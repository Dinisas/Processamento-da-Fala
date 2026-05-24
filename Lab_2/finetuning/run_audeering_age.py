#!/usr/bin/env python3
"""
Small hyperparameter sweep over audeering wav2vec2 age regression.

Two sweep axes:
  - SSL transformer LAYER to mean-pool: -1 (last) plus 20, 16, 12.
    Different layers cache to separate dirs under
      lab2_data/<part>/audeering_w2v2_layer<N>_pooled/  (or .._last_pooled).
  - HEAD architecture / hyperparameters:
      * 3 deterministic linear heads (Ridge x2, SVR-RBF)
      * 3 MLP architectures, each trained with 3 SEEDS (variance estimate)

Total = 48 (layer x head) runs. Targets ~10-15 min on the first run (the
3 new-layer extractions dominate) and ~1-2 min on subsequent runs because
features are cached per layer on disk via pf_tools.SLPdata.

The script never touches audeering's age/gender classifier head: we load
the checkpoint as a base `Wav2Vec2Model`, so HF's prefix-stripping silently
drops the `age.*` / `gender.*` weights and only the wav2vec2 backbone
weights are mapped.

Outputs:
  - Top-20 (layer, head) configs ranked by dev MAE.
  - MLP results aggregated across seeds (mean / std / min per architecture).
  - Top-3 submission CSVs (g<group>_sweep_L<layer>_<head>.csv).
  - sweep_results.csv with every individual run for offline analysis.

Usage:
    python finetuning/run_audeering_age.py
    python finetuning/run_audeering_age.py --trainset train     # full train, slow first run
    python finetuning/run_audeering_age.py --duration 8         # shorter clips for speed
"""

import argparse
import copy
import csv
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

# MPS fallback for any op without a Metal kernel.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LAB2_DIR))

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    WavLMModel,
    logging as tf_logging,
)

from pf_tools import SLPdata, prepare_slp_data, create_submission_file


# ---------------------------------------------------------------------
# Backbone registry.
#   - model_id     : HF hub id
#   - model_class  : Wav2Vec2Model or WavLMModel
#   - prefix       : SLPdata cache key prefix (separates caches across backbones)
#   - layers       : default single-layer sweep set for that backbone
#   - fusion_layers: default multi-layer fusion set for that backbone
# Adding another backbone (HuBERT, MMS, XLS-R) is one entry here.
# ---------------------------------------------------------------------

BACKBONES = {
    'audeering': {
        'model_id': 'audeering/wav2vec2-large-robust-24-ft-age-gender',
        'model_class': Wav2Vec2Model,
        'prefix': 'audeering_w2v2',
        'layers': [-1, 16, 12],
        'fusion_layers': [-1, 20, 16, 12, 8],
        'note': 'wav2vec2-large supervised-fine-tuned on TIMIT/VoxCeleb2/CommonVoice/aGender '
                '(English-dominant). 24 transformer layers, 1024-d.',
    },
    'wavlm': {
        'model_id': 'microsoft/wavlm-base-plus',
        'model_class': WavLMModel,
        'prefix': 'wavlm_bp',
        # WavLM Base+ has 12 transformer layers (vs audeering's 24). We keep -1 as the
        # "last hidden state" sentinel; positive N picks hidden_states[N] (1..12).
        'layers': [-1, 8, 6],
        'fusion_layers': [-1, 10, 8, 6, 4, 2],
        'note': 'WavLM Base+, the exact backbone Yang et al. arXiv:2502.12007 used. '
                '12 transformer layers, 768-d, purely SSL (no supervised fine-tune).',
    },
    'xlsr53': {
        'model_id': 'facebook/wav2vec2-large-xlsr-53',
        'model_class': Wav2Vec2Model,
        'prefix': 'xlsr53',
        # Same wav2vec2-large architecture as audeering (24 layers, 1024-d), so the
        # same layer choices port over. Different pretraining: pure SSL on 53
        # languages including Portuguese, no supervised age head.
        'layers': [-1, 16, 12],
        'fusion_layers': [-1, 20, 16, 12, 8],
        'note': 'wav2vec2-large pretrained SSL on 53 languages incl. Portuguese '
                '(no supervised fine-tune). Same architecture as audeering — '
                '24 transformer layers, 1024-d — different pretraining data.',
    },
}


AUD_MODEL_ID = BACKBONES['audeering']['model_id']  # legacy alias

# Default layer sets live on the backbone configs in BACKBONES. They differ per
# backbone (audeering has 24 layers, WavLM Base+ has 12). Override via --layers.
SEEDS  = [42, 7, 1]                    # seeds for MLP variance


def parse_layers(s):
    """Parse '-1 16 12' or '-1,16,12' into [-1, 16, 12]."""
    return [int(x) for x in str(s).replace(',', ' ').split()]

# (name, h1, h2, dropout, lr, weight_decay). Paper's small head is in here on
# purpose — the previous sweep showed it ties with / beats the larger variants
# at the best layer (L16) and has lower seed-to-seed std.
MLP_CONFIGS = [
    ('mlp_128_64_d03',  128, 64,  0.3, 1e-3, 1e-4),   # paper recipe
    ('mlp_256_128_d04', 256, 128, 0.4, 1e-3, 1e-4),
    ('mlp_512_256_d04', 512, 256, 0.4, 1e-3, 1e-4),
]


def _lin(estimator):
    """Wrap a linear estimator in a pipeline that drops near-constant feature
    dims and standardises the survivors. Without this, audeering's middle
    layers have huge per-dim scale heterogeneity (max|x| up to ~560) and
    Ridge / SVR end up with `inf` coefficients that overflow at predict time."""
    return make_pipeline(
        VarianceThreshold(threshold=1e-4),
        StandardScaler(),
        estimator,
    )


LINEAR_CONFIGS = [
    ('ridge_a1',   lambda: _lin(Ridge(alpha=1.0))),
    ('ridge_a10',  lambda: _lin(Ridge(alpha=10.0))),
    ('svr_rbf',    lambda: _lin(SVR(kernel='rbf', C=10.0, gamma='scale'))),
]

AGE_MIN, AGE_MAX = 20.0, 90.0


def transform_id_for(layer, prefix='audeering_w2v2'):
    """Cache key per (backbone prefix, layer). The default 'audeering_w2v2' prefix
    reproduces the on-disk cache from earlier runs."""
    suffix = 'last_pooled' if layer == -1 else f'layer{layer}_pooled'
    return f'{prefix}_{suffix}'


def load_backbone(backbone_name, device):
    """Load the chosen backbone in eval/frozen mode. Returns (fe, model, cfg)."""
    cfg = BACKBONES[backbone_name]
    fe = Wav2Vec2FeatureExtractor.from_pretrained(cfg['model_id'])
    tf_logging.set_verbosity_error()  # silence unexpected-keys warnings
    bk = cfg['model_class'].from_pretrained(cfg['model_id']).eval().to(device)
    tf_logging.set_verbosity_warning()
    for p in bk.parameters():
        p.requires_grad = False
    return fe, bk, cfg


# ---------------------------------------------------------------------
# Model + helpers
# ---------------------------------------------------------------------

class AgeMLP(nn.Module):
    def __init__(self, input_dim, h1, h2, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h1, h2),        nn.ReLU(), nn.Dropout(p),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x)


class FusionAgeMLP(nn.Module):
    """SUPERB-style multi-layer fusion. Each utterance is represented as a
    stack of mean-pooled per-layer embeddings of shape (n_layers, input_dim).
    A learnable scalar per layer is softmax-normalised at every forward pass;
    the layers are combined as a weighted sum, then passed through the same
    paper-recipe MLP. Zero-init -> uniform initial weights via softmax."""

    def __init__(self, n_layers, input_dim, h1, h2, p):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h1, h2),        nn.ReLU(), nn.Dropout(p),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        # x: (B, n_layers, input_dim)
        w = torch.softmax(self.layer_weights, dim=0).view(1, -1, 1)
        fused = (x * w).sum(dim=1)        # -> (B, input_dim)
        return self.mlp(fused)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def make_extractor(feat_ext, backbone, device, layer=-1, duration=10.0):
    use_hidden = (layer != -1)
    def extract(filename):
        audio, _ = librosa.load(str(filename), sr=16000, mono=True, duration=duration)
        inputs = feat_ext(audio, sampling_rate=16000, return_tensors='pt', padding=True)
        x = inputs['input_values'].to(device)
        with torch.no_grad():
            if use_hidden:
                h = backbone(x, output_hidden_states=True).hidden_states[layer]
            else:
                h = backbone(x).last_hidden_state
        return h.mean(dim=1).cpu().numpy().astype(np.float32)
    return extract


def aggregate(slpdata):
    d = prepare_slp_data(slpdata)
    raw = d['label'][:, 1]
    y = np.array([np.nan if v == '?' else float(v) for v in raw], dtype=float)
    return d['data'], y, d['identifiers']


def extract_layer_features(datadir, layer, fe, bk, device, partitions, duration,
                           prefix='audeering_w2v2'):
    transform_id = transform_id_for(layer, prefix)
    extract = make_extractor(fe, bk, device, layer=layer, duration=duration)
    X, y, ids = {}, {}, {}
    for p in partitions:
        slp = SLPdata(datadir, p,
                      transform_id=transform_id,
                      audio_transform=extract,
                      chunk_transform=None, chunk_size=0, chunk_hop=0)
        X[p], y[p], ids[p] = aggregate(slp)
    return X, y, ids


def train_mlp(model, Xt, yt, Xd, yd, device,
              lr=1e-3, weight_decay=1e-4, batch_size=32,
              max_epochs=200, patience=20):
    """MSE + Adam + target normalisation + ReduceLROnPlateau + early stop."""
    model = model.to(device)
    y_mean = float(np.mean(yt))
    y_std = float(np.std(yt) + 1e-8)
    yt_norm = (np.asarray(yt, dtype=np.float32) - y_mean) / y_std

    Xt_t = torch.as_tensor(Xt, dtype=torch.float32)
    yt_t = torch.as_tensor(yt_norm, dtype=torch.float32).view(-1, 1)
    Xd_t = torch.as_tensor(Xd, dtype=torch.float32).to(device)
    yd_np = np.asarray(yd, dtype=float).ravel()

    loader = DataLoader(TensorDataset(Xt_t, yt_t), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    crit = nn.MSELoss()

    best_mae = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds_norm = model(Xd_t).cpu().numpy().ravel()
        preds = preds_norm * y_std + y_mean
        mae = float(np.mean(np.abs(preds - yd_np)))
        sched.step(mae)
        if mae < best_mae - 1e-4:
            best_mae = mae
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model._age_y_mean = y_mean
    model._age_y_std = y_std

    # Train MAE at the best-by-dev checkpoint — diagnostic for overfitting.
    # If train MAE << dev MAE the model has memorised the training fold;
    # if they're close we're at the feature/capacity ceiling, not overfitting.
    Xt_eval = torch.as_tensor(Xt, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        train_preds_norm = model(Xt_eval).cpu().numpy().ravel()
    train_preds = train_preds_norm * y_std + y_mean
    train_mae = float(np.mean(np.abs(train_preds - np.asarray(yt, dtype=float).ravel())))

    return model, best_mae, train_mae


def predict_np(model, X, device):
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(Xt).cpu().numpy().ravel()
    if hasattr(model, '_age_y_mean'):
        out = out * model._age_y_std + model._age_y_mean
    return out


# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

def run_sweep(args):
    datadir = str(LAB2_DIR / 'lab2_data') + '/'
    device = select_device()
    bb_cfg = BACKBONES[args.backbone]
    layers = args.layers or bb_cfg['layers']
    print(f'device  : {device}')
    print(f'backbone: {args.backbone}  ({bb_cfg["model_id"]})')
    print(f'   note : {bb_cfg["note"]}')
    print(f'trainset: {args.trainset}')
    print(f'mode    : single-layer sweep')
    print(f'layers  : {layers}')
    print(f'mlp seeds: {SEEDS}')

    print(f'\n-- loading {args.backbone} backbone --')
    t0 = time.time()
    fe, bk, _ = load_backbone(args.backbone, device)
    print(f'  ready in {time.time()-t0:.1f}s  hidden={bk.config.hidden_size}  '
          f'layers={len(bk.encoder.layers)}')

    partitions = [args.trainset, 'dev', 'evl']
    if args.trainset != 'train_small':
        partitions = list(dict.fromkeys(partitions))  # dedupe, preserve order

    all_results = []   # list of dicts: layer, head, seed, dev_mae, dt, dev_hyp, evl_hyp, fileids

    for layer in layers:
        layer_t0 = time.time()
        print(f'\n========= LAYER {layer}  '
              f'({transform_id_for(layer, bb_cfg["prefix"])}) =========')
        X, y, ids = extract_layer_features(
            datadir, layer, fe, bk, device, partitions, args.duration,
            prefix=bb_cfg['prefix'],
        )
        print(f'  features ready (cache or extract): {time.time()-layer_t0:.1f}s')

        # Drop near-constant feature dims on the training fold.
        vt = VarianceThreshold(threshold=1e-6).fit(X[args.trainset])
        Xt = vt.transform(X[args.trainset]).astype(np.float32)
        Xd = vt.transform(X['dev']).astype(np.float32)
        Xe = vt.transform(X['evl']).astype(np.float32)
        yt, yd = y[args.trainset], y['dev']
        print(f'  kept {Xt.shape[1]}/{X[args.trainset].shape[1]} dims  '
              f'X.std={Xt.std():.3f}  max|x|={np.abs(Xt).max():.2f}  '
              f'y.mean={yt.mean():.2f}  y.std={yt.std():.2f}')

        # --- linear heads (deterministic) ---
        if args.mlp_only:
            print('  [--mlp-only] skipping linear baselines for this run')
        for name, factory in (() if args.mlp_only else LINEAR_CONFIGS):
            t0 = time.time()
            model = factory()
            model.fit(Xt, yt)
            train_hyp = np.clip(model.predict(Xt), AGE_MIN, AGE_MAX)
            train_mae = float(mean_absolute_error(yt, train_hyp))
            dev_hyp = np.clip(model.predict(Xd), AGE_MIN, AGE_MAX)
            evl_hyp = np.clip(model.predict(Xe), AGE_MIN, AGE_MAX)
            mae = float(mean_absolute_error(yd, dev_hyp))
            dt = time.time() - t0
            all_results.append(dict(
                layer=layer, head=name, seed=None,
                dev_mae=mae, train_mae=train_mae, dt=dt,
                dev_hyp=dev_hyp, evl_hyp=evl_hyp,
                fid_dev=ids['dev'], fid_evl=ids['evl'],
            ))
            print(f'  L{layer:3d}  {name:24s}        '
                  f'dev MAE {mae:6.3f}  train {train_mae:5.2f}  '
                  f'gap {mae-train_mae:+5.2f}  ({dt:5.1f}s)')

        # --- MLP architectures x seeds ---
        for mlp_name, h1, h2, p, lr, wd in MLP_CONFIGS:
            for seed in SEEDS:
                t0 = time.time()
                set_seeds(seed)
                model = AgeMLP(input_dim=Xt.shape[1], h1=h1, h2=h2, p=p)
                model, _, train_mae = train_mlp(
                    model, Xt, yt, Xd, yd, device,
                    lr=lr, weight_decay=wd,
                )
                dev_hyp = np.clip(predict_np(model, Xd, device), AGE_MIN, AGE_MAX)
                evl_hyp = np.clip(predict_np(model, Xe, device), AGE_MIN, AGE_MAX)
                mae = float(mean_absolute_error(yd, dev_hyp))
                dt = time.time() - t0
                tag = f'{mlp_name}_s{seed}'
                all_results.append(dict(
                    layer=layer, head=tag, seed=seed,
                    dev_mae=mae, train_mae=train_mae, dt=dt,
                    dev_hyp=dev_hyp, evl_hyp=evl_hyp,
                    fid_dev=ids['dev'], fid_evl=ids['evl'],
                ))
                print(f'  L{layer:3d}  {tag:24s}        '
                      f'dev MAE {mae:6.3f}  train {train_mae:5.2f}  '
                      f'gap {mae-train_mae:+5.2f}  ({dt:5.1f}s)')

    # ===== Summary tables =====
    all_sorted = sorted(all_results, key=lambda r: r['dev_mae'])

    print('\n========= TOP 20 (layer, head) BY DEV MAE =========')
    print(f'{"rank":>4s}  {"layer":>5s}  {"head":24s}  {"dev MAE":>8s}  '
          f'{"train":>6s}  {"gap":>6s}  {"time":>7s}')
    print('-' * 76)
    for i, r in enumerate(all_sorted[:20], 1):
        gap = r['dev_mae'] - r['train_mae']
        print(f'{i:>4d}  L{r["layer"]:>4d}  {r["head"]:24s}  '
              f'{r["dev_mae"]:8.3f}  {r["train_mae"]:6.2f}  {gap:+6.2f}  {r["dt"]:6.1f}s')

    # MLP aggregate across seeds
    print('\n========= MLPs AVERAGED ACROSS SEEDS =========')
    agg = defaultdict(list)
    for r in all_results:
        if r['seed'] is None:
            continue
        base = r['head'].rsplit('_s', 1)[0]
        agg[(r['layer'], base)].append(r['dev_mae'])
    agg_sorted = sorted(agg.items(), key=lambda kv: np.mean(kv[1]))
    print(f'{"layer":>5s}  {"arch":24s}  {"mean":>7s}  {"std":>6s}  {"min":>6s}  '
          f'{"max":>6s}')
    print('-' * 64)
    for (layer, base), maes in agg_sorted:
        a = np.asarray(maes)
        print(f'L{layer:>4d}  {base:24s}  {a.mean():7.3f}  {a.std():6.3f}  '
              f'{a.min():6.3f}  {a.max():6.3f}')

    # Write top-3 submissions
    print('\n========= TOP-3 SUBMISSION CSVs =========')
    for r in all_sorted[:3]:
        resdir = (LAB2_DIR / 'lab2_data' / args.trainset / 'models'
                  / f'sweep_L{r["layer"]}_{r["head"]}_{args.backbone}')
        resdir.mkdir(parents=True, exist_ok=True)
        with open(resdir / 'dev.pkl', 'wb') as f:
            pickle.dump({'hyp': r['dev_hyp'], 'fileids': r['fid_dev']}, f)
        with open(resdir / 'evl.pkl', 'wb') as f:
            pickle.dump({'hyp': r['evl_hyp'], 'fileids': r['fid_evl']}, f)
        sub = LAB2_DIR / (f'g{args.group}_{args.trainset}_{args.backbone}_'
                      f'sweep_L{r["layer"]}_{r["head"]}.csv')
        create_submission_file(str(resdir), str(sub))
        print(f'  L{r["layer"]:>3d}  {r["head"]:24s}  dev MAE {r["dev_mae"]:6.3f}  '
              f'-> {sub.name}')

    # ===== Top-3 prediction ensemble =====
    print('\n========= ENSEMBLE OF TOP-3 =========')
    top3 = all_sorted[:3]
    fid_dev = top3[0]['fid_dev']
    fid_evl = top3[0]['fid_evl']
    dev_ens = np.mean([r['dev_hyp'] for r in top3], axis=0)
    evl_ens = np.mean([r['evl_hyp'] for r in top3], axis=0)
    dev_mae_ens = float(mean_absolute_error(yd, dev_ens))
    ens_dir = (LAB2_DIR / 'lab2_data' / args.trainset / 'models'
               / f'sweep_ensemble_top3_{args.backbone}')
    ens_dir.mkdir(parents=True, exist_ok=True)
    with open(ens_dir / 'dev.pkl', 'wb') as f:
        pickle.dump({'hyp': dev_ens, 'fileids': fid_dev}, f)
    with open(ens_dir / 'evl.pkl', 'wb') as f:
        pickle.dump({'hyp': evl_ens, 'fileids': fid_evl}, f)
    ens_sub = LAB2_DIR / (f'g{args.group}_{args.trainset}_{args.backbone}_'
                      f'ensemble_top3.csv')
    create_submission_file(str(ens_dir), str(ens_sub))
    rank1 = top3[0]['dev_mae']
    delta = rank1 - dev_mae_ens
    print(f'  ensemble dev MAE {dev_mae_ens:6.3f}  '
          f'(rank-1 single = {rank1:.3f}, delta = {delta:+.3f})  '
          f'-> {ens_sub.name}')

    # CSV of all runs — name includes backbone so audeering / wavlm don't overwrite.
    csv_path = LAB2_DIR / f'sweep_results_{args.backbone}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer', 'head', 'seed', 'dev_mae', 'train_mae', 'gap', 'time_s'])
        for r in all_results:
            gap = r['dev_mae'] - r['train_mae']
            w.writerow([r['layer'], r['head'], r['seed'],
                        f'{r["dev_mae"]:.4f}', f'{r["train_mae"]:.4f}',
                        f'{gap:+.4f}', f'{r["dt"]:.2f}'])
    print(f'\nFull results written to {csv_path.name}')

    # Overfitting verdict — a quick aggregate cue.
    gaps = [r['dev_mae'] - r['train_mae']
            for r in all_results if r['head'].startswith('mlp_')]
    if gaps:
        gaps_arr = np.asarray(gaps)
        print(f'\nMLP dev-train gap: mean={gaps_arr.mean():+.2f}  '
              f'median={float(np.median(gaps_arr)):+.2f}  '
              f'min={gaps_arr.min():+.2f}  max={gaps_arr.max():+.2f}')
        if gaps_arr.mean() > 1.5:
            print('  -> dev - train gap > 1.5: classic overfitting signature.')
        elif gaps_arr.mean() > 0.5:
            print('  -> moderate gap; mild overfitting but probably not the bottleneck.')
        else:
            print('  -> small gap: train and dev are close → at the feature/capacity '
                  'ceiling, not overfitting.')


# ---------------------------------------------------------------------
# Multi-layer fusion mode
# ---------------------------------------------------------------------

def run_fusion(args):
    datadir = str(LAB2_DIR / 'lab2_data') + '/'
    device = select_device()
    bb_cfg = BACKBONES[args.backbone]
    layers = args.layers or bb_cfg['fusion_layers']

    print(f'device  : {device}')
    print(f'backbone: {args.backbone}  ({bb_cfg["model_id"]})')
    print(f'   note : {bb_cfg["note"]}')
    print(f'trainset: {args.trainset}')
    print(f'mode    : MULTI-LAYER FUSION (learnable softmax weights per layer)')
    print(f'layers  : {layers}')
    print(f'mlp seeds: {SEEDS}')

    print(f'\n-- loading {args.backbone} backbone --')
    t0 = time.time()
    fe, bk, _ = load_backbone(args.backbone, device)
    print(f'  ready in {time.time()-t0:.1f}s  hidden={bk.config.hidden_size}  '
          f'layers={len(bk.encoder.layers)}')

    partitions = [args.trainset, 'dev', 'evl']
    if args.trainset != 'train_small':
        partitions = list(dict.fromkeys(partitions))

    # Extract per-layer features for every layer in the fusion set.
    print(f'\n-- extracting / loading features for {len(layers)} layers --')
    per_layer_X = {}
    ref_y = ref_ids = None
    for layer in layers:
        t0 = time.time()
        X, y, ids = extract_layer_features(
            datadir, layer, fe, bk, device, partitions, args.duration,
            prefix=bb_cfg['prefix'],
        )
        per_layer_X[layer] = X
        if ref_y is None:
            ref_y, ref_ids = y, ids
        else:
            for part in partitions:
                if not (ids[part] == ref_ids[part]).all():
                    raise RuntimeError(
                        f'fileid order mismatch on partition {part!r} for layer {layer}. '
                        'Cannot stack features across layers.'
                    )
        print(f'  L{layer:3d}: {time.time()-t0:6.1f}s')

    # Stack per partition: (N, n_layers, input_dim).
    Xs = {}
    for part in partitions:
        Xs[part] = np.stack(
            [per_layer_X[L][part] for L in layers], axis=1
        ).astype(np.float32)

    Xt = Xs[args.trainset]
    Xd = Xs['dev']
    Xe = Xs['evl']
    yt = ref_y[args.trainset]
    yd = ref_y['dev']
    print(f'\n  stacked X shape: train={Xt.shape}  dev={Xd.shape}  evl={Xe.shape}')
    print(f'  targets: mean={yt.mean():.2f}  std={yt.std():.2f}')

    # Train FusionAgeMLP for each architecture x seed.
    print(f'\n-- training fusion MLPs ({len(MLP_CONFIGS)} archs × '
          f'{len(SEEDS)} seeds = {len(MLP_CONFIGS)*len(SEEDS)} runs) --')
    all_results = []
    for mlp_name, h1, h2, p, lr, wd in MLP_CONFIGS:
        for seed in SEEDS:
            t0 = time.time()
            set_seeds(seed)
            model = FusionAgeMLP(
                n_layers=len(layers), input_dim=Xt.shape[-1],
                h1=h1, h2=h2, p=p,
            )
            model, _, train_mae = train_mlp(
                model, Xt, yt, Xd, yd, device,
                lr=lr, weight_decay=wd,
            )
            with torch.no_grad():
                w_learned = torch.softmax(
                    model.layer_weights.detach().cpu(), dim=0
                ).numpy()

            dev_hyp = np.clip(predict_np(model, Xd, device), AGE_MIN, AGE_MAX)
            evl_hyp = np.clip(predict_np(model, Xe, device), AGE_MIN, AGE_MAX)
            mae = float(mean_absolute_error(yd, dev_hyp))
            dt = time.time() - t0

            tag = f'fusion_{mlp_name}_s{seed}'
            weights_str = ' '.join(f'L{L}:{w:.2f}' for L, w in zip(layers, w_learned))
            print(f'  {tag:32s}  dev {mae:6.3f}  train {train_mae:5.2f}  '
                  f'gap {mae-train_mae:+5.2f}  w[{weights_str}]  ({dt:5.1f}s)')

            all_results.append(dict(
                layer='fusion', head=tag, seed=seed,
                dev_mae=mae, train_mae=train_mae, dt=dt,
                dev_hyp=dev_hyp, evl_hyp=evl_hyp,
                fid_dev=ref_ids['dev'], fid_evl=ref_ids['evl'],
                layer_weights={int(L): float(w) for L, w in zip(layers, w_learned)},
            ))

    all_sorted = sorted(all_results, key=lambda r: r['dev_mae'])

    print('\n========= ALL FUSION RESULTS (sorted by dev MAE) =========')
    print(f'{"rank":>4s}  {"head":32s}  {"dev MAE":>8s}  '
          f'{"train":>6s}  {"gap":>6s}  {"time":>7s}')
    print('-' * 76)
    for i, r in enumerate(all_sorted, 1):
        gap = r['dev_mae'] - r['train_mae']
        print(f'{i:>4d}  {r["head"]:32s}  {r["dev_mae"]:8.3f}  '
              f'{r["train_mae"]:6.2f}  {gap:+6.2f}  {r["dt"]:6.1f}s')

    # Average learned softmax weights across all runs — cleanest "which
    # layer matters" signal we can produce.
    print('\n========= AVERAGE LEARNED LAYER WEIGHTS (across all runs) =========')
    avg_w = {L: float(np.mean([r['layer_weights'][L] for r in all_results]))
             for L in layers}
    for L, w in sorted(avg_w.items(), key=lambda kv: -kv[1]):
        bar = '#' * int(round(w * 40))
        print(f'  L{L:3d}: {w:6.3f}  {bar}')

    # Submission CSVs for the top-3 fusion runs.
    print('\n========= TOP-3 FUSION SUBMISSION CSVs =========')
    for r in all_sorted[:3]:
        resdir = (LAB2_DIR / 'lab2_data' / args.trainset / 'models'
                  / f'{r["head"]}_{args.backbone}')
        resdir.mkdir(parents=True, exist_ok=True)
        with open(resdir / 'dev.pkl', 'wb') as f:
            pickle.dump({'hyp': r['dev_hyp'], 'fileids': r['fid_dev']}, f)
        with open(resdir / 'evl.pkl', 'wb') as f:
            pickle.dump({'hyp': r['evl_hyp'], 'fileids': r['fid_evl']}, f)
        sub = LAB2_DIR / (f'g{args.group}_{args.trainset}_{args.backbone}_'
                      f'{r["head"]}.csv')
        create_submission_file(str(resdir), str(sub))
        print(f'  {r["head"]:32s}  dev MAE {r["dev_mae"]:6.3f}  -> {sub.name}')

    # Top-3 ensemble.
    top3 = all_sorted[:3]
    dev_ens = np.mean([r['dev_hyp'] for r in top3], axis=0)
    evl_ens = np.mean([r['evl_hyp'] for r in top3], axis=0)
    dev_mae_ens = float(mean_absolute_error(yd, dev_ens))
    ens_dir = (LAB2_DIR / 'lab2_data' / args.trainset / 'models'
               / f'fusion_ensemble_top3_{args.backbone}')
    ens_dir.mkdir(parents=True, exist_ok=True)
    with open(ens_dir / 'dev.pkl', 'wb') as f:
        pickle.dump({'hyp': dev_ens, 'fileids': top3[0]['fid_dev']}, f)
    with open(ens_dir / 'evl.pkl', 'wb') as f:
        pickle.dump({'hyp': evl_ens, 'fileids': top3[0]['fid_evl']}, f)
    ens_sub = LAB2_DIR / (f'g{args.group}_{args.trainset}_{args.backbone}_'
                      f'fusion_ensemble_top3.csv')
    create_submission_file(str(ens_dir), str(ens_sub))
    rank1 = top3[0]['dev_mae']
    print(f'\n========= FUSION ENSEMBLE OF TOP-3 =========')
    print(f'  dev MAE {dev_mae_ens:6.3f}  '
          f'(rank-1 single = {rank1:.3f}, delta = {rank1-dev_mae_ens:+.3f})  '
          f'-> {ens_sub.name}')

    # Per-run CSV including learned weights — name includes backbone.
    csv_path = LAB2_DIR / f'fusion_results_{args.backbone}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['head', 'seed', 'dev_mae', 'train_mae', 'gap', 'time_s'] \
               + [f'w_L{L}' for L in layers]
        w.writerow(cols)
        for r in all_results:
            gap = r['dev_mae'] - r['train_mae']
            row = [r['head'], r['seed'], f'{r["dev_mae"]:.4f}',
                   f'{r["train_mae"]:.4f}', f'{gap:+.4f}', f'{r["dt"]:.2f}']
            row += [f'{r["layer_weights"][L]:.4f}' for L in layers]
            w.writerow(row)
    print(f'\nFusion results written to {csv_path.name}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--trainset', default='train_small',
                        choices=('train_small', 'train', 'big_train_falar'),
                        help='training partition (default train_small). '
                             '"big_train_falar" is the FalAR-streamed expansion '
                             'built by finetuning/build_train_falar.py — only '
                             'works after that script has been run.')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='max audio duration in seconds (default 10)')
    parser.add_argument('--group', default='07',
                        help='student group prefix for submission CSVs')
    parser.add_argument('--mlp-only', action='store_true',
                        help='skip the linear baselines (Ridge, SVR); useful '
                             'on the slow --trainset train run where MLPs '
                             'are known to dominate.')
    parser.add_argument('--layers', type=parse_layers, default=None,
                        help='space- or comma-separated transformer layers to use, '
                             'e.g. "-1 16 12" or "17,18,19". Defaults: '
                             '[-1, 16, 12] for single-layer sweep, '
                             '[-1, 20, 16, 12, 8] for --fusion.')
    parser.add_argument('--fusion', action='store_true',
                        help='multi-layer fusion mode: extract features for '
                             'every layer in --layers (or the fusion default), '
                             'learn a softmax-weighted combination, then train '
                             'the MLP heads on the fused vector. Single command, '
                             'self-contained — no single-layer sweep is run.')
    parser.add_argument('--backbone', default='audeering',
                        choices=tuple(BACKBONES.keys()),
                        help='which pretrained SSL backbone to extract from. '
                             '"audeering" = wav2vec2-large supervised-fine-tuned on '
                             'TIMIT/VoxCeleb2/Common Voice/aGender. '
                             '"wavlm" = WavLM Base+ (the paper\'s actual backbone, '
                             'purely SSL, 12 layers, 768-d). '
                             '"xlsr53" = wav2vec2-large XLS-R 53, purely SSL on 53 '
                             'languages incl. Portuguese (same arch as audeering, '
                             'no age supervision). Caches are kept under separate '
                             'directories per backbone, so switching is safe.')
    args = parser.parse_args()
    if args.fusion:
        run_fusion(args)
    else:
        run_sweep(args)


if __name__ == '__main__':
    main()
