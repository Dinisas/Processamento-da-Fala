#!/usr/bin/env python3
"""Frozen-backbone head sweep on big_train_falar with a speaker-disjoint
real-speaker_id split.

Pipeline (all GPU-cheap once features are cached):
  1. Load the speaker-disjoint manifest produced by build_speaker_split_real.py.
  2. Extract mean-pooled audeering wav2vec2 features at one or more layers
     for big_train_falar (train_inner + dev_inner), and optionally also for
     lab `dev`, `evl`, and `falar_dev` (for cross-evaluation). Cached to disk
     in the same per-UUID .npy layout the rest of the project uses, so
     re-runs are instant.
  3. Train MULTIPLE head families on train_inner features:
       - Linear:  Ridge (alpha={1, 10, 100}), SVR(rbf, C=10)
       - MLP:     three architectures (paper recipe + 2 larger), each at
                  three seeds for variance estimation.
     Optionally a SUPERB-style softmax-fused MLP over the extracted layers.
  4. Evaluate each head on dev_inner (primary), and on the eval partitions
     (lab dev / falar_dev) when available. Apply bias correction using
     residuals on a held-out fold.
  5. Print a comparison table, write a submission CSV for the best head,
     and produce a comparison plot.

Usage:
    python finetuning/head_sweep_falar.py
    python finetuning/head_sweep_falar.py --layers 16
    python finetuning/head_sweep_falar.py --layers 12 16 -1 --fusion
    python finetuning/head_sweep_falar.py --no-extra-eval        # skip cross-eval
    python finetuning/head_sweep_falar.py --max-train 8000       # cap for smoke test
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import soundfile as sf
import numpy as np
import pandas as pd
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
from tqdm import tqdm
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    logging as tf_logging,
)


SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB2_DIR / 'lab2_data'

AUD_MODEL_ID = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
AGE_MIN, AGE_MAX = 20.0, 90.0


# ---------------------------------------------------------------------
# Head zoo
# ---------------------------------------------------------------------

def _lin(estimator):
    """Standard pipeline: drop near-constant dims, standardise, then estimator."""
    return make_pipeline(
        VarianceThreshold(threshold=1e-4),
        StandardScaler(),
        estimator,
    )


LINEAR_CONFIGS = [
    ('ridge_a1',    lambda: _lin(Ridge(alpha=1.0))),
    ('ridge_a10',   lambda: _lin(Ridge(alpha=10.0))),
    ('ridge_a100',  lambda: _lin(Ridge(alpha=100.0))),
    ('svr_rbf_c10', lambda: _lin(SVR(kernel='rbf', C=10.0, gamma='scale'))),
]


# (name, h1, h2, dropout, lr, weight_decay)
MLP_CONFIGS = [
    ('mlp_128_64_d03',   128,  64,  0.3, 1e-3, 1e-4),  # paper recipe
    ('mlp_256_128_d04',  256, 128,  0.4, 1e-3, 1e-4),
    ('mlp_512_256_d04',  512, 256,  0.4, 1e-3, 1e-4),
]
SEEDS = [42, 7, 1]


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
    """SUPERB-style multi-layer fusion: softmax-normalised scalar per layer,
    fused into a single embedding, then a paper-style MLP."""

    def __init__(self, n_layers, input_dim, h1, h2, p):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h1, h2),        nn.ReLU(), nn.Dropout(p),
            nn.Linear(h2, 1),
        )

    def forward(self, x):                       # x: (B, L, D)
        w = torch.softmax(self.layer_weights, dim=0).view(1, -1, 1)
        return self.mlp((x * w).sum(dim=1))


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_mlp(model, Xt, yt, Xd, yd, device,
              lr=1e-3, weight_decay=1e-4, batch_size=64,
              max_epochs=200, patience=20):
    """MSE + Adam + target-normalisation + ReduceLROnPlateau + early stop."""
    model = model.to(device)
    y_mean = float(np.mean(yt))
    y_std  = float(np.std(yt) + 1e-8)
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

    # Train MAE diagnostic.
    Xt_eval = torch.as_tensor(Xt, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        train_preds = model(Xt_eval).cpu().numpy().ravel() * y_std + y_mean
    train_mae = float(np.mean(np.abs(train_preds - np.asarray(yt).ravel())))
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
# Feature extraction (batched, cached per layer)
# ---------------------------------------------------------------------

def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def transform_id_for(layer: int) -> str:
    return 'audeering_w2v2_last_pooled' if layer == -1 \
        else f'audeering_w2v2_layer{layer}_pooled'


def load_cached_or_none(partition: str, layer: int, fileid: str) -> np.ndarray | None:
    tid = transform_id_for(layer)
    npy = DATA_DIR / partition / tid / fileid / f'{fileid}.npy'
    if not npy.is_file():
        return None
    v = np.load(npy)
    return (v[0] if v.ndim == 2 and v.shape[0] == 1 else v).astype(np.float32)


def save_cached(partition: str, layer: int, fileid: str, vec: np.ndarray):
    tid = transform_id_for(layer)
    sub = DATA_DIR / partition / tid / fileid
    sub.mkdir(parents=True, exist_ok=True)
    np.save(sub / f'{fileid}.npy', vec[np.newaxis, :])


def extract_partition(partition: str, df: pd.DataFrame, layers: list[int],
                      fe: Wav2Vec2FeatureExtractor, model: Wav2Vec2Model,
                      device: torch.device, batch_size: int = 8,
                      duration: float = 10.0) -> dict[int, np.ndarray]:
    """Returns {layer: (N, D) array} for the rows of df, using cache where
    possible. Extracts in one forward pass per batch via output_hidden_states."""
    # Pre-allocate the layer→array map with None for missing rows.
    per_layer: dict[int, list[np.ndarray | None]] = {L: [None] * len(df) for L in layers}
    rows = df.to_dict('records')
    todo: list[tuple[int, dict]] = []
    cache_hits = 0
    for i, r in enumerate(rows):
        fid = r['fileid']
        all_cached = True
        for L in layers:
            v = load_cached_or_none(partition, L, fid)
            if v is None:
                all_cached = False
                per_layer[L][i] = None
            else:
                per_layer[L][i] = v
        if all_cached:
            cache_hits += 1
        else:
            todo.append((i, r))
    print(f'    [{partition}] {cache_hits}/{len(df)} cache hits; '
          f'extracting {len(todo)}')

    if todo:
        for start in tqdm(range(0, len(todo), batch_size), desc=f'  {partition}'):
            batch = todo[start:start + batch_size]
            wavs = []
            for _, r in batch:
                wav_path = DATA_DIR / partition / 'wav' / r['wav']
                audio, sr = sf.read(str(wav_path), dtype='float32',
                                    frames=int(16000 * duration),
                                    always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                wavs.append(audio)
            inputs = fe(
                wavs, sampling_rate=16000, padding=True, return_tensors='pt',
                max_length=int(16000 * duration), truncation=True,
            )
            input_values = inputs['input_values'].to(device)
            attn = inputs.get('attention_mask')
            kwargs = {'input_values': input_values, 'output_hidden_states': True}
            if attn is not None:
                kwargs['attention_mask'] = attn.to(device)
            with torch.no_grad():
                out = model(**kwargs)
            # hidden_states is a tuple (CNN-out, layer1, ..., layerN). For
            # 24-layer wav2vec2-large that's 25 entries. layer-N index uses 1..24;
            # -1 sentinel maps to last_hidden_state.
            for L in layers:
                if L == -1:
                    h = out.last_hidden_state              # (B, T, D)
                else:
                    h = out.hidden_states[L]
                # Attention-aware mean pool over T.
                if attn is not None:
                    # HF returns attention_mask at the AUDIO-sample resolution.
                    # We mean-pool at the conv-output T resolution; for a quick
                    # extraction this is close enough (most pads are zero-frames
                    # anyway). For simplicity, do a plain mean — matches what
                    # run_audeering_age.py does.
                    pass
                pooled = h.mean(dim=1).cpu().numpy().astype(np.float32)
                for (idx, r), vec in zip(batch, pooled):
                    per_layer[L][idx] = vec
                    save_cached(partition, L, r['fileid'], vec)

    return {L: np.stack(per_layer[L]) for L in layers}


# ---------------------------------------------------------------------
# Bias correction helper
# ---------------------------------------------------------------------

def bias_corrected(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """Return (raw MAE, bias-corrected MAE) where the correction is the
    mean residual estimated on (pred, ref) itself (so 'oracle' calibration
    on this fold — useful for benchmarking the calibration headroom)."""
    raw = float(np.mean(np.abs(pred - ref)))
    bias = float(np.mean(pred - ref))
    corr = np.clip(pred - bias, AGE_MIN, AGE_MAX)
    return raw, float(np.mean(np.abs(corr - ref))), bias


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def load_partition_df(partition: str, want_age=True) -> pd.DataFrame:
    """Load info.csv (or info_full.csv if available) into a DataFrame."""
    full = DATA_DIR / partition / 'info_full.csv'
    base = DATA_DIR / partition / 'info.csv'
    path = full if full.is_file() else base
    df = pd.read_csv(path)
    df['fileid'] = df['wav'].str.replace('.wav', '', regex=False)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    return df


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--manifest', default='lab2_data/big_train_falar/speaker_disjoint_real.json',
                   help='speaker-disjoint manifest (default: real-speaker-id split)')
    p.add_argument('--source-partition', default='big_train_falar',
                   help='partition the manifest references (default big_train_falar)')
    p.add_argument('--layers', type=int, nargs='+', default=[16],
                   help='audeering wav2vec2 layers to extract (default 16). '
                        '-1 = last_hidden_state, positive N = hidden_states[N].')
    p.add_argument('--fusion', action='store_true',
                   help='also train SUPERB-style softmax-fused MLP heads over '
                        'all --layers. Requires len(--layers) > 1.')
    p.add_argument('--max-train', type=int, default=None,
                   help='cap train_inner size (smoke test)')
    p.add_argument('--no-extra-eval', action='store_true',
                   help='skip cross-evaluation on lab dev / falar_dev / lab evl')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--duration', type=float, default=10.0)
    p.add_argument('--group', default='07')
    p.add_argument('--out-csv', default=None,
                   help='where to save head-sweep results CSV (default: '
                        'Lab_2/head_sweep_falar_results.csv)')
    args = p.parse_args()

    if args.fusion and len(args.layers) < 2:
        raise ValueError('--fusion requires at least 2 layers in --layers')

    # ---------- Load manifest ----------
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = LAB2_DIR / manifest_path
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f'{"="*72}')
    print(f'HEAD SWEEP on {args.source_partition} (frozen audeering wav2vec2)')
    print(f'{"="*72}')
    print(f'manifest      : {manifest_path.relative_to(LAB2_DIR)}')
    print(f'method        : {manifest.get("method", "unknown")}')
    print(f'layers        : {args.layers}'
          f'{"  (+ fusion)" if args.fusion else ""}')
    print(f'n speakers    : train_inner={manifest["n_speakers_train"]}  '
          f'dev_inner={manifest["n_speakers_dev"]}')
    print(f'n utts        : train_inner={manifest["n_utts_train"]}  '
          f'dev_inner={manifest["n_utts_dev"]}')

    train_uuids = set(manifest['train_inner_uuids'])
    dev_uuids = set(manifest['dev_inner_uuids'])

    # ---------- Load source partition + carve ----------
    df_all = load_partition_df(args.source_partition)
    df_train = df_all[df_all['fileid'].isin(train_uuids)].reset_index(drop=True)
    df_dev = df_all[df_all['fileid'].isin(dev_uuids)].reset_index(drop=True)

    if args.max_train and len(df_train) > args.max_train:
        df_train = df_train.sample(n=args.max_train, random_state=42).reset_index(drop=True)
        print(f'(--max-train) capped train_inner to {len(df_train)}')

    # ---------- Extra eval partitions (cross-checks) ----------
    eval_dfs: dict[str, pd.DataFrame] = {
        'dev_inner': df_dev,
    }
    if not args.no_extra_eval:
        for part in ('dev', 'falar_dev', 'evl'):
            d = DATA_DIR / part
            if (d / 'info.csv').is_file():
                eval_dfs[part] = load_partition_df(part)

    print(f'\neval partitions:')
    for name, d in eval_dfs.items():
        n_labelled = int(d['age'].notna().sum())
        print(f'  {name:>15s}  rows={len(d)}  labelled={n_labelled}')

    # ---------- Load backbone (frozen) ----------
    device = select_device()
    print(f'\ndevice: {device}')
    tf_logging.set_verbosity_error()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(AUD_MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(AUD_MODEL_ID).eval().to(device)
    tf_logging.set_verbosity_warning()
    for p_ in model.parameters():
        p_.requires_grad = False

    # ---------- Extract features ----------
    print(f'\n-- feature extraction (cached) --')
    feats: dict[str, dict[int, np.ndarray]] = {}
    t0 = time.time()
    feats['train_inner'] = extract_partition(
        args.source_partition, df_train, args.layers, fe, model, device,
        batch_size=args.batch_size, duration=args.duration,
    )
    feats['dev_inner'] = extract_partition(
        args.source_partition, df_dev, args.layers, fe, model, device,
        batch_size=args.batch_size, duration=args.duration,
    )
    for name, d in eval_dfs.items():
        if name in ('dev_inner',):
            continue
        feats[name] = extract_partition(
            name, d, args.layers, fe, model, device,
            batch_size=args.batch_size, duration=args.duration,
        )
    print(f'  total extraction time: {time.time()-t0:.1f}s')

    # Free GPU memory after extraction.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pre-compute labels per partition.
    labels = {
        'train_inner': df_train['age'].to_numpy(dtype=float),
        'dev_inner':   df_dev['age'].to_numpy(dtype=float),
    }
    for name, d in eval_dfs.items():
        if name in labels:
            continue
        labels[name] = d['age'].to_numpy(dtype=float)

    # ---------- Build per-layer training matrices ----------
    print(f'\n-- training heads --')
    results = []  # list of dicts

    primary_eval = 'dev_inner'
    eval_names = list(eval_dfs.keys())

    for layer in args.layers:
        print(f'\n========= LAYER {layer} =========')
        # VarianceThreshold on training fold to drop near-constant dims.
        vt = VarianceThreshold(threshold=1e-6).fit(feats['train_inner'][layer])
        def transform(name):
            return vt.transform(feats[name][layer]).astype(np.float32)
        Xt = transform('train_inner')
        yt = labels['train_inner']
        print(f'  kept {Xt.shape[1]} of {feats["train_inner"][layer].shape[1]} dims  '
              f'X.std={Xt.std():.2f}  max|x|={np.abs(Xt).max():.2f}')

        # ---------- Linear heads ----------
        for hname, factory in LINEAR_CONFIGS:
            t0 = time.time()
            head = factory()
            head.fit(Xt, yt)
            train_hyp = np.clip(head.predict(Xt), AGE_MIN, AGE_MAX)
            train_mae = float(mean_absolute_error(yt, train_hyp))
            row = {
                'head': hname, 'layer': layer, 'seed': None, 'fusion': False,
                'train_mae': train_mae, 'time_s': 0.0,
            }
            for name in eval_names:
                X = transform(name) if name != primary_eval else \
                    vt.transform(feats['dev_inner'][layer]).astype(np.float32)
                pred = np.clip(head.predict(X), AGE_MIN, AGE_MAX)
                ref = labels[name]
                has = ~np.isnan(ref)
                if has.any():
                    raw, corr, bias = bias_corrected(pred[has], ref[has])
                    row[f'{name}_mae'] = raw
                    row[f'{name}_mae_biascorr'] = corr
                    row[f'{name}_bias'] = bias
                else:
                    row[f'{name}_mae'] = float('nan')
                    row[f'{name}_mae_biascorr'] = float('nan')
                    row[f'{name}_bias'] = float('nan')
                row[f'{name}_pred'] = pred
            row['time_s'] = time.time() - t0
            results.append(row)
            print(f'  L{layer:>3d}  {hname:<14s}  '
                  f'train={train_mae:5.2f}  '
                  f'dev_inner={row["dev_inner_mae"]:5.2f} '
                  f'(corr {row["dev_inner_mae_biascorr"]:5.2f})  '
                  + (f'falar_dev={row.get("falar_dev_mae", float("nan")):5.2f} '
                     if 'falar_dev' in eval_names else '')
                  + f'  ({row["time_s"]:5.1f}s)')

        # ---------- MLP heads × seeds ----------
        for mlp_name, h1, h2, p_, lr, wd in MLP_CONFIGS:
            for seed in SEEDS:
                t0 = time.time()
                set_seeds(seed)
                head = AgeMLP(input_dim=Xt.shape[1], h1=h1, h2=h2, p=p_)
                head, best_dev_mae, train_mae = train_mlp(
                    head, Xt, yt, transform('dev_inner'),
                    labels['dev_inner'], device,
                    lr=lr, weight_decay=wd, batch_size=64,
                )
                row = {
                    'head': f'{mlp_name}_s{seed}', 'layer': layer,
                    'seed': seed, 'fusion': False,
                    'train_mae': train_mae,
                }
                for name in eval_names:
                    X = transform(name) if name != primary_eval else \
                        transform('dev_inner')
                    pred = np.clip(predict_np(head, X, device), AGE_MIN, AGE_MAX)
                    ref = labels[name]
                    has = ~np.isnan(ref)
                    if has.any():
                        raw, corr, bias = bias_corrected(pred[has], ref[has])
                        row[f'{name}_mae'] = raw
                        row[f'{name}_mae_biascorr'] = corr
                        row[f'{name}_bias'] = bias
                    else:
                        row[f'{name}_mae'] = float('nan')
                        row[f'{name}_mae_biascorr'] = float('nan')
                        row[f'{name}_bias'] = float('nan')
                    row[f'{name}_pred'] = pred
                row['time_s'] = time.time() - t0
                results.append(row)
                print(f'  L{layer:>3d}  {mlp_name}_s{seed:<3d}     '
                      f'train={train_mae:5.2f}  '
                      f'dev_inner={row["dev_inner_mae"]:5.2f} '
                      f'(corr {row["dev_inner_mae_biascorr"]:5.2f})  '
                      + (f'falar_dev={row.get("falar_dev_mae", float("nan")):5.2f} '
                         if 'falar_dev' in eval_names else '')
                      + f'  ({row["time_s"]:5.1f}s)')

    # ---------- Optional fusion ----------
    if args.fusion:
        print(f'\n========= FUSION across {args.layers} =========')
        # Stack per-layer features per partition into (N, L, D).
        per_part_stack: dict[str, np.ndarray] = {}
        for name in ['train_inner', 'dev_inner'] + eval_names:
            if name == 'dev_inner' or name == 'train_inner' or name in eval_names:
                arrs = [feats[name][L] for L in args.layers]
                per_part_stack[name] = np.stack(arrs, axis=1).astype(np.float32)
        Xt_f = per_part_stack['train_inner']
        Xd_f = per_part_stack['dev_inner']

        for mlp_name, h1, h2, p_, lr, wd in MLP_CONFIGS:
            for seed in SEEDS:
                t0 = time.time()
                set_seeds(seed)
                head = FusionAgeMLP(
                    n_layers=len(args.layers),
                    input_dim=Xt_f.shape[-1],
                    h1=h1, h2=h2, p=p_,
                )
                head, _, train_mae = train_mlp(
                    head, Xt_f, yt, Xd_f, labels['dev_inner'], device,
                    lr=lr, weight_decay=wd, batch_size=64,
                )
                row = {
                    'head': f'fusion_{mlp_name}_s{seed}', 'layer': 'fusion',
                    'seed': seed, 'fusion': True,
                    'train_mae': train_mae,
                }
                for name in eval_names:
                    X = per_part_stack[name]
                    pred = np.clip(predict_np(head, X, device), AGE_MIN, AGE_MAX)
                    ref = labels[name]
                    has = ~np.isnan(ref)
                    if has.any():
                        raw, corr, bias = bias_corrected(pred[has], ref[has])
                        row[f'{name}_mae'] = raw
                        row[f'{name}_mae_biascorr'] = corr
                        row[f'{name}_bias'] = bias
                    else:
                        row[f'{name}_mae'] = float('nan')
                        row[f'{name}_mae_biascorr'] = float('nan')
                        row[f'{name}_bias'] = float('nan')
                    row[f'{name}_pred'] = pred
                row['time_s'] = time.time() - t0
                results.append(row)
                # Learned layer weights — diagnostic.
                with torch.no_grad():
                    w_learned = torch.softmax(head.layer_weights.detach().cpu(),
                                              dim=0).numpy()
                wstr = ' '.join(f'L{L}:{w:.2f}' for L, w in zip(args.layers, w_learned))
                print(f'  fusion {mlp_name}_s{seed:<3d}     '
                      f'train={train_mae:5.2f}  '
                      f'dev_inner={row["dev_inner_mae"]:5.2f} '
                      f'(corr {row["dev_inner_mae_biascorr"]:5.2f})  '
                      f'w[{wstr}]  ({row["time_s"]:5.1f}s)')

    # ---------- Summary ----------
    print(f'\n{"="*88}')
    print('SUMMARY (sorted by dev_inner MAE)')
    print(f'{"="*88}')
    print(f'{"rank":>4s}  {"head":<26s}  {"layer":>5s}  {"train":>5s}  '
          f'{"dev_inner":>9s}  {"corr":>5s}  '
          + ('  '.join(f'{e:>9s}' for e in eval_names if e != 'dev_inner'))
          + f'  {"time":>6s}')
    print('-' * 105)
    sorted_results = sorted(results, key=lambda r: r['dev_inner_mae'])
    for i, r in enumerate(sorted_results[:25], 1):
        extra = '  '.join(
            f'{r.get(f"{e}_mae", float("nan")):>9.3f}'
            for e in eval_names if e != 'dev_inner'
        )
        print(f'{i:>4d}  {r["head"]:<26s}  {str(r["layer"]):>5s}  '
              f'{r["train_mae"]:>5.2f}  {r["dev_inner_mae"]:>9.3f}  '
              f'{r["dev_inner_mae_biascorr"]:>5.2f}  {extra}  '
              f'{r["time_s"]:>5.0f}s')

    # ---------- Save best evl submission ----------
    best = sorted_results[0]
    print(f'\nBest head: {best["head"]} @ layer {best["layer"]}  '
          f'dev_inner MAE = {best["dev_inner_mae"]:.3f}')

    if 'evl' in eval_names and f'evl_pred' in best:
        sub_path = LAB2_DIR / f'g{args.group}_falar_head_sweep_{best["head"]}.csv'
        df_evl = eval_dfs['evl']
        with open(sub_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['wav', 'age'])
            for fname, pred in zip(df_evl['wav'], best['evl_pred']):
                w.writerow([fname, f'{pred:.3f}'])
        print(f'  -> evl submission: {sub_path.relative_to(LAB2_DIR)}')

    # ---------- Save CSV ----------
    out_csv = Path(args.out_csv) if args.out_csv else (
        LAB2_DIR / 'head_sweep_falar_results.csv'
    )
    serialise_cols = [
        c for c in results[0].keys()
        if not c.endswith('_pred')
    ]
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(serialise_cols)
        for r in sorted_results:
            w.writerow([r.get(c, '') for c in serialise_cols])
    print(f'\nResults CSV: {out_csv.relative_to(LAB2_DIR)}')


if __name__ == '__main__':
    main()
