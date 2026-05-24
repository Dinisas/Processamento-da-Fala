#!/usr/bin/env python3
"""External-test inference for the audeering wav2vec2 age finetune.

Loads one or more saved checkpoints from `lab2_data/<trainset>/models/
finetune_audeering_age_<run_id>/checkpoint-N/` and runs forward pass on a
target partition (default: falar_dev). Computes MAE if labels are present
in the target's info.csv.

The point: yesterday's B_lr6e5_s42 checkpoint (lab-dev MAE 4.91) and today's
E_disjoint_s42 checkpoint (dev_inner MAE 6.42) were both trained from lab
`train`, which is a downsampled subset of FaLAR's train split. FaLAR's
train and dev splits are speaker-disjoint, so falar_dev (~7300 utts with
real labels) is a CLEAN external test for both models — no shared speakers
with either training fold.

The comparison answers: "does the leaky 4.91 model actually generalise
better than the honest 6.42 model on truly held-out data, or was its dev
advantage purely speaker-memorisation?"

For each run:
  * load the latest checkpoint-N (or --checkpoint to pick one)
  * recompute y_mean / y_std from the *training* fold used by that run
    (lab `train` for the leaky model, train_inner via the split manifest
    for the speaker-disjoint model) so predictions are denormalised the
    same way training did them
  * forward pass on the target partition, batched
  * write predictions to <output_dir>/falar_dev_pred.pkl and a CSV
  * report MAE / MSE / mean absolute residual

Usage:
    # compare both 4.91 and 6.42 models on falar_dev
    python finetuning/infer_external.py \
        --runs B_lr6e5_s42:none E_disjoint_s42:train/speaker_disjoint_split.json \
        --target falar_dev

    # single-run
    python finetuning/infer_external.py --runs B_lr6e5_s42:none --target falar_dev
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LAB2_DIR))

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    logging as tf_logging,
)

from finetune_audeering_age import (
    AUD_MODEL_ID, AGE_MIN, AGE_MAX,
    load_partition_dataframe, select_device_str,
)


def parse_run_spec(spec: str) -> tuple[str, str | None]:
    """Parse 'run_id:manifest_path' or 'run_id:none'."""
    if ':' not in spec:
        return spec, None
    rid, manifest = spec.split(':', 1)
    return rid, (None if manifest.lower() in ('none', '', '-') else manifest)


def latest_checkpoint(run_dir: Path) -> Path:
    """Pick the most recent checkpoint-N subdir (highest N)."""
    candidates = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('checkpoint-')),
        key=lambda p: int(p.name.split('-')[1]),
    )
    if not candidates:
        raise FileNotFoundError(f'no checkpoint-N subdir found in {run_dir}')
    return candidates[-1]


def compute_y_stats(args, split_manifest: str | None) -> tuple[float, float]:
    """Recompute y_mean / y_std the exact same way the training run did,
    so we can denormalise predictions consistently."""
    datadir = LAB2_DIR / 'lab2_data'
    df_train = load_partition_dataframe(datadir, args.trainset)
    if split_manifest:
        manifest_path = Path(split_manifest)
        if not manifest_path.is_absolute():
            manifest_path = LAB2_DIR / 'lab2_data' / manifest_path
        with open(manifest_path) as f:
            manifest = json.load(f)
        train_inner = set(manifest['train_inner_uuids'])
        df_train = df_train[df_train['fileid'].isin(train_inner)]
    ages = [float(a) for a in df_train['age'].tolist() if not pd.isna(a)]
    return float(np.mean(ages)), float(np.std(ages) + 1e-8)


def load_target(args) -> pd.DataFrame:
    """Load target partition info.csv + add abs_path. Handles both UUID-style
    (lab evl) and structured (FaLAR) wav filenames."""
    datadir = LAB2_DIR / 'lab2_data'
    info_path = datadir / args.target / 'info.csv'
    if not info_path.is_file():
        raise FileNotFoundError(f'missing {info_path}')
    df = pd.read_csv(info_path)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['fileid'] = df['wav'].str.replace('.wav', '', regex=False)
    df['abs_path'] = (datadir / args.target / 'wav' / df['wav']).astype(str)
    return df


def predict_one(model, fe, df, device, duration, batch_size):
    """Batched forward pass. Returns (logits_array, ages_array_or_None)."""
    model.eval()
    model.to(device)
    max_samples = int(16000 * duration)
    logits_list = []
    n = len(df)
    rows = df.to_dict('records')

    for start in tqdm(range(0, n, batch_size), desc='inference'):
        batch_rows = rows[start:start + batch_size]
        wavs = []
        for r in batch_rows:
            audio, _ = librosa.load(
                r['abs_path'], sr=16000, mono=True, duration=duration,
            )
            wavs.append(audio.astype(np.float32))
        inputs = fe(
            wavs, sampling_rate=16000, padding=True,
            return_tensors='pt', max_length=max_samples, truncation=True,
        )
        with torch.no_grad():
            input_values = inputs['input_values'].to(device)
            attn = inputs.get('attention_mask')
            kwargs = {'input_values': input_values}
            if attn is not None:
                kwargs['attention_mask'] = attn.to(device)
            out = model(**kwargs)
        logits = out.logits.squeeze(-1).detach().cpu().numpy()
        logits_list.append(logits)

    logits_all = np.concatenate(logits_list, axis=0).astype(np.float32)
    has_age = ~df['age'].isna()
    ages = df.loc[has_age, 'age'].to_numpy(dtype=float)
    has_age_mask = has_age.to_numpy()
    return logits_all, ages, has_age_mask


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--runs', nargs='+', required=True,
                   help='one or more run specs as `<run_id>:<manifest_or_none>`. '
                        'Example: B_lr6e5_s42:none E_disjoint_s42:train/'
                        'speaker_disjoint_split.json')
    p.add_argument('--target', default='falar_dev',
                   help='target partition under lab2_data/ (default falar_dev)')
    p.add_argument('--trainset', default='train',
                   help='trainset the runs were trained on (default train)')
    p.add_argument('--duration', type=float, default=10.0,
                   help='max audio duration in seconds (default 10, same as training)')
    p.add_argument('--batch-size', type=int, default=8,
                   help='inference batch size (default 8)')
    p.add_argument('--checkpoint', default=None,
                   help='override checkpoint subdir name (e.g. checkpoint-2030); '
                        'default = highest-numbered checkpoint in the run dir')
    p.add_argument('--max-rows', type=int, default=None,
                   help='cap target rows for a quick smoke test (default: all)')
    p.add_argument('--out-dir', default=None,
                   help='where to write per-run prediction files (default: '
                        'next to each run dir under final_<target>/)')
    args = p.parse_args()

    datadir = LAB2_DIR / 'lab2_data'
    device = select_device_str()
    print(f'device   : {device}')
    print(f'target   : {args.target}')
    print(f'trainset : {args.trainset}   (for y_mean/y_std recompute)')
    print(f'runs     :')
    for r in args.runs:
        rid, mf = parse_run_spec(r)
        print(f'    {rid:<24s}  manifest = {mf or "-"}')

    # ---------- Load target once ----------
    df_target = load_target(args)
    if args.max_rows:
        df_target = df_target.head(args.max_rows).reset_index(drop=True)
    has_age_count = int((~df_target['age'].isna()).sum())
    print(f'\ntarget rows: {len(df_target)}   '
          f'with age labels: {has_age_count}   '
          f'(MAE computable: {"yes" if has_age_count else "no"})')

    # Shared feature extractor (audeering's).
    fe = Wav2Vec2FeatureExtractor.from_pretrained(AUD_MODEL_ID)

    # ---------- Run each model ----------
    all_results = []
    for r in args.runs:
        rid, manifest = parse_run_spec(r)
        run_dir = datadir / args.trainset / 'models' / f'finetune_audeering_age_{rid}'
        if not run_dir.is_dir():
            print(f'\n!! {rid}: missing run dir {run_dir} — skipping')
            continue
        ck = (run_dir / args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
        print(f'\n{"="*72}')
        print(f'RUN {rid}')
        print(f'  checkpoint: {ck.relative_to(LAB2_DIR)}')

        # y_stats consistent with how this run was trained.
        y_mean, y_std = compute_y_stats(args, manifest)
        print(f'  y_mean={y_mean:.3f}  y_std={y_std:.3f}')

        tf_logging.set_verbosity_error()
        model = Wav2Vec2ForSequenceClassification.from_pretrained(str(ck))
        tf_logging.set_verbosity_warning()

        t0 = time.time()
        logits, ref_ages, has_age_mask = predict_one(
            model, fe, df_target, device, args.duration, args.batch_size,
        )
        dt = time.time() - t0
        # Denormalise + clip to plausible age range.
        preds = logits * y_std + y_mean
        preds = np.clip(preds, AGE_MIN, AGE_MAX)

        # MAE / MSE on labelled rows only.
        if has_age_count:
            preds_with_age = preds[has_age_mask]
            mae = float(mean_absolute_error(ref_ages, preds_with_age))
            mse = float(np.mean((preds_with_age - ref_ages) ** 2))
            resid = preds_with_age - ref_ages
            mean_signed = float(resid.mean())
            print(f'  -> {args.target} MAE  : {mae:.3f}')
            print(f'  -> {args.target} MSE  : {mse:.3f}')
            print(f'  -> mean signed bias  : {mean_signed:+.3f} years '
                  f'({"over-predicts" if mean_signed > 0 else "under-predicts"})')
        else:
            mae = mse = mean_signed = float('nan')
            print(f'  (no labels in target — only predictions saved)')
        print(f'  inference time       : {dt:.1f}s ({dt/60:.1f} min)')

        # Persist predictions.
        out_dir = (Path(args.out_dir) if args.out_dir
                   else run_dir / f'final_{args.target}')
        out_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = out_dir / f'{args.target}_pred.pkl'
        csv_path = out_dir / f'{args.target}_pred.csv'
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'hyp': preds,
                'fileids': df_target['fileid'].to_numpy(),
                'wav': df_target['wav'].to_numpy(),
                'y_mean': y_mean, 'y_std': y_std,
                'run_id': rid,
                'checkpoint': str(ck.relative_to(LAB2_DIR)),
            }, f)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['wav', 'age_pred', 'age_ref'])
            for i, row in df_target.iterrows():
                ref = '' if pd.isna(row['age']) else f'{row["age"]:.2f}'
                w.writerow([row['wav'], f'{preds[i]:.3f}', ref])
        print(f'  saved -> {pkl_path.relative_to(LAB2_DIR)}')
        print(f'  saved -> {csv_path.relative_to(LAB2_DIR)}')

        all_results.append({
            'run_id': rid,
            'checkpoint': str(ck.relative_to(LAB2_DIR)),
            'mae': mae,
            'mse': mse,
            'mean_signed_bias': mean_signed,
            'n_eval': int(has_age_mask.sum()),
            'time_s': dt,
        })

        # Free up memory between runs.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- Summary ----------
    if all_results:
        print(f'\n{"="*72}')
        print(f'SUMMARY on {args.target}  (n={all_results[0]["n_eval"]})')
        print(f'{"="*72}')
        print(f'  {"run_id":<24s}  {"MAE":>7s}  {"MSE":>7s}  {"bias":>7s}  {"time":>7s}')
        print('-' * 70)
        for r in all_results:
            mae_s = f'{r["mae"]:7.3f}' if not np.isnan(r['mae']) else '   n/a'
            mse_s = f'{r["mse"]:7.2f}' if not np.isnan(r['mse']) else '   n/a'
            bias_s = (f'{r["mean_signed_bias"]:+7.2f}'
                      if not np.isnan(r['mean_signed_bias']) else '   n/a')
            print(f'  {r["run_id"]:<24s}  {mae_s}  {mse_s}  {bias_s}  '
                  f'{r["time_s"]:>6.0f}s')

        # If two runs, print the delta — that's the headline.
        if len(all_results) == 2 and all(not np.isnan(r['mae']) for r in all_results):
            d = all_results[0]['mae'] - all_results[1]['mae']
            sign = ('worse' if d > 0 else 'better')
            print(f'\n  {all_results[0]["run_id"]} is {abs(d):.3f} MAE {sign} '
                  f'than {all_results[1]["run_id"]} on {args.target}.')


if __name__ == '__main__':
    main()
