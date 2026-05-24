#!/usr/bin/env python3
"""
Load a saved fine-tuned audeering age checkpoint and run inference on ANY
partition (lab dev, lab evl, falar_dev, ...).

The point: you don't have to re-train just to change which dev set you score
against. Train once -> infer many.

Usage
    # Last night's adversarial model, evaluated on the 117-row lab dev
    # (this replaces the formal "Run A" experiment with a 5-minute inference):
    python finetuning/eval_checkpoint.py \\
        --run-id G_adv_falar_s42 --trainset big_train_falar \\
        --split-manifest lab2_data/big_train_falar/falar_speaker_split.json \\
        --on dev

    # Same model on the 7,319-row diagnostic FalAR dev:
    python finetuning/eval_checkpoint.py \\
        --run-id G_adv_falar_s42 --trainset big_train_falar \\
        --split-manifest lab2_data/big_train_falar/falar_speaker_split.json \\
        --on falar_dev

    # Old 4.91 baseline checkpoint, evaluated on lab evl for a submission CSV:
    python finetuning/eval_checkpoint.py \\
        --run-id B_lr6e5_s42 --trainset train \\
        --on evl

Outputs
    - dev.pkl (or <on>.pkl) under  <checkpoint>/eval_<on>/
    - submission CSV at           Lab_2/g<group>_eval_<run-id>_on_<on>.csv
    - MAE + RMSE printed to stdout (skipped on unlabelled partitions like evl)
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

# MPS fallback for any op without a Metal kernel.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LAB2_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    logging as tf_logging,
)

# Reuse helpers from the finetune script so the eval path is byte-for-byte
# the same as training (same audio duration, same FE config, same collator).
from finetune_audeering_age import (
    AUD_MODEL_ID,
    AGE_MIN,
    AGE_MAX,
    W2V2RegressionCollator,
    build_dataset,
    load_partition_dataframe,
    select_device_str,
)


def find_checkpoint(run_dir: Path) -> Path:
    """Locate the saved weights inside a run dir.
    Order of preference: 'final/', then 'final_<anything>/', then the
    highest-numbered 'checkpoint-N/'. Falls back to run_dir itself if it
    already has a config.json (already a checkpoint dir)."""
    if (run_dir / 'config.json').is_file():
        return run_dir
    if (run_dir / 'final' / 'config.json').is_file():
        return run_dir / 'final'
    final_dirs = sorted(p for p in run_dir.glob('final*') if p.is_dir())
    for p in final_dirs:
        if (p / 'config.json').is_file():
            return p
    checkpoints = sorted(
        (p for p in run_dir.glob('checkpoint-*') if p.is_dir()),
        key=lambda p: int(p.name.split('-')[-1]),
    )
    for p in reversed(checkpoints):
        if (p / 'config.json').is_file():
            return p
    sys.exit(f'No usable checkpoint found inside {run_dir}')


def compute_norm_stats(datadir: Path, trainset: str, split_manifest):
    """Reproduce the y_mean / y_std that the model was trained with.
    If --split-manifest was used during training, recompute on the train_inner
    carve (otherwise the stats drift and predictions denormalise to wrong ages)."""
    df = load_partition_dataframe(datadir, trainset)
    n_total = len(df)
    if split_manifest:
        mpath = Path(split_manifest)
        if not mpath.is_absolute():
            mpath = LAB2_DIR / mpath
        with open(mpath) as f:
            manifest = json.load(f)
        train_inner = set(manifest['train_inner_uuids'])
        df = df[df['fileid'].isin(train_inner)].reset_index(drop=True)
    ages = pd.to_numeric(df['age'], errors='coerce').dropna().to_numpy()
    y_mean = float(np.mean(ages))
    y_std = float(np.std(ages) + 1e-8)
    return y_mean, y_std, len(ages), n_total


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--run-id', required=True,
                    help='run id used during training (e.g. G_adv_falar_s42). '
                         'Checkpoint located at '
                         'lab2_data/<trainset>/models/finetune_audeering_age_<run-id>/')
    ap.add_argument('--trainset', required=True,
                    choices=('train_small', 'train', 'big_train_falar'),
                    help='the partition the model was trained on (defines '
                         'where the checkpoint lives + the y_mean/y_std stats)')
    ap.add_argument('--on', required=True,
                    help='partition to evaluate on (dev, evl, falar_dev, ...)')
    ap.add_argument('--split-manifest', default=None,
                    help='if training used --split-manifest, pass it here too '
                         'so y_mean/y_std are computed from the same '
                         'train_inner carve (otherwise denormalisation is wrong)')
    ap.add_argument('--subset', default='all',
                    choices=('all', 'train_inner', 'dev_inner'),
                    help='when --split-manifest is set AND --on matches the '
                         'trainset, filter the partition to just the manifest '
                         'carve. Pass "train_inner" to compute the model\'s MAE '
                         'on its training rows (the memorisation floor — needed '
                         'for an overfitting diagnostic). Default "all" keeps '
                         'every row in --on.')
    ap.add_argument('--duration', type=float, default=10.0,
                    help='max audio duration in seconds (must match training)')
    ap.add_argument('--batch-size', type=int, default=8,
                    help='eval batch size (default 8)')
    ap.add_argument('--checkpoint', default=None,
                    help='override the auto-located checkpoint dir (advanced)')
    ap.add_argument('--out-csv', default=None,
                    help='where to write the submission CSV '
                         '(default: Lab_2/g<group>_eval_<run-id>_on_<on>.csv)')
    ap.add_argument('--group', default='07')
    args = ap.parse_args()

    datadir = LAB2_DIR / 'lab2_data'
    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
        if not ckpt_dir.is_absolute():
            ckpt_dir = LAB2_DIR / ckpt_dir
    else:
        run_dir = (datadir / args.trainset / 'models'
                   / f'finetune_audeering_age_{args.run_id}')
        if not run_dir.is_dir():
            sys.exit(f'Run dir not found: {run_dir}\n'
                     f'  Available runs under {run_dir.parent}:\n  '
                     + '\n  '.join(sorted(p.name for p in run_dir.parent.glob('*')))
                       if run_dir.parent.is_dir() else '')
        ckpt_dir = find_checkpoint(run_dir)

    print(f'checkpoint : {ckpt_dir}')

    print(f'\n-- computing normalisation stats --')
    y_mean, y_std, n_used, n_total = compute_norm_stats(
        datadir, args.trainset, args.split_manifest,
    )
    print(f'  trainset {args.trainset!r}: {n_total} total rows  -> {n_used} used '
          + ('(carved by manifest)' if args.split_manifest else '(no carve)'))
    print(f'  y_mean = {y_mean:.4f}   y_std = {y_std:.4f}')

    device = select_device_str()
    print(f'\n-- loading model on {device} --')
    fe = Wav2Vec2FeatureExtractor.from_pretrained(AUD_MODEL_ID)
    # Suppress the unexpected-keys warning that fires when loading a
    # speaker-adversarial checkpoint into the base class (it harmlessly
    # drops the speaker_classifier head — we never need it for inference).
    tf_logging.set_verbosity_error()
    model = Wav2Vec2ForSequenceClassification.from_pretrained(str(ckpt_dir))
    tf_logging.set_verbosity_warning()
    # Force regression. The saved config has `problem_type=regression` but
    # `num_labels=None`, so HF re-derives num_labels from leftover id2label
    # entries inherited from audeering's original 3-head checkpoint. That
    # mis-routes the model's internal loss to CrossEntropy (which barfs on
    # Float labels). Pinning both fields here is belt-and-suspenders; we also
    # skip labels entirely in the inference loop below so no loss is computed.
    model.config.problem_type = 'regression'
    model.config.num_labels = 1
    model.eval()
    model.to(device)

    print(f'\n-- building eval dataset on partition {args.on!r} --')
    df_eval = load_partition_dataframe(datadir, args.on)
    print(f'  rows: {len(df_eval)}')
    has_labels = df_eval['age'].notna().any()
    print(f'  labelled: {has_labels}')

    ds_eval = build_dataset(
        df_eval, fe, duration=args.duration,
        has_labels=has_labels, desc=f'{args.on:>10s}',
    )
    # W2V2LazyDataset is a plain torch Dataset; it's NOT an HF datasets.Dataset,
    # so no .set_format() needed. The collator handles per-batch tensor packing.

    # Manual inference loop. We deliberately strip 'labels' from each batch
    # before calling the model so HF's internal loss path (which routes by
    # the saved config and can mis-fire on regression checkpoints with
    # leftover id2label) never gets invoked. We only want logits.
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    out_dir = ckpt_dir / f'eval_{args.on}'
    out_dir.mkdir(parents=True, exist_ok=True)

    collator = W2V2RegressionCollator(fe)
    loader = DataLoader(
        ds_eval, batch_size=args.batch_size,
        shuffle=False, num_workers=0, collate_fn=collator,
    )

    print(f'\n-- running inference --')
    all_preds = []
    use_amp = (device == 'cuda')
    with torch.no_grad():
        for batch in tqdm(loader, desc='infer', unit='batch'):
            batch.pop('labels', None)        # do not let the model compute loss
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(**batch)
            else:
                out = model(**batch)
            logits = out.logits.float().squeeze(-1).cpu().numpy()
            all_preds.append(logits)
    preds_norm = np.concatenate(all_preds).squeeze()

    # Denormalise + clip to the project's age range (matches the sweep scripts).
    preds = preds_norm * y_std + y_mean
    preds = np.clip(preds, AGE_MIN, AGE_MAX)
    fileids = df_eval['fileid'].to_numpy()

    print(f'\n========= RESULTS =========')
    print(f'  run_id     : {args.run_id}')
    print(f'  on         : {args.on}  ({len(preds)} rows)')
    print(f'  preds      : mean={preds.mean():.2f}  std={preds.std():.2f}  '
          f'min={preds.min():.1f}  max={preds.max():.1f}')

    if has_labels:
        ages_true = pd.to_numeric(df_eval['age'], errors='coerce').to_numpy()
        valid = ~np.isnan(ages_true)
        mae = float(mean_absolute_error(ages_true[valid], preds[valid]))
        rmse = float(np.sqrt(np.mean((ages_true[valid] - preds[valid]) ** 2)))
        bias = float(np.mean(preds[valid] - ages_true[valid]))
        print(f'  MAE        : {mae:.4f}')
        print(f'  RMSE       : {rmse:.4f}')
        print(f'  bias (pred-true mean): {bias:+.2f}   (n={int(valid.sum())})')
    else:
        print(f'  (unlabelled partition; MAE skipped)')

    # Pickle in the same format the sweep / finetune scripts use.
    pkl_path = out_dir / f'{args.on}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump({'hyp': preds, 'fileids': fileids}, f)
    print(f'\n  pickle -> {pkl_path.relative_to(LAB2_DIR)}')

    # Submission-style CSV.
    csv_path = (Path(args.out_csv) if args.out_csv else
                LAB2_DIR / f'g{args.group}_eval_{args.run_id}_on_{args.on}.csv')
    if not csv_path.is_absolute():
        csv_path = LAB2_DIR / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['fileId', 'Age'])
        for fid, p in zip(fileids, preds):
            w.writerow([fid, f'{float(p):.1f}'])
    print(f'  csv    -> {csv_path.relative_to(LAB2_DIR)}')


if __name__ == '__main__':
    main()
