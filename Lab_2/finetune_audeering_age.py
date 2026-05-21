#!/usr/bin/env python3
"""
Backbone fine-tuning of audeering/wav2vec2-large-robust-24-ft-age-gender
for age regression on our Portuguese lab data.

This is purposely a SMALL, CHEAP experiment designed to answer one question:
  "Does unfreezing the transformer give a real dev-MAE improvement on this
   dataset, or does it just memorise train_small?"

Approach (the same script scales up on a rented GPU):

  - Load `Wav2Vec2ForSequenceClassification` initialised from audeering's
    wav2vec2 backbone weights, with a fresh regression head (single scalar
    output). Audeering's age/gender heads are reported as unexpected keys
    and dropped during loading — they are never used, satisfying the
    "no audeering classifier" constraint.
  - `freeze_feature_encoder()` keeps the CNN frozen (the standard recipe);
    the 24 transformer layers + the new regression head are trainable.
  - HF `Trainer`, custom MSE loss, dev-MAE early-model-selection, linear
    LR schedule with 10% warm-up.

Local defaults are sized for an M-series Mac on MPS with 16 GB RAM:
  batch 2, gradient_accumulation 4 (effective 8), 3 epochs on train_small.
  Expected runtime: ~30-90 min, ~$0.

To scale on a rented GPU (A100 / A10G / 4090 / L4), override:
  --trainset train --epochs 6 --batch-size 8 --grad-accum 2
  ...this is roughly a $1-3 run.

Outputs:
  - dev MAE per epoch printed to stdout + saved to finetune_results.csv
  - best-by-dev-MAE checkpoint at lab2_data/<trainset>/models/finetune_audeering_age/
  - dev.pkl / evl.pkl in the existing sweep format
  - g<group>_<trainset>_finetune_audeering_age.csv  (Kaggle submission)

Usage:
    python finetune_audeering_age.py                 # local default
    python finetune_audeering_age.py --epochs 5
    python finetune_audeering_age.py --trainset train --epochs 6 --batch-size 8 --grad-accum 2
"""

import argparse
import copy
import csv
import gc
import os
import pickle
import sys
import time
from pathlib import Path

# Allow MPS to fall back to CPU for any op without a Metal kernel
# (wav2vec2 has a couple of these — GroupNorm in particular).
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    logging as tf_logging,
)

from pf_tools import create_submission_file


AUD_MODEL_ID = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
AGE_MIN, AGE_MAX = 20.0, 90.0    # post-prediction clip (matches sweep script)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def select_device_str():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_partition_dataframe(datadir, partition):
    """Load <datadir>/<partition>/info.csv into a DataFrame with columns:
    wav, gender, age (numeric or NaN), fileid (basename), abs_path."""
    info_path = datadir / partition / 'info.csv'
    df = pd.read_csv(info_path)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')      # '?' -> NaN
    df['fileid'] = df['wav'].str.replace('.wav', '', regex=False)
    df['abs_path'] = (datadir / partition / 'wav' / df['wav']).astype(str)
    return df


def build_dataset(df, feat_ext, duration=10.0, has_labels=True, desc='loading'):
    """Load every audio file with librosa, run the feature extractor, and
    store the (variable-length) `input_values` directly. Avoids HF's `Audio`
    feature decoder which now requires `torchcodec`; uses the same librosa
    path as the rest of the project. evl rows (no label) get a placeholder
    of 0.0 — we never train on evl, only predict."""
    max_samples = int(16000 * duration)
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        audio, _ = librosa.load(
            row['abs_path'], sr=16000, mono=True, duration=duration,
        )
        inputs = feat_ext(
            audio,
            sampling_rate=16000,
            max_length=max_samples,
            truncation=True,
            padding=False,                  # collator pads per-batch
        )
        rec = {'input_values': inputs['input_values'][0]}
        if has_labels and not pd.isna(row['age']):
            rec['labels'] = float(row['age'])
        else:
            rec['labels'] = 0.0
        records.append(rec)
    return Dataset.from_list(records)


def freeze_lower_layers(model, n_top_layers_to_train):
    """Freeze all but the top N transformer layers of the wav2vec2 backbone.
    The CNN feature encoder is frozen separately via `freeze_feature_encoder`.
    Returns the number of layers frozen (purely for logging)."""
    layers = model.wav2vec2.encoder.layers
    n_total = len(layers)
    n_to_freeze = max(n_total - n_top_layers_to_train, 0)
    for i, layer in enumerate(layers):
        if i < n_to_freeze:
            for p in layer.parameters():
                p.requires_grad = False
    return n_to_freeze, n_total


class W2V2RegressionCollator:
    """Pad variable-length `input_values` to the longest in the batch and
    attach float `labels`. Avoids the wasteful 'pad-everything-to-10s'
    approach when the batch happens to contain short clips."""

    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, features):
        input_features = [{'input_values': f['input_values']} for f in features]
        batch = self.fe.pad(input_features, padding=True, return_tensors='pt')
        batch['labels'] = torch.tensor(
            [f['labels'] for f in features], dtype=torch.float32
        )
        return batch


def make_compute_metrics(y_mean=0.0, y_std=1.0):
    """Factory that returns a compute_metrics callable. The model is trained
    on normalised targets (zero mean, unit std on the training fold), so its
    raw outputs live near 0; here we denormalise them before computing MAE/MSE
    against the raw-age reference labels."""
    def compute_metrics(eval_pred):
        preds_norm = eval_pred.predictions.squeeze(-1)
        preds = preds_norm * y_std + y_mean
        refs = eval_pred.label_ids                     # raw ages, never normalised
        return {
            'mae': float(mean_absolute_error(refs, preds)),
            'mse': float(np.mean((preds - refs) ** 2)),
        }
    return compute_metrics


class RegressionTrainer(Trainer):
    """MSE on the squeezed single-logit output, with target normalisation.

    Without normalisation, raw ages (~50, std ~10) make MSE start at ~2500
    on a fresh random head; gradients explode, default clipping caps the
    step size, and the model never escapes "predict near 0". Normalising
    targets to ~N(0, 1) keeps initial loss on the order of 1 and lets the
    head actually move. Set `trainer.y_mean` / `trainer.y_std` after init."""
    y_mean = 0.0
    y_std = 1.0

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs.pop('labels').float()
        normalised = (labels - self.y_mean) / self.y_std
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = nn.functional.mse_loss(logits, normalised)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

def run(args):
    datadir = HERE / 'lab2_data'
    device = select_device_str()
    print(f'device  : {device}')
    print(f'trainset: {args.trainset}')
    print(f'epochs  : {args.epochs}   batch: {args.batch_size}  '
          f'grad_accum: {args.grad_accum}   '
          f'effective batch: {args.batch_size * args.grad_accum}')
    print(f'lr      : {args.lr}')

    # ----- model -----
    print(f'\n-- loading audeering wav2vec2 (regression head) --')
    t0 = time.time()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(AUD_MODEL_ID)
    tf_logging.set_verbosity_error()        # silence expected unexpected-keys
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        AUD_MODEL_ID,
        num_labels=1,
        problem_type='regression',
    )
    tf_logging.set_verbosity_warning()

    model.freeze_feature_encoder()
    if args.top_layers is not None:
        n_frozen, n_total_layers = freeze_lower_layers(model, args.top_layers)
        print(f'  frozen lower transformer layers: {n_frozen}/{n_total_layers}  '
              f'(training top {args.top_layers} + regression head)')
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'  loaded in {time.time()-t0:.1f}s')
    print(f'  trainable params: {n_trainable/1e6:.1f} M / {n_total/1e6:.1f} M '
          f'({100*n_trainable/n_total:.1f}%)')

    # ----- data -----
    print(f'\n-- building HF datasets (librosa + feature extractor) --')
    df_train = load_partition_dataframe(datadir, args.trainset)
    df_dev   = load_partition_dataframe(datadir, 'dev')
    df_evl   = load_partition_dataframe(datadir, 'evl')
    print(f'  {args.trainset}: {len(df_train)}   dev: {len(df_dev)}   evl: {len(df_evl)}')

    # Target normalisation stats computed on the training fold only. Raw
    # labels stay raw inside the Dataset; the RegressionTrainer normalises
    # them just before computing MSE, and compute_metrics / final inference
    # denormalise predictions back to age in years.
    train_ages = [float(a) for a in df_train['age'].tolist() if not pd.isna(a)]
    y_mean = float(np.mean(train_ages))
    y_std  = float(np.std(train_ages) + 1e-8)
    print(f'  target normalisation: y_mean={y_mean:.2f}  y_std={y_std:.2f}')

    ds = DatasetDict({
        'train': build_dataset(df_train, fe, duration=args.duration,
                               has_labels=True,  desc=f'{args.trainset:>10s}'),
        'dev':   build_dataset(df_dev,   fe, duration=args.duration,
                               has_labels=True,  desc=f'{"dev":>10s}'),
        'evl':   build_dataset(df_evl,   fe, duration=args.duration,
                               has_labels=False, desc=f'{"evl":>10s}'),
    })
    ds.set_format('torch', columns=['input_values', 'labels'])

    # ----- training -----
    suffix = f'_{args.run_id}' if args.run_id else ''
    output_dir = (HERE / 'lab2_data' / args.trainset / 'models'
                  / f'finetune_audeering_age{suffix}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # If --max-epochs is set, use it as the ceiling and let EarlyStopping decide
    # the real stopping point. Otherwise fall back to the fixed --epochs schedule.
    use_early_stop = args.max_epochs is not None
    n_epochs = args.max_epochs if use_early_stop else args.epochs

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 4),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=n_epochs,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        logging_steps=10,
        logging_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='eval_mae',
        greater_is_better=False,
        report_to=[],                       # no wandb / tensorboard
        dataloader_num_workers=0,           # MPS can be picky with workers
        remove_unused_columns=False,
        # only enable fp16 on CUDA; MPS / CPU stick with fp32
        fp16=(device == 'cuda' and not args.no_fp16),
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )

    callbacks = []
    if use_early_stop:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.patience,
            early_stopping_threshold=0.01,
        ))

    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['dev'],
        data_collator=W2V2RegressionCollator(fe),
        compute_metrics=make_compute_metrics(y_mean, y_std),
        callbacks=callbacks,
    )
    trainer.y_mean = y_mean
    trainer.y_std  = y_std

    if use_early_stop:
        print(f'\n-- training (max {n_epochs} epochs, '
              f'early-stop patience={args.patience}) --')
    else:
        print(f'\n-- training ({n_epochs} epochs, no early stopping) --')
    train_t0 = time.time()
    trainer.train()
    train_dt = time.time() - train_t0
    print(f'\n  training finished in {train_dt:.1f}s ({train_dt/60:.1f} min)')

    # ----- evaluation (denormalise model outputs to age in years) -----
    print('\n-- final dev prediction --')
    dev_pred = trainer.predict(ds['dev'])
    dev_hyp = np.clip(
        dev_pred.predictions.squeeze(-1) * y_std + y_mean,
        AGE_MIN, AGE_MAX,
    )
    dev_ref = np.asarray(dev_pred.label_ids, dtype=float)
    dev_mae = float(mean_absolute_error(dev_ref, dev_hyp))
    print(f'  best-model dev MAE: {dev_mae:.3f}')

    print('\n-- evl prediction --')
    evl_pred = trainer.predict(ds['evl'])
    evl_hyp = np.clip(
        evl_pred.predictions.squeeze(-1) * y_std + y_mean,
        AGE_MIN, AGE_MAX,
    )

    # ----- submission CSV (same pkl format the sweep script uses) -----
    resdir = output_dir / 'final'
    resdir.mkdir(parents=True, exist_ok=True)
    with open(resdir / 'dev.pkl', 'wb') as f:
        pickle.dump({'hyp': dev_hyp,
                     'fileids': np.asarray(df_dev['fileid'].tolist())}, f)
    with open(resdir / 'evl.pkl', 'wb') as f:
        pickle.dump({'hyp': evl_hyp,
                     'fileids': np.asarray(df_evl['fileid'].tolist())}, f)
    sub_path = HERE / f'g{args.group}_{args.trainset}_finetune_audeering_age{suffix}.csv'
    create_submission_file(str(resdir), str(sub_path))
    print(f'  submission -> {sub_path.name}')

    # ----- per-epoch trajectory (the actual insight) -----
    print('\n========= DEV MAE TRAJECTORY =========')
    train_losses, dev_maes = {}, {}
    for entry in trainer.state.log_history:
        if 'eval_mae' in entry and 'epoch' in entry:
            dev_maes[round(entry['epoch'], 4)] = entry['eval_mae']
        elif 'loss' in entry and 'epoch' in entry and 'eval_mae' not in entry:
            # Keep the most recent train-loss log per epoch (overwrites within epoch).
            train_losses[round(entry['epoch'], 4)] = entry['loss']

    print(f'{"epoch":>6s}  {"train_loss":>12s}  {"dev MAE":>8s}')
    print('-' * 34)
    all_epochs = sorted(set(list(train_losses.keys()) + list(dev_maes.keys())))
    for epoch in all_epochs:
        tl = train_losses.get(epoch)
        dm = dev_maes.get(epoch)
        tl_s = f'{tl:12.4f}' if tl is not None else '           -'
        dm_s = f'{dm:8.3f}' if dm is not None else '       -'
        print(f'{epoch:6.2f}  {tl_s}  {dm_s}')

    # Verdict — what to do with the result.
    sorted_items = sorted(dev_maes.items())
    sorted_devs = [v for _, v in sorted_items]
    if sorted_devs:
        baseline = 6.0   # frozen-feature ceiling on this dataset
        first, last = sorted_devs[0], sorted_devs[-1]
        best = min(sorted_devs)
        best_epoch = min(sorted_items, key=lambda kv: kv[1])[0]
        n_epochs_done = len(sorted_devs)
        stop_msg = ''
        if use_early_stop and n_epochs_done < n_epochs:
            stop_msg = (f' (early-stopped at epoch {n_epochs_done} '
                        f'of ceiling {n_epochs})')
        print(f'\n  frozen-feature reference (sweep ensemble): {baseline:.2f}')
        print(f'  this run: first epoch dev = {first:.3f}, '
              f'final = {last:.3f}, best = {best:.3f} '
              f'@ epoch {best_epoch:g}{stop_msg}')
        if best < baseline - 0.5:
            print('  -> meaningful improvement. Fine-tuning works for our data; '
                  'rent the GPU and scale up.')
        elif best < baseline - 0.1:
            print('  -> small improvement. Worth scaling once on a GPU; '
                  'try more epochs and larger batch.')
        else:
            print('  -> no clear gain over frozen features. Debug LR / freeze '
                  'policy before spending on GPU.')

    # CSV trajectory for offline plotting / comparison.
    csv_path = HERE / f'finetune_results{suffix}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'dev_mae'])
        for epoch in all_epochs:
            w.writerow([
                epoch,
                f'{train_losses[epoch]:.6f}' if epoch in train_losses else '',
                f'{dev_maes[epoch]:.6f}'     if epoch in dev_maes     else '',
            ])
    print(f'\nTrajectory CSV: {csv_path.name}')

    # Free GPU memory before the next grid cell (no-op when run stand-alone).
    del trainer, model, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dev_mae


# ---------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------

# Grid cells. Each row: (run_id, lr, epochs, top_layers, seed).
# `epochs` is interpreted as the EarlyStopping ceiling when --grid is run
# with --max-epochs; otherwise it's the fixed number of epochs.
GRID = [
    # ----- Phase A: seed sweep at the winning recipe (uncertainty on 5.19) -----
    ('A_lr5e5_s42',  5e-5, 40, None,  42),
    ('A_lr5e5_s7',   5e-5, 40, None,   7),
    ('A_lr5e5_s13',  5e-5, 40, None,  13),
    ('A_lr5e5_s21',  5e-5, 40, None,  21),
    ('A_lr5e5_s99',  5e-5, 40, None,  99),

    # ----- Phase B: LR neighbourhood at seed 42 -----
    ('B_lr3e5_s42',  3e-5, 40, None,  42),
    ('B_lr4e5_s42',  4e-5, 40, None,  42),
    ('B_lr6e5_s42',  6e-5, 40, None,  42),
    ('B_lr7e5_s42',  7e-5, 40, None,  42),
    ('B_lr8e5_s42',  8e-5, 40, None,  42),

    # ----- Phase D: lower LR with high ceiling (1e-5 deserved its fair shot) -----
    ('D_lr1e5_s42',  1e-5, 60, None,  42),
    ('D_lr2e5_s42',  2e-5, 60, None,  42),
]


def run_grid(args):
    print(f'\n{"="*80}')
    print(f'GRID SEARCH — {len(GRID)} runs')
    print(f'{"="*80}')
    for i, (run_id, lr, epochs, top_n, seed) in enumerate(GRID, 1):
        print(f'  {i:>2d}. {run_id:<14s}  lr={lr:.0e}  epochs={epochs}  '
              f'top_layers={top_n}  seed={seed}')

    grid_csv = HERE / f'grid_results_{args.trainset}.csv'
    results = []

    for i, (run_id, lr, epochs, top_n, seed) in enumerate(GRID, 1):
        print(f'\n\n{"="*80}')
        print(f'GRID CELL {i}/{len(GRID)}: {run_id}')
        print(f'  lr={lr}  epochs={epochs}  top_layers={top_n}  seed={seed}')
        print(f'{"="*80}')

        cell_args = copy.deepcopy(args)
        cell_args.lr = lr
        cell_args.epochs = epochs
        # In grid mode the per-cell `epochs` is used as the early-stopping
        # ceiling unless the user explicitly disabled early stopping via
        # --no-early-stop on the grid invocation.
        if getattr(args, 'no_early_stop', False):
            cell_args.max_epochs = None
        else:
            cell_args.max_epochs = epochs
        cell_args.patience = getattr(args, 'patience', 5)
        cell_args.top_layers = top_n
        cell_args.seed = seed
        cell_args.run_id = run_id
        cell_args.grid = False           # do not recurse

        t0 = time.time()
        try:
            dev_mae = run(cell_args)
            status = 'ok'
        except Exception as e:
            print(f'\n  -> {run_id} FAILED: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()
            dev_mae = float('nan')
            status = f'failed:{type(e).__name__}'
        dt = time.time() - t0

        results.append({
            'run_id': run_id, 'lr': lr, 'epochs': epochs,
            'top_layers': top_n if top_n is not None else 'all',
            'seed': seed, 'dev_mae': dev_mae, 'time_s': dt, 'status': status,
        })

        # Write the running CSV after every cell so an interrupted run keeps its results.
        with open(grid_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['run_id', 'lr', 'epochs', 'top_layers', 'seed',
                        'dev_mae', 'time_s', 'status'])
            for r in results:
                dm = f'{r["dev_mae"]:.4f}' if not np.isnan(r["dev_mae"]) else 'NaN'
                w.writerow([r['run_id'], r['lr'], r['epochs'], r['top_layers'],
                            r['seed'], dm, f'{r["time_s"]:.1f}', r['status']])

    # Final summary
    print(f'\n\n{"="*80}')
    print('GRID SUMMARY (sorted by dev MAE)')
    print(f'{"="*80}')
    valid = [r for r in results if not np.isnan(r['dev_mae'])]
    valid.sort(key=lambda r: r['dev_mae'])
    print(f'{"rank":>4s}  {"run_id":<14s}  {"lr":>8s}  {"ep":>3s}  '
          f'{"top":>5s}  {"seed":>4s}  {"dev MAE":>8s}  {"time":>8s}')
    print('-' * 70)
    for i, r in enumerate(valid, 1):
        print(f'{i:>4d}  {r["run_id"]:<14s}  {r["lr"]:>8.0e}  {r["epochs"]:>3d}  '
              f'{str(r["top_layers"]):>5s}  {r["seed"]:>4d}  '
              f'{r["dev_mae"]:>8.3f}  {r["time_s"]:>7.0f}s')
    failed = [r for r in results if np.isnan(r['dev_mae'])]
    if failed:
        print(f'\nFailed cells ({len(failed)}):')
        for r in failed:
            print(f'  {r["run_id"]:<14s}  {r["status"]}')
    print(f'\nGrid CSV: {grid_csv.name}')
    if valid:
        best = valid[0]
        print(f'\nBest: {best["run_id"]} -> dev MAE {best["dev_mae"]:.3f}.  '
              f'Submission: g{args.group}_{args.trainset}_finetune_audeering_age_'
              f'{best["run_id"]}.csv')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--trainset', default='train_small',
                        choices=('train_small', 'train'),
                        help='training partition (default train_small)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of training epochs (default 3). '
                             'Ignored when --max-epochs is given.')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='if set, enables early stopping. Training runs up '
                             'to this many epochs but stops once dev MAE stops '
                             'improving (see --patience). Recommended: 40.')
    parser.add_argument('--patience', type=int, default=5,
                        help='early-stop patience: number of consecutive eval '
                             'rounds without dev-MAE improvement (>=0.01) before '
                             'stopping. Only used with --max-epochs. Default 5.')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='per-device batch size (default 2 for MPS; 8-16 on a GPU)')
    parser.add_argument('--grad-accum', type=int, default=4,
                        help='gradient accumulation steps. '
                             'effective batch = batch_size * grad_accum (default 4)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='AdamW learning rate (default 5e-5)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='max audio duration in seconds (default 10)')
    parser.add_argument('--group', default='07',
                        help='student group prefix for submission CSV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='trade compute for memory — useful on small-VRAM GPUs')
    parser.add_argument('--no-fp16', action='store_true',
                        help='disable fp16 even on CUDA (debugging)')
    parser.add_argument('--top-layers', type=int, default=None,
                        help='if set, freeze all but the top N transformer '
                             'layers (in addition to the always-frozen CNN). '
                             'Reduces trainable params and memory. Default: '
                             'all 24 layers trainable.')
    parser.add_argument('--run-id', default=None,
                        help='unique tag appended to output paths (checkpoint '
                             'dir, submission CSV, trajectory CSV). Required '
                             'in --grid mode (set automatically).')
    parser.add_argument('--grid', action='store_true',
                        help='run the predefined GRID of fine-tune configs. '
                             'Cells use early stopping; the per-cell `epochs` '
                             'field is treated as the early-stop ceiling.')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='in --grid mode, disable early stopping and run '
                             'each cell for its fixed number of epochs.')
    args = parser.parse_args()
    if args.grid:
        run_grid(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
