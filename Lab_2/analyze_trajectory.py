"""Quick overfitting analysis for finetune trajectory CSVs.

Reads a finetune_results_*.csv (per-step train_loss + per-epoch dev_mae),
collapses train_loss to a per-epoch mean, converts it to an approximate
train MAE in age units, and prints the train-vs-dev gap alongside dev MAE.

Train loss in the CSV is MSE on z-scored targets, so

    train_RMSE_age  =  sqrt(train_loss_mean) * y_std
    train_MAE_age   ~= train_RMSE_age / 1.25   # normal-residual factor

The 1.25 factor is the RMSE/MAE ratio for normally distributed residuals; it
slightly underestimates train MAE when residuals are heavy-tailed but is good
enough to see the gap trend.

Usage:
    python analyze_trajectory.py finetune_results_es_lr5e5.csv
    python analyze_trajectory.py finetune_results_es_lr5e5.csv --y-std 10.44
    python analyze_trajectory.py f1.csv f2.csv f3.csv         # compare runs
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


def load_trajectory(csv_path: Path, y_std: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Per-epoch mean of train loss (treat each logged step within an epoch
    # as a sample of that epoch's loss). Floor the epoch fractional part to
    # group all sub-epoch logs under the integer epoch they belong to.
    train_df = df.dropna(subset=['train_loss']).copy()
    # HF logs at epoch=X.YY come from the step that just completed; it belongs
    # to the epoch you'd round UP to. ceil handles both mid-epoch (e.g. 0.05 -> 1)
    # and boundary (e.g. 1.00 -> 1) cases. Treat the rare epoch=0 log (some
    # warmup configs log at step 0) as belonging to epoch 1.
    train_df['epoch_int'] = train_df['epoch'].apply(
        lambda x: max(1, math.ceil(x))
    ).astype(int)
    train_per_epoch = train_df.groupby('epoch_int')['train_loss'].mean()

    # Dev MAE is logged at integer epoch boundaries.
    dev_df = df.dropna(subset=['dev_mae']).copy()
    dev_df['epoch_int'] = dev_df['epoch'].round().astype(int)
    dev_per_epoch = dev_df.groupby('epoch_int')['dev_mae'].mean()

    out = pd.DataFrame({
        'train_loss_norm': train_per_epoch,
        'dev_mae':         dev_per_epoch,
    }).dropna()

    out['train_rmse_age'] = (out['train_loss_norm'] ** 0.5) * y_std
    out['train_mae_est']  = out['train_rmse_age'] / 1.25
    out['gap']            = out['dev_mae'] - out['train_mae_est']
    return out


def report(name: str, traj: pd.DataFrame) -> None:
    print(f'\n{"="*72}')
    print(f'{name}')
    print(f'{"="*72}')
    print(f'{"epoch":>5s}  {"train_loss":>11s}  {"train_MAE":>10s}  '
          f'{"dev_MAE":>8s}  {"gap":>6s}')
    print('-' * 50)
    best_epoch = traj['dev_mae'].idxmin()
    best_dev   = traj['dev_mae'].min()
    for ep, row in traj.iterrows():
        marker = '  <-- best' if ep == best_epoch else ''
        print(f'{ep:>5d}  {row.train_loss_norm:>11.4f}  '
              f'{row.train_mae_est:>10.3f}  {row.dev_mae:>8.3f}  '
              f'{row.gap:>+6.2f}{marker}')

    # Summary diagnostics.
    print()
    print(f'  best dev MAE        : {best_dev:.3f} @ epoch {best_epoch}')
    print(f'  train MAE at best   : ~{traj.loc[best_epoch, "train_mae_est"]:.2f}')
    print(f'  gap at best         : {traj.loc[best_epoch, "gap"]:+.2f} '
          f'(dev - train_MAE)')

    last_epoch = traj.index.max()
    final_gap = traj.loc[last_epoch, 'gap']
    print(f'  gap at final epoch  : {final_gap:+.2f}')

    # Where did the gap start growing past the best-epoch gap?
    best_gap = traj.loc[best_epoch, 'gap']
    after_best = traj.loc[traj.index > best_epoch]
    if len(after_best):
        widening = after_best[after_best['gap'] > best_gap + 0.5]
        if len(widening):
            ep = widening.index[0]
            print(f'  overfitting onset   : ~epoch {ep} '
                  f'(gap grew by >0.5 MAE past best)')
        else:
            print('  overfitting onset   : not detected after the best epoch '
                  '(dev didn\'t collapse — could likely have trained longer)')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('csvs', nargs='+', type=Path,
                   help='one or more finetune_results_*.csv trajectory files')
    p.add_argument('--y-std', type=float, default=10.44,
                   help='label std used for z-score normalisation during training '
                        '(default 10.44, matches the "train" partition)')
    args = p.parse_args()

    for csv in args.csvs:
        if not csv.exists():
            print(f'!! {csv} not found', file=sys.stderr)
            continue
        traj = load_trajectory(csv, args.y_std)
        report(csv.stem, traj)


if __name__ == '__main__':
    main()
