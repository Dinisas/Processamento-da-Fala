#!/usr/bin/env python3
"""Average two or more submission CSVs into a single ensemble CSV.

Each input CSV is the standard {fileId, Age} format produced by
finetune_audeering_age.py or eval_checkpoint.py. We join on fileId and
take the mean Age across all inputs. Missing fileids (present in some
inputs but not others) are dropped with a warning unless --keep-partial
is set, in which case rows present in any input are kept and averaged
over whichever inputs contain them.

Use cases
    - K-fold ensemble: average the k per-fold evl prediction CSVs.
    - Multi-seed ensemble: average the 4.91 seed runs (A_lr5e5_s7,
      A_lr5e5_s13, ...).
    - Cross-model ensemble: average the 4.91 model's evl predictions with
      a different recipe's.

Usage
    python finetuning/average_predictions.py \\
        --csv path/to/run1.csv path/to/run2.csv path/to/run3.csv \\
        --out g07_ensemble.csv

    # K-fold sweep:
    python finetuning/average_predictions.py \\
        --csv g07_eval_kfold_*_s42_on_evl_labeled.csv \\
        --out g07_kfold_ensemble.csv
"""

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--csv', nargs='+', required=True,
                    help='input submission CSVs to average (>= 2)')
    ap.add_argument('--out', required=True,
                    help='output submission CSV')
    ap.add_argument('--weights', nargs='+', type=float, default=None,
                    help='optional per-input weights (same length as --csv); '
                         'defaults to equal weighting')
    ap.add_argument('--keep-partial', action='store_true',
                    help='keep fileids present in some-but-not-all inputs '
                         '(default: drop them and warn)')
    args = ap.parse_args()

    import pandas as pd
    import numpy as np

    if len(args.csv) < 2:
        sys.exit('need >= 2 input CSVs to ensemble')
    if args.weights and len(args.weights) != len(args.csv):
        sys.exit(f'--weights has {len(args.weights)} values but '
                 f'{len(args.csv)} CSVs')

    weights = (np.array(args.weights, dtype=float) if args.weights
               else np.ones(len(args.csv)))
    weights = weights / weights.sum()

    dfs = []
    for path, w in zip(args.csv, weights):
        df = pd.read_csv(path)
        if 'fileId' not in df.columns or 'Age' not in df.columns:
            sys.exit(f'{path} is missing fileId or Age column; '
                     f'columns are {list(df.columns)}')
        df = df.set_index('fileId')[['Age']].rename(columns={'Age': path})
        dfs.append(df)
        print(f'  {path}: {len(df):>4d} rows, weight {w:.3f}')

    # Outer join so we can detect missing fileids; we'll filter or warn later.
    merged = dfs[0].join(dfs[1:], how='outer')

    n_full = int(merged.notna().all(axis=1).sum())
    n_partial = len(merged) - n_full
    if n_partial:
        if args.keep_partial:
            print(f'\n  WARNING: {n_partial} fileids present in only some '
                  f'inputs; averaging over whichever have them')
        else:
            print(f'\n  dropping {n_partial} fileids not present in all inputs')
            merged = merged.dropna()

    # Weighted average — for keep-partial rows we renormalise over the
    # weights of inputs that actually have a value (treat NaN as missing).
    arr = merged.to_numpy()
    if args.keep_partial:
        # broadcast weights across columns then mask
        w = np.tile(weights, (len(merged), 1))
        mask = ~np.isnan(arr)
        w = w * mask
        denom = w.sum(axis=1)
        denom = np.where(denom == 0, 1.0, denom)
        merged['Age'] = (np.nan_to_num(arr) * w).sum(axis=1) / denom
    else:
        merged['Age'] = arr @ weights

    out = merged[['Age']].reset_index()
    out['Age'] = out['Age'].round(1)
    out.to_csv(args.out, index=False)
    print(f'\nensemble written: {len(out)} rows -> {args.out}')
    print(f'  Age summary: mean={out["Age"].mean():.2f}  '
          f'std={out["Age"].std():.2f}  '
          f'min={out["Age"].min():.1f}  max={out["Age"].max():.1f}')


if __name__ == '__main__':
    main()
