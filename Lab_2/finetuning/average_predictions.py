"""Average two or more submission CSVs into one ensemble CSV (join on fileId, mean Age)."""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
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

    arr = merged.to_numpy()
    if args.keep_partial:
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
