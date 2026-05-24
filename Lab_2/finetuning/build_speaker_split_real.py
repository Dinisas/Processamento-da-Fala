#!/usr/bin/env python3
"""Build a speaker-disjoint train_inner / dev_inner manifest using FaLAR's
REAL `speaker_id` column from `info_full.csv` — zero clustering, zero
leakage by construction.

This is the ground-truth version of build_speaker_split.py:
  - build_speaker_split.py       uses ECAPA clustering on lab `train` (no IDs)
  - build_speaker_split_real.py  uses real speaker_id on big_train_falar (has IDs)

The output manifest has the same shape as the clustering version, so the
existing `finetune_audeering_age.py --split-manifest` flag and head-sweep
scripts can consume either kind of split without changes.

Usage:
    python finetuning/build_speaker_split_real.py
    python finetuning/build_speaker_split_real.py --val-frac 0.15
    python finetuning/build_speaker_split_real.py --partition falar_dev   # sanity build
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB2_DIR / 'lab2_data'
IMGS_DIR = LAB2_DIR / 'imgs'


def split_speakers(
    speaker_sizes: pd.Series, speaker_ages: pd.Series, speaker_genders: pd.Series,
    val_frac: float, seed: int,
) -> tuple[set[str], set[str]]:
    """Stratified speaker-disjoint split.

    Buckets speakers by (gender, age_decade), randomly picks val_frac of
    speakers from each bucket. This keeps the dev_inner's age + gender
    distribution close to the training fold's distribution — otherwise the
    naive "smaller-first" greedy strategy on big_train_falar packs all the
    small/peripheral speakers into dev, which skews the eval.
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'speaker_id': speaker_sizes.index,
        'size': speaker_sizes.values,
        'age': speaker_ages.reindex(speaker_sizes.index).values,
        'gender': speaker_genders.reindex(speaker_sizes.index).astype(str).values,
    })
    df['age_bucket'] = (df['age'] // 10).astype('Int64')  # 20s, 30s, ...

    dev = set()
    for (gender, age_bucket), group in df.groupby(['gender', 'age_bucket']):
        speakers = group['speaker_id'].tolist()
        rng.shuffle(speakers)
        # Take roughly val_frac of speakers in each bucket.
        n_dev = max(1, int(round(val_frac * len(speakers)))) if len(speakers) >= 2 else 0
        dev.update(speakers[:n_dev])

    train = set(speaker_sizes.index) - dev
    return train, dev


def plot_split(df: pd.DataFrame, train_mask: np.ndarray, dev_mask: np.ndarray,
               speaker_sizes: pd.Series, out_prefix: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available, skipping plots')
        return

    IMGS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Real speaker size distribution + which side of the split they landed.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    is_dev = pd.Series(False, index=speaker_sizes.index)
    is_dev.loc[df.loc[dev_mask, 'speaker_id'].astype(str).unique()] = True
    train_sizes = speaker_sizes[~is_dev].values
    dev_sizes = speaker_sizes[is_dev].values
    bins = np.linspace(0, speaker_sizes.max(), 40)
    ax.hist(train_sizes, bins=bins, alpha=0.55, color='steelblue',
            label=f'train_inner speakers (n={len(train_sizes)})')
    ax.hist(dev_sizes, bins=bins, alpha=0.55, color='darkorange',
            label=f'dev_inner speakers (n={len(dev_sizes)})')
    ax.set_yscale('log')
    ax.set_xlabel('utterances per speaker')
    ax.set_ylabel('# speakers (log scale)')
    ax.set_title(f'Speaker-size distribution in the speaker-disjoint split  '
                 f'(real speaker_id from FaLAR)')
    ax.legend()
    fig.tight_layout()
    out = IMGS_DIR / f'{out_prefix.name}_sizes.png'
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f'  -> {out.relative_to(LAB2_DIR)}')

    # 2) Age distribution overlay.
    has_age = df['age'].notna()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.arange(20, 81, 2)
    ax.hist(df.loc[has_age, 'age'], bins=bins, alpha=0.4, color='gray',
            density=True, label=f'all (n={int(has_age.sum())})')
    ax.hist(df.loc[train_mask & has_age, 'age'], bins=bins, alpha=0.55,
            color='steelblue', density=True,
            label=f'train_inner (n={int((train_mask & has_age).sum())})')
    ax.hist(df.loc[dev_mask & has_age, 'age'], bins=bins, alpha=0.55,
            color='darkorange', density=True,
            label=f'dev_inner (n={int((dev_mask & has_age).sum())})')
    ax.set_xlabel('age (years)')
    ax.set_ylabel('density')
    ax.set_title('Age distribution: speaker-disjoint split on real speaker_id')
    ax.legend()
    fig.tight_layout()
    out = IMGS_DIR / f'{out_prefix.name}_ages.png'
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f'  -> {out.relative_to(LAB2_DIR)}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--partition', default='big_train_falar',
                   help='partition with info_full.csv (default big_train_falar)')
    p.add_argument('--val-frac', type=float, default=0.20,
                   help='fraction of utts on dev_inner (default 0.20)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=None,
                   help='where to write JSON (default: '
                        'lab2_data/<partition>/speaker_disjoint_real.json)')
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()

    info_path = DATA_DIR / args.partition / 'info_full.csv'
    if not info_path.is_file():
        raise FileNotFoundError(f'{info_path} missing — need info_full.csv '
                                'with a speaker_id column from FaLAR')

    print(f'{"="*72}')
    print(f'BUILD SPEAKER-DISJOINT SPLIT (real speaker_id)')
    print(f'{"="*72}')
    print(f'partition: {args.partition}')
    print(f'val_frac : {args.val_frac}')
    print(f'seed     : {args.seed}')

    df = pd.read_csv(info_path)
    if 'speaker_id' not in df.columns:
        raise ValueError(f'{info_path} has no speaker_id column')
    df['fileid'] = df['wav'].str.replace('.wav', '', regex=False)
    df['speaker_id'] = df['speaker_id'].astype(str)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    print(f'\nrows in info_full.csv : {len(df)}')
    print(f'unique speaker_id     : {df["speaker_id"].nunique()}')
    n_age = int(df['age'].notna().sum())
    print(f'with age labels       : {n_age}  '
          f'(mean={df["age"].mean():.2f}  std={df["age"].std():.2f}  '
          f'range=[{int(df["age"].min())}, {int(df["age"].max())}])')

    speaker_sizes = df.groupby('speaker_id').size()
    # Each speaker_id maps to a single (age, gender) in FaLAR — take the first.
    speaker_ages = df.groupby('speaker_id')['age'].first()
    speaker_genders = df.groupby('speaker_id')['gender'].first() \
        if 'gender' in df.columns else pd.Series('U', index=speaker_sizes.index)
    print(f'\nspeaker size stats: '
          f'min={speaker_sizes.min()}  '
          f'median={int(speaker_sizes.median())}  '
          f'mean={speaker_sizes.mean():.1f}  '
          f'max={speaker_sizes.max()}')

    # ---------- Split ----------
    train_sp, dev_sp = split_speakers(
        speaker_sizes, speaker_ages, speaker_genders, args.val_frac, args.seed,
    )
    train_mask = df['speaker_id'].isin(train_sp).to_numpy()
    dev_mask = df['speaker_id'].isin(dev_sp).to_numpy()
    n_train = int(train_mask.sum())
    n_dev = int(dev_mask.sum())

    print(f'\nspeakers: train_inner={len(train_sp):>5d}   dev_inner={len(dev_sp):>5d}')
    print(f'utts   : train_inner={n_train:>5d}   dev_inner={n_dev:>5d}   '
          f'(dev fraction = {n_dev/(n_train+n_dev):.1%})')
    assert not (train_sp & dev_sp), 'speaker overlap!'

    # Age stats per side.
    print(f'\nage distribution per side:')
    for name, mask in [('train_inner', train_mask), ('dev_inner', dev_mask)]:
        m = mask & df['age'].notna().to_numpy()
        ages = df.loc[m, 'age'].to_numpy()
        if len(ages):
            print(f'  {name:>12s}: n={int(m.sum()):>5d}  '
                  f'mean={ages.mean():5.2f}  std={ages.std():4.2f}  '
                  f'min={ages.min():4.1f}  max={ages.max():4.1f}')

    # Gender balance per side.
    if 'gender' in df.columns:
        print(f'\ngender balance per side:')
        for name, mask in [('train_inner', train_mask), ('dev_inner', dev_mask)]:
            gc = df.loc[mask, 'gender'].value_counts(normalize=True).to_dict()
            gc_str = '  '.join(f'{k}={v:.2%}' for k, v in sorted(gc.items()))
            print(f'  {name:>12s}: {gc_str}')

    # ---------- Plots ----------
    out_path = (Path(args.out) if args.out else
                DATA_DIR / args.partition / 'speaker_disjoint_real.json')
    if not args.no_plot:
        print(f'\nplots:')
        plot_split(df, train_mask, dev_mask, speaker_sizes,
                   IMGS_DIR / f'speaker_split_real_{args.partition}')

    # ---------- Persist manifest ----------
    manifest = {
        'partition': args.partition,
        'method': 'real_speaker_id',
        'val_frac': args.val_frac,
        'seed': args.seed,
        'n_speakers_total': int(df['speaker_id'].nunique()),
        'n_speakers_train': len(train_sp),
        'n_speakers_dev': len(dev_sp),
        'n_utts_train': n_train,
        'n_utts_dev': n_dev,
        'train_inner_uuids': df.loc[train_mask, 'fileid'].tolist(),
        'dev_inner_uuids':   df.loc[dev_mask, 'fileid'].tolist(),
        'uuid_to_speaker': {f: s for f, s
                            in zip(df['fileid'], df['speaker_id'])},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nManifest -> {out_path.relative_to(LAB2_DIR)}')


if __name__ == '__main__':
    main()
