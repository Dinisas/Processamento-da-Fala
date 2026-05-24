#!/usr/bin/env python3
"""Build a speaker-disjoint train_inner / dev_inner manifest using REAL FalAR
speaker_ids (the ones we wrote to info_full.csv at download time).

This replaces the ECAPA-clustering approximation in build_speaker_split.py:
we no longer infer pseudo-speakers from embedding similarity — we use the
ground-truth FalAR speaker_id column directly. The output schema is identical
so finetune_audeering_age.py's `--split-manifest` flag consumes it unchanged.

What "speaker-disjoint" guarantees here
    Every speaker_id is assigned to exactly one side. No utterance from the
    same FalAR speaker can appear in both train_inner and dev_inner, so a
    model that learns "this voice = this age" can't reuse memorised speaker
    -> age associations to score well on dev_inner. The dev_inner MAE then
    reflects generalisation to UNSEEN speakers, which is the only fair
    measurement of age-from-voice learning vs. identity memorisation.

Stratification
    A naive random split by speaker leaves age distributions mismatched — if
    the random draw happens to send all 70-year-olds to dev, MAE looks worse
    for reasons unrelated to the model. We stratify by per-speaker age bucket
    (default 5-year bins) so each bucket contributes ~val_frac of its rows to
    dev_inner.

Output manifest (compatible with finetune_audeering_age.py --split-manifest):
    {
      "trainset":          "big_train_falar",
      "method":            "falar_speaker_id",
      "val_frac":          0.20,
      "seed":              42,
      "n_speakers_total":  <int>,
      "n_speakers_train":  <int>,
      "n_speakers_dev":    <int>,
      "n_utts_train":      <int>,
      "n_utts_dev":        <int>,
      "train_inner_uuids": [<FalAR ID>, ...],
      "dev_inner_uuids":   [<FalAR ID>, ...],
      "speakers_train":    [<speaker_id>, ...],
      "speakers_dev":      [<speaker_id>, ...],
      "uuid_to_speaker":   {<FalAR ID>: <speaker_id>, ...}
    }

Usage
    python finetuning/build_falar_speaker_split.py
    python finetuning/build_falar_speaker_split.py --val-frac 0.15 --bin-width 10
    python finetuning/build_falar_speaker_split.py --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB2_DIR / 'lab2_data'


def load_info_full(partition: str) -> pd.DataFrame:
    """Load info_full.csv (must include speaker_id). Falls back to info.csv +
    a clear error message if the sidecar is missing."""
    pdir = DATA_DIR / partition
    info_full = pdir / 'info_full.csv'
    if not info_full.is_file():
        info = pdir / 'info.csv'
        msg = (f"info_full.csv missing for partition '{partition}'.\n"
               f"  Looked at: {info_full}\n"
               f"  This script needs the speaker_id column. Re-run the FalAR "
               f"build with --write-speaker-id:\n"
               f"    python finetuning/build_train_falar.py "
               f"--partition {partition} --write-speaker-id ...")
        if info.is_file():
            msg += "\n  (info.csv exists but lacks speaker_id.)"
        sys.exit(msg)

    df = pd.read_csv(info_full)
    required = {'wav', 'speaker_id', 'age'}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"info_full.csv missing required columns: {sorted(missing)}")

    df['fileid'] = df['wav'].astype(str).str.replace(r'\.wav$', '', regex=True)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['speaker_id'] = pd.to_numeric(df['speaker_id'], errors='coerce')

    # Drop rows that can't be assigned: missing speaker_id OR missing age.
    n0 = len(df)
    df = df.dropna(subset=['speaker_id', 'age']).reset_index(drop=True)
    dropped = n0 - len(df)
    if dropped:
        print(f'  dropped {dropped} rows with missing speaker_id or age '
              f'({100*dropped/n0:.1f}%)')
    return df


def stratified_speaker_split(df: pd.DataFrame, val_frac: float,
                             bin_width: float, seed: int):
    """Per-bucket (by per-speaker mean age) random speaker split, targeting
    val_frac of UTTERANCES on the dev side. Returns (train_speakers, dev_speakers)
    as sets of speaker_id ints.

    Algorithm
      1. For each speaker, compute mean age across their rows.
      2. Bucket speakers by floor(mean_age / bin_width).
      3. Within each bucket, shuffle speakers; greedily assign to dev_inner
         until that bucket's utt count crosses val_frac of the bucket total.
         Single-speaker buckets default to train_inner (preserves age
         coverage on the training side).
      4. Every other speaker goes to train_inner. Speakers are never split.
    """
    rng = np.random.RandomState(seed)

    spk_stats = (df.groupby('speaker_id')
                 .agg(mean_age=('age', 'mean'),
                      n_utts=('fileid', 'count'))
                 .reset_index())
    spk_stats['bucket'] = (spk_stats['mean_age'] // bin_width).astype(int)

    train_speakers = set()
    dev_speakers = set()

    by_bucket = defaultdict(list)
    for _, row in spk_stats.iterrows():
        by_bucket[int(row['bucket'])].append(
            (int(row['speaker_id']), int(row['n_utts']), float(row['mean_age']))
        )

    print(f'  buckets ({bin_width:.0f}-year wide):')
    for b in sorted(by_bucket):
        entries = by_bucket[b]
        n_spk = len(entries)
        n_utts = sum(n for _, n, _ in entries)
        target = int(round(val_frac * n_utts))
        # Shuffle speakers within the bucket so the random seed actually
        # randomises which speakers end up in dev for this bucket.
        bucket_dev_spk = 0
        bucket_dev_utts = 0
        # Single-speaker bucket: send to train (we'd rather over-cover train
        # than create a dev_inner with all utterances from one speaker).
        if n_spk == 1:
            sid = entries[0][0]
            train_speakers.add(sid)
            print(f'    age {b*bin_width:>4.0f}+: {n_spk:>4d} spk  '
                  f'{n_utts:>5d} utts  -> 0 dev_spk, 0 dev_utts '
                  f'(single-speaker bucket sent to train)')
            continue
        # Best-fit greedy: at each step pick whichever remaining speaker
        # brings the bucket's dev_utts closest to target. Stop as soon as no
        # remaining speaker can improve the fit. Shuffle once for stable
        # tie-breaking under the seed.
        order = list(rng.permutation(n_spk))
        remaining = [entries[i] for i in order]
        while remaining:
            cur_diff = abs(bucket_dev_utts - target)
            best_idx = None
            best_diff = cur_diff
            for i, (_, n, _) in enumerate(remaining):
                new_diff = abs(bucket_dev_utts + n - target)
                if new_diff < best_diff:
                    best_diff = new_diff
                    best_idx = i
            if best_idx is None:
                break  # adding any remaining speaker only makes the fit worse
            sid, n, _ = remaining.pop(best_idx)
            dev_speakers.add(sid)
            bucket_dev_spk += 1
            bucket_dev_utts += n
        for sid, _, _ in remaining:
            train_speakers.add(sid)
        print(f'    age {b*bin_width:>4.0f}+: {n_spk:>4d} spk  '
              f'{n_utts:>5d} utts  -> {bucket_dev_spk:>3d} dev_spk, '
              f'{bucket_dev_utts:>4d} dev_utts '
              f'({100*bucket_dev_utts/n_utts:>4.1f}%)')

    assert train_speakers.isdisjoint(dev_speakers), 'speaker overlap'
    return train_speakers, dev_speakers


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--trainset', default='big_train_falar',
                   help='partition to split (default: big_train_falar). The '
                        'partition must contain info_full.csv with speaker_id.')
    p.add_argument('--val-frac', type=float, default=0.20,
                   help='target fraction of UTTERANCES on the dev_inner side '
                        '(default 0.20)')
    p.add_argument('--bin-width', type=float, default=5.0,
                   help='age bucket width in years for stratification '
                        '(default 5)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=None,
                   help='where to write the JSON manifest (default: '
                        'lab2_data/<trainset>/falar_speaker_split.json)')
    args = p.parse_args()

    out_path = Path(args.out) if args.out else (
        DATA_DIR / args.trainset / 'falar_speaker_split.json'
    )

    print(f'{"="*72}')
    print(f'BUILD FALAR SPEAKER-DISJOINT SPLIT FROM `{args.trainset}`')
    print(f'{"="*72}')
    print(f'val_frac   : {args.val_frac}')
    print(f'bin_width  : {args.bin_width} years')
    print(f'seed       : {args.seed}')

    print(f'\n-- loading info_full.csv --')
    df = load_info_full(args.trainset)
    n_utts = len(df)
    n_spk = df['speaker_id'].nunique()
    print(f'  utterances: {n_utts}')
    print(f'  speakers  : {n_spk}')
    print(f'  age range : {df["age"].min():.1f} - {df["age"].max():.1f}  '
          f'(mean {df["age"].mean():.1f})')

    print(f'\n-- splitting speakers (stratified by per-speaker age bucket) --')
    train_speakers, dev_speakers = stratified_speaker_split(
        df, args.val_frac, args.bin_width, args.seed,
    )

    df_train = df[df['speaker_id'].isin(train_speakers)]
    df_dev = df[df['speaker_id'].isin(dev_speakers)]

    print(f'\n-- result --')
    print(f'  speakers: {len(train_speakers):>5d} train_inner   '
          f'{len(dev_speakers):>5d} dev_inner')
    print(f'  utts:     {len(df_train):>5d} train_inner   '
          f'{len(df_dev):>5d} dev_inner   '
          f'(dev fraction = {len(df_dev)/(len(df_train)+len(df_dev)):.1%})')

    # Age distribution sanity check.
    print(f'\n-- age distribution check --')
    for name, sub in [('train_inner', df_train), ('dev_inner', df_dev)]:
        a = sub['age'].to_numpy()
        print(f'  {name:>12s}: n={len(a):>5d}  '
              f'mean={a.mean():5.2f}  std={a.std():4.2f}  '
              f'min={a.min():4.1f}  max={a.max():4.1f}')

    # KL divergence sanity (5-year bins, smoothed).
    bins = np.arange(20, 81, 5)
    p_train, _ = np.histogram(df_train['age'].to_numpy(), bins=bins)
    p_dev, _ = np.histogram(df_dev['age'].to_numpy(), bins=bins)
    p_train = (p_train + 1) / (p_train.sum() + len(p_train))  # add-one smooth
    p_dev = (p_dev + 1) / (p_dev.sum() + len(p_dev))
    kl = float(np.sum(p_dev * np.log(p_dev / p_train)))
    print(f'  KL(dev_inner || train_inner) = {kl:.4f}  '
          f'(<0.05 ideal, <0.15 acceptable, >0.30 suspicious)')

    # Speaker-disjoint INVARIANT — defensive double-check.
    overlap = train_speakers & dev_speakers
    if overlap:
        sys.exit(f'INTERNAL ERROR: {len(overlap)} speakers in both sides')
    df_overlap = set(df_train['fileid']) & set(df_dev['fileid'])
    if df_overlap:
        sys.exit(f'INTERNAL ERROR: {len(df_overlap)} fileids in both sides')

    # ---- write manifest ----
    manifest = {
        'trainset': args.trainset,
        'method': 'falar_speaker_id',
        'val_frac': args.val_frac,
        'bin_width': args.bin_width,
        'seed': args.seed,
        'n_speakers_total': int(n_spk),
        'n_speakers_train': len(train_speakers),
        'n_speakers_dev': len(dev_speakers),
        'n_utts_train': int(len(df_train)),
        'n_utts_dev': int(len(df_dev)),
        'train_inner_uuids': df_train['fileid'].tolist(),
        'dev_inner_uuids':   df_dev['fileid'].tolist(),
        'speakers_train':    sorted(int(s) for s in train_speakers),
        'speakers_dev':      sorted(int(s) for s in dev_speakers),
        'uuid_to_speaker':   {fid: int(s) for fid, s in
                              zip(df['fileid'].tolist(),
                                  df['speaker_id'].astype(int).tolist())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    rel = out_path.relative_to(LAB2_DIR) if out_path.is_relative_to(LAB2_DIR) else out_path
    print(f'\nManifest written -> {rel}')

    print('\nNow run:')
    print(f'  python finetuning/finetune_audeering_age.py \\')
    print(f'    --trainset {args.trainset} \\')
    print(f'    --split-manifest {rel} \\')
    print(f'    --epochs 4 --batch-size 4 --grad-accum 4')


if __name__ == '__main__':
    main()
