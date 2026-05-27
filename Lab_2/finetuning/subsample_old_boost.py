"""Sub-sample an old-boost partition to N utts per old speaker (hardlinks the wavs)."""

import argparse
import os
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB2_DIR / 'lab2_data'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default='lab_train_old_boost',
                    help='source partition with info_full.csv to sub-sample '
                         '(default lab_train_old_boost)')
    ap.add_argument('--utts-per-speaker', type=int, required=True,
                    help='how many utterances per old speaker to keep in the '
                         'sub-sampled partition')
    ap.add_argument('--out-name', default=None,
                    help='output partition name (default: '
                         '<src>_n<utts_per_speaker>)')
    ap.add_argument('--seed', type=int, default=42,
                    help='random seed for sub-sampling (default 42)')
    ap.add_argument('--copy', action='store_true',
                    help='copy wavs instead of hardlinking (slower, doubles '
                         'disk usage). Hardlinks work on Windows NTFS without '
                         'admin privileges but only within the same volume.')
    args = ap.parse_args()

    import pandas as pd

    src_dir = DATA_DIR / args.src
    src_wav = src_dir / 'wav'
    src_full = src_dir / 'info_full.csv'

    if not src_full.is_file():
        sys.exit(f'missing {src_full} — script needs info_full.csv with the '
                 f'speaker_id column to identify which rows are "new" vs lab train')

    out_name = args.out_name or f'{args.src}_n{args.utts_per_speaker}'
    out_dir = DATA_DIR / out_name
    out_wav = out_dir / 'wav'
    out_wav.mkdir(parents=True, exist_ok=True)

    print(f'src        : {src_dir}')
    print(f'dst        : {out_dir}')
    print(f'N per speaker (new rows only): {args.utts_per_speaker}')
    print(f'link mode  : {"copy" if args.copy else "hardlink"}')

    full = pd.read_csv(src_full)
    # Lab train rows have NaN speaker_id; new old-boost rows have an int.
    lab_mask = full['speaker_id'].isna()
    lab_rows = full[lab_mask].reset_index(drop=True)
    new_rows = full[~lab_mask].reset_index(drop=True)
    print(f'\nsource counts: {len(lab_rows)} lab train + {len(new_rows)} new boost'
          f' = {len(full)} total')

    # Deterministic per-speaker subsample
    def take_first_n(group):
        n = min(args.utts_per_speaker, len(group))
        return group.sample(n=n, random_state=args.seed)

    if args.utts_per_speaker == 0:
        subsampled = new_rows.iloc[:0]
    else:
        subsampled = (
            new_rows.groupby('speaker_id', group_keys=False)
            .apply(take_first_n)
            .reset_index(drop=True)
        )

    combined = pd.concat([lab_rows, subsampled], ignore_index=True)
    print(f'\nsubsampled new rows: {len(subsampled)} (from {len(new_rows)})')
    print(f'total rows in new partition: {len(combined)}')

    # Write info.csv + info_full.csv
    info_path = out_dir / 'info.csv'
    info_full_path = out_dir / 'info_full.csv'
    combined[['wav', 'gender', 'age']].to_csv(info_path, index=False)
    combined.to_csv(info_full_path, index=False)
    print(f'  wrote {info_path.relative_to(LAB2_DIR)}')
    print(f'  wrote {info_full_path.relative_to(LAB2_DIR)}')

    # Place wavs (hardlink or copy)
    print(f'\n-- placing {len(combined)} wavs --')
    placed = 0
    missing_src = []
    for _, r in combined.iterrows():
        wav = r['wav']
        src_p = src_wav / wav
        dst_p = out_wav / wav
        if dst_p.exists():
            placed += 1
            continue
        if not src_p.exists():
            missing_src.append(wav)
            continue
        try:
            if args.copy:
                shutil.copy2(src_p, dst_p)
            else:
                os.link(src_p, dst_p)
            placed += 1
        except OSError as e:
            # Hardlink can fail across filesystems; fall back to copy.
            try:
                shutil.copy2(src_p, dst_p)
                placed += 1
            except Exception as e2:
                print(f'  WARN: could not place {wav}: {e2}')
    print(f'  placed {placed}/{len(combined)}')
    if missing_src:
        print(f'  WARN: {len(missing_src)} src wavs not found '
              f'(first few: {missing_src[:5]})')

    # Age distribution preview
    print(f'\n========= NEW PARTITION AGE DISTRIBUTION =========')
    combined['age_num'] = pd.to_numeric(combined['age'], errors='coerce')
    bins = list(range(20, 81, 5))
    combined['bin'] = pd.cut(combined['age_num'], bins=bins, right=False)
    bucket = combined.groupby('bin', observed=True).agg(n=('wav', 'count')).reset_index()
    print(bucket.to_string(index=False))
    print(f'\n  mean age     : {combined["age_num"].mean():.2f}')
    print(f'  lab train alone: {lab_rows["age"].astype(float).mean():.2f}')
    print(f'  total rows   : {len(combined)}')

    print(f'\nNext: register {out_name!r} in pf_tools.RELEASE_CONFIGS + '
          f'finetune_audeering_age.py --trainset choices, then train:')
    print(f'  python finetuning/finetune_audeering_age.py \\')
    print(f'    --trainset {out_name} --lr 6e-5 --batch-size 16 \\')
    print(f'    --grad-accum 1 --max-epochs 40 --seed 42 \\')
    print(f'    --run-id O_n{args.utts_per_speaker}_s42')


if __name__ == '__main__':
    main()
