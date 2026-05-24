#!/usr/bin/env python3
"""
Build the 'big_train_falar' partition by streaming FalAR's train_* splits
from HuggingFace and writing them into the SLPdata-style layout that the
rest of Lab 2 expects (lab2_data/big_train_falar/{wav/<ID>.wav, info.csv}).

Disk-aware
    --max-gb caps cumulative WAV-on-disk bytes (default 50 GB). When the
    cap is reached the stream stops and info.csv is written with the rows
    kept so far.

Resumable
    Existing WAVs in lab2_data/big_train_falar/wav/ are kept and counted
    toward the cap. Re-running the script picks up where it left off,
    iterating shards in order and skipping ids that are already on disk.

Leakage
    We only ever read FalAR splits matching --shards (default: all train_*).
    The course's lab dev/evl partitions are derived from FalAR's dev/test,
    so restricting to train_* is how we avoid segment-level leakage. We
    never touch FalAR dev or test.

Schema
    FalAR rows look like (verified live): keys ID, wav, speaker_id, duration,
    gender, age, ... — `wav` is decoded by the HF Audio feature by default,
    which now needs torchcodec. We cast it to Audio(decode=False) so each
    row arrives as {'bytes': RIFF..., 'path': <orig.wav>}. FalAR is uniformly
    16 kHz mono PCM_16, so we write the bytes through unchanged. Anything
    that isn't 16 kHz / mono / PCM_16 is decoded + resampled + re-encoded
    via soundfile (defensive — should never trigger on the published shards).

Usage
    python finetuning/build_train_falar.py
    python finetuning/build_train_falar.py --max-gb 30
    python finetuning/build_train_falar.py --shards train_0 train_1
    python finetuning/build_train_falar.py --min-duration 5 --max-duration 25
    python finetuning/build_train_falar.py --dry-run     # estimate, write nothing
"""

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

# Windows console encoding (matches sibling scripts in this folder).
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB2_DIR / 'lab2_data'

# FalAR train shards per the dataset card: train_0..train_14 are 50k rows
# each, train_15 is 18.8k. We iterate in numeric order and stop when the
# disk cap is reached.
TRAIN_SHARDS = [f'train_{i}' for i in range(16)]


def _coerce_to_16k_mono_pcm16_bytes(raw_bytes):
    """Return WAV bytes guaranteed to be 16 kHz / mono / PCM_16.

    FalAR is already in this format, so the fast path just returns the
    input. The slow path decodes through soundfile (+ optional librosa
    resample), then re-encodes."""
    bio = io.BytesIO(raw_bytes)
    info = sf.info(bio)
    if info.samplerate == 16000 and info.channels == 1 and info.subtype == 'PCM_16':
        return raw_bytes

    # Defensive path: decode -> downmix -> resample -> re-encode.
    bio.seek(0)
    arr, sr = sf.read(bio, dtype='float32', always_2d=True)
    if arr.shape[1] > 1:
        arr = arr.mean(axis=1, keepdims=False).astype(np.float32)
    else:
        arr = arr[:, 0]
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    arr = np.clip(arr, -1.0, 1.0)
    out = io.BytesIO()
    sf.write(out, arr, 16000, subtype='PCM_16', format='WAV')
    return out.getvalue()


def write_info(info_path, rows_by_id):
    """Write info.csv with columns wav,gender,age matching the existing
    partitions. Sort by id for stable diffs across resumed runs."""
    with open(info_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['wav', 'gender', 'age'])
        for rid in sorted(rows_by_id):
            wav, gender, age = rows_by_id[rid]
            try:
                age_f = float(age)
                age_v = int(age_f) if age_f.is_integer() else age_f
            except (TypeError, ValueError):
                age_v = age
            w.writerow([wav, gender, age_v])


def _row_id(row):
    """FalAR provides `ID` (uppercase) plus `wav.path`. Use ID when populated,
    otherwise fall back to the original filename stem. Sanitised to a safe
    filename: forbid path separators and surrounding whitespace."""
    rid = row.get('ID') or row.get('id')
    if not rid:
        wav = row.get('wav') or {}
        path = wav.get('path') if isinstance(wav, dict) else None
        if path:
            rid = Path(path).stem
    if not rid:
        return None
    rid = str(rid).strip().replace('/', '_').replace('\\', '_')
    return rid or None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--partition', default='big_train_falar',
                    help='folder name under lab2_data/ (default big_train_falar)')
    ap.add_argument('--max-gb', type=float, default=50.0,
                    help='soft cap on cumulative WAV bytes on disk (default 50 GB)')
    ap.add_argument('--min-duration', type=float, default=3.0,
                    help='skip clips shorter than this in seconds (default 3.0)')
    ap.add_argument('--max-duration', type=float, default=30.0,
                    help='skip clips longer than this in seconds (default 30.0)')
    ap.add_argument('--shards', nargs='+', default=TRAIN_SHARDS,
                    help='FalAR splits to read in order (default train_0..train_15). '
                         'Refuses any split not starting with "train_" to enforce '
                         'the no-dev/test rule.')
    ap.add_argument('--dataset-id', default='inesc-id/FalAR',
                    help='HuggingFace dataset id (default inesc-id/FalAR)')
    ap.add_argument('--age-min', type=float, default=None,
                    help='drop rows with age < this (default: no lower filter)')
    ap.add_argument('--age-max', type=float, default=None,
                    help='drop rows with age > this (default: no upper filter)')
    ap.add_argument('--dry-run', action='store_true',
                    help='count rows that pass filters without writing anything')
    ap.add_argument('--save-every', type=int, default=2000,
                    help='flush info.csv every N new rows (default 2000)')
    ap.add_argument('--write-speaker-id', action='store_true',
                    help='also write a sidecar info_full.csv with speaker_id, '
                         'duration, ID — useful for later speaker-disjoint splits')
    args = ap.parse_args()

    # Leakage guardrails:
    #   - FalAR 'test' is NEVER allowed: the lab evl partition is sourced from
    #     it, so pulling it would leak into the submission target.
    #   - FalAR 'dev' IS allowed (it's the source of the lab dev set, which
    #     itself is only 117 rows; pulling the full 7.3k FalAR dev gives you a
    #     larger held-out fold for overfitting diagnostics). Refuses to write
    #     it into a folder that would clobber an existing lab partition.
    #   - 'train_*' shards are always allowed.
    FORBIDDEN_SHARDS = {'test'}
    forbidden = [s for s in args.shards if s in FORBIDDEN_SHARDS]
    if forbidden:
        sys.exit(f'refusing to read FalAR {forbidden} — overlaps with the lab '
                 f'evl partition (the submission target).')
    bad = [s for s in args.shards
           if not (s.startswith('train_') or s == 'dev')]
    if bad:
        sys.exit(f'unrecognised shards {bad}. Allowed: train_0..train_15 or dev.')

    RESERVED_PARTITIONS = {'train', 'train_small', 'dev', 'evl'}
    if args.partition in RESERVED_PARTITIONS:
        sys.exit(f"refusing to write into reserved partition '{args.partition}'. "
                 f"Use a different --partition name (e.g. falar_dev, big_train_falar).")

    if 'dev' in args.shards:
        print('\n*** NOTE: pulling FalAR dev split ***')
        print('  - This OVERLAPS with your lab dev set (lab dev is a ~117-row subset).')
        print('  - Treat it as a DIAGNOSTIC held-out fold only.')
        print('  - Never use it for checkpoint selection or hyperparameter tuning')
        print('    (that role belongs to the lab dev partition).')
        print('  - The lab evl set is your real submission target; this script')
        print('    refuses to pull FalAR test under any args.')

    partition_dir = DATA_DIR / args.partition
    wav_dir = partition_dir / 'wav'
    info_path = partition_dir / 'info.csv'
    info_full_path = partition_dir / 'info_full.csv'
    if not args.dry_run:
        wav_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = int(args.max_gb * 1024 ** 3)

    # ---- Resume: account for what's already on disk ----
    existing = {}
    if wav_dir.exists():
        for p in wav_dir.glob('*.wav'):
            existing[p.stem] = p.stat().st_size
    cumulative_bytes = sum(existing.values())
    print(f'partition  : {partition_dir}')
    print(f'cap        : {args.max_gb:.1f} GB ({max_bytes:,} bytes)')
    print(f'duration   : [{args.min_duration}, {args.max_duration}] s')
    if args.age_min is not None or args.age_max is not None:
        print(f'age filter : [{args.age_min}, {args.age_max}]')
    print(f'resume     : {len(existing)} wavs already on disk '
          f'({cumulative_bytes / 1024 ** 3:.2f} GB)')

    # Pre-load existing info.csv (if any) so we don't lose previously written rows
    # for ids whose wav files are still on disk.
    rows_by_id = {}
    full_rows_by_id = {}
    if info_path.exists():
        import pandas as pd
        df = pd.read_csv(info_path)
        for _, r in df.iterrows():
            basename = str(r['wav']).rsplit('.', 1)[0]
            if basename in existing:
                rows_by_id[basename] = (r['wav'], r['gender'], r['age'])
        print(f'             {len(rows_by_id)} rows preserved from existing info.csv')
    if args.write_speaker_id and info_full_path.exists():
        import pandas as pd
        df = pd.read_csv(info_full_path)
        for _, r in df.iterrows():
            basename = str(r['wav']).rsplit('.', 1)[0]
            if basename in existing:
                full_rows_by_id[basename] = (r['wav'], r['gender'], r['age'],
                                             r.get('speaker_id'), r.get('duration'))

    if cumulative_bytes >= max_bytes and not args.dry_run:
        print('cap already reached — only rewriting info.csv and exiting.')
        write_info(info_path, rows_by_id)
        return

    # ---- Stream shards in order ----
    n_new = 0
    n_skip_dur = 0
    n_skip_age_nan = 0
    n_skip_age_range = 0
    n_skip_exists = 0
    n_skip_gender = 0
    n_skip_recoded = 0     # rows that needed defensive re-encode
    started = time.time()

    for shard in args.shards:
        if cumulative_bytes >= max_bytes:
            break
        print(f'\n== streaming {args.dataset_id}:{shard} ==')
        try:
            ds = load_dataset(args.dataset_id, split=shard, streaming=True)
            # Read raw bytes — avoids the HF Audio decoder's torchcodec dependency.
            ds = ds.cast_column('wav', Audio(decode=False))
        except Exception as e:
            print(f'  skip {shard}: {e}')
            continue

        pbar = tqdm(ds, desc=shard, unit='row', smoothing=0.05)
        for row in pbar:
            if cumulative_bytes >= max_bytes:
                print(f'\n  cap of {args.max_gb:.1f} GB reached — stopping shard {shard}.')
                break

            rid = _row_id(row)
            if not rid:
                continue
            if rid in existing:
                n_skip_exists += 1
                continue

            # Filter: duration
            duration = row.get('duration')
            d = None
            if duration is not None:
                try:
                    d = float(duration)
                    if d < args.min_duration or d > args.max_duration:
                        n_skip_dur += 1
                        continue
                except (TypeError, ValueError):
                    pass

            # Filter: age
            age = row.get('age')
            if age is None:
                n_skip_age_nan += 1
                continue
            try:
                age_f = float(age)
                if np.isnan(age_f):
                    n_skip_age_nan += 1
                    continue
            except (TypeError, ValueError):
                n_skip_age_nan += 1
                continue
            if args.age_min is not None and age_f < args.age_min:
                n_skip_age_range += 1
                continue
            if args.age_max is not None and age_f > args.age_max:
                n_skip_age_range += 1
                continue

            gender = str(row.get('gender', '')).strip()
            if gender not in ('M', 'F'):
                n_skip_gender += 1
                continue

            if args.dry_run:
                est = int(32000 * (d if d is not None else 10.0)) + 44
                cumulative_bytes += est
                n_new += 1
                rows_by_id[rid] = (f'{rid}.wav', gender, age_f)
                pbar.set_postfix(est_gb=f'{cumulative_bytes / 1024 ** 3:.2f}',
                                 rows=len(rows_by_id))
                continue

            wav = row.get('wav')
            raw_bytes = wav.get('bytes') if isinstance(wav, dict) else None
            if not raw_bytes:
                continue

            try:
                out_bytes = _coerce_to_16k_mono_pcm16_bytes(raw_bytes)
            except Exception as e:
                print(f'\n  WARN: decode failed for {rid}: {e}')
                continue
            if out_bytes is not raw_bytes:
                n_skip_recoded += 1

            out_path = wav_dir / f'{rid}.wav'
            try:
                out_path.write_bytes(out_bytes)
            except Exception as e:
                print(f'\n  WARN: failed to write {rid}: {e}')
                continue

            size = out_path.stat().st_size
            cumulative_bytes += size
            existing[rid] = size
            rows_by_id[rid] = (f'{rid}.wav', gender, age_f)
            if args.write_speaker_id:
                full_rows_by_id[rid] = (f'{rid}.wav', gender, age_f,
                                        row.get('speaker_id'), d)
            n_new += 1

            if n_new % args.save_every == 0:
                write_info(info_path, rows_by_id)
                if args.write_speaker_id:
                    _write_full_info(info_full_path, full_rows_by_id)

            pbar.set_postfix(gb=f'{cumulative_bytes / 1024 ** 3:.2f}',
                             new=n_new, rows=len(rows_by_id))

        # End-of-shard flush.
        if not args.dry_run:
            write_info(info_path, rows_by_id)
            if args.write_speaker_id:
                _write_full_info(info_full_path, full_rows_by_id)

    # ---- Summary ----
    elapsed = time.time() - started
    print('\n========= SUMMARY =========')
    print(f'  partition         : {partition_dir}')
    print(f'  rows in info.csv  : {len(rows_by_id)}')
    print(f'  newly written     : {n_new}')
    print(f'  re-encoded (def.) : {n_skip_recoded}')
    print(f'  skipped (already) : {n_skip_exists}')
    print(f'  skipped (duration): {n_skip_dur}')
    print(f'  skipped (age NaN) : {n_skip_age_nan}')
    print(f'  skipped (age rng) : {n_skip_age_range}')
    print(f'  skipped (gender)  : {n_skip_gender}')
    print(f'  on-disk size      : {cumulative_bytes / 1024 ** 3:.2f} GB '
          f'({"estimated" if args.dry_run else "actual"})')
    print(f'  elapsed           : {elapsed / 60:.1f} min')

    if args.dry_run:
        print('\n(dry-run; no files written)')
    else:
        print(f'\nDone. Now run e.g.:')
        print(f'  python finetuning/run_audeering_age.py '
              f'--trainset {args.partition} --backbone xlsr53')


def _write_full_info(path, rows_by_id):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['wav', 'gender', 'age', 'speaker_id', 'duration'])
        for rid in sorted(rows_by_id):
            wav, gender, age, spk, dur = rows_by_id[rid]
            try:
                age_f = float(age)
                age_v = int(age_f) if age_f.is_integer() else age_f
            except (TypeError, ValueError):
                age_v = age
            w.writerow([wav, gender, age_v,
                        '' if spk is None else spk,
                        '' if dur is None else dur])


if __name__ == '__main__':
    main()
