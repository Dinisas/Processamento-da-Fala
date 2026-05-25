#!/usr/bin/env python3
"""
Augment lab train with extra FalAR utterances of speakers whose mean age
is >= --age-min, to fix the 65+ under-prediction bias we measured in
diagnose_4_91.py (slope = -0.37, |residual| up to 14 years for true age 75).

Why this approach
    The diagnostic showed bias is statistical regression-toward-mean
    driven by data scarcity at the tails (FalAR train has only 63 unique
    speakers in the 65-75 bucket, contributing ~190 utts of "old voice"
    to lab train). We can't add new 65+ voices — FalAR doesn't have any
    more — but each of those 63 speakers has ~684 utts on average in
    the broader FalAR train pool. Pulling 20 utts/speaker = ~1,260 extra
    rows of "what these voices sound like" gives the model more session
    variation per voice (different recording days, vocal warmth, prosody),
    so it can learn each old voice's age more confidently.

How it works
    1. Load _falar_train_index.csv (built by enumerate_falar_speakers.py).
       Per-speaker mean age comes from averaging the speaker's per-utt
       ages across all train_* shards.
    2. Filter to speakers with mean_age >= --age-min. Default 60.
    3. For each such speaker, sample --utts-per-speaker utterances at
       random from their available pool (excluding bad-duration ones).
    4. Hash lab train wavs (raw bytes MD5) so we can skip any FalAR row
       whose audio matches something already in lab train. This keeps
       the boost partition free of duplicates.
    5. Fetch the chosen rows from their parquet shards via DuckDB
       (column projection: only fetch ID + wav, not the unused columns).
    6. Write each fetched wav to the new partition, plus copy lab train
       wavs in so the partition is self-contained.
    7. Build the combined info.csv (lab train rows + new boost rows) and
       info_full.csv (with FalAR speaker_id for the new rows, NaN for
       lab train rows whose speaker_id we don't have).

Output
    lab2_data/<partition>/
        wav/                     # lab train wavs + N additional per speaker
        info.csv                 # SLPdata-compatible: wav, gender, age
        info_full.csv            # wav, gender, age, speaker_id, duration

Usage
    PS> $env:HF_TOKEN = "hf_..."
    python finetuning/build_old_speaker_boost.py
    python finetuning/build_old_speaker_boost.py --age-min 65 --utts-per-speaker 30
    python finetuning/build_old_speaker_boost.py --copy-mode symlink   # save disk
"""

import argparse
import csv
import hashlib
import io
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB2_DIR / 'lab2_data'

HF_RESOLVE_PREFIX = 'hf://datasets/inesc-id/FalAR/data/'


# ---------------------------------------------------------------------
# Audio helpers (re-used from build_train_falar.py — FalAR is uniformly
# 16 kHz mono PCM_16 so the fast path returns bytes unchanged).
# ---------------------------------------------------------------------

def hash_bytes(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def _fetch_via_hub_download(picks_df, lab_hashes, new_wav_dir, hf_token,
                            cleanup=True):
    """Alternate fetch path: download whole parquet files via
    huggingface_hub.hf_hub_download (battle-tested HF client, built-in
    retries, no Snappy/range-request issues), then extract the wanted
    rows locally with pyarrow.

    With cleanup=True (default): each parquet is downloaded into a fresh
    temp dir and that dir is wiped after extraction. Peak disk ~500 MB.
    Re-running re-downloads (since cache is wiped between files).

    With cleanup=False: parquets live in the persistent HF cache (~/.cache/
    huggingface/hub by default). Re-runs hit cache. But 294 parquets at
    ~500 MB each = ~150 GB — only viable if you have the disk."""
    import shutil
    import tempfile
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    import pyarrow.parquet as pq

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    by_file = list(picks_df.groupby('filename'))
    print(f'\n-- fetching {len(picks_df)} utts via hf_hub_download '
          f'({len(by_file)} parquets) --')
    print(f'  HF cache: {os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")}')

    written = sum(1 for _, sub in by_file for rid in sub['ID']
                  if (new_wav_dir / f'{rid}.wav').is_file())
    skipped_dup = 0
    skipped_fail = 0
    if written:
        print(f'  resume: {written} wavs already on disk')

    pbar = (tqdm(by_file, desc='hub-fetch', unit='pq') if tqdm else by_file)
    for fname, sub in pbar:
        wanted_ids = [str(i) for i in sub['ID'].tolist()
                      if not (new_wav_dir / f'{i}.wav').is_file()]
        if not wanted_ids:
            continue

        # Strip the 'hf://datasets/inesc-id/FalAR/' prefix to get the repo-
        # relative path.
        if not fname.startswith('hf://datasets/inesc-id/FalAR/'):
            note = f'unexpected fname format: {fname}'
            if hasattr(pbar, 'write'):
                pbar.write(note)
            else:
                print(note)
            continue
        repo_path = fname[len('hf://datasets/inesc-id/FalAR/'):]

        # Optionally use a per-file temp dir so cache doesn't accumulate.
        tmp_cache = tempfile.mkdtemp(prefix='falar_pq_') if cleanup else None

        try:
            # Download via hf_hub_download (HF client w/ retries built in).
            try:
                local_path = hf_hub_download(
                    repo_id='inesc-id/FalAR',
                    filename=repo_path,
                    repo_type='dataset',
                    token=hf_token,
                    cache_dir=tmp_cache,  # None = use default HF cache
                )
            except (HfHubHTTPError, OSError, Exception) as e:
                note = f'  download failed for {repo_path}: {str(e)[:120]}'
                if hasattr(pbar, 'write'):
                    pbar.write(note)
                else:
                    print(note)
                skipped_fail += len(wanted_ids)
                continue

            # Read locally with column projection + ID filter (pushed down to
            # parquet row groups by pyarrow).
            try:
                table = pq.read_table(
                    local_path,
                    columns=['ID', 'wav'],
                    filters=[('ID', 'in', wanted_ids)],
                )
            except Exception as e:
                note = f'  pyarrow read failed for {repo_path}: {str(e)[:120]}'
                if hasattr(pbar, 'write'):
                    pbar.write(note)
                else:
                    print(note)
                skipped_fail += len(wanted_ids)
                continue

            df = table.to_pandas()
        finally:
            # Wipe the temp cache for this file BEFORE moving on so peak disk
            # stays at ~500 MB. local_path becomes invalid here; we already
            # extracted the table contents into memory above.
            if tmp_cache:
                shutil.rmtree(tmp_cache, ignore_errors=True)

        for _, r in df.iterrows():
            rid = str(r['ID'])
            w = r['wav']
            raw_bytes = w['bytes'] if isinstance(w, dict) else None
            if raw_bytes is None:
                skipped_fail += 1
                continue
            try:
                out_bytes = coerce_to_16k_mono_pcm16(raw_bytes)
            except Exception:
                skipped_fail += 1
                continue
            if lab_hashes:
                h = hashlib.md5(out_bytes).hexdigest()
                if h in lab_hashes:
                    skipped_dup += 1
                    continue
            out_path = new_wav_dir / f'{rid}.wav'
            try:
                out_path.write_bytes(out_bytes)
                written += 1
            except Exception:
                skipped_fail += 1

        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(written=written, dup=skipped_dup, fail=skipped_fail)

    return written, skipped_dup, skipped_fail


def _finalise_partition(picked, lab_df, new_wav_dir, new_dir, args):
    """Write info.csv + info_full.csv based on which wavs are actually on disk.
    Used by both fetch paths so they share the finalisation logic."""
    new_rows = []
    for p in picked:
        rid = str(p['ID'])
        out_path = new_wav_dir / f'{rid}.wav'
        if not out_path.is_file():
            continue
        new_rows.append({
            'wav': f'{rid}.wav',
            'gender': p['gender'],
            'age': int(round(p['age'])) if float(p['age']).is_integer() else float(p['age']),
            'speaker_id': p['speaker_id'],
            'duration': p['duration'],
        })

    lab_for_info = lab_df.copy()
    if 'speaker_id' in lab_for_info.columns:
        lab_for_info = lab_for_info[['wav', 'gender', 'age']]
    combined_info = pd.concat([
        lab_for_info[['wav', 'gender', 'age']],
        pd.DataFrame(new_rows)[['wav', 'gender', 'age']] if new_rows
        else pd.DataFrame(columns=['wav', 'gender', 'age']),
    ], ignore_index=True)
    combined_info.to_csv(new_dir / 'info.csv', index=False)
    print(f'\nwrote {new_dir / "info.csv"}: {len(combined_info)} rows '
          f'({len(lab_df)} lab train + {len(new_rows)} new)')

    lab_full = lab_for_info.copy()
    lab_full['speaker_id'] = pd.NA
    lab_full['duration'] = pd.NA
    if new_rows:
        new_full = pd.DataFrame(new_rows)[['wav', 'gender', 'age', 'speaker_id', 'duration']]
    else:
        new_full = pd.DataFrame(columns=['wav', 'gender', 'age', 'speaker_id', 'duration'])
    combined_full = pd.concat([lab_full, new_full], ignore_index=True)
    combined_full.to_csv(new_dir / 'info_full.csv', index=False)
    print(f'wrote {new_dir / "info_full.csv"}: {len(combined_full)} rows')

    print(f'\n========= NEW PARTITION AGE DISTRIBUTION =========')
    combined_info['age_bin'] = pd.cut(combined_info['age'],
                                      bins=list(range(20, 81, 5)),
                                      right=False)
    print(combined_info.groupby('age_bin', observed=True)
          .agg(n=('wav', 'count'))
          .reset_index()
          .to_string(index=False))
    print(f'\n  mean age: {combined_info["age"].mean():.2f}  '
          f'(lab train alone: {lab_df["age"].mean():.2f})')
    print(f'  new total rows: {len(combined_info)}')


def _run_with_retry(con, q, batch_i, n_batches, pbar, transient_tokens=None):
    """Run a DuckDB query with retries.

    Two error classes:
      - HTTP transient (429/503/504/Timeout): rate-limit-style. Retry up
        to 7 times with exponential backoff (15s..300s). These almost
        always resolve.
      - Parquet corruption (Snappy/decompression/Failed-to-read/Read past
        end): deterministic. HF is serving a permanently bad CDN chunk
        OR the file has bad bytes. Retrying 7 times wastes ~10 min for
        no gain. Cap at 2 attempts with 10s between (1 fast retry in case
        of a one-off bad transfer, then give up and skip).
      - Everything else: give up immediately.

    Returns the resulting DataFrame, or None if retries exhausted."""
    HTTP_TOKENS = ('429', '503', '504', 'Timeout')
    PARQUET_TOKENS = ('Snappy', 'decompression', 'Failed to read',
                      'Read past end', 'unexpected end')

    http_attempts = 0
    parquet_attempts = 0
    while True:
        try:
            return con.sql(q).df()
        except Exception as e:
            msg = str(e)
            is_http = any(tok in msg for tok in HTTP_TOKENS)
            is_parquet = any(tok in msg for tok in PARQUET_TOKENS)

            if is_http and http_attempts < 7:
                wait = min(300, 15 * (2 ** http_attempts))
                http_attempts += 1
                note = (f'batch {batch_i}/{n_batches}: HTTP transient '
                        f'(http_attempt {http_attempts}/7), sleeping {wait}s. '
                        f'msg: {msg[:120]}')
                if hasattr(pbar, 'write'):
                    pbar.write(note)
                else:
                    print(note)
                time.sleep(wait)
                continue

            if is_parquet and parquet_attempts < 2:
                parquet_attempts += 1
                wait = 10
                note = (f'batch {batch_i}/{n_batches}: parquet error '
                        f'(parquet_attempt {parquet_attempts}/2), '
                        f'sleeping {wait}s. msg: {msg[:120]}')
                if hasattr(pbar, 'write'):
                    pbar.write(note)
                else:
                    print(note)
                time.sleep(wait)
                continue

            # Either retries exhausted, or non-transient error.
            note = (f'batch {batch_i}: giving up on this query. '
                    f'msg: {msg[:200]}')
            if hasattr(pbar, 'write'):
                pbar.write(note)
            else:
                print(note)
            return None


def coerce_to_16k_mono_pcm16(raw_bytes):
    bio = io.BytesIO(raw_bytes)
    info = sf.info(bio)
    if info.samplerate == 16000 and info.channels == 1 and info.subtype == 'PCM_16':
        return raw_bytes
    bio.seek(0)
    arr, sr = sf.read(bio, dtype='float32', always_2d=True)
    if arr.shape[1] > 1:
        arr = arr.mean(axis=1).astype(np.float32)
    else:
        arr = arr[:, 0]
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    arr = np.clip(arr, -1.0, 1.0)
    out = io.BytesIO()
    sf.write(out, arr, 16000, subtype='PCM_16', format='WAV')
    return out.getvalue()


# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--partition', default='lab_train_old_boost',
                    help='new partition folder name under lab2_data/ '
                         '(default lab_train_old_boost)')
    ap.add_argument('--base-trainset', default='train',
                    help='lab partition to use as the base (default train)')
    ap.add_argument('--falar-index',
                    default=str(DATA_DIR / '_falar_train_index.csv'),
                    help='FalAR train index produced by enumerate_falar_speakers.py')
    ap.add_argument('--age-min', type=float, default=60.0,
                    help='speakers with mean_age >= this are "old" (default 60)')
    ap.add_argument('--utts-per-speaker', type=int, default=20,
                    help='how many additional FalAR utts to pull per old speaker '
                         '(default 20)')
    ap.add_argument('--min-duration', type=float, default=3.0)
    ap.add_argument('--max-duration', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--copy-mode', default='copy',
                    choices=('copy', 'symlink'),
                    help='how to populate the base partition wavs in the new '
                         'partition (default copy; symlink saves disk but '
                         'needs admin/dev-mode on Windows)')
    ap.add_argument('--no-hash-check', action='store_true',
                    help='skip the lab-train hash de-dup pass (faster but '
                         'may include duplicates of utts already in lab train)')
    ap.add_argument('--parquets-per-batch', type=int, default=15,
                    help='# parquet files per DuckDB read_parquet call '
                         '(default 15). DuckDB parallelises within a single '
                         'query, so batching is faster than per-file queries.')
    ap.add_argument('--use-hub-download', action='store_true',
                    help='download whole parquet files via huggingface_hub '
                         '(robust, retries built-in) and extract rows locally '
                         'with pyarrow. Slower per-file but immune to the '
                         'Snappy/range-request issues DuckDB hits on HF CDN. '
                         'Each parquet is ~500 MB; without --keep-cache they '
                         'are downloaded to a temp dir and deleted after '
                         'extraction (peak disk ~500 MB).')
    ap.add_argument('--keep-cache', action='store_true',
                    help='only with --use-hub-download: keep parquets in the '
                         'HF cache after extraction (default: delete each '
                         'after use so disk stays bounded).')
    ap.add_argument('--finalise-only', action='store_true',
                    help='skip the fetch loop entirely; just write info.csv + '
                         'info_full.csv based on whichever wavs are already on '
                         'disk under the partition. Useful to start training '
                         'with a partial boost while continuing to fetch the '
                         'rest in another terminal.')
    args = ap.parse_args()

    # Sanity checks
    base_dir = DATA_DIR / args.base_trainset
    if not (base_dir / 'info.csv').is_file():
        sys.exit(f'missing {base_dir / "info.csv"}')
    if not Path(args.falar_index).is_file():
        sys.exit(f'missing {args.falar_index} — run enumerate_falar_speakers.py first')

    new_dir = DATA_DIR / args.partition
    new_wav_dir = new_dir / 'wav'
    new_wav_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    # ---------------- load lab train ----------------
    lab_df = pd.read_csv(base_dir / 'info.csv')
    print(f'lab train  : {len(lab_df)} rows from {base_dir}')

    # ---------------- load FalAR index ----------------
    print(f'\n-- loading FalAR index --')
    fal = pd.read_csv(args.falar_index, low_memory=False)
    print(f'  {len(fal):,} rows, {fal["speaker_id"].nunique()} unique speakers')

    # Per-speaker mean age + utt counts
    spk = fal.groupby('speaker_id').agg(
        mean_age=('age', 'mean'),
        n_utts=('ID', 'count'),
        gender=('gender', 'first'),
    ).reset_index()
    print(f'  per-speaker mean age: min {spk["mean_age"].min():.1f}, '
          f'max {spk["mean_age"].max():.1f}, '
          f'mean {spk["mean_age"].mean():.1f}')

    old_spk = spk[spk['mean_age'] >= args.age_min].copy()
    print(f'\n  speakers with mean_age >= {args.age_min}: '
          f'{len(old_spk)} (out of {len(spk)})')
    if len(old_spk) == 0:
        sys.exit(f'no speakers found with mean_age >= {args.age_min}; '
                 f'consider lowering --age-min')

    # ---------------- pick utts per old speaker ----------------
    print(f'\n-- selecting {args.utts_per_speaker} utts per old speaker --')
    picked = []   # list of dicts: speaker_id, ID, age, gender, duration, filename
    spk_summary = []
    for _, s in old_spk.iterrows():
        sid = int(s['speaker_id'])
        pool = fal[fal['speaker_id'] == sid].copy()
        pool = pool[(pool['duration'] >= args.min_duration)
                    & (pool['duration'] <= args.max_duration)]
        if len(pool) == 0:
            continue
        n_to_pick = min(args.utts_per_speaker, len(pool))
        # random pick (deterministic via rng)
        idx = rng.choice(len(pool), size=n_to_pick, replace=False)
        sub = pool.iloc[idx]
        for _, r in sub.iterrows():
            picked.append({
                'speaker_id': sid,
                'ID': r['ID'],
                'age': float(r['age']),
                'gender': r['gender'],
                'duration': float(r['duration']),
                'filename': r['filename'],
            })
        spk_summary.append({
            'speaker_id': sid,
            'mean_age': float(s['mean_age']),
            'n_picked': n_to_pick,
            'pool_size': len(pool),
        })
    print(f'  picked {len(picked)} rows from {len(spk_summary)} speakers')

    summary_df = pd.DataFrame(spk_summary).sort_values('mean_age')
    print(f'\n  per-speaker pick summary (sorted by mean_age):')
    print(summary_df.to_string(index=False,
                               float_format=lambda x: f'{x:.1f}'))

    # ---------------- copy / link lab train wavs into new partition ----------------
    print(f'\n-- copying lab train wavs to {new_wav_dir} ({args.copy_mode}) --')
    t0 = time.time()
    base_wav_dir = base_dir / 'wav'
    n_copied = 0
    for _, r in lab_df.iterrows():
        src = base_wav_dir / r['wav']
        dst = new_wav_dir / r['wav']
        if dst.exists():
            continue
        try:
            if args.copy_mode == 'symlink':
                os.symlink(str(src.resolve()), str(dst))
            else:
                shutil.copy2(src, dst)
            n_copied += 1
        except Exception as e:
            print(f'  WARN: failed to {args.copy_mode} {r["wav"]}: {e}')
    print(f'  {n_copied} wavs from lab train placed ({time.time()-t0:.1f}s)')

    # ---------------- hash lab train wavs for de-dup ----------------
    lab_hashes = set()
    if not args.no_hash_check:
        print(f'\n-- hashing {len(lab_df)} lab train wavs for de-dup --')
        t0 = time.time()
        for _, r in lab_df.iterrows():
            p = new_wav_dir / r['wav']
            if p.is_file():
                lab_hashes.add(hash_bytes(p))
        print(f'  hashed in {time.time()-t0:.1f}s')

    # ---------------- fetch new utts ----------------
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if not hf_token:
        sys.exit('no HF_TOKEN env var set — needed for HuggingFace auth '
                 '(both DuckDB and hf_hub_download paths). '
                 'PS> $env:HF_TOKEN = "hf_..."')

    picks_df = pd.DataFrame(picked)

    if args.finalise_only:
        print(f'\n-- --finalise-only: skipping fetch, writing info.csv from '
              f'{sum(1 for f in new_wav_dir.glob("*.wav"))} on-disk wavs --')
        _finalise_partition(picked, lab_df, new_wav_dir, new_dir, args)
        return

    if args.use_hub_download:
        written, skipped_dup, skipped_fail = _fetch_via_hub_download(
            picks_df, lab_hashes, new_wav_dir, hf_token,
            cleanup=not args.keep_cache,
        )
        print(f'\n-- fetch done ({written} written incl. resume, '
              f'{skipped_dup} dup, {skipped_fail} fail) --')
        # Skip the DuckDB block below.
        _finalise_partition(picked, lab_df, new_wav_dir, new_dir, args)
        return

    import duckdb
    con = duckdb.connect()
    con.sql('INSTALL httpfs; LOAD httpfs;')
    con.execute("CREATE SECRET hf_secret (TYPE huggingface, TOKEN ?)",
                [hf_token])
    print(f'\n-- fetching {len(picked)} utts via DuckDB --')

    # Group picks by parquet filename so each parquet is opened once.
    by_file = list(picks_df.groupby('filename'))
    print(f'  spread across {len(by_file)} parquet files')

    # ---- Resume: skip wavs already on disk ----
    pre_existing = sum(1 for _, sub in by_file
                       for rid in sub['ID']
                       if (new_wav_dir / f'{rid}.wav').is_file())
    if pre_existing:
        print(f'  resume: {pre_existing} wavs already on disk; will skip')

    written = pre_existing
    skipped_dup = 0
    skipped_fail = 0
    overall_t0 = time.time()

    # Batch multiple parquets into ONE DuckDB query so it parallelises HTTP
    # range requests + row-group skipping internally. Previous per-file
    # loop hit ~10s/file × 294 files = 50 min just for transport overhead.
    pbatch = args.parquets_per_batch
    batches = [by_file[i:i + pbatch] for i in range(0, len(by_file), pbatch)]
    print(f'  fetching in {len(batches)} batches of <= {pbatch} parquets each')

    try:
        from tqdm import tqdm
        pbar = tqdm(batches, desc='fetch', unit='batch')
    except ImportError:
        pbar = batches

    for batch_i, batch in enumerate(pbar):
        urls = [name for name, _ in batch]
        all_ids = [str(rid) for _, sub in batch for rid in sub['ID'].tolist()]
        # Skip IDs whose wav is already on disk (resume).
        all_ids = [rid for rid in all_ids
                   if not (new_wav_dir / f'{rid}.wav').is_file()]
        if not all_ids:
            continue

        url_list = ', '.join(repr(u) for u in urls)
        id_list = ', '.join(repr(i) for i in all_ids)
        q = f"""
            SELECT ID, wav
            FROM read_parquet([{url_list}])
            WHERE ID IN ({id_list})
        """

        # Retry policy: HTTP transients get exponential backoff (up to 7
        # attempts); parquet corruption (Snappy etc.) gets only 2 attempts
        # then we move on, since those errors are deterministic and not
        # worth waiting on.
        df = _run_with_retry(con, q, batch_i + 1, len(batches), pbar)
        if df is None:
            # Whole-batch retries exhausted. Fall back to per-file queries
            # so we can isolate which parquet(s) are persistently bad and
            # still rescue the rest of the batch.
            note = (f'batch {batch_i+1}: batched query failed permanently; '
                    f'falling back to per-file queries')
            if hasattr(pbar, 'write'):
                pbar.write(note)
            else:
                print(note)
            df_parts = []
            for url, sub in batch:
                ids_in_file = [str(i) for i in sub['ID'].tolist()
                               if not (new_wav_dir / f'{i}.wav').is_file()]
                if not ids_in_file:
                    continue
                id_list_one = ', '.join(repr(i) for i in ids_in_file)
                q_one = (f"SELECT ID, wav FROM read_parquet('{url}') "
                         f"WHERE ID IN ({id_list_one})")
                df_one = _run_with_retry(con, q_one, batch_i + 1,
                                          len(batches), pbar)
                if df_one is not None:
                    df_parts.append(df_one)
                else:
                    note = (f'  skipping {url.rsplit("/",1)[-1]} — '
                            f'{len(ids_in_file)} IDs lost to permanent error')
                    if hasattr(pbar, 'write'):
                        pbar.write(note)
                    else:
                        print(note)
            if not df_parts:
                continue
            df = pd.concat(df_parts, ignore_index=True)

        for _, r in df.iterrows():
            rid = str(r['ID'])
            w = r['wav']
            raw_bytes = w['bytes'] if isinstance(w, dict) else None
            if raw_bytes is None:
                skipped_fail += 1; continue
            try:
                out_bytes = coerce_to_16k_mono_pcm16(raw_bytes)
            except Exception as e:
                skipped_fail += 1; continue
            if lab_hashes:
                h = hashlib.md5(out_bytes).hexdigest()
                if h in lab_hashes:
                    skipped_dup += 1; continue
            out_path = new_wav_dir / f'{rid}.wav'
            try:
                out_path.write_bytes(out_bytes)
                written += 1
            except Exception:
                skipped_fail += 1

        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(written=written, dup=skipped_dup, fail=skipped_fail)

    print(f'\n-- fetch done in {(time.time()-overall_t0)/60:.1f} min '
          f'({written} written incl. resume, {skipped_dup} dup, '
          f'{skipped_fail} fail) --')

    _finalise_partition(picked, lab_df, new_wav_dir, new_dir, args)

    print(f'\nNow:')
    print(f'  1. Add {args.partition!r} to pf_tools.SLPdata.RELEASE_CONFIGS (stub url).')
    print(f'  2. Add {args.partition!r} to the --trainset choices in')
    print(f'     finetune_audeering_age.py.')
    print(f'  3. Run:')
    print(f'     python finetuning/finetune_audeering_age.py \\')
    print(f'       --trainset {args.partition} --lr 6e-5 --batch-size 16 \\')
    print(f'       --grad-accum 1 --max-epochs 40 --seed 42 \\')
    print(f'       --run-id O_oldboost_s42')


if __name__ == '__main__':
    main()
