"""Check for data leakage between evl / dev / train WAV partitions.

For each WAV in lab2_data/{partition}/wav/, computes an MD5 of the raw file
bytes. Then reports any hash that appears in more than one partition.

By default also computes a content-only hash (decoded PCM samples + sample
rate), which catches files that are sonically identical but re-encoded with
different headers. Disable with --no-content-hash to run faster.

Usage:
    python check_partition_overlap.py
    python check_partition_overlap.py --partitions evl dev train train_small
    python check_partition_overlap.py --no-content-hash
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / 'lab2_data'


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def content_md5(path: Path) -> str:
    """Hash of the decoded audio (sample rate + raw PCM samples).

    Catches reencoded duplicates: same audio, different WAV header / metadata
    will share this hash even when their file-byte MD5s differ.
    """
    import soundfile as sf  # local import so --no-content-hash users don't need it
    samples, sr = sf.read(str(path), dtype='float32', always_2d=False)
    h = hashlib.md5()
    h.update(str(sr).encode())
    h.update(samples.tobytes())
    return h.hexdigest()


def hash_partition(part: str, want_content: bool) -> dict[str, tuple[str, str]]:
    """Return {fileid: (byte_hash, content_hash)} for one partition.

    `content_hash` is '' when --no-content-hash is set.
    """
    wav_dir = DATA / part / 'wav'
    if not wav_dir.exists():
        print(f'  !! {wav_dir} not found, skipping')
        return {}

    paths = sorted(wav_dir.glob('*.wav'))
    out = {}
    for i, p in enumerate(paths, 1):
        fid = p.stem
        bh = file_md5(p)
        ch = content_md5(p) if want_content else ''
        out[fid] = (bh, ch)
        if i % 500 == 0:
            print(f'    {part}: {i}/{len(paths)}')
    print(f'  {part:<12s}  {len(out):>5d} files hashed')
    return out


def collect_overlaps(parts: dict[str, dict[str, tuple[str, str]]],
                     idx: int, label: str,
                     expected_subsets: list[frozenset[str]],
                     focus: set[str] | None) -> None:
    """Find any hash (at position `idx`) shared between any two partitions.

    Overlaps whose partition set matches one of `expected_subsets` (e.g.
    {train, train_small}) are reported as expected and not counted as
    leakage. When `focus` is non-empty, only overlaps that involve at
    least one focus partition (e.g. {'evl', 'dev'}) are shown.
    """
    hash_to_locations: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for part, entries in parts.items():
        for fid, hashes in entries.items():
            h = hashes[idx]
            if h:
                hash_to_locations[h].append((part, fid))

    cross = {h: locs for h, locs in hash_to_locations.items()
             if len({p for p, _ in locs}) > 1}

    expected_count = 0
    leak_count = 0
    leak_examples = []
    for h, locs in cross.items():
        parts_set = frozenset(p for p, _ in locs)
        if parts_set in expected_subsets:
            expected_count += 1
            continue
        if focus and not (parts_set & focus):
            continue
        leak_count += 1
        if len(leak_examples) < 30:
            leak_examples.append((h, locs))

    if expected_count:
        print(f'\n  expected subset overlap: {expected_count} hash(es) '
              f'(e.g. train_small ⊂ train) — not leakage')

    if leak_count == 0:
        print(f'\n  -> no {label} LEAKAGE across partitions. Clean.')
        return

    print(f'\n  -> {leak_count} {label} hash(es) appear across UNEXPECTED '
          f'partition pairs (LEAKAGE):')
    for h, locs in leak_examples:
        parts_set = sorted({p for p, _ in locs})
        print(f'     {h[:10]}…  in {parts_set}')
        for p, fid in locs:
            print(f'        {p:<12s}  {fid}')
    if leak_count > len(leak_examples):
        print(f'     ... and {leak_count - len(leak_examples)} more')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--partitions', nargs='+',
                    default=['evl', 'dev', 'train'],
                    help='partitions to compare (default: evl dev train)')
    ap.add_argument('--no-content-hash', action='store_true',
                    help='skip the decoded-audio hash. Faster, but will miss '
                         're-encoded duplicates with different WAV headers.')
    ap.add_argument('--focus', nargs='+', default=['evl', 'dev'],
                    help='only report overlaps involving these partitions. '
                         'Default: evl dev (the splits where leakage matters).')
    ap.add_argument('--no-focus', action='store_true',
                    help='disable --focus filtering, show every overlap')
    ap.add_argument('--expected-subset', nargs='+', action='append',
                    default=None,
                    help='partition pair known to be a subset relation '
                         '(repeat for multiple). Default: train_small train. '
                         'Use as: --expected-subset train_small train')
    args = ap.parse_args()

    # Default expected subset: train_small ⊂ train.
    if args.expected_subset is None:
        expected_subsets = [frozenset({'train_small', 'train'})]
    else:
        expected_subsets = [frozenset(pair) for pair in args.expected_subset]
    focus = None if args.no_focus else set(args.focus)

    print('Hashing WAVs:')
    parts = {p: hash_partition(p, want_content=not args.no_content_hash)
             for p in args.partitions}

    if focus:
        print(f'\nFocus partitions (leakage shown only when involving these): '
              f'{sorted(focus)}')
        print(f'Expected subsets (not counted as leakage): '
              f'{[sorted(s) for s in expected_subsets]}')

    print('\n========= BYTE-IDENTICAL OVERLAPS =========')
    collect_overlaps(parts, idx=0, label='byte-identical',
                     expected_subsets=expected_subsets, focus=focus)

    if not args.no_content_hash:
        print('\n========= AUDIO-CONTENT OVERLAPS =========')
        print('(same decoded samples + sample rate; catches re-encoded duplicates)')
        collect_overlaps(parts, idx=1, label='audio-content',
                         expected_subsets=expected_subsets, focus=focus)

    # Counts summary.
    print('\n========= SUMMARY =========')
    for p, entries in parts.items():
        print(f'  {p:<12s}  {len(entries):>5d} files')


if __name__ == '__main__':
    main()
