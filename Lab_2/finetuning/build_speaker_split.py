#!/usr/bin/env python3
"""Build a speaker-disjoint train_inner / dev_inner manifest from the train
partition, used by Phase 3 of the leakage-mitigation plan.

The Lab 2 dataset has no explicit speaker IDs, but the speaker_leakage_audit.py
diagnostic showed that ~26% of dev utterances have a same-speaker neighbour
in train (cosine ECAPA > 0.70, often at very different ages). To measure how
much of our 4.91 dev MAE is genuine age learning vs. speaker memorisation, we
need an internal validation fold the model could NOT have leaked from.

Approach:
  1. Load cached ECAPA embeddings for the train partition.
  2. Agglomerative clustering with cosine distance, average linkage, and a
     cosine merge threshold tuned via --threshold (default 0.65 — between the
     leakage-detection threshold 0.70 and the chance-similarity floor ~0.55,
     errs toward over-merging so within-cluster utterances are very likely
     the same speaker).
  3. Each cluster = pseudo-speaker. Split pseudo-speakers (NOT utterances)
     80/20 by count → train_inner / dev_inner. Random with fixed seed.
  4. Write the UUID manifest to a JSON file the finetune script can consume.

The output JSON has the shape:
  {
    "trainset": "train",
    "threshold": 0.65,
    "seed": 42,
    "n_clusters": <int>,
    "train_inner_uuids": [<UUID>, ...],
    "dev_inner_uuids":  [<UUID>, ...],
    "train_inner_clusters": [<cluster_id>, ...],
    "dev_inner_clusters":  [<cluster_id>, ...],
    "uuid_to_cluster": {<UUID>: <cluster_id>}
  }

Usage:
    python finetuning/build_speaker_split.py
    python finetuning/build_speaker_split.py --threshold 0.60 --val-frac 0.2
    python finetuning/build_speaker_split.py --out custom_split.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Windows console encoding.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB2_DIR / 'lab2_data'


def load_train_ecapa(partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ECAPA embeddings for `partition`, joined with info.csv ages.

    Returns (uuids, X, ages). Drops UUIDs missing from the cache.
    """
    info_path = DATA_DIR / partition / 'info.csv'
    cache = DATA_DIR / partition / 'spkrec-ecapa-voxceleb'
    df = pd.read_csv(info_path)
    df['uuid'] = df['wav'].str.replace('.wav', '', regex=False)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    rows = []
    for _, row in df.iterrows():
        uuid = row['uuid']
        npy = cache / uuid / f'{uuid}.npy'
        if not npy.is_file():
            sub = cache / uuid
            if not sub.is_dir():
                continue
            cand = sorted(p for p in sub.iterdir() if p.suffix == '.npy')
            if not cand:
                continue
            npy = cand[0]
        v = np.load(npy, allow_pickle=True)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        elif v.ndim == 2:
            v = v.mean(axis=0)
        rows.append((uuid, v.astype(np.float32), row['age']))

    uuids = np.array([r[0] for r in rows])
    X = np.stack([r[1] for r in rows], axis=0)
    ages = np.array([r[2] for r in rows], dtype=float)
    return uuids, X, ages


def l2_normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def cluster_speakers(X: np.ndarray, cosine_threshold: float) -> np.ndarray:
    """Agglomerative clustering with average linkage on cosine distance.

    distance_threshold = 1 - cosine_threshold. Two embeddings get merged into
    the same cluster if their average-linkage distance falls below this.
    """
    distance_threshold = 1.0 - cosine_threshold
    # sklearn's AgglomerativeClustering with `metric='cosine'` works directly on
    # raw vectors; no need to L2-normalise (it computes cosine distance).
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average',
    )
    labels = agg.fit_predict(X)
    return labels.astype(np.int64)


def split_clusters(labels: np.ndarray, val_frac: float, seed: int
                   ) -> tuple[set[int], set[int]]:
    """Split unique cluster IDs into train_inner / dev_inner sets.

    Aims for val_frac fraction of UTTERANCES (not clusters) on the dev side,
    by greedily picking smaller clusters first until the count is hit. This
    avoids the degenerate "single huge cluster lands in dev_inner" case.
    """
    rng = np.random.RandomState(seed)
    unique, counts = np.unique(labels, return_counts=True)
    n_total = int(counts.sum())
    target_val = int(round(val_frac * n_total))

    order = np.arange(len(unique))
    rng.shuffle(order)
    # Sort clusters by size (smaller first); ties broken by the shuffled order.
    by_size = sorted(zip(counts[order], unique[order]), key=lambda kv: kv[0])

    dev_clusters = set()
    dev_count = 0
    for sz, cid in by_size:
        if dev_count + sz > target_val * 1.15:
            # Don't blow past 15% over the target; skip and try smaller ones.
            continue
        dev_clusters.add(int(cid))
        dev_count += int(sz)
        if dev_count >= target_val:
            break

    train_clusters = set(int(c) for c in unique) - dev_clusters
    return train_clusters, dev_clusters


def plot_split(labels: np.ndarray, ages: np.ndarray, has_age: np.ndarray,
               train_mask: np.ndarray, dev_mask: np.ndarray, imgs_dir: Path):
    """Two plots: cluster-size distribution and per-side age distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available, skipping plots')
        return

    imgs_dir.mkdir(parents=True, exist_ok=True)
    cluster_sizes = np.bincount(labels)

    # 1. Cluster size histogram (log-scale y to see the long tail).
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.arange(1, cluster_sizes.max() + 2) - 0.5
    ax.hist(cluster_sizes, bins=bins, edgecolor='black', alpha=0.85)
    ax.set_yscale('log')
    ax.set_xlabel('utterances per cluster (= pseudo-speaker)')
    ax.set_ylabel('# clusters (log scale)')
    ax.set_title(f'Pseudo-speaker cluster size distribution  '
                 f'(n_clusters={len(cluster_sizes)})')
    ax.axvline(np.median(cluster_sizes), color='red', linestyle='--', alpha=0.6,
               label=f'median = {int(np.median(cluster_sizes))}')
    ax.axvline(cluster_sizes.mean(), color='orange', linestyle='--', alpha=0.6,
               label=f'mean = {cluster_sizes.mean():.1f}')
    ax.legend()
    fig.tight_layout()
    out = imgs_dir / 'speaker_split_cluster_sizes.png'
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f'  cluster-size plot saved -> {out.relative_to(LAB2_DIR)}')

    # 2. Age distribution overlay: full train vs train_inner vs dev_inner.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.arange(20, 81, 2)
    ax.hist(ages[has_age], bins=bins, alpha=0.4, label=f'train (all, n={int(has_age.sum())})',
            color='gray', density=True)
    ax.hist(ages[train_mask & has_age], bins=bins, alpha=0.55,
            label=f'train_inner (n={int((train_mask & has_age).sum())})',
            color='steelblue', density=True)
    ax.hist(ages[dev_mask & has_age], bins=bins, alpha=0.55,
            label=f'dev_inner (n={int((dev_mask & has_age).sum())})',
            color='darkorange', density=True)
    ax.set_xlabel('age (years)')
    ax.set_ylabel('density')
    ax.set_title('Age distribution: full train vs speaker-disjoint inner split')
    ax.legend()
    fig.tight_layout()
    out = imgs_dir / 'speaker_split_age_dist.png'
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f'  age-dist plot saved -> {out.relative_to(LAB2_DIR)}')


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--trainset', default='train',
                   choices=('train', 'train_small'),
                   help='partition to split (default: train)')
    p.add_argument('--threshold', type=float, default=0.65,
                   help='cosine similarity merge threshold; pairs with cosine '
                        '>= this end up in the same cluster (default 0.65 — '
                        'slightly looser than the 0.70 leakage threshold so '
                        'that ambiguous pairs are still kept together)')
    p.add_argument('--val-frac', type=float, default=0.20,
                   help='fraction of UTTERANCES (not clusters) on the '
                        'dev_inner side (default 0.20)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=None,
                   help='where to write the JSON manifest (default: '
                        'Lab_2/lab2_data/<trainset>/speaker_disjoint_split.json)')
    p.add_argument('--no-plot', action='store_true',
                   help='skip the cluster-size + age-distribution matplotlib plots')
    args = p.parse_args()

    out_path = Path(args.out) if args.out else (
        DATA_DIR / args.trainset / 'speaker_disjoint_split.json'
    )

    print(f'{"="*72}')
    print(f'BUILD SPEAKER-DISJOINT SPLIT FROM `{args.trainset}`')
    print(f'{"="*72}')
    print(f'cosine merge threshold : {args.threshold}')
    print(f'val_frac               : {args.val_frac}')
    print(f'seed                   : {args.seed}')

    # ---------- Load ----------
    print(f'\n-- loading {args.trainset} ECAPA embeddings --')
    uuids, X, ages = load_train_ecapa(args.trainset)
    print(f'  {len(uuids)} utterances loaded  (X.shape={X.shape})')
    has_age = ~np.isnan(ages)
    print(f'  {int(has_age.sum())} have age labels  '
          f'(age mean={ages[has_age].mean():.1f}  std={ages[has_age].std():.2f})')

    # ---------- Cluster ----------
    print(f'\n-- agglomerative clustering (cosine, avg linkage, '
          f'threshold={args.threshold}) --')
    labels = cluster_speakers(X, args.threshold)
    n_clusters = len(np.unique(labels))
    cluster_sizes = np.bincount(labels)
    print(f'  n_clusters: {n_clusters}')
    print(f'  cluster size distribution: '
          f'min={cluster_sizes.min()}  '
          f'median={int(np.median(cluster_sizes))}  '
          f'mean={cluster_sizes.mean():.1f}  '
          f'max={cluster_sizes.max()}')
    # Histogram of cluster sizes (compact text version).
    bins = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 9999)]
    print(f'  cluster size buckets:')
    for lo, hi in bins:
        n = int(((cluster_sizes >= lo) & (cluster_sizes <= hi)).sum())
        share = n / n_clusters
        bar = '#' * int(round(share * 40))
        print(f'    [{lo:>3d}..{hi:>4d}]: {n:>5d}  ({100*share:>5.1f}%)  {bar}')

    # ---------- Split ----------
    print(f'\n-- splitting clusters (target val_frac={args.val_frac:.0%}) --')
    train_clusters, dev_clusters = split_clusters(labels, args.val_frac, args.seed)

    train_mask = np.array([c in train_clusters for c in labels])
    dev_mask = ~train_mask

    n_train = int(train_mask.sum())
    n_dev = int(dev_mask.sum())
    print(f'  clusters: {len(train_clusters):>5d} train_inner   '
          f'{len(dev_clusters):>5d} dev_inner')
    print(f'  utts:     {n_train:>5d} train_inner   '
          f'{n_dev:>5d} dev_inner   '
          f'(dev fraction = {n_dev/(n_train+n_dev):.1%})')

    # Verify cluster disjoint-ness.
    assert train_clusters & dev_clusters == set(), 'cluster overlap!'

    # Age distribution sanity.
    def summarise(mask, name):
        m = mask & has_age
        if m.sum() == 0:
            print(f'  {name}: no age labels')
            return
        a = ages[m]
        print(f'  {name:>12s}: n={int(m.sum()):>5d}  '
              f'age mean={a.mean():5.2f}  std={a.std():4.2f}  '
              f'min={a.min():4.1f}  max={a.max():4.1f}')

    print(f'\n-- age distribution check --')
    summarise(train_mask, 'train_inner')
    summarise(dev_mask, 'dev_inner')

    # ---------- Plots ----------
    if not args.no_plot:
        plot_split(labels, ages, has_age, train_mask, dev_mask,
                   LAB2_DIR / 'imgs')

    # ---------- Persist ----------
    manifest = {
        'trainset': args.trainset,
        'threshold': args.threshold,
        'val_frac': args.val_frac,
        'seed': args.seed,
        'n_clusters_total': int(n_clusters),
        'n_clusters_train': len(train_clusters),
        'n_clusters_dev': len(dev_clusters),
        'n_utts_train': n_train,
        'n_utts_dev': n_dev,
        'train_inner_uuids': uuids[train_mask].tolist(),
        'dev_inner_uuids':   uuids[dev_mask].tolist(),
        'uuid_to_cluster': {str(u): int(c) for u, c in zip(uuids, labels)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nManifest written -> {out_path.relative_to(LAB2_DIR)}')


if __name__ == '__main__':
    main()
