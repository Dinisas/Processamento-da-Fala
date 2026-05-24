#!/usr/bin/env python3
"""Validate the ECAPA-based pseudo-speaker clustering against ground-truth
FaLAR `speaker_id` labels.

We do the clustering BLIND (only using ECAPA embeddings, just like
build_speaker_split.py does on lab `train` where no speaker_id is available),
then compare predicted clusters to the real speaker_ids from info_full.csv
to answer:

  1. Does agglomerative clustering on ECAPA recover the real speaker
     structure?
  2. Which cosine threshold is optimal? (we used 0.65 by intuition; here
     we measure it against ground truth)
  3. How do predicted clusters compare to real speakers in terms of size
     distribution and per-speaker / per-cluster purity?

Outputs:
  - Lab_2/imgs/clustering_metrics_vs_threshold.png — ARI/AMI/V/h/c vs threshold
  - Lab_2/imgs/clustering_size_dist.png            — predicted vs real cluster sizes
  - Lab_2/imgs/clustering_per_speaker_purity.png   — for each real speaker, n predicted clusters spanned
  - Lab_2/imgs/clustering_per_cluster_purity.png   — for each predicted cluster, n real speakers contained
  - Lab_2/imgs/clustering_pca_topk.png             — PCA scatter coloured by real speaker (top-K)
  - Lab_2/clustering_validation.json               — raw numbers per threshold

Usage:
    python finetuning/clustering_validation.py
    python finetuning/clustering_validation.py --partition falar_dev
    python finetuning/clustering_validation.py --max-utts 5000
    python finetuning/clustering_validation.py --no-subsample          # use everything
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Windows console encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import soundfile as sf  # used to load WAVs directly without librosa's
# lazy-loader, which conflicts with speechbrain's import machinery on Win.
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB2_DIR / 'lab2_data'
IMGS_DIR = LAB2_DIR / 'imgs'


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------
# ECAPA extraction (caches to disk in the same layout as lab dev/train)
# ---------------------------------------------------------------------

def extract_ecapa(df: pd.DataFrame, partition: str, device: torch.device,
                  batch_size: int = 16) -> np.ndarray:
    """Extract ECAPA embeddings for each row in df. Cached on disk under
    lab2_data/<partition>/spkrec-ecapa-voxceleb/<basename>/<basename>.npy
    (same layout the rest of the project uses)."""
    # librosa already imported at module level (must precede speechbrain).
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    cache_dir = DATA_DIR / partition / 'spkrec-ecapa-voxceleb'
    cache_dir.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray | None] = [None] * len(df)
    needs: list[tuple[int, dict]] = []
    rows = df.to_dict('records')
    for i, r in enumerate(rows):
        base = r['wav'].replace('.wav', '')
        npy = cache_dir / base / f'{base}.npy'
        if npy.is_file():
            v = np.load(npy)
            embeddings[i] = (v[0] if v.ndim == 2 and v.shape[0] == 1 else v).astype(np.float32)
        else:
            needs.append((i, r))

    print(f'  cache hits: {len(df) - len(needs)}/{len(df)}')

    if needs:
        print(f'  extracting ECAPA for {len(needs)} utts (batch={batch_size})')
        # Windows doesn't allow SpeechBrain's default SYMLINK strategy without
        # admin rights; force COPY so we work on a stock user account.
        classifier = EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-ecapa-voxceleb',
            savedir=str(LAB2_DIR / 'models_cache' / 'spkrec-ecapa-voxceleb'),
            run_opts={'device': str(device)},
            local_strategy=LocalStrategy.COPY,
        )
        for start in tqdm(range(0, len(needs), batch_size), desc='ECAPA'):
            batch = needs[start:start + batch_size]
            wavs = []
            for _, r in batch:
                wav_path = DATA_DIR / partition / 'wav' / r['wav']
                # FaLAR is 16 kHz mono PCM_16 by construction; cap at 10 s.
                audio, sr = sf.read(str(wav_path), dtype='float32',
                                    frames=int(16000 * 10.0), always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    # Defensive — should never trigger on FaLAR.
                    raise RuntimeError(f'unexpected sr={sr} in {wav_path}')
                wavs.append(torch.tensor(audio))
            max_len = max(int(w.shape[0]) for w in wavs)
            padded = torch.stack([
                torch.cat([w, torch.zeros(max_len - w.shape[0])]) for w in wavs
            ]).to(device)
            lengths = torch.tensor(
                [w.shape[0] / max_len for w in wavs], dtype=torch.float32,
            ).to(device)
            with torch.no_grad():
                embs = classifier.encode_batch(padded, lengths).squeeze(1).cpu().numpy()
            for (idx, r), emb in zip(batch, embs):
                emb = emb.astype(np.float32)
                embeddings[idx] = emb
                base = r['wav'].replace('.wav', '')
                sub = cache_dir / base
                sub.mkdir(parents=True, exist_ok=True)
                np.save(sub / f'{base}.npy', emb[np.newaxis, :])

    return np.stack(embeddings)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def purity_stats(real_idx: np.ndarray, pred_idx: np.ndarray) -> dict:
    """Per-speaker and per-cluster purity stats.

    - n_clusters_per_speaker[s]: for real speaker s, how many distinct
      predicted clusters their utterances span.
    - main_cluster_fraction[s]: fraction of s's utterances that landed in
      that speaker's most-populated predicted cluster.
    - n_speakers_per_cluster[c]: for predicted cluster c, how many distinct
      real speakers contribute utterances.
    - main_speaker_fraction[c]: fraction of cluster c's utterances from its
      majority real speaker.
    Means/medians are reported on the dict.
    """
    df = pd.DataFrame({'r': real_idx, 'p': pred_idx})
    # Per real speaker
    by_speaker = df.groupby('r')
    n_clusters_per_speaker = by_speaker['p'].nunique().values
    main_cluster_fraction = (
        by_speaker['p'].apply(lambda x: x.value_counts(normalize=True).iloc[0]).values
    )
    # Per predicted cluster
    by_cluster = df.groupby('p')
    n_speakers_per_cluster = by_cluster['r'].nunique().values
    main_speaker_fraction = (
        by_cluster['r'].apply(lambda x: x.value_counts(normalize=True).iloc[0]).values
    )

    return {
        'n_clusters_per_speaker': n_clusters_per_speaker,
        'main_cluster_fraction': main_cluster_fraction,
        'n_speakers_per_cluster': n_speakers_per_cluster,
        'main_speaker_fraction': main_speaker_fraction,
    }


def cluster_at(Z, threshold: float, real_idx: np.ndarray) -> dict:
    """Run fcluster at cosine threshold (linkage distance = 1 - cosine)."""
    labels = fcluster(Z, t=(1.0 - threshold), criterion='distance')
    n_pred = int(labels.max())
    ari = float(adjusted_rand_score(real_idx, labels))
    ami = float(adjusted_mutual_info_score(real_idx, labels))
    h, c, v = homogeneity_completeness_v_measure(real_idx, labels)
    pur = purity_stats(real_idx, labels)
    return {
        'threshold': threshold,
        'n_predicted_clusters': n_pred,
        'ari': ari, 'ami': ami,
        'homogeneity': float(h), 'completeness': float(c), 'v_measure': float(v),
        'mean_clusters_per_speaker': float(pur['n_clusters_per_speaker'].mean()),
        'median_clusters_per_speaker': float(np.median(pur['n_clusters_per_speaker'])),
        'mean_main_cluster_fraction': float(pur['main_cluster_fraction'].mean()),
        'mean_speakers_per_cluster': float(pur['n_speakers_per_cluster'].mean()),
        'mean_main_speaker_fraction': float(pur['main_speaker_fraction'].mean()),
        'labels': labels,
        '_purity': pur,
    }


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_metrics_vs_threshold(results: list[dict], n_real: int, out_path: Path):
    import matplotlib.pyplot as plt
    thresh = [r['threshold'] for r in results]
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for name, key, color in [
        ('ARI', 'ari', 'C0'),
        ('AMI', 'ami', 'C1'),
        ('V-measure', 'v_measure', 'C2'),
        ('homogeneity (cluster purity)', 'homogeneity', 'C3'),
        ('completeness (speaker captured)', 'completeness', 'C4'),
    ]:
        ax.plot(thresh, [r[key] for r in results], 'o-', label=name, color=color)
    ax.set_xlabel('cosine merge threshold (higher = stricter)')
    ax.set_ylabel('score (1 = perfect)')
    ax.set_ylim(0, 1.02)
    ax.set_title(f'ECAPA clustering vs ground-truth FaLAR speaker_id  '
                 f'(n_utts={len(results[0]["labels"])}, n_real_speakers={n_real})')
    ax.axvline(0.65, color='gray', linestyle='--', alpha=0.5,
               label='our default (0.65)')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left')
    # Annotate n_pred_clusters on secondary axis.
    ax2 = ax.twinx()
    ax2.plot(thresh, [r['n_predicted_clusters'] for r in results],
             'k--', alpha=0.5, label='n predicted clusters')
    ax2.axhline(n_real, color='red', linestyle=':', alpha=0.6,
                label=f'n real speakers ({n_real})')
    ax2.set_ylabel('n predicted clusters')
    ax2.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  saved -> {out_path.relative_to(LAB2_DIR)}')


def plot_size_dist(best: dict, real_idx: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt
    pred = pd.Series(best['labels'])
    real = pd.Series(real_idx)
    pred_sizes = pred.value_counts().values
    real_sizes = real.value_counts().values
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    bins = np.arange(1, max(pred_sizes.max(), real_sizes.max()) + 2) - 0.5
    ax.hist(real_sizes, bins=bins, alpha=0.55, label=f'real speakers (n={len(real_sizes)})',
            color='steelblue')
    ax.hist(pred_sizes, bins=bins, alpha=0.55,
            label=f'predicted clusters @ thresh {best["threshold"]:.2f} (n={len(pred_sizes)})',
            color='darkorange')
    ax.set_yscale('log')
    ax.set_xlabel('utterances per group')
    ax.set_ylabel('count (log scale)')
    ax.set_title('Group-size distribution: predicted clusters vs real speakers')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  saved -> {out_path.relative_to(LAB2_DIR)}')


def plot_per_speaker_purity(best: dict, out_path: Path):
    import matplotlib.pyplot as plt
    pur = best['_purity']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram: how many predicted clusters each real speaker spans.
    n = pur['n_clusters_per_speaker']
    axes[0].hist(n, bins=np.arange(1, n.max() + 2) - 0.5,
                 color='steelblue', edgecolor='black')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('# predicted clusters this real speaker is spread across')
    axes[0].set_ylabel('# real speakers (log scale)')
    axes[0].set_title(f'How fragmented are real speakers?  (1 = perfect)\n'
                      f'mean={n.mean():.2f}  median={int(np.median(n))}  '
                      f'thresh={best["threshold"]:.2f}')
    axes[0].axvline(1, color='green', linestyle='--', alpha=0.5,
                    label='ideal (= 1 cluster)')
    axes[0].legend()

    # Histogram: fraction of each real speaker's utts that land in their majority cluster.
    f = pur['main_cluster_fraction']
    axes[1].hist(f, bins=np.linspace(0, 1, 21),
                 color='steelblue', edgecolor='black')
    axes[1].set_xlabel('fraction of speaker utts in their majority predicted cluster')
    axes[1].set_ylabel('# real speakers')
    axes[1].set_title(f'Speaker-recovery fraction  (1 = perfect)\n'
                      f'mean={f.mean():.3f}  median={float(np.median(f)):.3f}')
    axes[1].axvline(1, color='green', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  saved -> {out_path.relative_to(LAB2_DIR)}')


def plot_per_cluster_purity(best: dict, out_path: Path):
    import matplotlib.pyplot as plt
    pur = best['_purity']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    n = pur['n_speakers_per_cluster']
    axes[0].hist(n, bins=np.arange(1, n.max() + 2) - 0.5,
                 color='darkorange', edgecolor='black')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('# real speakers mixed into this predicted cluster')
    axes[0].set_ylabel('# predicted clusters (log scale)')
    axes[0].set_title(f'How impure are predicted clusters?  (1 = pure)\n'
                      f'mean={n.mean():.2f}  median={int(np.median(n))}')
    axes[0].axvline(1, color='green', linestyle='--', alpha=0.5,
                    label='ideal (= 1 speaker)')
    axes[0].legend()

    f = pur['main_speaker_fraction']
    axes[1].hist(f, bins=np.linspace(0, 1, 21),
                 color='darkorange', edgecolor='black')
    axes[1].set_xlabel('fraction of cluster utts from its majority real speaker')
    axes[1].set_ylabel('# predicted clusters')
    axes[1].set_title(f'Cluster homogeneity  (1 = pure)\n'
                      f'mean={f.mean():.3f}  median={float(np.median(f)):.3f}')
    axes[1].axvline(1, color='green', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  saved -> {out_path.relative_to(LAB2_DIR)}')


def plot_pca_topk(X: np.ndarray, real_idx: np.ndarray, best: dict,
                  top_k: int, out_path: Path):
    """PCA scatter showing the top-K most-utterance real speakers, colour-
    coded by real speaker. Each marker is annotated with its predicted
    cluster id — so if same-coloured points share marker text, they were
    clustered together correctly."""
    import matplotlib.pyplot as plt
    counts = pd.Series(real_idx).value_counts()
    top_speakers = counts.head(top_k).index.values
    mask = pd.Series(real_idx).isin(top_speakers).values
    Xs = X[mask]
    rs = real_idx[mask]
    ps = best['labels'][mask]

    pca = PCA(n_components=2, random_state=0).fit(Xs)
    proj = pca.transform(Xs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    cmap = plt.get_cmap('tab20', top_k)
    spk_to_color = {s: cmap(i) for i, s in enumerate(top_speakers)}
    for s in top_speakers:
        m = rs == s
        ax.scatter(proj[m, 0], proj[m, 1], color=spk_to_color[s], s=30,
                   alpha=0.8, edgecolor='black', linewidth=0.4,
                   label=f'spk {s} (n={int(m.sum())})')
    ax.set_xlabel(f'PC1  ({100*pca.explained_variance_ratio_[0]:.1f}% var)')
    ax.set_ylabel(f'PC2  ({100*pca.explained_variance_ratio_[1]:.1f}% var)')
    ax.set_title(f'ECAPA PCA — top-{top_k} real speakers\n'
                 f'colour = real speaker_id   (well-separated colours = clustering should work)')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  saved -> {out_path.relative_to(LAB2_DIR)}')


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--partition', default='big_train_falar',
                   choices=('big_train_falar', 'falar_dev'),
                   help='partition to validate (must have info_full.csv with speaker_id)')
    p.add_argument('--max-utts', type=int, default=10000,
                   help='subsample cap for tractability (default 10000)')
    p.add_argument('--no-subsample', action='store_true',
                   help='use all utterances (slow + memory-heavy on big_train_falar)')
    p.add_argument('--thresholds', nargs='+', type=float,
                   default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
                   help='cosine merge thresholds to sweep')
    p.add_argument('--top-k-pca', type=int, default=15,
                   help='# real speakers to display in the PCA scatter')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()

    IMGS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load info_full.csv (must have speaker_id) ----------
    info_path = DATA_DIR / args.partition / 'info_full.csv'
    if not info_path.is_file():
        raise FileNotFoundError(f'missing {info_path} '
                                f'(need speaker_id column from FaLAR)')
    df = pd.read_csv(info_path)
    if 'speaker_id' not in df.columns:
        raise ValueError(f'{info_path} has no speaker_id column')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['speaker_id'] = df['speaker_id'].astype(str)

    print(f'{"="*72}')
    print(f'CLUSTERING VALIDATION on {args.partition}')
    print(f'{"="*72}')
    print(f'rows in info_full.csv : {len(df)}')
    print(f'unique real speakers  : {df["speaker_id"].nunique()}')

    if not args.no_subsample and len(df) > args.max_utts:
        # Stratified-ish subsample: keep all utterances of a randomly chosen
        # subset of speakers, so we don't fragment speakers across the cut.
        speakers = df['speaker_id'].unique()
        rng = np.random.RandomState(args.seed)
        rng.shuffle(speakers)
        keep = set()
        running = 0
        # Take speakers in random order until we hit the cap.
        size_by_spk = df.groupby('speaker_id').size().to_dict()
        for s in speakers:
            sz = size_by_spk[s]
            if running + sz > args.max_utts * 1.02:
                continue
            keep.add(s)
            running += sz
            if running >= args.max_utts:
                break
        df = df[df['speaker_id'].isin(keep)].reset_index(drop=True)
        print(f'subsampled (whole-speakers): {len(df)} utts, '
              f'{df["speaker_id"].nunique()} real speakers '
              f'(seed={args.seed})')

    # ---------- Extract ECAPA (cache to disk) ----------
    device = select_device()
    print(f'\ndevice: {device}')
    t0 = time.time()
    X = extract_ecapa(df, args.partition, device, batch_size=args.batch_size)
    print(f'ECAPA done in {time.time()-t0:.1f}s   X.shape={X.shape}')

    real_idx = pd.factorize(df['speaker_id'].values)[0]
    n_real = int(real_idx.max() + 1)

    # ---------- Compute pairwise cosine + linkage once ----------
    print(f'\ncomputing pairwise cosine distance matrix '
          f'(condensed size ~ {len(X)*(len(X)-1)//2:,})')
    t0 = time.time()
    dist = pdist(X.astype(np.float32), metric='cosine')
    print(f'  pdist {dist.nbytes/1e6:.1f} MB  in {time.time()-t0:.1f}s')

    print('computing average-linkage hierarchy...')
    t0 = time.time()
    Z = linkage(dist, method='average')
    print(f'  linkage in {time.time()-t0:.1f}s')

    # ---------- Sweep thresholds ----------
    print(f'\n{"thresh":>7s}  {"n_pred":>6s}  '
          f'{"ARI":>5s}  {"AMI":>5s}  {"V":>5s}  '
          f'{"hom":>5s}  {"comp":>5s}  '
          f'{"mean cls/spk":>11s}  {"main-spk frac":>13s}')
    print('-' * 88)
    results = []
    for t in sorted(args.thresholds):
        r = cluster_at(Z, t, real_idx)
        results.append(r)
        print(f'{t:>7.2f}  {r["n_predicted_clusters"]:>6d}  '
              f'{r["ari"]:>5.3f}  {r["ami"]:>5.3f}  {r["v_measure"]:>5.3f}  '
              f'{r["homogeneity"]:>5.3f}  {r["completeness"]:>5.3f}  '
              f'{r["mean_clusters_per_speaker"]:>11.2f}  '
              f'{r["mean_main_speaker_fraction"]:>13.3f}')

    # Pick the "best" threshold by V-measure (balanced homogeneity + completeness).
    best = max(results, key=lambda r: r['v_measure'])
    print(f'\nBest by V-measure: threshold = {best["threshold"]:.2f}  '
          f'V={best["v_measure"]:.3f}  '
          f'(ARI={best["ari"]:.3f}, AMI={best["ami"]:.3f})')

    # Our chosen threshold (0.65) for direct comparison.
    chosen = next((r for r in results if abs(r['threshold'] - 0.65) < 1e-6), None)
    if chosen and chosen is not best:
        print(f'Our chosen 0.65:    V={chosen["v_measure"]:.3f}  '
              f'(ARI={chosen["ari"]:.3f}, AMI={chosen["ami"]:.3f})  '
              f'{"--- pretty close to best" if best["v_measure"] - chosen["v_measure"] < 0.02 else "--- meaningfully worse than best"}')

    # ---------- Plots ----------
    print(f'\nplotting...')
    plot_metrics_vs_threshold(results, n_real, IMGS_DIR / 'clustering_metrics_vs_threshold.png')
    plot_size_dist(best, real_idx, IMGS_DIR / 'clustering_size_dist.png')
    plot_per_speaker_purity(best, IMGS_DIR / 'clustering_per_speaker_purity.png')
    plot_per_cluster_purity(best, IMGS_DIR / 'clustering_per_cluster_purity.png')
    plot_pca_topk(X, real_idx, best, args.top_k_pca,
                  IMGS_DIR / 'clustering_pca_topk.png')

    # ---------- Save JSON summary ----------
    summary = {
        'partition': args.partition,
        'n_utts': len(df),
        'n_real_speakers': n_real,
        'subsample_seed': args.seed,
        'results': [
            {k: v for k, v in r.items() if k not in ('labels', '_purity')}
            for r in results
        ],
        'best_threshold_by_v_measure': best['threshold'],
        'best_v_measure': best['v_measure'],
    }
    out_json = LAB2_DIR / 'clustering_validation.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nsummary written -> {out_json.relative_to(LAB2_DIR)}')


if __name__ == '__main__':
    main()
