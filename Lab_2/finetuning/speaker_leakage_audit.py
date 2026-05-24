#!/usr/bin/env python3
"""Speaker-leakage audit for the Lab 2 age dataset.

The parliament-derived dataset can plausibly contain the same person at multiple
ages (a politician recorded across decades). If that's true and the train/dev
partitions weren't built speaker-disjoint, a deep model can memorise speaker
identity instead of learning paralinguistic age cues — dev MAE then reflects
"how often did the test speaker also appear in train" more than ageing ability.

This script answers two questions, both cheaply (no GPU, uses cached embeddings):

  Phase 1 — Overlap audit:
    For every dev utterance, find its nearest training utterance by ECAPA
    cosine similarity. Report:
      - distribution of max-similarities,
      - count of dev utterances with sim > threshold (probably-same-speaker),
      - for each leaked pair: dev_age, train_age, |dAge| (a same-person-at-
        different-ages signal needs |dAge| > 5 to be informative).
    Same audit against evl → train, so we know whether the lab-side eval is
    speaker-disjoint from train by design.

  Phase 2 — Speaker-only age baseline:
    Train a plain Ridge on ECAPA(192) → age over train, evaluate on dev.
    If the Ridge dev MAE is close to the wav2vec2 finetune number (4.91),
    the finetune is largely riding on speaker identity. If much worse
    (>~7), the finetune's gain is paralinguistic and the 4.91 is honest.
    Repeats with the cached x-vectors as a sanity check.

Outputs are printed and also saved to:
    Lab_2/speaker_leakage_audit.json
    Lab_2/imgs/speaker_leakage_hist.png  (histogram of max similarities)

Usage:
    python finetuning/speaker_leakage_audit.py
    python finetuning/speaker_leakage_audit.py --threshold 0.65
    python finetuning/speaker_leakage_audit.py --no-plot       # skip png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows default cp1252 console can't encode em-dashes / arrows we print.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
LAB2_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB2_DIR / 'lab2_data'
AGE_MIN, AGE_MAX = 20.0, 90.0
FINETUNE_BASELINE_DEV_MAE = 4.91   # B_lr6e5_s42, the headline we're auditing


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_partition(partition: str, emb_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load info.csv + cached single-vector embeddings.

    Returns (uuids, X, ages). UUIDs missing from the embedding cache are
    dropped silently with a one-line warning printed. ages is NaN for evl.
    """
    info_path = DATA_DIR / partition / 'info.csv'
    cache_path = DATA_DIR / partition / emb_dir
    if not info_path.is_file():
        raise FileNotFoundError(f'missing {info_path}')
    if not cache_path.is_dir():
        raise FileNotFoundError(f'missing {cache_path}')

    df = pd.read_csv(info_path)
    df['uuid'] = df['wav'].str.replace('.wav', '', regex=False)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    rows = []
    missing = 0
    for _, row in df.iterrows():
        uuid = row['uuid']
        npy_path = cache_path / uuid / f'{uuid}.npy'
        if not npy_path.is_file():
            # Some caches store the single chunk under a different name; pick
            # the first .npy in the UUID dir if the canonical one is absent.
            sub = cache_path / uuid
            if not sub.is_dir():
                missing += 1
                continue
            cand = sorted(p for p in sub.iterdir() if p.suffix == '.npy')
            if not cand:
                missing += 1
                continue
            npy_path = cand[0]
        v = np.load(npy_path, allow_pickle=True)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        elif v.ndim == 2:
            v = v.mean(axis=0)        # average over chunks if any
        rows.append((uuid, v.astype(np.float32), row['age']))

    if missing:
        print(f'  [{partition}/{emb_dir}] {missing} UUIDs from info.csv missing from cache (skipped)')

    uuids = np.array([r[0] for r in rows])
    X = np.stack([r[1] for r in rows], axis=0)
    ages = np.array([r[2] for r in rows], dtype=float)
    return uuids, X, ages


def l2_normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


# ---------------------------------------------------------------------
# Phase 1 — overlap audit
# ---------------------------------------------------------------------

def overlap_report(name: str, src_X: np.ndarray, src_ages: np.ndarray, src_uuids: np.ndarray,
                   ref_X: np.ndarray, ref_ages: np.ndarray, ref_uuids: np.ndarray,
                   threshold: float) -> dict:
    """For each src utterance, find max cosine sim to any ref utterance.

    Returns a small dict with the headline numbers + the per-utterance table
    (as a list of dicts) for serialisation.
    """
    src_n = l2_normalise(src_X)
    ref_n = l2_normalise(ref_X)
    sims = src_n @ ref_n.T                            # (n_src, n_ref) cosine

    max_sim = sims.max(axis=1)
    argmax_ref = sims.argmax(axis=1)

    leaked_mask = max_sim > threshold
    n_leaked = int(leaked_mask.sum())

    # Distribution percentiles — useful to spot bimodality.
    pct = np.percentile(max_sim, [10, 25, 50, 75, 90, 95, 99])

    print(f'\n--- {name}  (n_src={len(src_X)}, n_ref={len(ref_X)}) ---')
    print(f'  max-sim percentiles  [10/25/50/75/90/95/99]: '
          f'{"  ".join(f"{p:.3f}" for p in pct)}')
    print(f'  n_src with max-sim > {threshold:.2f}: {n_leaked} '
          f'({100*n_leaked/len(src_X):.1f}%)')

    # Distribution of |dAge| for the leaked pairs — same person at different
    # ages is the actual leakage signal (vs trivially same recording).
    abs_dage = []
    leak_rows = []
    src_has_age = ~np.isnan(src_ages)
    ref_has_age = ~np.isnan(ref_ages)
    for i in np.where(leaked_mask)[0]:
        j = argmax_ref[i]
        s = float(max_sim[i])
        s_age = float(src_ages[i]) if src_has_age[i] else float('nan')
        r_age = float(ref_ages[j]) if ref_has_age[j] else float('nan')
        d = abs(s_age - r_age) if not (np.isnan(s_age) or np.isnan(r_age)) else float('nan')
        leak_rows.append({
            'src_uuid': str(src_uuids[i]),
            'src_age': s_age,
            'ref_uuid': str(ref_uuids[j]),
            'ref_age': r_age,
            'cosine': s,
            'abs_dage': d,
        })
        if not np.isnan(d):
            abs_dage.append(d)

    if abs_dage:
        a = np.asarray(abs_dage)
        n_same_age = int((a < 2).sum())
        n_diff_age = int((a >= 5).sum())
        print(f'  of those, |dAge|<2yr: {n_same_age}   |dAge|>=5yr: {n_diff_age}   '
              f'mean |dAge|: {a.mean():.2f}')

    # Show the 10 most extreme leaks (highest similarity).
    if leak_rows:
        top = sorted(leak_rows, key=lambda r: -r['cosine'])[:10]
        print(f'  top-10 most similar src→ref pairs:')
        print(f'    {"cosine":>7s}  {"src_age":>7s}  {"ref_age":>7s}  {"|dAge|":>6s}')
        for r in top:
            d_s = f'{r["abs_dage"]:>6.1f}' if not np.isnan(r['abs_dage']) else '     -'
            s_age_s = f'{r["src_age"]:>7.1f}' if not np.isnan(r['src_age']) else '      -'
            r_age_s = f'{r["ref_age"]:>7.1f}' if not np.isnan(r['ref_age']) else '      -'
            print(f'    {r["cosine"]:>7.3f}  {s_age_s}  {r_age_s}  {d_s}')

    return {
        'name': name,
        'n_src': int(len(src_X)),
        'n_ref': int(len(ref_X)),
        'threshold': threshold,
        'n_leaked': n_leaked,
        'percentiles': {str(p): float(v) for p, v in zip([10, 25, 50, 75, 90, 95, 99], pct)},
        'mean_abs_dage_when_leaked': float(np.mean(abs_dage)) if abs_dage else None,
        'leaks': leak_rows,
        'max_sim': max_sim.tolist(),
    }


def plot_histogram(reports: list[dict], out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available, skipping plot')
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for r in reports:
        ax.hist(r['max_sim'], bins=40, alpha=0.55,
                label=f'{r["name"]}  (n_leaked@{r["threshold"]:.2f}={r["n_leaked"]})',
                range=(0, 1))
    ax.axvline(reports[0]['threshold'], color='red', linestyle='--', alpha=0.6,
               label=f'threshold={reports[0]["threshold"]:.2f}')
    ax.set_xlabel('max cosine similarity to any train utterance (ECAPA)')
    ax.set_ylabel('# utterances')
    ax.set_title('Speaker-overlap audit — dev/evl nearest train neighbour')
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  histogram saved -> {out_path.relative_to(LAB2_DIR)}')


def plot_leak_age_scatter(report: dict, out_path: Path):
    """For the leaked dev↔train pairs (cos > threshold), scatter the dev_age
    vs train_age. Points on the diagonal y=x mean "same person, ~same age in
    both" (trivial leakage). Off-diagonal points mean "same person, very
    different age" — the parliament-recorded-across-decades scenario."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pairs = [(r['src_age'], r['ref_age'], r['cosine'])
             for r in report['leaks']
             if not (np.isnan(r['src_age']) or np.isnan(r['ref_age']))]
    if not pairs:
        return
    src_ages = np.array([p[0] for p in pairs])
    ref_ages = np.array([p[1] for p in pairs])
    cos = np.array([p[2] for p in pairs])

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sc = ax.scatter(ref_ages, src_ages, c=cos, cmap='viridis',
                    s=50, edgecolor='k', alpha=0.85)
    lo = min(ref_ages.min(), src_ages.min()) - 2
    hi = max(ref_ages.max(), src_ages.max()) + 2
    ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5, label='same age')
    # Shaded band of |dAge| < 5
    ax.fill_between([lo, hi], [lo - 5, hi - 5], [lo + 5, hi + 5],
                    color='red', alpha=0.08, label='|dAge| < 5y')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('train neighbour age (years)')
    ax.set_ylabel('dev utterance age (years)')
    ax.set_title(f'Leaked dev↔train pairs (cosine > {report["threshold"]:.2f})\n'
                 f'n={len(pairs)}  off-diagonal = same person, different age')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('ECAPA cosine similarity')
    ax.legend(loc='lower right')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  age scatter saved -> {out_path.relative_to(LAB2_DIR)}')


def plot_cos_vs_dage(report: dict, out_path: Path):
    """For ALL dev utterances with a labelled neighbour, scatter cosine
    similarity vs |dAge|. High cosine + high |dAge| = parliament-leakage."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = [(r['cosine'], r['abs_dage']) for r in report['leaks']
            if not np.isnan(r['abs_dage'])]
    if not rows:
        return
    cos = np.array([r[0] for r in rows])
    dage = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(cos, dage, s=40, alpha=0.7, edgecolor='k')
    ax.axhline(5, color='red', linestyle='--', alpha=0.5, label='|dAge|=5y')
    ax.axvline(report['threshold'], color='blue', linestyle='--', alpha=0.5,
               label=f'cos={report["threshold"]:.2f}')
    ax.set_xlabel('ECAPA cosine similarity (dev ↔ nearest train)')
    ax.set_ylabel('|age difference| (years)')
    ax.set_title('Leaked dev points: high cosine + high |dAge| = same person, recorded years apart')
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  cos-vs-dAge saved -> {out_path.relative_to(LAB2_DIR)}')


# ---------------------------------------------------------------------
# Phase 2 — speaker-only age baseline
# ---------------------------------------------------------------------

def ridge_baseline(name: str, Xt: np.ndarray, yt: np.ndarray,
                   Xd: np.ndarray, yd: np.ndarray, alpha: float = 10.0) -> dict:
    """Plain ridge regression speaker-embedding → age."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(Xt, yt)
    train_hyp = np.clip(model.predict(Xt), AGE_MIN, AGE_MAX)
    dev_hyp = np.clip(model.predict(Xd), AGE_MIN, AGE_MAX)
    train_mae = float(mean_absolute_error(yt, train_hyp))
    dev_mae = float(mean_absolute_error(yd, dev_hyp))
    print(f'  {name:24s}  train MAE {train_mae:5.2f}   dev MAE {dev_mae:5.2f}   '
          f'gap {dev_mae-train_mae:+5.2f}')
    return {'name': name, 'train_mae': train_mae, 'dev_mae': dev_mae}


# ---------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------

def print_verdict(audit: dict, ridge_results: dict, ft_baseline: float, threshold: float):
    n_dev = audit['n_src']
    n_dev_leaked = audit['n_leaked']
    pct_leaked = 100 * n_dev_leaked / n_dev if n_dev else 0.0
    ecapa_dev = ridge_results['ecapa']['dev_mae']
    xvec_dev = ridge_results['xvect']['dev_mae']
    best_ridge = min(ecapa_dev, xvec_dev)
    gap_to_finetune = best_ridge - ft_baseline

    print(f'\n{"="*72}')
    print('VERDICT')
    print(f'{"="*72}')
    print(f'  dev utterances with a >{threshold:.2f}-similar train neighbour: '
          f'{n_dev_leaked}/{n_dev} ({pct_leaked:.1f}%)')
    print(f'  best speaker-only Ridge dev MAE: {best_ridge:.2f}  '
          f'(ECAPA={ecapa_dev:.2f}, x-vec={xvec_dev:.2f})')
    print(f'  wav2vec2 finetune dev MAE      : {ft_baseline:.2f}')
    print(f'  ridge − finetune               : {gap_to_finetune:+.2f}')
    print()

    if pct_leaked < 5 and best_ridge > ft_baseline + 2:
        print('  -> LIKELY OK. Few near-duplicate speakers; ECAPA alone is much')
        print('     worse than the finetune, so the finetune is genuinely learning')
        print('     paralinguistic age cues. 4.91 is largely honest. You can')
        print('     skip Phases 3-4; quote with the seed-spread caveat.')
    elif best_ridge < ft_baseline + 0.5:
        print('  -> LIKELY LEAKED. Speaker embedding alone gets close to the')
        print('     finetune dev MAE. Strong evidence the wav2vec2 model is')
        print('     riding on speaker identity. Proceed with Phase 3 (speaker-')
        print('     disjoint CV) and Phase 4b (adversarial finetune).')
    else:
        print('  -> MIXED. ECAPA contributes something but isn\'t the whole')
        print('     story. Proceed with Phase 3 to measure the leakage budget')
        print('     directly; decide on 4b after seeing dev_inner MAE.')


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--threshold', type=float, default=0.70,
                        help='cosine threshold for "probably same speaker" '
                             '(default 0.70 for ECAPA-VoxCeleb)')
    parser.add_argument('--trainset', default='train',
                        choices=('train', 'train_small'),
                        help='partition to treat as reference (default: train, '
                             'the actual finetune training partition)')
    parser.add_argument('--no-plot', action='store_true',
                        help='skip the matplotlib histogram')
    parser.add_argument('--out-json', default=None,
                        help='where to write the audit JSON (default: '
                             'Lab_2/speaker_leakage_audit.json)')
    args = parser.parse_args()

    out_json = Path(args.out_json) if args.out_json else (LAB2_DIR / 'speaker_leakage_audit.json')

    print(f'{"="*72}')
    print(f'SPEAKER-LEAKAGE AUDIT')
    print(f'{"="*72}')
    print(f'trainset reference partition : {args.trainset}')
    print(f'cosine threshold             : {args.threshold}')
    print(f'wav2vec2 finetune dev MAE    : {FINETUNE_BASELINE_DEV_MAE}  (B_lr6e5_s42)')

    # ---------- Load ECAPA + x-vector embeddings for all partitions ----------
    print(f'\n-- loading cached embeddings --')
    ecapa = {}
    xvect = {}
    for part in (args.trainset, 'dev', 'evl'):
        u_e, X_e, y_e = load_partition(part, 'spkrec-ecapa-voxceleb')
        u_x, X_x, y_x = load_partition(part, 'spkrec-xvect-voxceleb')
        ecapa[part] = (u_e, X_e, y_e)
        xvect[part] = (u_x, X_x, y_x)
        print(f'  {part:>11s}: ECAPA {X_e.shape}  x-vector {X_x.shape}')

    u_train, X_train, y_train = ecapa[args.trainset]
    u_dev,   X_dev,   y_dev   = ecapa['dev']
    u_evl,   X_evl,   y_evl   = ecapa['evl']

    # ---------- Phase 1: overlap audit ----------
    print(f'\n{"="*72}')
    print('PHASE 1 — speaker-overlap audit (ECAPA cosine)')
    print(f'{"="*72}')

    dev_report = overlap_report(
        f'dev   -> {args.trainset}', X_dev, y_dev, u_dev,
        X_train, y_train, u_train, args.threshold,
    )
    evl_report = overlap_report(
        f'evl   -> {args.trainset}', X_evl, y_evl, u_evl,
        X_train, y_train, u_train, args.threshold,
    )
    # also dev → dev to check intra-partition near-dupes (sanity)
    # (excluding self by masking the diagonal)
    src_n = l2_normalise(X_dev)
    sims_self = src_n @ src_n.T
    np.fill_diagonal(sims_self, -1.0)
    max_self = sims_self.max(axis=1)
    n_intra = int((max_self > args.threshold).sum())
    print(f'\n--- dev intra-partition (excluding self) ---')
    print(f'  n_dev with >{args.threshold:.2f}-similar other-dev neighbour: '
          f'{n_intra}/{len(X_dev)}  ({100*n_intra/len(X_dev):.1f}%)')

    if not args.no_plot:
        imgs = LAB2_DIR / 'imgs'
        plot_histogram([dev_report, evl_report], imgs / 'speaker_leakage_hist.png')
        plot_leak_age_scatter(dev_report, imgs / 'speaker_leakage_age_scatter.png')
        plot_cos_vs_dage(dev_report, imgs / 'speaker_leakage_cos_vs_dage.png')

    # ---------- Phase 2: speaker-only Ridge baseline ----------
    print(f'\n{"="*72}')
    print('PHASE 2 — speaker-only age baseline (Ridge α=10)')
    print(f'{"="*72}')
    print(f'  (a perfect-leakage scenario approaches the wav2vec2 finetune number)')
    print()

    # Mask out any train rows without an age label (shouldn't happen for train
    # but defensive — the loader emits NaN for "?").
    mask_t = ~np.isnan(y_train)
    mask_d = ~np.isnan(y_dev)
    yt = y_train[mask_t]
    yd = y_dev[mask_d]

    ecapa_res = ridge_baseline(
        'ECAPA(192) → age',
        X_train[mask_t], yt,
        X_dev[mask_d], yd,
    )
    u_train_x, X_train_x, y_train_x = xvect[args.trainset]
    u_dev_x,   X_dev_x,   y_dev_x   = xvect['dev']
    mask_tx = ~np.isnan(y_train_x)
    mask_dx = ~np.isnan(y_dev_x)
    xvect_res = ridge_baseline(
        'x-vector(512) → age',
        X_train_x[mask_tx], y_train_x[mask_tx],
        X_dev_x[mask_dx], y_dev_x[mask_dx],
    )

    # Also: concatenated ECAPA + x-vector (uses both speaker models).
    if (u_train == u_train_x).all() and (u_dev == u_dev_x).all():
        combo_tr = np.concatenate([X_train[mask_t], X_train_x[mask_t]], axis=1)
        combo_de = np.concatenate([X_dev[mask_d], X_dev_x[mask_d]], axis=1)
        combo_res = ridge_baseline(
            'ECAPA + x-vector → age',
            combo_tr, yt, combo_de, yd,
        )
    else:
        # UUIDs out of order between caches — rare; skip the combo.
        print('  (UUID order differs across ECAPA / x-vec caches; skipping concat)')
        combo_res = None

    # ---------- Verdict ----------
    ridge_results = {'ecapa': ecapa_res, 'xvect': xvect_res}
    if combo_res:
        ridge_results['combo'] = combo_res

    print_verdict(dev_report, ridge_results, FINETUNE_BASELINE_DEV_MAE,
                  args.threshold)

    # ---------- Persist ----------
    audit = {
        'finetune_baseline_dev_mae': FINETUNE_BASELINE_DEV_MAE,
        'threshold': args.threshold,
        'phase1': {
            'dev_vs_train': dev_report,
            'evl_vs_train': evl_report,
            'dev_intra_n_above_threshold': n_intra,
        },
        'phase2_ridge': ridge_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(audit, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    print(f'\nAudit JSON: {out_json.relative_to(LAB2_DIR)}')


if __name__ == '__main__':
    main()
