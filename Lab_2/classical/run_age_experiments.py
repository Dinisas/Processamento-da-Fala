"""Classical openSMILE sweep: GeMAPS/eGeMAPS/ComParE functionals + SVR variants."""
import os, sys, time, pickle, warnings
from pathlib import Path

LAB2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB2_DIR))

import numpy as np
import opensmile
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pf_tools import SLPdata, prepare_slp_data

warnings.filterwarnings('ignore', category=UserWarning)

DATADIR = str(LAB2_DIR / 'lab2_data') + os.sep
TRAINSET = 'train_small'
RNG = 35731
CHECKPOINT = '/tmp/age_exp_checkpoint.pkl'

SMILE_SETS = {
    'gemaps':       opensmile.FeatureSet.GeMAPSv01b,
    'egemaps':      opensmile.FeatureSet.eGeMAPSv02,
    'compare2016':  opensmile.FeatureSet.ComParE_2016,
}


def make_extractor(fs):
    smile = opensmile.Smile(feature_set=fs, feature_level=opensmile.FeatureLevel.Functionals)
    return lambda x: smile.process_file(x).to_numpy()


def load_partition(transform_id, fs):
    extractor = make_extractor(fs)
    out = {}
    for part in (TRAINSET, 'dev', 'evl'):
        slp = SLPdata(DATADIR, part,
                      transform_id=transform_id,
                      audio_transform=extractor,
                      chunk_transform=None, chunk_size=-1, chunk_hop=-1)
        d = prepare_slp_data(slp)
        age = d['label'][:, 1]
        if part != 'evl':
            age = age.astype(np.float32)
        out[part] = (d['data'], age, d['identifiers'])
    return out


def eval_model(name, pipe, parts, results, save=True):
    Xtr, ytr, _ = parts[TRAINSET]
    Xdv, ydv, fids_dv = parts['dev']
    Xev, _, fids_ev = parts['evl']
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    pred_dv = pipe.predict(Xdv)
    pred_ev = pipe.predict(Xev)
    mae = mean_absolute_error(ydv, pred_dv)
    mse = mean_squared_error(ydv, pred_dv)
    results[name] = {
        'mae': mae, 'mse': mse,
        'pred_dv': pred_dv, 'pred_ev': pred_ev,
        'fids_dv': fids_dv, 'fids_ev': fids_ev,
        'ytrue_dv': ydv,
    }
    print(f'  {name:60s} MAE={mae:6.3f}  MSE={mse:7.2f}  ({time.time()-t0:.1f}s)', flush=True)
    if save:
        with open(CHECKPOINT, 'wb') as f:
            pickle.dump(results, f)
    return results[name]


def main():
    results = {}
    cached = {}
    print('=== Loading cached features ===', flush=True)
    for name, fs in SMILE_SETS.items():
        cached[name] = load_partition(name, fs)
        n_tr = cached[name][TRAINSET][0].shape
        print(f'  {name}: train={n_tr}', flush=True)

    # ----- PHASE 1+2: feature x quick model sweep -----
    print('\n=== Phase 1+2: feature x model sweep ===', flush=True)
    for fname, parts in cached.items():
        eval_model(f'{fname} | linear SVR (no scaler)',
                   SVR(kernel='linear'), parts, results)
        eval_model(f'{fname} | linear SVR + scaler',
                   Pipeline([('sc', StandardScaler()),
                             ('svr', SVR(kernel='linear'))]),
                   parts, results)
        eval_model(f'{fname} | rbf SVR + scaler (defaults)',
                   Pipeline([('sc', StandardScaler()),
                             ('svr', SVR(kernel='rbf'))]),
                   parts, results)

    best_name = min(results, key=lambda k: results[k]['mae'])
    best_feat = best_name.split(' | ')[0]
    print(f'\nBest after P1+2: {best_name}  MAE={results[best_name]["mae"]:.3f}', flush=True)

    # ----- PHASE 3: small grid on the BEST feature set, plus a grid on egemaps
    # (egemaps is small-dim so grid is cheap; if best is compare2016 we still
    # cover ComParE with a tinier hand-grid below). -----
    def small_grid(parts, label):
        Xtr, ytr, _ = parts[TRAINSET]
        grid = {
            'svr__C':       [1.0, 10.0, 100.0],
            'svr__epsilon': [0.5, 1.0, 2.0],
            'svr__gamma':   ['scale', 0.01],
        }
        base = Pipeline([('sc', StandardScaler()), ('svr', SVR(kernel='rbf'))])
        t0 = time.time()
        gs = GridSearchCV(base, grid, cv=5,
                          scoring='neg_mean_absolute_error',
                          n_jobs=4, refit=True)
        gs.fit(Xtr, ytr)
        print(f'  [{label}] CV best MAE={-gs.best_score_:.3f} '
              f'params={gs.best_params_} ({time.time()-t0:.1f}s)', flush=True)
        eval_model(f'{label} | SVR rbf CV-tuned', gs.best_estimator_, parts, results)

    print('\n=== Phase 3: small CV grid (rbf SVR) ===', flush=True)
    # Always run on egemaps (cheap, 88 dims) so we have a strong reference
    small_grid(cached['egemaps'], 'egemaps')
    # If best is something else, also tune on it
    if best_feat != 'egemaps':
        small_grid(cached[best_feat], best_feat)

    # ----- PHASE 4: alternative regressors on best feature set -----
    print('\n=== Phase 4: alternative regressors ===', flush=True)
    for fname in {best_feat, 'egemaps'}:
        parts = cached[fname]
        eval_model(f'{fname} | KernelRidge rbf alpha=1.0',
                   Pipeline([('sc', StandardScaler()),
                             ('kr', KernelRidge(alpha=1.0, kernel='rbf'))]),
                   parts, results)
        eval_model(f'{fname} | KernelRidge rbf alpha=0.1',
                   Pipeline([('sc', StandardScaler()),
                             ('kr', KernelRidge(alpha=0.1, kernel='rbf'))]),
                   parts, results)
        eval_model(f'{fname} | HistGBR defaults',
                   HistGradientBoostingRegressor(random_state=RNG),
                   parts, results)

    # ----- PHASE 5: ensemble of top-3 distinct -----
    print('\n=== Phase 5: ensemble of top 3 ===', flush=True)
    ranked = sorted(results.items(), key=lambda kv: kv[1]['mae'])
    print('Top 8 individual:', flush=True)
    for n, r in ranked[:8]:
        print(f'  {r["mae"]:6.3f}  {n}', flush=True)

    top = [r for _, r in ranked[:3]]
    avg_dv = np.mean([r['pred_dv'] for r in top], axis=0)
    avg_ev = np.mean([r['pred_ev'] for r in top], axis=0)
    ydv = top[0]['ytrue_dv']
    mae_ens = mean_absolute_error(ydv, avg_dv)
    print(f'\nEnsemble (avg top 3) MAE={mae_ens:.3f}', flush=True)

    print('\n=== FINAL TABLE (sorted by dev MAE) ===', flush=True)
    for n, r in sorted(results.items(), key=lambda kv: kv[1]['mae']):
        print(f'  {r["mae"]:6.3f}  {n}', flush=True)
    print(f'  {mae_ens:6.3f}  >>> ENSEMBLE (avg top 3) <<<', flush=True)

    # Persist ensemble + single-best for the submission
    out_dir = os.path.join(DATADIR, TRAINSET, 'models', 'age', 'svr_ensemble_v1')
    os.makedirs(out_dir, exist_ok=True)
    pickle.dump({'hyp': avg_dv, 'fileids': top[0]['fids_dv']},
                open(os.path.join(out_dir, 'dev.pkl'), 'wb'))
    pickle.dump({'hyp': avg_ev, 'fileids': top[0]['fids_ev']},
                open(os.path.join(out_dir, 'evl.pkl'), 'wb'))
    print(f'\nSaved ensemble dev/evl predictions to {out_dir}', flush=True)

    best_single_name, best_single = ranked[0]
    out_dir2 = os.path.join(DATADIR, TRAINSET, 'models', 'age', 'svr_best_v1')
    os.makedirs(out_dir2, exist_ok=True)
    pickle.dump({'hyp': best_single['pred_dv'], 'fileids': best_single['fids_dv']},
                open(os.path.join(out_dir2, 'dev.pkl'), 'wb'))
    pickle.dump({'hyp': best_single['pred_ev'], 'fileids': best_single['fids_ev']},
                open(os.path.join(out_dir2, 'evl.pkl'), 'wb'))
    print(f'Saved single-best ({best_single_name}) to {out_dir2}', flush=True)


if __name__ == '__main__':
    main()
