"""Continuation: ComParE RBF, GeMAPS CV grid, kernel ridge + gradient boosting, final ensemble."""
import os, sys, time, pickle, warnings
from pathlib import Path

LAB2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB2_DIR))

import numpy as np
import opensmile
from sklearn.svm import SVR, LinearSVR
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
    out = {}
    for part in (TRAINSET, 'dev', 'evl'):
        slp = SLPdata(DATADIR, part, transform_id=transform_id,
                      audio_transform=make_extractor(fs),
                      chunk_transform=None, chunk_size=-1, chunk_hop=-1)
        d = prepare_slp_data(slp)
        age = d['label'][:, 1]
        if part != 'evl':
            age = age.astype(np.float32)
        out[part] = (d['data'], age, d['identifiers'])
    return out


def eval_model(name, pipe, parts, results):
    Xtr, ytr, _ = parts[TRAINSET]
    Xdv, ydv, fids_dv = parts['dev']
    Xev, _, fids_ev = parts['evl']
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    pred_dv = pipe.predict(Xdv)
    pred_ev = pipe.predict(Xev)
    mae = mean_absolute_error(ydv, pred_dv)
    mse = mean_squared_error(ydv, pred_dv)
    results[name] = dict(mae=mae, mse=mse,
                         pred_dv=pred_dv, pred_ev=pred_ev,
                         fids_dv=fids_dv, fids_ev=fids_ev, ytrue_dv=ydv)
    print(f'  {name:60s} MAE={mae:6.3f}  MSE={mse:7.2f}  ({time.time()-t0:.1f}s)', flush=True)
    pickle.dump(results, open(CHECKPOINT, 'wb'))


def main():
    results = pickle.load(open(CHECKPOINT, 'rb'))
    print(f'Resuming with {len(results)} results from checkpoint', flush=True)

    cached = {name: load_partition(name, fs) for name, fs in SMILE_SETS.items()}
    print('Features ready', flush=True)

    # Fast remaining ComParE paths
    print('\n=== ComParE remaining (fast variants) ===', flush=True)
    eval_model('compare2016 | linear SVR + scaler (LinearSVR fast)',
               Pipeline([('sc', StandardScaler()),
                         ('svr', LinearSVR(C=1.0, max_iter=5000))]),
               cached['compare2016'], results)
    eval_model('compare2016 | rbf SVR + scaler (defaults)',
               Pipeline([('sc', StandardScaler()),
                         ('svr', SVR(kernel='rbf'))]),
               cached['compare2016'], results)

    # ----- Phase 3: small CV grid on gemaps (leader) and on egemaps -----
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

    print('\n=== Phase 3: CV grid (rbf SVR) ===', flush=True)
    small_grid(cached['gemaps'], 'gemaps')
    small_grid(cached['egemaps'], 'egemaps')

    # ----- Phase 4: alt regressors -----
    print('\n=== Phase 4: alternative regressors ===', flush=True)
    for fname in ('gemaps', 'egemaps'):
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

    # ----- Phase 5: ensemble -----
    print('\n=== Phase 5: ensemble of top 3 ===', flush=True)
    ranked = sorted(results.items(), key=lambda kv: kv[1]['mae'])
    print('Top 8:', flush=True)
    for n, r in ranked[:8]:
        print(f'  {r["mae"]:6.3f}  {n}', flush=True)
    top = [r for _, r in ranked[:3]]
    avg_dv = np.mean([r['pred_dv'] for r in top], axis=0)
    avg_ev = np.mean([r['pred_ev'] for r in top], axis=0)
    ydv = top[0]['ytrue_dv']
    mae_ens = mean_absolute_error(ydv, avg_dv)
    print(f'\nEnsemble (avg top 3) MAE={mae_ens:.3f}', flush=True)

    print('\n=== FINAL TABLE ===', flush=True)
    for n, r in sorted(results.items(), key=lambda kv: kv[1]['mae']):
        print(f'  {r["mae"]:6.3f}  {n}', flush=True)
    print(f'  {mae_ens:6.3f}  >>> ENSEMBLE (avg top 3) <<<', flush=True)

    # Save submission pickles
    out_dir = os.path.join(DATADIR, TRAINSET, 'models', 'age', 'svr_ensemble_v1')
    os.makedirs(out_dir, exist_ok=True)
    pickle.dump({'hyp': avg_dv, 'fileids': top[0]['fids_dv']},
                open(os.path.join(out_dir, 'dev.pkl'), 'wb'))
    pickle.dump({'hyp': avg_ev, 'fileids': top[0]['fids_ev']},
                open(os.path.join(out_dir, 'evl.pkl'), 'wb'))
    print(f'\nSaved ensemble to {out_dir}', flush=True)

    best_name, best = ranked[0]
    out_dir2 = os.path.join(DATADIR, TRAINSET, 'models', 'age', 'svr_best_v1')
    os.makedirs(out_dir2, exist_ok=True)
    pickle.dump({'hyp': best['pred_dv'], 'fileids': best['fids_dv']},
                open(os.path.join(out_dir2, 'dev.pkl'), 'wb'))
    pickle.dump({'hyp': best['pred_ev'], 'fileids': best['fids_ev']},
                open(os.path.join(out_dir2, 'evl.pkl'), 'wb'))
    print(f'Saved best single ({best_name}) to {out_dir2}', flush=True)


if __name__ == '__main__':
    main()
