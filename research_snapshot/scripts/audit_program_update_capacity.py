"""Fit real continuous-program controls; distinguish capacity from selection."""
import argparse
from itertools import product
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from controlled_feedback import save, sha, utc


def arrays(root):
    cfg = json.loads((root/'PROTOCOL373.json').read_text())
    source, profile = Path(cfg['source_archive']), Path(cfg['profile_archive'])
    for folder, digest in [(source, cfg['source_manifest_sha256']), (profile, cfg['profile_manifest_sha256'])]:
        assert sha(folder/'SOURCE_MANIFEST.json') == digest
    with np.load(source/'private_inputs/training_queries_private.npz', allow_pickle=False) as z:
        x, y, groups = z['x'], z['y'], z['groups']
    with np.load(source/'private_inputs/discovery_truth.npz', allow_pickle=False) as z:
        dx, dy = z['x'], z['y']
    with np.load(profile/'PROFILES371.npz', allow_pickle=False) as z:
        assert np.array_equal(groups, z['train_groups'])
        p, dp, names = z['cv_prediction'], z['development_prediction'], z['programs'].tolist()
    assert len(y) == 47 and len(dy) == 8 and p.shape == (47, 11) and dp.shape == (8, 11)
    return cfg, dict(raw_input=(x, dx), full_prediction=(p, dp)), y, dy, groups, p, dp, names


def predict(artifact, x, parent):
    z = artifact['scaler'].transform(x)
    value = artifact['model'].predict(z)+artifact['target_mean']
    if artifact['update'] == 'residual':
        value = value+parent
    return np.maximum(value, 0)


def fit(root):
    assert not (root/'HANDLE373.json').exists(), 'Never implicitly restart fits'
    cfg, spaces, y, dy, groups, p, dp, names = arrays(root)
    save(root/'HANDLE373.json', dict(pid=os.getpid(), utc=utc(), state='fitting'))
    begin, events = time.monotonic(), []
    for representation, scope, update, algorithm in product(cfg['representations'], cfg['data_scopes'], cfg['updates'], cfg['learners']):
        spec = '__'.join([representation, scope, update, algorithm])
        a, b = spaces[representation]
        for held in range(8):
            assert time.monotonic()-begin < cfg['budget']['fit_wall_seconds']
            available = np.arange(8) != held
            parent = int(np.argmin(np.mean(np.abs(dp[available]-dy[available, None]), axis=0)))
            tx, ty, base = b[available], dy[available], dp[available, parent]
            if scope == 'mixed_54':
                tx, ty, base = np.vstack([a, tx]), np.r_[y, ty], np.r_[p[:, parent], base]
            target = ty-base if update == 'residual' else ty
            scaler = StandardScaler()
            z = scaler.fit_transform(tx)
            mean = float(np.mean(target))
            if algorithm == 'ridge10':
                model = Ridge(alpha=10)
                bandwidth = None
            else:
                distances = np.sqrt(np.sum((z[:, None]-z[None, :])**2, axis=2))
                bandwidth = float(np.median(distances[distances > 0]))
                model = KernelRidge(alpha=1, kernel='rbf', gamma=1/(2*bandwidth**2))
            event = dict(spec=spec, held=held, rows=len(ty), parent=names[parent],
                fitting_development_cells=np.flatnonzero(available).tolist(),
                fitting_source_cells=groups.tolist() if scope == 'mixed_54' else [],
                target='nativeRULcycles' if update == 'direct' else 'signedRULresidual',
                actual_target_sha256=__import__('hashlib').sha256(np.ascontiguousarray(target).tobytes()).hexdigest(),
                estimator=type(model).__name__, status='started')
            events.append(event)
            save(root/'FIT_EVENTS373.json', events)
            model.fit(z, target-mean)
            coefficients = model.coef_ if algorithm == 'ridge10' else model.dual_coef_
            assert np.isfinite(coefficients).all()
            artifact = dict(scaler=scaler, model=model, target_mean=mean, update=update,
                representation=representation, data_scope=scope, parent=names[parent], parent_index=parent,
                fitting_development_cells=event['fitting_development_cells'], fitting_source_cells=event['fitting_source_cells'])
            folder = root/'models'/spec
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/('held_'+str(held)+'.joblib')
            joblib.dump(artifact, path)
            warm = float(predict(artifact, b[held:held+1], dp[held:held+1, parent])[0])
            event.update(status='completed', model_path=str(path.relative_to(root)), model_sha256=sha(path),
                learned_coefficient_norm=float(np.linalg.norm(coefficients)), training_target_mean=mean,
                kernel_bandwidth=bandwidth, warm_prediction=warm, parent_prediction=float(dp[held, parent]))
            save(root/'FIT_EVENTS373.json', events)
    assert len(events) == cfg['budget']['new_RUL_model_fits_max'] == 128
    save(root/'FITS_TERMINAL373.json', dict(exit_code=0, completed_actual_RUL_fits=128,
        learned_scalers=128, new_LLM_calls=0, fit_seconds=time.monotonic()-begin))
    print(json.dumps(dict(actual_RUL_fits=128, learned_scalers=128, new_LLM_calls=0)))


def verify(root):
    cfg, spaces, y, dy, groups, p, dp, names = arrays(root)
    assert json.loads((root/'FITS_TERMINAL373.json').read_text())['exit_code'] == 0
    assert not (root/'RESULT373.json').exists(), 'One verification, no outcome-based retry'
    events = json.loads((root/'FIT_EVENTS373.json').read_text())
    scores = {}
    for event in events:
        assert event['status'] == 'completed'
        path = root/event['model_path']
        assert sha(path) == event['model_sha256']
        artifact = joblib.load(path)
        held, parent = event['held'], artifact['parent_index']
        assert set(artifact['fitting_development_cells']) == set(range(8))-{held}
        _, b = spaces[artifact['representation']]
        cold = float(predict(artifact, b[held:held+1], dp[held:held+1, parent])[0])
        np.testing.assert_allclose(cold, event['warm_prediction'], rtol=1e-10, atol=1e-8)
        error, old = abs(cold-float(dy[held])), abs(float(dp[held, parent])-float(dy[held]))
        scores.setdefault(event['spec'], []).append(dict(held=held, prediction=cold, actual=float(dy[held]),
            absolute_error=error, parent_error=old, harm=max(0, error-old), gain=max(0, old-error),
            beats_frozen_pool_oracle=error < float(np.min(np.abs(dp[held]-dy[held])))-1e-9))
    aggregate = {name: dict(MAE=float(np.mean([r['absolute_error'] for r in rows])),
        parent_MAE=float(np.mean([r['parent_error'] for r in rows])),
        harm=float(np.mean([r['harm'] for r in rows])), gain=float(np.mean([r['gain'] for r in rows])),
        strict_parent_improvements=sum(r['absolute_error'] < r['parent_error']-1e-9 for r in rows),
        strict_parent_regressions=sum(r['absolute_error'] > r['parent_error']+1e-9 for r in rows),
        cells_beyond_old_pool=sum(r['beats_frozen_pool_oracle'] for r in rows)) for name, rows in scores.items()}
    best = min(aggregate, key=lambda i: aggregate[i]['MAE'])
    result = dict(utc=utc(), rows=scores, aggregate=aggregate, best_exploratory_spec=best,
        frozen_pool_oracle_MAE=float(np.mean(np.min(np.abs(dp-dy[:, None]), axis=1))),
        real_new_RUL_model_fits=128, learned_scalers=128, independent_cold_predict_calls=128,
        actual_models_used_to_derive_inputs_fitted_now=0, new_LLM_calls=0,
        study_role='Capacitycontrol,notinnovativeAgentmethodorhead-to-headgain. Extra7historicaldevlabelsofflineavailabletoALLnewrecipes;originalA367AgentdidnothavethoseRULlabels.',
        prospective_calendar_deployment_claim=False, all8_already_opened=True,
        independent_confirmation_scenarios=0, novelty_confirmed=False, final22_opened=False, majority_complete=False)
    save(root/'RESULT373.json', result)
    save(root/'TERMINAL373.json', dict(exit_code=0, utc=utc()))
    print(json.dumps(dict(best_exploratory_spec=best, metrics=aggregate[best],
        old_pool_oracle=result['frozen_pool_oracle_MAE'], new_RUL_fits=128, no_Agent_advantage_claim=True)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['fit', 'verify'])
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    globals()[a.action](a.run.resolve())
