"""Real train-only RUL regressors for measured XJTU causal feature records.

Targets are minutes to the original terminal acquisition. This module knows no
native unit names, test lifetimes, split seeds, or final evaluation fractions.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from real_bearing_rul_model import fit, predict
from real_rul_model import RULNetwork, write_json


BANK = ['median_life', 'ridge10', 'knn5', 'rf200', 'et200', 'hgb100']


def matrix(raw):
    """A fixed, label-free transform; scalers below are fitted on fitting rows."""
    x = np.asarray(raw['features'], dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == 34 and np.isfinite(x).all()
    x = x.copy()
    x[:, :31] = np.sign(x[:, :31]) * np.log1p(np.abs(x[:, :31]))
    return x


def metrics(values, labels, units):
    p, y, u = map(np.asarray, (values, labels, units))
    assert p.shape == y.shape == u.shape and np.isfinite(p).all()
    rows = []
    for unit in sorted(set(u.tolist())):
        error = p[u == unit] - y[u == unit]
        rows.append(dict(unit_id=int(unit), queries=len(error),
            mae_minutes=float(np.mean(np.abs(error))),
            rmse_minutes=float(np.sqrt(np.mean(error ** 2))),
            bias_minutes=float(np.mean(error))))
    return dict(mean_bearing_mae_minutes=float(np.mean([r['mae_minutes'] for r in rows])),
        pooled_mae_minutes=float(np.mean(np.abs(p-y))),
        pooled_rmse_minutes=float(np.sqrt(np.mean((p-y)**2))), per_bearing=rows)


def fit_bank_model(name, raw, seed=278):
    x, y = matrix(raw), np.asarray(raw['labels'], dtype=float)
    if name == 'median_life':
        lifetimes = {}
        for scenario in sorted(set(raw['scenario_ids'].tolist())):
            values = []
            for unit in sorted(set(raw['unit_ids'][raw['scenario_ids'] == scenario].tolist())):
                mask = raw['unit_ids'] == unit
                endpoints = raw['elapsed_minutes'][mask] + y[mask]
                assert np.all(endpoints == endpoints[0])
                values.append(float(endpoints[0]))
            lifetimes[int(scenario)] = float(np.median(values))
        model = lifetimes
    else:
        factories = {
            'ridge10': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
            'knn5': lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(5, weights='distance', n_jobs=2)),
            'rf200': lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                max_features=.8, random_state=seed, n_jobs=2),
            'et200': lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3,
                random_state=seed, n_jobs=2),
            'hgb100': lambda: HistGradientBoostingRegressor(max_iter=100, learning_rate=.05,
                l2_regularization=1, random_state=seed),
        }
        model = factories[name]().fit(x, y)
    return dict(dataset='XJTU', format='xjtu_npz', prediction_unit='minutes',
                name=name, model=model, fitting_units=sorted(set(raw['unit_ids'].tolist())))


def predict_bank(model, raw):
    if model['name'] == 'median_life':
        p = np.asarray([model['model'][int(s)]-t for s,t in
            zip(raw['scenario_ids'], raw['elapsed_minutes'])], dtype=float)
    else:
        p = model['model'].predict(matrix(raw))
    p = np.maximum(0, p)
    assert np.isfinite(p).all()
    return p


def train(data, output, architecture='mlp', epochs_override=None):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assert architecture == 'mlp'
    config = json.loads((data/'training_config.json').read_text())
    raw = dict(np.load(data/'train.npz', allow_pickle=False))
    assert set(raw['unit_ids'].tolist()) == set(config['fit_unit_ids'])
    epochs = config['epochs'] if epochs_override is None else epochs_override
    assert isinstance(epochs, int) and not isinstance(epochs, bool) and epochs >= 1
    checkpoint, initial, audit = fit(matrix(raw), raw['labels'], 'mlp', epochs,
        config, config['seed'], validation=None)
    common = dict(dataset='XJTU', format='xjtu_npz', prediction_unit='minutes')
    torch.save(dict(common, **checkpoint), output/'model.pt')
    torch.save(dict(common, **initial), output/'untrained_model.pt')
    audit.update(dataset='XJTU', target_unit='minutes', real_training=True,
        test_labels_accessed=False, development_labels_accessed=False,
        fitting_units=config['fit_unit_ids'], fitting_scope='train_only',
        selection='Fixed last epoch; no development feedback',
        source='Actual measured records and original terminal acquisition labels')
    write_json(output/'training_audit.json', audit)
    return audit


def infer(checkpoint_path, input_path, output_path):
    raw = dict(np.load(input_path, allow_pickle=False))
    assert raw['dataset'].item() == 'XJTU' and 'labels' not in raw
    if str(checkpoint_path).endswith('.joblib'):
        checkpoint = joblib.load(checkpoint_path)
        assert checkpoint['dataset'] == 'XJTU'
        values = predict_bank(checkpoint, raw)
    else:
        torch.set_num_threads(2)
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        assert checkpoint['format'] == 'xjtu_npz'
        model = RULNetwork(checkpoint['architecture'], 1, checkpoint['features'])
        model.load_state_dict(checkpoint['state_dict'])
        x = (matrix(raw)-checkpoint['mean'].numpy())/checkpoint['std'].numpy()
        values = predict(model, torch.from_numpy(x.astype(np.float32)[:, None, :]), checkpoint['target_scale'])
    ids = raw['sample_ids'].tolist()
    assert len(set(ids)) == len(ids) == len(values)
    result = dict(unit='minutes', predictions=[dict(sample_id=str(i), unit_id=int(u),
        scenario_id=int(s), predicted_rul=float(v)) for i,u,s,v in
        zip(ids, raw['unit_ids'], raw['scenario_ids'], values)])
    write_json(output_path, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    infer(a.model, a.input, a.output)
