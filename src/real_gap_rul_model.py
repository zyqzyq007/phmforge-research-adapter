"""Genuine train-only regressors for explicitly qualified measured RUL sources.

LDM_RTF uses remaining acquisitions to documented threshold termination.
IMS_ENDPOINT uses remaining acquisitions to recording end, an explicit proxy.
No native final unit names, final lengths or scores occur in this module.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from real_bearing_rul_model import fit, predict
from real_rul_model import RULNetwork, write_json


def matrix(raw):
    x = np.asarray(raw['features'], dtype=float)
    assert x.ndim == 2 and np.isfinite(x).all()
    return np.sign(x) * np.log1p(np.abs(x))


def fitting_indices(raw, maximum=250):
    return np.concatenate([idx[np.unique(np.linspace(0, len(idx)-1, min(maximum, len(idx))).astype(int))]
        for u in np.unique(raw['unit_ids']) for idx in [np.flatnonzero(raw['unit_ids'] == u)]])


def train(data, output, architecture='mlp', epochs_override=None):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((data/'training_config.json').read_text())
    raw = dict(np.load(data/'train.npz', allow_pickle=False))
    assert set(raw['unit_ids'].tolist()) == set(config['fit_unit_ids'])
    assert architecture == 'mlp'
    epochs = config['epochs'] if epochs_override is None else epochs_override
    assert isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0
    indices = fitting_indices(raw)
    fitted, initial, audit = fit(matrix(raw)[indices], raw['labels'][indices], architecture,
        epochs, config, config['seed'])
    common = dict(dataset=str(raw['dataset'].item()), format='measured_endpoint_npz',
                  prediction_unit='acquisitions', target_definition=config['target_definition'])
    torch.save(dict(common, **fitted), output/'model.pt')
    torch.save(dict(common, **initial), output/'untrained_model.pt')
    audit.update(common, real_training=True, target_unit='acquisitions',
        training_rows_available=len(raw['labels']), fitting_unit_ids=config['fit_unit_ids'],
        training_indices=indices.tolist(), final_labels_accessed=False, development_used_for_fitting=False)
    write_json(output/'training_audit.json', audit)
    return audit


def fit_bank(data, output):
    data, output = Path(data), Path(output)
    raw = dict(np.load(data/'train.npz', allow_pickle=False))
    dev = dict(np.load(data/'val.npz', allow_pickle=False))
    idx = fitting_indices(raw); x, y = matrix(raw)[idx], raw['labels'][idx]
    specs = [('ridge10', make_pipeline(StandardScaler(), Ridge(alpha=10)), list(range(x.shape[1]))),
             ('et200', ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=321, n_jobs=2), list(range(x.shape[1]))),
             ('hgb100', HistGradientBoostingRegressor(max_iter=100, l2_regularization=1, random_state=321), list(range(x.shape[1])))]
    if raw['dataset'].item() == 'LDM_RTF':
        for name, cols in [('vibration', list(range(20))+[29]), ('temperature', [26,27,28,29]),
                           ('operating_conditions', list(range(20,26))+[29])]:
            specs.append((name, HistGradientBoostingRegressor(max_iter=100, l2_regularization=1, random_state=321), cols))
    output.mkdir(parents=True, exist_ok=True); rows=[]
    for name, model, cols in specs:
        model.fit(x[:,cols], y)
        checkpoint=dict(dataset=str(raw['dataset'].item()), model=model, columns=cols,
                        fitting_units=np.unique(raw['unit_ids']).tolist())
        file=output/(name+'.joblib');joblib.dump(checkpoint,file)
        p=np.maximum(0,model.predict(matrix(dev)[:,cols]))
        rows.append(dict(name=name, checkpoint=file.name, genuine_fit=True, fitting_rows=len(idx),
                         development_mae=float(np.mean(abs(p-dev['labels'])))))
    write_json(output/'bank_manifest.json',dict(starting_model='hgb100', models=rows,
        selection='Starting model fixed before looking at development errors; all alternatives retained.'))
    return rows


def infer(checkpoint_path, input_path, output_path):
    checkpoint_path, input_path, output_path = map(Path, (checkpoint_path,input_path,output_path))
    raw=dict(np.load(input_path,allow_pickle=False));assert 'labels' not in raw
    if checkpoint_path.suffix == '.joblib':
        checkpoint=joblib.load(checkpoint_path)
        values=np.maximum(0,checkpoint['model'].predict(matrix(raw)[:,checkpoint['columns']]))
    else:
        torch.set_num_threads(2)
        checkpoint=torch.load(checkpoint_path,weights_only=True,map_location='cpu')
        model=RULNetwork(checkpoint['architecture'],1,checkpoint['features'])
        model.load_state_dict(checkpoint['state_dict'])
        x=(matrix(raw)-checkpoint['mean'].numpy())/checkpoint['std'].numpy()
        values=predict(model,torch.from_numpy(x.astype(np.float32)[:,None,:]),checkpoint['target_scale'])
    assert checkpoint['dataset'] == raw['dataset'].item()
    assert np.isfinite(values).all() and len(set(raw['sample_ids'].tolist()))==len(values)
    result=dict(unit='acquisitions',predictions=[dict(sample_id=str(s),unit_id=int(u),scenario_id=int(c),predicted_rul=float(v))
        for s,u,c,v in zip(raw['sample_ids'],raw['unit_ids'],raw['scenario_ids'],values)])
    write_json(output_path,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();infer(a.model,a.input,a.output)
