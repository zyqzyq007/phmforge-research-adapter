"""Replay A305's already fitted components; never search weights or fit targets."""
import argparse
import json
from pathlib import Path
import runpy
import sys

import joblib
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cohort', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    out = Path(a.output_dir)
    z = np.load(a.input, allow_pickle=False)
    assert 'labels' not in z and 'groups' not in z
    x = z['features'].astype(float)
    ids = z['sample_ids'].astype(str)
    rows = []

    def emit(name, pred, role):
        pred = np.asarray(pred)
        assert pred.shape == (len(ids),) and set(pred.tolist()) <= set(range(5))
        (out / (name + '.json')).write_text(json.dumps({'predictions': [
            dict(sample_id=str(i), predicted_label=int(v)) for i, v in zip(ids, pred)
        ]}) + '\n')
        rows.append(dict(name=name, role=role))

    sys.argv = ['/app/solution.py', '--input', a.input, '--output', str(out/'complete.json')]
    runpy.run_path('/app/solution.py', run_name='__main__')
    rows.append(dict(name='complete', role='unchanged submitted program'))
    if a.cohort == 'a':
        from modeling import spectral_features
        model = joblib.load('/matched/matched_et_spectral_leaf2.joblib')
        emit('discarded_spectral_recipe', model.predict(spectral_features(z['waveforms'])),
             'actual discarded recipe;one prescore matched refit,not original saved weights')
    elif a.cohort == 'a_review':
        from modeling import spectral_features
        for name, file, features in [
            ('spectral', 'rotor_spectral_extra_trees', spectral_features(z['waveforms'])),
            ('logistic', 'rotor_raw_logistic', x),
        ]:
            emit('component_'+name, joblib.load('/app/models/'+file+'.joblib').predict(features),
                 'saved component ablation;not automatically a discarded complete program')
    elif a.cohort in ['b', 'b_review']:
        from rotor_model import augment_load
        bundle = joblib.load('/app/rotor_model.joblib')
        for component in bundle['components']:
            features = x if component['load_kind'] == 'none' else augment_load(
                x, z['operating_load_Nm'].astype(float), component['load_kind'])
            emit('component_'+component['name'], component['model'].predict(features),
                 'saved component ablation;record classifier inapplicable without groups')
    elif a.cohort in ['c', 'c_review']:
        from rotor_pipeline import transform_features
        bundle = joblib.load('/app/artifacts/rotor_ensemble.joblib')
        views = transform_features(x)
        for name, model in bundle['models'].items():
            emit('component_'+name, model.predict(views[bundle['model_views'][name]]),
                 'saved component ablation;record classifier inapplicable without groups')
    elif a.cohort == 'd_review':
        from rotor_pipeline import observation_input, normalize_by_load
        bundle = joblib.load('/app/model.joblib')
        features, loads, _ = observation_input(z)
        normalized = normalize_by_load(features, loads)
        for name, key in [('svc', 'model'), ('extra_trees', 'ensemble_model')]:
            emit('component_'+name, bundle[key].predict(normalized),
                 'saved component;fixed unsupervised per-load adaptation retained')
    if a.cohort in ['a', 'c']:
        frozen = json.loads(Path('/controls/FROZEN_SELECTION.json').read_text())
        for candidate in frozen['candidates']:
            model = joblib.load('/controls/'+candidate['name']+'_train.joblib')
            emit('bank_'+candidate['name'], model.predict(x), 'prespecified fitted bank')
    (out/'VARIANTS.json').write_text(json.dumps(dict(
        variants=rows, new_supervised_fits=0, final_labels_read=False,
        d_rule='existing per-load unsupervised normalization executes unchanged'
    ), indent=2)+'\n')


if __name__ == '__main__':
    main()
