"""Actual fitted references for the publisher's engine QA annotations."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import real_fault_model
from real_rul_model import write_json


def train(data, output, epochs=10):
    data, output = Path(data), Path(output)
    audit = real_fault_model.train(data, output, 'mlp', epochs)
    with np.load(data / 'train.npz', allow_pickle=False) as fit:
        x = fit['features'].astype(np.float32)
        classifiers = {}
        for name in ('state', 'timing'):
            labels = fit['states' if name == 'state' else 'timing'].astype(str)
            known = labels != ''
            if known.any():
                model = ExtraTreesClassifier(n_estimators=100, random_state=2026102309, n_jobs=2)
                model.fit(x[known], labels[known])
                classifiers[name] = model
        joblib.dump(classifiers, output / 'qa_heads.joblib', compress=3)
    audit['fitted_publisher_qa_heads'] = sorted(classifiers)
    audit['causal_or_physical_risk_validation'] = False
    write_json(output / 'training_audit.json', audit)
    return audit


def infer(checkpoint_path, input_path, output_path):
    result = real_fault_model.infer(checkpoint_path, input_path, output_path)
    with np.load(input_path, allow_pickle=False) as data:
        x = data['features'].astype(np.float32)
    classifiers = joblib.load(Path(checkpoint_path).parent / 'qa_heads.joblib')
    for name in ('state', 'timing'):
        values = classifiers[name].predict(x).tolist() if name in classifiers else [None] * len(x)
        for row, value in zip(result['predictions'], values):
            row[name] = value
    write_json(output_path, result)
    return result
