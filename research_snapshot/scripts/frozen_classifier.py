"""Fixed ordinary classifiers and observation-only inference for real PHM data."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def estimators(seed):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return [
        ('lda_auto', make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))),
        ('lda_01', make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage=.1))),
        ('logistic_c1', make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=seed))),
        ('svc_c1', make_pipeline(StandardScaler(), SVC(C=1, kernel='rbf', gamma='scale'))),
        ('rf300', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=2)),
        ('hgb100', HistGradientBoostingClassifier(max_iter=100, random_state=seed)),
    ]


def infer(model_path, input_path, output_path):
    data = np.load(input_path, allow_pickle=False)
    if 'labels' in data.files or 'groups' in data.files:
        raise ValueError('Inference requires observation-only input, without labels/groups')
    ids = data['sample_ids'].astype(str)
    x = data['features'].astype(float)
    if len(set(ids.tolist())) != len(ids) or len(x) != len(ids) or not np.isfinite(x).all():
        raise ValueError('Invalid observation identities or features')
    predicted = joblib.load(model_path).predict(x)
    rows = [dict(sample_id=str(i), predicted_label=y.item() if hasattr(y, 'item') else y)
            for i, y in zip(ids, predicted)]
    result = dict(predictions=rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    infer(a.model, a.input, a.output)
