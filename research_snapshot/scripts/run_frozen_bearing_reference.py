"""Apply the unchanged A11 feature/3NN reference to new complete records."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    source = args.source.resolve()
    expected = '575d0d9ab6ef05479d70519c6f4322ab092159c7aef2d22f9702b3ed508bbc5b'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('frozen_reference_features', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = run / 'reference83'
    out.mkdir(exist_ok=False)
    tasks = [j for j in json.loads((run / 'plan83.json').read_text())['jobs'] if j['id'].startswith('Ottawa_')]
    cache = {}
    summary = []
    started = time.monotonic()
    for mode in ['timefreq_stat', 'envelope_order']:
        for job in tasks:
            work = run / job['workspace']
            refs = json.loads((work / 'references.json').read_text())
            queries = json.loads((work / 'queries.json').read_text())
            x, y = [], []
            for r in refs:
                key = (mode, r['record_id'])
                if key not in cache:
                    cache[key] = module.features(work / r['file'], mode)
                x.extend(cache[key])
                y.extend([r['class_code']] * 10)
            scaler = StandardScaler().fit(x)
            model = KNeighborsClassifier(n_neighbors=3).fit(scaler.transform(x), y)
            rows = []
            for q in queries:
                key = (mode, q['case_id'])
                if key not in cache:
                    cache[key] = module.features(work / q['file'], mode)
                votes = model.predict(scaler.transform(cache[key]))
                code = int(np.argmax(np.bincount(votes, minlength=5)))
                rows.append({'case_id': q['case_id'], 'decision_kind': 'diagnosis', 'class_code': code, 'window_votes': votes.tolist()})
            (out / (job['id'] + '_' + mode + '.json')).write_text(json.dumps({'results': rows}, indent=2) + '\n')
            summary.append({'job_id': job['id'], 'mode': mode, 'reference_recordings': 5, 'dependent_training_windows': 50, 'query_recordings': 5, 'reference_class_order': [r['class_code'] for r in refs], 'query_labels_not_used': True})
    (out / 'receipt83.json').write_text(json.dumps({'frozen_feature_source_sha256': expected, 'scaler': 'training references only', 'model': '3NN, majority of10windows', 'rows': summary, 'unique_mode_record_pairs': len(cache), 'elapsed_seconds': time.monotonic() - started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Ordinary fixed task reference, not SOTA; no test selection and both predeclared modes retained'}, indent=2) + '\n')
    print(json.dumps({'reference_flows': len(summary), 'queries_per_mode': 30, 'elapsed_seconds': time.monotonic() - started}))


if __name__ == '__main__':
    main()
