"""Independent arithmetic/provenance checks; never fits a scientific selector."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def arrays(p):
    with np.load(p, allow_pickle=False) as data:
        return dict(data)


def audit(run):
    checks = []

    def ok(condition, name):
        assert condition, name
        checks.append(name)

    def close(a, b, name):
        ok(np.allclose(a, b, rtol=1e-8, atol=1e-7), name)

    protocol = read(run / 'PROTOCOL454.json')
    manifests = {}
    for key, conf in protocol['archives'].items():
        path = Path(conf['path']) / 'SOURCE_MANIFEST.json'
        ok(digest(path) == conf['manifest_sha256'], 'manifest:' + key)
        manifests[conf['path']] = {r['path']: r['sha256'] for r in read(path)['files']}
    for role in ['train', 'query', 'outcomes']:
        for record in read(run / (role + '_BINDINGS454.json')):
            ok(digest(run / record['copy']) == record['sha256'] ==
               manifests[record['archive']][record['member']], 'binding:' + record['copy'])
    for path, sha in read(run / 'CODE_FREEZE454.json').items():
        ok(digest(run / 'code' / path) == sha, 'code:' + path)
    fits = read(run / 'FITS_SEALED454.json')
    predictions_seal = read(run / 'ALL_PREDICTIONS_SEALED454.json')
    for path, sha in fits['files'].items():
        ok(digest(run / path) == sha, 'frozen_fit:' + path)
    ok(digest(run / 'TRAIN_INPUT454.json') == fits['train_sha256'], 'train_seal')
    ok(digest(run / 'PREDICTIONS454.json') == predictions_seal['prediction_sha256'], 'prediction_seal')
    ok(digest(run / 'BASE_PREDICTIONS454.json') == predictions_seal['base_sha256'], 'base_seal')
    ok(fits['utc'] < predictions_seal['utc'], 'fit_before_prediction')
    train = read(run / 'TRAIN_INPUT454.json')
    queries = read(run / 'QUERY_FEATURES454.json')
    bank = {b['case']: b for b in read(run / 'BASE_PREDICTIONS454.json')}
    members = read(run / 'LEGAL_MEMBERS454.json')
    legal = {r['id']: r for r in members}
    locations = read(run / 'PREDICTION_BINDINGS454.json')
    com_parts = [arrays(run / f'query/COMFAULDA/evaluation/common_p00/cold_repeat/input{i}.npz') for i in range(8)]
    ok(all('labels' not in d for d in com_parts), 'query_com_no_labels')
    com = {k: np.concatenate([d[k] for d in com_parts]) for k in ['features', 'groups', 'sample_ids', 'rotation_hz']}
    hust = arrays(run / 'query/HUST_BATTERY/private_inputs/development_observations.npz')
    ok('y' not in hust and 'cap' not in hust, 'query_hust_no_current_capacity_or_labels')
    for source in train['sources']:
        training_path = 'train/COMFAULDA/private_inputs/train.npz' if source == 'COMFAULDA' else 'train/HUST_BATTERY/private_inputs/training_queries_private.npz'
        for label, raw, expected in [('train', arrays(run / training_path), train['sources'][source]),
                                     ('query', com if source == 'COMFAULDA' else hust, queries[source])]:
            if source == 'COMFAULDA':
                ids = sorted(set(raw['groups']))
                physical = [np.concatenate([raw['features'][raw['groups'] == i].mean(axis=0),
                                            raw['features'][raw['groups'] == i].std(axis=0),
                                            [raw['rotation_hz'][raw['groups'] == i].mean()]]) for i in ids]
            else:
                ids = raw['groups'].astype(str).tolist()
                physical = np.concatenate([raw['x'], raw['past_cap'], raw['rates'],
                                           raw['age'].reshape(-1, 1), raw['cycle'].reshape(-1, 1)], axis=1)
            ok(ids == expected['unit_ids'], source + label + '_unit_alignment')
            close(physical, expected['x'], source + label + '_physical_features')
        tf = read(run / 'fitted' / source / 'TRANSFORM.json')
        x = np.asarray(train['sources'][source]['x'])
        close(x.mean(axis=0), tf['mean'], source + '_training_only_scaler')
        close(x.std(axis=0), np.where(x.std(axis=0) < 1e-14, 0, tf['scale']), source + '_training_scale')
        for label, physical, stored_z in [('train', x, tf['z']), ('query', queries[source]['x'], queries[source]['z'])]:
            z = ((np.asarray(physical) - tf['mean']) / tf['scale'] - tf['pca_mean']) @ np.asarray(tf['components']).T
            z /= np.sqrt(tf['explained_variance'])
            close(z, stored_z, source + label + '_pca_arithmetic')
    min_neighbor_gap = []
    for case in train['cases']:
        key, source = case['case'], case['source']
        z = np.asarray(read(run / 'fitted' / source / 'TRANSFORM.json')['z'])
        q = np.asarray(queries[source]['z'])
        y = np.asarray(case['errors'])
        for j, pid in enumerate(case['ids']):
            row = legal[key + '/' + pid]
            unit_errors = {}
            for fold in row['legal_report']['cv']:
                for unit in fold['records' if source == 'COMFAULDA' else 'cells']:
                    unit_errors[unit['group' if source == 'COMFAULDA' else 'cell']] = (float(unit['actual'] != unit['predicted']) if source == 'COMFAULDA' else unit['MAE'])
            close([unit_errors[i] for i in train['sources'][source]['unit_ids']], y[:, j], key + pid + '_oof_targets')
            location = locations[key + '/' + pid]
            raw = arrays(run / 'query' / location['key'] / location['member'])
            mapping = dict(zip(raw['sample_ids'].astype(str), raw['prediction']))
            expected = []
            if source == 'COMFAULDA':
                for group in queries[source]['unit_ids']:
                    votes = Counter(int(mapping[str(s)]) for s in com['sample_ids'][com['groups'] == group])
                    expected.append(min(votes, key=lambda label: (-votes[label], label)))
            else:
                expected = [mapping[str(s)] for s in hust['sample_ids']]
            close(expected, np.asarray(bank[key]['predictions'])[:, j], key + pid + '_raw_base_predictions')
        zc = z - z.mean(axis=0)
        coef = np.linalg.solve(zc.T @ zc + np.eye(z.shape[1]), zc.T @ (y - y.mean(axis=0)))
        risk = (q - z.mean(axis=0)) @ coef + y.mean(axis=0)
        close(risk, bank[key]['risk_scores']['local_ridge'], key + '_independent_ridge')
        k = int(np.ceil(np.sqrt(len(z))))
        distances = ((q[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
        order = np.argsort(distances, axis=1)
        sorted_distances = np.take_along_axis(distances, order, axis=1)
        gap = float((sorted_distances[:, k] - sorted_distances[:, k - 1]).min())
        ok(gap > 1e-10, key + '_no_ambiguous_knn_boundary')
        min_neighbor_gap.append(gap)
        close(y[order[:, :k]].mean(axis=1), bank[key]['risk_scores']['local_knn'], key + '_independent_knn')
        close(np.tile(y.mean(axis=0), (8, 1)), bank[key]['risk_scores']['global_cv'], key + '_global_cv')
    com_truth = {r['group']: r['actual'] for r in read(run / 'outcomes/COMFAULDA/evaluation/common_p00/RESULT.json')['records']}
    h = arrays(run / 'outcomes/HUST_BATTERY/private_inputs/discovery_truth.npz')
    ok(np.array_equal(h['sample_ids'], hust['sample_ids']), 'truth_sample_ids')
    truths = {'COMFAULDA': com_truth, 'HUST_BATTERY': dict(zip(hust['groups'].astype(str), h['y']))}
    result = read(run / 'RESULT454.json')
    ok(len(result['rows']) == 60 and len(train['cases']) == 12 and len(members) == 111, 'full_denominator')
    ok(len({r['content_identity'] for r in members}) == 45, 'distinct_artifacts')
    metrics = []
    for row in result['rows']:
        b = bank[row['case']]
        p = np.asarray(b['predictions'])
        ids, method, source = b['ids'], row['method'], row['source']
        old = ids.index('p00')
        if method in b['risk_scores']:
            risk = np.asarray(b['risk_scores'][method])
            choices = [old if scores[old] <= min(scores) + 1e-12 else next(i for i, v in enumerate(scores) if v <= min(scores) + 1e-12) for scores in risk]
        elif method == 'keep':
            choices = [old] * 8
        else:
            choices = None
        if choices is not None:
            ok(row['selected'] == [ids[i] for i in choices], row['case'] + method + '_choices')
            output = np.array([values[i] for values, i in zip(p, choices)])
        elif source == 'HUST_BATTERY':
            output = p.mean(axis=1)
        else:
            output = []
            for values in p:
                c = Counter(values)
                tied = [label for label, n in c.items() if n == max(c.values())]
                output.append(values[old] if values[old] in tied else min(tied))
            output = np.array(output)
        close(output, row['predictions'], row['case'] + method + '_output')
        truth = np.array([truths[source][i] for i in row['unit_ids']])
        error = (output != truth).astype(float) if source == 'COMFAULDA' else abs(output - truth)
        base_error = (p != truth[:, None]).astype(float) if source == 'COMFAULDA' else abs(p - truth[:, None])
        values = dict(loss=error.mean(), harm=np.maximum(error - base_error[:, old], 0).mean(),
                      gain=np.maximum(base_error[:, old] - error, 0).mean(),
                      regret_to_query_oracle=(error - base_error.min(axis=1)).mean())
        close(error, row['errors'], row['case'] + method + '_unit_errors')
        close([row[k] for k in values], list(values.values()), row['case'] + method + '_metrics')
        metrics.append(dict(source=source, method=method, **values))
    for method, sources in result['summaries'].items():
        for source, values in sources.items():
            for metric, value in values.items():
                close(value, np.mean([r[metric] for r in metrics if r['source'] == source and r['method'] == method]), source + method + metric)
    s = result['summaries']
    for method in ['local_knn', 'local_ridge']:
        qualified = all(s[method][src]['loss'] < s[base][src]['loss'] - 1e-12 for src in truths for base in ['global_cv', 'equal_ensemble'])
        qualified &= all(s[method][src]['harm'] <= s['global_cv'][src]['harm'] + 1e-12 and s[method][src]['gain'] >= s['global_cv'][src]['gain'] - 1e-12 for src in truths)
        ok(bool(qualified) == result['capability_qualified'][method], method + '_gate')
    report = dict(passed=True, checks=len(checks), check_names=checks, utc=datetime.now(timezone.utc).isoformat(),
                  independent_linear_solves=12, scientific_refits=0, minimum_knn_boundary_gap=min(min_neighbor_gap),
                  scope='Arithmetic, provenance, feature/label boundary and frozen rule audit; not independent scientific confirmation.')
    (run / 'AUDIT454.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'check_names'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run)
