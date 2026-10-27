"""Post-hoc diagnosis of frozen local-risk selectors; no fitting or selection changes."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(min=float(a.min()), median=float(np.median(a)), mean=float(a.mean()),
                p95=float(np.quantile(a, .95)), max=float(a.max()))


def main(run):
    start = time.monotonic()
    checks = []

    def verify(condition, message):
        assert condition, message
        checks.append(message)

    protocol = read(run / 'PROTOCOL455.json')
    archive = Path(protocol['source_archive'])
    verify(sha(archive / 'SOURCE_MANIFEST.json') == protocol['source_manifest_sha256'], 'archive_manifest')
    manifest = {r['path']: r['sha256'] for r in read(archive / 'SOURCE_MANIFEST.json')['files']}
    for record in read(run / 'INPUT_BINDINGS455.json'):
        verify(sha(run / 'inputs' / record['member']) == manifest[record['member']] == record['sha256'], 'input:' + record['member'])
    root = run / 'inputs'
    train = read(root / 'TRAIN_INPUT454.json')
    queries = read(root / 'QUERY_FEATURES454.json')
    bank = read(root / 'BASE_PREDICTIONS454.json')
    results = read(root / 'RESULT454.json')
    legal = read(root / 'LEGAL_MEMBERS454.json')
    legal_by_id = {r['id']: r for r in legal}
    original = {(r['case'], r['method']): r for r in results['rows']}
    with np.load(root / 'query/HUST_BATTERY/private_inputs/development_observations.npz', allow_pickle=False) as d:
        hust_query = dict(d)
    with np.load(root / 'outcomes/HUST_BATTERY/private_inputs/discovery_truth.npz', allow_pickle=False) as d:
        hust_truth = dict(d)
    with np.load(root / 'train/HUST_BATTERY/private_inputs/training_queries_private.npz', allow_pickle=False) as d:
        hust_train = dict(d)
    verify(np.array_equal(hust_query['sample_ids'], hust_truth['sample_ids']), 'hust_truth_ids')
    verify('cap' not in hust_query and 'y' not in hust_query, 'hust_legal_query_fields')
    truths = dict(COMFAULDA={r['group']: r['actual'] for r in read(root / 'outcomes/COMFAULDA/evaluation/common_p00/RESULT.json')['records']},
                  HUST_BATTERY=dict(zip(hust_query['groups'], hust_truth['y'])))
    coverage = {}
    for family, source in train['sources'].items():
        tf = read(root / 'fitted' / family / 'TRANSFORM.json')
        z, qz = np.asarray(tf['z']), np.asarray(queries[family]['z'])
        distances = np.sqrt(((z[:, None] - z[None]) ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        qdist = np.sqrt(((qz[:, None] - z[None]) ** 2).sum(axis=2))
        k = int(np.ceil(np.sqrt(len(z))))
        train_neighbor = np.sort(distances, axis=1)[:, :k].mean(axis=1)
        query_neighbor = np.sort(qdist, axis=1)[:, :k].mean(axis=1)
        x, qx = np.asarray(source['x']), np.asarray(queries[family]['x'])
        verify(x.shape[1] == qx.shape[1], family + '_feature_width')
        for i in range(len(qz)):
            brute = sorted(float(np.linalg.norm(qz[i] - row)) for row in z)[:k]
            verify(np.isclose(query_neighbor[i], np.mean(brute)), family + '_distance_' + str(i))
        residuals = []
        for raw in [x, qx]:
            centered = (raw - tf['mean']) / tf['scale'] - tf['pca_mean']
            projection = (centered @ np.asarray(tf['components']).T) @ np.asarray(tf['components'])
            residuals.append(np.sqrt(((centered - projection) ** 2).mean(axis=1)))
        outside = (qx < x.min(axis=0) - 1e-10) | (qx > x.max(axis=0) + 1e-10)
        coverage[family] = dict(train_units=len(x), query_units=len(qx), raw_features=x.shape[1],
            k=k, train_LOO_k_neighbor_distance=summary(train_neighbor), query_k_neighbor_distance=summary(query_neighbor),
            query_distance_over_training95=int((query_neighbor > np.quantile(train_neighbor, .95)).sum()),
            training_PCA_residual_rms=summary(residuals[0]), query_PCA_residual_rms=summary(residuals[1]),
            query_residual_over_training95=int((residuals[1] > np.quantile(residuals[0], .95)).sum()),
            out_of_training_coordinate_range_per_query=outside.sum(axis=1).tolist(),
            unit_ids=queries[family]['unit_ids'], neighbor_distances=query_neighbor.tolist(),
            PCA_residuals=residuals[1].tolist(),
            caveat='Descriptive coverage proxies, not a distribution-equivalence test or a decision rule; full-rank small-sample PCA has near-zero training residual by construction.')
    ordering, routes, artifacts = [], [], {}
    for case in bank:
        family, key, ids = case['source'], case['case'], case['ids']
        p = np.asarray(case['predictions'])
        y = np.array([truths[family][unit] for unit in case['unit_ids']])
        errors = (p != y[:, None]).astype(float) if family == 'COMFAULDA' else abs(p - y[:, None])
        global_ids = original[key, 'global_cv']['selected']
        for j, pid in enumerate(ids):
            evidence = legal_by_id[key + '/' + pid]
            digest = evidence['content_identity']
            cv = float(np.mean(case['risk_scores']['global_cv'], axis=0)[j])
            item = artifacts.setdefault(digest, dict(source=family, members=[], CV_mean_error=cv,
                full_query_mean_error=float(errors[:, j].mean()), query_errors=errors[:, j].tolist(),
                initial=evidence['initial'], chosen_counts=dict(global_cv=0, local_knn=0, local_ridge=0)))
            verify(np.allclose(item['query_errors'], errors[:, j]), key + pid + '_duplicate_artifact_predictions')
            verify(abs(item['CV_mean_error'] - cv) < 1e-7, key + pid + '_duplicate_artifact_cv')
            item['members'].append(key + '/' + pid)
            for method in item['chosen_counts']:
                item['chosen_counts'][method] += original[key, method]['selected'].count(pid)
        for method in ['global_cv', 'local_knn', 'local_ridge']:
            risk = np.asarray(case['risk_scores'][method])
            for i, unit in enumerate(case['unit_ids']):
                signs_true, signs_pred = [], []
                for j in range(len(ids)):
                    for k in range(j + 1, len(ids)):
                        a, b = errors[i, j] - errors[i, k], risk[i, j] - risk[i, k]
                        signs_true.append(0 if abs(a) <= 1e-10 else int(np.sign(a)))
                        signs_pred.append(0 if abs(b) <= 1e-10 else int(np.sign(b)))
                a, b = np.array(signs_true), np.array(signs_pred)
                informative = a != 0
                correct = (a == b) & informative
                reversed_pairs = (a == -b) & informative
                tied_predictions = (b == 0) & informative
                verify(int(correct.sum() + reversed_pairs.sum() + tied_predictions.sum()) == int(informative.sum()), key + method + unit + '_pair_partition')
                ordering.append(dict(source=family, case=key, unit=unit, method=method,
                    comparable=int(informative.sum()), correctly_ordered=int(correct.sum()), reversed=int(reversed_pairs.sum()),
                    predicted_ties=int(tied_predictions.sum()), true_ties=int((~informative).sum()),
                    fraction_correct=float(correct.sum() / informative.sum()) if informative.any() else None,
                    fraction_negative_risk=float((risk[i] < 0).mean())))
                if method == 'global_cv':
                    continue
                selected = original[key, method]['selected'][i]
                selected_index, global_index = ids.index(selected), ids.index(global_ids[i])
                routes.append(dict(source=family, case=key, unit=unit, method=method,
                    selected=selected, global_selected=global_ids[i], changed=selected != global_ids[i],
                    estimated_error_difference=float(risk[i, selected_index] - risk[i, global_index]),
                    actual_error_difference=float(errors[i, selected_index] - errors[i, global_index]),
                    full_error=float(errors[i, selected_index]), global_full_error=float(errors[i, global_index])))
    aggregates = {}
    for family in truths:
        aggregates[family] = {}
        for method in ['global_cv', 'local_knn', 'local_ridge']:
            rows = [r for r in ordering if r['source'] == family and r['method'] == method]
            comparison = [r for r in routes if r['source'] == family and r['method'] == method]
            changed = [r for r in comparison if r['changed']]
            n = sum(r['comparable'] for r in rows)
            aggregates[family][method] = dict(pairwise_correct=sum(r['correctly_ordered'] for r in rows) / n,
                pairwise_reversed=sum(r['reversed'] for r in rows) / n,
                pairwise_predicted_ties=sum(r['predicted_ties'] for r in rows) / n,
                comparable_pair_instances=n, instance_mean_fraction_correct=float(np.mean([r['fraction_correct'] for r in rows if r['fraction_correct'] is not None])),
                negative_risk_fraction=float(np.mean([r['fraction_negative_risk'] for r in rows])),
                changed_query_pool_instances=len(changed), changed_actual_better=sum(r['actual_error_difference'] < -1e-10 for r in changed),
                changed_actual_worse=sum(r['actual_error_difference'] > 1e-10 for r in changed),
                changed_actual_same=sum(abs(r['actual_error_difference']) <= 1e-10 for r in changed),
                mean_actual_error_difference=float(np.mean([r['actual_error_difference'] for r in comparison])) if comparison else 0.)
    jobs = read(root / 'train/HUST_BATTERY/JOBS366.json')
    verify(np.array_equal(np.unique(hust_train['cycle']), np.unique(hust_query['cycle'])), 'same_train_query_cycle')
    verify(np.array_equal(np.unique(hust_train['age']), np.unique(hust_query['age'])), 'same_train_query_age')
    context = dict(fitting_cells={k: len(v['fitting_cells']) for k, v in jobs.items()},
                   heldout_cells={k: len(v.get('heldout_cells', [])) for k, v in jobs.items()},
                   query_cycles=np.unique(hust_query['cycle']).tolist(), completed_cycles=np.unique(hust_query['age']).tolist(),
                   training_query_cycles=np.unique(hust_train['cycle']).tolist(),
                   training_rul_range=[float(hust_train['y'].min()), float(hust_train['y'].max())],
                   opened_query_rul_range=[float(hust_truth['y'].min()), float(hust_truth['y'].max())],
                   fit_context_difference_verified=True, causal_effect_identified=False)
    result = dict(iteration='A455',utc=datetime.now(timezone.utc).isoformat(),coverage=coverage,ordering=ordering,
                  routes=routes,artifacts=artifacts,aggregates=aggregates,hust_context=context,
                  scope='Post-hoc 12correlatedpools/2sources; all old query outcomes. No fitted selector changes.',
                  new_fits=0,new_forward=0,new_LLM=0,new_Agent=0,independent_confirmation=0,
                  seconds=time.monotonic()-start)
    (run/'RESULT455.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (run/'AUDIT455.json').write_text(json.dumps(dict(passed=True,checks=len(checks),names=checks,
        independent_confirmation=False,new_fits=0),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(aggregates=aggregates,coverage=coverage,hust_context=context,checks=len(checks)),ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
