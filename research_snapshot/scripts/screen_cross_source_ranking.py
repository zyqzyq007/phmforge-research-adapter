"""A429: source-disjoint, bounded training of validation-evidence rankers.

Uses previously opened real PHM program results as a meta-learning corpus.
This is a qualification experiment, not a new PHM fit or Agent comparison.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

EPS = 1e-12
RULES = ['cv', 'development', 'all_group_mean', 'worst_group']
META = ['ridge_two_summary_features', 'ridge_evidence_shape']


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranks(values):
    a = np.asarray(values, dtype=float)
    # Average normalized rank, including ties; the full tied pool is rank .5.
    delta = a[:, None] - a[None, :]
    return ((delta > EPS).sum(1) + .5 * ((np.abs(delta) <= EPS).sum(1) - 1)) / (len(a) - 1)


def features(case):
    loss = np.asarray(case['group_losses'], dtype=float)
    dev_cols = case['development_columns']
    cv_cols = [j for j in range(loss.shape[1]) if j not in dev_cols]
    assert loss.ndim == 2 and np.isfinite(loss).all() and (loss >= 0).all()
    assert len(loss) == len(case['candidate_ids']) >= 2 and cv_cols and dev_cols
    cv, dev = loss[:, cv_cols], loss[:, dev_cols]
    cv_rank, dev_rank = ranks(cv.mean(1)), ranks(dev.mean(1))
    gap = float((dev.mean() - cv.mean()) / (dev.mean() + cv.mean() + EPS))
    matrix = np.column_stack([
        cv_rank, dev_rank, ranks(cv.max(1)), ranks(dev.max(1)),
        ranks(loss.std(1)), -(cv <= cv.min(0) + EPS).mean(1),
        -(dev <= dev.min(0) + EPS).mean(1), (cv_rank - dev_rank) * gap,
    ])
    matrix -= matrix.mean(0)
    controls = dict(cv=cv.mean(1).tolist(), development=dev.mean(1).tolist(),
                    all_group_mean=loss.mean(1).tolist(), worst_group=loss.max(1).tolist())
    return matrix, controls


def choice(scores, old):
    tied = [i for i, v in enumerate(scores) if v <= min(scores) + EPS]
    return old if old in tied else tied[0]


def normalized(loss):
    loss = np.asarray(loss, dtype=float)
    scale = float(np.ptp(loss))
    return (loss - loss.min()) / scale if scale > EPS else np.zeros(len(loss))


def source_mean(rows, value):
    groups = defaultdict(list)
    for row in rows:
        groups[row['cluster']].append(row[value])
    return float(np.mean([np.mean(vals) for vals in groups.values()]))


def load_corpus(run):
    cfg = read(run / 'PROTOCOL429.json')
    cases, gold, checks = [], {}, 0
    for source in cfg['sources']:
        archive = Path(source['archive'])
        assert sha(archive / 'SOURCE_MANIFEST.json') == source['manifest_sha256']
        index = {e['path']: e for e in read(archive / 'SOURCE_MANIFEST.json')['files']}
        for entry in source['files']:
            assert sha(run / entry['input']) == entry['sha256'] == index[entry['relative']]['sha256']
            checks += 1
        raw_cases = read(run / source['files'][0]['input'])
        raw_result = read(run / source['files'][1]['input'])
        raw_scores = raw_result['candidate_scores' if source['tag'] == 'A328' else 'scores']
        for c in raw_cases:
            assert c['id'] not in gold
            matrix, controls = features(c)
            cluster = cfg['cluster_map'][c['source']]
            scores = raw_scores[c['id']]
            assert set(scores) == set(c['candidate_ids'])
            values = [float(scores[k]['loss']) for k in c['candidate_ids']]
            assert np.isfinite(values).all() and min(values) >= 0
            cases.append(dict(case=c['id'], cluster=cluster, ids=c['candidate_ids'],
                              original_incumbent=c['original_incumbent'],
                              features=matrix.tolist(), controls=controls, raw_validation=c))
            gold[c['id']] = dict(losses=values, raw=scores)
    assert len(cases) == cfg['expected_pools'] and sum(len(c['ids']) for c in cases) == cfg['expected_memberships']
    assert len({c['cluster'] for c in cases}) == 4
    return cfg, cases, gold, checks


def make_fold(cfg, cases, gold, held):
    train = [c for c in cases if c['cluster'] != held]
    test = [c for c in cases if c['cluster'] == held]
    groups = Counter(c['cluster'] for c in train)
    assert len(groups) == 3 and held not in groups
    rows = []
    history = []
    for c in train:
        y = normalized(gold[c['case']]['losses'])
        old = c['ids'].index(c['original_incumbent'])
        history.append(dict(case=c['case'], cluster=c['cluster'],
                            **{r: float(y[choice(c['controls'][r], old)]) for r in RULES}))
        target = y - y.mean()
        for cid, feature, value in zip(c['ids'], c['features'], target):
            rows.append(dict(case=c['case'], cluster=c['cluster'], candidate=cid,
                             x=feature, y=float(value),
                             weight=1. / (len(groups) * groups[c['cluster']] * len(c['ids']))))
    history_scores = {r: source_mean(history, r) for r in RULES}
    rule = min(RULES, key=lambda r: (history_scores[r], RULES.index(r)))
    return dict(held_cluster=held, lambda_=cfg['ridge_lambda'], train=rows,
                history_rows=history, history_scores=history_scores, history_rule=rule,
                test=test, target_source_outcomes_in_input=False)


def fit_fold(path):
    start = time.monotonic()
    fold = read(path)
    train = fold['train']
    assert all(r['cluster'] != fold['held_cluster'] for r in train)
    assert all(c['cluster'] == fold['held_cluster'] for c in fold['test'])
    full = np.asarray([r['x'] for r in train])
    y = np.asarray([r['y'] for r in train])
    w = np.asarray([r['weight'] for r in train])
    assert np.isclose(w.sum(), 1.)
    outputs = {}
    for name, width in zip(META, [2, 8]):
        x = full[:, :width]
        prior = np.zeros(width)
        prior[:2] = .5
        lam = fold['lambda_']
        lhs = x.T @ (w[:, None] * x) + lam * np.eye(width)
        rhs = x.T @ (w * y) + lam * prior
        beta = np.linalg.solve(lhs, rhs)
        assert np.isfinite(beta).all()
        outputs[name] = dict(beta=beta.tolist(), prior=prior.tolist(),
                             residual=float(np.max(np.abs(lhs @ beta - rhs))),
                             scores={c['case']: (np.asarray(c['features'])[:, :width] @ beta).tolist()
                                     for c in fold['test']})
    return dict(held_cluster=fold['held_cluster'], train_rows=len(train),
                train_sources=sorted({r['cluster'] for r in train}),
                input_sha256=sha(path), history_rule=fold['history_rule'], models=outputs,
                actual_meta_fits=2, seconds=time.monotonic() - start)


def make_decisions(cases, outputs):
    decisions = []
    for c in cases:
        out = outputs[c['cluster']]
        scores = dict(c['controls'])
        scores['history_selected_validation_rule'] = scores[out['history_rule']]
        scores.update({m: out['models'][m]['scores'][c['case']] for m in META})
        for old, cid in enumerate(c['ids']):
            for method in ['keep', *RULES, 'history_selected_validation_rule', *META]:
                selected = old if method == 'keep' else choice(scores[method], old)
                decisions.append(dict(case=c['case'], cluster=c['cluster'], method=method,
                                      incumbent=cid, selected=c['ids'][selected],
                                      original_anchor=cid == c['original_incumbent']))
    return decisions


def assess(decisions, cases, gold):
    lookup = {c['case']: c for c in cases}
    rows = []
    for d in decisions:
        c = lookup[d['case']]
        values = gold[d['case']]['losses']
        n = normalized(values)
        old, new = c['ids'].index(d['incumbent']), c['ids'].index(d['selected'])
        rows.append(dict(**d, loss=values[new], old_loss=values[old],
                         regret=float(n[new]), harm=float(max(0, n[new] - n[old])),
                         gain=float(max(0, n[old] - n[new])), opportunity=float(n[old]),
                         updated=new != old))
    summaries = {}
    for method in sorted({r['method'] for r in rows}):
        summaries[method] = {}
        for panel in ['original', 'all_anchors']:
            selected = [r for r in rows if r['method'] == method and (panel == 'all_anchors' or r['original_anchor'])]
            pools = []
            for c in cases:
                rr = [r for r in selected if r['case'] == c['case']]
                pools.append(dict(case=c['case'], cluster=c['cluster'],
                                  **{k: float(np.mean([v[k] for v in rr])) for k in ['regret', 'harm', 'gain', 'opportunity']}))
            total = {k: source_mean(pools, k) for k in ['regret', 'harm', 'gain', 'opportunity']}
            total['gain_capture'] = total['gain'] / total['opportunity'] if total['opportunity'] > EPS else None
            total['by_source'] = {s: {k: float(np.mean([r[k] for r in pools if r['cluster'] == s]))
                                      for k in ['regret', 'harm', 'gain', 'opportunity']}
                                  for s in sorted({c['cluster'] for c in cases})}
            total['decisions'] = len(selected)
            summaries[method][panel] = total
    main = summaries[META[1]]['original']
    hist = summaries['history_selected_validation_rule']['original']
    controls = [*RULES, 'history_selected_validation_rule', META[0]]
    wins = sum(main['by_source'][s]['regret'] < hist['by_source'][s]['regret'] - EPS for s in main['by_source'])
    ties = sum(abs(main['by_source'][s]['regret'] - hist['by_source'][s]['regret']) <= EPS for s in main['by_source'])
    gate = (all(main['regret'] < summaries[c]['original']['regret'] - EPS for c in controls)
            and all(main['harm'] <= summaries[c]['original']['harm'] + EPS
                    and (main['gain_capture'] or 0) >= (summaries[c]['original']['gain_capture'] or 0) - EPS
                    for c in ['history_selected_validation_rule', META[0]]) and wins > 2)
    return dict(rows=rows, summaries=summaries, primary_gate=bool(gate),
                source_comparison_with_history_rule=dict(win=wins, tie=ties, loss=4-wins-ties))


def run_science(run):
    start = time.monotonic()
    assert not (run / 'STARTED429.json').exists(), 'No implicit restart'
    cfg, cases, gold, checks = load_corpus(run)
    save(run / 'STARTED429.json', dict(utc=utc(), status='running', source_checks=checks))
    sources = sorted({c['cluster'] for c in cases})
    for source in sources:
        save(run / 'fold_inputs' / (source + '.json'), make_fold(cfg, cases, gold, source))
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(fit_fold, [run / 'fold_inputs' / (s + '.json') for s in sources]))
    outputs = {o['held_cluster']: o for o in values}
    for s, o in outputs.items():
        save(run / 'fold_outputs' / (s + '.json'), o)
    decisions = make_decisions(cases, outputs)
    save(run / 'DECISIONS429.json', dict(utc=utc(), decisions=decisions))
    files = [run / 'DECISIONS429.json', *sorted((run / 'fold_outputs').glob('*.json')),
             *sorted((run / 'fold_inputs').glob('*.json'))]
    save(run / 'ALL_CHOICES_SEALED429.json', dict(utc=utc(),
         files=[dict(path=str(p.relative_to(run)), sha256=sha(p)) for p in files],
         held_source_outcomes_not_used_to_fit_or_select=True,
         all_outcomes_previously_known_to_researcher=True, not_a_blind_confirmation=True))
    result = assess(decisions, cases, gold)
    result.update(utc=utc(), status='complete', seconds=time.monotonic()-start,
                  actual_meta_model_fits=8, new_PHM_fits=0, new_LLM_calls=0,
                  new_model_forwards=0, new_Agent_stages=0, new_independent_confirmations=0,
                  pools=len(cases), candidate_memberships=sum(len(c['ids']) for c in cases),
                  source_clusters=len(sources), confirmation_panels_opened=False,
                  same_information_Agent_comparison=False, goal_complete=False)
    save(run / 'RESULT429.json', result)
    print(json.dumps(dict(status='science_complete', gate=result['primary_gate'],
                          comparison=result['source_comparison_with_history_rule'], seconds=result['seconds'])))


def audit(run):
    start = time.monotonic()
    cfg, cases, gold, checks = load_corpus(run)
    for c in cases:
        for cid in c['ids']:
            raw = gold[c['case']]['raw'][cid]
            if c['cluster'] == 'rotor':
                recalculated = sum(v['actual'] != v['predicted'] for v in raw['records']) / len(raw['records'])
            elif c['cluster'] == 'electric_motor':
                recalculated = sum(1-v['correct']/v['n'] for v in raw['records']) / len(raw['records'])
            elif c['cluster'] == 'gear_lab':
                cm = raw['confusion']
                recalls = []
                for j in range(len(cm)):
                    denominator = sum(cm[j]) + sum(row[j] for row in cm)
                    recalls.append(2*cm[j][j]/denominator if denominator else 0.)
                recalculated = 1-sum(recalls)/len(recalls)
            else:
                errors = defaultdict(list)
                for query in raw['queries']:
                    errors[query['unit_id']].append(abs(query['actual']-query['predicted']))
                recalculated = sum(sum(v)/len(v) for v in errors.values())/len(errors)
            assert abs(recalculated-raw['loss']) < 1e-10, (c['case'], cid)
            checks += 1
    for item in read(run / 'ALL_CHOICES_SEALED429.json')['files']:
        assert sha(run / item['path']) == item['sha256']; checks += 1
    outputs = {}
    for source in sorted({c['cluster'] for c in cases}):
        fold = read(run / 'fold_inputs' / (source + '.json'))
        assert fold == make_fold(cfg, cases, gold, source); checks += 1
        out = read(run / 'fold_outputs' / (source + '.json'))
        assert out['train_sources'] == sorted({c['cluster'] for c in cases} - {source})
        assert {c['case'] for c in fold['test']} == {c['case'] for c in cases if c['cluster'] == source}
        assert set(r['case'] for r in fold['train']).isdisjoint(c['case'] for c in fold['test'])
        checks += 3
        for method, width in zip(META, [2, 8]):
            x = np.asarray([r['x'][:width] for r in fold['train']])
            y = np.asarray([r['y'] for r in fold['train']])
            rootw = np.sqrt([r['weight'] for r in fold['train']])
            prior = np.zeros(width); prior[:2] = .5
            # Separate augmented least-squares (SVD), not the normal equations.
            aa = np.vstack([rootw[:, None] * x, np.sqrt(fold['lambda_']) * np.eye(width)])
            bb = np.concatenate([rootw * y, np.sqrt(fold['lambda_']) * prior])
            beta = np.linalg.lstsq(aa, bb, rcond=None)[0]
            assert np.allclose(beta, out['models'][method]['beta'], atol=1e-12, rtol=1e-12)
            checks += width
            for c in fold['test']:
                prediction = np.asarray(c['features'])[:, :width] @ beta
                assert np.allclose(prediction, out['models'][method]['scores'][c['case']], atol=1e-12, rtol=1e-12)
                checks += len(c['ids'])
        outputs[source] = out
    decisions = read(run / 'DECISIONS429.json')['decisions']
    assert decisions == make_decisions(cases, outputs); checks += len(decisions)
    result = read(run / 'RESULT429.json')
    # Independent scalar aggregation, without the vector normalization or assess().
    for method in result['summaries']:
        for panel in ['original', 'all_anchors']:
            pools = []
            for c in cases:
                vals = gold[c['case']]['losses']; low, high = min(vals), max(vals)
                dr = [d for d in decisions if d['method'] == method and d['case'] == c['case']
                      and (panel == 'all_anchors' or d['original_anchor'])]
                sums = Counter()
                for d in dr:
                    old, new = vals[c['ids'].index(d['incumbent'])], vals[c['ids'].index(d['selected'])]
                    scale = high - low
                    for key, numerator in [('regret', new-low), ('harm', max(new-old, 0)),
                                           ('gain', max(old-new, 0)), ('opportunity', old-low)]:
                        sums[key] += numerator / scale if scale > EPS else 0.
                pools.append(dict(cluster=c['cluster'], **{k: v/len(dr) for k, v in sums.items()}))
            for key in ['regret', 'harm', 'gain', 'opportunity']:
                value = source_mean(pools, key)
                assert abs(value - result['summaries'][method][panel][key]) < 1e-12
                checks += 1
    main = result['summaries'][META[1]]['original']
    history = result['summaries']['history_selected_validation_rule']['original']
    win = sum(main['by_source'][s]['regret'] < history['by_source'][s]['regret']-EPS
              for s in main['by_source'])
    tie = sum(abs(main['by_source'][s]['regret']-history['by_source'][s]['regret']) <= EPS
              for s in main['by_source'])
    controls = [*RULES, 'history_selected_validation_rule', META[0]]
    expected_gate = (all(main['regret'] < result['summaries'][k]['original']['regret']-EPS for k in controls)
                     and all(main['harm'] <= result['summaries'][k]['original']['harm']+EPS
                             and (main['gain_capture'] or 0) >= (result['summaries'][k]['original']['gain_capture'] or 0)-EPS
                             for k in ['history_selected_validation_rule', META[0]]) and win > 2)
    assert result['primary_gate'] == bool(expected_gate)
    assert result['source_comparison_with_history_rule'] == dict(win=win, tie=tie, loss=4-win-tie)
    checks += 2
    save(run / 'AUDIT429.json', dict(utc=utc(), passed=True, checks=checks,
         source_binding_checks=4, source_folds_disjoint=True, G1_G2_in_same_fold=True,
         independent_SVD_model_resolves=8, held_source_scores_absent_from_fold_inputs=True,
         independent_metric_recount=True, raw_candidate_losses_recounted=107,
         source_result_sha256=sha(run/'RESULT429.json'),
         seconds=time.monotonic()-start, new_PHM_fits=0, new_LLM_calls=0))
    print(json.dumps(dict(status='audit_complete', checks=checks, seconds=time.monotonic()-start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'audit'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    (run_science if args.action == 'run' else audit)(args.run.resolve())
