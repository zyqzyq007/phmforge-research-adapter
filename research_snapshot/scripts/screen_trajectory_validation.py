"""Bounded real-cache comparison of snapshot versus observed-prefix evidence.

This is a development control, not a new PHM learner or an Agent experiment.
Selection and scoring are separate commands; selection never reads query labels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from rul_prediction_transport import choose


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def bound(cfg, path):
    path = Path(path)
    assert sha(path) == cfg['bindings'][str(path)], str(path)
    return path


def source_data(cfg, task):
    source = Path(cfg['source'])
    units = []
    for row in cfg['units'][task]:
        unit = packet(bound(cfg, source / row['path']))
        unit.update(id=row['id'], bank=row['bank'])
        units.append(unit)
    spec = cfg['tasks'][task]
    scaler = packet(bound(cfg, spec['scaler']))
    previous = read(bound(cfg, spec['previous_choices']))
    return units, scaler, previous


def prefix_indices(clock, fractions):
    positions = clock[0] + (clock[:, None] - clock[0]) * fractions
    return np.maximum(0, np.searchsorted(clock, positions, side='right') - 1)


def select_task(cfg, task):
    started = time.monotonic()
    spec, source = cfg['tasks'][task], Path(cfg['source'])
    units, scaler, previous = source_data(cfg, task)
    prior = {row['sample_id']: row for row in previous['rows']}
    fractions = np.asarray(cfg['prefix_fractions'])
    normal = lambda values: (np.sign(values) * np.log1p(abs(values)) - scaler['mean']) / scaler['std']
    for unit in units:
        assert np.all(np.diff(unit['elapsed']) > 0)
        np.testing.assert_allclose(unit['features'][:, spec['elapsed_column']], unit['elapsed'], atol=1e-6, rtol=0)
        unit['normalized'] = normal(unit['features'])
        unit['prefix_indices'] = prefix_indices(unit['elapsed'], fractions)
        unit['signature'] = unit['normalized'][unit['prefix_indices']]
    rows = []
    for case in cfg['cases'][task]:
        query = packet(bound(cfg, source / case['input']))
        assert not set(query).intersection({'labels', 'actual_rul', 'rul', 'targets'})
        current = query['features'][0]
        clock = float(current[spec['elapsed_column']])
        history = query['history_features']
        history_time = history[:, spec['elapsed_column']]
        assert np.all(np.diff(history_time) > 0) and history_time[-1] <= clock + 1e-6
        np.testing.assert_allclose(history[-1], current, rtol=1e-6, atol=1e-6)
        targets = history_time[0] + (clock - history_time[0]) * fractions
        qindices = np.maximum(0, np.searchsorted(history_time, targets, side='right') - 1)
        signature = normal(history[qindices])
        signature[-1] = normal(current)
        distances = {}
        for unit in units:
            delta = unit['normalized'] - normal(current)
            one = np.einsum('ij,ij->i', delta, delta)
            delta = unit['signature'] - signature
            sequence = np.einsum('ijk,ijk->i', delta, delta) / len(fractions)
            distances[unit['id']] = {'snapshot': one, 'prefix': sequence}
        decisions, evidence = {}, {}
        for representation in ['snapshot', 'prefix']:
            for scope in ['all', 'at_risk']:
                name = representation + '_' + scope
                eligible = [u for u in units if scope == 'all' or u['elapsed'][0] <= clock <= u['end']]
                references, losses = [], []
                for unit in eligible:
                    distance = distances[unit['id']][representation]
                    index = int(np.argmin(distance))
                    loss = abs(unit['residual'][index])
                    losses.append(loss)
                    references.append(dict(unit=unit['id'], index=index, elapsed=float(unit['elapsed'][index]),
                        distance=float(distance[index]), prefix_indices=unit['prefix_indices'][index].tolist(),
                        losses=loss.tolist(), bank=unit['bank']))
                risk = np.mean(losses, axis=0) if losses else np.asarray(previous['global_CV'])
                decisions[name] = spec['names'][choose(spec['names'], risk, spec['incumbent'])]
                evidence[name] = dict(references=references, risk=risk.tolist(), fallback=not losses)
                if representation == 'snapshot':
                    old = prior[case['sample_id']]
                    assert decisions[name] == old['choices']['feature_' + scope]
                    assert [r['unit'] for r in references] == old['support']['feature_' + scope]['units']
                    if references:
                        np.testing.assert_allclose(risk, old['support']['feature_' + scope]['estimated_MAE'], atol=1e-7, rtol=0)
        rows.append(dict(task=task, sample_id=case['sample_id'], unit=int(query['unit_ids'][0]),
            current_elapsed=clock, query_prefix_indices=qindices.tolist(),
            query_prefix_times=history_time[qindices].tolist(), choices=decisions, evidence=evidence))
    return dict(rows=rows, seconds=time.monotonic() - started,
        source_units=len(units), source_rows=sum(len(u['elapsed']) for u in units))


def select(root, cfg):
    assert not (root / 'CHOICES420.json').exists()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {task: executor.submit(select_task, cfg, task) for task in cfg['tasks']}
        results = {task: future.result() for task, future in futures.items()}
    assert time.monotonic() - started <= cfg['budget']['selection_seconds']
    save(root / 'CHOICES420.json', dict(utc=utc(), tasks=results, seconds=time.monotonic() - started,
        code_sha256=sha(root / 'code/screen_trajectory_validation.py'),
        target_labels_read=False, new_PHM_fits=0, new_model_forwards=0, new_LLM_calls=0))
    print(json.dumps({'phase': 'choices_sealed', 'seconds': time.monotonic()-started,
        'queries': {task: len(value['rows']) for task, value in results.items()}}))


def score_table(cfg, task):
    spec, source = cfg['tasks'][task], Path(cfg['source'])
    original = read(bound(cfg, spec['cached_scores']))
    if task == 'LDM_RTF':
        variants = {row['id']: row for row in original['variants'] if row.get('dataset') == task}
        cached = {name: {r['sample_id']: r for r in variants['fixed_model' if name == 'mlp' else 'fixed_' + name]['metrics']['event_predictions']} for name in spec['names']}
    else:
        variants = {row['variant']['id']: row for row in original['variants']}
        cached = {name: {r['sample_id']: r for r in variants['reference_mlp' if name == 'mlp' else 'bank_' + name]['results']['final']['metrics']['per_query']} for name in spec['names']}
    actual = cached[spec['incumbent']]
    predictions = {}
    for case in cfg['cases'][task]:
        output = read(bound(cfg, source / case['output'] / 'predictions.json'))['banks']['full']
        predictions[case['sample_id']] = output
        for name in spec['names']:
            old = cached[name][case['sample_id']]
            assert old['actual_rul'] == actual[case['sample_id']]['actual_rul']
            np.testing.assert_allclose(output[name], old['predicted_rul'], rtol=1e-6, atol=1e-6)
    assert set(predictions) == set(actual)
    return predictions, actual


def metrics(decisions, predictions, actual, incumbent):
    losses, old, best = {}, {}, {}
    for sid, row in actual.items():
        label = row['actual_rul']
        losses[sid] = abs(predictions[sid][decisions[sid]] - label)
        old[sid] = abs(predictions[sid][incumbent] - label)
        best[sid] = min(abs(value - label) for value in predictions[sid].values())
    units = sorted({row['unit_id'] for row in actual.values()})
    per_unit = []
    for unit in units:
        ids = [sid for sid, row in actual.items() if row['unit_id'] == unit]
        per_unit.append(dict(unit=unit, MAE=math.fsum(losses[sid] for sid in ids)/len(ids),
            selection_loss=math.fsum(losses[sid]-best[sid] for sid in ids)/len(ids)))
    opportunity = math.fsum(old[sid]-best[sid] for sid in actual)
    return dict(mean_bearing_MAE=math.fsum(r['MAE'] for r in per_unit)/len(units),
        mean_selection_loss=math.fsum(r['selection_loss'] for r in per_unit)/len(units),
        mean_query_harm=math.fsum(max(losses[sid]-old[sid], 0) for sid in actual)/len(actual),
        mean_query_gain=math.fsum(max(old[sid]-losses[sid], 0) for sid in actual)/len(actual),
        query_gain_capture=math.fsum(max(old[sid]-losses[sid], 0) for sid in actual)/opportunity if opportunity else None,
        per_bearing=per_unit)


def score(root, cfg):
    assert not (root / 'RESULT420.json').exists()
    choices = read(root / 'CHOICES420.json')
    save(root / 'SCORE_STARTED420.json', dict(utc=utc(), choices_sha256=sha(root / 'CHOICES420.json')))
    tasks = {}
    for task, spec in cfg['tasks'].items():
        predictions, actual = score_table(cfg, task)
        rows = choices['tasks'][task]['rows']
        policies = {name: {row['sample_id']: row['choices'][name] for row in rows} for name in rows[0]['choices']}
        old = read(bound(cfg, spec['previous_choices']))['rows']
        policies.update({'r9_' + name: {r['sample_id']: r['choices'][name] for r in old} for name in old[0]['choices']})
        policies.update({'fixed_' + name: {sid: name for sid in actual} for name in spec['names']})
        if task == 'XJTU':
            closure = read(bound(cfg, cfg['closure_control']))
            policies.update({'closure_' + name: {sid: value['model'] for sid in actual} for name, value in closure['methods'].items()})
        results = {name: metrics(value, predictions, actual, spec['incumbent']) for name, value in policies.items()}
        controls = [name for name in policies if name.startswith(('r9_', 'fixed_', 'closure_'))]
        best = min(controls, key=lambda name: (results[name]['mean_bearing_MAE'], name))
        main, control = results['prefix_at_risk'], results[best]
        gate = (main['mean_bearing_MAE'] < control['mean_bearing_MAE']-1e-6
            and main['mean_query_harm'] <= control['mean_query_harm']+1e-6
            and main['query_gain_capture'] >= control['query_gain_capture']-1e-12)
        tasks[task] = dict(methods=results, policies=policies, strongest_fixed_control=best,
            primary_gate=gate, changed_from_snapshot=sum(r['choices']['prefix_at_risk'] != r['choices']['snapshot_at_risk'] for r in rows))
    result = dict(utc=utc(), tasks=tasks, primary_gate=all(v['primary_gate'] for v in tasks.values()),
        new_PHM_fits=0, new_model_forwards=0, new_LLM_calls=0,
        data='Previously observed real LDM and XJTU discovery cohorts; not independent confirmation.',
        novelty_established=False, ordinary_Agent_comparison=False, confirmation_panel_opened=False,
        decision='Advance only if both frozen source gates pass; otherwise stop this representation, no threshold/fraction tuning on these outcomes.')
    save(root / 'RESULT420.json', result)
    print(json.dumps({'phase': 'scored', 'gate': result['primary_gate'], 'sources': {
        task: {'primary': v['methods']['prefix_at_risk'], 'control': v['strongest_fixed_control'],
               'control_MAE': v['methods'][v['strongest_fixed_control']]['mean_bearing_MAE'], 'gate': v['primary_gate']} for task,v in tasks.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['select', 'score'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    config = read(root / 'PROTOCOL420.json')
    for name, value in read(root / 'CODE_FREEZE420.json').items():
        assert sha(root / name) == value
    globals()[args.action](root, config)
