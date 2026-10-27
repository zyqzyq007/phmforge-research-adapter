"""Replay frozen PHM programs with matched complete/prefix observations.

No candidate editing, extra model search, or LLM calls. The independent audit
evaluates exported numeric tree nodes without sklearn prediction methods.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(name):
    import builtins
    import pickle
    import runpy
    import types
    import numpy as np
    import pandas as pd

    started = time.monotonic()
    spec = json.loads(Path('case_spec.json').read_text()) if Path('case_spec.json').exists() else {}
    family = spec.get('family', name)
    g = runpy.run_path('original.py')
    assert not g['HAS_LGB'], 'Frozen source execution used sklearn, not LightGBM'
    models = g['models'] if family == 'rejected' else [g['model']]
    iterations = [len(m._predictors) for m in models]
    if family != 'rejected':
        iterations = [int(g['best_iter'])]
    Path('fitted.pkl').write_bytes(pickle.dumps(models))
    before_hash = digest('fitted.pkl')
    save('FIT_CHECKPOINT.json', {'dev_mae': float(g['dev_mae']),
                                'iterations': iterations, 'models_sha256': before_hash})
    f = g['build_features'] if family == 'rejected' else g['feats']
    assert f.__closure__ is None
    safe_globals = {'np': np, 'pd': pd, '__builtins__': builtins.__dict__}
    safe_globals.update({key: g[key] for key in ['SENS', 'ID', 'CYC', 'CAP']})
    safe_globals['RUL'] = 'RUL'
    f = types.FunctionType(f.__code__, safe_globals, f.__name__, f.__defaults__)
    dv = pd.read_csv('input/development.csv')
    x_all, *rest = f(dv.drop(columns='RUL'))
    rows = rest[-1]
    assert np.max(np.abs(x_all - g['Xdv'])) < 1e-7
    all_by_row = x_all[np.argsort(rows)]
    full, prefix, targets, query_rows = [], [], [], []
    for engine, frame in dv.groupby('engine_id', sort=True):
        frame = frame.sort_values('cycle')
        for fraction in [.25, .5, .75]:
            cut = int(np.floor(len(frame) * fraction))
            observed = frame.iloc[:cut].drop(columns='RUL')
            xp, *other = f(observed)
            selected_row = np.flatnonzero(other[-1] == cut - 1)
            assert len(selected_row) == 1 and observed.cycle.max() == frame.iloc[cut - 1].cycle
            original_row = int(frame.index[cut - 1])
            full.append(all_by_row[original_row])
            prefix.append(xp[selected_row[0]])
            targets.append(float(dv.iloc[original_row].RUL))
            query_rows.append({'engine': int(engine), 'fraction': fraction, 'cut_rows': cut,
                               'cycle': int(frame.iloc[cut - 1].cycle), 'row': original_row})
    full, prefix, targets = np.array(full), np.array(prefix), np.array(targets)
    n = len(g['SENS'])
    block = [] if family == 'initial' else ([full.shape[1] - 1] if family == 'selected'
                                          else list(range(6 * n + 1, 7 * n + 1)))
    block = spec.get('expected_changed_block', block)
    restored = prefix.copy()
    restored[:, block] = full[:, block]
    outside = np.ones(full.shape[1], dtype=bool)
    outside[block] = False
    outside_max = float(np.max(np.abs(full[:, outside] - prefix[:, outside])))
    assert outside_max < 1e-7, ('Unexpected changing feature', outside_max)
    matrices = [x_all, full, prefix, restored]
    combined = np.concatenate(matrices)
    individual = []
    arrays = {}
    metadata = []
    for mi, (model, count) in enumerate(zip(models, iterations)):
        assert model.__class__.__name__ == 'HistGradientBoostingRegressor'
        assert model.loss in ['absolute_error', 'squared_error']
        for stage, p in enumerate(model.staged_predict(combined), 1):
            if stage == count:
                individual.append(np.clip(p, 0, g['CAP']))
                break
        trees = []
        for ti, predictors in enumerate(model._predictors[:count]):
            assert len(predictors) == 1
            nodes = predictors[0].nodes
            assert not np.any(nodes['is_categorical'])
            key = f'm{mi}_t{ti}'
            arrays[key] = nodes
            trees.append(key)
        metadata.append({'baseline': float(model._baseline_prediction.ravel()[0]), 'trees': trees})
    predictions = np.mean(individual, axis=0)
    boundaries = np.cumsum([0] + [len(x) for x in matrices])
    pred = {k: predictions[a:b] for k, a, b in zip(
        ['all', 'full', 'prefix', 'restored'], boundaries[:-1], boundaries[1:])}
    original = np.asarray(g['predictions'])
    assert np.max(np.abs(pred['all'] - original)) < 1e-7
    assert hashlib.sha256(pickle.dumps(models)).hexdigest() == before_hash
    expected = json.loads(Path('source_result.json').read_text())['candidate_branch1_reported_scores'][family]
    if spec.get('require_original_score', True):
        assert abs(float(g['dev_mae']) - expected) < 1e-5
    assert abs(np.mean(np.abs(pred['all'] - dv.RUL.to_numpy()[rows])) - float(g['dev_mae'])) < 1e-5
    np.savez_compressed('numeric.npz', **arrays, x_all=x_all, full=full, prefix=prefix,
                        restored=restored, targets=targets, all_targets=dv.RUL.to_numpy()[rows],
                        **{f'pred_{key}': value for key, value in pred.items()})
    save('trees.json', metadata)
    save('queries.json', query_rows)
    save('RESULT.json', {'name': name, 'family': family, 'source_mae': expected,
                         'require_original_score': spec.get('require_original_score', True),
                         'replayed_mae': float(g['dev_mae']),
                         'matched_mae': {k: float(np.mean(np.abs(pred[k] - targets)))
                                         for k in ['full', 'prefix', 'restored']},
                         'feature_block': block, 'outside_block_max_change': outside_max,
                         'changed_features': np.flatnonzero(np.any(full != prefix, axis=0)).tolist(),
                         'same_weights': True, 'models_sha256': before_hash,
                         'queries': len(targets), 'engines': int(dv.engine_id.nunique()),
                         'seconds': time.monotonic() - started, 'safe_globals': sorted(safe_globals)})


def run(root):
    from mlestar_phm_adapter import make_executor
    started = time.monotonic()
    generic = (root / 'PLAN.json').exists()
    plan = json.loads((root / ('PLAN.json' if generic else 'PLAN432.json')).read_text())
    executor = make_executor(root / 'workspace', root / 'executions', root / 'code/program_training_entry.py')
    prepared = []
    for name in plan['programs']:
        cwd = root / 'workspace' / name
        cwd.mkdir(exist_ok=False)
        shutil.copytree(root / 'tasks/cmapss_runtime_check', cwd / 'input')
        shutil.copy2(root / 'candidates' / (name + '.py'), cwd / 'original.py')
        shutil.copy2(root / 'RESULT431.json', cwd / 'source_result.json')
        shutil.copy2(root / 'code/probe_prefix_context.py', cwd / 'probe_prefix_context.py')
        if name in plan.get('case_specs', {}):
            save(cwd / 'case_spec.json', plan['case_specs'][name])
        prepared.append((name, cwd))
    output = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(executor, f'import probe_prefix_context\nprobe_prefix_context.worker({name!r})\n',
                               cwd, 'replay.py', 300): name for name, cwd in prepared}
        for future in as_completed(futures):
            name = futures[future]
            try:
                output[name] = future.result()
            except Exception:
                output[name] = {'returncode': 1, 'error': traceback.format_exc()}
            save(root / ('PROGRESS.json' if generic else 'PROGRESS432.json'), output)
            print(json.dumps({'name': name, 'returncode': output[name]['returncode'],
                              'seconds': time.monotonic() - started}), flush=True)
    save(root / ('SCIENCE_TERMINAL.json' if generic else 'SCIENCE_TERMINAL432.json'), {'seconds': time.monotonic() - started,
                                           'complete': all(v['returncode'] == 0 for v in output.values()),
                                           'programs': output})
    if not all(v['returncode'] == 0 for v in output.values()):
        raise RuntimeError('Preserved replay failure; inspect checkpoint before any recovery')


def audit(root):
    import numpy as np
    started = time.monotonic()
    generic = (root / 'PLAN.json').exists()
    plan = json.loads((root / 'PLAN.json').read_text()) if generic else {}
    cases = {}
    for name in plan.get('audit_programs', ['initial', 'selected', 'rejected']):
        cwd = root / 'workspace' / name
        d = np.load(cwd / 'numeric.npz', allow_pickle=False)
        metadata = json.loads((cwd / 'trees.json').read_text())
        case = json.loads((cwd / 'RESULT.json').read_text())
        queries = json.loads((cwd / 'queries.json').read_text())
        for key in ['all', 'full', 'prefix', 'restored']:
            x = d['x_all' if key == 'all' else key]
            model_preds = []
            for model in metadata:
                p = np.full(len(x), model['baseline'])
                for tree in model['trees']:
                    nodes = d[tree]
                    at = np.zeros(len(x), dtype=int)
                    live = np.arange(len(x))
                    while len(live):
                        active = nodes[at[live]]
                        leaf = active['is_leaf'].astype(bool)
                        p[live[leaf]] += active['value'][leaf]
                        live, active = live[~leaf], active[~leaf]
                        if len(live):
                            value = x[live, active['feature_idx']]
                            left = np.where(np.isnan(value), active['missing_go_to_left'],
                                            value <= active['num_threshold']).astype(bool)
                            at[live] = np.where(left, active['left'], active['right'])
                model_preds.append(np.clip(p, 0, 125))
            prediction = np.mean(model_preds, axis=0)
            error = float(np.max(np.abs(prediction - d['pred_' + key])))
            assert error < 1e-7, (name, key, error)
            truth = d['all_targets'] if key == 'all' else d['targets']
            case['audit_' + key] = {'max_prediction_delta': error,
                                    'mae': float(np.mean(np.abs(prediction - truth)))}
        fractions = np.array([q['fraction'] for q in queries])
        case['prefix_mae_25_50'] = float(np.abs(d['pred_prefix'] - d['targets'])[fractions < .75].mean())
        case['prefix_mae_75'] = float(np.abs(d['pred_prefix'] - d['targets'])[fractions == .75].mean())
        case['query_signature'] = hashlib.sha256(json.dumps(queries, sort_keys=True).encode()).hexdigest()
        cases[name] = case
    assert len({v['query_signature'] for v in cases.values()}) == 1
    initial = cases[plan.get('reference', 'initial')]
    selected = cases[plan.get('primary', 'selected')]
    simple_choice = min(cases, key=lambda name: (cases[name]['prefix_mae_25_50'], name))
    save(root / ('AUDIT.json' if generic else 'AUDIT432.json'), {'passed': True, 'seconds': time.monotonic() - started,
                                'cases': cases, 'simple_prefix_choice': simple_choice,
                                'primary_rank_reversal': selected['audit_full']['mae'] < initial['audit_full']['mae']
                                and selected['audit_prefix']['mae'] > initial['audit_prefix']['mae'],
                                'new_LLM_calls': 0, 'new_Agent_stages': 0,
                                'new_independent_confirmation': 0, 'goal_complete': False})
    print(json.dumps({'cases': {k: v['matched_mae'] for k, v in cases.items()},
                      'simple_prefix_choice': simple_choice, 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['run', 'audit'])
    p.add_argument('root', type=Path)
    a = p.parse_args()
    (run if a.mode == 'run' else audit)(a.root)
