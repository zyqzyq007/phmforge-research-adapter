"""Cold prediction of frozen RUL pipelines, then sealed external scoring."""
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def worker():
    import numpy as np
    import pandas as pd
    import pickle
    spec = json.loads(Path('case.json').read_text())
    source = Path('source_program.py').read_text()
    function_name = 'build_features' if spec['family'] == 'rejected' else 'feats'
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    namespace = dict(np=np, pd=pd, ID='engine_id', CYC='cycle', RUL='RUL', CAP=125,
                     SENS=spec['sensor_columns'])
    # Exact authored function, with no original training module globals.
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'frozen_features.py', 'exec'), namespace)
    raw = np.loadtxt('test.txt')
    assert raw.shape[1] == len(spec['columns']) == 26
    frame = pd.DataFrame(raw, columns=spec['columns'])
    assert len(frame.engine_id.unique()) == 100
    frame['engine_id'] = frame.engine_id.astype(int)
    frame['cycle'] = frame.cycle.astype(int)
    last = frame.sort_values(['engine_id', 'cycle']).groupby('engine_id', sort=True).tail(1)
    full, *rest = namespace[function_name](frame)
    rows = rest[-1]
    x = full[np.argsort(rows)][last.index]
    models = pickle.loads(Path('fitted.pkl').read_bytes())
    before = hashlib.sha256(pickle.dumps(models)).hexdigest()
    individual = []
    for model, count in zip(models, spec['iterations']):
        assert model.__class__.__name__ == 'HistGradientBoostingRegressor'
        assert 0 < count <= len(model._predictors)
        for i, value in enumerate(model.staged_predict(x), 1):
            if i == count:
                individual.append(np.clip(value, 0., 125.))
                break
    prediction = np.mean(individual, axis=0)
    assert prediction.shape == (100,) and np.isfinite(prediction).all()
    assert hashlib.sha256(pickle.dumps(models)).hexdigest() == before
    np.savez_compressed('prediction.npz', x=x, prediction=prediction,
                        engine_ids=last.engine_id.to_numpy(), cycles=last.cycle.to_numpy())
    save('PREDICTION.json', {'name': spec['name'], 'model_sha256': digest('fitted.pkl'),
                            'source_program_sha256': digest('source_program.py'),
                            'rows': len(frame), 'engines': 100, 'same_weights': True,
                            'new_fits': 0, 'feature_function': function_name})


def run(root):
    from mlestar_phm_adapter import make_executor
    started = time.monotonic()
    plan = json.loads((root / 'PLAN434.json').read_text())
    executor = make_executor(root / 'workspace', root / 'executions', root / 'code/program_training_entry.py',
                             execution_mode='predict')
    prepared = []
    for name in plan['programs']:
        cwd = root / 'workspace' / name
        cwd.mkdir()
        for filename in ['case.json', 'fitted.pkl', 'source_program.py']:
            shutil.copy2(root / 'models' / name / filename, cwd / filename)
        shutil.copy2(root / 'public/test.txt', cwd / 'test.txt')
        shutil.copy2(root / 'code/transport_cmapss_programs.py', cwd / 'transport_cmapss_programs.py')
        prepared.append((name, cwd))
    output = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(executor, 'import transport_cmapss_programs as t\nt.worker()\n',
                               cwd, 'predict.py', 120): name for name, cwd in prepared}
        for future in as_completed(futures):
            name = futures[future]
            output[name] = future.result()
            save(root / 'PROGRESS434.json', output)
            print(json.dumps({'case': name, 'returncode': output[name]['returncode']}), flush=True)
    all_done = len(output) == 6 and all(v['returncode'] == 0 for v in output.values())
    save(root / 'PREDICTION_TERMINAL434.json', {'seconds': time.monotonic()-started,
                                             'complete': all_done, 'processes': output})
    assert all_done, 'Preserved prediction failure; no labels read'
    for p in (root / 'executions').glob('*/training_audit.json'):
        audit = json.loads(p.read_text())
        assert audit['mode'] == 'predict' and audit['completed_estimator_fits'] == 0
    save(root / 'ALL_PREDICTIONS_SEALED434.json', {
        'sealed_unix': time.time(), 'code_freeze_sha256': digest(root / 'CODE_FREEZE434.json'),
        'files': {name: digest(root / 'workspace' / name / 'prediction.npz') for name in plan['programs']},
        'labels_read_by_workers': False})


def score(root):
    import math
    import numpy as np
    started = time.monotonic()
    plan = json.loads((root / 'PLAN434.json').read_text())
    seal = json.loads((root / 'ALL_PREDICTIONS_SEALED434.json').read_text())
    assert seal['code_freeze_sha256'] == digest(root / 'CODE_FREEZE434.json')
    for path, sha in json.loads((root / 'CODE_FREEZE434.json').read_text())['files'].items():
        assert digest(root / path) == sha
    label_source = Path(plan['external_labels']['path'])
    blob = label_source.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == plan['external_labels']['sha256']
    (root / 'private').mkdir()
    (root / 'private/official_RUL.txt').write_bytes(blob)
    truth = np.fromstring(blob.decode(), sep=' ')
    assert truth.shape == (100,) and np.isfinite(truth).all()
    scores, errors, predictions, checks = {}, {}, {}, 0
    for name in plan['programs']:
        folder = root / 'workspace' / name
        assert digest(folder / 'prediction.npz') == seal['files'][name]
        d = np.load(folder / 'prediction.npz', allow_pickle=False)
        assert np.array_equal(d['engine_ids'], np.arange(1, 101))
        trees = json.loads((root / 'models' / name / 'trees.json').read_text())
        nodes_by_tree = np.load(root / 'models' / name / 'numeric.npz', allow_pickle=False)
        x, values = d['x'], []
        for model in trees:
            value = np.full(len(x), model['baseline'])
            for key in model['trees']:
                nodes = nodes_by_tree[key]
                active = np.arange(len(x))
                at = np.zeros(len(x), dtype=int)
                while len(active):
                    node = nodes[at[active]]
                    leaf = node['is_leaf'].astype(bool)
                    value[active[leaf]] += node['value'][leaf]
                    active, node = active[~leaf], node[~leaf]
                    if len(active):
                        observed = x[active, node['feature_idx']]
                        left = np.where(np.isnan(observed), node['missing_go_to_left'],
                                        observed <= node['num_threshold']).astype(bool)
                        at[active] = np.where(left, node['left'], node['right'])
            values.append(np.clip(value, 0., 125.))
        independent = np.mean(values, axis=0)
        delta = float(np.max(abs(independent - d['prediction'])))
        assert delta < 1e-7
        e = abs(independent - truth)
        mae = float(e.mean())
        assert abs(math.fsum(map(float, e))/100 - mae) < 1e-10
        errors[name] = e
        predictions[name] = independent.tolist()
        scores[name] = {'MAE_official_uncapped': mae,
                         'RMSE_official_uncapped': float(np.sqrt(np.mean((independent-truth)**2))),
                         'MAE_capped_secondary': float(np.mean(abs(independent-np.minimum(truth, 125)))),
                         'independent_prediction_max_delta': delta}
        checks += 202
    for name, e in errors.items():
        change = e - errors['initial']
        scores[name].update(harm_from_initial=float(np.maximum(change, 0).mean()),
                            gain_from_initial=float(np.maximum(-change, 0).mean()),
                            engine_win_tie_loss={'win': int((change < -1e-7).sum()),
                                                'tie': int((abs(change) <= 1e-7).sum()),
                                                'loss': int((change > 1e-7).sum())})
    selected = plan['frozen_repair_selection']
    passes = all(scores[selected]['MAE_official_uncapped'] < scores[control]['MAE_official_uncapped']-1e-9
                 for control in plan['primary_controls'])
    old = scores['selected']
    joint = passes and scores[selected]['harm_from_initial'] <= old['harm_from_initial']+1e-9 and scores[selected]['gain_from_initial'] >= old['gain_from_initial']-1e-9
    result = {'scores': scores, 'frozen_repair_selection': selected, 'primary_transport_gate': passes,
              'joint_descriptive_gate_vs_old_branch': joint, 'new_fits': 0, 'new_LLM_calls': 0,
              'actual_cold_processes': 6, 'predicted_engines_per_program': 100, 'source_clusters': 1,
              'independent_confirmation': 0, 'new_Agent_stages': 0, 'audit_checks': checks,
              'audit_seconds': time.monotonic()-started, 'goal_complete': False,
              'ordinary_Agent_comparison': False, 'all_handles_terminal': True, 'archive_pending': True}
    save(root / 'PREDICTIONS_AND_TRUTH434.json', {'truth': truth.tolist(), 'predictions': predictions})
    save(root / 'RESULT434.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    mode, path = sys.argv[1:]
    (run if mode == 'run' else score)(Path(path))
