"""Resume a recorded PHM coding task with frozen data and isolated evaluation.

This is a new review of an archived submission, not a claim to reproduce an
original stochastic Agent trajectory. Native broker experiments remain available
as frozen source and separately replayable fit/predict records.
"""
import json
from pathlib import Path, PurePosixPath
import shutil
from concurrent.futures import ThreadPoolExecutor

from .data import write_json
from .environment import Environment, _parse_command, replay


def replay_case(env, case_id, out):
    """Re-execute every final query of a case, including query-isolated RUL."""
    case = next((c for c in env.cases() if c['id'] == case_id), None)
    if case is None or not case['replay_records']: raise ValueError('No recorded final executions for this case')
    out = Path(out).resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    def one(item):
        number, record = item
        try: return replay(env, record, out/str(number))
        except Exception as ex: return {'status': 'failed', 'record': record, 'error': str(ex)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, enumerate(case['replay_records'])))
    result = {'case': case_id, 'executions': rows, 'queries': len(rows),
              'status': 'completed' if all(r['status']=='completed' for r in rows) else 'failed',
              'new_agent_run': False, 'scope': 'historical program reproduction'}
    write_json(out/'CASE_RESULT.json', result)
    if result['status'] != 'completed': raise RuntimeError('Some final queries failed; inspect CASE_RESULT.json')
    return result


def recipe(env, case_id):
    case = next((c for c in env.cases() if c['id'] == case_id), None)
    if case is None: raise KeyError('Unknown case; use environment list')
    if case['kind'] != 'legacy_coding':
        raise ValueError('Native broker sessions use their frozen runtime and PROCESS records')
    if case['test_inputs_visible_during_development'] is not False:
        raise ValueError('This historical case does not declare a training-only development mount')
    if len(case['replay_records']) != 1:
        raise ValueError('Case must have one unambiguous recorded external execution')
    plan = env.read_json(case['plan'])
    job = next(j for j in plan['jobs'] if j['id'] == case['job'])
    base = str(PurePosixPath(case['plan']).parent)
    mounts = env.read_json(env.translate(job['mounts']))['mounts']
    allowed = {'/opt/python312', '/opt/phm_python/extra', '/tools', '/data', '/reference'}
    for mount in mounts:
        if mount['destination'] not in allowed:
            raise ValueError('Review the additional historical mount before enabling this case: '+mount['destination'])
        key = env.translate(mount['source'])
        if mount['destination'] == '/data':
            for name in env.files:
                if name.startswith(key+'/') and PurePosixPath(name).name in ('test.npz', 'test_labels.json'):
                    raise ValueError('A final-data file is present in the historical development tree')
    prompt = env.object(env.translate(job['prompt_file'])).read_text()
    workspace = job['workspace']
    workspace = env.translate(workspace) if workspace.startswith('/') else base+'/'+workspace
    if not any(k.startswith(workspace+'/') for k in env.files):
        recorded, _, _ = _parse_command(env.read_json(case['replay_records'][0]))
        workspace = env.translate(next(src for _, src, dst in recorded if dst == '/app'))
    return dict(case=case, job=job, mounts=mounts, prompt=prompt, workspace=workspace)


def prepare(env, case_id, out):
    """Create a coordinator recipe; only listed public mounts may enter an Agent."""
    values = recipe(env, case_id)
    out = Path(out).resolve()
    if out == env.root or out.is_relative_to(env.root) or env.root.is_relative_to(out):
        raise ValueError('Workspace must be outside the immutable environment')
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    mounts = []
    for number, mount in enumerate(values['mounts']):
        local = out/'mounts'/str(number)
        key = env.translate(mount['source'])
        env.materialize(key, local)
        mounts.append((local, mount['destination']))
    result = dict(case=values['case'], mounts=[dict(source=str(p), destination=t) for p,t in mounts],
                  workspace_source=values['workspace'], mode='review_historical_submission',
                  original_agent_trajectory_reproduced=False, hidden_data_mounted=False)
    write_json(out/'WORKBENCH.json', result)
    (out/'ORIGINAL_TASK.txt').write_text(values['prompt'])
    env.materialize(values['workspace'], out/'starter', writable=True)
    return result


def score_predictions(env, case, predictions):
    """Recompute classification/RUL subtask metrics from frozen external labels."""
    rows = predictions.get('predictions', predictions) if isinstance(predictions, dict) else predictions
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError('This output requires its original task-specific scorer; no numerical score inferred')
    identity = 'sample_id' if all('sample_id' in r for r in rows) else 'unit_id'
    if not all(identity in r for r in rows): raise ValueError('Prediction identities are missing')
    ids = [r[identity] for r in rows]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate prediction identities')
    base = str(PurePosixPath(case['plan']).parent)
    preferred = base+'/hidden/'+case['dataset']+'/labels.json'
    candidates = [preferred] if preferred in env.files else []
    candidates += [k for k in env.files if k.endswith('/labels.json') and '/hidden/' in k and k != preferred
                   and case['dataset'] in PurePosixPath(k).parts]
    gold_key = None
    for key in candidates:
        gold = env.read_json(key)
        if isinstance(gold, dict): gold = gold.get('labels', gold.get('truth', []))
        if not isinstance(gold, list) or len(gold) != len(rows): continue
        if all(isinstance(g, dict) and identity in g for g in gold) and {g[identity] for g in gold} == set(ids):
            gold_key = key; break
    if gold_key is None: raise FileNotFoundError('No preserved external labels match all submitted identities')
    from verified_benchmark_scoring import classification, regression
    if all('predicted_label' in r for r in rows) and all('label' in r for r in gold):
        metrics = classification(rows, gold)
    elif all('predicted_rul' in r for r in rows) and all('actual_rul' in r for r in gold):
        metrics = regression(rows, gold, identity=identity)
    else:
        raise ValueError('Use the preserved task-specific scorer for this output contract')
    return dict(metrics=metrics, gold_snapshot_key=gold_key, numerical_subtask_only=True,
                full_original_scenario_score=False, scoring='fresh external calculation')


def run(store, case_id, out, cli, base_url, timeout=1200, max_requests=40,
        max_input_tokens=4000000, max_output_tokens=65536, max_output_per_request=8192, smoke=False):
    """Run the same coding-agent harness against an archived task's public inputs."""
    from .agent import run_agent
    env = Environment(store)
    out = Path(out).resolve()
    if out.exists(): raise FileExistsError(out)
    values = recipe(env, case_id)
    frozen = prepare(env, case_id, out.with_name(out.name+'_inputs'))
    mounts = [(Path(m['source']), m['destination']) for m in frozen['mounts']]
    def setup(ws):
        shutil.copytree(out.with_name(out.name+'_inputs')/'starter', ws, symlinks=True)
        (ws/'TASK.md').write_text(values['prompt'])
        return {'source': values['workspace'], 'restored_historical_submission': True,
                'new_training_claimed': False, 'mode': 'review_historical_submission'}
    def evaluate(ws, target):
        result = replay(env, values['case']['replay_records'][0], target,
                        timeout=300, compare=False, submission=ws)
        score = score_predictions(env, values['case'], json.loads(Path(result['prediction_file']).read_text()))
        write_json(target/'SCORE.json', score)
        return dict(execution=result, score=score)
    context = dict(root=env.root, meta=values['case'], prepare=setup, evaluate=evaluate, mounts=mounts,
                   prompt='Review the historical PHM submission in /app/solution.py and its model files. '
                          'Read /app/TASK.md for the original task and data contract. '
                          'Only training and development data are mounted at /data; the historical reference is at /reference. '
                          'Use the provided PHMForge MCP tools or local code. You may retain, repair or replace the submission. '
                          'Save a complete runnable solution.py and its models in /app. '
                          'This is a new review, not a replay of the historical Agent decisions. Do not seek final data or scores.')
    return run_agent(env.root, out, cli, base_url, timeout=timeout, max_requests=max_requests,
                     max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens,
                     max_output_per_request=max_output_per_request, smoke=smoke, _context=context)
