"""Separate structural, real-execution and live-model qualification gates."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import sys

from mlestar_phm_adapter import load_official, make_executor, make_model, make_search, save


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_data(root):
    import numpy as np
    shared = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks240')
    raw = shared / 'public/datasets/CMAPSS_FD001/train.txt'
    config_path = raw.with_name('training_config.json')
    config = json.loads(config_path.read_text())
    manifest = json.loads((shared / 'FILE_MANIFEST.json').read_text())
    matching = [v for v in manifest['files'] if v.get('path') == str(raw.relative_to(shared))]
    assert len(matching) == 1 and matching[0]['sha256'] == sha(raw)
    task = root / 'tasks/cmapss_runtime_check'
    task.mkdir(parents=True, exist_ok=False)
    data = np.loadtxt(raw)
    assert data.shape[1] == 26 and np.isfinite(data).all()
    train_ids, dev_ids = set(config['train_engine_ids']), set(config['validation_engine_ids'])
    assert train_ids.isdisjoint(dev_ids) and train_ids | dev_ids == set(data[:, 0].astype(int))
    maximum = {e: data[data[:, 0] == e, 1].max() for e in train_ids | dev_ids}
    y = np.asarray([min(125., maximum[int(row[0])] - row[1]) for row in data])
    matrix = np.column_stack([data, y])
    names = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3']
    names += [f'sensor_{i}' for i in range(1, 22)] + ['RUL']
    for name, ids in [('train', train_ids), ('development', dev_ids)]:
        selected = matrix[np.isin(data[:, 0].astype(int), sorted(ids))]
        np.savetxt(task / (name + '.csv'), selected, delimiter=',', header=','.join(names), comments='')
    # These are public development observations, not hidden final-test records.
    dev = matrix[np.isin(data[:, 0].astype(int), sorted(dev_ids))]
    np.savetxt(task / 'test.csv', dev[:, :-1], delimiter=',', header=','.join(names[:-1]), comments='')
    description = ('Runtime qualification on official NASA CMAPSS FD001 training trajectories. '
                   'Predict RUL in cycles from the current row of operating settings and 21 sensors. '
                   'train.csv has 80 complete engine trajectories; development.csv has the other '
                   '20 engines and their labels. RUL is capped at 125. Do not use engine_id as a '
                   'feature. Fit only on train.csv; validate on development.csv; do not randomly '
                   'split rows or fit preprocessing on development. Metric is mean absolute error. '
                   'test.csv is the SAME public development observations without RUL, solely for '
                   'checking output plumbing; it is not a hidden test set. Save submission.csv '
                   'with one RUL prediction per row in test.csv. No official test file or final '
                   'test labels are present. This run qualifies execution, not generalization. '
                   'Use locally installed numpy, pandas, scipy, sklearn or torch; network access '
                   'inside generated Python is disabled. Keep training lightweight for this smoke run.')
    (task / 'task_description.txt').write_text(description + '\n')
    save(root / 'DATA_QUALIFICATION430.json', dict(
        raw_path=str(raw), raw_sha256=sha(raw), raw_manifest_sha256=sha(shared / 'FILE_MANIFEST.json'),
        config_path=str(config_path), config_sha256=sha(config_path),
        rows=len(data), train_engines=80, development_engines=20,
        source='NASA official simulation, explicitly authorized by user',
        hidden_final_access=False, synthetic_rows_created=0,
        observation_view='Current sensor row; runtime smoke adaptation, not original sequence-model replication',
        files={p.name: sha(p) for p in task.iterdir()}))
    return task


def build(root):
    return load_official(
        root / 'source/machine_learning_engineering',
        dict(agent_model='deepseek-flash', data_dir=str(root / 'tasks'),
             workspace_dir=str(root / 'agent_workspace'), task_name='cmapss_runtime_check'),
        make_model(root / 'ipc/model.sock'), make_search(root / 'search_broker', limit=4),
        make_executor(root / 'agent_workspace', root / 'executions', root / 'code/program_training_entry.py'))


def structural(root):
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types
    from mlestar_phm_adapter import responses_payload
    module, inventory = build(root)
    llms = [r for r in inventory if 'model' in r]
    assert llms and all(r['model'] == 'deepseek-flash' for r in llms)
    assert not any('google_search' in r.get('tools', []) for r in inventory)
    stages = module.mle_pipeline_agent.sub_agents
    assert len(stages) == 4 and all(a.name.startswith(prefix) for a, prefix in zip(
        stages, ['initialization', 'refinement', 'ensemble', 'submission']))
    request = LlmRequest(contents=[
        types.Content(role='user', parts=[types.Part(text='Find public documentation')]),
        types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(
            id='call_1', name='search_web', args={'query': 'RandomForestRegressor documentation'}))]),
        types.Content(role='user', parts=[types.Part(function_response=types.FunctionResponse(
            id='call_1', name='search_web', response={'result': 'schema fixture, not an executed search'}))])],
        config=types.GenerateContentConfig(system_instruction='Preserve identities', tools=[types.Tool(
            function_declarations=[types.FunctionDeclaration(name='search_web', parameters=types.Schema(
                type='OBJECT', properties={'query': types.Schema(type='STRING')}, required=['query']))])]))
    converted = responses_payload(request)
    assert converted['input'][1]['call_id'] == converted['input'][2]['call_id'] == 'call_1'
    assert converted['tools'][0]['parameters']['type'] == 'object'
    save(root / 'ADAPTER_STRUCTURE430.json', dict(
        passed=True, agents=inventory, total_agents=len(inventory), routed_llm_agents=len(llms),
        real_search_agents=sum('search_web' in r.get('tools', []) for r in inventory),
        official_bootstrap_suppressed=True, protocol_fixture_only=True, new_LLM_calls=0,
        full_pipeline_executed=False))
    print(json.dumps({'structural_passed': True, 'agents': len(inventory), 'llms': len(llms)}))


def execute(root):
    task = prepare_data(root)
    cwd = root / 'agent_workspace/qualification'
    cwd.mkdir(parents=True, exist_ok=False)
    shutil.copytree(task, cwd / 'input')
    source = '''import json, os, socket
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
private = ['/home/jupyter-yyf-zyq/.bashrc', '/data/yyf-zyq/PHM_20261010', '/srv/yyf-zyq/PHM_20261010', '/home/jupyter-yyf-zyq/Bo_2/PHM_20261010']
assert not any(Path(p).exists() for p in private)
assert not any('API_KEY' in k or 'SECRET' in k for k in os.environ)
assert os.environ['TMPDIR'].startswith('/work/')
train = pd.read_csv('input/train.csv'); dev = pd.read_csv('input/development.csv')
assert not set(train.engine_id) & set(dev.engine_id)
columns = [c for c in train if c not in ['RUL', 'engine_id']]
model = RandomForestRegressor(n_estimators=16, max_depth=6, random_state=42, n_jobs=2)
model.fit(train[columns].iloc[::5], train.RUL.iloc[::5])
prediction = model.predict(dev[columns])
assert np.isfinite(prediction).all() and np.unique(prediction).size > 1
pd.DataFrame({'RUL':prediction}).to_csv('predictions.csv',index=False)
import joblib
joblib.dump(model, 'trained_model.joblib')
print('Final Validation Performance:', mean_absolute_error(dev.RUL, prediction))
print(json.dumps({'private_paths_absent': True, 'secrets_absent': True, 'trees': len(model.estimators_), 'train_engines': train.engine_id.nunique(), 'dev_engines': dev.engine_id.nunique()}))
'''
    executor = make_executor(root / 'agent_workspace', root / 'executions', root / 'code/program_training_entry.py')
    result = executor(source, str(cwd), 'qualification.py', 180)
    save(root / 'EXECUTOR_PROCESS430.json', result)
    assert result['returncode'] == 0, result['stderr'][-1600:]
    import numpy as np
    import pandas as pd
    pred = pd.read_csv(cwd / 'predictions.csv').RUL.to_numpy()
    truth = pd.read_csv(task / 'development.csv').RUL.to_numpy()
    audit_paths = list((root / 'executions').glob('*/training_audit.json'))
    assert len(audit_paths) == 1
    audit = json.loads(audit_paths[0].read_text())
    assert audit['completed_estimator_fits'] == 1 and audit['estimator_fit_events'][0]['class_name'] == 'RandomForestRegressor'
    reported = float(result['stdout'].split('Final Validation Performance:')[1].splitlines()[0])
    actual = float(np.mean(np.abs(pred - truth)))
    assert abs(reported - actual) < 1e-9
    save(root / 'REAL_EXECUTION430.json', dict(passed=True, new_PHM_fits=1,
        trees=16, predictions=len(pred), declared_score=reported, independently_recomputed_MAE=actual,
        evaluation_population='Public development rows only; not hidden final evaluation',
        authored_by='Controller qualification code, not an Agent submission',
        audit_sha256=sha(audit_paths[0]), trained_model_sha256=sha(cwd / 'trained_model.joblib'),
        new_LLM_calls=0, method_success_claim=False))
    print(json.dumps({'real_execution_passed': True, 'fits': 1, 'development_predictions': len(pred)}))


async def live(root):
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    module, inventory = build(root)
    # A real official retriever, not a substitute toy conversational Agent.
    agent = module.root_agent.find_agent('model_retriever_agent_1')
    assert agent is not None
    ws = root / 'agent_workspace/cmapss_runtime_check/1/model_candidates'
    ws.mkdir(parents=True, exist_ok=False)
    sessions = InMemorySessionService()
    state = dict(workspace_dir=str(root / 'agent_workspace'), task_name='cmapss_runtime_check',
                 num_model_candidates=2,
                 task_summary=(root / 'tasks/cmapss_runtime_check/task_description.txt').read_text())
    session = await sessions.create_session(app_name='qualification430', user_id='research', state=state)
    runner = Runner(agent=agent, app_name='qualification430', session_service=sessions)
    events = []
    try:
        async for event in runner.run_async(user_id='research', session_id=session.id,
                new_message=types.Content(role='user', parts=[types.Part(
                    text='Use the available real web search to check current official documentation, then produce the requested two model candidates. This is a runtime qualification.')])):
            events.append(event.model_dump(mode='json', exclude_none=True))
            save(root / 'LIVE_EVENTS430.json', events)
    finally:
        final = await sessions.get_session(app_name='qualification430', user_id='research', session_id=session.id)
        save(root / 'LIVE_STATE430.json', final.state)
    candidates = list(ws.glob('model_*.txt'))
    requests = list((root / 'search_broker').glob('*.request.json'))
    assert len(candidates) == 2 and requests
    assert all(p.with_name(p.name.replace('.request.', '.response.')).exists() for p in requests)
    save(root / 'LIVE_MODEL430.json', dict(passed=True, official_agent=agent.name,
        candidate_files={p.name: sha(p) for p in candidates}, searches=len(requests),
        full_pipeline_executed=False, real_model_requests='See budgeted gateway receipts'))
    print(json.dumps({'live_official_retriever_passed': True, 'candidates': 2, 'searches': len(requests)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--stage', choices=['structural', 'execute', 'live'], required=True)
    args = parser.parse_args()
    if args.stage == 'live':
        asyncio.run(live(args.run.resolve()))
    else:
        globals()[args.stage](args.run.resolve())
