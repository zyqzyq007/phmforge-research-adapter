"""A367: ordinary native coding on measured batteries; no final22 access."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import uuid
import joblib
import numpy as np
import torch
from author_net import CRNN
from controlled_feedback import save, sha, utc
from native_validation_comparison import closure
from free_program_comparison import arguments, environment, execute
from replay_diagnostic_cli import isolate_filesystem
from qualify_native_cli_mcp import probe
from hust_window_features import cell_examples
from hust_battery_reference import network_predictions
from hust_program_runtime import Runtime, OBSERVABLE, score

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
PYTHON = Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12')
DEPS = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')
read = lambda p: json.loads(Path(p).read_text())


def prepare(root):
    config = read(root / 'PROTOCOL337.json')
    assert not (root / 'PREPARATION367.json').exists()
    source, ref = Path(config['data_archive']), Path(config['reference_archive'])
    for path, digest in [(source, config['data_manifest_sha256']), (ref, config['reference_manifest_sha256'])]:
        assert sha(path / 'SOURCE_MANIFEST.json') == digest
        assert read(path / 'ARCHIVE_VERIFIED.json')['source_cleanup'] == 'complete'
    assert sha(Path(config['cli'])) == config['cli_sha256']
    torch.set_num_threads(2)
    qualified, train, queries, development = read(source / 'QUALIFICATION365.json'), [], [], []
    bindings = {}
    def examples(item, cycles):
        path = source / 'private' / item['cell'] / 'PRIVATE_CELL.npz'
        assert sha(path) == item['curves_sha256']
        bindings[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
        d = cell_examples(raw, cycles(raw))
        n = len(d['y'])
        d['groups'] = np.repeat(item['cell'], n)
        d['sample_ids'] = np.asarray([item['cell']+'_'+str(c) for c in d['cycle']])
        d['past_cap'] = d['cap'][:, :9].copy()
        d['rates'] = np.repeat(raw['discharge_rates'][None, :], n, axis=0)
        return d
    for item in qualified['cells']:
        if item['role'] == 'train':
            train.append(examples(item, lambda raw: range(39, int(raw['cycle'][-1])+1, 10)))
            queries.append(examples(item, lambda _: [config['query_cycle']]))
        elif item['role'] == 'development':
            d = examples(item, lambda _: [config['query_cycle']])
            d['groups'] = np.repeat('query_'+str(len(development)), len(d['y']))
            d['sample_ids'] = d['groups'].copy()
            development.append(d)
    combine = lambda rows: {k: np.concatenate([d[k] for d in rows]) for k in rows[0]}
    train, queries, development = map(combine, [train, queries, development])
    assert len(set(train['groups'])) == 47 and len(development['y']) == 8
    for name, data in [('train', train), ('training_queries_private', queries), ('discovery_truth', development)]:
        np.savez_compressed(root / 'private_inputs' / (name + '.npz'), **data)
    np.savez_compressed(root / 'public_data/train.npz', **train)
    observations = {k: development[k] for k in OBSERVABLE}
    for parent in ['private_inputs', 'public_data']:
        np.savez_compressed(root / parent / 'development_observations.npz', **observations)
    shutil.copy2(ref / 'JOBS366.json', root / 'JOBS366.json')
    jobs = read(root / 'JOBS366.json')
    model_names = read(ref / 'PROTOCOL366.json')['models']
    predictions, reports = {}, {name: [] for name in model_names}
    for job_id, job in jobs.items():
        data = development if job_id == 'full' else {k: queries[k][np.isin(queries['groups'], job['heldout_cells'])] for k in queries}
        for name in model_names:
            if name == 'author_crnn':
                net = CRNN(ni=100, nc=4, no=64, nh=64)
                net.load_state_dict(torch.load(ref / 'models' / job_id / 'author_crnn.pt', weights_only=True)['state_dict'])
                p, _ = network_predictions(net, data['nn'])
            else:
                p = joblib.load(ref / 'models' / job_id / (name+'.joblib'))['model'].predict(data['x'])
                if name == 'lifetime_ridge10':
                    p -= data['age']
                p = np.maximum(p, 0)
            if job_id == 'full':
                predictions[name] = p
            else:
                reports[name].append(dict(fold=job_id, **score(data, p)))
    # Current discovery truth is never returned to Actor or used to choose an arm.
    np.savez_compressed(root / 'private_inputs/REFERENCE_PREDICTIONS.npz', **predictions)
    initial = {}
    for name in model_names:
        ident = 'p00' if name == 'author_crnn' else 'ref_'+name
        folder = root / 'reference' / ident
        (folder / 'source').mkdir(parents=True)
        (folder / 'full_fit/artifacts').mkdir(parents=True)
        code = (PROJECT / 'scripts/hust_reference_program.py').read_text().replace("MODEL_KIND = 'author_crnn'", 'MODEL_KIND = '+repr(name))
        (folder / 'source/program.py').write_text(code)
        suffix = '.pt' if name == 'author_crnn' else '.joblib'
        shutil.copy2(ref / 'models/full' / (name+suffix), folder / 'full_fit/artifacts' / (name+suffix))
        report = dict(status='complete', program_id=ident, model=name, cv=reports[name],
            fitting_cells=sorted(set(train['groups'].tolist())), initial_models_reused=True,
            fit_entry_executions=0, source_reference_training='A366:fourrealfitentries,notrecounted',
            identity_invariance_passed=False, final_data_read=False,
            artifacts={str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()})
        save(folder / 'REPORT.json', report)
        initial[ident] = str(folder)
    metadata = dict(task='Rolling RUL at partial charge of cycle500 for measured LFP/graphite cells, 30C;one original source research adaptation.',
        independent_units='47historicaltrainingcells;8discoverycells;officialfinal22notmounted.',
        target='Author-providedremainingcyclesincludingcurrentunfinishedcycle;notindependentlyverifiedexact880mAhphysicalcrossing.',
        inputs=dict(x='173legalcurve/pastcapacity/age/ratefeatures', nn='4x100x10normalizedpartialchargecurves,last30cycles,sparse10',
            past_cap='9alreadycompleteddischargecapacities,mAh;allbeforecurrentquery', age='completedcycles', rates='3documenteddischargerates',
            train_targets='yRULcycles;cap10capacitiesallowedforhistoricaltrainonly'),
        forbidden='NoqueryRUL,currentunfinisheddischargecapacity,laterqueries,sourcearchive/networkorotherAgentaccess.',
        program_interface='program.py fit --train PATH --model-dir DIR;predict --input ONEQUERY.npz --model-dir DIR --output JSON:{rul:[nonnegativefloat]}.',
        allowed_online='Predict may learn from current partialcharge and the nine PAST capacities,<=100optimizersteps/query;originaltrainedartifact read-only. No current/futureRUL label exists.',
        metric='MeanMAEperphysicaldiscoverycell,cycles;all8retained.Wholetraining-cellCVat500given;ordinaryAgentcanselectanyinitialmodelorwritearbitrarycompleteprogram.',
        evaluation='AllchoicessealedbeforehiddendiscoveryRULscorejoin;8developmentcellsareexploration,notconfirmation. No current discovery score in toolfeedback.')
    save(root / 'public_data/metadata.json', metadata)
    engine = dict(input_root=str(root / 'private_inputs'), common_tools=str(root / 'common_tools'),
        reference_jobs=str(root / 'JOBS366.json'), fit_seconds=240, predict_seconds=45, program_seconds=900,
        prediction_workers=6, allow_past_capacity_online_fit=True, online_optimizer_steps_max=100)
    save(root / 'ENGINE_CONFIG.json', engine)
    rows = []
    for repeat in range(1, 3):
        run = root / 'subjects' / ('ordinary_'+str(repeat))
        for d in ['app/.runtime/tmp', 'app/.runtime/mpl', 'runtime/tmp', 'runtime/mpl', 'logs', 'plans', 'audit', 'broker', 'primary', 'feedback', 'ipc', 'scripts']:
            (run / d).mkdir(parents=True, exist_ok=False)
        (run / 'app/AGENTS.md').write_text('Useonlylegaltrainandpastqueryinformation;newcandidatefitsviaregister_program;nosourcearchive/network/otherAgents;never/tmp.\n')
        shutil.copy2(root / 'ENGINE_CONFIG.json', run / 'ENGINE_CONFIG.json')
        for name in ['model_catalog187.json']:
            shutil.copy2(root / name, run / name)
        (run / 'cli_config187.toml').write_text((root / 'base_cli_config.toml').read_text() + (root / 'MCP_CONFIG.txt').read_text())
        shutil.copy2(root / 'common_tools/isolated_model_gateway.py', run / 'scripts/isolated_model_gateway.py')
        save(run / 'mounts.json', dict(mounts=[dict(source=str(p), destination=q) for p, q in [
            (PYTHON.parent.parent, '/opt/python312'), (DEPS, '/opt/phm_python/extra'),
            (root / 'common_tools', '/tools'), (root / 'public_data', '/data'),
            (root / 'reference', '/reference'), (run / 'feedback', '/feedback')]]))
        save(run / 'BROKER_CONFIG.json', dict(initial_programs=initial, contract=str(root / 'public_data/metadata.json'),
            max_programs=2, request_directory='app/.runtime/feedback_requests'))
        save(run / 'PROTOCOL187.json', dict(cli=config['cli'], expected_mcp_tools=['mcp__phmforge__'+n for n in ['info', 'register_program', 'submit']]))
        save(run / 'MODEL_GATEWAY_CONFIG.json', dict(routes=[dict(job=run.name, socket=str(run / 'ipc/s.sock'))],
            base_url='https://api.deepseek.com/', model='deepseek-flash', max_requests=60, max_input_tokens=4000000,
            max_output_tokens=131072, max_output_per_request=32768, max_inflight=1,
            denied_tool_namespaces=['collaboration'], denied_tool_names=['spawn_agent', 'followup_task', 'send_message', 'list_agents'], buffer_verified_responses=True))
        shutil.copy2(root / 'PROMPT.txt', run / 'plans/prompt.txt')
        rows.append(dict(id=run.name, arm='ordinary', repeat=repeat, path=str(run)))
    save(root / 'SESSIONS337.json', rows)
    save(root / 'PREPARATION367.json', dict(passed=True, source_bindings=bindings, train_rows=len(train['y']),
        CV_at500_cells=47, discovery_queries=8, reference_predict_calls=28, new_fits=0, final22_files_opened=0))


def qualify(root):
    assert read(root / 'PREPARATION367.json')['passed']
    cfg = read(root / 'PROTOCOL337.json')
    assert sha(Path(cfg['cli'])) == cfg['cli_sha256']
    rows = []
    for row in read(root / 'SESSIONS337.json'):
        run = Path(row['path'])
        ns = arguments(root, run)
        fixture = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        fixture.bind(str(run / 'ipc/s.sock'))
        fixture.listen(1)
        folder = run / 'audit/isolation'
        folder.mkdir()
        check = """import os,socket,numpy as np,json
from pathlib import Path
c=dict(private_absent=not any(Path(p).exists() for p in ['/home/jupyter-yyf-zyq/.bashrc','/data/yyf-zyq/PHM_20261010','/home/jupyter-yyf-zyq/Bo_2/PHM_20261010']),secret_absent=not any('API_KEY'in k or 'SECRET'in k for k in os.environ),tmp_legal=os.environ['TMPDIR'].startswith('/app/.runtime/'),query_keys=set(np.load('/data/development_observations.npz').files)=={'x','nn','past_cap','age','rates','cycle','groups','sample_ids'},no_query_targets=not Path('/data/discovery_truth.npz').exists())
try:s=socket.create_connection(('1.1.1.1',443),timeout=1);s.close();c['network_blocked']=False
except OSError:c['network_blocked']=True
print(json.dumps(c));assert all(c.values())
"""
        env = environment(root)
        env.update(PATH='/opt/python312/bin:/usr/bin:/bin', PYTHONPATH='/opt/phm_python/extra:/tools',
            TMPDIR='/app/.runtime/tmp', TMP='/app/.runtime/tmp', TEMP='/app/.runtime/tmp')
        command = isolate_filesystem([str(ns.cli), 'sandbox', '-P', 'iot-offline', '-C', '/app', '--', '/opt/python312/bin/python3.12', '-B', '-c', check], ns, run / 'app', folder, None)
        try:
            p = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
        finally:
            fixture.close()
            (run / 'ipc/s.sock').unlink()
        save(folder / 'RESULT.json', dict(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr))
        assert p.returncode == 0, p.stderr[-1600:]
        result = probe(run, run, dict(id=run.name, mounts=str(run / 'mounts.json')), 'fixed_discovery', (run / 'cli_config187.toml').read_text())
        assert result['passed']
        worker = None
        try:
            with (run / 'logs/transport.log').open('w') as log:
                worker = subprocess.Popen([str(PYTHON), '-B', str(root / 'code/hust_program_tools.py'), 'serve', '--run', str(run)],
                    env=environment(root), stdout=log, stderr=subprocess.STDOUT)
            for _ in range(200):
                if (run / 'BROKER_READY.json').exists():
                    break
                assert worker.poll() is None
                time.sleep(.1)
            assert (run / 'BROKER_READY.json').exists()
            fixture = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            fixture.bind(str(run / 'ipc/s.sock'))
            fixture.listen(1)
            folder = run / 'audit/transport'
            folder.mkdir()
            check = "from hust_program_tools import client;import json;r=client({'action':'info'});assert r['ok'] and len(r['result']['programs'])==7;assert all('development' not in v for v in r['result']['programs'].values());print(json.dumps({'passed':True,'actual_model_calls':0}))"
            command = isolate_filesystem([str(ns.cli), 'sandbox', '-P', 'iot-offline', '-C', '/app', '--', '/opt/python312/bin/python3.12', '-B', '-c', check], ns, run / 'app', folder, None)
            p = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
            save(folder / 'RESULT.json', dict(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr))
            assert p.returncode == 0, p.stderr[-1600:]
        finally:
            if worker is not None:
                worker.terminate()
                worker.wait(timeout=10)
            fixture.close()
            (run / 'ipc/s.sock').unlink(missing_ok=True)
            (run / 'BROKER_READY.json').unlink(missing_ok=True)
        rows.append(dict(id=row['id'], isolation=True, native_MCP=True, real_file_transport=True, real_model_calls=0))
    save(root / 'NATIVE_QUALIFICATION367.json', dict(passed=True, rows=rows))


def preflight(root):
    from hust_program_tools import Broker
    cfg = read(root / 'PROTOCOL337.json')
    run = root / 'qualification_subject'
    for d in ['app', 'broker', 'feedback']:
        (run / d).mkdir(parents=True)
    shutil.copy2(root / 'ENGINE_CONFIG.json', run / 'ENGINE_CONFIG.json')
    original = read(root / 'subjects/ordinary_1/BROKER_CONFIG.json')
    save(run / 'BROKER_CONFIG.json', original)
    code = (PROJECT / 'scripts/hust_reference_program.py').read_text().replace("MODEL_KIND = 'author_crnn'", "MODEL_KIND = 'ridge10'")
    (run / 'app/program.py').write_text(code)
    broker = Broker(run)
    report = broker.dispatch(dict(action='register_program', files=['program.py'], claim='ExactRidge genuine registration qualification', rival='Infrastructure misreports fitted results'))
    save(root / 'PREFLIGHT_REGISTER_RESULT367.json', report)
    assert report['status'] == 'complete' and report['fit_entry_executions'] == 4
    assert broker.dispatch(dict(action='info'))['programs']['p01']['confirmed_estimator_fits'] == 4
    assert broker.dispatch(dict(action='submit', program_id='p01', reason='Infrastructure qualification only'))['program_id'] == 'p01'
    engine = Runtime(run)
    engine.deadline = time.monotonic()+600
    data = {k: v for k, v in np.load(root / 'private_inputs/development_observations.npz').items()}
    expected = np.load(root / 'private_inputs/REFERENCE_PREDICTIONS.npz')
    for name, path in original['initial_programs'].items():
        folder = Path(path)
        pred, _ = engine.predict(run / ('reference_'+name), folder / 'source', folder / 'full_fit/artifacts', data)
        model = read(folder / 'REPORT.json')['model']
        np.testing.assert_allclose(pred, expected[model], rtol=1e-6, atol=1e-4)
    save(root / 'REGISTRATION_PREFLIGHT367.json', dict(passed=True, actual_fit_entries=4, actual_learned_regressors=4,
        registration_cold_processes=63, reference_cold_processes=56, reference_programs=7,
        register_info_submit=True, query_RUL_not_evaluated=True, final22_read=False, model_calls=0))


def agents(root):
    assert read(root / 'NATIVE_QUALIFICATION367.json')['passed']
    assert read(root / 'REGISTRATION_PREFLIGHT367.json')['passed']
    assert not (root / 'HANDLE367.json').exists(), 'Never implicitly restart native episodes'
    for name, digest in read(root / 'EFFECTIVE_CODE_FREEZE337.json').items():
        assert sha(root / name) == digest
    save(root / 'HANDLE367.json', dict(pid=os.getpid(), utc=utc(), state='running'))
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(execute, root, r) for r in read(root / 'SESSIONS337.json')]):
            results.append(f.result())
            save(root / 'PROGRESS367.json', results)
    save(root / 'ALL_CHOICES_SEALED367.json', dict(utc=utc(), results=results, discovery_scores_joined=False))
    save(root / 'TERMINAL367.json', dict(utc=utc(), exit_code=0, sessions=len(results), final22_read=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'qualify', 'preflight', 'agents'])
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    globals()[a.action](a.run.resolve())
