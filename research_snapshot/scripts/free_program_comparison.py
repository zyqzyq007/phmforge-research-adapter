"""A337: native coding, real authored-program fits, and sealed paired checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace
import uuid

from controlled_feedback import save, sha, utc
from native_validation_comparison import closure
from program_revision_runtime import Runtime
from replay_diagnostic_cli import replay, isolate_filesystem
from run_full_benchmark_campaign import clean_env
from run_validation_transfer import start_gateway
from qualify_native_cli_mcp import probe

PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
PYTHON=Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS=Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')
OUT=PROJECT/'results/freeprogram337'

COMMON='''Improve or retain the supplied actually trained four-class PHM diagnosis program. You are a coding Agent: you may write any genuine feature extraction and supported supervised classifier program, not merely select a manual four-model library. Read /data/metadata.json, /reference/source/program.py and /reference/REPORT.json, then use the phmforge info tool. Legal measurements are /data/train.npz and development.npz; independent_check_observations.npz has observations only. These are measured laboratory recordings from one rotating-machine rig. There are only eight training acquisition groups, not 72 independent devices. Do not assume any visible validation view predicts the unknown final conditions.

The real tool freezes files from /app, fits TWO complete-condition leave-out models and ONE full training model using only training labels, cold-tests development and opaque-ID invariance, and returns complete actual errors. Up to FOUR authored versions may be registered, besides the common starting program p00. Write program.py and optional .py/.json helpers, then register_program(files,claim,rival). Both fit and predict entry points must match the starting program. A changed version must be registered again; editing a previously registered file does not change its immutable version. All supervised model fitting for your submitted candidates must go through register_program. Do not fit unregistered models locally, fabricate measurements, create fault examples, initialize without learning, or use constant/mock training. Supported audited sklearn classifiers and PyTorch optimizers are available; use reproducible seeds and at most two numerical threads. The prediction worker receives ONE acquisition at a time, no labels, with fitted artifacts read-only. Predictions must not depend on group/sample IDs or class counts of a batch. The shared /tools/phm_program_features.py is optional; arbitrary actual feature code is permitted.

The same optional probe tool is available to you: once, precommit all currently registered complete programs and obtain their true labeled results on one fresh measured validation condition. This is a legal independent check, NOT the final test, and no extrapolation guarantee is supplied. Only programs registered BEFORE that call are independently checked. Further code written afterwards does not inherit their evidence. Final observations, labels, conditions, counts, scores, other runs and other Agents are unavailable. Do not access the network, spawn agents or call another model. No change is guaranteed to help.

Minimize whole-acquisition classification error on unseen conditions. Retain useful improvements and avoid replacing effective code with worse code; investigate actual conflicting evidence. Choose one actually registered program, including retaining p00, using phmforge submit(program_id,reason). Save report.md describing actual changes, checks, selection and uncertainty. The accepted tool submission is the sealed deliverable. If MCP discovery is unavailable, use python3 /tools/program_revision_tools.py client --request with the same JSON action fields. Budget: 2400 seconds total, 50 model requests, 2500000 input tokens, 131072 output tokens total, 32768 output tokens/request; each registered version at most 600 seconds total, each fit 180 seconds and each per-record prediction 90 seconds. All tools and any subsequent controller check consume this same episode wall budget. Use /app/.runtime for scratch, never /tmp. Do not describe a prose score as a final result.
'''

METHOD='''\nAdditional method workflow: treat each proposed replacement as a falsifiable change transaction. State the concrete reason it should improve and a rival explanation (for example, exploiting one condition rather than preserving diagnosis). Prefer small, discriminating changes and keep earlier complete versions. Compare whole-program per-condition results, not merely the component you changed. Assemble your candidate versions BEFORE spending the single independent probe. Probe that frozen bank before submission, then select only a version actually covered by its result. Evidence is attached to the exact source and fitted artifacts, not inherited by a later edit. Explicitly retain negative evidence.

Your final method commit is deterministic among the precommitted successfully checked programs: for each version collect its two whole-condition CV record errors, development record error and independent-check record error; minimize the worst of these four, then their mean; on a complete tie retain p00, then earlier registration order. Do not adjust this rule or invent a threshold. This rule is a provisional design under test, not a theorem or a safety guarantee. An external controller will apply this same frozen rule to your actual bank; your coding and hypotheses must still aim to generate genuinely useful improvements within the same budget.
'''


def read(path):return json.loads(Path(path).read_text())


def environment(root):
    env=clean_env(root)
    env.update(PYTHONPATH=str(DEPS)+':'+str(root/'code'),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    return env


def episode_config(root, run):
    cfg=read(root/'PROTOCOL337.json')
    limits=run/'EPISODE_LIMITS.json'
    if limits.exists():
        for key,value in read(limits).items():
            assert key in ['episode_seconds','episode_requests','episode_input_tokens','episode_output_tokens']
            assert 0 < value <= cfg[key], 'Per-stage limits may only deduct prior stage cost'
            cfg[key]=value
    return cfg


def arguments(root,run):
    cfg=episode_config(root,run)
    return SimpleNamespace(cli=Path(cfg['cli']),model='deepseek-flash',output=run/'primary',
        plan=run/'plans/unused.json',timeout=cfg['episode_seconds'],analysis_tools=True,
        permission_profile='iot-offline',isolation_config=run/'cli_config187.toml',
        model_gateway_socket=run/'ipc/s.sock',model_catalog=run/'model_catalog187.json',
        runtime_mounts_file=run/'mounts.json',workspace_alias='/app',credential_env_file=None,
        credential_env_name='DEEPSEEK_API_KEY',snapshot_generated_source=True,cleanup_runtime_cache=True)


def status(root,**changes):
    p=root/'STATUS337.json';old=read(p) if p.exists() else {}
    old.update(utc=utc(),**changes);save(p,old);save(OUT/'current_run.json',dict(old,run=str(root)))


def prepare(root):
    assert not (root/'PREPARATION337.json').exists()
    cfg=read(root/'PROTOCOL337.json');assert read(root/'reference/p00/REPORT.json')['identity_invariance_passed']
    assert sha(Path(cfg['cli']))==cfg['cli_sha256']
    for name in closure(['free_program_comparison','program_revision_tools','program_revision_runtime','program_training_entry','qualify_native_cli_mcp']):
        shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    for name in ['isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'common_tools'/name)
    (root/'scripts').mkdir(exist_ok=True);shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',root/'scripts/isolated_model_gateway.py')
    (root/'subjects').mkdir(exist_ok=True)
    prompts=read(root/cfg['prompt_file']) if cfg.get('prompt_file') else dict(ordinary_coding=COMMON,witness_transaction=COMMON+METHOD)
    save(root/'PROMPT_FREEZE337.json',prompts)
    rows=[]
    for arm in cfg['arms']:
        for repeat in range(1,cfg['repeats']+1):
            run=root/'subjects'/(arm+'_'+str(repeat));run.mkdir()
            for d in ['app/.runtime/tmp','app/.runtime/mpl','app/artifacts','runtime/tmp','runtime/mpl','logs','plans','audit','broker','primary','feedback','ipc','scripts']:(run/d).mkdir(parents=True)
            (run/'app/AGENTS.md').write_text('Use only this task legal measurements. All supervised candidate fitting through register_program. Final information unavailable. Write scratch under /app/.runtime; never /tmp. No network or agent delegation.\n')
            text=(root/'base_cli_config.toml').read_text()+'''[mcp_servers.phmforge]
command = "/opt/python312/bin/python3.12"
args = ["-B", "/tools/program_revision_tools.py", "mcp"]
required = true
env = { PYTHONPATH = "/opt/phm_python/extra:/tools", PYTHONDONTWRITEBYTECODE = "1", TMPDIR = "/app/.runtime/tmp", TMP = "/app/.runtime/tmp", TEMP = "/app/.runtime/tmp", MPLCONFIGDIR = "/app/.runtime/mpl", OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "2", MKL_NUM_THREADS = "2" }
startup_timeout_sec = 60
tool_timeout_sec = 720
'''
            if cfg.get('enable_paired_evidence',False):
                text=text.replace('MKL_NUM_THREADS = "2" }','MKL_NUM_THREADS = "2", PHM_PAIRED_EVIDENCE = "1" }')
            (run/'cli_config187.toml').write_text(text)
            for name in ['model_catalog187.json','ENGINE_CONFIG.json']:shutil.copy2(root/name,run/name)
            shutil.copy2(root/'scripts/isolated_model_gateway.py',run/'scripts/isolated_model_gateway.py')
            mounts=[(PYTHON,'/opt/python312'),(DEPS,'/opt/phm_python/extra'),(root/'common_tools','/tools'),
                (root/'public_data','/data'),(root/'reference/p00','/reference'),(run/'feedback','/feedback')]
            save(run/'mounts.json',dict(mounts=[dict(source=str(p),destination=q) for p,q in mounts]))
            save(run/'BROKER_CONFIG.json',dict(baseline=str(root/'reference/p00'),contract=str(root/'public_data/metadata.json'),max_programs=cfg['max_registered_programs'],request_directory='app/.runtime/feedback_requests',defer_independent_check=cfg.get('defer_independent_check',False),
                enable_paired_evidence=cfg.get('enable_paired_evidence',False),probe_reusable=cfg.get('probe_reusable',False)))
            tool_names=['info','register_program','probe','submit']+(['paired_evidence'] if cfg.get('enable_paired_evidence',False) else [])
            save(run/'PROTOCOL187.json',dict(cli=cfg['cli'],expected_mcp_tools=['mcp__phmforge__'+k for k in tool_names]))
            save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=run.name,socket=str(run/'ipc/s.sock'))],
                base_url='https://api.deepseek.com/',model='deepseek-flash',max_requests=cfg['episode_requests'],
                max_input_tokens=cfg['episode_input_tokens'],max_output_tokens=cfg['episode_output_tokens'],
                max_output_per_request=cfg['output_per_request'],max_inflight=1,
                denied_tool_namespaces=['collaboration'],denied_tool_names=['spawn_agent','followup_task','interrupt_agent','list_agents','send_message','wait_agent'],buffer_verified_responses=True))
            (run/'plans/prompt.txt').write_text(read(root/'PROMPT_FREEZE337.json')[arm])
            rows.append(dict(id=run.name,arm=arm,repeat=repeat,path=str(run)))
    save(root/'SESSIONS337.json',rows)
    save(root/'EFFECTIVE_CODE_FREEZE337.json',{str(p.relative_to(root)):sha(p) for d in ['code','common_tools','scripts'] for p in (root/d).glob('*.py')})
    save(root/'PREPARATION337.json',dict(passed=True,utc=utc(),sessions=4,baseline_real_fits=3,baseline_identity_check_passed=True,new_scientific_LLM_calls=0,final_data_read=False))
    status(root,phase='prepared',controller_status='idle',planned_sessions=4,completed_sessions=0,final_scores_opened=False)


def preflight(root):
    """Exercise the real success response, not merely tool discovery/info."""
    assert not (root/'REGISTRATION_PREFLIGHT337.json').exists()
    run=root/'qualification_subject';run.mkdir()
    for d in ['app/.runtime/feedback_requests','feedback','broker','logs']:(run/d).mkdir(parents=True)
    shutil.copy2(root/'ENGINE_CONFIG.json',run/'ENGINE_CONFIG.json')
    files=[]
    for source in sorted((root/'reference/p00/source').rglob('*')):
        if source.is_file():
            relative=source.relative_to(root/'reference/p00/source');target=run/'app'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);files.append(str(relative))
    save(run/'BROKER_CONFIG.json',dict(baseline=str(root/'reference/p00'),contract=str(root/'public_data/metadata.json'),max_programs=1,request_directory='app/.runtime/feedback_requests'))
    worker=None
    def call(request):
        ident=uuid.uuid4().hex;save(run/'app/.runtime/feedback_requests'/(ident+'.json'),request)
        target=run/'feedback/responses'/(ident+'.json');deadline=time.monotonic()+650
        while time.monotonic()<deadline:
            if target.exists():return read(target)
            assert worker.poll() is None;time.sleep(.1)
        raise TimeoutError('Actual registration preflight timed out')
    try:
        with (run/'logs/broker.log').open('w') as log:
            worker=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B','-u',str(root/'code/program_revision_tools.py'),'serve','--run',str(run)],env=environment(root),stdout=log,stderr=subprocess.STDOUT)
        for _ in range(200):
            if (run/'BROKER_READY.json').exists():break
            assert worker.poll() is None;time.sleep(.1)
        registered=call(dict(action='register_program',files=files,claim='Exact starting program transport qualification.',rival='Host tool can misserialize a genuinely completed fit.'))
        assert registered['ok'] and registered['result']['status']=='complete',registered
        assert registered['result']['program_id']=='p01' and registered['result']['confirmed_estimator_fits']==3
        info=call(dict(action='info'));assert info['result']['programs']['p01']['fit_entry_executions']==3
        submitted=call(dict(action='submit',program_id='p01',reason='Qualification only;not an Actor or scientific selection.'))
        assert submitted['ok'] and submitted['result']['program_id']=='p01'
        save(root/'REGISTRATION_PREFLIGHT337.json',dict(passed=True,utc=utc(),actual_fits=3,actual_cold_predictions=16,register_then_info_then_submit=True,model_calls=0,independent_packet_opened=False,final_data_read=False))
    finally:
        if worker is not None:
            worker.terminate();worker.wait(timeout=10)


def qualify(root):
    assert read(root/'PREPARATION337.json')['passed'];assert not (root/'NATIVE_QUALIFICATION337.json').exists()
    rows=[]
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path']);ns=arguments(root,run)
        previous=[run/'audit/isolation/RESULT.json',run/'audit/native_mcp'/run.name/'fixed_discovery/RESULT.json',run/'audit/transport/RESULT.json']
        if all(p.exists() for p in previous) and read(previous[0])['exit_code']==0 and read(previous[1])['passed'] and read(previous[2])['exit_code']==0:
            rows.append(dict(id=row['id'],isolation=True,mcp_catalog=True,file_transport=True,model_calls=0,reused_completed_same_input_qualification=True))
            continue
        fixture=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);fixture.bind(str(run/'ipc/s.sock'));fixture.listen(1)
        folder=run/'audit/isolation';folder.mkdir()
        check="""import os,socket,json
from pathlib import Path
c=dict(private_paths_absent=not any(Path(p).exists() for p in ['/home/jupyter-yyf-zyq/.bashrc','/data/yyf-zyq/PHM_20261010','/srv/yyf-zyq/PHM_20261010/_shared','/srv/yyf-zyq/PHM_20261010/20261024T174000Z_freeprogram337/private_inputs','/home/jupyter-yyf-zyq/Bo_2/PHM_20261010']),keys_absent=not any('API_KEY' in k or 'SECRET' in k for k in os.environ),temporary_path=os.environ['TMPDIR'].startswith('/app/.runtime/'),public_allowlist={p.name for p in Path('/data').iterdir()}=={'train.npz','development.npz','independent_check_observations.npz','metadata.json'},final_absent=not Path('/data/final.npz').exists())
try:s=socket.create_connection(('1.1.1.1',443),timeout=1);s.close();c['network_blocked']=False
except OSError:c['network_blocked']=True
print(json.dumps(c));assert all(c.values())
"""
        check=check.replace('/srv/yyf-zyq/PHM_20261010/20261024T174000Z_freeprogram337/private_inputs',str(root/'private_inputs'))
        env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra:/tools',PATH='/opt/python312/bin:/usr/bin:/bin',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp')
        command=isolate_filesystem([str(ns.cli),'sandbox','-P','iot-offline','-C','/app','--','/opt/python312/bin/python3.12','-B','-c',check],ns,run/'app',folder,None)
        try:p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
        finally:fixture.close();(run/'ipc/s.sock').unlink()
        save(folder/'RESULT.json',dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,model_calls=0,command=command));assert p.returncode==0,p.stderr[-1600:]
        native=probe(run,run,dict(id=run.name,mounts=str(run/'mounts.json')),'fixed_discovery',(run/'cli_config187.toml').read_text());assert native['passed']
        worker=None
        try:
            with (run/'logs/transport.log').open('w') as log:
                worker=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B','-u',str(root/'code/program_revision_tools.py'),'serve','--run',str(run)],env=environment(root),stdout=log,stderr=subprocess.STDOUT)
            for _ in range(200):
                if (run/'BROKER_READY.json').exists():break
                assert worker.poll() is None;time.sleep(.1)
            assert (run/'BROKER_READY.json').exists()
            fixture=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);fixture.bind(str(run/'ipc/s.sock'));fixture.listen(1)
            folder=run/'audit/transport';folder.mkdir()
            check="from program_revision_tools import client;import json;r=client({'action':'info'});assert r['ok'] and r['result']['programs']['p00']['fit_entry_executions']==3;print(json.dumps({'passed':True,'actual_baseline':True,'model_calls':0}))"
            command=isolate_filesystem([str(ns.cli),'sandbox','-P','iot-offline','-C','/app','--','/opt/python312/bin/python3.12','-B','-c',check],ns,run/'app',folder,None)
            p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
            save(folder/'RESULT.json',dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,model_calls=0));assert p.returncode==0,p.stderr[-1600:]
        finally:
            if worker is not None:worker.terminate();worker.wait(timeout=10)
            fixture.close();(run/'ipc/s.sock').unlink(missing_ok=True);(run/'BROKER_READY.json').unlink(missing_ok=True)
        rows.append(dict(id=row['id'],isolation=True,mcp_catalog=True,file_transport=True,model_calls=0))
    save(root/'NATIVE_QUALIFICATION337.json',dict(passed=True,rows=rows,actual_model_calls=0,utc=utc()))
    status(root,phase='native_qualification_passed',controller_status='idle')


def choose(state,reports=None,restricted=None):
    options=[]
    for ident,v in state['programs'].items():
        if restricted is not None and ident not in restricted:continue
        r=v['report'];loss=[x['record_error'] for x in r['cv']]+[r['development']['record_error']]
        if reports is not None:
            if reports.get(ident,{}).get('status')!='complete':continue
            loss.append(reports[ident]['record_error'])
        options.append((max(loss),sum(loss)/len(loss),ident!='p00',ident,loss))
    assert options,'No valid checked version'
    best=min(options)
    return dict(program_id=best[3],worst_record_error=best[0],mean_condition_error=best[1],condition_losses=best[4])


def execute(root,row):
    run=Path(row['path']);cfg=episode_config(root,run);broker=None;gateway=None;started=time.monotonic()
    result=dict(id=row['id'],arm=row['arm'],repeat=row['repeat'],completed=False)
    try:
        with (run/'logs/broker.log').open('w') as log:
            broker=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B','-u',str(root/'code'/cfg.get('broker_script','program_revision_tools.py')),'serve','--run',str(run)],env=environment(root),stdout=log,stderr=subprocess.STDOUT)
        for _ in range(200):
            if (run/'BROKER_READY.json').exists():break
            assert broker.poll() is None;time.sleep(.1)
        assert (run/'BROKER_READY.json').exists()
        save(run/'EPISODE_BUDGET.json',dict(start_epoch=time.time(),deadline_epoch=time.time()+cfg['episode_seconds'],stage=cfg.get('stage','coding')))
        gateway=start_gateway(run)
        job=dict(id=row['id'],condition=row['arm'],repeat=row['repeat'],workspace=str(run/'app'),prompt_file=str(run/'plans/prompt.txt'),answer_mode='text')
        result.update(native=replay(job,arguments(root,run)))
    except Exception as error:result.update(error_type=type(error).__name__,error=str(error))
    finally:
        if broker is not None:
            broker.terminate()
            try:broker.wait(timeout=10)
            except subprocess.TimeoutExpired:broker.kill();broker.wait()
    state=read(run/'broker/STATE.json');submission=state.get('submission');result.update(submission=submission,actor_submission_valid=submission is not None)
    if cfg.get('build_stage_only',False):
        result.update(completed=result.get('native',{}).get('status')=='finished' and submission is not None,
            seconds=time.monotonic()-started,final_scores_joined=False,preliminary_handoff_only=True,
            registered_programs=sorted(state['programs']))
        if gateway is not None:
            (run/'MODEL_GATEWAY_STOP').touch()
            try:gateway.wait(timeout=30)
            except subprocess.TimeoutExpired:gateway.terminate();gateway.wait(timeout=10)
        save(run/'BUILD_RESULT.json',result);return result
    packet=state.get('probe');remaining=cfg['episode_seconds']-(time.monotonic()-started)
    if packet is None and remaining>5:
        # Completed Actor receives no further feedback. This is charged to its
        # remaining wall allowance and supplies a same-bank gate diagnostic.
        where=run/'controller_packet';where.mkdir()
        save(where/'PRECOMMIT.json',dict(utc=utc(),precommitted_program_ids=sorted(state['programs']),code_and_weights={i:v['report']['artifacts'] for i,v in state['programs'].items()},actor_finished=True))
        engine=Runtime(run);engine.config['program_seconds']=min(600,remaining)
        reports=engine.independent({i:Path(v['path']) for i,v in state['programs'].items()},where)
        packet=dict(status='complete',program_ids=sorted(state['programs']),reports=reports,controller_only=True)
    result['packet']=packet
    if packet and packet['status']=='complete':
        try:
            result['transaction']=choose(state,packet['reports'],packet['program_ids'])
            result['without_new_check']=choose(state,None,packet['program_ids'])
            result['completed']=submission is not None
        except AssertionError as error:result['transaction_error']=str(error)
    result['fixed_all_programs_no_check']=choose(state)
    for kind in ['development','mean_cv']:
        ids=list(state['programs']);key=lambda i:((state['programs'][i]['report']['development']['record_error'] if kind=='development' else sum(x['record_error'] for x in state['programs'][i]['report']['cv'])/2),i!='p00',i)
        result['fixed_'+kind]=dict(program_id=min(ids,key=key))
    result.update(seconds=time.monotonic()-started,remaining_seconds=cfg['episode_seconds']-(time.monotonic()-started),fallback=not result['completed'],final_scores_joined=False)
    if gateway is not None:
        (run/'MODEL_GATEWAY_STOP').touch()
        try:gateway.wait(timeout=30)
        except subprocess.TimeoutExpired:gateway.terminate();gateway.wait(timeout=10)
    save(run/'DECISIONS337.json',result);return result


def agents(root):
    assert read(root/'NATIVE_QUALIFICATION337.json')['passed'];assert not (root/'ALL_CHOICES_SEALED337.json').exists()
    if read(root/'PROTOCOL337.json').get('engineering_recovery_of'):assert read(root/'REGISTRATION_PREFLIGHT337.json')['passed']
    for name,digest in read(root/'EFFECTIVE_CODE_FREEZE337.json').items():assert sha(root/name)==digest
    rows=read(root/'SESSIONS337.json');results=[]
    save(root/'HANDLE337.json',dict(pid=os.getpid(),status='running',phase='native_coding',utc=utc()))
    status(root,phase='native_coding_running',controller_status='running',controller_pid=os.getpid(),completed_sessions=0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(execute,root,row) for row in rows]):
            results.append(f.result());save(root/'PROGRESS337.json',results)
            status(root,phase='native_coding_running',completed_sessions=len(results))
    save(root/'ALL_CHOICES_SEALED337.json',dict(utc=utc(),episodes=results,final_scores_joined=False,code_freeze_sha256=sha(root/'EFFECTIVE_CODE_FREEZE337.json'),protocol_sha256=sha(root/'PROTOCOL337.json')))
    save(root/'HANDLE337.json',dict(pid=os.getpid(),status='terminal',phase='choices_sealed',utc=utc()))
    status(root,phase='choices_sealed',controller_status='terminal',completed_sessions=4,valid_actor_submissions=sum(r['actor_submission_valid'] for r in results),final_scores_opened=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','qualify','agents']);p.add_argument('--run',type=Path,required=True);a=p.parse_args();globals()[a.action](a.run)
