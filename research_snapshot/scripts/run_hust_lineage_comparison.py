"""Same-budget single-context battery program generation with one parent fork."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from controlled_feedback import save, sha, utc
from native_validation_comparison import closure
from free_program_comparison import arguments, environment, execute
from replay_diagnostic_cli import isolate_filesystem
from qualify_native_cli_mcp import probe
from hust_lineage_tools import Broker, parent_hint

PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261025T104200Z_hust367')
PYTHON=Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS=Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')
read=lambda p:json.loads(Path(p).read_text())

COMMON='''Improve or retain the supplied complete rolling-RUL program p00 for measured LFP/graphite cells. Read /data/metadata.json, /reference/p00/source/program.py and phmforge info. p00 and ref_prior are two genuinely trained previous authored programs; seven other actually trained references are also available. Historical whole-cell training CV is supplied for all. Only partial-charge/current legal measurements and PAST capacities are available for the eight evaluation queries; their current RUL labels and unfinished discharge capacities are hidden. All47 historical training cells and their labels are legal. Official22 reserved cells are completely unavailable. The task is mean absolute RUL error per physical query cell, in cycles. The source label is author-provided remaining cycles including the current unfinished cycle.

You may freely write complete feature/estimator/prediction programs. Use up to FOUR register_program calls. Each call freezes your source, genuinely trains three whole-cell CV versions and a full training version, and cold-predicts single queries. Training and CV labels are confined to their actual fold. All new supervised candidate fits must go through this tool; do not fit unregistered models locally or create synthetic fault examples. Do not use full-training reference weights inside a CV fold: write a self-contained fit that learns only from its passed training file. No network, delegation, other models or hidden source archives. Workers expose /tools/author_net.py and supported sklearn/PyTorch; at most two numerical threads. Prediction artifacts are read-only; legitimate <=100-step adaptation to nine PAST capacities is allowed, never current/future RUL. Program interface: program.py fit --train NPZ --model-dir DIR; predict --input ONEQUERY_NPZ --model-dir DIR --output JSON with {rul:[nonnegative_number]}.

Both policies have the same lineage_hint tool. It reports the whole-cell CV best and a non-best parent within one paired standard error whose per-cell absolute-error profile differs most. This is a search suggestion with no deployment guarantee; it is not a final acceptance rule. Immutable earlier code is available under /feedback/archive/ID/source after info/lineage_hint. Both policies may inspect, modify, combine, or disregard this evidence. Preserve source versions and state the actual parent in the claim when registering. Low validation error, similarity, novelty and conservatism alone do not prove unseen performance.

Submit one complete registered program using phmforge submit; p00 and all references remain valid choices. Select using the available complete-program evidence, and report uncertainty and actual changes in report.md. Final selection is free. Budget2400seconds,60model requests,4Minput tokens,131072output tokens,32768output/request; four registration attempts, each <=900seconds,fit<=240seconds,predict<=45seconds. All calls count within the episode. Use /app/.runtime,never /tmp. If MCP discovery fails use python3 /tools/hust_lineage_tools.py client --request with the same JSON actions. An accepted tool submission is required, not a prose score.
'''
METHOD='''
Generation policy under test: reserve one continuation for a near-tied but behaviorally different parent. Generate your first two complete new programs normally. Immediately before the third registration, call lineage_hint on the current complete archive. Use its recommended_parent as the starting source for that ONE new version: copy that immutable source into a new working version, identify an actual training-CV error pattern to improve and a rival explanation, then implement a substantive refinement of that parent. Record parent=ID and the receipt UTC in your pre-fit claim and short lineage.md. If no non-best near-tied alternative exists, the tool recommends the best and you must explicitly report that the planned contrast did not activate. Never manufacture a worse program to trigger it. The fourth registration, if used, and final selection are unrestricted. If fewer than two new programs complete, report non-activation rather than pretending to test the policy. This adapts established stepping-stone search; effectiveness and novelty are not established.
'''


def prepare(root):
    assert not (root/'PREPARATION442.json').exists()
    cfg=read(root/'PROTOCOL442.json')
    assert sha(SOURCE/'SOURCE_MANIFEST.json')==cfg['source_manifest_sha256']
    index={e['path']:e for e in read(SOURCE/'SOURCE_MANIFEST.json')['files']}
    bindings=[]
    def bound(relative, destination):
        p=SOURCE/relative
        assert sha(p)==index[relative]['sha256']
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,destination)
        bindings.append(dict(source=str(p),sha256=sha(p),destination=str(destination.relative_to(root))))
    for folder in ['runtime/tmp','runtime/mpl','code','common_tools','reference','logs','audit']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    for name in ['base_cli_config.toml','model_catalog187.json','MCP_CONFIG.txt','retention_policy.json']:
        bound(name,root/name)
    for p in (SOURCE/'common_tools').glob('*.py'):
        bound(str(p.relative_to(SOURCE)),root/'common_tools'/p.name)
    for name in closure(['run_hust_lineage_comparison','hust_lineage_tools','hust_program_runtime','hust_program_tools','free_program_comparison','qualify_native_cli_mcp']):
        p=PROJECT/'scripts'/(name+'.py')
        shutil.copyfile(p,root/'code'/p.name);shutil.copyfile(p,root/'common_tools'/p.name)
    shutil.copyfile(SOURCE/'common_tools/author_net.py',root/'code/author_net.py')
    (root/'scripts').mkdir(exist_ok=True)
    shutil.copyfile(root/'code/isolated_model_gateway.py',root/'scripts/isolated_model_gateway.py')
    initial={}
    mapping={'p00':'subjects/ordinary_2/programs/p02','ref_prior':'subjects/ordinary_2/programs/p01'}
    for p in sorted((SOURCE/'reference').iterdir()):
        mapping['ref_author_crnn' if p.name=='p00' else p.name]=str(p.relative_to(SOURCE))
    for ident,relative in mapping.items():
        original=SOURCE/relative;dest=root/'reference'/ident
        for sub in ['source','full_fit/artifacts']:
            for p in (original/sub).rglob('*'):
                if p.is_file():bound(str(p.relative_to(SOURCE)),dest/p.relative_to(original))
        report=read(original/'REPORT.json')
        report.update(program_id=ident,initial_models_reused=True,reference_archive=str(SOURCE),
                      original_program_path=relative,original_report_sha256=sha(original/'REPORT.json'),
                      fit_entry_executions=0,reference_training_recounted=False,
                      artifacts={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()})
        save(dest/'REPORT.json',report);initial[ident]=str(dest)
    assert len(initial)==9
    engine=read(SOURCE/'ENGINE_CONFIG.json')
    engine.update(input_root=str(SOURCE/'private_inputs'),common_tools=str(root/'common_tools'),
                  reference_jobs=str(SOURCE/'JOBS366.json'),program_seconds=900,fit_seconds=240,
                  predict_seconds=45,prediction_workers=6)
    save(root/'ENGINE_CONFIG.json',engine)
    # Explicit hash references avoid recopying the large shared measurements.
    for relative in ['public_data/train.npz','public_data/development_observations.npz',
                     'private_inputs/train.npz','private_inputs/training_queries_private.npz',
                     'private_inputs/development_observations.npz','JOBS366.json']:
        assert sha(SOURCE/relative)==index[relative]['sha256']
        bindings.append(dict(source=str(SOURCE/relative),sha256=sha(SOURCE/relative),readonly_reference=True))
    common=COMMON.replace('Budget2400seconds,60model requests,4Minput tokens,131072output tokens,32768output/request',
        f"Budget{cfg['episode_seconds']}seconds,{cfg['episode_requests']}model requests,{cfg['episode_input_tokens']}input tokens,{cfg['episode_output_tokens']}output tokens,{cfg['output_per_request']}output/request")
    rows=[]
    for arm in cfg['arms']:
        for repeat in [1,2]:
            run=root/'subjects'/(arm+'_'+str(repeat))
            for d in ['app/.runtime/tmp','app/.runtime/mpl','runtime/tmp','runtime/mpl','logs','plans','audit','broker','primary','feedback','ipc','scripts']:
                (run/d).mkdir(parents=True)
            (run/'app/AGENTS.md').write_text('Only legal historical labels/past query information. Fit submitted candidates through register_program. No network/delegation/other models. Scratch /app/.runtime; never /tmp.\n')
            shutil.copyfile(root/'ENGINE_CONFIG.json',run/'ENGINE_CONFIG.json')
            shutil.copyfile(root/'model_catalog187.json',run/'model_catalog187.json')
            shutil.copyfile(root/'common_tools/isolated_model_gateway.py',run/'scripts/isolated_model_gateway.py')
            (run/'cli_config187.toml').write_text((root/'base_cli_config.toml').read_text()+(root/'MCP_CONFIG.txt').read_text().replace('hust_program_tools.py','hust_lineage_tools.py'))
            save(run/'mounts.json',dict(mounts=[dict(source=str(p),destination=q) for p,q in [
                (PYTHON,'/opt/python312'),(DEPS,'/opt/phm_python/extra'),(root/'common_tools','/tools'),
                (SOURCE/'public_data','/data'),(root/'reference','/reference'),(run/'feedback','/feedback')]]))
            save(run/'BROKER_CONFIG.json',dict(initial_programs=initial,contract=str(SOURCE/'public_data/metadata.json'),max_programs=4,request_directory='app/.runtime/feedback_requests'))
            save(run/'PROTOCOL187.json',dict(cli=cfg['cli'],expected_mcp_tools=['mcp__phmforge__'+n for n in ['info','lineage_hint','register_program','submit']]))
            save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=run.name,socket=str(run/'ipc/s.sock'))],base_url='https://api.deepseek.com/',model='deepseek-flash',max_requests=cfg['episode_requests'],max_input_tokens=cfg['episode_input_tokens'],max_output_tokens=cfg['episode_output_tokens'],max_output_per_request=cfg['output_per_request'],max_inflight=1,denied_tool_namespaces=['collaboration'],denied_tool_names=['spawn_agent','followup_task','send_message','list_agents'],buffer_verified_responses=True))
            (run/'plans/prompt.txt').write_text(common+(METHOD if arm=='lineage' else ''))
            rows.append(dict(id=run.name,arm=arm,repeat=repeat,path=str(run)))
    save(root/'SESSIONS337.json',rows)
    save(root/'PROTOCOL337.json',dict(cfg,broker_script='hust_lineage_tools.py',build_stage_only=True,stage='coding'))
    save(root/'PROMPTS442.json',dict(ordinary=common,lineage=common+METHOD))
    save(root/'PREPARATION442.json',dict(passed=True,bindings=bindings,new_fits=0,new_model_calls=0,
                initial_parent_hint=parent_hint({i:dict(report=read(Path(p)/'REPORT.json')) for i,p in initial.items()}),
                final22_opened=False,discovery_truth_read=False))
    save(root/'EFFECTIVE_CODE_FREEZE337.json',{str(p.relative_to(root)):sha(p) for folder in ['code','common_tools'] for p in (root/folder).glob('*.py')})


def qualify(root):
    assert read(root/'PREPARATION442.json')['passed']
    results=[]
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path']);ns=arguments(root,run)
        native=probe(run,run,dict(id=run.name,mounts=str(run/'mounts.json')),'fixed_discovery',(run/'cli_config187.toml').read_text())
        assert native['passed']
        fixture=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);fixture.bind(str(run/'ipc/s.sock'));fixture.listen(1)
        worker=None
        try:
            with (run/'logs/qualification_broker.log').open('w') as log:
                worker=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B',str(root/'code/hust_lineage_tools.py'),'serve','--run',str(run)],env=environment(root),stdout=log,stderr=subprocess.STDOUT)
            for _ in range(200):
                if (run/'BROKER_READY.json').exists():break
                assert worker.poll() is None;time.sleep(.1)
            assert (run/'BROKER_READY.json').exists()
            check="""import os,json,socket,numpy as np
from pathlib import Path
from hust_lineage_tools import client
assert not any(Path(p).exists() for p in ['/home/jupyter-yyf-zyq/.bashrc','/data/yyf-zyq/PHM_20261010','/data/discovery_truth.npz'])
assert not any('API_KEY' in k or 'SECRET' in k for k in os.environ)
assert os.environ['TMPDIR'].startswith('/app/.runtime/')
assert set(np.load('/data/development_observations.npz').files)=={'x','nn','past_cap','age','rates','cycle','groups','sample_ids'}
try: s=socket.create_connection(('1.1.1.1',443),timeout=1);s.close();network=False
except OSError: network=True
assert network
r=client({'action':'lineage_hint'});assert r['ok'];h=r['result'];assert h['cells']==47 and len(h['candidates'])==9
assert all(Path(p+'/program.py').is_file() for p in h['code_paths'].values())
print(json.dumps({'passed':True,'network_blocked':True,'cells':h['cells'],'parents':len(h['candidates']),'model_calls':0}))
"""
            folder=run/'audit/qualification';folder.mkdir()
            command=isolate_filesystem([str(ns.cli),'sandbox','-P','iot-offline','-C','/app','--','/opt/python312/bin/python3.12','-B','-c',check],ns,run/'app',folder,None)
            env=environment(root);env.update(PATH='/opt/python312/bin:/usr/bin:/bin',PYTHONPATH='/opt/phm_python/extra:/tools',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp')
            p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
            save(folder/'RESULT.json',dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,command=command))
            assert p.returncode==0,p.stderr[-1800:]
        finally:
            if worker is not None:worker.terminate();worker.wait(timeout=10)
            fixture.close();(run/'ipc/s.sock').unlink(missing_ok=True);(run/'BROKER_READY.json').unlink(missing_ok=True)
        results.append(dict(id=run.name,native_MCP=True,real_transport=True,private_labels_absent=True,new_fits=0,new_model_calls=0))
    save(root/'QUALIFICATION442.json',dict(passed=True,rows=results))


def agents(root):
    assert read(root/'QUALIFICATION442.json')['passed']
    assert not (root/'HANDLE442.json').exists(),'No implicit restart'
    for p,h in read(root/'EFFECTIVE_CODE_FREEZE337.json').items():assert sha(root/p)==h
    save(root/'HANDLE442.json',dict(pid=os.getpid(),utc=utc(),state='running'))
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(execute,root,row) for row in read(root/'SESSIONS337.json')]):
            results.append(future.result());save(root/'PROGRESS442.json',results)
    save(root/'ALL_CHOICES_SEALED442.json',dict(utc=utc(),results=results,external_scores_read=False))
    save(root/'TERMINAL442.json',dict(utc=utc(),status='completed',episodes=len(results)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','qualify','agents']);p.add_argument('--run',type=Path,required=True);a=p.parse_args();globals()[a.action](a.run)
