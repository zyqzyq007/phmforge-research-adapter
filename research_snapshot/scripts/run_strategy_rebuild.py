"""Budgeted PHM adaptation: full-code review versus short-strategy reconstruction.

This is not a reproduction of What Fits' validation-information guarantee:
both downstream arms retain legal training-cell CV and may retain the incumbent.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import numpy as np

from controlled_feedback import save, sha, utc
from free_program_comparison import arguments, environment, execute
from mlestar_phm_adapter import post_socket
from qualify_native_cli_mcp import probe
from native_validation_comparison import closure
from replay_diagnostic_cli import isolate_filesystem
from run_hust_lineage_comparison import prepare as base_prepare, SOURCE, PYTHON
from run_hust_lineage_followup import extract_bound
from run_validation_transfer import start_gateway
from screen_score_blinding import output_text
from evaluate_hust_lineage import cost, prediction

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
read = lambda p: json.loads(Path(p).read_text())

COMMON = '''Review and, if justified, improve a genuinely trained rolling-RUL program p00 for measured LFP cells. Read /data/metadata.json and phmforge info. The task is mean absolute RUL error per physical cell in cycles. All47 training cells and their historical labels are legal. Eight query observation histories contain no current RUL or unfinished discharge capacity. Official22 reserved cells and external scores are unavailable. The author label counts remaining cycles including the unfinished current cycle.

You may write complete feature/estimator/prediction programs. At most FOUR register_program attempts: each freezes your code, really fits three whole-cell CV models and one full model, and cold-predicts each query separately. Fit only on the passed training file; do not reuse full-data weights inside a CV fold. All supervised candidate fitting must use register_program, not local unregistered training. No synthetic fault data, mock learning, network, delegation, other models or hidden archives. Supported sklearn/PyTorch and /tools/author_net.py are available, at most two numerical threads. Up to100-step adaptation to the nine PAST observed capacities is allowed, never current/future RUL. Interface: program.py fit --train NPZ --model-dir DIR; program.py predict --input ONEQUERY_NPZ --model-dir DIR --output JSON, output {"rul":[nonnegative_number]}.

The same tools info, register_program(files,claim,rival), submit(program_id,reason) expose actual whole-cell training CV. p00 may always be retained without accessing its source. Choose one complete registered program; report actual modifications, evidence and uncertainty in report.md. Low CV error alone does not certify unseen performance. The accepted tool submission is required, not a prose score. Budget2400seconds including preparatory strategy compression if used,60model requests,12000000input tokens,131072output tokens,32768output/request; four registrations, each900seconds,fit240seconds,predict45seconds. Actual remaining caps are in /app/budget.json. Use /app/.runtime, never /tmp. If MCP discovery fails: python3 /tools/hust_program_tools.py client --request with the same JSON action. New source versions must be registered again.
'''
ORDINARY = '''\nOrdinary review: the incumbent's complete source is under /reference/source. Inspect it and available training evidence, then freely retain, modify, or replace it as you judge best. Preserve working versions.\n'''
REBUILD = '''\nStrategy-reconstruction policy: /reference/strategy.md contains the compressed strategy of the same incumbent. Its original source and exploration transcript are unavailable. Implement that strategy afresh in your own program, preserving its key modeling intent; verify and register one genuinely trained reconstruction before considering further revisions. Describe a missing or ambiguous detail as such. You may then inspect the same training-CV evidence, revise within the common budget, or retain p00. Compression is not a performance guarantee. Record what was reconstructed in reconstruction.md.\n'''
COMPRESS = '''Read the supplied incumbent PHM program and its training-validation report. Distill its actual prediction and training strategy into a self-contained prompt for a fresh coding Agent, at most128 cl100k_base tokens. Preserve the most important model, features, hyperparameters, blending and past-only adaptation details. Include exact values where essential. Describe the strategy, do not copy code, raw weights, arrays, record IDs or paths. Do not suggest a new strategy or assume the incumbent is successful. No external scores are available. Output only the short strategy. Excess tokens will be deterministically truncated. This single-pass PHM adaptation does not reproduce the author's iterative compressor.\n'''


def setup(root):
    cfg=read(root/'PROTOCOL461.json')
    assert not (root/'PREPARATION461.json').exists()
    save(root/'PROTOCOL442.json',cfg)
    base_prepare(root)
    archive=Path(cfg['incumbent_archive'])
    assert sha(archive/'SOURCE_MANIFEST.json')==cfg['incumbent_manifest_sha256']
    old=root/'incumbent_source'
    bindings=extract_bound(archive,old,lambda p:p in [f'subjects/ordinary_{i}/broker/STATE.json' for i in [1,2]])
    selected={i:read(old/f'subjects/ordinary_{i}/broker/STATE.json')['submission']['program_id'] for i in [1,2]}
    prefixes=[f'subjects/ordinary_{i}/programs/{selected[i]}/' for i in [1,2]]
    bindings.update(extract_bound(archive,old,lambda p:any(p.startswith(s) for s in prefixes) and
        ('/source/' in p or '/full_fit/artifacts/' in p or p.endswith('/REPORT.json') or p.endswith('/development/PREDICTIONS.npz')
         or ('/development/' in p and p.endswith('/predictions.json')))))
    common_config=read(root/'PROTOCOL337.json');common_config.update(broker_script='hust_program_tools.py',build_stage_only=True)
    save(root/'PROTOCOL337.json',common_config)
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path']);i=row['repeat'];origin=old/prefixes[i-1];inc=root/'incumbents'/str(i)
        if not inc.exists():
            shutil.copytree(origin,inc)
            report=read(inc/'REPORT.json');report.update(program_id='p00',initial_models_reused=True,
                original_program_id=selected[i],reference_training_recounted=False)
            report['artifacts']={str(p.relative_to(inc)):sha(p) for p in inc.rglob('*') if p.is_file() and ('source' in p.parts or 'artifacts' in p.parts)}
            save(inc/'REPORT.json',report)
        broker=read(run/'BROKER_CONFIG.json');broker.update(initial_programs={'p00':str(inc)},max_programs=4)
        save(run/'BROKER_CONFIG.json',broker)
        (run/'cli_config187.toml').write_text((run/'cli_config187.toml').read_text().replace('hust_lineage_tools.py','hust_program_tools.py'))
        p=read(run/'PROTOCOL187.json');p['expected_mcp_tools']=['mcp__phmforge__'+x for x in ['info','register_program','submit']];save(run/'PROTOCOL187.json',p)
        view=root/'views'/row['id'];view.mkdir(parents=True)
        if row['arm']=='ordinary':shutil.copytree(inc/'source',view/'source')
        else:(view/'PENDING.txt').write_text('Only the frozen short strategy is added after compression. No incumbent source.\n')
        mounts=read(run/'mounts.json')
        for m in mounts['mounts']:
            if m['destination']=='/reference':m['source']=str(view)
        save(run/'mounts.json',mounts)
        prompt=COMMON+(ORDINARY if row['arm']=='ordinary' else REBUILD)
        (run/'plans/prompt.txt').write_text(prompt)
        save(run/'app/budget.json',{k:cfg[k] for k in ['episode_seconds','episode_requests','episode_input_tokens','episode_output_tokens']})
        if row['arm']=='rebuild':
            comp=root/'compressors'/row['id']
            for name in ['runtime/tmp','runtime/mpl','logs','ipc']:(comp/name).mkdir(parents=True)
            save(comp/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=row['id'],socket=str(comp/'ipc/s.sock'))],
                base_url='https://api.deepseek.com/',model='deepseek-flash',max_requests=1,max_input_tokens=100000,
                max_output_tokens=cfg['compression_max_output'],max_output_per_request=cfg['compression_max_output'],max_inflight=1,buffer_verified_responses=True))
            save(comp/'VISIBLE.json',dict(source={str(p.relative_to(inc/'source')):p.read_text() for p in (inc/'source').rglob('*') if p.is_file()},
                legal_training_report={k:v for k,v in read(inc/'REPORT.json').items() if k not in ['artifacts']},task=COMMON))
    save(root/'INCUMBENT_BINDINGS461.json',dict(files=bindings,selection=selected,
        rule='All two normally completed A444 ordinary submissions, not selected by external performance. Their external performance has previously been seen by researchers.'))
    save(root/'PROMPTS461.json',dict(common=COMMON,ordinary=ORDINARY,rebuild=REBUILD,compressor=COMPRESS))
    for module in closure(['run_strategy_rebuild']):
        shutil.copyfile(PROJECT/'scripts'/(module+'.py'),root/'code'/(module+'.py'))
    freeze={str(p.relative_to(root)):sha(p) for d in ['code','common_tools'] for p in (root/d).glob('*.py')}
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path'])
        for name in ['plans/prompt.txt','BROKER_CONFIG.json','ENGINE_CONFIG.json','mounts.json','cli_config187.toml']:
            freeze[str((run/name).relative_to(root))]=sha(run/name)
    for name in ['PROTOCOL461.json','PROTOCOL337.json','PROMPTS461.json','INCUMBENT_BINDINGS461.json']:
        freeze[name]=sha(root/name)
    save(root/'LAUNCH_FREEZE461.json',freeze)
    stages=len(read(root/'SESSIONS337.json'))
    save(root/'PREPARATION461.json',dict(passed=True,new_fits=0,new_model_calls=0,incumbents=selected,scenarios=1,related_programs=2,planned_native_stages=stages))
    print(json.dumps(dict(prepared=True,incumbents=selected,planned_native_stages=stages)),flush=True)


def qualify(root):
    out=[]
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path']);ns=arguments(root,run)
        catalog=run/'audit/native_mcp'/run.name/'fixed_discovery/RESULT.json'
        if catalog.exists():
            q=read(catalog)
            assert (catalog.parent/'config.toml').read_text()==(run/'cli_config187.toml').read_text()
        else:
            q=probe(run,run,dict(id=run.name,mounts=str(run/'mounts.json')),'fixed_discovery',(run/'cli_config187.toml').read_text())
        assert q['passed']
        fixture=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);fixture.bind(str(run/'ipc/s.sock'));fixture.listen(1)
        with (run/'logs/qualification_broker.log').open('w') as log:
            worker=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B',str(root/'code/hust_program_tools.py'),'serve','--run',str(run)],env=environment(root),stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(200):
                if (run/'BROKER_READY.json').exists():break
                assert worker.poll() is None;time.sleep(.1)
            code='''import os,json,socket,numpy as np
from pathlib import Path
from hust_program_tools import client
assert not any(Path(p).exists() for p in ['/home/jupyter-yyf-zyq/.bashrc','/home/jupyter-yyf-zyq/Bo_2/PHM_20261010/AGNETS.md','/data/yyf-zyq/PHM_20261010/20261025T104200Z_hust367/private_inputs','/data/discovery_truth.npz','PRIVATE_INCUMBENT_ROOT'])
assert not any('API_KEY' in k or 'SECRET' in k for k in os.environ)
assert os.environ['TMPDIR'].startswith('/app/.runtime/')
assert set(np.load('/data/development_observations.npz').files)=={'x','nn','past_cap','age','rates','cycle','groups','sample_ids'}
try:s=socket.create_connection(('1.1.1.1',443),timeout=1);s.close();blocked=False
except OSError:blocked=True
assert blocked
r=client({'action':'info'});assert r['ok'] and set(r['result']['programs'])=={'p00'}
assert r['result']['training_versions_remaining']==4
assert bool(list(Path('/reference').rglob('program.py')))==EXPECTED_SOURCE
print(json.dumps({'passed':True,'model_calls':0,'incumbents':1,'source_visible':EXPECTED_SOURCE}))
'''.replace('EXPECTED_SOURCE',str(row['arm']=='ordinary')).replace('PRIVATE_INCUMBENT_ROOT',str(root/'incumbents'))
            folder=run/'audit/qualification';folder.mkdir()
            command=isolate_filesystem([str(ns.cli),'sandbox','-P','iot-offline','-C','/app','--','/opt/python312/bin/python3.12','-B','-c',code],ns,run/'app',folder,None)
            env=environment(root);env.update(PATH='/opt/python312/bin:/usr/bin:/bin',PYTHONPATH='/opt/phm_python/extra:/tools',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp')
            p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
            save(folder/'RESULT.json',dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,command=command))
            assert p.returncode==0,p.stderr[-1500:]
            out.append(dict(id=row['id'],passed=True,actual_MCP=True,private_labels_absent=True,source_visibility_correct=True))
        finally:
            worker.terminate();worker.wait(timeout=10);fixture.close();(run/'ipc/s.sock').unlink(missing_ok=True);(run/'BROKER_READY.json').unlink(missing_ok=True)
    save(root/'QUALIFICATION461.json',dict(passed=True,rows=out,new_fits=0,new_model_calls=0))
    print(json.dumps(dict(qualified=True,sessions=len(out),model_calls=0)),flush=True)


def compress(root,row):
    run=Path(row['path']);comp=root/'compressors'/row['id'];cfg=read(root/'PROTOCOL461.json');started=time.monotonic();gateway=start_gateway(comp)
    try:
        response=post_socket(comp/'ipc/s.sock',dict(model='deepseek-flash',instructions=COMPRESS,
            input=json.dumps(read(comp/'VISIBLE.json')),tools=[],stream=False,max_output_tokens=cfg['compression_max_output']))
        save(comp/'RESPONSE.json',response);text=output_text(response).strip();assert text
        raw=comp/'raw_strategy.txt';raw.write_text(text)
        target=root/'views'/row['id']/'strategy.md'
        author=root/'repro/author_code/pipeline'
        snippet="from pathlib import Path;from compress_iterative import count_tokens,truncate_to_budget,strip_comments;import sys,json;t=strip_comments(Path(sys.argv[1]).read_text());v=truncate_to_budget(t,128);Path(sys.argv[2]).write_text(v);print(json.dumps({'tokens':count_tokens(v),'truncated':t!=v}))"
        env=dict(os.environ,PYTHONPATH=str(author),TIKTOKEN_CACHE_DIR=str(root/'runtime/tiktoken'),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root/'runtime/tmp'),TMP=str(root/'runtime/tmp'),TEMP=str(root/'runtime/tmp'))
        p=subprocess.run(['/usr/bin/python3','-B','-c',snippet,str(raw),str(target)],env=env,capture_output=True,text=True,timeout=30)
        assert p.returncode==0,p.stderr[-1200:];tokens=json.loads(p.stdout);assert tokens['tokens']<=128
    finally:
        (comp/'MODEL_GATEWAY_STOP').touch()
        try:gateway.wait(timeout=30)
        except subprocess.TimeoutExpired:gateway.terminate();gateway.wait(timeout=10)
    receipts=[read(p) for p in comp.glob('model_gateway/*/*_receipt.json')];assert len(receipts)==1
    usage=receipts[0]['usage'];prior=cfg.get('carried_failed_prefix',{}).get(str(row['repeat']),{})
    limits=dict(episode_seconds=max(1,int(cfg['episode_seconds']-(time.monotonic()-started)-prior.get('seconds',0))),
        episode_requests=cfg['episode_requests']-1-prior.get('requests',0),episode_input_tokens=cfg['episode_input_tokens']-usage['input_tokens']-prior.get('input_tokens',0),
        episode_output_tokens=cfg['episode_output_tokens']-usage['output_tokens']-prior.get('output_tokens',0))
    save(run/'EPISODE_LIMITS.json',limits);save(run/'app/budget.json',limits)
    g=read(run/'MODEL_GATEWAY_CONFIG.json');g.update(max_requests=limits['episode_requests'],max_input_tokens=limits['episode_input_tokens'],max_output_tokens=limits['episode_output_tokens']);save(run/'MODEL_GATEWAY_CONFIG.json',g)
    save(comp/'TERMINAL461.json',dict(passed=True,seconds=time.monotonic()-started,usage=usage,carried_failed_prefix=prior,**tokens,
        strategy_sha256=sha(target),source_code_forwarded=False,external_scores_read=False))


def run(root):
    assert read(root/'QUALIFICATION461.json')['passed'] and not (root/'HANDLE461.json').exists()
    for p,h in read(root/'LAUNCH_FREEZE461.json').items():assert sha(root/p)==h
    save(root/'HANDLE461.json',dict(pid=os.getpid(),utc=utc(),state='running'))
    results=[]
    def one(row):
        try:
            if row['arm']=='rebuild':compress(root,row)
            result=execute(root,row)
        except Exception as e:
            result=dict(id=row['id'],arm=row['arm'],repeat=row['repeat'],completed=False,error_type=type(e).__name__,error=str(e)[:1200])
        save(Path(row['path'])/'TERMINAL461.json',result)
        return result
    try:
        with ThreadPoolExecutor(max_workers=read(root/'PROTOCOL461.json')['max_parallel_agents']) as pool:
            for f in as_completed([pool.submit(one,row) for row in read(root/'SESSIONS337.json')]):
                r=f.result();results.append(r);save(root/'PROGRESS461.json',results)
                print(json.dumps(dict(id=r['id'],completed=r['completed'],ended=len(results))),flush=True)
        save(root/'ALL_CHOICES_SEALED461.json',dict(utc=utc(),results=results,external_scores_read=False))
    finally:
        save(root/'HANDLE461.json',dict(pid=os.getpid(),utc=utc(),state='terminal'))
    save(root/'TERMINAL461.json',dict(utc=utc(),science_terminal=True,stages=len(results),evaluation_pending=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','qualify','run']);p.add_argument('--run',type=Path,required=True);a=p.parse_args();globals()[a.action](a.run)
