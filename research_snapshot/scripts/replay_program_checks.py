"""Replay authored checks on a common incumbent and sealed proposed programs.

This is terminal-artifact analysis, not exact intermediate-state replay or a
damage-accuracy oracle. No test or predictor is repaired by this runner.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
BASE = Path('/data/yyf-zyq/PHM_20261010')
SOURCES = [BASE/'20261025T200200Z_patch392', BASE/'20261025T202300Z_patch392r1']
PYTHON = Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def utc(): return datetime.now(timezone.utc).isoformat()
def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


HOOK = '''import atexit,json,os,sys
from collections import Counter
calls=Counter(); numeric={'numpy_lstsq':0}
def track(frame,event,arg):
    if event=='call' and frame.f_code.co_filename=='/app/solution.py':
        calls[frame.f_code.co_name]+=1
    return track
sys.setprofile(track)
try:
    import numpy as np
    original=np.linalg.lstsq
    def count(*a,**k):
        numeric['numpy_lstsq']+=1
        return original(*a,**k)
    np.linalg.lstsq=count
except ImportError:
    pass
def finish():
    sys.setprofile(None)
    with open('/app/.runtime/profile_'+str(os.getpid())+'.json','w') as f:
        json.dump({'solution_functions':dict(calls),'numerical_calls':numeric},f)
atexit.register(finish)
'''


def prepare(root):
    assert not root.exists()
    for directory in ['runtime/tmp','runtime/mpl','code','audit','hooks','inputs','reference','jobs']:
        (root/directory).mkdir(parents=True, exist_ok=True)
    bindings=[]
    def copy(source, relative, target):
        receipt=read(source/'ARCHIVE_VERIFIED.json')
        manifest=source/receipt['manifest_filename']
        assert sha(manifest)==receipt['manifest_sha256']
        index={r['path']:r for r in read(manifest)['files']}
        assert sha(source/relative)==index[relative]['sha256']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative,target)
        bindings.append(dict(archive=str(source),manifest_sha256=receipt['manifest_sha256'],member=relative,sha256=sha(target),target=str(target.relative_to(root))))
    for p in ['healthy_reference.npz','reference_geometry.json']:
        copy(SOURCES[0],'public/data/'+p,root/'inputs'/p)
    for p in (SOURCES[0]/'reference').glob('*.py'):
        copy(SOURCES[0],'reference/'+p.name,root/'reference'/p.name)
    subjects=[];checks=[];trace_claims=[]
    for source in SOURCES:
        for episode in read(source/'SESSIONS392.json'):
            if not episode['stages']: continue
            stage=episode['stages'][0]
            prefix='episodes/'+episode['id']+'/'+stage
            ident=('initial' if source==SOURCES[0] else 'recovery')+'_'+episode['id']
            app=source/prefix/'app'
            folder=root/'subjects'/ident/'app';folder.mkdir(parents=True)
            for p in app.rglob('*'):
                if p.is_file() and '.runtime' not in p.parts and 'data' not in p.relative_to(app).parts and p.suffix in ['.py','.md','.json']:
                    copy(source,str(p.relative_to(source)),folder/p.relative_to(app))
            events_path=source/prefix/'primary'/(episode['id']+'_'+stage)/'events.jsonl'
            events=[]
            for line in events_path.read_text().splitlines():
                d=json.loads(line)
                if d.get('type')=='item.completed':events.append(d.get('item',{}))
            # This rule is fixed before results: all persistent check_*.py; if
            # absent, the last recorded public-predict command with assertions.
            scripts=sorted(folder.rglob('check_*.py'))
            own=[]
            for script in scripts:
                text=script.read_text();tree=ast.parse(text)
                check=dict(id=f'check{len(checks):02d}',subject=ident,kind='persistent',path=str(script.relative_to(folder)),
                           sha256=sha(script),assertions=sum(isinstance(n,ast.Assert) for n in ast.walk(tree)),
                           historical_command_mentions=[e['id'] for e in events if e.get('type')=='command_execution' and script.name in e.get('command','')])
                checks.append(check);own.append(check['id'])
            if not scripts:
                candidates=[e for e in events if e.get('type')=='command_execution' and 'assert' in e.get('command','') and '.predict(' in e.get('command','')]
                if candidates:
                    e=candidates[-1]
                    check=dict(id=f'check{len(checks):02d}',subject=ident,kind='inline_shell',command=e['command'],event_id=e['id'],historical_exit_code=e.get('exit_code'),
                               command_sha256=hashlib.sha256(e['command'].encode()).hexdigest())
                    checks.append(check);own.append(check['id'])
            subjects.append(dict(id=ident,source=str(source),episode=episode['id'],stage=stage,path=str(folder),
                                 solution_sha256=sha(folder/'solution.py'),checks=own,
                                 limitation='Sealed initial-stage program; original instantaneous code state at every test event not reconstructed.'))
            for e in events:
                if e.get('type')=='agent_message':
                    trace_claims.append(dict(subject=ident,event_id=e['id'],text=e.get('text',''),source=str(events_path)))
    assert len(subjects)==6
    (root/'hooks/sitecustomize.py').write_text(HOOK)
    shutil.copy2(Path(__file__),root/'code'/Path(__file__).name)
    protocol=dict(id='A393',frozen_utc=utc(),mode='Exploratory terminal-test cross-state replay',
                  source_runs=list(map(str,SOURCES)),subjects=len(subjects),checks=len(checks),jobs=2*len(checks),
                  conditions=['incumbent','proposal'],concurrency=4,per_check_seconds=180,total_wall_seconds=1200,maximum_disk_GiB=3,
                  selection='All persistent check_*.py from six initial-stage final workspaces; if none, last completed command containing assert and .predict(; absent subjects retained.',
                  predictions=['If checks only establish software or healthy behavior, they will often pass on both states or discriminate a property unrelated to damage accuracy.',
                               'Private-helper or candidate-specific dependencies may make cross-state replay incomparable, not evidence that the incumbent is physically wrong.',
                               'Checks may identify legitimate contract improvement; preserve these as contrary evidence.'],
                  analysis='Report all jobs, assertion presence, failures and exact commands; do not equate exit0 with valid localization evidence. Both outcomes already known, no confirmatory statistics.',
                  exclusions='No changes to test assertions or candidate programs; no missing-helper repair. Missing authored check is missing evidence, not failure.',
                  new_LLM_budget=0,new_supervised_fit_budget=0,new_hidden_query_budget=0,measured_inputs_only=True,
                  cache_reuse=False,independent_scenarios=0,majority_goal_complete=False,script_sha256=sha(root/'code'/Path(__file__).name),
                  hook_sha256=sha(root/'hooks/sitecustomize.py'))
    save(root/'PROTOCOL393.json',protocol);save(root/'SUBJECTS393.json',subjects);save(root/'CHECKS393.json',checks)
    save(root/'SOURCE_BINDINGS393.json',bindings);save(root/'TRACE_CLAIMS393.json',trace_claims)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',shared_dependencies=dict(path=str(DEPS.parent),usage='Pinned read-only Python libraries; source archives bound by SHA.'),retain=['All checks, original programs, commands, outputs, profiles, failures, source bindings and protocol']))
    save(PROJECT/'results/trace393/PROTOCOL393.json',protocol)
    print(json.dumps(dict(subjects=len(subjects),checks=len(checks),planned_jobs=2*len(checks),no_check_subjects=[s['id'] for s in subjects if not s['checks']])))


def execute(root):
    cfg=read(root/'PROTOCOL393.json');assert sha(Path(__file__))==cfg['script_sha256']
    assert sha(root/'hooks/sitecustomize.py')==cfg['hook_sha256']
    assert not (root/'STARTED393.json').exists(),'No implicit retry'
    for binding in read(root/'SOURCE_BINDINGS393.json'):
        assert sha(root/binding['target'])==binding['sha256']
    start=time.monotonic();save(root/'STARTED393.json',dict(utc=utc(),pid=os.getpid()))
    subjects={s['id']:s for s in read(root/'SUBJECTS393.json')}
    checks=read(root/'CHECKS393.json');jobs=[(check,state) for check in checks for state in cfg['conditions']]
    def worker(job):
        check,state=job;target=root/'jobs'/check['id']/state;app=target/'app'
        shutil.copytree(subjects[check['subject']]['path'],app)
        for directory in ['.runtime/tmp','.runtime/mpl']:(app/directory).mkdir(parents=True,exist_ok=True)
        if state=='incumbent':shutil.copy2(root/'reference/incumbent.py',app/'solution.py')
        command=['bwrap','--unshare-all','--die-with-parent','--new-session']
        for p in ['/usr','/lib','/lib64','/etc']:
            if Path(p).exists():command+=['--ro-bind',p,p]
        command+=['--symlink','usr/bin','/bin','--ro-bind',str(PYTHON),'/opt/python312','--ro-bind',str(DEPS),'/opt/deps',
                  '--ro-bind',str(root/'hooks'),'/hooks','--ro-bind',str(root/'reference'),'/reference',
                  '--bind',str(app),'/app','--ro-bind',str(root/'inputs'),'/app/data','--proc','/proc','--dev','/dev',
                  '--clearenv','--setenv','HOME','/app','--setenv','PATH','/opt/python312/bin:/usr/bin:/bin',
                  '--setenv','PYTHONPATH','/hooks:/opt/deps:/app','--setenv','PYTHONDONTWRITEBYTECODE','1',
                  '--setenv','TMPDIR','/app/.runtime/tmp','--setenv','TMP','/app/.runtime/tmp','--setenv','TEMP','/app/.runtime/tmp',
                  '--setenv','MPLCONFIGDIR','/app/.runtime/mpl','--setenv','OPENBLAS_NUM_THREADS','2','--setenv','OMP_NUM_THREADS','2',
                  '--setenv','MKL_NUM_THREADS','2','--chdir','/app']
        if check['kind']=='persistent':command+=['/opt/python312/bin/python3.12','-B','/app/'+check['path']]
        else:command+=['/bin/sh','-c',check['command']]
        save(target/'COMMAND.json',command)
        t=time.monotonic();error=None
        with (target/'stdout.log').open('w') as out,(target/'stderr.log').open('w') as err:
            p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
            try:code=p.wait(timeout=min(cfg['per_check_seconds'],max(1,cfg['total_wall_seconds']-(time.monotonic()-start))))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);code=p.wait();error='check_timeout'
        profiles=[read(p) for p in (app/'.runtime').glob('profile_*.json')]
        stderr=(target/'stderr.log').read_text(errors='replace')
        result=dict(check=check['id'],subject=check['subject'],state=state,exit_code=code,error=error,seconds=time.monotonic()-t,
                    source_sha256=sha(app/'solution.py'),stdout_sha256=sha(target/'stdout.log'),stderr_sha256=sha(target/'stderr.log'),
                    profiles=profiles,exception_tail=stderr[-1600:])
        expected=sha(root/'reference/incumbent.py') if state=='incumbent' else subjects[check['subject']]['solution_sha256']
        assert result['source_sha256']==expected,'A test modified the program'
        save(target/'RESULT.json',result);return result
    results=[]
    with ThreadPoolExecutor(max_workers=cfg['concurrency']) as pool:
        for future in as_completed([pool.submit(worker,j) for j in jobs]):
            row=future.result();results.append(row);save(root/'RESULTS393.json',results)
            save(PROJECT/'results/trace393/current_run.json',dict(run=str(root),state='running',pid=os.getpid(),completed=len(results),planned=len(jobs),utc=utc()))
            print(json.dumps(dict(completed=len(results),check=row['check'],state=row['state'],exit_code=row['exit_code'])),flush=True)
    assert len(results)==cfg['jobs']
    pairs=[]
    for check in checks:
        rows={r['state']:r for r in results if r['check']==check['id']}
        pairs.append(dict(check=check['id'],subject=check['subject'],kind=check['kind'],path=check.get('path'),
                          assertions=check.get('assertions'),incumbent_exit=rows['incumbent']['exit_code'],proposal_exit=rows['proposal']['exit_code'],
                          requires_manual_scope_and_comparability_review=True))
    summary=dict(utc=utc(),seconds=time.monotonic()-start,jobs=len(results),pairs=pairs,actual_predict_calls=sum(p.get('solution_functions',{}).get('predict',0) for r in results for p in r['profiles']),
                 numerical_lstsq_calls=sum(p.get('numerical_calls',{}).get('numpy_lstsq',0) for r in results for p in r['profiles']),
                 new_LLM_calls=0,new_supervised_fits=0,new_hidden_query_calls=0,independent_confirmation_scenarios=0,
                 scientific_state='terminal',goal_complete=False)
    save(root/'SUMMARY393.json',summary);save(PROJECT/'results/trace393/RESULT393.json',summary)
    save(root/'TERMINAL393.json',dict(utc=utc(),exit_code=0,jobs=len(results)))
    save(PROJECT/'results/trace393/current_run.json',dict(run=str(root),state='terminal_manual_audit_pending',pid=os.getpid(),completed=len(results),planned=len(jobs),utc=utc()))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    if args.action=='prepare':prepare(args.run)
    else:execute(args.run)
