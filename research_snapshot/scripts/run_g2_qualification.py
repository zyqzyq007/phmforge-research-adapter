"""Wait for the verified source archive,then export and really fit the basis.

Never restarts acquisition,opens final scores,or calls an LLM. The parent goal
owns publication of the source archive; this producer archives only its own run.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from controlled_feedback import save,sha,utc
from prepare_forge_coding_baseline import RUNTIME
from run_full_benchmark_campaign import clean_env


def fit_counts(root):
    started=completed=0
    for path in root.glob('basis/*/fit_events.jsonl'):
        raw=path.read_text();lines=raw.splitlines() if raw.endswith('\n') else raw.splitlines()[:-1]
        events=[json.loads(x) for x in lines]
        started+=sum(x['status']=='started' for x in events);completed+=sum(x['status']=='completed' for x in events)
    return dict(real_fit_calls_started=started,real_fit_calls_completed=completed)


def update(root,cfg,phase,**extra):
    previous=root/'PIPELINE_STATUS295.json';value=json.loads(previous.read_text()) if previous.exists() else {}
    value.update(iteration='A295',run_id=root.name,run=str(root),producer_pid=os.getpid(),updated_utc=utc(),phase=phase,
        goal_status='active',new_LLM_calls=0,new_Agent_cases=0,new_final_scores=0,**fit_counts(root),**extra)
    save(previous,value);save(Path(cfg['project'])/cfg['result_directory']/'current_run.json',value);return value


def terminate_owned_fits(root):
    for path in root.glob('basis/*/status.json'):
        state=json.loads(path.read_text())
        if state.get('status')!='running':continue
        pid=state.get('pid');proc=Path('/proc')/str(pid)/'cmdline'
        if proc.exists():
            command=proc.read_bytes().replace(b'\x00',b' ').decode()
            assert str(root) in command and 'controlled_feedback.py' in command and 'fit' in command
            assert os.getpgid(pid)==pid;os.killpg(pid,signal.SIGTERM)
            for _ in range(20):
                if not proc.exists() or not proc.read_bytes():break
                time.sleep(.1)
            if proc.exists() and proc.read_bytes():os.killpg(pid,signal.SIGKILL)
        save(root/'audit'/('worker_before_termination_'+path.parent.name+'.json'),state)
        state.update(status='externally_observed_terminated',ended_utc=utc());save(path,state)


def stage(root,cfg,name,arguments):
    for row in json.loads((root/'CODE_FREEZE295.json').read_text())['files']:assert sha(root/row['path'])==row['sha256']
    command=[sys.executable,'-B',str(Path(__file__).with_name(arguments[0])),*arguments[1:]]
    save(root/'audit'/(name+'_command.json'),command)
    env=clean_env(root);env.update(PYTHONPATH=str(root/'code')+':'+str(RUNTIME/'deps'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (root/'logs'/(name+'.log')).open('wb') as log:
        process=subprocess.Popen(command,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'audit'/(name+'_process.json'),dict(pid=process.pid,started_utc=utc(),maximum_seconds=cfg['stage_timeout_seconds'][name]))
        deadline=time.monotonic()+cfg['stage_timeout_seconds'][name]
        try:
            while process.poll() is None:
                update(root,cfg,name,status='executing',stage_pid=process.pid)
                if time.monotonic()>deadline:raise TimeoutError('Frozen '+name+' deadline reached')
                time.sleep(5)
            assert process.returncode==0,name+' failed;no automatic retry'
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            terminate_owned_fits(root);raise
    save(root/'audit'/(name+'_terminal.json'),dict(pid=process.pid,exit_code=process.returncode,ended_utc=utc()))


def archive_own_run(root,cfg,final):
    project=Path(cfg['project']);destination=Path('/data/yyf-zyq/PHM_20261010')/root.name
    save(root/'OWNERSHIP.json',dict(owner='run_g2_qualification',writers_stopped=True,status='terminal_archiving'))
    logpath=project/cfg['result_directory']/'archive.log'
    command=[sys.executable,'-B',str(Path(__file__).with_name('archive_run_locked.py')),'--workspace',str(project),'--run',str(root),'--destination',str(destination)]
    with logpath.open('wb') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=cfg['archive_timeout_seconds'])
    assert result.returncode==0,'Own-run archive failed;preserve source and inspect log'
    receipt=json.loads((destination/'ARCHIVE_VERIFIED.json').read_text())
    assert receipt['status']=='verified' and sha(destination/receipt['manifest_filename'])==receipt['manifest_sha256'] and not root.exists()
    with (project/'results/.archive_index.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);path=project/'results/archive_index.json';index=json.loads(path.read_text())
        assert not any(x['run_id']==root.name for x in index['archives']);index['archives'].append(receipt);index['updated_utc']=utc();save(path,index)
    save(project/cfg['result_directory']/'current_run.json',dict(final,status='completed_archived_verified' if final['status']=='completed' else 'failed_archived_verified',
        archive=str(destination),archive_receipt=receipt,producer_terminal=True,updated_utc=utc()))


def run(root):
    cfg=json.loads((root/'PROTOCOL295.json').read_text());source=Path(cfg['source_run']);archive=Path(cfg['source_archive'])
    assert not (root/'PRODUCER_STARTED295.json').exists(),'No implicit restart'
    save(root/'PRODUCER_STARTED295.json',dict(pid=os.getpid(),started_utc=utc(),new_LLM_calls=0))
    deadline=time.monotonic()+cfg['maximum_wait_seconds'];final=None
    try:
        while not (archive/'ARCHIVE_VERIFIED.json').exists():
            try:current=json.loads((source/'SOURCE_STATUS292.json').read_text())
            except FileNotFoundError:
                if (archive/'ARCHIVE_VERIFIED.json').exists():break
                if time.monotonic()>deadline:raise TimeoutError('Source/archive handoff wait budget reached')
                time.sleep(2);continue
            process=Path('/proc')/str(cfg['source_controller_pid'])/'cmdline'
            try:live=str(source) in process.read_bytes().replace(b'\x00',b' ').decode()
            except FileNotFoundError:live=False
            terminal=current['status']=='native_members_complete_waiting_condition_contract'
            if not live and not terminal:
                time.sleep(1)
                if (archive/'ARCHIVE_VERIFIED.json').exists():break
                try:current=json.loads((source/'SOURCE_STATUS292.json').read_text())
                except FileNotFoundError:continue
                terminal=current['status']=='native_members_complete_waiting_condition_contract'
                if not terminal:raise RuntimeError('Source controller terminal without all native members;no restart or fitting')
            if time.monotonic()>deadline:raise TimeoutError('Archive wait budget reached;not permission to restart acquisition')
            update(root,cfg,'waiting_for_verified_source_archive',status='waiting',source_controller_verified_live=bool(live),
                source_verified_bytes=current['verified_bytes'],source_expected_bytes=current['expected_compressed_bytes'],
                native_members_passed=sum(x.get('native_audit_passed') is True for x in current.get('outcomes',[])),
                next_parent_action='When all6native members and original controller are terminal,archive source292 with the canonical helper;this producer then starts automatically.')
            time.sleep(10)
        receipt=json.loads((archive/'ARCHIVE_VERIFIED.json').read_text())
        assert receipt['status']=='verified' and sha(archive/receipt['manifest_filename'])==receipt['manifest_sha256']
        update(root,cfg,'source_archive_verified',status='prepared',source_controller_verified_live=False,native_members_passed=6,source_archive=str(archive))
        plan=archive/'audit/qualification_precommit/g2_ensemble_source_precommit.json'
        assert sha(plan)==cfg['source_plan_sha256']
        stage(root,cfg,'export',['prepare_planetary_probe.py','export-bound-source','--run',str(root),'--source-archive',str(archive),'--source-plan',str(plan)])
        stage(root,cfg,'basis',['prepare_probability_bank.py','--run',str(root)])
        qualification=json.loads((root/'BASIS_QUALIFICATION.json').read_text());assert qualification['passed']
        bank=np.load(root/qualification['bank'],allow_pickle=False);declared=json.loads(plan.read_text())['probe_identifiability_before_Agent']
        rows=[json.loads((root/m['directory']/'fit_report.json').read_text()) for m in qualification['models']]
        threshold=declared['at_least_one_basis_joint_CV_development_macro_F1_greater_than']
        checks=dict(has_joint_CV_development_signal=any(x['cv']['macro_f1']>threshold and x['development']['macro_f1']>threshold for x in rows),
                    development_basis_predictions_differ=not np.all(bank['development'].argmax(axis=2)==bank['development'][:,0,:].argmax(axis=1)[:,None]))
        result=dict(passed=all(checks.values()),checks=checks,threshold=threshold,utc=utc(),final_scores_opened=False,
            interpretation='Feasibility only;not an Agent result or overfitting mechanism. No automatic seed/recipe search after failure.')
        save(root/'PROBE_IDENTIFIABILITY.json',result)
        final=update(root,cfg,'qualified_waiting_Agent_protocol' if result['passed'] else 'completed_not_identifiable',status='completed',
            actual_cold_processes=qualification['actual_cold_processes'],ready_for_Agent_protocol=bool(result['passed']),final_scores_opened=False)
    except BaseException as error:
        terminate_owned_fits(root)
        final=update(root,cfg,'qualification_failed_no_automatic_retry',status='failed',error_type=type(error).__name__,error=str(error),final_scores_opened=False)
    archive_own_run(root,cfg,final);print(json.dumps({k:v for k,v in final.items() if k not in ['run']}))
    return 0 if final['status']=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();raise SystemExit(run(args.run.resolve()))
