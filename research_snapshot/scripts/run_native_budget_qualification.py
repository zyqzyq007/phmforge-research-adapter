"""A444: two fresh ordinary coding qualifications, without external scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import tarfile
from controlled_feedback import save, sha, utc
from run_hust_lineage_comparison import prepare, qualify
from free_program_comparison import execute

read=lambda p:json.loads(Path(p).read_text())
EXTRA='''\nExecution-qualification requirement: complete at least TWO genuinely trained new candidate programs through register_program, inspect their actual whole-cell validation, then submit one registered choice. You may retain any initial program if it is preferable. This tests whether a complete coding iteration can finish under the shared budget; it is not a new research-method policy. No external RUL outcome feedback is available. All failed registrations and resources count.\n'''


def setup(root):
    assert not (root/'PREPARATION444.json').exists()
    cfg=read(root/'PROTOCOL444.json');assert read(root/'PROTOCOL442.json')==cfg
    prepare(root)
    archive=Path(cfg['initial_parent_qualification_reused_from'])
    assert sha(archive/'SOURCE_MANIFEST.json')==cfg['initial_parent_qualification_manifest']
    index={e['path']:e for e in read(archive/'SOURCE_MANIFEST.json')['files']}
    bound=[]
    for p in (root/'reference').rglob('*'):
        if p.is_file():
            rel=str(p.relative_to(root));assert sha(p)==index[rel]['sha256'];bound.append(rel)
    for name in ['hust_program_runtime.py','program_revision_runtime.py','program_training_entry.py','author_net.py','hust_window_features.py']:
        rel='common_tools/'+name;assert sha(root/rel)==index[rel]['sha256'];bound.append(rel)
    with tarfile.open(archive/'RUN_CONTENTS.tar.gz','r|gz') as bundle:
        for member in bundle:
            if member.name=='PARENT_QUALIFICATION442.json':
                raw=bundle.extractfile(member).read();assert __import__('hashlib').sha256(raw).hexdigest()==index[member.name]['sha256']
                (root/'REUSED_PARENT_QUALIFICATION444.json').write_bytes(raw);break
        else:raise RuntimeError('Missing genuine parent qualification')
    assert read(root/'REUSED_PARENT_QUALIFICATION444.json')['passed']
    prompts={}
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path']);p=run/'plans/prompt.txt';prompt=p.read_text()+EXTRA;p.write_text(prompt)
        gateway=read(run/'MODEL_GATEWAY_CONFIG.json')
        assert gateway['max_input_tokens']==12000000 and '12000000input tokens' in prompt
        assert gateway['max_requests']==60 and gateway['model']=='deepseek-flash'
        prompts[row['id']]=prompt
    save(root/'PROMPTS444.json',prompts)
    save(root/'PREPARATION444.json',dict(passed=True,initial_reference_and_runtime_bindings=bound,
        reused_cold_parent_checks=72,new_prediction_processes=0,new_fits=0,new_LLM_calls=0,
        source_archive=str(archive),external_RUL_truth_read=False,final22_read=False))
    shutil.copyfile(Path(__file__),root/'code'/Path(__file__).name)
    freeze=read(root/'EFFECTIVE_CODE_FREEZE337.json');freeze['code/'+Path(__file__).name]=sha(Path(__file__))
    save(root/'EFFECTIVE_CODE_FREEZE337.json',freeze)
    qualify(root)
    assert len(read(root/'QUALIFICATION442.json')['rows'])==2
    paths={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and 'clients-cache' not in p.parts
           and (p.name in ['PROTOCOL444.json','PROTOCOL442.json','PROTOCOL337.json','ENGINE_CONFIG.json','BROKER_CONFIG.json',
                         'MODEL_GATEWAY_CONFIG.json','mounts.json','cli_config187.toml','prompt.txt','SESSIONS337.json','EFFECTIVE_CODE_FREEZE337.json']
                or p.parent in [root/'code',root/'scripts'])}
    save(root/'LAUNCH_FREEZE444.json',paths)
    print(json.dumps(dict(prepared=True,native_transports_qualified=2,reference_bindings=len(bound),new_LLM_calls=0)),flush=True)


def run(root):
    assert read(root/'PREPARATION444.json')['passed'] and read(root/'QUALIFICATION442.json')['passed']
    assert not (root/'HANDLE444.json').exists(),'No implicit restart'
    for p,digest in read(root/'LAUNCH_FREEZE444.json').items():assert sha(root/p)==digest
    save(root/'HANDLE444.json',dict(pid=os.getpid(),utc=utc(),state='running'))
    rows=[]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(execute,root,row) for row in read(root/'SESSIONS337.json')]):
                rows.append(future.result());save(root/'PROGRESS444.json',rows)
        save(root/'ALL_CHOICES_SEALED444.json',dict(utc=utc(),results=rows,external_RUL_truth_read=False))
        checked=[]
        for row in rows:
            subject=root/'subjects'/row['id'];state=read(subject/'broker/STATE.json')
            fresh={k:v for k,v in state['programs'].items() if not v['report'].get('initial_models_reused')}
            for item in state['programs'].values():
                for p,digest in item['report']['artifacts'].items():assert sha(Path(item['path'])/p)==digest
            selected=state.get('submission')
            if selected:assert selected['program_report_sha256']==sha(Path(state['programs'][selected['program_id']]['path'])/'REPORT.json')
            checked.append(dict(id=row['id'],normally_completed=row['completed'],complete_new_programs=len(fresh),
                registration_attempts=state['attempts'],selected=selected['program_id'] if selected else None,
                execution_passed=bool(row['completed'] and len(fresh)>=2 and selected)))
        result=dict(utc=utc(),rows=checked,execution_gate_passed=all(r['execution_passed'] for r in checked),
            qualification_admission='pending_command_and_cost_audit',external_RUL_truth_read=False,
            final22_read=False,method_comparison=False,goal_complete=False)
        save(root/'QUALIFICATION_RESULT444.json',result)
        save(root/'TERMINAL444.json',dict(utc=utc(),exit_code=0,episodes=2,external_RUL_truth_read=False))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        save(root/'TERMINAL444.json',dict(utc=utc(),exit_code=1,error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        value=read(root/'HANDLE444.json');value.update(state='terminal',terminal_utc=utc());save(root/'HANDLE444.json',value)


def audit(root):
    """Audit execution only; deliberately never open external RUL truth."""
    from audit_program_trial import python_review
    from evaluate_hust_lineage import cost
    import numpy as np
    assert read(root/'HANDLE444.json')['state']=='terminal'
    assert read(root/'TERMINAL444.json')['exit_code']==0
    assert not (root/'audit/EXECUTION_AUDIT444.json').exists()
    checks=0
    for relative,digest in read(root/'LAUNCH_FREEZE444.json').items():
        assert sha(root/relative)==digest;checks+=1
    sealed=read(root/'ALL_CHOICES_SEALED444.json')['results']
    assert len(sealed)==2
    rows=[];commands=[];fit_records=[]
    for row in read(root/'SESSIONS337.json'):
        subject=Path(row['path']);state=read(subject/'broker/STATE.json')
        build=read(subject/'BUILD_RESULT.json')
        assert build==next(x for x in sealed if x['id']==row['id']);checks+=1
        fresh=[]
        for ident,item in state['programs'].items():
            program=Path(item['path']);report=read(program/'REPORT.json')
            assert report==item['report'] and report['status']=='complete';checks+=1
            for relative,digest in report['artifacts'].items():
                assert sha(program/relative)==digest;checks+=1
            if report.get('initial_models_reused'):continue
            assert report['training_only'] and report['whole_cell_split'] and report['fit_entry_executions']==4
            for fold in report['cv']:
                with np.load(program/fold['fold']/'train.npz',allow_pickle=False) as data:
                    train_groups=set(map(str,data['groups']))
                assert not train_groups & {str(c['cell']) for c in fold['cells']};checks+=1
            fresh.append(dict(id=ident,source_sha256=sha(program/'source/program.py'),
                fit_entries=report['fit_entry_executions'],estimator_fits=report['confirmed_estimator_fits'],
                nonzero_optimizer_steps=report['nonzero_sampled_optimizer_steps']))
        submission=state.get('submission')
        if submission:
            assert submission['program_report_sha256']==sha(Path(state['programs'][submission['program_id']]['path'])/'REPORT.json');checks+=1
        receipts=[read(p) for p in (subject/'model_gateway'/subject.name).glob('*_receipt.json')]
        gateway=read(subject/'model_gateway'/subject.name/'status.json')
        assert gateway['input_tokens']==sum(v.get('usage',{}).get('input_tokens',0) for v in receipts)
        assert gateway['output_tokens']==sum(v.get('usage',{}).get('output_tokens',0) for v in receipts)
        assert gateway['inflight_count']==0;checks+=3
        assert all(v.get('usage') and v['returned_model_ids']==['deepseek-flash'] for v in receipts);checks+=len(receipts)
        for number,line in enumerate((subject/'primary'/subject.name/'events.jsonl').read_text().splitlines(),1):
            event=json.loads(line);item=event.get('item',{})
            if event.get('type')=='item.completed' and item.get('type')=='command_execution':
                commands.append(dict(episode=subject.name,line=number,command=item['command'],
                    exit_code=item.get('exit_code'),**python_review(item['command'])))
        rows.append(dict(id=subject.name,normal=build['completed'],new_programs=fresh,
            selected=submission['program_id'] if submission else None,requests=gateway['requests'],
            input_tokens=gateway['input_tokens'],output_tokens=gateway['output_tokens'],seconds=build['seconds']))
    for p in root.glob('subjects/*/programs/*/**/PROCESS.json'):
        process=read(p)
        if process.get('mode')!='fit':continue
        path=p.parent/'training_audit.json';value=read(path) if path.exists() else {}
        fit_records.append(dict(path=str(p.relative_to(root)),returncode=process.get('returncode'),
            audit_status=value.get('status'),completed_estimator_fits=value.get('completed_estimator_fits',0),
            nonzero_optimizer_steps=value.get('nonzero_sampled_optimizer_steps',0)))
    counted=cost(root)
    counted.update(qualification_prediction_processes=0,parent_qualification_reused=72)
    assert counted['fit_entries_with_process_record']==len(fit_records);checks+=1
    counted['cached_input_tokens']=sum(read(p).get('usage',{}).get('input_tokens_details',{}).get('cached_tokens',0)
        for p in root.glob('subjects/*/model_gateway/*/*_receipt.json'))
    flagged=[c for c in commands if c['possible_learning_calls'] or c['unresolved']]
    output=dict(utc=utc(),passed=True,checks=checks,rows=rows,cost=counted,fit_records=fit_records,
        command_count=len(commands),flagged_commands=flagged,manual_review_pending=True,
        external_RUL_truth_read=False,final22_read=False,new_LLM_calls=0,new_fits=0,new_prediction_processes=0,
        result_sha256=sha(root/'QUALIFICATION_RESULT444.json'),auditor_sha256=sha(Path(__file__)))
    save(root/'audit/COMMANDS444.json',commands)
    save(root/'audit/EXECUTION_AUDIT444.json',output)
    shutil.copyfile(Path(__file__),root/'audit'/Path(__file__).name)
    print(json.dumps(dict(passed=True,checks=checks,commands=len(commands),flagged=len(flagged),cost=counted)),flush=True)


def watch(root):
    """Bounded post-seal audit; never restart or modify a scientific episode."""
    import time
    handle=read(root/'HANDLE444.json')
    os.kill(handle['pid'],0)
    marker=root/'AUDIT_WATCH444.json'
    state=dict(pid=os.getpid(),controller_pid=handle['pid'],utc=utc(),state='running')
    save(marker,state)
    try:
        deadline=time.monotonic()+read(root/'PROTOCOL444.json')['episode_seconds']+180
        while time.monotonic()<deadline:
            if (root/'TERMINAL444.json').exists() and read(root/'HANDLE444.json')['state']=='terminal':
                audit(root)
                state.update(state='terminal',audit_completed=True,utc=utc())
                break
            os.kill(handle['pid'],0)
            time.sleep(15)
        else:
            state.update(state='terminal',audit_completed=False,reason='Observation deadline; no restart or scientific termination',utc=utc())
    except BaseException as error:
        state.update(state='terminal',audit_completed=False,error_type=type(error).__name__,error=str(error),utc=utc())
        raise
    finally:
        save(marker,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','run','audit','watch']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.run.resolve())
