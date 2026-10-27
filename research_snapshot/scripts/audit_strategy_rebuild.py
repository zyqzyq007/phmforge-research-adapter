"""Post-seal audit and paired scoring of bounded strategy reconstruction.

No label access in audit(); score() requires an explicit completed command and
source review. Two incumbents are related discovery cases, not independent tasks.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from audit_program_trial import python_review
from controlled_feedback import save, sha, utc
from evaluate_hust_lineage import prediction

read=lambda p:json.loads(Path(p).read_text())


def roots(root):
    return [Path(read(root/'PROTOCOL461.json')['ordinary_live_run']),root]


def receipt_audit(root):
    requests=[];receipts=[]
    for kind in ['subjects','compressors']:
        requests.extend(root.glob(kind+'/*/model_gateway/*/*_request.json'))
        for p in root.glob(kind+'/*/model_gateway/*/*_receipt.json'):
            r=read(p);prefix=str(r['sequence']).zfill(3)
            assert sha(p.parent/(prefix+'_request.json'))==r['request_sha256']
            if r.get('response_sha256'):assert sha(p.parent/(prefix+'_response.sse'))==r['response_sha256']
            assert r['model_requested']=='deepseek-flash'
            assert r.get('usage') and r['returned_model_ids']==['deepseek-flash']
            receipts.append(dict(path=str(p),sha256=sha(p),**r))
    processes=[dict(path=str(p),**read(p)) for p in root.glob('subjects/*/programs/*/**/PROCESS.json')]
    fits=[dict(path=str(p),**read(p)) for p in root.glob('subjects/*/programs/*/**/training_audit.json') if read(p).get('mode')=='fit']
    return dict(requests=len(requests),receipts=len(receipts),missing_receipts=len(requests)-len(receipts),
        input_tokens=sum(r['usage']['input_tokens'] for r in receipts),output_tokens=sum(r['usage']['output_tokens'] for r in receipts),
        cached_input_tokens=sum(r['usage'].get('input_tokens_details',{}).get('cached_tokens',0) for r in receipts),
        fit_entry_processes=sum(p.get('mode')=='fit' for p in processes),fit_entry_audits=len(fits),
        successful_fit_processes=sum(p.get('mode')=='fit' and p.get('returncode')==0 for p in processes),
        completed_estimator_fits=sum(r.get('completed_estimator_fits',0) for r in fits),
        optimizer_steps=sum(r.get('optimizer_steps',0) for r in fits),
        nonzero_optimizer_steps=sum(r.get('nonzero_sampled_optimizer_steps',0) for r in fits),
        cold_prediction_processes=sum(p.get('mode')=='predict' for p in processes),
        receipt_rows=receipts,fit_records=fits)


def audit(root):
    assert not (root/'audit/EXECUTION_AUDIT462.json').exists()
    binding=read(root/'PAIRED_CONTROL_BINDING462.json');ordinary,method=roots(root)
    assert sha(ordinary/'PROTOCOL461.json')==binding['ordinary_protocol_sha256']
    assert sha(ordinary/'LAUNCH_FREEZE461.json')==binding['ordinary_launch_freeze_sha256']
    commands=[];rows=[];checks=0;costs={}
    for trial in [ordinary,method]:
        assert read(trial/'HANDLE461.json')['state']=='terminal'
        seal=read(trial/'ALL_CHOICES_SEALED461.json');assert len(seal['results'])==len(read(trial/'SESSIONS337.json'))
        for relative,h in read(trial/'LAUNCH_FREEZE461.json').items():assert sha(trial/relative)==h;checks+=1
        costs[trial.name]=receipt_audit(trial)
        for row in read(trial/'SESSIONS337.json'):
            subject=Path(row['path']);terminal=read(subject/'TERMINAL461.json')
            assert terminal==next(r for r in seal['results'] if r['id']==row['id']);checks+=1
            events=subject/'primary'/subject.name/'events.jsonl'
            # The two failed A461 compressors never launched a coding stage.
            if trial==ordinary and row['arm']=='rebuild':
                assert not events.exists() and not terminal['completed'];checks+=1
                continue
            state=read(subject/'broker/STATE.json');fresh=[]
            for ident,item in state['programs'].items():
                program=Path(item['path']);report=read(program/'REPORT.json')
                assert report==item['report'] and report['status']=='complete';checks+=1
                for relative,h in report['artifacts'].items():assert sha(program/relative)==h;checks+=1
                if report.get('initial_models_reused'):continue
                assert report['training_only'] and report['whole_cell_split'] and report['fit_entry_executions']==4
                assert report['confirmed_estimator_fits']>0 or report['nonzero_sampled_optimizer_steps']>0
                for fold in report['cv']:
                    with np.load(program/fold['fold']/'train.npz',allow_pickle=False) as data:train_groups=set(map(str,data['groups']))
                    assert not train_groups & {str(c['cell']) for c in fold['cells']};checks+=1
                fresh.append(dict(id=ident,source_hash=sha(program/'source/program.py'),fit_entries=report['fit_entry_executions'],
                    actual_estimator_fits=report['confirmed_estimator_fits'],actual_optimizer_steps=report['actual_optimizer_steps']))
            submitted=state.get('submission')
            if submitted:
                assert sha(Path(state['programs'][submitted['program_id']]['path'])/'REPORT.json')==submitted['program_report_sha256'];checks+=1
            if events.exists():
                for number,line in enumerate(events.read_text().splitlines(),1):
                    event=json.loads(line);item=event.get('item',{})
                    if event.get('type')=='item.completed' and item.get('type')=='command_execution':
                        commands.append(dict(trial=trial.name,episode=row['id'],line=number,command=item['command'],
                            exit_code=item.get('exit_code'),**python_review(item['command'])))
            account=read(subject/'model_gateway'/subject.name/'status.json') if (subject/'model_gateway'/subject.name/'status.json').exists() else {}
            assert account.get('inflight_count',0)==0
            prefix=[]
            if row['arm']=='rebuild':
                prefix=[read(trial/'compressors'/row['id']/'TERMINAL461.json')]
                limits=read(subject/'EPISODE_LIMITS.json');original=read(trial/'PROTOCOL461.json');comp=prefix[0];prior=comp['carried_failed_prefix']
                assert limits['episode_requests']==original['episode_requests']-1-prior['requests']
                assert limits['episode_input_tokens']==original['episode_input_tokens']-comp['usage']['input_tokens']-prior['input_tokens']
                assert limits['episode_output_tokens']==original['episode_output_tokens']-comp['usage']['output_tokens']-prior['output_tokens']
                assert limits['episode_seconds']+comp['seconds']+prior['seconds']<=original['episode_seconds']+1.0
                assert sha(Path(prior['receipt_path']))==prior['receipt_sha256'];checks+=5
                assert sha(trial/'views'/row['id']/'strategy.md')==comp['strategy_sha256'];checks+=1
            rows.append(dict(id=row['id'],arm=row['arm'],repeat=row['repeat'],trial=str(trial),completed=terminal['completed'],
                registration_attempts=state['attempts'],new_programs=fresh,submission=submitted,account=account,prefix=prefix))
    assert len(rows)==4
    flagged=[c for c in commands if c['possible_learning_calls'] or c['unresolved']]
    save(root/'audit/COMMANDS462.json',commands)
    save(root/'audit/COST462.json',costs)
    save(root/'audit/EXECUTION_AUDIT462.json',dict(passed=True,utc=utc(),checks=checks,rows=rows,
        command_count=len(commands),flagged_commands=flagged,manual_review_pending=True,
        final_labels_read=False,independent_confirmation=0,auditor_sha256=sha(Path(__file__))))
    print(json.dumps(dict(passed=True,checks=checks,commands=len(commands),flagged=len(flagged))),flush=True)


def score(root):
    assert not (root/'RESULT462.json').exists()
    audited=read(root/'audit/EXECUTION_AUDIT462.json');assert audited['passed']
    reviewed=read(root/'audit/MANUAL_REVIEW462.json')
    assert reviewed['review_complete'] and reviewed['execution_audit_sha256']==sha(root/'audit/EXECUTION_AUDIT462.json')
    assert reviewed['commands_sha256']==sha(root/'audit/COMMANDS462.json')
    cfg=read(root/'PROTOCOL461.json');source=Path(cfg['source_archive'])
    assert sha(source/'SOURCE_MANIFEST.json')==cfg['source_manifest_sha256']
    index={r['path']:r for r in read(source/'SOURCE_MANIFEST.json')['files']}
    truth=source/'private_inputs/discovery_truth.npz';assert sha(truth)==index[str(truth.relative_to(source))]['sha256']
    with np.load(truth,allow_pickle=False) as data:actual=data['y'].copy();ids=data['sample_ids'].copy()
    assert actual.shape==(8,) and len(set(ids))==8
    rows=[]
    for row in audited['rows']:
        subject=Path(row['trial'])/'subjects'/row['id'];state=read(subject/'broker/STATE.json');bank={}
        for ident,item in state['programs'].items():
            program=Path(item['path']);values=prediction(program/'development',ids)
            errors=np.abs(values-actual);report=item['report']
            cv=np.array([c['MAE'] for fold in report['cv'] for c in fold['cells']]);assert cv.shape==(47,)
            bank[ident]=dict(predictions=values.tolist(),errors=errors.tolist(),MAE=float(errors.mean()),CV=float(cv.mean()))
        base=np.array(bank['p00']['errors']);selected=state['submission']['program_id'] if state.get('submission') else None
        fixed=min(bank,key=lambda i:(bank[i]['CV'],i!='p00',i))
        def metric(ident):
            error=np.array(bank[ident]['errors']);return dict(MAE=float(error.mean()),harm=float(np.maximum(error-base,0).mean()),
                gain=float(np.maximum(base-error,0).mean()),bank_selection_loss=float(error.mean()-min(v['MAE'] for v in bank.values())))
        rows.append(dict(id=row['id'],arm=row['arm'],case=row['repeat'],completed=row['completed'],selected=selected,
            compliant=reviewed['episodes'][row['id']]['compliant'],reconstruction_activated=reviewed['episodes'][row['id']].get('reconstruction_activated'),
            initial_MAE=bank['p00']['MAE'],selected_metrics=metric(selected) if selected else None,
            fixed_CV=dict(id=fixed,**metric(fixed)),bank=bank,
            fallback_p00_reported_separately=metric('p00') if selected is None else None))
    paired=[]
    for case in [1,2]:
        ordinary=next(r for r in rows if r['case']==case and r['arm']=='ordinary')
        method=next(r for r in rows if r['case']==case and r['arm']=='rebuild')
        np.testing.assert_allclose(ordinary['bank']['p00']['predictions'],method['bank']['p00']['predictions'],rtol=0,atol=0)
        union_best=min(v['MAE'] for r in [ordinary,method] for v in r['bank'].values())
        paired.append(dict(case=case,ordinary=ordinary['selected_metrics'],rebuild=method['selected_metrics'],
            union_best_MAE=union_best,shared_pool_is_posthoc_diagnostic_not_available_to_both_actors=True))
    means={arm:{k:float(np.mean([r['selected_metrics'][k] for r in rows if r['arm']==arm])) for k in ['MAE','harm','gain']}
        if all(r['selected_metrics'] for r in rows if r['arm']==arm) else None for arm in ['ordinary','rebuild']}
    eligible=all(r['completed'] and r['compliant'] and r['selected_metrics'] is not None for r in rows)
    activated=all(r['reconstruction_activated'] for r in rows if r['arm']=='rebuild')
    successful=next(r for r in rows if r['arm']=='rebuild' and r['case']==2)
    retained=bool(successful['selected_metrics'] and successful['selected_metrics']['MAE']<=successful['initial_MAE']+1e-9)
    numeric=bool(means['ordinary'] and means['rebuild'] and means['rebuild']['MAE']<means['ordinary']['MAE']
        and means['rebuild']['harm']<=means['ordinary']['harm'] and means['rebuild']['gain']>=means['ordinary']['gain'])
    result=dict(utc=utc(),rows=rows,paired=paired,means=means,complete_compliant_comparison=eligible,
        reconstruction_activated=activated,old_success_preserved=retained,numeric_gate=numeric,
        exploratory_primary_passed=bool(eligible and activated and retained and numeric),
        physical_source_clusters=1,related_incumbents=2,physical_query_cells=8,independent_confirmation=0,
        new_method_innovation=False,goal_complete=False,truth_sha256=sha(truth),sample_ids=ids.tolist(),truth=actual.tolist())
    save(root/'RESULT462.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','paired','truth','sample_ids']}),flush=True)


def watch(root):
    """Bounded live-process monitor. Audit after both terminate, never score."""
    marker=root/'audit/WATCH462.json'
    assert not marker.exists(), 'Do not start duplicate watchers'
    started=time.monotonic()
    save(marker,dict(pid=os.getpid(),utc=utc(),state='running',operation='wait_then_execution_audit',final_label_access=False))
    try:
        while time.monotonic()-started<3600:
            observations=[]
            for trial in roots(root):
                h=read(trial/'HANDLE461.json');pid=h['pid'];proc=Path('/proc')/str(pid)
                alive=proc.exists()
                if alive:
                    try:
                        raw=(proc/'cmdline').read_bytes().split(b'\0')
                        state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
                        alive=state!='Z'
                        if alive:assert str(trial).encode() in raw, 'PID does not refer to the experiment controller'
                    except FileNotFoundError:alive=False
                observations.append(dict(run=str(trial),pid=pid,live=alive,handle_state=h['state'],sealed=(trial/'ALL_CHOICES_SEALED461.json').exists()))
            save(root/'audit/WATCH_OBSERVATION462.json',dict(utc=utc(),observations=observations))
            if all(not r['live'] and r['handle_state']=='terminal' and r['sealed'] for r in observations):
                audit(root)
                save(marker,dict(pid=os.getpid(),utc=utc(),state='terminal',execution_audit_complete=True,manual_review_pending=True,final_label_access=False))
                return
            if any(not r['live'] and (r['handle_state']!='terminal' or not r['sealed']) for r in observations):
                raise RuntimeError('Controller missing without a complete seal; retained state, no restart')
            time.sleep(15)
        raise TimeoutError('Bounded audit wait expired; no scientific restart or inferred terminal')
    except BaseException as error:
        save(marker,dict(pid=os.getpid(),utc=utc(),state='terminal',execution_audit_complete=False,
            error_type=type(error).__name__,error=str(error),final_label_access=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','score','watch']);p.add_argument('--run',required=True,type=Path);a=p.parse_args();globals()[a.action](a.run)
