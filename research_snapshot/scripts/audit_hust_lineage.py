"""Independent raw-output and training provenance audit after A442 sealing."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from audit_program_trial import python_review
from controlled_feedback import save, sha, utc

read=lambda p:json.loads(Path(p).read_text())


def audit(root):
    followup=(root/'PROTOCOL445.json').exists()
    tag='445' if followup else '442'
    assert read(root/('TERMINAL445.json' if followup else 'CONTROLLER_TERMINAL442.json'))['exit_code']==0
    assert read(root/('HANDLE'+tag+'.json'))['state']=='terminal'
    result=read(root/('RESULT'+tag+'.json')); cfg=read(root/('PROTOCOL'+tag+'.json'))
    assert len(read(root/('ALL_CHOICES_SEALED444.json' if followup else 'ALL_CHOICES_SEALED442.json'))['results'])==(2 if followup else 4)
    ordinary_index={}
    if followup:
        archive=Path(cfg['ordinary_archive'])
        assert sha(archive/'SOURCE_MANIFEST.json')==cfg['ordinary_manifest_sha256']
        ordinary_index={x['path']:x for x in read(archive/'SOURCE_MANIFEST.json')['files']}
        prior=read(root/'audit/reused_training/audit/EXECUTION_AUDIT444.json')
        assert prior['passed'] and prior['checks']==269
        assert len(read(root/'reused_ordinary/ALL_CHOICES_SEALED444.json')['results'])==2
    checks=0
    for relative,digest in read(root/('LAUNCH_FREEZE'+tag+'.json')).items():
        assert sha(root/relative)==digest,relative; checks+=1
    source=Path(cfg['source_archive'])
    index={e['path']:e for e in read(source/'SOURCE_MANIFEST.json')['files']}
    assert sha(source/'SOURCE_MANIFEST.json')==cfg['source_manifest_sha256']
    inputs={}
    for name in ['training_queries_private','discovery_truth']:
        p=source/'private_inputs'/(name+'.npz')
        assert sha(p)==index[str(p.relative_to(source))]['sha256'];checks+=1
        with np.load(p,allow_pickle=False) as z:inputs[name]={k:z[k].copy() for k in z.files}
    truth=inputs['discovery_truth']; queries=inputs['training_queries_private']
    commands=[]; statistics=[]; audited=set()
    for row in result['rows']:
        reused=followup and row['arm']=='ordinary'
        base_root=root/'reused_ordinary' if reused else root
        run=base_root/'subjects'/row['id'];state=read(run/'broker/STATE.json');errors={};predictions={}
        for ident,item in state['programs'].items():
            initial=item['report'].get('initial_models_reused')
            folder=((root/'reference'/ident if initial else run/'programs'/ident)
                    if followup else Path(item['path']))
            report=read(folder/'REPORT.json')
            for name,digest in report['artifacts'].items():
                if reused and not initial:
                    assert ordinary_index[str((folder/name).relative_to(base_root))]['sha256']==digest
                else:assert sha(folder/name)==digest
                checks+=1
            if report.get('initial_models_reused'):
                original=source/report['original_program_path']/'REPORT.json'
                assert sha(original)==report['original_report_sha256']==index[str(original.relative_to(source))]['sha256'];checks+=1
                predroot=root/'qualification/parents'/ident
            else:
                predroot=folder/'development'
                for fold in report['cv']:
                    stage=(root/'audit/reused_training'/folder.relative_to(base_root)/fold['fold']
                           if reused else folder/fold['fold'])
                    z=np.load(stage/'check/PREDICTIONS.npz',allow_pickle=False)
                    lookup={str(i):j for j,i in enumerate(queries['sample_ids'])}
                    indices=[lookup[str(i)] for i in z['sample_ids']]
                    raw=[read(stage/'check'/('prediction'+str(j))/'predictions.json')['rul'][0] for j in range(len(indices))]
                    assert np.array_equal(z['prediction'],raw);checks+=1
                    if reused:
                        # Original real-input train/CV disjointness was checked
                        # before archiving; bind that receipt and these outputs.
                        for p in (stage/'check').rglob('*'):
                            if p.is_file():
                                rel=str(p.relative_to(root/'audit/reused_training'))
                                assert sha(p)==ordinary_index[rel]['sha256'];checks+=1
                    else:
                        with np.load(stage/'train.npz',allow_pickle=False) as train:
                            assert not set(train['groups'])&set(queries['groups'][indices]);checks+=1
                    bycell={}
                    for j,i in enumerate(indices):bycell.setdefault(str(queries['groups'][i]),[]).append(abs(raw[j]-float(queries['y'][i])))
                    reconstructed={g:math.fsum(v)/len(v) for g,v in bycell.items()}
                    claimed={c['cell']:c['MAE'] for c in fold['cells']}
                    assert set(claimed)==set(reconstructed)
                    for g,v in reconstructed.items():assert abs(v-claimed[g])<1e-8;checks+=1
                    assert abs(math.fsum(reconstructed.values())/len(reconstructed)-fold['mean_cell_MAE'])<1e-8;checks+=1
            values=[read(predroot/('prediction'+str(i))/'predictions.json')['rul'][0] for i in range(8)]
            with np.load(predroot/'PREDICTIONS.npz',allow_pickle=False) as z:
                assert np.array_equal(z['sample_ids'],truth['sample_ids']) and np.array_equal(z['prediction'],values);checks+=2
            delta=[abs(float(p)-float(y)) for p,y in zip(values,truth['y'])]
            errors[ident]=delta;predictions[ident]=values
            claimed=row['programs'][ident]
            assert claimed['errors' if followup else 'absolute_errors']==delta and abs(claimed['MAE']-math.fsum(delta)/8)<1e-8;checks+=2
            if followup:
                cv_values=[c['MAE'] for fold in report['cv'] for c in fold['cells']]
                assert len(cv_values)==47 and abs(claimed['CV']-math.fsum(cv_values)/47)<1e-8;checks+=1
            audited.add(str(folder))
        selected=row['selected']
        if selected is not None:
            delta=[a-b for a,b in zip(errors[selected],errors['p00'])]
            metrics=dict(MAE=math.fsum(errors[selected])/8,harm=math.fsum(max(x,0) for x in delta)/8,
                         gain=math.fsum(max(-x,0) for x in delta)/8)
            for k,v in metrics.items():assert abs(v-row['metrics' if followup else 'selected_metrics'][k])<1e-8;checks+=1
            assert abs(metrics['MAE']-(math.fsum(errors['p00'])/8+metrics['harm']-metrics['gain']))<1e-8;checks+=1
        else:metrics=None
        statistics.append(dict(id=row['id'],arm=row['arm'],completed=row['completed'],metrics=metrics))
        if followup:
            cv_choice=min(row['programs'],key=lambda p:(row['programs'][p]['CV'],p!='p00',list(row['programs']).index(p)))
            assert cv_choice==row['fixed_CV_choice'];checks+=1
            assert abs(math.fsum(errors[cv_choice])/8-row['fixed_CV_metrics']['MAE'])<1e-8;checks+=1
            import statistics as stat
            for key,combine in [('equal_mean_control',stat.mean),('equal_median_control',stat.median)]:
                combined=[combine(values) for values in zip(*predictions.values())]
                combined_errors=[abs(p-float(y)) for p,y in zip(combined,truth['y'])]
                assert abs(math.fsum(combined_errors)/8-row[key]['MAE'])<1e-8;checks+=1
        if reused:continue
        events=run/'primary'/row['id']/'events.jsonl'
        for number,line in enumerate(events.read_text().splitlines(),1):
            event=json.loads(line);item=event.get('item',{})
            if event.get('type')=='item.completed' and item.get('type')=='command_execution':
                command=item['command']
                commands.append(dict(episode=row['id'],line=number,command=command,exit_code=item.get('exit_code'),**python_review(command)))
    summaries=({arm:dict(means=value) for arm,value in result['means'].items()}
               if followup else result['arms'])
    for arm,summary in summaries.items():
        rows=[r for r in statistics if r['arm']==arm]
        complete=all(r['completed'] and r['metrics'] is not None for r in rows)
        assert (summary['means'] is not None)==complete;checks+=1
        if complete:
            for key in ['MAE','harm','gain']:
                assert abs(math.fsum(r['metrics'][key] for r in rows)/2-summary['means'][key])<1e-8;checks+=1
    fits=[]
    for p in root.glob('subjects/*/programs/*/**/PROCESS.json'):
        process=read(p)
        if process.get('mode')=='fit':fits.append(dict(path=str(p.relative_to(root)),returncode=process.get('returncode'),audit_present=(p.parent/'training_audit.json').exists()))
    assert len(fits)==result['cost_new_only' if followup else 'cost']['fit_entries_with_process_record'];checks+=1
    costs=None
    if followup:
        receipts=[]
        for p in root.glob('subjects/*/model_gateway/*/*_receipt.json'):
            receipt=read(p);prefix=str(receipt['sequence']).zfill(3)
            assert sha(p.parent/(prefix+'_request.json'))==receipt['request_sha256'];checks+=1
            assert sha(p.parent/(prefix+'_response.sse'))==receipt['response_sha256'];checks+=1
            assert receipt['model_requested']=='deepseek-flash' and receipt['returned_model_ids']==['deepseek-flash'];checks+=1
            receipts.append(receipt)
        costs=dict(actual_requests=len(list(root.glob('subjects/*/model_gateway/*/*_request.json'))),
            receipts=len(receipts),input_tokens=sum(r['usage']['input_tokens'] for r in receipts),
            output_tokens=sum(r['usage']['output_tokens'] for r in receipts),
            cached_input_tokens=sum(r['usage'].get('input_tokens_details',{}).get('cached_tokens',0) for r in receipts))
        for key in ['actual_requests','receipts','input_tokens','output_tokens']:
            assert costs[key]==result['cost_new_only'][key];checks+=1
    flagged=[c for c in commands if c['possible_learning_calls'] or c['unresolved']]
    save(root/('audit/COMMANDS'+tag+'.json'),commands)
    output=dict(utc=utc(),passed=True,checks=checks,unique_complete_programs=len(audited),
        fit_processes=fits,executed_commands=len(commands),flagged_commands=flagged,
        command_review_pending=bool(flagged),result_sha256=sha(root/('RESULT'+tag+'.json')),
        auditor_sha256=sha(Path(__file__)),command_parser_sha256=sha(Path(__file__).with_name('audit_program_trial.py')),
        new_cost_recount=costs,reused_ordinary_episodes=2 if followup else 0,
        final22_read=False,new_fits=0,new_LLM_calls=0,new_model_predictions=0)
    save(root/('audit/INDEPENDENT_AUDIT'+tag+'.json'),output)
    print(json.dumps({k:output[k] for k in ['passed','checks','unique_complete_programs','executed_commands','command_review_pending']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    audit(p.parse_args().run.resolve())
