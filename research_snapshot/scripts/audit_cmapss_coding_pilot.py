"""Independent numerical and execution accounting audit after A435 is terminal."""
import argparse,collections,hashlib,json,math
from decimal import Decimal,localcontext
from pathlib import Path
from controlled_feedback import save,utc,sha

def main(root):
    assert (root/'EVALUATION_TERMINAL435.json').exists()
    for filename in ['HANDLE435.json','WATCH435.json']:
        pid=json.loads((root/filename).read_text())['pid']
        assert not Path('/proc',str(pid)).exists(),'Audit only terminal workers'
    checks=[]
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        assert value,name
    plan=json.loads((root/'EVALUATION_PLAN435.json').read_text())
    label=Path(plan['label_source']);check('official label identity',sha(label)==plan['label_sha256'])
    truth=[float(x) for x in label.read_text().split()];check('100 official labels',len(truth)==100)
    expected=json.loads((root/'RESULT435.json').read_text())
    bind=json.loads((root/'EXTERNAL_PROGRAMS_SEALED435.json').read_text())
    pred={}
    for ident,job in bind['jobs'].items():
        d=json.loads((root/'external_predictions'/ident/'RESULT.json').read_text())
        if d['status']!='complete':continue
        by_id={int(x['engine']):x['prediction'] for x in d['rows']}
        check(ident+':complete IDs',set(by_id)==set(range(1,101)) and len(d['rows'])==100)
        pred[ident]=[float(by_id[i]) for i in range(1,101)]
        for rel,digest in bind['artifacts'][ident].items():check(ident+':frozen '+rel,sha(Path(job['path'])/rel)==digest)
    base=[abs(a-b) for a,b in zip(pred['p00'],truth)]
    audit_scores={}
    for ident,values in pred.items():
        errs=[abs(a-b) for a,b in zip(values,truth)]
        score=dict(MAE=math.fsum(errs)/100,RMSE=math.sqrt(math.fsum((a-b)**2 for a,b in zip(values,truth))/100),
            capped125_MAE=math.fsum(abs(a-min(b,125)) for a,b in zip(values,truth))/100,
            harm=math.fsum(max(e-b,0) for e,b in zip(errs,base))/100,
            gain=math.fsum(max(b-e,0) for e,b in zip(errs,base))/100,
            wins=sum(e<b for e,b in zip(errs,base)),ties=sum(e==b for e,b in zip(errs,base)),
            losses=sum(e>b for e,b in zip(errs,base)))
        with localcontext() as ctx:
            ctx.prec=40
            decimal_mae=sum((abs(Decimal(str(a))-Decimal(str(b))) for a,b in zip(values,truth)),Decimal(0))/100
        check(ident+':independent decimal MAE',abs(float(decimal_mae)-score['MAE'])<1e-10)
        for key,value in score.items():check(ident+':'+key,math.isclose(value,expected['scores'][ident][key],rel_tol=1e-10,abs_tol=1e-10))
        audit_scores[ident]=score
    requested=receipted=inputs=outputs=0
    request_http=collections.Counter();returned=collections.Counter();stages=[]
    for row in json.loads((root/'SESSIONS337.json').read_text()):
        run=Path(row['path']);requests=list(run.glob('model_gateway/*/*_request.json'))
        receipts=list(run.glob('model_gateway/*/*_receipt.json'));requested+=len(requests);receipted+=len(receipts)
        for receipt in receipts:
            info=json.loads(receipt.read_text());prefix=receipt.name.split('_')[0]
            check(str(receipt.relative_to(root))+':request hash',sha(receipt.with_name(prefix+'_request.json'))==info['request_sha256'])
            if 'response_sha256' in info:
                check(str(receipt.relative_to(root))+':response hash',sha(receipt.with_name(prefix+'_response.sse'))==info['response_sha256'])
            check(str(receipt.relative_to(root))+':requested model',info['model_requested']=='deepseek-flash')
            request_http[str(info.get('upstream_status'))]+=1;returned.update(info.get('returned_model_ids',[]))
            inputs+=info.get('usage',{}).get('input_tokens',0) or 0;outputs+=info.get('usage',{}).get('output_tokens',0) or 0
        state=json.loads((run/'broker/STATE.json').read_text());native=json.loads((run/'primary'/row['id']/'status.json').read_text())
        report=json.loads((run/'BUILD_RESULT.json').read_text())
        check(row['id']+':native terminal',native['status'] in ['finished','execution_error','timeout','invalid_answer'])
        check(row['id']+':submission identity',state['submission']==report['submission'])
        check(row['id']+':normal definition',report['completed']==(native['status']=='finished' and state['submission'] is not None))
        commands=[]
        for line in (run/'primary'/row['id']/'events.jsonl').open():
            try:event=json.loads(line)
            except ValueError:continue
            item=event.get('item',{})
            if item.get('type')=='command_execution' and event.get('type')=='item.completed':
                commands.append(dict(command=item.get('command'),exit_code=item.get('exit_code')))
        save(run/'COMMANDS_FOR_REVIEW435.json',commands)
        stages.append(dict(id=row['id'],normal=report['completed'],native_status=native['status'],
            valid_tool_submission=state['submission'] is not None,registered_attempts=state['attempts'],
            complete_new_programs=len(state['programs'])-3,checks=len(state['checks']),commands=len(commands),
            actual_requests=len(requests),actual_receipts=len(receipts)))
    fits=[]
    for f in root.glob('subjects/*/programs/*/full_fit/PROCESS.json'):
        proc=json.loads(f.read_text());check(str(f.relative_to(root))+':fit mode',proc['mode']=='fit')
        audit=f.with_name('training_audit.json');d=json.loads(audit.read_text()) if audit.exists() else {}
        fits.append(dict(path=str(f.relative_to(root)),returncode=proc.get('returncode'),
            learning_audit_present=bool(d),estimator_fits=d.get('completed_estimator_fits',0),
            optimizer_steps=d.get('optimizer_steps',0),nonzero_sampled_optimizer_steps=d.get('nonzero_sampled_optimizer_steps',0)))
    pred_audits=list((root/'external_predictions').rglob('training_audit.json'))
    for f in pred_audits:
        d=json.loads(f.read_text());check(str(f.relative_to(root))+':no final fit',d['mode']=='predict' and d['completed_estimator_fits']==0 and d['optimizer_steps']==0)
    result=dict(utc=utc(),passed=True,checks=len(checks),details=checks,numerical_scores=audit_scores,stages=stages,
        costs=dict(actual_model_requests=requested,actual_receipts=receipted,upstream_status=dict(request_http),
            returned_model_ids=dict(returned),input_tokens=inputs,output_tokens=outputs,
            scientific_fit_entry_attempts=len(fits),qualification_fit_entry_attempts=1,
            completed_scientific_estimator_fits=sum(f['estimator_fits'] for f in fits),
            optimizer_steps=sum(f['optimizer_steps'] for f in fits),nonzero_sampled_optimizer_steps=sum(f['nonzero_sampled_optimizer_steps'] for f in fits),
            cold_final_prediction_audits=len(pred_audits)),fit_entries=fits,
        behavioral_compliance='Command and generated-source manual review still required; instrumentation alone does not prove no unregistered fit.',
        no_additional_model_training_or_prediction=True)
    save(root/'INDEPENDENT_AUDIT435.json',result)
    print(json.dumps(dict(passed=True,checks=len(checks),costs=result['costs'],stages=stages)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())

