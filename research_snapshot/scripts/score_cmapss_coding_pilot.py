"""Seal all native choices before cold one-engine predictions and external RUL scoring."""
import argparse,json,math,os,shutil,tarfile,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from controlled_feedback import save,sha,utc
from cmapss_program_runtime import Runtime

def evaluate(root):
    assert (root/'ALL_CHOICES_SEALED435.json').exists() and (root/'TERMINAL435.json').exists()
    assert not (root/'EVALUATION_HANDLE435.json').exists(),'Never repeat external evaluation implicitly'
    controller=json.loads((root/'HANDLE435.json').read_text())['pid']
    assert not Path('/proc',str(controller)).exists(),'Native controller must be terminal'
    plan=json.loads((root/'EVALUATION_PLAN435.json').read_text())
    for path,digest in plan['evaluation_code_sha256'].items():assert sha(Path(path))==digest
    save(root/'EVALUATION_HANDLE435.json',dict(pid=os.getpid(),utc=utc(),state='running'))
    raw_path=Path(plan['test_source']);assert sha(raw_path)==plan['test_sha256']
    raw=np.loadtxt(raw_path);engines=sorted(set(raw[:,0].astype(int)))
    assert engines==list(range(1,101)) and raw.shape[1]==26
    columns=np.load(root/'inputs/train.npz',allow_pickle=False)['columns']
    jobs={i:dict(path=str(root/'reference'/i),host=str(root)) for i in ['p00','ref_selected','ref_rejected']}
    choices={}
    for row in json.loads((root/'SESSIONS337.json').read_text()):
        run=Path(row['path']);state=json.loads((run/'broker/STATE.json').read_text())
        for ident,v in state['programs'].items():
            if ident not in jobs:jobs[row['id']+'/'+ident]=dict(path=v['path'],host=str(run))
        selected=state.get('submission')
        if selected:
            ident=selected['program_id'];choices[row['id']]=ident if ident in ['p00','ref_selected','ref_rejected'] else row['id']+'/'+ident
        else:choices[row['id']]=None
    save(root/'EXTERNAL_PROGRAMS_SEALED435.json',dict(utc=utc(),choices=choices,jobs=jobs,
        choice_receipt_sha256=sha(root/'ALL_CHOICES_SEALED435.json'),
        artifacts={k:json.loads((Path(v['path'])/'REPORT.json').read_text())['artifacts'] for k,v in jobs.items()}))
    started=time.monotonic()
    def one_program(item):
        ident,job=item;folder=root/'external_predictions'/ident;folder.mkdir(parents=True)
        program=Path(job['path']);engine=Runtime(Path(job['host']))
        engine.deadline=time.monotonic()+plan['program_seconds']
        report=json.loads((program/'REPORT.json').read_text())
        for rel,digest in report['artifacts'].items():assert sha(program/rel)==digest
        def one(unit):
            values=raw[raw[:,0]==unit];inp=folder/(str(unit)+'.npz')
            np.savez_compressed(inp,values=values,columns=columns)
            out=folder/str(unit)
            _,audit=engine.command(program/'source',program/'full_fit/artifacts',inp,out,'predict')
            p=np.asarray(json.loads((out/'predictions.json').read_text())['rul'],dtype=float)
            assert p.shape==(len(values),) and np.isfinite(p).all() and np.all(p>=0)
            return dict(engine=int(unit),cycle=int(values[-1,1]),prediction=float(p[-1]),
                        audit_sha256=sha(out/'training_audit.json'))
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,engines))
            for rel,digest in report['artifacts'].items():assert sha(program/rel)==digest
            result=dict(status='complete',rows=rows,prediction_count=100,weights_and_source_unchanged=True)
        except Exception as error:result=dict(status='failed',error_type=type(error).__name__,error=str(error))
        save(folder/'RESULT.json',result);return ident,result
    outputs={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one_program,item) for item in jobs.items()]):
            ident,result=f.result();outputs[ident]=result
            save(root/'EXTERNAL_PROGRESS435.json',dict(completed=len(outputs),total=len(jobs),statuses={k:v['status'] for k,v in outputs.items()}))
    save(root/'ALL_PREDICTIONS_SEALED435.json',dict(utc=utc(),seconds=time.monotonic()-started,
        results={k:sha(root/'external_predictions'/k/'RESULT.json') for k in outputs}))
    # Only now read the unchanged external official labels.
    label_path=Path(plan['label_source']);assert sha(label_path)==plan['label_sha256'];truth=np.loadtxt(label_path)
    assert truth.shape==(100,) and np.isfinite(truth).all()
    if outputs['p00']['status']!='complete':raise RuntimeError('Common starting program external evaluation failed')
    baseline=np.asarray([r['prediction'] for r in outputs['p00']['rows']]);base_error=np.abs(baseline-truth)
    scores={}
    for ident,value in outputs.items():
        if value['status']!='complete':scores[ident]=dict(status='failed');continue
        pred=np.asarray([r['prediction'] for r in value['rows']]);error=np.abs(pred-truth)
        mae=float(error.mean());independent=math.fsum(abs(float(p)-float(y)) for p,y in zip(pred,truth))/100
        assert math.isclose(mae,independent,rel_tol=1e-12,abs_tol=1e-12)
        scores[ident]=dict(status='complete',MAE=mae,RMSE=float(np.sqrt(np.mean((pred-truth)**2))),
            capped125_MAE=float(np.mean(np.abs(pred-np.minimum(truth,125)))),
            harm=float(np.maximum(error-base_error,0).mean()),gain=float(np.maximum(base_error-error,0).mean()),
            wins=int((error<base_error).sum()),ties=int((error==base_error).sum()),losses=int((error>base_error).sum()))
    episodes=json.loads((root/'ALL_CHOICES_SEALED435.json').read_text())['results'];rows=[]
    for episode in episodes:
        choice=choices[episode['id']]
        rows.append(dict(id=episode['id'],arm=episode['arm'],normal=episode['completed'],
                         selected=choice,score=scores.get(choice),seconds=episode['seconds']))
    aggregate={}
    for arm in ['ordinary','causal_repair']:
        members=[r for r in rows if r['arm']==arm]
        valid=len(members)==2 and all(r['normal'] and r['score'] and r['score']['status']=='complete' for r in members)
        aggregate[arm]=dict(all_two_normal_and_scored=valid,assigned=2)
        if valid:aggregate[arm].update({k:float(np.mean([r['score'][k] for r in members])) for k in ['MAE','harm','gain']})
    valid=all(v['all_two_normal_and_scored'] for v in aggregate.values())
    gate=False
    if valid:
        a,b=aggregate['ordinary'],aggregate['causal_repair']
        gate=b['MAE']<a['MAE'] and b['MAE']<scores['p00']['MAE'] and b['harm']<=a['harm'] and b['gain']>=a['gain']
    fixed=plan['fixed_repair_preexisting_score']
    result=dict(utc=utc(),source_clusters=1,independent_confirmation=0,goal_complete=False,
        primary_pilot_gate=gate,all_four_normal_and_scored=valid,episodes=rows,aggregate=aggregate,scores=scores,
        fixed_repair_control=fixed,beats_fixed_repair=(valid and aggregate['causal_repair']['MAE']<fixed['MAE']),
        final_truth_sha256=sha(label_path),prediction_seconds=time.monotonic()-started,
        independent_scalar_MAE_checks=sum(v['status']=='complete' for v in scores.values()),
        selection_regret_not_compared_across_different_pools=True,
        limitations=['Historically opened source; development pilot only.','Two repeats, one dataset, not independent task majority.',
                     'Fixed researcher repair and historical ordinary successes remain strong counterevidence.'])
    save(root/'RESULT435.json',result)
    save(root/'EVALUATION_TERMINAL435.json',dict(utc=utc(),status='completed',programs=len(outputs),primary_pilot_gate=gate))
    print(json.dumps(dict(primary_pilot_gate=gate,aggregate=aggregate)),flush=True)

def output_cap(root):
    """Precommitted output-only control; never changes an Agent submission."""
    assert (root/'EVALUATION_TERMINAL435.json').exists()
    plan_path=root/'OUTPUT_CAP_ANALYSIS_PLAN435.json'
    plan=json.loads(plan_path.read_text())
    result=json.loads((root/'RESULT435.json').read_text())
    assert plan['utc'] < result['utc'], 'Plan must predate the current scores'
    binding=json.loads((root/'EXTERNAL_PROGRAMS_SEALED435.json').read_text())
    evaluation=json.loads((root/'EVALUATION_PLAN435.json').read_text())
    label=Path(evaluation['label_source'])
    assert sha(label)==evaluation['label_sha256']
    final_truth=[float(v) for v in label.read_text().split()]
    views={}; hashes={str(plan_path.relative_to(root)):sha(plan_path)}
    def read(path):
        hashes[str(path.relative_to(root))]=sha(path)
        return json.loads(path.read_text())
    def score(truth, prediction):
        before=[abs(p-y) for p,y in zip(prediction,truth)]
        after=[abs(min(p,125.0)-y) for p,y in zip(prediction,truth)]
        n=len(truth); assert len(prediction)==n and n>0
        mae=math.fsum(before)/n; clipped=math.fsum(after)/n
        assert math.isclose(mae,float(np.mean(before)),abs_tol=1e-10)
        assert math.isclose(clipped,float(np.mean(after)),abs_tol=1e-10)
        return dict(n=n,original_MAE=mae,clipped_MAE=clipped,delta_MAE=clipped-mae,
            harm=math.fsum(max(a-b,0) for a,b in zip(after,before))/n,
            gain=math.fsum(max(b-a,0) for a,b in zip(after,before))/n,
            affected=sum(p>125 for p in prediction),
            wins=sum(a<b for a,b in zip(after,before)),
            ties=sum(a==b for a,b in zip(after,before)),
            losses=sum(a>b for a,b in zip(after,before)))
    for ident,job in sorted(binding['jobs'].items()):
        if ident.startswith('ref_'): continue
        program=Path(job['path'])
        full=read(program/'development/PREDICTIONS.json')
        row=dict(full=score([y for r in full for y in r['truth']],
                            [p for r in full for p in r['prediction']]))
        if '/' in ident:
            subject,pid=ident.split('/'); check_path=root/'subjects'/subject/'checks'/pid/'CHECK.json'
        else:
            copies=sorted(root.glob('subjects/*/checks/'+ident+'/CHECK.json'))
            check_path=copies[0] if copies else root/'reference_checks'/ident/'CHECK.json'
            if copies:
                canonical=read(copies[0])['rows']
                for copy in copies[1:]: assert read(copy)['rows']==canonical
        if check_path.exists():
            prefixes=read(check_path)['rows']
            row['prefix']=score([r['truth'] for r in prefixes],[r['prediction'] for r in prefixes])
        else: row['prefix']=None
        final=read(root/'external_predictions'/ident/'RESULT.json')
        assert final['status']=='complete'
        lookup={int(r['engine']):r['prediction'] for r in final['rows']}
        row['final']=score(final_truth,[lookup[i] for i in range(1,101)])
        assert math.isclose(row['final']['original_MAE'],result['scores'][ident]['MAE'],abs_tol=1e-10)
        views[ident]=row
    save(root/'OUTPUT_CAP_RESULT435.json',dict(utc=utc(),intervention='min(prediction,125)',
        threshold_chosen_before_A435=True,plan_predates_current_scores=True,programs=views,
        source_hashes=hashes,primary_result_unchanged=True,new_training=0,new_inference=0,
        new_LLM_calls=0,independent_confirmation=0,limits=plan['limits']))
    print(json.dumps({k:{v:(None if s is None else [s['original_MAE'],s['clipped_MAE']])
                         for v,s in d.items()} for k,d in views.items()}),flush=True)

def watch(root):
    assert not (root/'WATCH435.json').exists()
    save(root/'WATCH435.json',dict(pid=os.getpid(),utc=utc(),state='waiting_for_all_choices'))
    deadline=time.monotonic()+2400
    while time.monotonic()<deadline:
        if (root/'TERMINAL435.json').exists():
            pid=json.loads((root/'HANDLE435.json').read_text())['pid']
            if not Path('/proc',str(pid)).exists():break
        time.sleep(2)
    else:raise TimeoutError('Native stage did not become terminal; no test scores opened')
    evaluate(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['watch','evaluate','output_cap']);p.add_argument('--run',type=Path,required=True);a=p.parse_args();globals()[a.action](a.run.resolve())
