"""External A337 scoring; refuses to open final arrays before all choices seal."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time

import numpy as np

from controlled_feedback import save,sha,utc
from program_revision_runtime import Runtime,load,score

SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261024T172000Z_programsource336')
PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')


def read(path):return json.loads(Path(path).read_text())


def independent_score(data,prediction):
    rows=[];matrix=np.zeros((4,4),dtype=int)
    for ident in np.unique(data['groups']):
        indices=np.flatnonzero(data['groups']==ident);ys=data['labels'][indices]
        assert len(set(ys.tolist()))==1
        counts={label:sum(int(prediction[i])==label for i in indices) for label in range(4)}
        guessed=min(counts,key=lambda label:(-counts[label],label));truth=int(ys[0]);matrix[truth,guessed]+=1
        rows.append(dict(group=str(ident),actual=truth,predicted=guessed))
    return dict(record_error=sum(r['actual']!=r['predicted'] for r in rows)/len(rows),confusion=matrix.tolist(),records=rows)


def run(root):
    seal=root/'ALL_CHOICES_SEALED337.json';assert seal.exists(),'All choices must be sealed first'
    choices=read(seal);assert not (root/'RESULTS337.json').exists(),'Never repeat completed final scoring'
    assert not choices['final_scores_joined'];assert len(choices['episodes'])==4
    assert sha(root/'EFFECTIVE_CODE_FREEZE337.json')==choices['code_freeze_sha256']
    for name,digest in read(root/'EFFECTIVE_CODE_FREEZE337.json').items():assert sha(root/name)==digest
    source_qualification=read(SOURCE/'QUALIFICATION336.json');path=SOURCE/'qualification/final.npz'
    assert sha(path)==source_qualification['files']['final.npz']['sha256']
    save(root/'FINAL_EVALUATION_STARTED337.json',dict(utc=utc(),choices_sha256=sha(seal),final_source_sha256=sha(path),evaluator_sha256=sha(Path(__file__)),all_program_choices_precede_final_read=True))
    # This is the first post-seal read of final numerical arrays in this study.
    data=load(path);catalog={'common_p00':root/'reference/p00'}
    for episode in choices['episodes']:
        state=read(root/'subjects'/episode['id']/'broker/STATE.json')
        for ident,v in state['programs'].items():
            if ident!='p00':catalog[episode['id']+'__'+ident]=Path(v['path'])
    def evaluate(item):
        name,program=item;report=read(program/'REPORT.json')
        for relative,digest in report['artifacts'].items():assert sha(program/relative)==digest
        engine=Runtime(root);engine.deadline=time.monotonic()+600
        output=root/'evaluation'/name;output.mkdir(parents=True)
        try:
            prediction,_=engine.predict(output/'forward',program/'source',program/'full_fit/artifacts',data)
            repeat,_=engine.predict(output/'cold_repeat',program/'source',program/'full_fit/artifacts',data)
            assert np.array_equal(prediction,repeat),'Nondeterministic cold prediction'
            independent=independent_score(data,prediction);result=score(data,prediction)
            assert result['confusion']==independent['confusion'] and abs(result['record_error']-independent['record_error'])<1e-12
            result.update(status='complete',cold_repeat_identical=True,independent_record_vote_verified=True,actual_prediction_processes=2*len(independent['records']))
        except Exception as error:result=dict(status='evaluation_error',error_type=type(error).__name__,error=str(error))
        save(output/'RESULT.json',result);return name,result
    evaluated={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(evaluate,item) for item in catalog.items()]):
            name,result=f.result();evaluated[name]=result;save(root/'EVALUATION_PROGRESS337.json',evaluated)
    assert evaluated['common_p00']['status']=='complete','Common baseline evaluation failed'
    baseline=evaluated['common_p00']['record_error'];rows=[]
    for episode in choices['episodes']:
        prefix=episode['id']+'__';state=read(root/'subjects'/episode['id']/'broker/STATE.json')
        def lookup(ident):return evaluated['common_p00' if ident=='p00' else prefix+ident]
        complete=[lookup(i)['record_error'] for i in state['programs'] if lookup(i)['status']=='complete']
        pool_best=min(complete);decisions={'keep':'p00'}
        if episode.get('submission'):decisions['actor_submission']=episode['submission']['program_id']
        for key in ['transaction','without_new_check','fixed_all_programs_no_check','fixed_development','fixed_mean_cv']:
            if key in episode:decisions[key]=episode[key]['program_id']
        scores={}
        for method,ident in decisions.items():
            value=lookup(ident);entry=dict(program_id=ident,evaluation_status=value['status'])
            if value['status']=='complete':
                loss=value['record_error'];opportunity=baseline-pool_best
                entry.update(record_error=loss,record_accuracy=1-loss,window_error=value['window_error'],macro_F1=value['macro_F1'],
                    common_bank_descriptive_selection_loss=loss-pool_best,harm=max(0.,loss-baseline),gain=baseline-loss,
                    gain_capture=(min(1.,max(0.,baseline-loss)/opportunity) if opportunity>0 else None))
            scores[method]=entry
        rows.append(dict(id=episode['id'],arm=episode['arm'],repeat=episode['repeat'],valid_actor_submission=episode['actor_submission_valid'],
            end_to_end_completed=episode['completed'],seconds=episode['seconds'],fallback=episode['fallback'],
            actor_matches_transaction=(episode.get('submission',{}).get('program_id')==episode.get('transaction',{}).get('program_id')) if episode.get('submission') else None,
            own_pool_best_error_descriptive_only=pool_best,programs_in_pool=len(state['programs']),methods=scores))
    summary={}
    for arm,method in [('ordinary_coding','actor_submission'),('witness_transaction','transaction')]:
        rr=[r for r in rows if r['arm']==arm];valid=[r for r in rr if r['end_to_end_completed'] and method in r['methods'] and r['methods'][method]['evaluation_status']=='complete']
        summary[arm]=dict(planned=len(rr),valid=len(valid),failures=len(rr)-len(valid),
            mean_record_error=(float(np.mean([r['methods'][method]['record_error'] for r in valid])) if len(valid)==len(rr) else None),
            mean_harm=(float(np.mean([r['methods'][method]['harm'] for r in valid])) if valid else None),
            mean_seconds=float(np.mean([r['seconds'] for r in rr])))
    ordinary=summary['ordinary_coding']['mean_record_error'];method=summary['witness_transaction']['mean_record_error']
    comparison='incomplete' if ordinary is None or method is None else ('method_win' if method<ordinary else 'tie' if method==ordinary else 'method_loss')
    audits=[]
    for p in root.rglob('training_audit.json'):
        if p.parent.name.startswith('prediction'):continue
        a=read(p)
        if a.get('mode')=='fit':audits.append((p,a))
    receipts=[read(p) for p in root.glob('subjects/*/model_gateway/*/*_receipt.json')]
    usage=[r.get('usage') for r in receipts]
    reused_fit_entries=sum(p.is_relative_to(root/'reference/p00') for p,_ in audits)
    dispatched=len(list(root.glob('subjects/*/model_gateway/*/*_request.json')))
    cost=dict(model_requests=len(receipts),all_usage_present=all(u is not None for u in usage),
        model_requests_dispatched=dispatched,model_request_receipts=len(receipts),missing_request_receipts=dispatched-len(receipts),
        input_tokens=sum((u or {}).get('input_tokens',0) for u in usage),output_tokens=sum((u or {}).get('output_tokens',0) for u in usage),
        returned_model_ids=sorted({model for r in receipts for model in r.get('returned_model_ids',[])}),
        real_fit_entry_executions=len(audits),completed_estimator_fits=sum(a['completed_estimator_fits'] for _,a in audits),
        reused_baseline_fit_entry_executions=reused_fit_entries,new_fit_entry_executions=len(audits)-reused_fit_entries,
        optimizer_steps=sum(a['optimizer_steps'] for _,a in audits),final_prediction_processes=sum(v.get('actual_prediction_processes',0) for v in evaluated.values()))
    result=dict(iteration='A337',utc=utc(),status='complete',scope='One measured-rig development cohort,two repetitions per arm;zero independent confirmation scenarios.',
        baseline_error=baseline,primary=summary,development_scenario_comparison=comparison,independent_confirmation_scenarios=0,
        own_pool_oracle_is_not_cross_arm_advantage=True,rows=rows,cost=cost,
        evaluated_programs=len(evaluated),evaluation_failures=sum(v['status']!='complete' for v in evaluated.values()),all_results=evaluated,
        majority_goal_complete=False,final_source_sha256=sha(path),choices_seal_sha256=sha(seal))
    save(root/'RESULTS337.json',result);save(PROJECT/'results/freeprogram337/RESULTS337.json',result)
    from free_program_comparison import status
    status(root,phase='external_evaluation_complete',controller_status='terminal',final_scores_opened=True,development_comparison=comparison)
    print(json.dumps(dict(primary=summary,comparison=comparison,cost=cost),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run(a.run)
