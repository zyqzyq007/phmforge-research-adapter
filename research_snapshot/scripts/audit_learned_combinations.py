"""Independent parameter equations and scoring; share only source-file parsing."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def main(root):
    spec=importlib.util.spec_from_file_location('frozen447',root/'evaluation_code/screen_learned_combinations.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    read=loader.read;check_count=0
    def check(value,label):
        nonlocal check_count
        assert value,label;check_count+=1
    def close(a,b,label):
        check(np.allclose(a,b,rtol=1e-8,atol=1e-7),label)
    cfg=read(root/'PROTOCOL447.json');result=read(root/'RESULT447.json')
    for name in ['LAUNCH_FREEZE447.json']:
        for path,digest in read(root/name).items():check(loader.sha(root/path)==digest,path)
    for name in ['LEGAL_BINDINGS447.json','OUTCOME_BINDINGS447.json']:
        for entry in read(root/name):check(loader.sha(entry['absolute_path'])==entry['sha256'],entry['absolute_path'])
    cases=read(root/'CASES447.json');seal=read(root/'ALL_FITS_SEALED447.json')
    check(len(cases)==12 and sum(len(c['candidate_ids']) for c in cases)==107,'full roster')
    all_rows=[];diagnostics=[];bound=loader.Bound()
    def native(c,d,p):
        if c['source'] in ['gear','gear_g2']:
            return 1-f1_score(d['y'],p,labels=list(range(c['class_count'])),average='macro',zero_division=0)
        errors=np.abs(d['y']-p) if c['source']=='rul' else (d['y']!=p).astype(float)
        return np.mean([np.mean(errors[d['groups']==g]) for g in set(d['groups'])])
    for case in cases:
        cid=case['id'];data=dict(np.load(root/'legal'/(cid+'.npz'),allow_pickle=False));model=read(root/'fits'/cid/'MODELS.json')
        check(loader.sha(root/'fits'/cid/'MODELS.json')==next(r['sha256'] for r in seal['models'] if r['case']==cid),'model sealed')
        classes=case['class_count'];m=len(case['candidate_ids']);n=len(data['y'])
        votes=np.eye(classes)[data['predictions'].astype(int)] if classes else data['predictions'][:,:,None]
        target=np.eye(classes)[data['y'].astype(int)] if classes else data['y'][:,None]
        weights=data['weights'];w=model['simplex']['weights'];scale=model['simplex']['scale']
        residual=np.tensordot(votes,w,axes=(1,0))-target
        objective=np.sum(weights[:,None]*residual**2)/weights.sum()/scale**2+1e-6*np.sum((np.asarray(w)-1/m)**2)
        close(objective,model['simplex']['objective'],'convex objective')
        grad=np.asarray([2*np.sum(weights[:,None]*residual*votes[:,j,:])/weights.sum()/scale**2+2e-6*(w[j]-1/m) for j in range(m)])
        close(grad,model['simplex']['gradient'],'convex gradient')
        support=np.asarray(w)>1e-7;stationary=np.mean(grad[support]);tol=1e-4*(1+np.max(np.abs(grad)))
        check(min(w)>=-1e-8 and abs(sum(w)-1)<1e-8,'simplex feasible')
        check(np.max(np.abs(grad[support]-stationary))<tol and np.all(grad[~support]>=stationary-tol),'convex KKT')
        rr=model['ridge'];x=(votes.reshape(n,-1)-rr['xmean'])/rr['xscale'];y=(target-rr['ymean'])/rr['yscale']
        beta=np.asarray(rr['coef']).reshape(-1,x.shape[1]).T
        errors=x@beta+rr['intercept']-y
        close(x.T@(weights[:,None]*errors)+beta,np.zeros_like(beta),'ridge normal equation')
        close(np.sum(weights[:,None]*errors,axis=0),np.zeros(y.shape[1]),'ridge intercept equation')
        distinct=len({tuple(data['predictions'][:,j].tolist()) for j in range(m)})
        diagnostics.append(dict(case=cid,development_groups=len(set(data['groups'])),development_prediction_vectors=distinct,
            all_members_correct=bool(np.all(data['predictions']==data['y'][:,None])),base_members=m,
            identical_votes_make_simplex_predictions_unidentifiable=distinct==1 and bool(classes)))
        external,old=loader.external(case,bound);old_index=case['candidate_ids'].index(case['original_incumbent'])
        p=external['predictions'];v=np.eye(classes)[p.astype(int)] if classes else p[:,:,None]
        scores=((v.reshape(len(p),-1)-rr['xmean'])/rr['xscale'])@beta+rr['intercept']
        scores=scores*np.asarray(rr['yscale'])+rr['ymean']
        predictions=dict(simplex_least_squares=np.tensordot(v,w,axes=(1,0)),ridge_calibration=scores,
                         equal_vote_or_mean=np.average(v,axis=1))
        for method,values in predictions.items():
            if classes:
                labels=[]
                for i,row in enumerate(values):
                    tied=np.flatnonzero(row>=max(row)-1e-12)
                    labels.append(int(p[i,old_index]) if int(p[i,old_index]) in tied else int(tied[0]))
                predictions[method]=np.asarray(labels)
            else:predictions[method]=np.maximum(values.reshape(-1),0)
        stored=dict(np.load(root/('PREDICTIONS_'+cid+'.npz'),allow_pickle=False))
        close(stored['actual'],external['y'],'truth source binding')
        check(np.array_equal(stored['sample_ids'],external['sample_ids']),'identity binding')
        for method,values in predictions.items():close(stored[method],values,'formula '+method)
        partial=root/'evaluation_attempt1'/('PREDICTIONS_'+cid+'.npz')
        if partial.exists():
            with np.load(partial,allow_pickle=False) as old_output:
                for key in old_output.files:check(np.array_equal(stored[key],old_output[key]),'shape repair unchanged existing output')
        for i,name in enumerate(case['candidate_ids']):close(native(case,external,p[:,i]),old[name]['loss'],'historical native score')
        for row in [r for r in result['rows'] if r['case']==cid]:
            value=native(case,external,stored[row['method']]);close(value,row['loss'],'independent native score')
            change=(value-row['original_loss'])/row['original_bank_span']
            close(change,row['normalized_change'],'normalized change');close(max(change,0),row['harm'],'harm');close(max(-change,0),row['gain'],'gain')
            all_rows.append(row)
    for method,summary in result['summaries'].items():
        for cluster,values in summary['clusters'].items():
            rows=[r for r in all_rows if r['method']==method and r['cluster']==cluster]
            for key,value in values.items():close(value,np.mean([r[key] for r in rows]),'source mean')
        for key in ['normalized_change','harm','gain']:close(summary[key],np.mean([c[key] for c in summary['clusters'].values()]),'macro mean')
    for method,controls in result['gates'].items():
        for control,gate in controls.items():
            a=result['summaries'][method];b=result['summaries'][control]
            deltas=[a['clusters'][c]['normalized_loss']-b['clusters'][c]['normalized_loss'] for c in a['clusters']]
            wins=sum(d < -1e-10 for d in deltas);ties=sum(abs(d)<=1e-10 for d in deltas)
            check(gate['wins']==wins and gate['ties']==ties and gate['losses']==4-wins-ties,'source outcomes')
            check(gate['passed']==(wins>=3 and a['harm']<=b['harm']+1e-10 and a['gain']>=b['gain']-1e-10),'gate')
    check(len(list((root/'fits').glob('*/MODELS.json')))==12,'24 learned combiners')
    check(sum(read(p)['returncode']==0 for p in (root/'fits').glob('*/PROCESS.json'))==11,'11 original successful processes')
    check(read(root/'fits/xjtu_bank/PROCESS.json')['returncode']==1 and read(root/'recovery_xjtu/PROCESS.json')['returncode']==0,'failed numerical attempt retained')
    report=dict(passed=True,utc=loader.utc(),checks=check_count,completed_supervised_combiners=24,failed_solver_attempts=1,
        actual_fit_processes=13,new_base_model_fits=0,new_LLM_calls=0,new_independent_confirmation=0,
        diagnostics=diagnostics,scope='Independent equations,sklearn task scoring,macro aggregation and decisions;source parser shared.')
    loader.save(root/'AUDIT447.json',report);print(json.dumps(dict(passed=True,checks=check_count,diagnostics=diagnostics)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);main(p.parse_args().run)
