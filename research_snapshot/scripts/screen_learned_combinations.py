"""Bounded supervised combination controls on all A328/A329 real candidate pools.

Base predictions are verified archived executions. New combiners fit only legal
development rows, in separate namespaces; old external outcomes join after seal.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

PYTHON = Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class Bound:
    def __init__(self): self.index={};self.rows={}
    def path(self,archive,relative):
        archive=Path(archive)
        if str(archive) not in self.index:
            receipt=read(archive/'ARCHIVE_VERIFIED.json')
            assert sha(archive/'SOURCE_MANIFEST.json')==receipt['manifest_sha256']
            self.index[str(archive)]={r['path']:r for r in read(archive/'SOURCE_MANIFEST.json')['files']}
        entry=self.index[str(archive)][relative];path=archive/relative
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],str(path)
        self.rows[str(path)]=dict(absolute_path=str(path),**entry)
        return path
    def json(self,a,p): return read(self.path(a,p))


def matrix(rows,candidate_ids):
    maps={c:{str(r.get('sample_id',r['group'])):r for r in rows[c]} for c in candidate_ids}
    first=maps[candidate_ids[0]];ids=sorted(first)
    assert all(len(m)==len(rows[c]) and set(m)==set(first) for c,m in maps.items())
    y=np.asarray([first[i]['actual'] for i in ids])
    groups=np.asarray([str(first[i]['group']) for i in ids])
    for m in maps.values():
        assert all(m[i]['actual']==first[i]['actual'] and str(m[i]['group'])==str(first[i]['group']) for i in ids)
    return dict(sample_ids=np.asarray(ids),groups=groups,y=y,
                predictions=np.asarray([[maps[c][i]['predicted'] for c in candidate_ids] for i in ids]))


def prepare(root):
    cfg=read(root/'PROTOCOL447.json');bound=Bound();cases=[];ordinary={};old_scores={}
    for spec in cfg['sources']:
        archive=Path(spec['archive']);assert sha(archive/'SOURCE_MANIFEST.json')==spec['manifest_sha256']
        catalog=bound.json(archive,'public/cases.json')
        episodes=bound.json(archive,'AGENT_CHOICES.json')['episodes']
        for c in catalog:
            raw=bound.json(archive,'public/rows/'+c['id']+'.json')
            if c['source']=='rul':
                context=bound.json(archive,'CAMPAIGN_CONFIG.json')['source_archives']['rul']
                models={m['name']:m for m in raw['models']};rows={}
                for cid in c['candidate_ids']:
                    pp=bound.json(context['path'],'c/workspaces/s278_frozen_r2/.runtime/bank_predictions/'+cid+'.json')['predictions']
                    signed={str(r['unit_id']):r for r in models[cid]['development']['per_bearing']}
                    assert all(r['queries']==1 for r in signed.values())
                    # Legal published signed error and actual legal prediction
                    # identify each development target; no private label lookup.
                    rows[cid]=[dict(sample_id=r['sample_id'],group=str(r['unit_id']),
                        actual=round(r['predicted_rul']-signed[str(r['unit_id'])]['bias_minutes'],9),
                        predicted=r['predicted_rul']) for r in pp]
            else: rows={cid:raw[cid]['development'] for cid in c['candidate_ids']}
            data=matrix(rows,c['candidate_ids'])
            counts=Counter(data['groups'].tolist());weights=np.asarray([1/counts[g] for g in data['groups']])
            class_count={'rotor':5,'electric_motor':4,'gear':3,'gear_g2':6}.get(c['source'],0)
            data.update(weights=weights,class_count=np.asarray(class_count),incumbent=np.asarray(c['candidate_ids'].index(c['original_incumbent'])))
            path=root/'legal'/(c['id']+'.npz');path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,**data)
            cases.append(dict(**c,archive=str(archive),cluster=cfg['clusters'][c['source']],
                              legal_rows=len(data['y']),legal_groups=len(counts),class_count=class_count))
            for e in episodes:
                assert e['completed'] and e['choice']
                matches=[r for r in e['choice']['choices'] if r['case']==c['id'] and r['incumbent']==c['original_incumbent']]
                assert len(matches)==1
                ordinary[c['id']+'::'+e['id']]=matches[0]['selected']
    assert len(cases)==cfg['expected_pools'] and sum(len(c['candidate_ids']) for c in cases)==cfg['expected_memberships']
    save(root/'CASES447.json',cases);save(root/'ORDINARY_CHOICES447.json',ordinary)
    save(root/'LEGAL_BINDINGS447.json',list(bound.rows.values()))
    (root/'code').mkdir();shutil.copyfile(__file__,root/'code'/Path(__file__).name)
    save(root/'LAUNCH_FREEZE447.json',{str(p.relative_to(root)):sha(p)
        for p in [root/'PROTOCOL447.json',root/'CASES447.json',root/'ORDINARY_CHOICES447.json',root/'LEGAL_BINDINGS447.json',
                  *sorted((root/'legal').glob('*.npz')),*sorted((root/'code').glob('*.py'))]})
    print(json.dumps(dict(prepared=12,legal_rows={c['id']:c['legal_rows'] for c in cases},new_fits=0)),flush=True)


def features(data):
    predictions=data['predictions'];classes=int(data['class_count'])
    if classes:
        return np.eye(classes)[predictions.astype(int)],np.eye(classes)[data['y'].astype(int)]
    return predictions[:,:,None].astype(float),data['y'][:,None].astype(float)


def fit_one(input_file,output,exact=False):
    from scipy.optimize import minimize
    from sklearn.linear_model import Ridge
    start=time.monotonic();data=dict(np.load(input_file,allow_pickle=False))
    voters,target=features(data);weights=data['weights'].astype(float)
    weights/=weights.sum();n,m,c=voters.shape;uniform=np.ones(m)/m
    # Scale only the regression least-squares objective; no final information.
    scale=max(float(np.sqrt(np.average((target[:,0]-np.average(target[:,0],weights=weights))**2,weights=weights))),1.) if c==1 else 1.
    v=voters/scale;t=target/scale;penalty=1e-6
    def objective(w):
        residual=np.einsum('nmc,m->nc',v,w)-t
        return float(np.einsum('n,nc,nc->',weights,residual,residual)+penalty*np.sum((w-uniform)**2))
    def gradient(w):
        residual=np.einsum('nmc,m->nc',v,w)-t
        return 2*np.einsum('n,nmc,nc->m',weights,v,residual)+2*penalty*(w-uniform)
    if exact:
        # Bounded numerical recovery for the failed six-member convex QP.
        # Every simplex face is considered under exactly the frozen objective.
        from itertools import combinations
        from types import SimpleNamespace
        assert m<=6
        q=np.einsum('n,nic,njc->ij',weights,v,v)+penalty*np.eye(m)
        b=np.einsum('n,nmc,nc->m',weights,v,t)+penalty*uniform
        feasible=[];faces=0
        for count in range(1,m+1):
            for active in combinations(range(m),count):
                active=list(active);faces+=1
                kkt=np.block([[q[np.ix_(active,active)],np.ones((count,1))],
                              [np.ones((1,count)),np.zeros((1,1))]])
                solution=np.linalg.solve(kkt,np.r_[b[active],1.])[:count]
                if solution.min() < -1e-10:continue
                w=np.zeros(m);w[active]=np.maximum(solution,0);w/=w.sum()
                feasible.append((objective(w),w))
        _,best=min(feasible,key=lambda pair:pair[0])
        opt=SimpleNamespace(x=best,success=True,nit=faces,message='Enumerated all63 simplex faces under unchanged objective')
    else:
        opt=minimize(objective,uniform,jac=gradient,bounds=[(0,1)]*m,
            constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(m))],
            method='SLSQP',options=dict(maxiter=1000,ftol=1e-12))
    assert opt.success and np.isfinite(opt.x).all() and abs(opt.x.sum()-1)<1e-8,str(opt.message)
    x=voters.reshape(n,m*c);y=target.copy();xmean=np.zeros(m*c);xscale=np.ones(m*c);ymean=np.zeros(c);yscale=np.ones(c)
    if c==1:
        xmean=np.average(x,axis=0,weights=weights);xscale=np.sqrt(np.average((x-xmean)**2,axis=0,weights=weights));xscale[xscale<1e-12]=1
        ymean=np.average(y,axis=0,weights=weights);yscale=np.sqrt(np.average((y-ymean)**2,axis=0,weights=weights));yscale[yscale<1e-12]=1
    model=Ridge(alpha=1.,fit_intercept=True).fit((x-xmean)/xscale,(y-ymean)/yscale,
        sample_weight=data['weights'])
    fit=dict(utc=utc(),input_sha256=sha(input_file),rows=n,groups=len(set(data['groups'].tolist())),
        source_targets_only=True,base_models_refitted=0,actual_supervised_combiners_fitted=2,
        simplex=dict(weights=opt.x.tolist(),initial_objective=objective(uniform),objective=objective(opt.x),
            gradient=gradient(opt.x).tolist(),iterations=int(opt.nit),success=bool(opt.success),scale=scale,
            solver='all_simplex_faces_same_objective' if exact else 'SLSQP'),
        ridge=dict(coef=np.asarray(model.coef_).tolist(),intercept=np.asarray(model.intercept_).tolist(),
            xmean=xmean.tolist(),xscale=xscale.tolist(),ymean=ymean.tolist(),yscale=yscale.tolist(),
            n_features=int(model.n_features_in_),alpha=1.,parameter_norm=float(np.linalg.norm(model.coef_))),
        seconds=time.monotonic()-start)
    save(output/'MODELS.json',fit)


def fit(root):
    cfg=read(root/'PROTOCOL447.json');cases=read(root/'CASES447.json')
    for p,digest in read(root/'LAUNCH_FREEZE447.json').items():assert sha(root/p)==digest
    def one(case):
        name=case['id'];out=root/'fits'/name;out.mkdir(parents=True);(out/'tmp/mpl').mkdir(parents=True)
        cmd=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
        for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
            if Path(p).exists():cmd+=['--ro-bind',p,p]
        for p,q in [(PYTHON,'/opt/python'),(DEPS,'/opt/deps'),(root/'code','/code'),(root/'legal'/(name+'.npz'),'/input.npz')]:cmd+=['--ro-bind',str(p),q]
        cmd+=['--bind',str(out),'/output','--proc','/proc','--dev','/dev','--chdir','/code','--remount-ro','/',
              '/opt/python/bin/python3.12','-B','/code/screen_learned_combinations.py','--stage','fit_one','--run','/output','--input','/input.npz']
        env=dict(PATH='/opt/python/bin:/usr/bin:/bin',PYTHONPATH='/opt/deps',PYTHONDONTWRITEBYTECODE='1',
                 TMPDIR='/output/tmp',TMP='/output/tmp',TEMP='/output/tmp',MPLCONFIGDIR='/output/tmp/mpl',
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=cfg['process_seconds'])
        save(out/'PROCESS.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,command=cmd))
        assert p.returncode==0,p.stderr[-2000:]
        return dict(case=name,sha256=sha(out/'MODELS.json'),actual_combiners=2)
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=cfg['fit_workers']) as pool:models=list(pool.map(one,cases))
    save(root/'ALL_FITS_SEALED447.json',dict(utc=utc(),models=models,actual_combiners=24,fit_processes=12,
        seconds=time.monotonic()-start,external_outcomes_joined=False,new_LLM_calls=0))
    print(json.dumps(dict(fitted_combiners=24,seconds=time.monotonic()-start)),flush=True)


def external(case,bound):
    archive=Path(case['archive']);ids=case['candidate_ids'];source=case['source']
    if source in ['rotor','rul']:
        scores=bound.json(archive,'FINAL_RESULTS328.json')['candidate_scores'][case['id']]
        field='records' if source=='rotor' else 'queries'
        return matrix({cid:[dict(sample_id=str(r.get('sample_id',r.get('group'))),
            group=str(r.get('group',r.get('unit_id'))),actual=r['actual'],predicted=r['predicted'])
            for r in scores[cid][field]] for cid in ids},ids),scores
    if source=='gear':
        src=Path(bound.json(archive,'CAMPAIGN_CONFIG.json')['source_archives']['gear']['path'])
        data=dict(np.load(bound.path(src,'hidden/test.npz'),allow_pickle=False))
        gold={str(i):(int(y),str(g)) for i,y,g in zip(data['sample_ids'],data['labels'],data['groups'])}
        rows={}
        for cid in ids:
            pred=bound.json(src,'evaluation282/'+case['id'].removeprefix('gear_')+'/'+cid+'/test/predictions.json')['predictions']
            rows[cid]=[dict(sample_id=r['sample_id'],actual=gold[r['sample_id']][0],group=gold[r['sample_id']][1],predicted=r['predicted_label']) for r in pred]
        scores=bound.json(archive,'FINAL_RESULTS328.json')['candidate_scores'][case['id']]
    elif source=='electric_motor':
        prefix='source_views/motor/';allocation=case['id'][-1]
        gold={r['sample_id']:r for r in bound.json(archive,prefix+allocation+'/hidden/ELECTRICMOTORVIBRATIONS/labels.json')}
        rows={cid:[dict(sample_id=r['sample_id'],actual=gold[r['sample_id']]['label'],group=gold[r['sample_id']]['group'],predicted=r['predicted_label'])
                   for r in bound.json(archive,prefix+'controls/'+case['id']+'/'+cid+'_train_external_predictions.json')['predictions']] for cid in ids}
        scores=bound.json(archive,'FINAL_RESULTS329.json')['scores'][case['id']]
    else:
        prefix='source_views/gear/';catalog=bound.json(archive,'G2_CANDIDATE_BINDING.json')
        data=dict(np.load(bound.path(archive,prefix+'hidden/test.npz'),allow_pickle=False))
        stacks=[]
        for name in catalog['ids'][:6]:
            p=bound.json(archive,prefix+'evaluation282/basis/'+name+'/test/probabilities.json')
            assert p['classes']==list(range(6))
            mapping={str(k):i for i,k in enumerate(p['sample_ids'])}
            stacks.append(np.asarray(p['probabilities'])[[mapping[str(i)] for i in data['sample_ids']]])
        stacks=np.stack(stacks,axis=1);rows={}
        for cid,w in zip(catalog['ids'],catalog['weights']):
            pred=np.einsum('nmc,m->nc',stacks,np.asarray(w)/sum(w)).argmax(1)
            rows[cid]=[dict(sample_id=str(i),actual=int(y),group=str(g),predicted=int(p)) for i,y,g,p in zip(data['sample_ids'],data['labels'],data['groups'],pred)]
        scores=bound.json(archive,'FINAL_RESULTS329.json')['scores'][case['id']]
    return matrix(rows,ids),scores


def metric(case,data,prediction):
    truth=data['y'];groups=data['groups']
    if case['source'] in ['gear','gear_g2']:
        terms=[]
        for label in range(case['class_count']):
            tp=int(((truth==label)&(prediction==label)).sum())
            denominator=int((truth==label).sum()+(prediction==label).sum())
            terms.append(2*tp/denominator if denominator else 0.)
        return 1-float(np.mean(terms))
    error=np.abs(truth-prediction) if case['source']=='rul' else truth!=prediction
    return float(np.mean([np.mean(error[groups==g]) for g in np.unique(groups)]))


def apply_models(data,model,incumbent):
    voters,_=features(data);n,m,c=voters.shape
    ridge=model['ridge'];flat=voters.reshape(n,-1)
    scores=((flat-ridge['xmean'])/ridge['xscale'])@np.asarray(ridge['coef']).T+ridge['intercept']
    scores=(scores*ridge['yscale']+ridge['ymean']).reshape(n,c)
    outputs=dict(simplex_least_squares=np.einsum('nmc,m->nc',voters,model['simplex']['weights']),
                 ridge_calibration=scores,equal_vote_or_mean=voters.mean(1))
    for name,pred in outputs.items():
        if int(data['class_count']):
            tied=pred>=pred.max(1,keepdims=True)-1e-12
            labels=pred.argmax(1);old=data['predictions'][:,incumbent].astype(int)
            labels[tied[np.arange(n),old]]=old[tied[np.arange(n),old]]
            outputs[name]=labels
        else:outputs[name]=np.maximum(pred[:,0],0.)
    return outputs


def evaluate(root):
    cfg=read(root/'PROTOCOL447.json');seal=read(root/'ALL_FITS_SEALED447.json')
    cases=read(root/'CASES447.json');ordinary=read(root/'ORDINARY_CHOICES447.json');bound=Bound();rows=[]
    for case in cases:
        path=root/'fits'/case['id']/'MODELS.json'
        assert sha(path)==next(r['sha256'] for r in seal['models'] if r['case']==case['id'])
        data,old=external(case,bound);ids=case['candidate_ids'];anchor=ids.index(case['original_incumbent'])
        data['class_count']=np.asarray(case['class_count'])
        base=[metric(case,data,data['predictions'][:,i]) for i in range(len(ids))]
        assert all(abs(v-old[cid]['loss'])<1e-8 for cid,v in zip(ids,base)),case['id']
        outputs=apply_models(data,read(path),anchor);outputs['keep']=data['predictions'][:,anchor]
        losses=np.asarray(case['group_losses']);dev=case['development_columns'];cv=[i for i in range(losses.shape[1]) if i not in dev]
        for name,columns in [('development_best',dev),('CV_best',cv)]:
            values=losses[:,columns].mean(1);tied=np.flatnonzero(values<=values.min()+1e-12)
            selected=anchor if anchor in tied else int(tied[0]);outputs[name]=data['predictions'][:,selected]
        for name in ['ordinary_1','ordinary_2']:outputs[name]=data['predictions'][:,ids.index(ordinary[case['id']+'::'+name])]
        span=max(base)-min(base);span=span if span>1e-12 else 1.
        for method,pred in outputs.items():
            loss=metric(case,data,pred);delta=(loss-base[anchor])/span
            rows.append(dict(case=case['id'],cluster=case['cluster'],method=method,loss=loss,
                original_loss=base[anchor],original_bank_span=span,normalized_loss=loss/span,
                normalized_change=delta,harm=max(0,delta),gain=max(0,-delta)))
        np.savez_compressed(root/('PREDICTIONS_'+case['id']+'.npz'),sample_ids=data['sample_ids'],groups=data['groups'],actual=data['y'],**outputs)
    summaries={}
    for method in cfg['methods']:
        by={}
        for cluster in sorted(set(cfg['clusters'].values())):
            rr=[r for r in rows if r['method']==method and r['cluster']==cluster]
            by[cluster]={k:float(np.mean([r[k] for r in rr])) for k in ['loss','normalized_loss','normalized_change','harm','gain']}
        summaries[method]=dict(clusters=by,**{k:float(np.mean([v[k] for v in by.values()])) for k in ['normalized_change','harm','gain']})
    gates={}
    for name in cfg['methods'][:2]:
        m=summaries[name];gates[name]={}
        for control in ['ordinary_1','ordinary_2','equal_vote_or_mean']:
            other=summaries[control]
            changes=[m['clusters'][c]['normalized_loss']-other['clusters'][c]['normalized_loss'] for c in m['clusters']]
            gates[name][control]=dict(wins=sum(d < -1e-10 for d in changes),ties=sum(abs(d)<=1e-10 for d in changes),
                losses=sum(d>1e-10 for d in changes),passed=sum(d < -1e-10 for d in changes)>=3
                    and m['harm']<=other['harm']+1e-10 and m['gain']>=other['gain']-1e-10)
    save(root/'OUTCOME_BINDINGS447.json',list(bound.rows.values()))
    save(root/'RESULT447.json',dict(utc=utc(),rows=rows,summaries=summaries,gates=gates,
        capability_passed=any(all(v['passed'] for v in controls.values()) for controls in gates.values()),
        independent_confirmation=False,new_Agent_comparison=False,new_combiners_fitted=24,new_base_fits=0,new_LLM_calls=0))
    print(json.dumps(dict(gates=gates,summaries={k:{a:v[a] for a in ['normalized_change','harm','gain']} for k,v in summaries.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--input',type=Path)
    p.add_argument('--stage',required=True,choices=['prepare','fit','fit_one','fit_one_exact','evaluate']);a=p.parse_args()
    if a.stage in ['fit_one','fit_one_exact']:fit_one(a.input,a.run,exact=a.stage=='fit_one_exact')
    else:globals()[a.stage](a.run.resolve())
