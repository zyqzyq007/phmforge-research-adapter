"""Independent accounting, split, isolation and numerical audit of A459."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def audit(root):
    started=time.monotonic();checks=[]
    def check(ok,label):
        checks.append(dict(label=label,passed=bool(ok)))
    cfg=read(root/'PROTOCOL459.json')
    for p,d in read(root/'LAUNCH_FREEZE459.json').items():check(sha(root/p)==d,'frozen:'+p)
    for r in cfg['resources']:check(sha(root/r['target'])==r['sha256'],'resource:'+r['target'])
    check(sha(cfg['train'])==cfg['train_sha256'],'source_training_SHA')
    train=load(cfg['train']);query=load(root/'inputs/training_queries.npz')
    truth={str(g):float(v) for g,v in zip(query['groups'],query['y']+query['age'])}
    check(len(truth)==47 and set(truth)==set(train['groups']),'all_source_47')
    for group,target in truth.items():
        check(np.allclose((train['y']+train['age'])[train['groups']==group],target,atol=1e-8,rtol=0),'source_label:'+group)
    folds=read(root/'SPLITS459.json');tasks=read(root/'FIT_TASKS459.json');base=read(root/'BASE_SEALED459.json')
    check(base['passed'] and len(tasks)==36 and len(base['rows'])==376,'full_base_denominator')
    outer={f['outer']:set(f['heldout_cells']) for f in folds};by_id={t['id']:t for t in tasks}
    completed_estimator_fits=0;optimizer_steps=0
    for t in tasks:
        z=load(root/t['input']);groups=set(z['groups']);out=root/'fits'/t['id']
        process=read(out/'PROCESS.json');a=read(out/'training_audit.json')
        check(sha(root/t['input'])==t['input_sha256'],'fit_input_SHA:'+t['id'])
        check(groups==set(t['fitting_cells']) and groups.isdisjoint(set(t['heldout_cells'])|outer[t['outer']]),'whole_cell_split:'+t['id'])
        ix=next(i for f in folds if f['outer']==t['outer'] for i in f['inner'] if i['inner']==t['inner'])['fit_indices']
        check(all(np.array_equal(z[k],v[ix]) for k,v in train.items()),'actual_source_rows:'+t['id'])
        check(process['returncode']==0 and a['status']=='completed' and a['completed_estimator_fits']>0,'real_fit:'+t['id'])
        binds={process['command'][i+2]:process['command'][i+1] for i,v in enumerate(process['command']) if v=='--ro-bind'}
        check('--unshare-net' in process['command'] and set(binds)=={'/usr','/bin','/sbin','/lib','/lib64','/etc','/opt/python312','/opt/deps','/tools','/program','/input.npz'}
              and binds['/input.npz']==str(root/t['input']) and binds['/program']==str(root/'original'/t['program']/'source'),'fit_isolation:'+t['id'])
        completed_estimator_fits+=a['completed_estimator_fits'];optimizer_steps+=a['optimizer_steps']
    component_calls=0
    for r in base['rows']:
        t=by_id[r['fit_id']];out=root/'inner_predictions'/r['id'];raw=read(out/'COMPONENT_TRACE.json')
        a=read(out/'training_audit.json');proc=read(out/'PROCESS.json')
        z=load(root/'queries'/(r['sample_id']+'.npz'));mask=query['sample_ids']==r['sample_id']
        check(r['group'] in t['heldout_cells'] and r['group'] not in t['fitting_cells'] and r['group'] not in outer[r['outer']],'inner_disjoint:'+r['id'])
        check(set(z)=={'x','nn','past_cap','age','rates','cycle','groups','sample_ids'} and all(np.array_equal(z[k],query[k][mask]) for k in z),'label_free_actual_query:'+r['id'])
        check(a['completed_estimator_fits']==a['optimizer_steps']==0 and proc['returncode']==0,'cold_inference:'+r['id'])
        check(sha(out/'COMPONENT_TRACE.json')==r['trace_sha256'] and np.isfinite(r['x']).all(),'prediction_SHA:'+r['id'])
        vals={v['name']:float(np.asarray(v['prediction']).ravel()[0]) for v in raw['calls']}
        spec=next(s for s in cfg['programs'] if s['name']==r['program'])
        names=[k for k in vals if k.startswith('models.')] if spec['kind']=='success' else spec['components']
        x=[vals[n]*(1000 if n=='models.svr' else 1) for n in names]
        check(np.allclose(x,r['x'],atol=1e-8,rtol=1e-10),'component_vector:'+r['id'])
        component_calls+=len(raw['calls'])
    meta=read(root/'META_FITS_SEALED459.json');check(meta['passed'] and len(meta['rows'])==12,'meta_denominator')
    for t in meta['rows']:
        z=load(root/t['input']);out=root/'meta_fits'/t['id'];m=read(out/'MODEL.json')
        wanted=next(f['fitting_cells'] for f in folds if f['outer']==t['outer'])
        rows=sorted([r for r in base['rows'] if r['program']==t['program'] and r['outer']==t['outer']],key=lambda r:r['group'])
        check(z['groups'].tolist()==wanted and set(z['groups']).isdisjoint(outer[t['outer']]),'meta_disjoint:'+t['id'])
        check(np.array_equal(z['x'],np.array([r['x'] for r in rows])) and np.array_equal(z['y'],np.array([truth[g] for g in wanted])),'meta_actual_training:'+t['id'])
        w=np.array(m['coefficient']);b=m['intercept'];x,y=z['x'],z['y'];xc=x-x.mean(0);yc=y-y.mean()
        grad=xc.T@(xc@w-yc)+12*w;pg=np.where(w>1e-8,grad,np.minimum(grad,0))
        kkt=np.linalg.norm(pg,np.inf)/max(1,np.linalg.norm(xc.T@yc,np.inf))
        check(np.all(w>=0) and kkt<1e-10 and abs(b-(y.mean()-x.mean(0)@w))<1e-8,'convex_solution:'+t['id'])
        check(m['fit_calls']==1 and m['input_sha256']==sha(root/t['input']) and read(out/'PROCESS.json')['returncode']==0,'actual_meta_fit:'+t['id'])
    cached=read(root/'CACHED459.json');pred=read(root/'SOURCE_PREDICTIONS_SEALED459.json');choices=read(root/'CHOICES_SEALED459.json')
    check(len(pred['rows'])==188 and len(choices['choices'])==4,'validation_denominator')
    errors={}
    for r in pred['rows']:
        raw=next(t for t in cached[r['program']] if t['task']['sample_id']==r['sample_id'])
        m=read(root/'meta_fits'/(r['program']+'__outer_'+str(r['outer']))/'MODEL.json')
        life=sum(a*b for a,b in zip(raw['x'],m['coefficient']))+m['intercept'];a,offset,scale,add,cap=raw['params']
        result=scale*max(a*life+offset-raw['age'],0)+add
        if cap>0:result=min(result,cap)
        check(abs(result-r['readout'])<1e-6 and abs(raw['original']-r['original'])<1e-8,'outer_readout:'+r['program']+':'+r['sample_id'])
        e=errors.setdefault(r['program'],{'original':[],'readout':[]})
        for k in e:e[k].append(abs(r[k]-(truth[r['group']]-raw['age'])))
    for c in choices['choices']:
        e=errors[c['program']];old=np.mean(e['original']);new=np.mean(e['readout'])
        check(np.isclose(c['original_CV_MAE'],old) and np.isclose(c['readout_CV_MAE'],new),'source_score:'+c['program'])
        check(c['selected']==('readout' if new<old-1e-8 else 'original'),'source_only_choice:'+c['program'])
    check(not choices['external_targets_joined'] and not choices['confirmation_opened'],'source_choice_scope')
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),failed=[c for c in checks if not c['passed']],
        completed_base_fit_entries=36,actual_completed_estimator_fit_events=completed_estimator_fits,
        optimizer_steps=optimizer_steps,readout_fit_entries=12,inner_cold_processes=376,
        qualification_cold_processes=4,inner_component_predict_calls=component_calls,
        seconds=time.monotonic()-started,independent_confirmation=0)
    (root/'AUDIT459.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'AUDIT_CHECKS459.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert result['passed'],'Preserve audit failures'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    audit(p.parse_args().run.resolve())
