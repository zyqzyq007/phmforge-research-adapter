"""Whole-cell nested validation of a frozen component readout intervention.

Base programs are trained without the outer validation cells, including the
inner out-of-fold features used to train the readout. Authored code generation
is historical and is explicitly not nested by this experiment.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import numpy as np

from controlled_feedback import save, sha, utc
from program_revision_runtime import Runtime, PYTHON, DEPS, load, subset
from fit_component_readouts import describe, compose


TRACE_WORKER = r'''
import json,runpy,sys
from pathlib import Path
import joblib
import numpy as np
assert sys.argv[1]=='predict'
args=sys.argv
input_path=Path(args[args.index('--input')+1])
output=Path(args[args.index('--output')+1])
with np.load(input_path,allow_pickle=False) as z:
    assert set(z.files)=={'x','nn','past_cap','age','rates','cycle','groups','sample_ids'}
    assert len(z['age'])==1
    age=float(z['age'][0])
assert not Path('/data').exists() and not Path('/srv').exists()
assert not Path('/home/jupyter-yyf-zyq/.bashrc').exists()
calls=[];metadata={}
class Observed:
    def __init__(self,model,name): self.model,self.name=model,name
    def __getattr__(self,name): return getattr(self.model,name)
    def predict(self,*args,**kwargs):
        result=self.model.predict(*args,**kwargs)
        calls.append({'name':self.name,'prediction':np.asarray(result).tolist()})
        return result
def wrap(value,name=''):
    if isinstance(value,dict):
        return {k:wrap(v,name+'.'+str(k) if name else str(k)) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        values=[]
        for i,v in enumerate(value):
            if isinstance(v,tuple) and len(v)==2 and isinstance(v[0],str):
                values.append((v[0],wrap(v[1],name+'.'+v[0])))
            else: values.append(wrap(v,name+'.'+str(i)))
        return tuple(values) if isinstance(value,tuple) else values
    if callable(getattr(value,'predict',None)): return Observed(value,name)
    return value
original_load=joblib.load
def traced_load(*args,**kwargs):
    bundle=original_load(*args,**kwargs)
    if isinstance(bundle,dict) and 'choice' in bundle: metadata['choice']=bundle['choice']
    return wrap(bundle)
joblib.load=traced_load
runpy.run_path(str(Path(__file__).with_name('original_main.py')),run_name='__main__')
joblib.load=original_load
assert len(calls)==len({c['name'] for c in calls})
prediction=json.loads(output.read_text())['rul']
output.with_name('COMPONENT_TRACE.json').write_text(json.dumps({
    'age':age,'calls':calls,'original':prediction,'metadata':metadata,
    'query_targets_visible':False,'authored_prediction_code_modified':False},indent=2)+'\n')
'''


def read(p):
    return json.loads(Path(p).read_text())


def restore(root, protocol_name='PROTOCOL459.json', receipt_name='RESTORED459.json'):
    cfg = read(root / protocol_name)
    grouped = defaultdict(list)
    for row in cfg['resources']:
        grouped[row['archive']].append(row)
    def one(item):
        archive, rows = item
        archive = Path(archive)
        assert sha(archive / 'SOURCE_MANIFEST.json') == cfg['archives'][str(archive)]
        index = {r['path']: r for r in read(archive / 'SOURCE_MANIFEST.json')['files']}
        missing = {}
        for row in rows:
            assert row['sha256'] == index[row['member']]['sha256']
            dst = root / row['target']
            dst.parent.mkdir(parents=True, exist_ok=True)
            assert not dst.exists()
            src = archive / row['member']
            if src.is_file():
                shutil.copyfile(src, dst)
            else:
                missing[row['member']] = row
        if missing:
            with tarfile.open(archive / 'RUN_CONTENTS.tar.gz', 'r|gz') as stream:
                for entry in stream:
                    name = entry.name.removeprefix('./')
                    if name not in missing:
                        continue
                    row = missing.pop(name)
                    assert entry.isfile() and '..' not in Path(name).parts
                    (root / row['target']).write_bytes(stream.extractfile(entry).read())
                    if not missing:
                        break
        assert not missing
        for row in rows:
            dst = root / row['target']
            assert sha(dst) == row['sha256'] and dst.stat().st_size == row['bytes']
        return len(rows)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        count = sum(pool.map(one, grouped.items()))
    save(root / receipt_name, dict(passed=True, resources=count,
        seconds=time.monotonic()-started, utc=utc(), external_targets_read=False))
    print(json.dumps(read(root / receipt_name)), flush=True)


def trace_description(spec, raw):
    if spec['kind'] != 'success':
        return describe(spec, raw)
    values = {r['name']: float(np.asarray(r['prediction']).ravel()[0]) for r in raw['calls']}
    names = [r['name'] for r in raw['calls'] if r['name'].startswith('models.')]
    assert len(names) == 7 and 'models.svr' in names
    base = np.array([values[n] * (1000. if n == 'models.svr' else 1.) for n in names])
    converted = dict(base_lifetimes=[base.tolist()], age=[raw['age']],
        stack=[values['stack']], median=[float(np.median(base))],
        consensus=[float(.8*np.median(base)+.2*base.mean())],
        hgb=[values['direct_hgb']], extra_trees=[values['direct_et']],
        choice=raw['metadata']['choice'], variants={'original':raw['original']})
    return describe(spec, converted)


def cached_rows(root, spec):
    success = spec['kind'] == 'success'
    tasks = read(root / 'cached' / ('reused/TASKS446.json' if success else 'TASKS.json'))
    rows = []
    for t in tasks:
        if t['context'] == 'full' or (not success and t['program'] != spec['name']):
            continue
        path = root/'cached'/('reused/predictions' if success else 'predictions')/t['id']
        path /= 'COMPONENTS.json' if success else 'COMPONENT_TRACE.json'
        x, age, params, original = describe(spec, read(path))
        rows.append(dict(task=t, x=x.tolist(), age=age, params=params.tolist(),
            original=original, trace=str(path.relative_to(root)), trace_sha256=sha(path)))
    assert len(rows) == 47
    return rows


def prepare(root):
    from sklearn.model_selection import GroupKFold
    cfg = read(root/'PROTOCOL459.json')
    assert read(root/'RESTORED459.json')['passed'] and sha(cfg['train']) == cfg['train_sha256']
    started = time.monotonic()
    train, query = load(cfg['train']), load(root/'inputs/training_queries.npz')
    assert len(set(query['groups'])) == 47 and np.all(query['age'] == 499)
    assert set(query['groups']) == set(train['groups'])
    jobs = read(root/'inputs/JOBS366.json')
    fields = ['x','nn','past_cap','age','rates','cycle','groups','sample_ids']
    all_queries = [t for t in read(root/'cached/reused/TASKS446.json') if t['context'] != 'full']
    for t in all_queries:
        mask = query['sample_ids'] == t['sample_id']; assert mask.sum() == 1
        actual = {k: query[k][mask] for k in fields}; old = load(root/'cached'/t['input'])
        assert set(old) == set(actual) and all(np.array_equal(actual[k],old[k]) for k in fields)
        p = root/'queries'/(t['sample_id']+'.npz'); p.parent.mkdir(exist_ok=True)
        np.savez(p, **actual)
    folds=[];tasks=[]
    for outer, (fit_idx, val_idx) in enumerate(GroupKFold(3).split(train['x'],groups=train['groups'])):
        previous = jobs['fold_'+str(outer)]
        assert fit_idx.tolist()==previous['fit_indices'] and val_idx.tolist()==previous['validation_indices']
        outer_fit = subset(train, fit_idx)
        o = dict(outer=outer,fit_indices=fit_idx.tolist(),validation_indices=val_idx.tolist(),
            fitting_cells=sorted(set(train['groups'][fit_idx])),heldout_cells=sorted(set(train['groups'][val_idx])),inner=[])
        assert set(o['fitting_cells']).isdisjoint(o['heldout_cells'])
        for inner,(fi,vi) in enumerate(GroupKFold(3).split(outer_fit['x'],groups=outer_fit['groups'])):
            base_fit=fit_idx[fi];base_val=fit_idx[vi]
            f=dict(inner=inner,fit_indices=base_fit.tolist(),validation_indices=base_val.tolist(),
                fitting_cells=sorted(set(train['groups'][base_fit])),heldout_cells=sorted(set(train['groups'][base_val])))
            assert set(f['fitting_cells']).isdisjoint(set(f['heldout_cells'])|set(o['heldout_cells']))
            p=root/'inputs'/('outer_'+str(outer)+'_inner_'+str(inner)+'.npz')
            np.savez(p,**subset(train,base_fit));f['input']=str(p.relative_to(root));f['sha256']=sha(p)
            o['inner'].append(f)
            for spec in cfg['programs']:
                tasks.append(dict(id=spec['name']+'__o'+str(outer)+'_i'+str(inner),program=spec['name'],
                    kind=spec['kind'],outer=outer,inner=inner,input=f['input'],input_sha256=f['sha256'],
                    fitting_cells=f['fitting_cells'],heldout_cells=f['heldout_cells']))
        folds.append(o)
    for spec in cfg['programs']:
        dst=root/'tracers'/spec['name']
        shutil.copytree(root/'original'/spec['name']/'source',dst)
        shutil.copyfile(dst/'program.py',dst/'original_main.py')
        (dst/'program.py').write_text(TRACE_WORKER)
    save(root/'SPLITS459.json',folds);save(root/'FIT_TASKS459.json',tasks)
    save(root/'CACHED459.json',{s['name']:cached_rows(root,s) for s in cfg['programs']})
    save(root/'ENGINE_CONFIG.json',dict(input_root=str(root/'inputs'),common_tools=str(root/'code'),
        fit_seconds=cfg['fit_seconds'],predict_seconds=cfg['predict_seconds']))
    assert len(tasks)==36 and sum(len(t['heldout_cells']) for t in tasks)==376
    for name in ['validate_component_readouts.py','fit_component_readouts.py',
                 'controlled_feedback.py','program_revision_runtime.py','program_training_entry.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'code'/name)
    paths=[p for folder in ['inputs','queries','tracers','original','cached','code'] for p in (root/folder).rglob('*') if p.is_file()]
    paths += [root/n for n in ['PROTOCOL459.json','SPLITS459.json','FIT_TASKS459.json','CACHED459.json','ENGINE_CONFIG.json']]
    save(root/'LAUNCH_FREEZE459.json',{str(p.relative_to(root)):sha(p) for p in paths})
    save(root/'PREPARED459.json',dict(passed=True,seconds=time.monotonic()-started,
        base_fit_tasks=36,inner_prediction_tasks=376,source_cells=47,utc=utc(),external_targets_read=False))
    print(json.dumps(read(root/'PREPARED459.json')),flush=True)


def qualification(root):
    cfg=read(root/'PROTOCOL459.json');cached=read(root/'CACHED459.json')
    for p,d in read(root/'LAUNCH_FREEZE459.json').items(): assert sha(root/p)==d
    def one(spec):
        name=spec['name'];row=next(r for r in cached[name] if r['task']['context']=='fold_0')
        out=root/'qualification'/name
        try:
            process,audit=Runtime(root).command(root/'tracers'/name,root/'original'/name/'fold_0/artifacts',
                root/'queries'/(row['task']['sample_id']+'.npz'),out,'predict')
            actual=trace_description(spec,read(out/'COMPONENT_TRACE.json'))
            assert all(np.allclose(a,b,rtol=1e-9,atol=1e-6) for a,b in zip(actual,[row['x'],row['age'],row['params'],row['original']]))
            return dict(program=name,status='complete',seconds=process['seconds'],
                component_calls=len(read(out/'COMPONENT_TRACE.json')['calls']),new_fits=0)
        except Exception as e:
            return dict(program=name,status='error',error_type=type(e).__name__,error=str(e))
    with ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(one,cfg['programs']))
    passed=all(r['status']=='complete' for r in rows)
    save(root/'QUALIFICATION459.json',dict(passed=passed,rows=rows,utc=utc(),cold_predictions=len(rows)))
    print(json.dumps(read(root/'QUALIFICATION459.json')),flush=True)
    assert passed,'Retain qualification errors; do not launch training'


def base(root):
    cfg=read(root/'PROTOCOL459.json');tasks=read(root/'FIT_TASKS459.json')
    assert read(root/'QUALIFICATION459.json')['passed'] and not (root/'BASE_HANDLE459.json').exists()
    for p,d in read(root/'LAUNCH_FREEZE459.json').items(): assert sha(root/p)==d
    start=time.monotonic();deadline=start+cfg['run_seconds']
    save(root/'BASE_HANDLE459.json',dict(state='running',pid=os.getpid(),utc=utc()))
    def fit(t):
        try:
            rt=Runtime(root);rt.deadline=deadline
            process,audit=rt.command(root/'original'/t['program']/'source',None,root/t['input'],root/'fits'/t['id'],'fit')
            return dict(**t,status='complete',seconds=process['seconds'],completed_estimator_fits=audit['completed_estimator_fits'],
                        optimizer_steps=audit['optimizer_steps'])
        except Exception as e:
            return dict(**t,status='error',error_type=type(e).__name__,error=str(e))
    def batch(todo,fn,stage):
        records=[]
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
            for f in as_completed([pool.submit(fn,t) for t in todo]):
                row=f.result();records.append(row);save(root/'records'/stage/(row['id']+'.json'),row)
                progress=dict(stage=stage,finished=len(records),total=len(todo),
                    errors=sum(r['status']!='complete' for r in records),seconds=time.monotonic()-start,utc=utc())
                save(root/'PROGRESS459.json',progress)
                if len(records)%4==0 or row['status']!='complete':print(json.dumps(progress),flush=True)
        return records
    fitted=batch(tasks,fit,'fit')
    fit_passed=all(r['status']=='complete' for r in fitted)
    save(root/'BASE_FITS_SEALED459.json',dict(passed=fit_passed,rows=fitted,utc=utc(),seconds=time.monotonic()-start))
    specs={s['name']:s for s in cfg['programs']}
    def predict(t):
        try:
            rt=Runtime(root);rt.deadline=deadline
            out=root/'inner_predictions'/t['id']
            process,audit=rt.command(root/'tracers'/t['program'],root/'fits'/t['fit_id']/'artifacts',
                root/'queries'/(t['sample_id']+'.npz'),out,'predict')
            raw=read(out/'COMPONENT_TRACE.json');x,age,params,original=trace_description(specs[t['program']],raw)
            return dict(**t,status='complete',seconds=process['seconds'],trace_sha256=sha(out/'COMPONENT_TRACE.json'),
                component_calls=len(raw['calls']),x=x.tolist(),age=age,params=params.tolist(),original=original)
        except Exception as e:
            return dict(**t,status='error',error_type=type(e).__name__,error=str(e))
    predicted=[]
    if fit_passed:
        queries=load(root/'inputs/training_queries.npz');by_group=dict(zip(queries['groups'],queries['sample_ids']))
        todo=[dict(id=t['id']+'__'+g,fit_id=t['id'],program=t['program'],outer=t['outer'],inner=t['inner'],
            group=g,sample_id=str(by_group[g])) for t in tasks for g in t['heldout_cells']]
        predicted=batch(todo,predict,'predict')
    passed=fit_passed and len(predicted)==376 and all(r['status']=='complete' for r in predicted)
    save(root/'BASE_SEALED459.json',dict(passed=passed,fit_entries_attempted=len(fitted),
        completed_fit_entries=sum(t['status']=='complete' for t in fitted),prediction_processes=len(predicted),
        rows=predicted,seconds=time.monotonic()-start,utc=utc(),external_targets_read=False))
    save(root/'BASE_HANDLE459.json',dict(state='terminal',passed=passed,pid=os.getpid(),utc=utc()))
    print(json.dumps({k:v for k,v in read(root/'BASE_SEALED459.json').items() if k!='rows'}),flush=True)
    assert passed,'Frozen budget or science failure; retain full denominator, no restart'


def meta_worker(root):
    from scipy.optimize import nnls
    assert not Path('/data').exists() and not Path('/srv').exists()
    z=load('/input.npz');assert set(z)=={'x','y','sample_ids','groups'}
    cfg=read('/protocol.json');x,y=z['x'],z['y'];mu=x.mean(axis=0);ym=y.mean()
    xc=x-mu;yc=y-ym;p=x.shape[1];started=time.monotonic()
    coef,residual=nnls(np.vstack([xc,np.sqrt(cfg['alpha'])*np.eye(p)]),
        np.r_[yc,np.zeros(p)],maxiter=1000)
    intercept=float(ym-mu@coef)
    grad=xc.T@(xc@coef-yc)+cfg['alpha']*coef
    pg=np.where(coef>1e-8,grad,np.minimum(grad,0))
    kkt=float(np.linalg.norm(pg,np.inf)/max(1,np.linalg.norm(xc.T@yc,np.inf)))
    save(root/'MODEL.json',dict(coefficient=coef.tolist(),intercept=intercept,alpha=cfg['alpha'],
        fit_seconds=time.monotonic()-started,fit_calls=1,training_rows=len(y),input_sha256=sha('/input.npz'),
        normalized_projected_gradient=kkt,objective=float(residual**2),utc=utc(),external_targets_visible=False))
    assert kkt<1e-10 and np.isfinite(coef).all()


def meta(root):
    cfg=read(root/'PROTOCOL459.json');base_rows=read(root/'BASE_SEALED459.json')
    assert base_rows['passed'];queries=load(root/'inputs/training_queries.npz')
    truth={str(g):float(v) for g,v in zip(queries['groups'],queries['y']+queries['age'])}
    folds=read(root/'SPLITS459.json');tasks=[]
    for spec in cfg['programs']:
        for fold in folds:
            rows=sorted([r for r in base_rows['rows'] if r['program']==spec['name'] and r['outer']==fold['outer']],key=lambda r:r['group'])
            assert [r['group'] for r in rows]==fold['fitting_cells']
            name=spec['name']+'__outer_'+str(fold['outer']);p=root/'meta_inputs'/(name+'.npz');p.parent.mkdir(exist_ok=True)
            np.savez(p,x=np.array([r['x'] for r in rows]),y=np.array([truth[r['group']] for r in rows]),
                sample_ids=np.array([r['sample_id'] for r in rows]),groups=np.array([r['group'] for r in rows]))
            tasks.append(dict(id=name,program=spec['name'],outer=fold['outer'],input=str(p.relative_to(root)),sha256=sha(p)))
    save(root/'META_INPUT_SEALED459.json',dict(utc=utc(),tasks=tasks,outer_validation_labels_in_fit=False))
    started=time.monotonic()
    def one(t):
        out=root/'meta_fits'/t['id'];out.mkdir(parents=True,exist_ok=False);(out/'tmp/mpl').mkdir(parents=True)
        cmd=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
        for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
            if Path(p).exists():cmd+=['--ro-bind',p,p]
        for src,dst in [(PYTHON,'/opt/python'),(DEPS,'/opt/deps'),(root/'code','/code'),
                        (root/t['input'],'/input.npz'),(root/'PROTOCOL459.json','/protocol.json')]:
            cmd+=['--ro-bind',str(src),dst]
        cmd+=['--bind',str(out),'/output','--proc','/proc','--dev','/dev','--chdir','/code','--remount-ro','/',
            '/opt/python/bin/python3.12','-B','/code/validate_component_readouts.py','--run','/output','--stage','meta_worker']
        env=dict(PATH='/opt/python/bin:/usr/bin:/bin',PYTHONPATH='/opt/deps:/code',PYTHONDONTWRITEBYTECODE='1',
            TMPDIR='/output/tmp',TMP='/output/tmp',TEMP='/output/tmp',MPLCONFIGDIR='/output/tmp/mpl',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2')
        try:
            p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=cfg['meta_seconds'])
            record=dict(**t,status='complete' if p.returncode==0 else 'error',returncode=p.returncode,
                        stdout=p.stdout,stderr=p.stderr,command=cmd,utc=utc())
        except Exception as e:
            record=dict(**t,status='error',error_type=type(e).__name__,error=str(e),command=cmd,utc=utc())
        save(out/'PROCESS.json',record);return record
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool: rows=list(pool.map(one,tasks))
    passed=all(t['status']=='complete' for t in rows)
    save(root/'META_FITS_SEALED459.json',dict(passed=passed,rows=rows,seconds=time.monotonic()-started,utc=utc()))
    print(json.dumps(dict(stage='meta',passed=passed,fit_entries=len(rows),seconds=time.monotonic()-started)),flush=True)
    assert passed,'Retain meta training failures; no automatic retry'


def choose(root):
    cfg=read(root/'PROTOCOL459.json');assert read(root/'META_FITS_SEALED459.json')['passed']
    cached=read(root/'CACHED459.json');predictions=[]
    for spec in cfg['programs']:
        for row in cached[spec['name']]:
            fold=int(row['task']['context'].split('_')[-1])
            model_path=root/'meta_fits'/(spec['name']+'__outer_'+str(fold))/'MODEL.json'
            model=read(model_path);lifetime=np.array([np.asarray(row['x'])@np.array(model['coefficient'])+model['intercept']])
            pred=float(compose(lifetime,np.array([row['age']]),np.array([row['params']]))[0])
            predictions.append(dict(program=spec['name'],sample_id=row['task']['sample_id'],group=row['task']['group'],
                outer=fold,readout=pred,original=row['original'],model_sha256=sha(model_path),trace_sha256=row['trace_sha256']))
    assert len(predictions)==188
    save(root/'SOURCE_PREDICTIONS_SEALED459.json',dict(utc=utc(),rows=predictions,source_targets_joined=False))
    query=load(root/'inputs/training_queries.npz');truth=dict(zip(query['sample_ids'],query['y']))
    choices=[]
    for spec in cfg['programs']:
        rows=[r for r in predictions if r['program']==spec['name']]
        errors={k:np.array([abs(r[k]-truth[r['sample_id']]) for r in rows]) for k in ['original','readout']}
        old,new=float(errors['original'].mean()),float(errors['readout'].mean())
        choices.append(dict(program=spec['name'],selected='readout' if new<old-1e-8 else 'original',
            original_CV_MAE=old,readout_CV_MAE=new,fold_MAE={str(f):{k:float(np.mean([abs(r[k]-truth[r['sample_id']]) for r in rows if r['outer']==f])) for k in errors} for f in range(3)},
            source_cells=len(rows)))
    save(root/'CHOICES_SEALED459.json',dict(utc=utc(),choices=choices,source_predictions_sha256=sha(root/'SOURCE_PREDICTIONS_SEALED459.json'),
        external_targets_joined=False,confirmation_opened=False))
    print(json.dumps(read(root/'CHOICES_SEALED459.json')),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--stage', required=True)
    a = p.parse_args()
    globals()[a.stage](a.run.resolve())
