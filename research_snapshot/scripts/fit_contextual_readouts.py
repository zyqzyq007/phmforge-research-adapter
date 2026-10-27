"""Matched branch-aware objective control on real nested component traces."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from controlled_feedback import save, sha, utc
from program_revision_runtime import PYTHON, DEPS, load
from fit_component_readouts import compose
from validate_component_readouts import restore


def read(p): return json.loads(Path(p).read_text())


def prepare(root):
    cfg=read(root/'PROTOCOL460.json')
    restore(root,'PROTOCOL460.json','RESTORED460.json')
    base=read(root/'nested/BASE_SEALED459.json');cached=read(root/'nested/CACHED459.json')
    q=load(root/'nested/inputs/training_queries.npz')
    y={str(s):float(v) for s,v in zip(q['sample_ids'],q['y'])};cases=[];tasks=[]
    for spec in cfg['programs']:
        name=spec['name']
        for fold in range(4):
            context='full' if fold==3 else 'outer_'+str(fold)
            if context=='full':
                source=[dict(x=r['x'],age=r['age'],params=r['params'],sample_id=r['task']['sample_id'],group=r['task']['group']) for r in cached[name]]
                queries=load(root/'full/queries'/(name+'.npz'))
                old_model=root/'full/numeric_reference'/name/'MODEL.json'
            else:
                source=[r for r in base['rows'] if r['program']==name and r['outer']==fold]
                target=[r for r in cached[name] if r['task']['context']=='fold_'+str(fold)]
                queries=dict(x=np.array([r['x'] for r in target]),age=np.array([r['age'] for r in target]),
                    params=np.array([r['params'] for r in target]),sample_ids=np.array([r['task']['sample_id'] for r in target]))
                old_model=root/'nested/meta_fits'/(name+'__'+context)/'MODEL.json'
            source=sorted(source,key=lambda r:r['sample_id']);case_id=name+'__'+context
            data=dict(x=np.array([r['x'] for r in source]),age=np.array([r['age'] for r in source]),
                params=np.array([r['params'] for r in source]),y=np.array([y[r['sample_id']] for r in source]),
                sample_ids=np.array([r['sample_id'] for r in source]),groups=np.array([r['group'] for r in source]))
            if context!='full':assert set(data['sample_ids']).isdisjoint(queries['sample_ids'])
            inp=root/'inputs'/(case_id+'.npz');inp.parent.mkdir(exist_ok=True);np.savez(inp,**data)
            query=root/'queries'/(case_id+'.npz');query.parent.mkdir(exist_ok=True);np.savez(query,**queries)
            assert set(queries)=={'x','age','params','sample_ids'}
            case=dict(id=case_id,program=name,kind=spec['kind'],context=context,
                input=str(inp.relative_to(root)),query=str(query.relative_to(root)),ordinary_model=str(old_model.relative_to(root)),
                fit_groups=data['groups'].tolist(),query_ids=queries['sample_ids'].tolist())
            cases.append(case)
            if spec['kind']=='anchor':
                s=data['params'][:,0]*data['params'][:,2]
                d=data['params'][:,2]*(data['params'][:,1]-data['age'])+data['params'][:,3]
                assert np.array_equal(s,np.ones(len(s))) and np.array_equal(d,-data['age'])
            else:
                for variant in ['scale_only','whole_program']:
                    tasks.append(dict(**case,variant=variant,task_id=case_id+'__'+variant))
    assert len(cases)==16 and len(tasks)==24
    save(root/'CASES460.json',cases);save(root/'TASKS460.json',tasks)
    for name in ['fit_contextual_readouts.py','fit_component_readouts.py','validate_component_readouts.py',
                 'controlled_feedback.py','program_revision_runtime.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'code'/name)
    paths=[p for folder in ['nested','full','inputs','queries','code'] for p in (root/folder).rglob('*') if p.is_file()]
    paths += [root/n for n in ['PROTOCOL460.json','CASES460.json','TASKS460.json']]
    save(root/'LAUNCH_FREEZE460.json',{str(p.relative_to(root)):sha(p) for p in paths})
    save(root/'PREPARED460.json',dict(passed=True,actual_new_fits=0,cases=len(cases),tasks=len(tasks),utc=utc()))
    print(json.dumps(read(root/'PREPARED460.json')),flush=True)


def worker(root,mode,variant):
    assert not Path('/data').exists() and not Path('/srv').exists()
    z=load('/input.npz');started=time.monotonic()
    if mode=='fit':
        from scipy.optimize import nnls
        assert set(z)=={'x','y','age','params','sample_ids','groups'}
        p=z['params'];s=p[:,0]*p[:,2];d=p[:,2]*(p[:,1]-z['age'])+p[:,3]
        target=(z['y']-d)/s if variant=='whole_program' else z['y']+z['age']
        mass=s*s;xm=np.average(z['x'],axis=0,weights=mass);ym=np.average(target,weights=mass)
        x=s[:,None]*(z['x']-xm);y=s*(target-ym);n=x.shape[1]
        coef,resid=nnls(np.vstack([x,np.sqrt(12.)*np.eye(n)]),np.r_[y,np.zeros(n)],maxiter=1000)
        intercept=float(ym-xm@coef);life=z['x']@coef+intercept
        actual=compose(life,z['age'],p);linear=s*life+d
        inactive=bool(np.allclose(actual,linear,atol=1e-6,rtol=1e-9))
        grad=x.T@(x@coef-y)+12*coef;pg=np.where(coef>1e-8,grad,np.minimum(grad,0))
        kkt=float(np.linalg.norm(pg,np.inf)/max(1,np.linalg.norm(x.T@y,np.inf)))
        save(root/'MODEL.json',dict(coefficient=coef.tolist(),intercept=intercept,alpha=12.,variant=variant,
            input_sha256=sha('/input.npz'),training_rows=len(target),fit_calls=1,fit_seconds=time.monotonic()-started,
            linear_region_qualified=inactive,training_full_program_MAE=float(np.abs(actual-z['y']).mean()),
            training_MAE_is_validation=False,normalized_projected_gradient=kkt,objective=float(resid**2),utc=utc()))
        assert kkt<1e-10
        assert variant!='whole_program' or inactive,'Full-program linear-region qualification failed; no repair'
    else:
        assert set(z)=={'x','age','params','sample_ids'}
        m=read('/model.json');life=z['x']@np.array(m['coefficient'])+m['intercept']
        actual=compose(life,z['age'],z['params'])
        assert np.isfinite(actual).all()
        save(root/'PREDICTIONS.json',dict(sample_ids=z['sample_ids'].tolist(),prediction=actual.tolist(),
            input_sha256=sha('/input.npz'),model_sha256=sha('/model.json'),targets_visible=False,
            actual_base_predict_calls=0,readout_predict_calls=1,seconds=time.monotonic()-started,utc=utc()))


def launch(root,mode):
    cfg=read(root/'PROTOCOL460.json');tasks=read(root/'TASKS460.json')
    if mode=='predict':assert read(root/'FIT_SEALED460.json')['passed']
    for p,d in read(root/'LAUNCH_FREEZE460.json').items():assert sha(root/p)==d
    start=time.monotonic()
    def one(t):
        out=root/mode/t['task_id'];out.mkdir(parents=True,exist_ok=False);(out/'tmp/mpl').mkdir(parents=True)
        cmd=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
        for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
            if Path(p).exists():cmd+=['--ro-bind',p,p]
        mounts=[(PYTHON,'/opt/python'),(DEPS,'/opt/deps'),(root/'code','/code'),
                (root/t['input' if mode=='fit' else 'query'],'/input.npz')]
        if mode=='predict':mounts.append((root/'fit'/t['task_id']/'MODEL.json','/model.json'))
        for src,dst in mounts:cmd+=['--ro-bind',str(src),dst]
        cmd+=['--bind',str(out),'/output','--proc','/proc','--dev','/dev','--chdir','/code','--remount-ro','/',
            '/opt/python/bin/python3.12','-B','/code/fit_contextual_readouts.py','--run','/output','--stage','worker','--mode',mode,'--variant',t['variant']]
        env=dict(PATH='/opt/python/bin:/usr/bin:/bin',PYTHONPATH='/opt/deps:/code',PYTHONDONTWRITEBYTECODE='1',
            TMPDIR='/output/tmp',TMP='/output/tmp',TEMP='/output/tmp',MPLCONFIGDIR='/output/tmp/mpl',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2')
        begin=time.monotonic()
        try:
            limit=min(cfg['fit_seconds' if mode=='fit' else 'predict_seconds'],cfg['whole_stage_seconds']-(time.monotonic()-start))
            assert limit>0
            p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=limit)
            record=dict(task_id=t['task_id'],status='complete' if p.returncode==0 else 'error',returncode=p.returncode,
                stdout=p.stdout,stderr=p.stderr,command=cmd,seconds=time.monotonic()-begin,utc=utc())
        except Exception as e:record=dict(task_id=t['task_id'],status='error',error_type=type(e).__name__,error=str(e),command=cmd,seconds=time.monotonic()-begin,utc=utc())
        save(out/'PROCESS.json',record);return record
    records=[]
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(one,t) for t in tasks]):
            records.append(f.result());save(root/'PROGRESS460.json',dict(stage=mode,finished=len(records),total=24,errors=sum(r['status']!='complete' for r in records),utc=utc()))
    passed=all(r['status']=='complete' for r in records)
    save(root/(mode.upper()+'_SEALED460.json'),dict(passed=passed,rows=records,seconds=time.monotonic()-start,utc=utc()))
    print(json.dumps(dict(stage=mode,passed=passed,processes=len(records),seconds=time.monotonic()-start)),flush=True)
    assert passed,'Preserve all failures; no scientific restart'


def choose(root):
    assert read(root/'PREDICT_SEALED460.json')['passed']
    cases=read(root/'CASES460.json');source=read(root/'nested/CACHED459.json');q=load(root/'nested/inputs/training_queries.npz')
    truth=dict(zip(q['sample_ids'],q['y']));predictions=[]
    for c in cases:
        z=load(root/c['query']);m=read(root/c['ordinary_model'])
        old=compose(z['x']@np.array(m['coefficient'])+m['intercept'],z['age'],z['params'])
        values={'ordinary':old.tolist()}
        for v in ['scale_only','whole_program']:
            values[v]=old.tolist() if c['kind']=='anchor' else read(root/'predict'/(c['id']+'__'+v)/'PREDICTIONS.json')['prediction']
        for i,s in enumerate(z['sample_ids']):
            row=dict(program=c['program'],context=c['context'],sample_id=str(s),predictions={k:v[i] for k,v in values.items()})
            if c['context']!='full':row['predictions']['original']=next(r['original'] for r in source[c['program']] if r['task']['sample_id']==s)
            predictions.append(row)
    save(root/'PREDICTIONS_SEALED460.json',dict(utc=utc(),rows=predictions,external_targets_joined=False))
    order=['original','ordinary','scale_only','whole_program'];choices=[]
    for name in sorted(source):
        rows=[r for r in predictions if r['program']==name and r['context']!='full'];assert len(rows)==47
        scores={v:float(np.mean([abs(r['predictions'][v]-truth[r['sample_id']]) for r in rows])) for v in order}
        def select(menu):
            best=menu[0]
            for v in menu[1:]:
                if scores[v]<scores[best]-1e-8:best=v
            return best
        choices.append(dict(program=name,CV_MAE=scores,selected=select(order),without_whole=select(order[:-1]),A459=select(order[:2])))
    save(root/'CHOICES_SEALED460.json',dict(utc=utc(),choices=choices,external_targets_joined=False,confirmation_opened=False))
    print(json.dumps(read(root/'CHOICES_SEALED460.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--stage',required=True)
    p.add_argument('--mode');p.add_argument('--variant');a=p.parse_args()
    if a.stage=='worker':worker(a.run,a.mode,a.variant)
    elif a.stage in ['fit','predict']:launch(a.run.resolve(),a.stage)
    else:globals()[a.stage](a.run.resolve())
