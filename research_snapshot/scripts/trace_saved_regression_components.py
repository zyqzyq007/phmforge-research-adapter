"""Trace saved regressors without changing authored prediction code or weights.

Configuration binds every original resource. Controls are algebra on actual
component outputs, not new training or a newly successful Agent submission.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean, median
import time

import numpy as np

from controlled_feedback import save, sha, utc
from program_revision_runtime import Runtime, load
from screen_score_blinding import restore


WORKER = r'''
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
calls=[]
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
    if callable(getattr(value,'predict',None)): return Observed(value,name)
    return value
original_load=joblib.load
def traced_load(*args,**kwargs): return wrap(original_load(*args,**kwargs))
joblib.load=traced_load
# original_main.py is a byte-identical copy of the original registration entry.
runpy.run_path(str(Path(__file__).with_name('original_main.py')),run_name='__main__')
joblib.load=original_load
assert len(calls)==len({c['name'] for c in calls})
prediction=json.loads(output.read_text())['rul']
output.with_name('COMPONENT_TRACE.json').write_text(json.dumps({
    'age':age,'calls':calls,'original':prediction,'query_targets_visible':False,
    'authored_prediction_code_modified':False},indent=2)+'\n')
'''


def read(path):
    return json.loads(Path(path).read_text())


def controls(kind, values, age):
    positive = lambda v: max(float(v), 0.)
    rul = lambda v: positive(v-age)
    if kind == 'dual':
        cn = ['ridge', 'pls', 'et', 'hgb_sq', 'hgb_abs']
        tn = ['trend_ridge', 'trend_et', 'trend_hgb']
        core = [values[n] for n in cn]; trend = [values[n] for n in tn]
        c = rul(.8*median(core)+.2*mean(core))
        t = rul(.7*median(trend)+.3*mean(trend))
        h,e = positive(values['all_hgb']),positive(values['all_et'])
        mix = lambda a,b: .65*a+.23*b+.10*h+.02*e
        out = dict(original=mix(c,t), core_only=c, without_trend=.88*c+.10*h+.02*e,
                   without_teachers=(.65*c+.23*t)/.88, trend_only=t)
        out.update({'core_'+n:mix(rul(values[n]),t) for n in cn})
        out.update({'trend_'+n.removeprefix('trend_'):mix(c,rul(values[n])) for n in tn})
    elif kind == 'anchor':
        names = ['ridge','knn','et','hgb_sq','hgb_abs','anchor_knn','anchor_et','anchor_ridge']
        v = [values[n] for n in names]; a = v[-3:]
        out = dict(original=rul(.50*median(v)+.27*mean(v)+.23*median(a)),
                   without_extra_anchor_weight=rul((.50*median(v)+.27*mean(v))/.77),
                   median_all=rul(median(v)), mean_all=rul(mean(v)), anchor_median=rul(median(a)))
        out.update({'base_'+n:rul(values[n]) for n in names})
    elif kind == 'compact':
        names = [n for n in values if n.startswith('local.')]
        v = [values[n] for n in names]
        c = rul(.8*median(v)+.2*mean(v))
        d = rul(.58*values['direct_ridge']+.42*values['direct_extra'])
        t = positive(.72*values['all_hgb']+.28*values['all_extra'])
        mix = lambda a,b: .76*a+.06*b+.18*t
        out = dict(original=mix(c,d), local_only=c, without_direct=.82*c+.18*t,
                   without_teachers=(.76*c+.06*d)/.82, direct_only=d, teacher_only=t)
        out.update({'local_'+n.split('.')[1]:mix(rul(values[n]),d) for n in names})
        out.update({'direct_'+n:mix(c,rul(values['direct_'+n])) for n in ['ridge','extra']})
    else:
        raise ValueError(kind)
    assert all(math.isfinite(v) and v >= 0 for v in out.values())
    return out


def prepare(root):
    cfg=read(root/'PROTOCOL.json'); grouped=defaultdict(list)
    assert not (root/'PREPARED.json').exists()
    for row in cfg['resources']: grouped[row['archive']].append(row)
    def one(item):
        name, rows=item; archive=Path(name)
        assert sha(archive/'SOURCE_MANIFEST.json')==cfg['archives'][name]
        index={r['path']:r for r in read(archive/'SOURCE_MANIFEST.json')['files']}
        missing={r['member'] for r in rows if not (archive/r['member']).is_file()}
        cache=root/'clients-cache'/archive.name
        if missing: restore(archive,cache,lambda p:p in missing)
        for r in rows:
            source=(cache if r['member'] in missing else archive)/r['member']
            assert sha(source)==r['sha256']==index[r['member']]['sha256']
            dest=root/r['target'];dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
        return len(rows)
    with ThreadPoolExecutor(max_workers=4) as pool: restored=sum(pool.map(one,grouped.items()))
    queries=read(root/'reused/TASKS446.json');tasks=[]
    for program in cfg['programs']:
        source=root/'programs'/program['name']/'source'
        shutil.copyfile(source/'program.py',source/'original_main.py')
        (source/'program.py').write_text(WORKER)
        for query in queries:
            old=root/'expected'/program['name']/(query['context']+'.npz')
            arr=load(old);expected=dict(zip(arr['sample_ids'].tolist(),arr['prediction'].tolist()))
            tasks.append(dict(query,id=program['name']+'__'+query['id'],program=program['name'],
                kind=program['kind'],source=str(source.relative_to(root)),expected=expected[query['sample_id']],
                artifacts='programs/'+program['name']+'/'+query['context']+'/artifacts'))
    assert len(tasks)==cfg['new_cold_processes']
    for task in tasks:
        z=load(root/task['input'])
        assert set(z)=={'x','nn','past_cap','age','rates','cycle','groups','sample_ids'}
        assert len(z['age'])==1 and str(z['sample_ids'][0])==task['sample_id']
    save(root/'TASKS.json',tasks)
    save(root/'ENGINE_CONFIG.json',dict(input_root=str(root/'queries'),common_tools=str(root/'code'),fit_seconds=0,predict_seconds=cfg['predict_seconds']))
    names=['PROTOCOL.json','TASKS.json','ENGINE_CONFIG.json']
    names += [str(p.relative_to(root)) for f in ['programs','queries','code','reused','expected'] for p in (root/f).rglob('*') if p.is_file()]
    save(root/'LAUNCH_FREEZE.json',{p:sha(root/p) for p in names})
    save(root/'PREPARED.json',dict(utc=utc(),passed=True,resource_files=restored,tasks=len(tasks),new_fits=0))
    print(json.dumps(read(root/'PREPARED.json')),flush=True)


def run(root):
    cfg=read(root/'PROTOCOL.json');tasks=read(root/'TASKS.json')
    assert read(root/'PREPARED.json')['passed'] and not (root/'HANDLE.json').exists()
    for path,digest in read(root/'LAUNCH_FREEZE.json').items(): assert sha(root/path)==digest
    start=time.monotonic();save(root/'HANDLE.json',dict(pid=os.getpid(),state='running',utc=utc()))
    results=[]
    def one(task):
        output=root/'predictions'/task['id']
        try:
            rt=Runtime(root);rt.deadline=start+cfg['stage_seconds']
            process,audit=rt.command(root/task['source'],root/task['artifacts'],root/task['input'],output,'predict')
            trace=read(output/'COMPONENT_TRACE.json')
            assert audit['completed_estimator_fits']==audit['optimizer_steps']==0
            actual=trace['original'][0]
            assert math.isclose(actual,task['expected'],rel_tol=1e-7,abs_tol=1e-5)
            values={r['name']:float(np.asarray(r['prediction']).reshape(-1)[0]) for r in trace['calls']}
            out=controls(task['kind'],values,trace['age'])
            assert math.isclose(actual,out['original'],rel_tol=1e-9,abs_tol=1e-6)
            assert list(out)==cfg['variants'][task['kind']]
            return dict(**task,status='complete',seconds=process['seconds'],trace_sha256=sha(output/'COMPONENT_TRACE.json'),component_calls=len(trace['calls']))
        except Exception as e:
            return dict(**task,status='error',error_type=type(e).__name__,error=str(e))
    # Match one original output per program before spending the remaining budget.
    anchors=[next(t for t in tasks if t['program']==p['name'] and t['context']=='full') for p in cfg['programs']]
    def batch(todo):
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
            for future in as_completed([pool.submit(one,t) for t in todo]):
                row=future.result();results.append(row);save(root/'records'/(row['id']+'.json'),row)
                save(root/'PROGRESS.json',dict(utc=utc(),finished=len(results),total=len(tasks),errors=sum(x['status']!='complete' for x in results)))
    batch(anchors)
    if all(t['status']=='complete' for t in results):
        seen={t['id'] for t in anchors};batch([t for t in tasks if t['id'] not in seen])
    passed=len(results)==len(tasks) and all(t['status']=='complete' for t in results)
    save(root/'SEALED.json',dict(utc=utc(),passed=passed,rows=results,seconds=time.monotonic()-start,
        cold_processes=len(results),component_calls=sum(t.get('component_calls',0) for t in results),new_fits=0,new_LLM_calls=0,confirmation_opened=False))
    save(root/'HANDLE.json',dict(pid=os.getpid(),state='terminal',passed=passed,utc=utc()))
    print(json.dumps({k:v for k,v in read(root/'SEALED.json').items() if k!='rows'}),flush=True)
    assert passed,'Retain failures; no automatic restart'


def evaluate(root):
    cfg=read(root/'PROTOCOL.json');sealed=read(root/'SEALED.json')
    assert sealed['passed'] and read(root/'HANDLE.json')['state']=='terminal'
    # Only the host evaluator reads these labels, after all predictions are sealed.
    truth={}
    for row in cfg['truth_resources']:
        source=Path(row['archive'])/row['member'];assert sha(source)==row['sha256']
        z=load(source);truth.update(zip(z['sample_ids'].tolist(),z['y'].tolist()))
    rows=[]
    for task in sealed['rows']:
        path=root/'predictions'/task['id']/'COMPONENT_TRACE.json';assert sha(path)==task['trace_sha256']
        trace=read(path);values={r['name']:float(np.asarray(r['prediction']).reshape(-1)[0]) for r in trace['calls']}
        rows.append(dict(**task,variants=controls(task['kind'],values,trace['age']),truth=float(truth[task['sample_id']])))
    # Reuse every original successful submission record; do not re-execute it.
    for task in read(root/'reused/TASKS446.json'):
        data=read(root/'reused'/'predictions'/task['id']/'COMPONENTS.json')
        rows.append(dict(**task,program=cfg['success_name'],status='cached',
            variants={k:v[0] for k,v in data['variants'].items()},truth=float(truth[task['sample_id']]),
            cache_source='A446'))
    metrics={};choices={}
    for program in [p['name'] for p in cfg['programs']]+[cfg['success_name']]:
        subset=[r for r in rows if r['program']==program];assert len(subset)==55
        metrics[program]={}
        for variant in subset[0]['variants']:
            metrics[program][variant]={}
            for split in ['CV','exploration']:
                take=[r for r in subset if (r['context']=='full')==(split=='exploration')]
                assert len(take)==(47 if split=='CV' else 8)
                errors=[abs(t['variants'][variant]-t['truth']) for t in take]
                differences=[e-abs(t['variants']['original']-t['truth']) for e,t in zip(errors,take)]
                metrics[program][variant][split]=dict(MAE=mean(errors),harm=mean(max(0,d) for d in differences),
                    gain=mean(max(0,-d) for d in differences),wins=sum(d < -1e-8 for d in differences),
                    losses=sum(d > 1e-8 for d in differences),per_cell_error=errors,ids=[t['sample_id'] for t in take])
        choices[program]=min(metrics[program],key=lambda v:(metrics[program][v]['CV']['MAE'],v!='original',v))
    save(root/'DETAILS.json',rows)
    save(root/'RESULT.json',dict(utc=utc(),metrics=metrics,ordinary_complete_CV_choices=choices,
        actual_new_cold_processes=sealed['cold_processes'],cached_success_rows=55,actual_new_component_predict_calls=sealed['component_calls'],
        new_fits=0,new_LLM_calls=0,new_Agent_stages=0,independent_confirmation=0,new_method_superiority=False))
    print(json.dumps({'seconds':sealed['seconds'],'choices':choices,'metrics':{p:{v:{s:x['MAE'] for s,x in m.items()} for v,m in d.items()} for p,d in metrics.items()}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','run','evaluate'],required=True)
    a=p.parse_args();globals()[a.stage](a.run.resolve())
