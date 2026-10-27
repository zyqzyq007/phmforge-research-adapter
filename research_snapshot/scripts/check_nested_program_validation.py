"""Fixed-code, whole-cell nested validation of an actual PHM program update."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from audit_program_update_capacity import predict
from controlled_feedback import save, sha, utc


def read(p):
    return json.loads(Path(p).read_text())


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def part(a, index):
    return {k: v[index] for k, v in a.items()}


def environment(root):
    env = dict(os.environ)
    for k in ['TMPDIR', 'TMP', 'TEMP']:
        env[k] = str(root/'runtime/tmp')
    env.update(MPLCONFIGDIR=str(root/'runtime/mpl'), PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return env


def parent_worker(root, job_path):
    job = read(job_path)
    spec = importlib.util.spec_from_file_location('frozen_parent', root/'code/parent_program.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = time.monotonic()
    events = []
    if job['mode'] == 'fit':
        # Record real estimator calls, including PLS, rather than count code paths.
        for cls in [module.Ridge, module.PLSRegression, module.ExtraTreesRegressor,
                    module.HistGradientBoostingRegressor]:
            old = cls.fit
            def wrapped(self, x, y, *args, _old=old, _name=cls.__name__, **kwargs):
                entry = dict(estimator=_name, rows=len(y), status='started',
                    target_sha256=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())
                events.append(entry); save(job_path.parent/'ESTIMATOR_EVENTS.json', events)
                result = _old(self, x, y, *args, **kwargs)
                entry['status'] = 'completed'; save(job_path.parent/'ESTIMATOR_EVENTS.json', events)
                return result
            cls.fit = wrapped
        module.fit(Path(job['input']), Path(job['artifacts']))
        assert len(events) == 5 and all(e['status'] == 'completed' for e in events)
    else:
        module.predict(Path(job['input']), Path(job['artifacts']), Path(job['output']))
        values = read(job['output'])['rul']
        assert np.isfinite(values).all()
    save(job_path.parent/(job['mode']+'_TERMINAL.json'), dict(exit_code=0, utc=utc(),
        seconds=time.monotonic()-started, completed_estimator_fits=len(events)))


def invoke(root, job, timeout):
    mode = job['mode']
    folder = root/'jobs'/job['name']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(mode+'_JOB.json')
    assert not path.exists(), 'Do not restart a job implicitly'
    save(path, job)
    cmd = [sys.executable, '-B', str(root/'code/check_nested_program_validation.py'),
           'worker', '--run', str(root), '--job', str(path)]
    with (folder/(mode+'.log')).open('w') as log:
        proc = subprocess.Popen(cmd, env=environment(root), stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        save(folder/(mode+'_HANDLE.json'), dict(pid=proc.pid, command=cmd, utc=utc()))
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            code = 124
    save(folder/(mode+'_PROCESS.json'), dict(returncode=code, utc=utc()))
    return job['name'], code


def fit(root):
    cfg = read(root/'PROTOCOL377.json')
    assert not (root/'HANDLE377.json').exists(), 'Never implicitly restart'
    save(root/'HANDLE377.json', dict(pid=os.getpid(), state='running', utc=utc()))
    start = time.monotonic()
    src = {k: Path(v['path']) for k,v in cfg['sources'].items()}
    for k,p in src.items():
        assert sha(p/'SOURCE_MANIFEST.json') == cfg['sources'][k]['manifest_sha256']
    source = src['source']; parent = source/'subjects/ordinary_2/programs/p01'
    assert sha(root/'code/parent_program.py') == sha(parent/'source/program.py')
    train = arrays(source/'private_inputs/train.npz')
    query = arrays(source/'private_inputs/training_queries_private.npz')
    dev = arrays(source/'private_inputs/development_observations.npz')
    # Existing discovery labels are legal for the OTHER seven historical cells.
    # Current held-cell labels are used only in the separate evaluation action.
    devlabels = arrays(source/'private_inputs/discovery_truth.npz')['y']
    jobs = read(source/'JOBS366.json')
    plans, prediction_jobs, bindings = [], [], []
    for outer, job in jobs.items():
        if outer == 'full': continue
        fit_cells, val_cells = set(job['fitting_cells']), set(job['heldout_cells'])
        ordered = sorted(fit_cells, key=lambda g: hashlib.sha256(('A377inner:'+g).encode()).hexdigest())
        for inner in range(3):
            valid = set(ordered[inner::3]); fitting = fit_cells-valid
            assert not fitting & val_cells and not valid & val_cells and not fitting & valid
            name = outer+'_inner_'+str(inner); folder = root/'models'/name
            folder.mkdir(parents=True)
            np.savez_compressed(folder/'train.npz', **part(train, np.isin(train['groups'], list(fitting))))
            qi = np.flatnonzero(np.isin(query['groups'], list(valid)))
            np.savez_compressed(folder/'input.npz', x=query['x'][qi], age=query['age'][qi])
            base = dict(name=name, input=str(folder/'train.npz'), artifacts=str(folder/'artifacts'),
                outer=outer, fitting_cells=sorted(fitting), heldout_cells=sorted(valid),
                untouched_outer_cells=sorted(val_cells), query_indices=qi.tolist())
            plans.append(dict(base, mode='fit'))
            prediction_jobs.append(dict(base, mode='predict', input=str(folder/'input.npz'),
                                        output=str(folder/'predictions.json')))
        folder = root/'models'/outer; folder.mkdir()
        np.savez_compressed(folder/'input.npz', x=dev['x'], age=dev['age'])
        checkpoint = parent/outer/'fit/artifacts/cycle500_lifetime_ensemble.joblib'
        bindings.append(dict(path=str(checkpoint), sha256=sha(checkpoint), fitting_cells=sorted(fit_cells)))
        prediction_jobs.append(dict(name=outer, mode='predict', input=str(folder/'input.npz'),
            artifacts=str(checkpoint.parent), output=str(folder/'predictions.json'),
            fitting_cells=sorted(fit_cells), query_development_cells=list(range(8))))
    save(root/'NESTED_SPLITS377.json', plans)
    save(root/'REUSED_PARENT_BINDINGS377.json', bindings)
    failures = []
    for stage, stagejobs in [('parent_fits',plans),('parent_predictions',prediction_jobs)]:
        with ThreadPoolExecutor(max_workers=cfg['budget']['workers']) as pool:
            futures = [pool.submit(invoke,root,j,cfg['budget']['per_job_seconds']) for j in stagejobs]
            for future in as_completed(futures):
                name, code = future.result()
                if code: failures.append(dict(stage=stage,name=name,exit_code=code))
                save(root/'PROGRESS377.json',dict(utc=utc(),stage=stage,
                    parent_fits_finished=len(list((root/'jobs').glob('*/fit_PROCESS.json'))),
                    parent_predicts_finished=len(list((root/'jobs').glob('*/predict_PROCESS.json'))),failures=failures))
        assert not failures, failures
        assert time.monotonic()-start < cfg['budget']['wall_seconds']
    events=[]
    for outer, job in jobs.items():
        if outer == 'full': continue
        ti=np.flatnonzero(np.isin(query['groups'],job['fitting_cells']))
        vi=np.flatnonzero(np.isin(query['groups'],job['heldout_cells']))
        inner=np.full(47,np.nan)
        for plan in [p for p in plans if p['outer']==outer]:
            inner[plan['query_indices']]=read(root/'models'/plan['name']/'predictions.json')['rul']
        assert np.isfinite(inner[ti]).all() and np.isnan(inner[vi]).all()
        outerdev=np.array(read(root/'models'/outer/'predictions.json')['rul'])
        old=arrays(parent/outer/'check/PREDICTIONS.npz')
        oldmap=dict(zip(old['sample_ids'].tolist(),old['prediction'].tolist()))
        base=np.array([oldmap[i] for i in query['sample_ids'][vi]])
        for held in range(8):
            available=np.flatnonzero(np.arange(8)!=held)
            x=np.vstack([query['x'][ti],dev['x'][available]])
            target=np.r_[query['y'][ti]-inner[ti],devlabels[available]-outerdev[available]]
            scaler=StandardScaler(); z=scaler.fit_transform(x); mean=float(target.mean())
            model=Ridge(alpha=10);model.fit(z,target-mean)
            artifact=dict(scaler=scaler,model=model,target_mean=mean,update='residual',
                parent='ordinary_2__p01',outer=outer,held=held,fitting_source_cells=job['fitting_cells'],
                fitting_development_cells=available.tolist())
            folder=root/'corrections'/outer/str(held);folder.mkdir(parents=True)
            joblib.dump(artifact,folder/'model.joblib')
            qx=np.vstack([query['x'][vi],dev['x'][held:held+1]])
            qb=np.r_[base,outerdev[held]]
            np.savez_compressed(folder/'input.npz',x=qx,parent=qb)
            warm=predict(artifact,qx,qb)
            event=dict(outer=outer,held=held,fitting_source_cells=job['fitting_cells'],
                fitting_development_cells=available.tolist(),validation_source_cells=job['heldout_cells'],
                source_indices=vi.tolist(),model_path=str((folder/'model.joblib').relative_to(root)),
                model_sha256=sha(folder/'model.joblib'),input_sha256=sha(folder/'input.npz'),
                target_sha256=hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest(),
                warm_predictions=warm.tolist(),parent_predictions=qb.tolist(),status='completed')
            events.append(event);save(root/'CORRECTION_EVENTS377.json',events)
    assert len(events)==24
    save(root/'FITS_TERMINAL377.json',dict(utc=utc(),exit_code=0,parent_fit_entries=9,
        parent_estimator_fits=45,correction_fits=24,correction_scalers=24,parent_predict_processes=12,
        warm_correction_predict_calls=24,elapsed_seconds=time.monotonic()-start,new_LLM_calls=0))
    print(json.dumps(read(root/'FITS_TERMINAL377.json')))


def optimal_alpha(base, candidate, truth, weight):
    delta=candidate-base
    nonzero=np.abs(delta)>1e-12
    options=np.unique(np.r_[0.,1.,np.clip((truth[nonzero]-base[nonzero])/delta[nonzero],0,1)])
    risk=np.sum(weight[:,None]*np.abs(base[:,None]+delta[:,None]*options-truth[:,None]),axis=0)
    return float(options[np.flatnonzero(risk<=risk.min()+1e-10)[0]])


def evaluate(root):
    assert not (root/'DECISIONS377.json').exists()
    cfg=read(root/'PROTOCOL377.json');src={k:Path(v['path']) for k,v in cfg['sources'].items()}
    assert read(root/'FITS_TERMINAL377.json')['exit_code']==0
    query=arrays(src['source']/'private_inputs/training_queries_private.npz')
    devtruth=arrays(src['source']/'private_inputs/discovery_truth.npz')['y']
    historical=read(src['partial']/'FIT_EVENTS374.json')
    events=read(root/'CORRECTION_EVENTS377.json')
    source_base=np.full((8,47),np.nan);source_new=source_base.copy()
    cold_rows=0
    for e in events:
        path=root/e['model_path'];assert sha(path)==e['model_sha256']
        q=arrays(path.parent/'input.npz');assert sha(path.parent/'input.npz')==e['input_sha256']
        a=joblib.load(path);assert e['held'] not in a['fitting_development_cells']
        assert not set(a['fitting_source_cells']) & set(e['validation_source_cells'])
        cold=predict(a,q['x'],q['parent']);cold_rows+=len(cold)
        np.testing.assert_allclose(cold,e['warm_predictions'],rtol=1e-10,atol=1e-8)
        source_base[e['held'],e['source_indices']]=q['parent'][:-1]
        source_new[e['held'],e['source_indices']]=cold[:-1]
    assert np.isfinite(source_new).all() and cold_rows==400
    decisions=[]
    for held in range(8):
        history=[e for e in historical if e['held']==held and e['kind']=='loo']
        db=np.array([e['parent_predictions'][0] for e in history])
        dc=np.array([e['warm_predictions'][0] for e in history])
        dy=devtruth[[e['probe'] for e in history]]
        sb,sc,sy=source_base[held],source_new[held],query['y']
        choices=dict(keep=0.,full=1.,
            existing_dev_cv=optimal_alpha(db,dc,dy,np.ones(7)/7),
            nested_source_cv=optimal_alpha(sb,sc,sy,np.ones(47)/47),
            equal_cohort_cv=optimal_alpha(np.r_[sb,db],np.r_[sc,dc],np.r_[sy,dy],np.r_[np.ones(47)/94,np.ones(7)/14]))
        prior=next(e for e in read(src['capacity']/'FIT_EVENTS373.json') if e['held']==held and e['spec']=='raw_input__mixed_54__residual__ridge10')
        b,c=prior['parent_prediction'],prior['warm_prediction']
        decisions.append(dict(held=held,alphas=choices,parent=b,candidate=c,
            source_full_minus_parent_MAE=float(np.mean(abs(sc-sy)-abs(sb-sy))),
            dev_full_minus_parent_MAE=float(np.mean(abs(dc-dy)-abs(db-dy))),
            source_min_alpha_MAE=float(np.mean(abs(sb+choices['nested_source_cv']*(sc-sb)-sy))),
            predictions={k:b+a*(c-b) for k,a in choices.items()}))
    save(root/'DECISIONS377.json',dict(rows=decisions,current_held_label_used_by_rules=False,
        all8_design_used=True,independent_confirmation=False))
    np.savez_compressed(root/'SOURCE_VALIDATION377.npz',parent=source_base,candidate=source_new,groups=query['groups'])
    scores={}
    for method in decisions[0]['alphas']:
        rows=[]
        for d in decisions:
            y=float(devtruth[d['held']]);b,c=d['parent'],d['candidate'];p=d['predictions'][method]
            old,error=abs(b-y),abs(p-y);oracle=abs(np.clip(y,min(b,c),max(b,c))-y)
            opportunity=old-oracle
            rows.append(dict(held=d['held'],absolute_error=error,regret=error-oracle,
                harm=max(0,error-old),gain=max(0,old-error),opportunity=opportunity,
                gain_capture=max(0,old-error)/opportunity if opportunity>1e-9 else None))
        scores[method]=dict(rows=rows,MAE=float(np.mean([r['absolute_error'] for r in rows])),
            regret=float(np.mean([r['regret'] for r in rows])),harm=float(np.mean([r['harm'] for r in rows])),
            gain=float(np.mean([r['gain'] for r in rows])),
            gain_capture=float(np.mean([r['gain_capture'] for r in rows if r['gain_capture'] is not None])))
    a,b=scores['nested_source_cv'],scores['existing_dev_cv']
    gate=(a['MAE']<b['MAE']-1e-9 and a['harm']<=b['harm']+1e-9 and a['gain_capture']>=b['gain_capture']-1e-9)
    result=dict(utc=utc(),scores=scores,primary_numeric_gate=gate,
        new_fits=33,parent_estimators=45,correction_estimators=24,parent_predict_processes=12,
        warm_correction_predict_calls=24,cold_correction_predict_calls=24,cold_rows=400,
        qualified_source_validation_rows_per_held=47,unique_source_cells=47,
        new_LLM_calls=0,final22_opened=False,A33872_opened=False,
        novelty_confirmed=False,independent_confirmation=False,majority_complete=False,
        limitation='Fixed-code conventional validation. Does not nest original Agent code search, changes training size, adds legal source-check information, and uses one already-opened development cohort. No fair Agent superiority claim.')
    save(root/'RESULT377.json',result);save(root/'TERMINAL377.json',dict(exit_code=0,utc=utc()))
    print(json.dumps(dict(gate=gate,metrics={k:{t:v for t,v in s.items() if t!='rows'} for k,s in scores.items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','evaluate','worker'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--job',type=Path)
    a=p.parse_args();root=a.run.resolve()
    if a.action=='worker':parent_worker(root,a.job)
    else:globals()[a.action](root)
