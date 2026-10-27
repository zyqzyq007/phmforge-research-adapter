"""Paired calibration data, independently varied residual and deployment parent."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from controlled_feedback import save, sha, utc


read=lambda p:json.loads(Path(p).read_text())
def data(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def inputs(root):
    cfg=read(root/'PROTOCOL379.json');src={k:Path(v['path']) for k,v in cfg['sources'].items()}
    for k,p in src.items():assert sha(p/'SOURCE_MANIFEST.json')==cfg['sources'][k]['manifest_sha256']
    source=data(src['source']/'private_inputs/training_queries_private.npz')
    dev=data(src['source']/'private_inputs/discovery_truth.npz')
    profile=data(src['profiles']/'PROFILES371.npz');column=profile['programs'].tolist().index('ordinary_2__p01')
    assert np.array_equal(profile['train_groups'],source['groups'])
    return cfg,src,source,dev,profile['cv_prediction'][:,column],profile['development_prediction'][:,column]


def fit(root):
    assert not (root/'HANDLE379.json').exists()
    cfg,src,source,dev,source_parent,full_parent=inputs(root)
    save(root/'HANDLE379.json',dict(pid=os.getpid(),utc=utc(),state='running'))
    start=time.monotonic();jobs=read(src['source']/'JOBS366.json')
    def one(fold):
        spec=jobs[fold];cal=np.flatnonzero(np.isin(source['groups'],spec['heldout_cells']))
        assert set(source['groups'][cal]).isdisjoint(spec['fitting_cells'])
        parent_path=src['nested']/'models'/fold/'predictions.json'
        fold_parent=np.array(read(parent_path)['rul'])
        events=[]
        for held in range(8):
            avail=np.flatnonzero(np.arange(8)!=held)
            x=np.vstack([source['x'][cal],dev['x'][avail]])
            scaler=StandardScaler();z=scaler.fit_transform(x)
            folder=root/'models'/fold/str(held);folder.mkdir(parents=True)
            # One common scaler per paired contrast; input entities identical.
            for residual_parent,dp in [('fold',fold_parent),('full',full_parent)]:
                target=np.r_[source['y'][cal]-source_parent[cal],dev['y'][avail]-dp[avail]]
                mean=float(target.mean());model=Ridge(alpha=10)
                event=dict(fold=fold,held=held,residual_parent=residual_parent,
                    parent_fitting_source_cells=spec['fitting_cells'],
                    correction_source_cells=source['groups'][cal].tolist(),
                    correction_development_cells=avail.tolist(),rows=len(x),status='started',
                    target_sha256=hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest())
                events.append(event);save(root/'events'/(fold+'.json'),events)
                model.fit(z,target-mean)
                path=folder/(residual_parent+'.joblib')
                joblib.dump(dict(model=model,scaler=scaler,target_mean=mean,residual_parent=residual_parent,
                    parent_fitting_source_cells=spec['fitting_cells'],correction_source_cells=event['correction_source_cells'],
                    correction_development_cells=avail.tolist()),path)
                correction=float((model.predict(scaler.transform(dev['x'][held:held+1]))+mean)[0])
                event.update(status='completed',model_path=str(path.relative_to(root)),model_sha256=sha(path),
                    coefficient_norm=float(np.linalg.norm(model.coef_)),warm_correction=correction,
                    full_parent=float(full_parent[held]),fold_parent=float(fold_parent[held]),
                    predictions={kind:max(0,float(b[held])+correction) for kind,b in [('full',full_parent),('fold',fold_parent)]})
                save(root/'events'/(fold+'.json'),events)
            assert time.monotonic()-start<cfg['budget']['wall_seconds']
        return events
    with ThreadPoolExecutor(max_workers=3) as pool:events=sum(list(pool.map(one,['fold_0','fold_1','fold_2'])),[])
    assert len(events)==48 and all(e['status']=='completed' for e in events)
    save(root/'FIT_EVENTS379.json',events)
    save(root/'FITS_TERMINAL379.json',dict(exit_code=0,utc=utc(),actual_Ridge_fits=48,
        actual_scaler_fits=24,warm_predict_calls=48,new_parent_fits=0,new_LLM_calls=0,seconds=time.monotonic()-start))
    print(json.dumps(read(root/'FITS_TERMINAL379.json')))


def evaluate(root):
    assert not (root/'DECISIONS379.json').exists()
    cfg,src,source,dev,source_parent,full_parent=inputs(root)
    assert read(root/'FITS_TERMINAL379.json')['exit_code']==0
    events=read(root/'FIT_EVENTS379.json');cold=[]
    for e in events:
        path=root/e['model_path'];assert sha(path)==e['model_sha256']
        a=joblib.load(path);held=e['held'];assert held not in a['correction_development_cells']
        assert not set(a['parent_fitting_source_cells'])&set(a['correction_source_cells'])
        cal=np.flatnonzero(np.isin(source['groups'],a['correction_source_cells']));avail=a['correction_development_cells']
        base=full_parent if e['residual_parent']=='full' else np.array(read(src['nested']/'models'/e['fold']/'predictions.json')['rul'])
        target=np.r_[source['y'][cal]-source_parent[cal],dev['y'][avail]-base[avail]]
        assert hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest()==e['target_sha256']
        value=float((a['model'].predict(a['scaler'].transform(dev['x'][held:held+1]))+a['target_mean'])[0])
        np.testing.assert_allclose(value,e['warm_correction'],rtol=1e-10,atol=1e-8)
        cold.append(dict(e,cold_correction=value))
    original={e['held']:e for e in read(src['capacity']/'FIT_EVENTS373.json') if e['spec']=='raw_input__mixed_54__residual__ridge10'}
    predictions={'original_parent':full_parent.tolist(),'old_mixed_correction':[original[i]['warm_prediction'] for i in range(8)]}
    predictions['fold_parent_mean']=[float(np.mean([e['fold_parent'] for e in cold if e['held']==i and e['residual_parent']=='fold'])) for i in range(8)]
    for fit_parent in ['fold','full']:
        for deploy_parent in ['fold','full']:
            key='calibrate_'+fit_parent+'__deploy_'+deploy_parent
            predictions[key]=[float(np.mean([max(0,e[deploy_parent+'_parent']+e['cold_correction']) for e in cold if e['held']==i and e['residual_parent']==fit_parent])) for i in range(8)]
    save(root/'DECISIONS379.json',dict(predictions=predictions,aggregation='uniform mean3folds fixedbeforefits',
        current_held_truth_used_by_rules=False,all8_design_used=True))
    scores={}
    for k,pred in predictions.items():
        p=np.array(pred);error=abs(p-dev['y']);old=abs(full_parent-dev['y'])
        scores[k]=dict(MAE=float(error.mean()),harm=float(np.maximum(error-old,0).mean()),
            gain=float(np.maximum(old-error,0).mean()),errors=error.tolist(),strict_improvements=int(np.sum(error<old-1e-9)),
            strict_regressions=int(np.sum(error>old+1e-9)))
    a=scores['calibrate_fold__deploy_fold'];b=scores['calibrate_full__deploy_full'];old=scores['old_mixed_correction']
    def passes(x,y):return x['MAE']<y['MAE']-1e-9 and x['harm']<=y['harm']+1e-9 and x['gain']>=y['gain']-1e-9
    result=dict(utc=utc(),scores=scores,paired_mechanism_gate=passes(a,b),strong_proposal_gate=passes(a,old),
        actual_Ridge_fits=48,actual_scalers=24,warm_predict_calls=48,cold_predict_calls=48,
        reused_parent_checkpoints=4,new_parent_fits=0,new_LLM_calls=0,final22_opened=False,A33872_opened=False,
        independent_confirmation=False,novelty_confirmed=False,Agent_comparison=False,
        note='Known fixed-predictor residual calibration control; source identity and7historicaldev labels identical within everypair. Inference and devresidual parents independentlyvaried; full/fold basecapacity andaggregation controlled, not a new Agent win.')
    save(root/'RESULT379.json',result);save(root/'TERMINAL379.json',dict(exit_code=0,utc=utc()))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','evaluate']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.run.resolve())
