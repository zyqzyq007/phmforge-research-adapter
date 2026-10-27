"""One paired information expansion, with a genuinely trained unchanged learner."""
import argparse,json,hashlib,time,os
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from controlled_feedback import save,sha,utc
from audit_program_update_capacity import predict

read=lambda p:json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def inputs(root):
    cfg=read(root/'PROTOCOL381.json');src={k:Path(v['path']) for k,v in cfg['sources'].items()}
    for k,p in src.items():assert sha(p/'SOURCE_MANIFEST.json')==cfg['sources'][k]['manifest_sha256']
    source=load(src['source']/'private_inputs/training_queries_private.npz')
    dev=load(src['source']/'private_inputs/discovery_truth.npz')
    profiles=load(src['profiles']/'PROFILES371.npz');index=profiles['programs'].tolist().index('ordinary_2__p01')
    assert np.array_equal(profiles['train_groups'],source['groups'])
    binding={v['public_path']:v for v in read(src['history']/'PRIVATE_BINDINGS380.json')}
    def extend(d,role):
        history=[]
        for ident in d['groups']:
            path=src['history']/'public'/role/(str(ident)+'.npz');key=str(path.relative_to(src['history']))
            assert sha(path)==binding[key]['public_sha256']
            h=load(path);assert np.array_equal(h['cycle'],np.arange(10,500))
            history.append((h['capacity_mAh'].reshape(49,10).mean(axis=1)/1000).astype(np.float32))
        return np.concatenate([d['x'],np.array(history,dtype=np.float32)],axis=1)
    sx,dx=extend(source,'train'),extend(dev,'development')
    assert sx.shape==(47,222) and dx.shape==(8,222)
    return cfg,src,source,dev,profiles['cv_prediction'][:,index],profiles['development_prediction'][:,index],sx,dx

def fit(root):
    assert not (root/'HANDLE381.json').exists();start=time.monotonic()
    cfg,src,s,d,sp,dp,sx,dx=inputs(root);save(root/'HANDLE381.json',dict(pid=os.getpid(),utc=utc()))
    events=[]
    for held in range(8):
        avail=np.flatnonzero(np.arange(8)!=held);x=np.vstack([sx,dx[avail]])
        target=np.r_[s['y']-sp,d['y'][avail]-dp[avail]];mean=float(target.mean())
        scaler=StandardScaler();z=scaler.fit_transform(x);model=Ridge(alpha=10)
        event=dict(held=held,status='started',fitting_source_cells=s['groups'].tolist(),
            fitting_development_cells=avail.tolist(),target_sha256=hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest())
        events.append(event);save(root/'FIT_EVENTS381.json',events)
        model.fit(z,target-mean)
        artifact=dict(model=model,scaler=scaler,target_mean=mean,update='residual',
            fitting_source_cells=event['fitting_source_cells'],fitting_development_cells=avail.tolist())
        path=root/'models'/('held_'+str(held)+'.joblib');path.parent.mkdir(exist_ok=True);joblib.dump(artifact,path)
        warm=float(predict(artifact,dx[held:held+1],dp[held:held+1])[0])
        event.update(status='completed',model_path=str(path.relative_to(root)),model_sha256=sha(path),
            coefficient_norm=float(np.linalg.norm(model.coef_)),warm_prediction=warm,parent_prediction=float(dp[held]))
        save(root/'FIT_EVENTS381.json',events)
    assert time.monotonic()-start<cfg['budget']['wall_seconds']
    save(root/'FITS_TERMINAL381.json',dict(exit_code=0,utc=utc(),new_Ridge_fits=8,new_scalers=8,warm_predictions=8,new_LLM_calls=0,seconds=time.monotonic()-start))
    print(json.dumps(read(root/'FITS_TERMINAL381.json')))

def evaluate(root):
    assert not(root/'PREDICTIONS381.json').exists()
    cfg,src,s,d,sp,dp,sx,dx=inputs(root);assert read(root/'FITS_TERMINAL381.json')['exit_code']==0
    original={e['held']:e for e in read(src['capacity']/'FIT_EVENTS373.json') if e['spec']=='raw_input__mixed_54__residual__ridge10'}
    rows=[]
    for e in read(root/'FIT_EVENTS381.json'):
        held=e['held'];path=root/e['model_path'];assert sha(path)==e['model_sha256'];a=joblib.load(path)
        assert held not in a['fitting_development_cells'];assert a['scaler'].n_samples_seen_==54
        avail=a['fitting_development_cells'];target=np.r_[s['y']-sp,d['y'][avail]-dp[avail]]
        assert hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest()==e['target_sha256']
        new=float(predict(a,dx[held:held+1],dp[held:held+1])[0]);np.testing.assert_allclose(new,e['warm_prediction'],rtol=1e-10,atol=1e-8)
        old=original[held];oldpath=src['capacity']/old['model_path'];assert sha(oldpath)==old['model_sha256']
        oldpred=float(predict(joblib.load(oldpath),d['x'][held:held+1],dp[held:held+1])[0]);np.testing.assert_allclose(oldpred,old['warm_prediction'],rtol=1e-10,atol=1e-8)
        rows.append(dict(held=held,parent=float(dp[held]),original_173=oldpred,expanded_222=new))
    save(root/'PREDICTIONS381.json',dict(rows=rows,current_label_not_used_for_fitting=True,all8_design_used=True))
    scores={}
    for method in ['parent','original_173','expanded_222']:
        error=np.abs(np.array([r[method] for r in rows])-d['y']);base=np.abs(dp-d['y']);diff=error-base
        scores[method]=dict(MAE=float(error.mean()),harm=float(np.maximum(diff,0).mean()),gain=float(np.maximum(-diff,0).mean()),
            errors=error.tolist(),parent_wins=int(np.sum(diff<-1e-9)),parent_losses=int(np.sum(diff>1e-9)))
    a,b=scores['expanded_222'],scores['original_173']
    gate=a['MAE']<b['MAE']-1e-9 and a['harm']<=b['harm']+1e-9 and a['gain']>=b['gain']-1e-9
    result=dict(utc=utc(),scores=scores,paired_information_gate=gate,new_Ridge_fits=8,new_scalers=8,warm_predict_calls=8,
        cold_predict_calls=16,old_model_replays=8,new_LLM_calls=0,novelty_confirmed=False,Agent_comparison=False,
        final22_opened=False,A33872_opened=False,independent_confirmation=False,
        note='Moreactualpastobservations;notAgentmechanismgain.NopastcapacitysurrogatescoreusedtorejectRULprograms.')
    save(root/'RESULT381.json',result);save(root/'TERMINAL381.json',dict(exit_code=0,utc=utc()));print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','evaluate']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.run.resolve())
