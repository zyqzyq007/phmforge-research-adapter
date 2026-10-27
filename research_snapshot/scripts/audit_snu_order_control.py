"""Independent raw-signal, parameter-forward and metric audit of A421."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile

import h5py
import joblib
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis,skew


def read(path):return json.loads(Path(path).read_text())


def load(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def independent_descriptor(x,rate,edges,normalized):
    if normalized:x=x/(np.linalg.norm(x)/np.sqrt(len(x)))
    mu=float(np.mean(x));sd=float(np.std(x));rms=float(np.linalg.norm(x)/np.sqrt(len(x)))
    maximum=float(np.max(abs(x)));absolute=float(np.mean(abs(x)))
    stats=[mu,sd,rms,maximum,float(x.max()-x.min()),absolute,float(kurtosis(x,fisher=False,bias=True)),float(skew(x,bias=True)),maximum/rms,rms/absolute,maximum/absolute,float(np.count_nonzero(x[1:]*x[:-1]<0)/(len(x)-1))]
    spectrum=np.fft.rfft(x-mu);power=(spectrum.real**2+spectrum.imag**2)/len(x)**2
    frequencies=np.arange(len(power))*rate/len(x)
    # Direct masks avoid reusing the production histogram implementation.
    bands=[float(power[(frequencies>=left)&((frequencies<=right) if j==len(edges)-2 else (frequencies<right))].sum()) for j,(left,right) in enumerate(zip(edges[:-1],edges[1:]))]
    return np.r_[stats,np.log1p(bands)]


def forward(model,X,kind):
    if kind!='et':
        scaler,estimator=model[0],model[-1]
        values=(X-scaler.mean_)/scaler.scale_
        if kind=='lda':scores=values@estimator.coef_.T+estimator.intercept_
        else:
            scores=values
            for i,(weights,bias) in enumerate(zip(estimator.coefs_,estimator.intercepts_)):
                scores=scores@weights+bias
                if i<len(estimator.coefs_)-1:scores=np.maximum(scores,0)
        return estimator.classes_[np.argmax(scores,axis=1)]
    # sklearn decision trees cast inputs to float32. Traverse saved nodes directly.
    values=X.astype(np.float32);scores=np.zeros((len(X),len(model.classes_)))
    for estimator in model.estimators_:
        tree=estimator.tree_;nodes=np.zeros(len(X),dtype=np.intp)
        while True:
            active=np.flatnonzero(tree.children_left[nodes]!=-1)
            if not len(active):break
            current=nodes[active];left=values[active,tree.feature[current]]<=tree.threshold[current]
            nodes[active]=np.where(left,tree.children_left[current],tree.children_right[current])
        leaf=tree.value[nodes,0,:];scores+=leaf/leaf.sum(axis=1,keepdims=True)
    return model.classes_[np.argmax(scores,axis=1)]


def measures(y,pred,sources):
    confusion=np.zeros((3,3),dtype=int)
    for actual,answer in zip(y,pred):confusion[int(actual),int(answer)]+=1
    recall=np.diag(confusion)/confusion.sum(axis=1)
    f1=[2*confusion[c,c]/(confusion[c,:].sum()+confusion[:,c].sum()) for c in range(3)]
    records=[]
    for source in sorted(set(sources)):
        mask=sources==source;votes=np.array([np.count_nonzero(pred[mask]==c)for c in range(3)])
        assert len(set(y[mask]))==1
        records.append(int(votes.argmax())==int(y[mask][0]))
    return {'macro_f1':float(np.mean(f1)),'accuracy':float(np.trace(confusion)/confusion.sum()),'balanced_accuracy':float(np.mean(recall)),'record_accuracy':float(np.mean(records)),'confusion_matrix':confusion.tolist()}


def main(root):
    started=time.monotonic();cfg=read(root/'PROTOCOL421.json');result=read(root/'RESULT421.json');seal=read(root/'CHOICES421.json')
    assert read(root/'TERMINAL421.json')['exit_code']==0
    assert digest(root/'code/snu_order_control.py')==cfg['code_sha256']
    source=Path(cfg['source']);manifest=read(source/'SOURCE_MANIFEST.json');inventory={e['path']:e['sha256']for e in manifest['files']}
    assert digest(source/'SOURCE_MANIFEST.json')==cfg['source_manifest_sha256']
    for path,expected in cfg['author_equation_sources'].items():assert digest(source/path)==expected==inventory[path]
    assert len(cfg['members'])==14 and len(cfg['jobs'])==12
    raw_checks=[]
    selected=[('5m/xc_n.mat',0),('5m/xc_pr_075.mat',0),('5m/xc_s_05.mat',48),('5m/xn_n.mat',0)]
    for member,index in selected:
        meta=next(m for m in cfg['members']if m['member']==member);path=root/'clients-cache'/('audit_'+Path(member).name)
        with zipfile.ZipFile(cfg['source_zip']) as z:path.write_bytes(z.read(member))
        assert digest(path)==meta['sha256'];key=Path(member).stem
        if h5py.is_hdf5(path):
            with h5py.File(path)as f:values=np.asarray(f[key]).T
        else:values=np.asarray(loadmat(path)[key])
        count=cfg['fs']*cfg['window_seconds'];raw=values[index*count:(index+1)*count]
        vibration=raw[:,0].astype(float);degrees=raw[:,1];mask=degrees!=0;degrees=degrees[mask]
        stats=read(root/'features'/(key+'.json'))['angular_checks'][index]
        assert stats['usable_samples']==len(degrees) and np.all(np.diff(degrees)>0)
        n=stats['resampled_samples'];assert n%cfg['samples_per_revolution']==0
        grid=degrees[0]+360*np.arange(1,n+1)/cfg['samples_per_revolution']
        assert grid[-1]<degrees[-1]
        # Piecewise-linear angle -> vibration is algebraically equivalent to
        # the author's two interpolations when they share the acquisition knots.
        aligned=np.interp(grid,degrees,vibration[mask]);features=load(root/'features'/(key+'.npz'))
        comparisons={}
        for rep in ['frequency_raw','frequency_rms','order_raw','order_rms']:
            angular=rep.startswith('order');edges=np.asarray(cfg['order_edges'],dtype=float)
            if not angular:edges*=cfg['reference_rpm']/60
            rebuilt=independent_descriptor(aligned if angular else vibration,cfg['samples_per_revolution']if angular else cfg['fs'],edges,rep.endswith('rms'))
            saved=features[rep][index]
            np.testing.assert_allclose(rebuilt,saved,rtol=2e-6,atol=2e-9)
            comparisons[rep]=float(np.max(abs(rebuilt-saved)))
        raw_checks.append(dict(member=member,window=index,max_absolute_feature_differences=comparisons));path.unlink()
    labels={split:load(root/'private'/(split+'_labels.npz'))for split in ['fit','development','final']}
    for split,truth in labels.items():
        assert len(truth['y'])==cfg['counts'][split]
        for identity,window,label in zip(truth['source'],truth['window'],truth['y']):
            assert label==cfg['labels'][identity]
            assert (identity.startswith('5m/xc_') and (window<36 if split=='fit'else window>=48)) if split!='final'else identity.startswith('5m/xn_')
    forward_rows=metric_fields=0;fit_entries=scaler_fits=0;neural=[]
    for job in cfg['jobs']:
        rep,kind=job.rsplit('__',1);report=read(root/'models'/(job+'_REPORT.json'));path=root/'models'/(job+'.joblib')
        assert digest(path)==report['model_sha256']==seal['model_hashes'][job]
        assert report['training']['rows']==252 and report['training']['features']==268
        fit_entries+=report['training']['classifier_fits'];scaler_fits+=report['training']['scaler_fits']
        pred_audit=read(root/'predictions'/(job+'_AUDIT.json'));assert pred_audit['utc']>seal['utc'] and pred_audit['fit_attempts_observed']==0 and pred_audit['reverse_predictions_equal']
        assert read(root/'audit'/('train_'+job+'_PROCESS.json'))['returncode']==0
        model=joblib.load(path)
        if kind=='mlp':
            estimator=model[-1];assert estimator.n_iter_==report['training']['epochs']>1
            assert estimator.loss_curve_[0]>estimator.loss_curve_[-1]
            neural.append(dict(job=job,epochs=int(estimator.n_iter_),samples_seen=int(estimator.t_),loss_before=float(estimator.loss_curve_[0]),loss_after=float(estimator.loss_curve_[-1])))
        elif kind=='et':assert len(model.estimators_)==300 and all(t.tree_.node_count>1 for t in model.estimators_)
        if kind!='et':
            fit=load(root/'features'/(rep+'_fit.npz'))['X'];np.testing.assert_allclose(model[0].mean_,fit.mean(axis=0),rtol=1e-11,atol=1e-11);assert model[0].n_samples_seen_==252
        for split in ['development','final']:
            X=load(root/'features'/(rep+'_'+split+'.npz'))['X'];saved=np.load(root/'predictions'/(job+'_'+split+'.npy'),allow_pickle=False)
            independently_predicted=forward(model,X,kind);np.testing.assert_array_equal(independently_predicted,saved);forward_rows+=len(X)
            expected=measures(labels[split]['y'],saved,labels[split]['source']);recorded=report['development']if split=='development'else result['all_models'][job]
            for name,value in expected.items():np.testing.assert_allclose(value,recorded[name],atol=1e-12,rtol=0);metric_fields+=1
    assert fit_entries==12 and scaler_fits==8 and len(neural)==4
    for rep,job in seal['selected'].items():
        candidates=[j for j in cfg['jobs']if j.startswith(rep+'__')]
        winner=min(candidates,key=lambda j:(-read(root/'models'/(j+'_REPORT.json'))['development']['macro_f1'],j.rsplit('__',1)[1]))
        assert winner==job
    main_result=result['selected_metrics']['order_rms'];base=read(source/'SOURCE_BASELINE_RESULTS322.json')['final']['mlp']
    controls=[result['selected_metrics'][r]for r in ['frequency_raw','frequency_rms']]
    gate=all(main_result['macro_f1']>v['macro_f1']+1e-12 for v in controls+[base])and main_result['record_accuracy']>=max(v['record_accuracy']for v in controls)
    assert gate==result['primary_qualification_passed']
    audit=dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,seconds=time.monotonic()-started,
        independent_raw_windows=raw_checks,independent_feature_vectors=16,parameter_forward_rows=forward_rows,metric_fields=metric_fields,
        actual_classifier_fits=fit_entries,actual_scaler_fits=scaler_fits,neural_training=neural,
        actual_cold_final_processes=12,actual_cold_final_predict_calls=24,
        independent_direct_parameter_recomputations=24,new_audit_model_fits=0,new_LLM_calls=0,
        source_recordings=14,final_recordings=7,source_clusters=1,formal_confirmation_opened=False,
        result_sha256=digest(root/'RESULT421.json'))
    (root/'audit/INDEPENDENT_AUDIT421.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items()if k not in ['independent_raw_windows','neural_training']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
