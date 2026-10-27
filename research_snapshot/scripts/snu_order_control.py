"""Real SNU signal controls: frequency/order representation x amplitude scaling.

Known signal processing, not an Agent method or original PHMForge completion.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np


REPRESENTATIONS = ['frequency_raw', 'frequency_rms', 'order_raw', 'order_rms']
ESTIMATORS = ['et', 'lda', 'mlp']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def packet(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def statistics(signal):
    mean, std = signal.mean(), signal.std()
    rms = float(np.sqrt(np.mean(signal*signal)))
    peak, absolute = float(np.max(np.abs(signal))), float(np.mean(np.abs(signal)))
    centered = signal-mean
    return [float(mean),float(std),rms,peak,float(np.ptp(signal)),absolute,
        float(np.mean(centered**4)/max(std**4,1e-30)),
        float(np.mean(centered**3)/max(std**3,1e-30)),
        peak/max(rms,1e-15),rms/max(absolute,1e-15),peak/max(absolute,1e-15),
        float(np.mean(signal[1:]*signal[:-1] < 0))]


def descriptor(signal, sample_rate, edges, normalized):
    if normalized:
        signal = signal/max(float(np.sqrt(np.mean(signal*signal))),1e-15)
    power = abs(np.fft.rfft(signal-signal.mean()))**2/len(signal)**2
    frequency = np.fft.rfftfreq(len(signal), d=1/sample_rate)
    bands = np.histogram(frequency,bins=edges,weights=power)[0]
    return np.asarray(statistics(signal)+np.log1p(bands).tolist())


def angular_resample(vibration, degrees, fs, samples_per_revolution):
    good = degrees != 0
    t = np.arange(len(vibration),dtype=float)[good]/fs
    vibration, degrees = vibration[good], degrees[good]
    assert len(degrees)>2 and np.all(np.diff(degrees)>0)
    step = 360/samples_per_revolution
    count = int(np.floor((degrees[-1]-degrees[0])/step))-1
    count -= count % samples_per_revolution
    assert count >= samples_per_revolution
    grid = degrees[0]+step*np.arange(1,count+1,dtype=float)
    # Author z_resampling.m: angle -> acquisition time -> vibration.
    at = np.interp(grid,degrees,t)
    result = np.interp(at,t,vibration)
    return result,dict(original_samples=len(good),usable_samples=len(degrees),
        resampled_samples=len(result),revolutions=count//samples_per_revolution,
        degree_start=float(degrees[0]),degree_end=float(degrees[-1]),
        max_grid_degree=float(grid[-1]),minimum_angle_increment=float(np.diff(degrees).min()))


def decode(root, cfg, member):
    import h5py
    from scipy.io import loadmat
    target = root/'clients-cache'/Path(member['member']).name
    with zipfile.ZipFile(cfg['source_zip']) as archive:
        target.write_bytes(archive.read(member['member']))
    assert sha(target)==member['sha256']
    key=Path(member['member']).stem
    if h5py.is_hdf5(target):
        with h5py.File(target) as data: values=np.asarray(data[key]).T
    else:
        values=np.asarray(loadmat(target)[key])
    assert values.shape==(7680001,3) and np.isfinite(values).all()
    return target,values


def extract_member(root,cfg,member):
    started=time.monotonic();path,values=decode(root,cfg,member)
    fs,step=cfg['fs'],cfg['window_seconds']*cfg['fs']
    order_edges=np.asarray(cfg['order_edges']);frequency_edges=order_edges*cfg['reference_rpm']/60
    results={name:[] for name in REPRESENTATIONS};rows=[];coarse=None;checks=[]
    for index in range(60):
        raw=values[index*step:(index+1)*step]
        signal=raw[:,0].astype(float)
        aligned,check=angular_resample(signal,raw[:,1],fs,cfg['samples_per_revolution'])
        for representation in REPRESENTATIONS:
            angular=representation.startswith('order')
            results[representation].append(descriptor(aligned if angular else signal,
                cfg['samples_per_revolution'] if angular else fs,
                order_edges if angular else frequency_edges,representation.endswith('rms')))
        split=('fit' if index<36 else 'development' if index>=48 else 'gap') if Path(member['member']).stem.startswith('xc_') else 'final'
        rows.append(dict(source=member['member'],window=index,split=split,
            label=cfg['labels'][member['member']]))
        angular_rpm=(raw[-1,1]-raw[0,1])/360*60/((len(raw)-1)/fs)
        # Author Data_region_extract.mlx saves Speed = RPM / 60 in column 3.
        reported_rps = float(raw[:, 2].mean())
        check.update(index=index,angle_derived_rpm=float(angular_rpm),reported_mean_rps=reported_rps,
            rpm_relative_difference=float(abs(angular_rpm-60*reported_rps)/max(abs(angular_rpm),1e-12)))
        checks.append(check)
        if index==0:
            power=abs(np.fft.rfft(signal-signal.mean()))**2/len(signal)**2
            coarse=np.asarray(statistics(signal)+[float(np.log1p(v.sum())) for v in np.array_split(power,32)])
    target=root/'features'/(path.stem+'.npz')
    np.savez_compressed(target,**{name:np.asarray(v) for name,v in results.items()})
    save(root/'features'/(path.stem+'.json'),dict(source=member['member'],source_sha256=member['sha256'],rows=rows,
        angular_checks=checks,coarse_first_window=coarse.tolist(),seconds=time.monotonic()-started))
    # Exact source exists in the verified ZIP; this is a declared rebuildable copy.
    path.unlink()
    return dict(member=member['member'],feature_path=str(target.relative_to(root)),sha256=sha(target),seconds=time.monotonic()-started)


def features(root,cfg):
    started=time.monotonic();assert sha(cfg['source_zip'])==cfg['source_zip_sha256']
    with ThreadPoolExecutor(max_workers=cfg['parallel']) as executor:
        futures=[executor.submit(extract_member,root,cfg,m) for m in cfg['members']]
        receipts=[f.result() for f in futures]
    rows=[];matrix={name:[] for name in REPRESENTATIONS}
    baseline={split:packet(Path(cfg['source'])/('features_'+split+'.npz')) for split in ['fit','development','final']}
    reference_checks=0
    for receipt in receipts:
        path=root/receipt['feature_path'];meta=read(path.with_suffix('.json'));data=packet(path)
        split=meta['rows'][0]['split'];old=baseline[split]
        idx=np.flatnonzero((old['source']==receipt['member']) & (old['window']==0))
        assert len(idx)==1
        np.testing.assert_allclose(meta['coarse_first_window'],old['X'][idx[0]],rtol=1e-12,atol=1e-12)
        reference_checks+=1;rows.extend(meta['rows'])
        for name in REPRESENTATIONS:matrix[name].extend(data[name])
    for split in ['fit','development','final']:
        indices=[i for i,row in enumerate(rows) if row['split']==split]
        metadata=dict(y=np.asarray([rows[i]['label'] for i in indices]),source=np.asarray([rows[i]['source'] for i in indices]),window=np.asarray([rows[i]['window'] for i in indices]))
        np.savez_compressed(root/'private'/(split+'_labels.npz'),**metadata)
        for name in REPRESENTATIONS:
            X=np.asarray(matrix[name])[indices]
            assert X.shape==(cfg['counts'][split],268) and np.isfinite(X).all()
            np.savez_compressed(root/'features'/(name+'_'+split+'.npz'),X=X)
    save(root/'FEATURES421.json',dict(utc=utc(),members=receipts,reference_reproductions=reference_checks,
        seconds=time.monotonic()-started,shape_features=268,source_labels_unchanged=True,
        simulated_signals=False,preprocessing='Interpolation of actual measured vibration using actual encoder angles;not synthetic faults.'))


def metric(truth,prediction):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score
    per_record=[]
    for identity in sorted(set(truth['source'])):
        mask=truth['source']==identity;counts=np.bincount(prediction[mask].astype(int),minlength=3)
        label=int(truth['y'][mask][0]);assert np.all(truth['y'][mask]==label)
        per_record.append(dict(source=str(identity),label=label,prediction=int(np.argmax(counts)),
            correct=bool(np.argmax(counts)==label),window_accuracy=float(np.mean(prediction[mask]==label))))
    return dict(macro_f1=float(f1_score(truth['y'],prediction,labels=[0,1,2],average='macro',zero_division=0)),
        accuracy=float(accuracy_score(truth['y'],prediction)),balanced_accuracy=float(balanced_accuracy_score(truth['y'],prediction)),
        confusion_matrix=confusion_matrix(truth['y'],prediction,labels=[0,1,2]).tolist(),
        record_accuracy=float(np.mean([r['correct'] for r in per_record])),per_record=per_record)


def train(root,cfg,job):
    import joblib
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    representation,kind=job.rsplit('__',1);X=packet(root/'features'/(representation+'_fit.npz'))['X']
    y=packet(root/'private/fit_labels.npz')['y'];started=time.monotonic()
    if kind=='et':model=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,random_state=322,n_jobs=2)
    elif kind=='lda':model=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
    else:model=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),max_iter=300,random_state=322))
    assert len(y)==252 and set(y)=={0,1,2}
    save(root/'models'/(job+'_STARTED.json'),dict(utc=utc(),rows=len(y),features=X.shape[1],kind=kind))
    model.fit(X,y)
    estimator=model if kind=='et' else model[-1]
    training=dict(classifier_fits=1,scaler_fits=int(kind!='et'),rows=len(y),features=X.shape[1],classes=estimator.classes_.tolist())
    if kind=='mlp':
        assert estimator.n_iter_>1 and np.isfinite(estimator.loss_curve_).all()
        training.update(epochs=int(estimator.n_iter_),samples_seen=int(estimator.t_),loss_curve=list(map(float,estimator.loss_curve_)),
            parameter_shapes=[list(a.shape) for a in estimator.coefs_+estimator.intercepts_],parameter_norms=[float(np.linalg.norm(a)) for a in estimator.coefs_+estimator.intercepts_])
    elif kind=='et':
        training.update(actual_trees=len(estimator.estimators_),nodes=[int(t.tree_.node_count) for t in estimator.estimators_]);assert min(training['nodes'])>1
    else:
        training.update(coefficient_norm=float(np.linalg.norm(estimator.coef_)),coefficient_shape=list(estimator.coef_.shape));assert training['coefficient_norm']>0
    model_path=root/'models'/(job+'.joblib');joblib.dump(model,model_path)
    dev=packet(root/'features'/(representation+'_development.npz'))['X'];pred=model.predict(dev)
    np.save(root/'predictions'/(job+'_development.npy'),pred)
    save(root/'models'/(job+'_REPORT.json'),dict(utc=utc(),job=job,training=training,model_sha256=sha(model_path),
        development=metric(packet(root/'private/development_labels.npz'),pred),seconds=time.monotonic()-started,final_labels_read=False))


def predict(root,cfg,job):
    import joblib
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    seal=read(root/'CHOICES421.json');report=read(root/'models'/(job+'_REPORT.json'))
    model_path=root/'models'/(job+'.joblib');assert sha(model_path)==report['model_sha256']==seal['model_hashes'][job]
    attempted=[]
    def forbidden(*args,**kwargs):
        attempted.append(True);raise RuntimeError('Fitting prohibited in cold prediction')
    for cls in [LinearDiscriminantAnalysis,ExtraTreesClassifier,MLPClassifier,StandardScaler]:cls.fit=forbidden
    model=joblib.load(model_path);rep=job.rsplit('__',1)[0];X=packet(root/'features'/(rep+'_final.npz'))['X']
    pred=model.predict(X);reverse=model.predict(X[::-1])[::-1]
    np.testing.assert_array_equal(pred,reverse);assert not attempted
    np.save(root/'predictions'/(job+'_final.npy'),pred)
    save(root/'predictions'/(job+'_AUDIT.json'),dict(utc=utc(),rows=len(X),predict_calls=2,
        fit_attempts_observed=len(attempted),reverse_predictions_equal=True,model_sha256=sha(model_path),final_labels_read=False))


def seal(root,cfg):
    reports={job:read(root/'models'/(job+'_REPORT.json')) for job in cfg['jobs']}
    selected={rep:min([j for j in cfg['jobs'] if j.startswith(rep+'__')],key=lambda j:(-reports[j]['development']['macro_f1'],j.rsplit('__',1)[1])) for rep in REPRESENTATIONS}
    save(root/'CHOICES421.json',dict(utc=utc(),selected=selected,development={k:v['development'] for k,v in reports.items()},
        model_hashes={k:v['model_sha256'] for k,v in reports.items()},final_prediction_files_exist=any((root/'predictions').glob('*_final.npy'))))
    assert not read(root/'CHOICES421.json')['final_prediction_files_exist']


def score(root,cfg):
    seal=read(root/'CHOICES421.json');truth=packet(root/'private/final_labels.npz')
    values={job:metric(truth,np.load(root/'predictions'/(job+'_final.npy'),allow_pickle=False)) for job in cfg['jobs']}
    baseline=read(Path(cfg['source'])/'SOURCE_BASELINE_RESULTS322.json')
    primary=values[seal['selected']['order_rms']]
    comparisons={rep:values[seal['selected'][rep]] for rep in REPRESENTATIONS}
    controls=[comparisons['frequency_raw'],comparisons['frequency_rms'],baseline['final']['mlp']]
    gate=all(primary['macro_f1']>c['macro_f1']+1e-12 for c in controls) and primary['record_accuracy']>=max(comparisons[r]['record_accuracy'] for r in ['frequency_raw','frequency_rms'])
    save(root/'RESULT421.json',dict(utc=utc(),primary='order_rms',selected=seal['selected'],selected_metrics=comparisons,all_models=values,
        primary_qualification_passed=gate,old_coarse_fixed_baselines=baseline['final'],new_classifier_fits=len(cfg['jobs']),
        new_scaler_fits=8,new_neural_models=4,new_Agent_sessions=0,new_LLM_calls=0,
        source_clusters=1,final_recordings=7,final_windows=420,independent_confirmation=False,
        original_PHMForge_task_complete=False,novelty_established=False,formal_confirmation_panel_opened=False))
    print(json.dumps({'selected':seal['selected'],'primary_gate':gate,'selected_metrics':{k:{m:v[m] for m in ['macro_f1','accuracy','record_accuracy']} for k,v in comparisons.items()}}),flush=True)


def run_stage(root,cfg,action,job=None):
    command=[sys.executable,'-B',str(root/'code/snu_order_control.py'),action,'--run',str(root)]
    if job:command+=['--job',job]
    name=action+('_'+job if job else '');started=time.monotonic()
    with (root/'audit'/(name+'.stdout')).open('w') as out,(root/'audit'/(name+'.stderr')).open('w') as err:
        process=subprocess.Popen(command,stdout=out,stderr=err)
        save(root/'audit'/(name+'_HANDLE.json'),dict(pid=process.pid,command=command,utc=utc()))
        try:code=process.wait(timeout=cfg['timeouts'][action])
        except subprocess.TimeoutExpired:process.kill();code=process.wait()
    save(root/'audit'/(name+'_PROCESS.json'),dict(command=command,returncode=code,seconds=time.monotonic()-started))
    return dict(stage=name,exit_code=code,seconds=time.monotonic()-started)


def run(root,cfg):
    started=time.monotonic();history=[]
    save(root/'STARTED421.json',dict(utc=utc(),controller_pid=__import__('os').getpid()))
    for phase in ['features','train','seal','predict','score']:
        if phase in ['train','predict']:
            with ThreadPoolExecutor(max_workers=cfg['parallel']) as pool:
                futures=[pool.submit(run_stage,root,cfg,phase,j) for j in cfg['jobs']]
                outputs=[f.result() for f in futures]
        else:outputs=[run_stage(root,cfg,phase)]
        history+=outputs;save(root/'PROGRESS421.json',dict(utc=utc(),completed=history,current_phase=phase))
        print(json.dumps({'phase':phase,'completed':len(outputs),'failed':sum(r['exit_code']!=0 for r in outputs),'seconds':time.monotonic()-started}),flush=True)
        if any(r['exit_code'] for r in outputs):
            save(root/'TERMINAL421.json',dict(utc=utc(),exit_code=1,failed_phase=phase,seconds=time.monotonic()-started));raise SystemExit(1)
    save(root/'TERMINAL421.json',dict(utc=utc(),exit_code=0,seconds=time.monotonic()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['features','train','seal','predict','score','run']);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--job')
    args=parser.parse_args();root=args.run.resolve();cfg=read(root/'PROTOCOL421.json')
    assert sha(root/'code/snu_order_control.py')==cfg['code_sha256']
    if args.action in ['train','predict']:globals()[args.action](root,cfg,args.job)
    else:globals()[args.action](root,cfg)
