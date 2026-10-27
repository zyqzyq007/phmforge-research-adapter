"""Run frozen HUST correction models on real held-family records, outside routing gates.

This is an exploratory component probe, not an autonomous Agent or physical intervention.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def worker(actor):
    import joblib
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    sys.path.insert(0,'/app')
    def forbidden_fit(*args, **kwargs):
        raise RuntimeError('This frozen-checkpoint probe forbids all fitting')
    for cls in [Pipeline,StandardScaler,LogisticRegression,SVC,LinearDiscriminantAnalysis]:
        cls.fit=forbidden_fit
    if actor=='b':
        from hust_features import extract_record_features
        from train_solution import labels_from_components
        checkpoint=Path('/app/artifacts/hust_component_model.joblib')
        model_names=['IR','OR','BALL','CLASS5']
    else:
        from order_features import extract_order_features
        checkpoint=Path('/app/artifacts/hust_record_model.joblib')
        model_names=['base_model','healthy_ir_model','compound_model']
    artifact=joblib.load(checkpoint)
    counter=[0]
    def call(model,x,prob=False):
        counter[0]+=1
        return model.predict_proba(x)[:,1] if prob else model.predict(x)
    datasets={}
    for part in ['train','val']:
        with np.load('/data/'+part+'.npz',allow_pickle=False) as d:
            if actor=='b':
                features,inverse=extract_record_features(d)
            else:
                window_features,inverse=extract_order_features(d['waveforms'],d['shaft_frequency_hz'],
                                                                 d['native_record_mean'],d['native_record_scale'])
                features=np.stack([window_features[inverse==i].mean(0) for i in np.unique(inverse)])
            groups=np.unique(inverse)
            assert np.array_equal(groups,np.arange(len(features)))
            meta=[]
            for i in groups:
                row={k:np.unique(d[k][inverse==i]) for k in ['labels','bearing_family','native_record_scale','groups']}
                assert all(len(v)==1 for v in row.values())
                meta.append(dict(label=int(row['labels'][0]),family=str(row['bearing_family'][0]),
                                 scale=float(row['native_record_scale'][0]),record_id=str(row['groups'][0]),
                                 windows=int(np.sum(inverse==i))))
            datasets[part]=dict(x=features,inverse=inverse,meta=meta,ids=d['sample_ids'].astype(str))
    train=datasets['train'];labels=np.array([r['label'] for r in train['meta']]);family=np.array([r['family'] for r in train['meta']])
    assert len(labels)==57 and set(family)=={'6204','6205','6206'}
    assert len(datasets['val']['meta'])==21 and {r['family'] for r in datasets['val']['meta']}=={'6207'}
    lineage=[]
    def verify_scaler(model, expected, name, fold):
        scaler=model.named_steps['standardscaler']
        delta=float(np.max(np.abs(scaler.mean_-expected.astype(float).mean(0))))
        assert int(scaler.n_samples_seen_)==len(expected) and delta<1e-10
        lineage.append(dict(model=name,fold=fold,fitting_records=len(expected),scaler_mean_max_abs=delta))
    rows=[]
    for fold in ['6204','6205','6206','val']:
        if fold=='val':
            d=datasets['val'];indices=np.arange(len(d['meta']));fit=np.ones(len(labels),bool)
            models=artifact['models'] if actor=='b' else {n:artifact[n] for n in model_names}
        else:
            d=train;indices=np.flatnonzero(family==fold);fit=family!=fold
            models={n:joblib.load('/folds/'+fold+'_'+n+'.joblib') for n in model_names}
        x=d['x'][indices]
        scale=np.array([d['meta'][i]['scale'] for i in indices])
        if actor=='b':
            for n in model_names:verify_scaler(models[n],train['x'][fit],n,fold)
            probs={n:call(models[n],x,True) for n in model_names}
            assert float(artifact['threshold'])==.5
            base=labels_from_components(np.column_stack([probs[n] for n in artifact['component_order']]),float(artifact['threshold']))
            threshold=float(artifact['class5_promotion_threshold'])
            shadow=np.where(probs['CLASS5']>threshold,5,4)
            before=base.copy();gate=base==4;full=base.copy();full[gate]=shadow[gate]
        else:
            matrix=train['x'][:,891:1188]
            expected={'base_model':matrix[fit],
                      'healthy_ir_model':np.log(np.array([r['scale'] for r in train['meta']])[fit&np.isin(labels,[0,1]),None]+1e-12),
                      'compound_model':matrix[fit&np.isin(labels,[4,5])]}
            for n in model_names:verify_scaler(models[n],expected[n],n,fold)
            base=np.asarray(call(models['base_model'],x[:,891:1188]),int)
            hi=np.asarray(call(models['healthy_ir_model'],np.log(scale[:,None]+1e-12)),int)
            shadow=np.asarray(call(models['compound_model'],x[:,891:1188]),int)
            before=base.copy()
            before[(base==3)&(hi==0)]=0
            before[np.isin(base,[0,1])&(hi==1)]=1
            gate=np.isin(base,[4,5]);full=before.copy();full[gate]=shadow[gate]
            threshold=None
        for j,i in enumerate(indices):
            row=dict(d['meta'][i],fold=fold,base=int(base[j]),before_target_component=int(before[j]),
                     complete=int(full[j]),natural_gate=bool(gate[j]),shadow_pair_prediction=int(shadow[j]))
            if actor=='b':row['class5_probability']=float(probs['CLASS5'][j])
            rows.append(row)
        if fold=='val':
            for name,pred in [('base',base),('full',full)]:
                saved=json.loads(Path('/folds/val_'+name+'.json').read_text())['predictions']
                expected={str(i):int(v) for i,v in zip(d['ids'],pred[d['inverse']],strict=True)}
                assert expected=={r['sample_id']:r['predicted_label'] for r in saved}
        save(Path('/output/PARTIAL.json'),dict(rows=rows,predict_calls=counter[0]))
    save(Path('/output/EVIDENCE.json'),dict(actor=actor,rows=rows,lineage=lineage,
         actual_estimator_predict_calls=counter[0],new_fits=0,new_LLM_calls=0,
         val_base_and_full_replay_matched=True,checkpoint_sha256=sha(checkpoint),
         original_b_promotion_threshold=threshold,source_labels_used_only_for_grouping_audit_and_evaluation=True,
         no_synthetic_signals_or_labels=True,final_data_mounted=False))


def run(root):
    assert not (root/'HANDLE388.json').exists()
    cfg=json.loads((root/'PROTOCOL388.json').read_text())
    save(root/'HANDLE388.json',dict(pid=os.getpid(),start_unix=time.time()))
    start=time.monotonic()
    def job(actor):
        plan=cfg['actors'][actor];out=root/'workers'/actor;out.mkdir()
        (out/'runtime/tmp').mkdir(parents=True);(out/'runtime/mpl').mkdir(parents=True)
        command=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net',
                 '--die-with-parent','--new-session']
        for path in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:command+=['--ro-bind',path,path]
        command+=['--ro-bind',cfg['python_root'],'/opt/python','--ro-bind',cfg['deps'],'/opt/deps',
                  '--ro-bind',plan['workspace'],'/app','--ro-bind',plan['folds'],'/folds',
                  '--ro-bind',str(root/'code'),'/code','--dir','/data',
                  '--ro-bind',plan['data']+'/train.npz','/data/train.npz',
                  '--ro-bind',plan['data']+'/val.npz','/data/val.npz',
                  '--bind',str(out),'/output','--proc','/proc','--dev','/dev',
                  '--chdir','/app','--remount-ro','/',
                  '/opt/python/bin/python3.12','-B','/code/probe_update_branches.py','worker','--actor',actor]
        save(out/'COMMAND.json',command)
        env=dict(os.environ,PYTHONPATH='/opt/deps',TMPDIR='/output/runtime/tmp',TMP='/output/runtime/tmp',
                 TEMP='/output/runtime/tmp',MPLCONFIGDIR='/output/runtime/mpl',PYTHONDONTWRITEBYTECODE='1',
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/'stdout').open('w') as stdout,(out/'stderr').open('w') as stderr:
            try:code=subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=600).returncode
            except subprocess.TimeoutExpired:code=124
        save(out/'TERMINAL.json',dict(exit_code=code))
        return dict(actor=actor,exit_code=code)
    states=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,a) for a in ['b','c']]):
            states.append(future.result());save(root/'PROGRESS388.json',states);print(json.dumps(states[-1]),flush=True)
    ok=all(s['exit_code']==0 for s in states)
    if ok:
        save(root/'EVIDENCE388.json',{a:json.loads((root/'workers'/a/'EVIDENCE.json').read_text()) for a in ['b','c']})
        save(root/'EVIDENCE_SEAL388.json',dict(sha256=sha(root/'EVIDENCE388.json'),sealed_unix=time.time()))
    save(root/'TERMINAL388.json',dict(exit_code=0 if ok else 1,rows=states,seconds=time.monotonic()-start))
    if not ok:raise RuntimeError('Retain failures; no automatic scientific retry')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['run','worker']);parser.add_argument('--run',type=Path);parser.add_argument('--actor',choices=['b','c'])
    args=parser.parse_args()
    worker(args.actor) if args.action=='worker' else run(args.run)
