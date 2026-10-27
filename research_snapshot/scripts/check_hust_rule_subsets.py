"""Fixed actual-rule factorial probe; exploratory, no threshold or feature search."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261022T023131Z_validation274')
APP = SOURCE/'a_review/evaluation/s274_hust_record_r1_review/app'
PUBLIC = SOURCE/'a_review/public_data/s274_hust_record_r1_review/datasets/HUST'
PYTHON = '/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu'
DEPS = '/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps'
MASKS = ['none', 'class1', 'class5', 'both']

def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def utc():return datetime.now(timezone.utc).isoformat()

def prepare(root):
    root.mkdir(exist_ok=False)
    for folder in ['code','app','runtime/tmp','runtime/mpl','workers']:(root/folder).mkdir(parents=True)
    inputs={'train.npz':PUBLIC/'train.npz','dev.npz':PUBLIC/'val.npz',
        'train_features.npz':APP/'.runtime/train_features_v2.npz',
        'dev_features.npz':APP/'.runtime/val_features_v2.npz',
        'original.joblib':APP/'artifacts/hust_hybrid.joblib',
        'before.joblib':APP/'artifacts/hust_band_svc.joblib'}
    external={'observations':SOURCE/'a_review/evaluation_data/s274_hust_record_r1_review/datasets/HUST/test.npz',
        'labels':SOURCE/'a_review/hidden/HUST/labels.json',
        'before_prediction':SOURCE/'a/evaluation/s274_hust_record_r1_initial/output/predictions.json',
        'after_prediction':SOURCE/'a_review/evaluation/s274_hust_record_r1_review/output/predictions.json'}
    m=SOURCE/'SOURCE_MANIFEST.json'
    assert sha(m)=='ed5a5e4eef7ecff596f99c91d2d6c5de83c0fa9f351994f9f6f0e0118fdc357f'
    inventory={x['path']:x for x in json.loads(m.read_text())['files']}
    bindings=[]
    paths=list(inputs.values())+list(external.values())+[APP/x for x in ['solution.py','train_hybrid.py','feature_extractor.py']]
    for p in paths:
        entry=inventory[str(p.relative_to(SOURCE))]
        assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
        bindings.append(dict(source_path=str(p),**entry))
    for name in ['solution.py','train_hybrid.py','feature_extractor.py']:shutil.copy2(APP/name,root/'app'/name)
    code=(root/'app/solution.py').read_text()
    for mask in MASKS:
        variant=code.replace('if p_class_5 > artifact["class5_probability_threshold"]:',
            f'if {mask in ("class5","both")} and p_class_5 > artifact["class5_probability_threshold"]:')
        variant=variant.replace('elif p_class_1 > artifact["class1_probability_threshold"]:',
            f'elif {mask in ("class1","both")} and p_class_1 > artifact["class1_probability_threshold"]:')
        assert variant!=code
        (root/'app'/f'{mask}.py').write_text(variant)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    plan=dict(iteration='A438',utc=utc(),source_archive=str(SOURCE),bindings=bindings,
        inputs={k:str(v) for k,v in inputs.items()},external={k:str(v) for k,v in external.items()},
        hypothesis='Whole-update validation can hide separable useful and harmful rule changes; original-threshold subsets may preserve a real improvement without its collateral harm.',
        variants=MASKS,held_families=['6204','6205'],
        unavailable_fold={'6206':'Original primary explicitly trains only6206; excluding6206 leaves zero primary training rows. No substitute primary or fabricated full-CV score.'},
        primary_selection='Among masks with development window error no worse than none, minimize pooled window error across both eligible whole-family folds; ties use lower development error, fewer enabled rules, then fixed mask order.',
        primary_gate='Selected partial mask strictly improves external window error over both original none and both; harm relative to none no higher than both and gain no lower than both.',
        controls=['original none','actual ordinary both','development-only optimum','eligible complete-program CV optimum'],
        budget={'new_classifier_fits':4,'new_scaler_fits':4,'new_PCA_fits':4,'primary_SVC_fits':0,'LLM_calls':0,'parallel_workers':3,'worker_seconds':300,'numeric_threads':1,'disk_bytes':2000000000,'GPU_hours':0},
        frozen_parameters=True,synthetic_data=False,source_already_opened=True,independent_confirmation=False,
        original_scope_override='One original-parameter subset capability check, no feature/threshold/seed search; ongoing method goal authorizes exploratory checks, not reuse as confirmation.',
        limitations=['One historical source and split; no native Agent method or independent majority comparison.',
            'Only2/3 family folds executable under original primary training contract; report missing coverage.',
            'Rule subset construction and partial revalidation have prior art; no novelty assertion.'],
        code_hashes={str(p.relative_to(root)):sha(p) for base in ['app','code'] for p in (root/base).glob('*.py')})
    save(root/'PROTOCOL438.json',plan)
    save(root/'FREEZE438.json',{'protocol_sha256':sha(root/'PROTOCOL438.json'),'utc':utc()})
    save(root/'retention_policy.json',{'excluded_rebuildable_directory':'clients-cache','shared_dependencies':{'path':str(Path(DEPS).parent),'usage':'Read-only numerical runtime; hashed original measured inputs and models retained in source archive'}})
    print(json.dumps({'prepared':str(root),'files_bound':len(bindings),'fits_planned':4}),flush=True)

def worker(task):
    import joblib
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0,'/app')
    import train_hybrid as trainer
    from feature_extractor import extract_features
    out=Path('/output');artifact=joblib.load('/inputs/original.joblib')
    started=time.monotonic();lineage=[]
    if task=='external':
        raw=np.load('/inputs/observations.npz',allow_pickle=False)
        assert 'labels' not in raw.files and 'groups' not in raw.files
        x,ids,names=extract_features('/inputs/observations.npz')
        np.savez_compressed(out/'real_features.npz',features=x,sample_ids=ids,feature_names=names)
        data={k:raw[k] for k in ['sample_ids','nominal_load_watts','native_record_scale']}
        labels=None
    else:
        name='dev' if task=='development' else 'train'
        raw=np.load('/inputs/'+name+'.npz',allow_pickle=False)
        data={k:raw[k] for k in ['sample_ids','nominal_load_watts','native_record_scale','bearing_family']}
        labels=raw['labels'];cache=np.load('/inputs/'+name+'_features.npz',allow_pickle=False)
        assert np.array_equal(cache['sample_ids'],data['sample_ids'])
        x=cache['features'];ids=cache['sample_ids'];names=cache['feature_names'].astype(str)
        if task!='development':
            held=task;family=data['bearing_family'].astype(str);fit=family!=held
            assert held in ['6204','6205'] and np.all(fit[family=='6206'])
            # Original normalization is per family/load; rebuilding the fitted subset must agree.
            norm=trainer._normalize_by_family_and_load(x,family,data['nominal_load_watts'])
            rebuilt=trainer._normalize_by_family_and_load(x[fit],family[fit],data['nominal_load_watts'][fit])
            assert np.array_equal(norm[fit],rebuilt)
            record_x,record_y,keys=trainer._record_medians(rebuilt,data['nominal_load_watts'][fit],data['native_record_scale'][fit],labels[fit])
            primary=artifact['primary_model'];old=joblib.load('/inputs/before.joblib')['model']
            for attr in ['support_vectors_','dual_coef_','intercept_','classes_']:
                assert np.allclose(getattr(primary[-1],attr),getattr(old[-1],attr),rtol=0,atol=1e-12)
            assert np.allclose(primary[0].mean_,old[0].mean_,rtol=0,atol=1e-12)
            for kind,indices,ncomp,c in [('raw_order_record_model',artifact['raw_envelope_indices'],8,.1),('band_record_model',artifact['band_indices'],16,10.)]:
                model=make_pipeline(StandardScaler(),PCA(n_components=ncomp,random_state=274),LogisticRegression(C=c,max_iter=3000))
                save(out/(kind+'_START.json'),{'utc':utc(),'fit_records':len(record_y),'held':held})
                model.fit(record_x[:,indices],record_y)
                artifact[kind]=model
                joblib.dump(model,out/(kind+'.joblib'),compress=3)
                assert int(model[0].n_samples_seen_)==len(record_y)
                assert np.allclose(model[0].mean_,record_x[:,indices].mean(0),rtol=0,atol=1e-12)
                transformed=model[1].transform(model[0].transform(record_x[:,indices]))
                scores=transformed@model[-1].coef_.T+model[-1].intercept_
                manual=model[-1].classes_[scores.argmax(1)]
                assert np.array_equal(manual,model.predict(record_x[:,indices]))
                lineage.append({'kind':kind,'fit_records':len(record_y),'fit_windows':int(fit.sum()),'fit_families':sorted(set(family[fit])),'excluded_family':held,'model_sha256':sha(out/(kind+'.joblib')),'linear_forward_check':True})
                save(out/'FITS.json',lineage)
            chosen=~fit;x=x[chosen];ids=ids[chosen];labels=labels[chosen]
            data={k:v[chosen] for k,v in data.items()}
    assert np.array_equal(np.asarray(names,dtype=str),artifact['all_feature_names'])
    artifact_path=out/'artifact.joblib';joblib.dump(artifact,artifact_path,compress=3)
    observation=out/'observations.npz';np.savez_compressed(observation,**data)
    predictions={}
    for mask in MASKS:
        spec=importlib.util.spec_from_file_location('subset_'+mask,'/app/'+mask+'.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ARTIFACT=artifact_path
        module.extract_features=lambda _: (x,ids,np.asarray(names,dtype=str))
        result=module.predict(observation,out/(mask+'.json'))
        predictions[mask]=[int(z['predicted_label']) for z in result['predictions']]
        assert [z['sample_id'] for z in result['predictions']]==ids.astype(str).tolist()
    save(out/'EVIDENCE.json',dict(task=task,ids=ids.astype(str).tolist(),labels=None if labels is None else labels.tolist(),
        predictions=predictions,lineage=lineage,actual_classifier_fits=len(lineage),actual_PCA_fits=len(lineage),actual_scaler_fits=len(lineage),
        primary_reused=True,primary_SVC_fits=0,complete_program_predictions=4,seconds=time.monotonic()-started,
        source_records=int(len(np.unique(np.column_stack([data['nominal_load_watts'],np.round(data['native_record_scale'],10)]),axis=0)))))

def run_worker(root, task, plan):
    out=root/'workers'/task
    (out/'runtime/tmp').mkdir(parents=True,exist_ok=False);(out/'runtime/mpl').mkdir()
    cmd=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
    for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:cmd+=['--ro-bind',p,p]
    cmd+=['--ro-bind',PYTHON,'/opt/python','--ro-bind',DEPS,'/opt/deps','--ro-bind',str(root/'app'),'/app','--ro-bind',str(root/'code'),'/code','--dir','/inputs']
    files={'original.joblib':plan['inputs']['original.joblib']}
    if task=='external':files['observations.npz']=plan['external']['observations']
    else:files.update(plan['inputs'])
    for name,path in files.items():cmd+=['--ro-bind',path,'/inputs/'+name]
    cmd+=['--bind',str(out),'/output','--proc','/proc','--dev','/dev','--chdir','/app','--remount-ro','/','/opt/python/bin/python3.12','-B','/code/'+Path(__file__).name,'worker','--task',task]
    save(out/'COMMAND.json',cmd)
    env=dict(os.environ,PYTHONPATH='/opt/deps',TMPDIR='/output/runtime/tmp',TMP='/output/runtime/tmp',TEMP='/output/runtime/tmp',MPLCONFIGDIR='/output/runtime/mpl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (out/'stdout').open('w') as stdout,(out/'stderr').open('w') as stderr:
        try:code=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,timeout=300).returncode
        except subprocess.TimeoutExpired:code=124
    save(out/'TERMINAL.json',{'utc':utc(),'exit_code':code})
    return {'task':task,'exit_code':code}

def execute(root):
    assert not (root/'HANDLE438.json').exists(),'No implicit restart'
    plan=json.loads((root/'PROTOCOL438.json').read_text());freeze=json.loads((root/'FREEZE438.json').read_text())
    assert sha(root/'PROTOCOL438.json')==freeze['protocol_sha256']
    for path,digest in plan['code_hashes'].items():assert sha(root/path)==digest
    save(root/'HANDLE438.json',{'utc':utc(),'pid':os.getpid()});started=time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        states=list(pool.map(lambda t:run_worker(root,t,plan),['development','6204','6205']))
    save(root/'VALIDATION_TERMINAL438.json',states)
    if any(x['exit_code'] for x in states):raise RuntimeError('Validation failure retained; no automatic retry')
    evidence={t:json.loads((root/'workers'/t/'EVIDENCE.json').read_text()) for t in ['development','6204','6205']}
    def error(e,mask):return sum(p!=y for p,y in zip(e['predictions'][mask],e['labels'],strict=True))/len(e['labels'])
    dev={m:error(evidence['development'],m) for m in MASKS}
    cv={m:sum(sum(p!=y for p,y in zip(evidence[t]['predictions'][m],evidence[t]['labels'],strict=True)) for t in ['6204','6205'])/sum(len(evidence[t]['labels']) for t in ['6204','6205']) for m in MASKS}
    complexity={'none':0,'class1':1,'class5':1,'both':2}
    eligible=[m for m in MASKS if dev[m]<=dev['none']]
    choice=min(eligible,key=lambda m:(cv[m],dev[m],complexity[m],MASKS.index(m)))
    seal={'utc':utc(),'choice':choice,'dev_error':dev,'CV_error':cv,'eligible':eligible,'covered_families':['6204','6205'],'missing_family':'6206','evidence_hashes':{t:sha(root/'workers'/t/'EVIDENCE.json') for t in evidence}}
    save(root/'CHOICE_SEALED438.json',seal);print(json.dumps(seal),flush=True)
    state=run_worker(root,'external',plan);save(root/'EXTERNAL_TERMINAL438.json',state)
    if state['exit_code']:raise RuntimeError('External execution failed; no implicit retry')
    ext=json.loads((root/'workers/external/EVIDENCE.json').read_text())
    save(root/'PREDICTIONS_SEALED438.json',{'utc':utc(),'sha256':sha(root/'workers/external/EVIDENCE.json')})
    save(root/'TERMINAL438.json',{'utc':utc(),'status':'predictions_sealed_pending_independent_scoring','seconds':time.monotonic()-started,'new_classifier_fits':4,'new_PCA_fits':4,'new_scaler_fits':4,'new_LLM_calls':0,'whole_program_predictions':16})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('--run',type=Path);p.add_argument('--task');a=p.parse_args()
    try:
        if a.action=='prepare':prepare(a.run)
        elif a.action=='worker':worker(a.task)
        else:execute(a.run)
    except Exception as e:
        if a.action=='run':save(a.run/'TERMINAL438.json',{'utc':utc(),'status':'failed','error_type':type(e).__name__,'error':str(e)})
        raise
