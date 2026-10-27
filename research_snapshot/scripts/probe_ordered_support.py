"""Fixed-parameter, bidirectional support holdouts for real rotor programs.

Ordinary blocked validation is a strong control, not a claimed new algorithm.
The final direction, data and outcomes are absent from every numerical worker.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261024T103209Z_hypotheses325')
PYTHON='/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu'
DEPS='/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps'
COMPONENTS=['P00','P10','P01','P11','RAW_LR']
CANDIDATES=['P00','P10','P01','P11','P11B']


def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def prepare(root):
    root.mkdir(exist_ok=False)
    for d in ['code','shared_tools','workers','runtime/tmp','runtime/mpl']:(root/d).mkdir(parents=True,exist_ok=True)
    receipt=json.loads((SOURCE/'ARCHIVE_VERIFIED.json').read_text())
    manifest=SOURCE/receipt['manifest_filename']
    assert sha(manifest)==receipt['manifest_sha256']=='ec2b43dc89d0b779156ad65425718531eccdfccab9d8c146ae701d27bce1ecb4'
    index={v['path']:v for v in json.loads(manifest.read_text())['files']}
    files=[SOURCE/'CAMPAIGN_CONFIG.json']
    for name in ['controlled_program_checks.py','controlled_feedback.py','rotor_feature_recipe.py']:
        files.append(SOURCE/'shared_tools'/name);shutil.copy2(files[-1],root/'shared_tools'/name)
    for block in ['block78','block910']:
        files += [SOURCE/'hidden'/block/'train.npz',SOURCE/'bank'/block/'training_designs.npz',SOURCE/'bank'/block/'VALIDATION_REPORTS.json']
        for c in COMPONENTS:
            files += [SOURCE/'bank'/block/c/f for f in ['probabilities.npz','fold_0.joblib','fold_4.joblib']]
    bindings=[]
    for p in files:
        item=index[str(p.relative_to(SOURCE))]
        assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
        bindings.append(dict(source_path=str(p),**item))
    original=json.loads((SOURCE/'CAMPAIGN_CONFIG.json').read_text())
    config={k:original[k] for k in ['tree_parameters','logistic','numerical_seed','numerical_threads_per_worker']}
    save(root/'MODEL_CONFIG.json',config)
    tasks=[]
    for block in ['block78','block910']:
        for direction in ['low','high']:
            for k in [2,3,4]:tasks.append(dict(id=f'{block}_{direction}{k}',block=block,kind=direction,k=k,reuse=(k==4)))
        for k in [2,3]:tasks.append(dict(id=f'{block}_interior{k}',block=block,kind='interior',k=k,reuse=False))
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    plan=dict(iteration='A391',frozen_utc=datetime.now(timezone.utc).isoformat(),source=str(SOURCE),source_manifest_sha256=receipt['manifest_sha256'],source_manifest_filename=receipt['manifest_filename'],bindings=bindings,tasks=tasks,
        question='Do larger observed-support gaps reveal the harmful spectral update when ordinary leave-one-load CV favors it?',
        known_scope='A325 two already-opened acquisition blocks from one physical motor; exploratory qualification only',
        hypothesis='Candidate update gains under interpolation can disappear under bidirectional range holdouts; test both directions without target information',
        rival='Changes are explained by smaller fitted samples rather than range geometry; cardinality-matched interior holdouts are controls',
        geometry=dict(loads=[.5,1.,1.5,2.,2.5],ordered_training_sizes=[2,3,4],low='first k observed loads, hold remaining higher loads',high='last k observed loads, hold remaining lower loads',interior2=[.5,2.5],interior3=[.5,1.5,2.5],interior4='reuse the three original middle-load leave-one-out scores'),
        all_classes_required_in_each_training_and_holdout_group=True,whole_acquisition_groups=True,
        fixed_original_recipes=True,new_candidates=False,parameter_search=False,
        primary='Mean of six complete-program whole-record error rates per block; each direction and k equally weighted; select minimum, ties use P00/P10/P01/P11/P11B order',
        diagnostic='For each k compare candidate-minus-P00 error in symmetric extrapolation versus matched-size interior checks; then average k equally',
        qualification='Both blocks select P00 or P01; P11B mean extrapolation error no greater than P11 in both blocks (limited beneficial intermediate-pair check); P11B-minus-P00 geometry contrast positive in both. Failure of any stops this rule; no retuning.',
        limits=['No improvement opportunity from incumbent P00 in the old five-candidate pool; intermediate-pair check is not goal gain-capture validation','Old ordinary Agent did not receive these new measurements; no same-information Agent advantage can be claimed','Training/context changes and observational load variation do not identify a unique physical cause','Repeated records across folds and acquisition blocks are not independent motors'],
        budget=dict(new_classifier_fits=60,new_standardscaler_fits=12,reused_endpoint_checkpoints=20,new_fit_prediction_calls=60,cold_reload_prediction_calls=60,reused_model_prediction_calls=20,workers=4,threads_per_worker=config['numerical_threads_per_worker'],seconds_per_task=480,total_seconds_cap=1800,new_LLM_calls=0,new_final_predictions=0,disk_estimate_MB=900,GPU_hours=0),
        no_final_inputs_mounted=True,no_final_direction_used_to_choose_splits=True,no_synthetic_data=True,
        prior_art=dict(url='https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881',title='Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure',verified='2026-10-25',read_scope='Abstract and predictor-space blocking table; blocking for extrapolation is established, not our novelty'),code_sha256=sha(root/'code'/Path(__file__).name))
    save(root/'PROTOCOL391.json',plan);save(root/'FREEZE391.json',dict(protocol_sha256=sha(root/'PROTOCOL391.json'),code_sha256=plan['code_sha256'],model_config_sha256=sha(root/'MODEL_CONFIG.json')))
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',shared_dependencies=dict(path=str(Path(DEPS).parent),usage='Original A325 recipe parameters; measured feature cache; no new final evaluation')))
    print(str(root),flush=True)


def worker(kind,k):
    import joblib
    import numpy as np
    sys.path.insert(0,'/recipe')
    from controlled_program_checks import estimator,feature_kind,metrics
    cfg=json.loads(Path('/inputs/config.json').read_text())
    with np.load('/inputs/train.npz',allow_pickle=False) as z:data={n:z[n] for n in ['labels','groups','sample_ids','operating_load_Nm']}
    with np.load('/inputs/designs.npz',allow_pickle=False) as z:x=dict(z)
    loads=np.array(sorted(set(data['operating_load_Nm'])))
    assert loads.tolist()==[.5,1.,1.5,2.,2.5] and len(set(data['groups']))==50
    chosen=loads[:k] if kind=='low' else (loads[-k:] if kind=='high' else loads[[0,4] if k==2 else [0,2,4]])
    fit=np.isin(data['operating_load_Nm'],chosen);held=~fit
    assert set(data['groups'][fit]).isdisjoint(data['groups'][held])
    assert set(data['labels'][fit])==set(range(5)) and set(data['labels'][held])==set(range(5))
    assert len(set(data['groups'][fit]))==10*k
    subset={n:v[held] for n,v in data.items()};prob={};events=[];calls=0
    for c in COMPONENTS:
        started=time.monotonic();reuse=k==4;target=x[feature_kind(c)][held]
        if reuse:
            model=joblib.load('/inputs/'+c+'.joblib');p=model.predict_proba(target);calls+=1
            with np.load('/inputs/'+c+'_prob.npz',allow_pickle=False) as z:cached=z['cv'][held]
            assert np.allclose(p,cached,rtol=0,atol=1e-12)
        else:
            model=estimator(cfg,c);model.fit(x[feature_kind(c)][fit],data['labels'][fit])
            p=model.predict_proba(target);calls+=1
            joblib.dump(model,'/output/'+c+'.joblib',compress=3)
            cold=joblib.load('/output/'+c+'.joblib').predict_proba(target);calls+=1
            assert np.allclose(p,cold,rtol=0,atol=1e-12)
        assert model.classes_.tolist()==list(range(5)) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
        if c=='RAW_LR':
            s=model.named_steps['standardscaler'];assert int(s.n_samples_seen_)==int(fit.sum())
            assert np.allclose(s.mean_,x['raw'][fit].mean(0),rtol=1e-10,atol=1e-10)
        event=dict(component=c,reused_checkpoint=reuse,new_classifier_fit=not reuse,seconds=time.monotonic()-started,cold_equal=True,fit_rows=int(fit.sum()),held_rows=int(held.sum()))
        events.append(event);prob[c]=p;save('/output/FIT_EVENTS.json',events)
        print(json.dumps(event),flush=True)
    prob['P11B']=.9*prob['P11']+.1*prob['RAW_LR']
    reports={c:metrics(subset,prob[c].argmax(1)) for c in CANDIDATES}
    np.savez_compressed('/output/probabilities.npz',**prob,sample_ids=data['sample_ids'][held],labels=data['labels'][held],groups=data['groups'][held],loads=data['operating_load_Nm'][held])
    save('/output/EVIDENCE.json',dict(kind=kind,k=k,fit_loads=chosen.tolist(),held_loads=loads[~np.isin(loads,chosen)].tolist(),fit_groups=sorted(set(data['groups'][fit])),held_groups=sorted(set(data['groups'][held])),reports=reports,events=events,pipeline_prediction_calls=calls,classifier_fits=0 if k==4 else 5,scaler_fits=0 if k==4 else 1,no_final_inputs=True))


def run(root):
    assert not (root/'HANDLE391.json').exists()
    plan=json.loads((root/'PROTOCOL391.json').read_text());frozen=json.loads((root/'FREEZE391.json').read_text())
    assert sha(root/'PROTOCOL391.json')==frozen['protocol_sha256'] and sha(root/'code'/Path(__file__).name)==frozen['code_sha256']
    assert sha(root/'MODEL_CONFIG.json')==frozen['model_config_sha256']
    save(root/'HANDLE391.json',dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat()));start=time.monotonic()
    def job(task):
        out=root/'workers'/task['id']
        for d in ['runtime/tmp','runtime/mpl']:(out/d).mkdir(parents=True,exist_ok=False)
        source=Path(plan['source']);bank=source/'bank'/task['block']
        mounts={'config.json':root/'MODEL_CONFIG.json','train.npz':source/'hidden'/task['block']/'train.npz','designs.npz':bank/'training_designs.npz'}
        if task['reuse']:
            fold='4' if task['kind']=='low' else '0'
            for c in COMPONENTS:mounts[c+'.joblib']=bank/c/('fold_'+fold+'.joblib');mounts[c+'_prob.npz']=bank/c/'probabilities.npz'
        cmd=['bwrap','--unshare-user','--unshare-pid','--unshare-ipc','--unshare-uts','--unshare-net','--die-with-parent','--new-session']
        for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:cmd+=['--ro-bind',p,p]
        cmd+=['--ro-bind',PYTHON,'/opt/python','--ro-bind',DEPS,'/opt/deps','--ro-bind',str(root/'shared_tools'),'/recipe','--ro-bind',str(root/'code'),'/code','--dir','/inputs']
        for name,path in mounts.items():cmd+=['--ro-bind',str(path),'/inputs/'+name]
        cmd+=['--bind',str(out),'/output','--proc','/proc','--dev','/dev','--chdir','/output','--remount-ro','/','/opt/python/bin/python3.12','-B','/code/'+Path(__file__).name,'worker','--kind',task['kind'],'--k',str(task['k'])]
        save(out/'COMMAND.json',cmd)
        env=dict(os.environ,PYTHONPATH='/opt/deps',TMPDIR='/output/runtime/tmp',TMP='/output/runtime/tmp',TEMP='/output/runtime/tmp',MPLCONFIGDIR='/output/runtime/mpl',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/'stdout').open('w') as stdout,(out/'stderr').open('w') as stderr:
            try:code=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,timeout=480).returncode
            except subprocess.TimeoutExpired:code=124
        save(out/'TERMINAL.json',dict(exit_code=code));return dict(id=task['id'],exit_code=code)
    states=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(job,t) for t in plan['tasks']]):
            states.append(future.result());save(root/'PROGRESS391.json',states);print(json.dumps(states[-1]),flush=True)
    ok=all(s['exit_code']==0 for s in states)
    if ok:
        evidence={t['id']:json.loads((root/'workers'/t['id']/'EVIDENCE.json').read_text()) for t in plan['tasks']}
        save(root/'EVIDENCE391.json',evidence);save(root/'EVIDENCE_SEAL391.json',dict(sha256=sha(root/'EVIDENCE391.json'),sealed_utc=datetime.now(timezone.utc).isoformat()))
    save(root/'TERMINAL391.json',dict(exit_code=0 if ok else 1,states=states,seconds=time.monotonic()-start))
    if not ok:raise RuntimeError('Failure retained; no automatic retry')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('--run',type=Path);p.add_argument('--kind',choices=['low','high','interior']);p.add_argument('--k',type=int);a=p.parse_args()
    if a.action=='prepare':prepare(a.run)
    elif a.action=='run':run(a.run)
    else:worker(a.kind,a.k)
