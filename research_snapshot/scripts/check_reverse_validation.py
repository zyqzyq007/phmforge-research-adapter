"""Fixed soft reverse-transfer probe on real, already opened PHM programs.

Candidate predictions are auxiliary regression targets, never ground-truth
fault labels. This is an exploratory transductive control, not a new Agent.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np

BASE=Path('/data/yyf-zyq/PHM_20261010')
CW=BASE/'20261026T060959Z_target413'
RO=BASE/'20261023T154500Z_safety314'
GE=BASE/'20261023T002126Z_ensemble298'
PYTHON=Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS=Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def arrays(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (keys or z.files)}
def bound(archive,names,dest=None):
    receipt=read(archive/'ARCHIVE_VERIFIED.json');manifest=archive/receipt['manifest_filename']
    assert sha(manifest)==receipt['manifest_sha256']
    index={x['path']:x for x in read(manifest)['files']};result={};pending=set(names)
    def use(name,raw):
        item=index[name];assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
        if dest is not None:
            path=dest/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);result[name]=path
        else:result[name]=raw
        pending.remove(name)
    for name in list(pending):
        if (archive/name).is_file():use(name,(archive/name).read_bytes())
    if pending:
        with tarfile.open(archive/receipt['bundle_filename'],'r|gz') as handle:
            for item in handle:
                if item.name in pending:use(item.name,handle.extractfile(item).read())
                if not pending:break
    assert not pending,pending
    return result,[dict(archive=str(archive),member=n,**index[n]) for n in names]
def probs(path,classes):
    rows=read(path)['predictions'];ids=[];out=[]
    for row in rows:
        ids.append(str(row['sample_id']));p=row.get('probabilities',row.get('class_probabilities'))
        v=[p[str(c)] for c in classes] if isinstance(p,dict) else p
        assert str(classes[int(np.argmax(v))])==str(row['predicted_label'])
        out.append(v)
    p=np.asarray(out,dtype=float);assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-6)
    return np.asarray(ids),p/p.sum(1,keepdims=True)
def metrics(y,p,k):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int)
    cm=np.bincount(y*k+p,minlength=k*k).reshape(k,k)
    denominator=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),denominator,out=np.zeros(k,dtype=float),where=denominator>0)
    return dict(accuracy=float(np.mean(y==p)),macro_f1=float(f.mean()))

def prepare(root,project):
    root.mkdir(exist_ok=False)
    for name in ['code','inputs','source','runtime/tmp','runtime/mpl','workers']:(root/name).mkdir(parents=True)
    for name in ['check_reverse_validation.py','controlled_feedback.py','prepare_fault_extensions.py','acquire_forge_pdmb_sources244.py']:
        shutil.copy2(Path(__file__).parent/name,root/'code'/name)
    protocol=dict(iteration='A440',utc=utc(),sources=['cwru','rotor','planetary_g2'],
        hypothesis='A fixed reverse predictor may discriminate candidate label geometry that ordinary development scores and target confidence miss.',
        algorithm='Use original native features for CWRU/rotor and existing stats_fft64 for PlanetaryG2. StandardScaler fitted only to original source train. Fit RBF kernel ridge from unlabeled target features to each existing candidate probability vector (soft predictions, not genuine labels). alpha=1, gamma=1/feature_count. Rank by original task metric on source development labels after reversing. Tie first registered candidate/reference. No tuning.',
        feature_scope='Native public representations shared across all candidates within each source; not the internal feature representation of each candidate.',
        controls=['actual ordinary Agent choice','forward development optimum','mean target maximum probability','keep original reference for two paired sources'],
        primary_gate='Strict improvement over ordinary on at least2/3 source clusters, lower equal-source mean regret, aggregate quality higher than both development and confidence controls; mean paired harm no greater and mean paired gain no lower than ordinary. G2 has no common original reference: harm/gain N/A there, never a full-goal confirmation.',
        denominator='CWRU duplicate initial/review predictions collapse into1 identical pool; rotor1; all12 original G2 pools retained and averaged within1 source. 14 pools,3 already opened sources, no independent confirmation.',
        permission='New transductive researcher control. Original Agents did not necessarily see final observations; no retrospective information-parity or Agent superiority claim. Target ground-truth forbidden to workers; external scorer reads it only after all choices seal.',
        pseudo_target_scope='Existing model output distributions are identified as regression targets; no source fault labels changed and no pseudo label is treated as benchmark ground truth.',
        budgets=dict(workers=3,worker_seconds=180,new_PHM_predictor_fits=0,auxiliary_kernel_fits=10,scaler_fits=3,new_LLM_calls=0,new_native_Agent_stages=0,disk_bytes=1000000000,memory_bytes=10000000000,GPU_hours=0),
        novelty='Reverse validation is prior art. This fixed soft-kernel adaptation is a control/admission test, not a faithful full DANN reproduction or a novel method.',
        stop='Any missing lineage or numerical failure remains in denominator. If primary gate fails stop this recipe; no source/feature/alpha/gamma/threshold search on opened target outcomes.',
        literature=[dict(url='https://adapt-python.github.io/adapt/generated/adapt.metrics.reverse_validation.html',claim='Author implementation describes reverse validation.'),
                    dict(url='https://arxiv.org/abs/2407.11676',claim='Official latest metadata identifies SKADA-Bench as TMLR, not accepted ICLR main conference.'),
                    dict(url='https://papers.phmsociety.org/index.php/phme/article/view/4926',claim='PHME2026 reports partial negative-transfer mitigation and residual failures; no reproduction claim.'),
                    dict(url='https://arxiv.org/html/2610.17805v1',claim='Selective dependency repair, empirical utility gates and verified rollback already have close prior art.'),
                    dict(url='https://arxiv.org/html/2604.02647v1',claim='Runtime traces as patch-validation constraints already proposed; venue not established.')])
    save(root/'PROTOCOL440.json',protocol)
    save(root/'FREEZE440.json',dict(utc=utc(),files={str(p.relative_to(root)):sha(p) for p in [root/'PROTOCOL440.json',*sorted((root/'code').glob('*.py'))]}))
    bindings=[]
    names=['source/train_labeled.npz','source/val_labeled.npz','inputs/test.npz']
    names += [f'predictions/{m}_{part}.npz' for m in ['mlp','ratio'] for part in ['val','test']]
    files,b=bound(CW,names,root/'source/cwru');bindings+=b
    train=arrays(files['source/train_labeled.npz']);val=arrays(files['source/val_labeled.npz']);target=arrays(files['inputs/test.npz'])
    members=['reference','Agent'];ps=[];pv=[];classes=None
    for model in ['mlp','ratio']:
        item=arrays(files[f'predictions/{model}_test.npz']);v=arrays(files[f'predictions/{model}_val.npz'])
        if classes is None:classes=sorted(item['classes'].tolist())
        assert np.array_equal(item['sample_ids'],target['sample_ids']) and np.array_equal(v['sample_ids'],val['sample_ids'])
        for data,destination in [(item,ps),(v,pv)]:
            own=data['classes'].tolist();assert set(own)==set(classes)
            assert np.array_equal(np.asarray(own)[data['probabilities'].argmax(1)],data['predicted'])
            aligned=data['probabilities'][:,[own.index(c) for c in classes]]
            assert np.array_equal(np.asarray(classes)[aligned.argmax(1)],data['predicted'])
            destination.append(aligned)
    encode=lambda ys:np.asarray([classes.index(v.item() if hasattr(v,'item') else v) for v in ys],dtype=int)
    np.savez_compressed(root/'inputs/cwru.npz',train_x=train['features'],val_x=val['features'],val_y=encode(val['labels']),target_x=target['features'],target_ids=target['sample_ids'],target_units=target['unit_ids'],target_p=np.stack(ps,1),val_p=np.stack(pv,1),classes=np.asarray(classes),members=np.asarray(members))
    rp='r/public_data/safety314_r_initial/datasets/ROTORBROKENBAR/'
    tp='r/evaluation_data/safety314_r_initial/datasets/ROTORBROKENBAR/test.npz'
    rr='r/evaluation/safety314_r_initial/output/'
    vr='r/audit/development_replay/safety314_r_initial/output/'
    audit_name='r/references/safety314_r_initial/training_audit.json'
    names=[rp+'train.npz',rp+'val.npz',tp,rr+'reference.json',rr+'predictions.json',vr+'predictions.json',audit_name]
    # Bind large read-only observations without duplicating them in the run.
    receipt=read(RO/'ARCHIVE_VERIFIED.json');manifest=RO/receipt['manifest_filename'];assert sha(manifest)==receipt['manifest_sha256']
    index={x['path']:x for x in read(manifest)['files']}
    for n in names:assert sha(RO/n)==index[n]['sha256'];bindings.append(dict(archive=str(RO),member=n,**index[n]))
    train=arrays(RO/(rp+'train.npz'),['features']);val=arrays(RO/(rp+'val.npz'),['features','labels','sample_ids']);target=arrays(RO/tp,['features','sample_ids'])
    classes=[0,1,2,3,4];ps=[]
    for n in ['reference.json','predictions.json']:
        ids,prob=probs(RO/(rr+n),classes);assert np.array_equal(ids,target['sample_ids']);ps.append(prob)
    # Reference validation accuracy is already recorded by real checkpoint selection.
    ids,agent_val=probs(RO/(vr+'predictions.json'),classes);assert np.array_equal(ids,val['sample_ids'])
    reference_audit=read(RO/audit_name);assert reference_audit['validation_rows']==len(val['labels'])
    np.savez_compressed(root/'inputs/rotor.npz',train_x=train['features'],val_x=val['features'],val_y=val['labels'],target_x=target['features'],target_ids=target['sample_ids'],target_p=np.stack(ps,1),classes=np.asarray(classes),members=np.asarray(members),agent_val_p=agent_val,reference_dev_quality=np.asarray(reference_audit['selected_validation_accuracy']))
    from controlled_feedback import transform
    receipt=read(GE/'ARCHIVE_VERIFIED.json');manifest=GE/receipt['manifest_filename'];assert sha(manifest)==receipt['manifest_sha256']
    index={x['path']:x for x in read(manifest)['files']}
    for n in ['hidden/train.npz','hidden/val.npz','hidden/test.npz','bank/probabilities.npz','PHASE_SEALED.json','review/PHASE_SEALED.json','BASIS_QUALIFICATION.json']:
        assert sha(GE/n)==index[n]['sha256'];bindings.append(dict(archive=str(GE),member=n,**index[n]))
    features={};ids_by_role={};labels=None
    for part in ['train','val','test']:
        with np.load(GE/'hidden'/(part+'.npz'),allow_pickle=False) as data:
            features[part]=transform(data['waveforms'],'stats_fft64');ids_by_role[part]=data['sample_ids']
            if part=='val':labels=data['labels']
    bank=arrays(GE/'bank/probabilities.npz');members=bank['basis_names'].tolist();ps=[]
    for name in members:
        n='evaluation282/basis/'+name+'/test/probabilities.json';assert sha(GE/n)==index[n]['sha256'];bindings.append(dict(archive=str(GE),member=n,**index[n]))
        item=read(GE/n);assert item['classes']==bank['classes'].tolist() and item['sample_ids']==ids_by_role['test'].tolist();ps.append(item['probabilities'])
    assert np.array_equal(bank['development_ids'],ids_by_role['val']) and np.array_equal(bank['development_labels'],labels)
    np.savez_compressed(root/'inputs/planetary_g2.npz',train_x=features['train'],val_x=features['val'],val_y=labels,target_x=features['test'],target_ids=ids_by_role['test'],target_p=np.stack(ps,1),val_p=bank['development'],classes=bank['classes'],members=bank['basis_names'])
    pools=[]
    for name in ['PHASE_SEALED.json','review/PHASE_SEALED.json']:
        phase=read(GE/name)
        for case in phase['cases']:
            pools.append(dict(stage=phase['stage'],id=case['id'],ordinary=case['selection']['candidate_id'],trials=[dict(id=t['id'],weights=t['weights']) for t in case['trials']]))
    assert len(pools)==12
    save(root/'inputs/POOLS440.json',pools)
    save(root/'SOURCE_BINDINGS440.json',dict(utc=utc(),files=bindings,feature_inputs={p.name:sha(p) for p in (root/'inputs').iterdir()},target_labels_loaded=False,rotor_reference_development_available=True,rotor_reference_development_basis='SHA-bound actual checkpoint-selection audit scalar, not fabricated per-row predictions.'))
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',shared_dependencies=dict(path=str(DEPS.parent),usage='Read-only scientific runtime; large original sources manifest-bound.')))
    print(dict(prepared=str(root),sources=3,auxiliary_fits=10,LLM_calls=0),flush=True)


def worker(source):
    from scipy.linalg import cho_factor,cho_solve
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import StandardScaler
    start=time.monotonic();d=arrays('/input.npz');out=Path('/output')
    assert 'target_y' not in d and 'test_labels' not in d
    scaler=StandardScaler().fit(d['train_x']);xt=scaler.transform(d['target_x']);xv=scaler.transform(d['val_x']);gamma=1/xt.shape[1]
    kernel=rbf_kernel(xt,xt,gamma=gamma);system=kernel+np.eye(len(xt));factor=cho_factor(system,lower=True)
    target=d['target_p'];assert np.isfinite(target).all() and np.allclose(target.sum(-1),1,atol=1e-6)
    coefficients=cho_solve(factor,target.reshape(len(xt),-1));reverse=(rbf_kernel(xv,xt,gamma=gamma)@coefficients).reshape(len(xv),target.shape[1],target.shape[2])
    residual=np.linalg.norm(system@coefficients-target.reshape(len(xt),-1))/max(np.linalg.norm(target),1e-12)
    assert residual<1e-8 and np.isfinite(reverse).all()
    np.savez_compressed(out/'fit.npz',mean=scaler.mean_,scale=scaler.scale_,coefficients=coefficients,reverse_val=reverse,gamma=np.asarray(gamma))
    save(out/'FIT440.json',dict(utc=utc(),source=source,auxiliary_models=target.shape[1],scaler_fits=1,target_rows=len(xt),source_validation_rows=len(xv),features=xt.shape[1],residual=float(residual),seconds=time.monotonic()-start,PHM_predictors_retrained=0,target_ground_truth_seen=False))


def dispatch(root,source):
    out=root/'workers'/source;out.mkdir();(out/'tmp').mkdir()
    cmd=['bwrap','--unshare-all','--die-with-parent','--new-session']
    for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
        if Path(p).exists():cmd+=['--ro-bind',p,p]
    for p,d in [(PYTHON,'/opt/python'),(DEPS,'/opt/deps'),(root/'code','/code'),(root/'inputs'/(source+'.npz'),'/input.npz')]:cmd+=['--ro-bind',str(p),d]
    cmd+=['--bind',str(out),'/output','--proc','/proc','--dev','/dev','--chdir','/code','--remount-ro','/','/opt/python/bin/python3.12','-B','/code/check_reverse_validation.py','worker','--source',source]
    env=dict(PATH='/opt/python/bin:/usr/bin:/bin',PYTHONPATH='/opt/deps:/code',PYTHONDONTWRITEBYTECODE='1',TMPDIR='/output/tmp',TMP='/output/tmp',TEMP='/output/tmp',MPLCONFIGDIR='/output/tmp/mpl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic()
    try:
        proc=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180);report=dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
    except subprocess.TimeoutExpired as e:report=dict(returncode=None,error='timeout',stdout=str(e.stdout),stderr=str(e.stderr))
    report.update(seconds=time.monotonic()-start,command=cmd);save(out/'PROCESS440.json',report)
    assert report['returncode']==0,report
    return source,read(out/'FIT440.json')


def choices(root):
    result={}
    pools=read(root/'inputs/POOLS440.json')
    for source in ['cwru','rotor','planetary_g2']:
        d=arrays(root/'inputs'/(source+'.npz'));reverse=arrays(root/'workers'/source/'fit.npz')['reverse_val'];k=len(d['classes']);metric='macro_f1' if source=='planetary_g2' else 'accuracy'
        cases=pools if source=='planetary_g2' else [dict(id=source,stage='paired',ordinary='Agent',trials=[dict(id=n,weights=[int(i==j) for j in range(2)]) for i,n in enumerate(['reference','Agent'])])]
        rows=[]
        for case in cases:
            candidates=[]
            for trial in case['trials']:
                w=np.asarray(trial['weights'],dtype=float);w/=w.sum();p=np.einsum('nmc,m->nc',d['target_p'],w);q=np.einsum('nmc,m->nc',reverse,w)
                value=dict(id=trial['id'],weights=w.tolist(),reverse=metrics(d['val_y'],q.argmax(1),k)[metric],confidence=float(p.max(1).mean()))
                if 'val_p' in d:value['development']=metrics(d['val_y'],np.einsum('nmc,m->nc',d['val_p'],w).argmax(1),k)[metric]
                elif trial['id']=='reference':value['development']=float(d['reference_dev_quality'])
                else:value['development']=metrics(d['val_y'],d['agent_val_p'].argmax(1),k)[metric]
                candidates.append(value)
            selected=dict(ordinary=case['ordinary'])
            for key in ['reverse','confidence','development']:
                selected[key]=max(candidates,key=lambda x:x[key])['id'] if all(x[key] is not None for x in candidates) else None
            rows.append(dict(id=case['id'],stage=case['stage'],metric=metric,candidates=candidates,selected=selected))
        result[source]=rows
    save(root/'CHOICES_SEALED440.json',dict(utc=utc(),sources=result,target_labels_loaded=False))
    return result


def evaluate(root):
    chosen=read(root/'CHOICES_SEALED440.json');result={};bindings=[]
    raw,b=bound(CW,['source/native_truth_external_only.json']);bindings+=b
    labels={r['unit_id']:r['label'] for r in json.loads(raw['source/native_truth_external_only.json'])}
    for source,cases in chosen['sources'].items():
        d=arrays(root/'inputs'/(source+'.npz'));classes=d['classes'].tolist();k=len(classes)
        if source=='cwru':y=np.asarray([classes.index(labels[x]) for x in d['target_units']])
        elif source=='planetary_g2':
            data=arrays(GE/'hidden/test.npz',['labels','sample_ids']);assert np.array_equal(data['sample_ids'],d['target_ids']);y=data['labels']
        else:
            raw,b=bound(RO,['r/hidden/ROTORBROKENBAR/labels.json']);bindings+=b
            records=json.loads(raw['r/hidden/ROTORBROKENBAR/labels.json']);mapping={r['sample_id']:r['label'] for r in records};y=np.asarray([mapping[x] for x in d['target_ids']])
        scored=[]
        for case in cases:
            metrics_by_id={}
            for c in case['candidates']:
                pred=np.einsum('nmc,m->nc',d['target_p'],c['weights']).argmax(1);metrics_by_id[c['id']]=metrics(y,pred,k)
            quality={n:(None if v is None else metrics_by_id[v][case['metric']]) for n,v in case['selected'].items()}
            oracle=max(v[case['metric']] for v in metrics_by_id.values());regret={n:None if v is None else oracle-v for n,v in quality.items()}
            reference=metrics_by_id['reference'][case['metric']] if source!='planetary_g2' else None
            scored.append(dict(id=case['id'],stage=case['stage'],candidates=metrics_by_id,selected=case['selected'],quality=quality,regret=regret,reference=reference,
                harm={n:None if v is None or reference is None else max(0,reference-v) for n,v in quality.items()},
                gain={n:None if v is None or reference is None else max(0,v-reference) for n,v in quality.items()}))
        result[source]=dict(rows=scored,mean_quality={key:float(np.mean([r['quality'][key] for r in scored])) if all(r['quality'][key] is not None for r in scored) else None for key in ['ordinary','reverse','confidence','development']},n=len(y))
    save(root/'EXTERNAL_SCORE_BINDINGS440.json',dict(utc=utc(),files=bindings,choices_sha256=sha(root/'CHOICES_SEALED440.json')))
    wins=sum(x['mean_quality']['reverse']>x['mean_quality']['ordinary']+1e-12 for x in result.values())
    loss=sum(x['mean_quality']['reverse']<x['mean_quality']['ordinary']-1e-12 for x in result.values())
    # Missing mandatory comparison stays missing; it never counts as a win.
    complete=all(x['mean_quality']['development'] is not None for x in result.values())
    gates=dict(majority=wins>=2,development_control_available=complete,
        mean_above_confidence=bool(np.mean([x['mean_quality']['reverse'] for x in result.values()])>np.mean([x['mean_quality']['confidence'] for x in result.values()])))
    gates['mean_above_development']=bool(complete and np.mean([x['mean_quality']['reverse'] for x in result.values()])>np.mean([x['mean_quality']['development'] for x in result.values()]))
    paired=[result[s]['rows'][0] for s in ['cwru','rotor']]
    gates['paired_harm_no_higher']=sum(x['harm']['reverse'] for x in paired)<=sum(x['harm']['ordinary'] for x in paired)+1e-12
    gates['paired_gain_no_lower']=sum(x['gain']['reverse'] for x in paired)>=sum(x['gain']['ordinary'] for x in paired)-1e-12
    final=dict(utc=utc(),sources=result,wins=wins,ties=3-wins-loss,losses=loss,gates=gates,primary_gate=all(gates.values()),independent_confirmation=False,majority_goal_complete=False,
        cost=dict(auxiliary_kernel_fits=10,scaler_fits=3,PHM_predictor_fits=0,LLM_calls=0,native_Agent_stages=0),
        decision='Admit only to a prospectively frozen same-information Agent pilot.' if all(gates.values()) else 'Stop this fixed reverse-kernel recipe; no target-driven tuning.')
    save(root/'RESULT440.json',final);return final


def run(root):
    for name,digest in read(root/'FREEZE440.json')['files'].items():assert sha(root/name)==digest,name
    for name,digest in read(root/'SOURCE_BINDINGS440.json')['feature_inputs'].items():assert sha(root/'inputs'/name)==digest,name
    assert not (root/'STARTED440.json').exists();save(root/'STARTED440.json',dict(utc=utc(),status='running'))
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as ex:fits=dict(ex.map(lambda s:dispatch(root,s),['cwru','rotor','planetary_g2']))
    save(root/'FITS440.json',fits);choices(root);result=evaluate(root)
    save(root/'TERMINAL440.json',dict(utc=utc(),status='completed',seconds=time.monotonic()-started,cost=result['cost']))
    print(dict(quality={k:v['mean_quality'] for k,v in result['sources'].items()},wins=result['wins'],ties=result['ties'],losses=result['losses'],gate=result['primary_gate']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('--run',type=Path);p.add_argument('--project',type=Path);p.add_argument('--source');a=p.parse_args()
    if a.action=='worker':worker(a.source)
    elif a.action=='prepare':prepare(a.run.resolve(),a.project.resolve())
    else:
        try:run(a.run.resolve())
        except Exception:
            save(a.run/'TERMINAL440.json',dict(utc=utc(),status='failed',traceback=traceback.format_exc()));raise
