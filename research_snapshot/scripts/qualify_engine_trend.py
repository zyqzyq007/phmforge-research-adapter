"""Real source qualification for the ten multi-cycle Engine question directions.

Train/development tuples share no source record, including through overlapping
QA tuples. Exclude every connected group touching an original question. Targets
are publisher condition/probability-bin annotations, not calibrated risk.
"""
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from acquire_forge_pdmb_sources244 import save, sha256
from prepare_verified_benchmarks import extract_bound
from qualify_engine_supervision import URL, SIZE
from real_fault_model import train, infer

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261023T140132Z_engine310')


def state(root,**values):
    p=root/'STATUS313.json';v=json.loads(p.read_text()) if p.exists() else {}
    v.update(updated_utc=datetime.now(timezone.utc).isoformat(),run=str(root),**values)
    save(p,v);save(PROJECT/'results/enginetrend313/current_run.json',v)


def main(root):
    assert not (root/'PROTOCOL313.json').exists()
    for f in ['audit','source','code','outputs','logs','runtime/tmp','runtime/mpl','clients-cache']:(root/f).mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(root/'runtime/tmp'),TMP=str(root/'runtime/tmp'),TEMP=str(root/'runtime/tmp'),MPLCONFIGDIR=str(root/'runtime/mpl'))
    path=SOURCE/'source/official_train_qa.jsonl';rows=[];parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for line in path.open():
        row=json.loads(line)
        if not isinstance(row['id'],list):continue
        ids=tuple(map(str,row['id']));assert len(ids)==10 and len(set(ids))==10
        labels={};cs=row['conversations']
        for i,c in enumerate(cs[:-1]):
            if c['from']!='human' or c.get('stage')!='3' or cs[i+1]['from']!='gpt':continue
            q=c['value'];answer=cs[i+1]['value'].strip();opts={a:b.strip() for a,b in re.findall(r'([a-z]):\s*([^\n]+)',q)}
            key='condition' if 'qualitative condition' in q else 'probability_bin' if 'probability range' in q else None
            if key:labels[key]=opts[answer]
        assert set(labels)=={'condition','probability_bin'}
        for i in ids:parent[find(i)]=find(ids[0])
        rows.append(dict(source_ids=ids,**labels))
    matrix=json.loads((PROJECT/'results/forgesources244_qa_correction/SOURCE_MATRIX244.json').read_text())['engine_task_matrix']
    reserved={str(i) for row in matrix for i in row['source_ids']};excluded={find(i) for i in reserved if i in parent}
    groups=defaultdict(list)
    for row in rows:
        group=find(row['source_ids'][0])
        if group not in excluded:groups[group].append(row)
    strata=defaultdict(list)
    for group,items in groups.items():
        dominant=Counter(r['condition'] for r in items).most_common(1)[0][0];strata[dominant].append(group)
    allocation={}
    for label,members in strata.items():
        ordered=sorted(members,key=lambda i:hashlib.sha256(('A313:'+i).encode()).hexdigest())
        nval=max(2,round(len(ordered)*.25));assert nval<len(ordered)
        for i,group in enumerate(ordered):allocation[group]='val' if i<nval else 'train'
    selection=[];identities={}
    for group,items in groups.items():
        for row in items:
            key='|'.join(row['source_ids']);sid='trend_'+hashlib.sha256(key.encode()).hexdigest()[:24]
            label=row['condition']+'|'+row['probability_bin']
            if sid in identities:
                assert identities[sid]==label,'Identical tuples have conflicting annotations'
                continue
            identities[sid]=label;selection.append(dict(row,sample_id=sid,label=label,part=allocation[group],group='qa_overlap_'+hashlib.sha256(group.encode()).hexdigest()[:16]))
    native=sorted({i for row in selection for i in row['source_ids']},key=int)
    sets={part:{i for row in selection if row['part']==part for i in row['source_ids']} for part in ['train','val']}
    assert not sets['train']&sets['val'] and not set(native)&reserved
    binding=extract_bound(Path('/data/yyf-zyq/PHM_20261010/20261020T140759Z_forgesources244_qa_correction'),root/'source/loader',['source/itformer_dataset.py'])
    protocol=dict(iteration='A313',purpose='Qualify genuine ten-cycle annotation prediction; no Agent or final original-task predictions.',
        source_url=URL,source_bytes=SIZE,source_pin='ea2051a15512212976c22c1a075ec549c7d48eef',qa_sha256=sha256(path),
        tuple_group_rule='Connected components of all overlapping published ten-cycle source-ID tuples; groups touching any original task ID excluded.',
        physical_engine_identity_certified=False,group_count=len(groups),excluded_original_connected_groups=len(excluded),
        selection=selection,unique_source_records=len(native),native_prefix_rows=60,
        prefix_rule='Pinned official itformer_dataset.py concatenates the first len(record)//10 rows of each source in the supplied order:10x60x33.',
        features='For each of10 native60x33 prefixes in order:mean,std,min,max,first,last;1980 real features.',
        fixed_models='MLP50epochs and ExtraTrees500trees,max_features1,minleaf1,seed2026102309; no final-score model selection.',
        max_source_worker_seconds=900,max_total_range_bytes=512*1024**2,source_workers=4,
        artifact_budget_bytes=2*1024**3,physical_failure_probability_calibrated=False,new_Agent_sessions=0,
        original_tasks_planned=[r['task_id'] for r in matrix if len(r['source_ids'])==10])
    save(root/'PROTOCOL313.json',protocol);save(root/'audit/LOADER_BINDING313.json',binding)
    shutil.copy2(path,root/'source/official_train_qa.jsonl')
    for f in ['qualify_engine_trend.py','probe_remote_hdf5.py','real_fault_model.py','qualify_engine_supervision.py']:
        shutil.copy2(PROJECT/'scripts'/f,root/'code'/f)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared',retention='Pinned runtime and h5py3.15.1 retained read-only; exact HTTP ranges and source prefixes retained.')))
    state(root,status='loading_real_native60row_prefixes',selected_tuples=len(selection),native_records=len(native),new_model_sessions=0,training_fits=0)
    def worker(index):
        ids=native[index::4];f=root/'audit'/('ids'+str(index)+'.json');save(f,ids)
        argv=[sys.executable,'-B',str(root/'code/probe_remote_hdf5.py'),'--url',URL,'--size',str(SIZE),'--ids',str(f),
            '--output',str(root/'outputs'/('native'+str(index))),'--max-records','0','--time-rows','60','--transport-attempts','3']
        save(root/'audit'/('command'+str(index)+'.json'),argv)
        with (root/'logs'/('native'+str(index)+'.log')).open('wb') as out:
            try:code=subprocess.run(argv,env=dict(os.environ),stdout=out,stderr=subprocess.STDOUT,timeout=900).returncode
            except subprocess.TimeoutExpired:code='timeout'
        save(root/'audit'/('exit'+str(index)+'.json'),dict(exit_code=code));return code
    with ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(worker,range(4)))
    if any(code!=0 for code in codes):
        state(root,status='source_incomplete_no_fitting',source_exit_codes=codes);raise RuntimeError('Source transport incomplete; no fake values or automatic full retry')
    return fit_run(root)


def fit_run(root):
    protocol=json.loads((root/'PROTOCOL313.json').read_text())
    selection=protocol['selection'];native={i for row in selection for i in row['source_ids']}
    arrays={}
    for path in (root/'outputs').glob('native*/*.npy'):
        assert path.stem in native and path.stem not in arrays
        a=np.load(path,allow_pickle=False)
        assert a.shape==(60,33) and np.isfinite(a).all()
        arrays[path.stem]=a
    assert set(arrays)==set(native)
    data=root/'public/datasets/ENGINE_QA_TREND';data.mkdir(parents=True)
    for part in ['train','val']:
        subset=[r for r in selection if r['part']==part];features=[]
        for row in subset:
            features.append(np.concatenate([np.concatenate([a.mean(0),a.std(0),a.min(0),a.max(0),a[0],a[-1]]) for a in [arrays[i] for i in row['source_ids']]]).astype(np.float32))
        np.savez_compressed(data/(part+'.npz'),features=np.asarray(features),labels=np.asarray([r['label'] for r in subset]),groups=np.asarray([r['group'] for r in subset]),sample_ids=np.asarray([r['sample_id'] for r in subset]),dataset=np.asarray('ENGINE_QA_TREND'))
    info=dict(dataset='ENGINE_QA_TREND',seed=2026102309,class_names=sorted({r['label'] for r in selection}),architectures=['mlp'],feature_count=1980,
        rows_by_split={p:sum(r['part']==p for r in selection) for p in ['train','val']},group_definition=protocol['tuple_group_rule'],
        data_origin='Official NASA simulator-derived EngineMT-QA sequences and publisher categorical annotations, not field risk measurements.',
        physical_probability_calibration=False,feature_definition=protocol['features'],physical_engine_grouping_verified=False)
    save(data/'dataset.json',info)
    state(root,status='actual_MLP_fitting_on_group_disjoint_source_tuples',original_source_exit_codes=[json.loads((root/'audit'/('exit'+str(i)+'.json')).read_text())['exit_code'] for i in range(4)],training_fits=1)
    audit=train(data,root/'reference','mlp',50);assert audit['nonzero_gradient_steps']>0 and audit['changed_parameter_tensors']>0
    fit=dict(np.load(data/'train.npz',allow_pickle=False));dev=dict(np.load(data/'val.npz',allow_pickle=False))
    observation=root/'development_observations.npz';np.savez_compressed(observation,**{k:v for k,v in dev.items() if k not in ['labels','groups']})
    prediction=infer(root/'reference/model.pt',observation,root/'development_predictions.json')
    by_id={r['sample_id']:r['predicted_label'] for r in prediction['predictions']}
    mlp=np.asarray([by_id[str(s)] for s in dev['sample_ids']]);model=ExtraTreesClassifier(n_estimators=500,max_features=1.0,min_samples_leaf=1,n_jobs=2,random_state=2026102309)
    state(root,status='actual_fixed_ExtraTrees_fitting',training_fits=2)
    model.fit(fit['features'],fit['labels']);pred=model.predict(dev['features']);joblib.dump(model,root/'reference/ordinary_extra_trees.joblib',compress=3)
    def metrics(p):
        return dict(n=len(p),joint_annotation_accuracy=float(np.mean(p==dev['labels'])),
            condition_accuracy=float(np.mean([a.split('|')[0]==b.split('|')[0] for a,b in zip(p,dev['labels'])])),
            probability_bin_annotation_accuracy=float(np.mean([a.split('|')[1]==b.split('|')[1] for a,b in zip(p,dev['labels'])])))
    result=dict(numerical_pipeline_qualified=True,MLP=metrics(mlp),ExtraTrees=metrics(pred),native_records=len(native),source_overlap=False,
        physical_engine_disjointness_certified=False,physical_risk_calibration=False,original_task_final_predictions=0,new_Agent_sessions=0,
        model_sha256=sha256(root/'reference/ordinary_extra_trees.joblib'))
    save(root/'QUALIFICATION313.json',result)
    state(root,status='real_source_and_two_fits_complete_annotation_only',training_fits=2,qualification=result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();main(args.run.resolve())
