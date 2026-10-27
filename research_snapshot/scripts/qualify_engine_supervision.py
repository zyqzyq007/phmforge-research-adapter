"""Bounded genuine numerical qualification of official EngineMT-QA supervision.

Not an Agent experiment and not physical risk calibration. No original-task
record enters fitting/development. Only official five-component annotations are
used; missing overall/cause annotations are never filled in.
"""
import argparse
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

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from real_fault_model import train, infer

PROJECT=Path(__file__).resolve().parents[1]
HEADS=['fan','lpc','hpc','hpt','lpt']
URL='https://huggingface.co/datasets/pandalin98/EngineMT-QA/resolve/ea2051a15512212976c22c1a075ec549c7d48eef/time_series_data.h5'
SIZE=18838039816


def main(run,source,train_per_class=12,val_per_class=4,prior=None,worker_timeout=750):
    for f in ['audit','code','source','outputs','logs','runtime/tmp','runtime/mpl','clients-cache']:(run/f).mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(run/'runtime/tmp'),TMP=str(run/'runtime/tmp'),
        TEMP=str(run/'runtime/tmp'),MPLCONFIGDIR=str(run/'runtime/mpl'))
    status=run/'STATUS309.json'
    def state(**changes):
        current=json.loads(status.read_text()) if status.exists() else {}
        current.update(updated_utc=datetime.now(timezone.utc).isoformat(),run=str(run),**changes)
        target='engine309' if prior is None else 'engine310'
        save(status,current);save(PROJECT/'results'/target/'current_run.json',current)
    assert not (run/'PROTOCOL309.json').exists()
    matrix=json.loads((PROJECT/'results/forgesources244_qa_correction/SOURCE_MATRIX244.json').read_text())['engine_task_matrix']
    reserved={str(i) for row in matrix for i in row['source_ids']}
    qa=source/'source/engine_train_qa.jsonl';records={}
    for line in qa.open():
        row=json.loads(line)
        if not isinstance(row['id'],str) or row['id'] in reserved:continue
        labels={};cs=row['conversations']
        for i,c in enumerate(cs[:-1]):
            if c['from']!='human' or c.get('stage')!='2' or 'health status' not in c['value'].lower() or cs[i+1]['from']!='gpt':continue
            match=[h for h in HEADS if re.search(r'\b'+h+r'\b',c['value'],re.I)]
            answer=cs[i+1]['value'].strip()
            if len(match)==1 and answer in ['a','b']:labels[match[0]]=int(answer=='b')
        assert set(labels)==set(HEADS)
        key=''.join(str(labels[h]) for h in HEADS)
        if row['id'] in records:assert records[row['id']]['joint_label']==key
        records[row['id']]=dict(source_id=row['id'],joint_label=key,component_annotation=labels)
    prior_plan=json.loads((prior/'PROTOCOL309.json').read_text()) if prior else None
    protected_train={v['source_id'] for v in prior_plan['selection'] if v['part']=='train'} if prior else set()
    protected_val={v['source_id'] for v in prior_plan['selection'] if v['part']=='val'} if prior else set()
    classes=sorted({v['joint_label'] for v in records.values()});selection=[]
    for label in classes:
        members=sorted([v for v in records.values() if v['joint_label']==label],
            key=lambda v:hashlib.sha256(('A309:'+v['source_id']).encode()).hexdigest())
        assert len(members)>=6
        # Fixed small qualification, not a full trained benchmark baseline.
        fit=[v for v in members if v['source_id'] not in protected_val][:train_per_class]
        fit_ids={v['source_id'] for v in fit}
        keep=[v for v in members if v['source_id'] in protected_val]
        held=keep+[v for v in members if v['source_id'] not in fit_ids|protected_train|protected_val][:max(0,val_per_class-len(keep))]
        for part,subset in [('train',fit),('val',held)]:
            for v in subset:selection.append(dict(v,part=part))
    # Small strata must not overlap between fitting and development.
    for label in classes:
        selected=[v for v in selection if v['joint_label']==label]
        held={v['source_id'] for v in selected if v['part']=='val'}
        selection=[v for v in selection if not (v['joint_label']==label and v['part']=='train' and v['source_id'] in held)]
    ids=[v['source_id'] for v in selection];assert len(ids)==len(set(ids)) and set(ids).isdisjoint(reserved)
    protocol=dict(iteration='A309' if prior is None else 'A310',purpose='Qualify real loading, genuine fitting and record-disjoint development for five official QA component annotations; no original-task final evaluation.',
        user_source_authorization='允许 NASA 官方仿真数据，禁止占位返回和伪训练',
        source_pin='ea2051a15512212976c22c1a075ec549c7d48eef',source_url=URL,source_bytes=SIZE,
        qa_path=str(qa),qa_sha256=sha256(qa),selection=selection,excluded_original_source_IDs=sorted(reserved,key=int),
        feature_rule='Per anonymous native channel: mean,std,min,max,first,last over all600 observations; no guessed channel names.',
        training='Real MLP,50epochs,seed2026102309; normalization fitting records only; no final labels or tasks.',
        qualification_metrics=['complete_native_loading','source_record_identity','actual_parameter_updates','development_joint_and_component_annotation_accuracy'],
        no_accuracy_threshold=True,no_Agent_sessions=True,physical_engine_IDs_unavailable=True,
        limitation='A record is one source cycle. Record separation does not establish physical-engine separation; annotation agreement is not calibrated physical risk or causality.',
        train_per_class=train_per_class,val_per_class=val_per_class,prior_source_qualification=str(prior) if prior else None,
        protected_prior_validation_IDs=sorted(protected_val,key=int),
        max_range_download_bytes=4*128*1024**2,max_wall_seconds=worker_timeout+150,artifact_budget_bytes=2*1024**3,
        source_native_values_not_simulated_locally=True,official_simulator_source=True)
    save(run/'PROTOCOL309.json',protocol)
    shutil.copy2(qa,run/'source/official_train_qa.jsonl')
    for f in ['qualify_engine_supervision.py','probe_remote_hdf5.py','real_fault_model.py','real_rul_model.py']:
        shutil.copy2(PROJECT/'scripts'/f,run/'code'/f)
    save(run/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',
        shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared',retention='Hash-bound scientific runtime; h5py3.15.1 is reused read-only from verified A304 archive.')))
    reused=[]
    if prior:
        target=run/'outputs/native_reuse';target.mkdir()
        for ident in ids:
            matches=list((prior/'outputs').glob('native*/'+ident+'.npy'))
            if not matches:continue
            assert len(matches)==1
            native=matches[0]
            receipt=json.loads((native.parent/'RESULT.json').read_text())
            expected=next(v for v in receipt['records'] if v['id']==ident)
            values=np.load(native,allow_pickle=False)
            assert hashlib.sha256(values.tobytes()).hexdigest()==expected['sha256']
            shutil.copy2(native,target/native.name)
            reused.append(dict(id=ident,path=str(native),source_file_sha256=sha256(native),array_sha256=expected['sha256']))
    save(run/'audit/REUSED_NATIVE_RECORDS.json',reused)
    remaining=[v for v in ids if v not in {x['id'] for x in reused}]
    state(status='loading_selected_original_HDF5_records',records=len(ids),reused_native_records=len(reused),new_model_sessions=0,training_fits=0,original_final_scores=0)
    def worker(index):
        chunk=remaining[index::4];idfile=run/'audit'/('ids'+str(index)+'.json');save(idfile,chunk)
        command=[sys.executable,'-B',str(PROJECT/'scripts/probe_remote_hdf5.py'),'--url',URL,'--size',str(SIZE),
            '--ids',str(idfile),'--output',str(run/'outputs'/('native'+str(index))),'--max-records','0']
        save(run/'audit'/('command'+str(index)+'.json'),command)
        with (run/'logs'/('native'+str(index)+'.log')).open('wb') as log:
            try:code=subprocess.run(command,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,timeout=worker_timeout).returncode
            except subprocess.TimeoutExpired:code='timeout'
        save(run/'audit'/('exit'+str(index)+'.json'),dict(exit_code=code));return code
    with ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(worker,range(4)))
    if any(v!=0 for v in codes):
        state(status='source_transport_incomplete_no_training',worker_exit_codes=codes)
        raise RuntimeError('Native source acquisition incomplete; preserve all failed transports; do not fake source values')
    return fit_run(run,codes)


def fit_run(run,codes):
    protocol=json.loads((run/'PROTOCOL309.json').read_text())
    selection=protocol['selection'];classes=sorted({v['joint_label'] for v in selection})
    def state(**changes):
        target=run/'STATUS309.json';current=json.loads(target.read_text())
        current.update(updated_utc=datetime.now(timezone.utc).isoformat(),**changes)
        save(target,current)
        save(PROJECT/'results'/('engine310' if protocol['iteration']=='A310' else 'engine309')/'current_run.json',current)
    state(status='native_records_loaded_fitting_real_MLP',training_fits=1)
    data=run/'public/datasets/ENGINE_QA_HEALTH';data.mkdir(parents=True)
    decoded=[]
    for v in selection:
        matches=list((run/'outputs').glob('native*/'+v['source_id']+'.npy'));assert len(matches)==1
        raw=np.load(matches[0],allow_pickle=False);assert raw.shape==(600,33) and np.isfinite(raw).all()
        features=np.concatenate([raw.mean(0),raw.std(0),raw.min(0),raw.max(0),raw[0],raw[-1]]).astype(np.float32)
        decoded.append(dict(v,features=features,native_sha256=sha256(matches[0])))
    for part in ['train','val']:
        rows=[v for v in decoded if v['part']==part]
        np.savez_compressed(data/(part+'.npz'),features=np.asarray([v['features'] for v in rows]),
            labels=np.asarray([v['joint_label'] for v in rows]),groups=np.asarray(['source_record_'+v['source_id'] for v in rows]),
            sample_ids=np.asarray(['engine_qa_'+v['source_id'] for v in rows]),dataset=np.asarray('ENGINE_QA_HEALTH'))
    info=dict(dataset='ENGINE_QA_HEALTH',seed=2026102309,class_names=classes,architectures=['mlp'],
        group_definition='One official source-cycle record; no fitting/development record overlap. Physical-engine identity unresolved.',
        source_sha256=sha256(run/'PROTOCOL309.json'),data_origin='Official N-CMAPSS-derived EngineMT-QA arrays and publisher QA annotations.',
        annotation_heads=HEADS,physical_risk_calibration=False,missing_cause_or_overall_labels_not_filled=True)
    save(data/'dataset.json',info)
    audit=train(data,run/'reference','mlp',50)
    assert audit['optimizer_steps']>0 and audit['nonzero_gradient_steps']>0 and audit['changed_parameter_tensors']>0
    dev=dict(np.load(data/'val.npz',allow_pickle=False));obs=run/'development_observations.npz'
    np.savez_compressed(obs,**{k:v for k,v in dev.items() if k not in ['labels','groups']})
    result=infer(run/'reference/model.pt',obs,run/'development_predictions.json')
    by_id={v['sample_id']:v['predicted_label'] for v in result['predictions']}
    actual=dev['labels'].astype(str);predicted=np.asarray([by_id[str(s)] for s in dev['sample_ids']])
    metrics=dict(n=len(actual),joint_annotation_accuracy=float(np.mean(actual==predicted)),
        component_annotation_accuracy={h:float(np.mean([a[i]==b[i] for a,b in zip(actual,predicted)])) for i,h in enumerate(HEADS)})
    save(run/'QUALIFICATION309.json',dict(passed=True,real_native_records=len(selection),genuine_optimizer_steps=audit['optimizer_steps'],
        metrics=metrics,reference_model_sha256=sha256(run/'reference/model.pt'),no_original_task_final_predictions=True,
        engine_disjointness_qualified=False,annotation_classification_pipeline_qualified=True,
        full_original_health_timing_risk_contract_qualified=False))
    state(status='annotation_numerical_pipeline_qualified_not_full_physical_contract',training_fits=1,metrics=metrics,
        new_model_sessions=0,original_final_scores=0,source_transport_exit_codes=codes)
    print(json.dumps(dict(records=len(selection),metrics=metrics,original_Agent_tasks_run=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--train-per-class',type=int,default=12);p.add_argument('--val-per-class',type=int,default=4)
    p.add_argument('--prior',type=Path);p.add_argument('--worker-timeout',type=int,default=750)
    a=p.parse_args();main(a.run.resolve(),a.source.resolve(),a.train_per_class,a.val_per_class,a.prior.resolve() if a.prior else None,a.worker_timeout)
