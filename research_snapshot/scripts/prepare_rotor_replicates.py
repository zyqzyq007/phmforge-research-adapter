"""Real whole-record Rotor replications; no final-score access or simulated data."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from acquire_forge_pdmb_sources244 import save, sha256
from prepare_classifier_bank import install
from prepare_fault_extensions import features
from prepare_selection_expansion import controls
from real_fault_model import train


def status(root, **changes):
    p=root/'PREPARATION_STATUS305.json'
    previous=json.loads(p.read_text()) if p.exists() else {}
    previous.update(utc=datetime.now(timezone.utc).isoformat(),**changes)
    save(p,previous)
    print(json.dumps(previous),flush=True)


def replay_bank(root, block, data, classes):
    """Reload every actual fold model; independently reconstruct OOF metrics."""
    cv=root/'bank_selection'/block
    report=json.loads((cv/'TRAINING_CV.json').read_text())
    with np.load(data/'train.npz',allow_pickle=False) as z:
        train_data=dict(z)
    index={str(v):i for i,v in enumerate(train_data['sample_ids'])}
    checks=[]
    for record in report['rows']:
        assert record['valid'], 'Stop before Agent admission if a frozen CV recipe fails'
        prediction=np.empty_like(train_data['labels']);assigned=np.zeros(len(prediction),int)
        for fold in report['folds']:
            fit=np.asarray([index[x] for x in fold['fit_ids']])
            held=np.asarray([index[x] for x in fold['heldout_ids']])
            assert not set(train_data['groups'][fit])&set(train_data['groups'][held])
            assert not set(train_data['operating_load_Nm'][fit])&set(train_data['operating_load_Nm'][held])
            model=joblib.load(cv/'cv_models'/(record['name']+'_fold'+str(fold['fold'])+'.joblib'))
            if hasattr(model,'named_steps') and 'standardscaler' in model.named_steps:
                assert np.allclose(model.named_steps['standardscaler'].mean_,
                    train_data['features'][fit].astype(float).mean(0),rtol=1e-10,atol=1e-10)
            prediction[held]=model.predict(train_data['features'][held].astype(float))
            assigned[held]+=1
        assert np.all(assigned==1)
        saved={x['sample_id']:x['predicted_label'] for x in json.loads(
            (cv/(record['name']+'_OOF.json')).read_text())['predictions']}
        assert np.array_equal(prediction,np.asarray([saved[str(i)] for i in train_data['sample_ids']]))
        acc=float(accuracy_score(train_data['labels'],prediction))
        f1=float(f1_score(train_data['labels'],prediction,labels=classes,average='macro',zero_division=0))
        assert abs(acc-record['accuracy'])<1e-12 and abs(f1-record['macro_f1'])<1e-12
        checks.append(dict(model=record['name'],fold_models=3,accuracy=acc,macro_f1=f1,
                           saved_parameters_replayed=True,normalization_fold_local=True))
    return checks


def prepare_block(root, config, block, source_records):
    selected=[x for x in source_records if x['acquisition'] in block['acquisitions']]
    assert len(selected)==80 and all(not x['previously_opened'] for x in selected)
    common=root/'common'/block['id']/'restored/ROTORBROKENBAR'
    data=common/'data';data.mkdir(parents=True,exist_ok=False)
    rows={p:[] for p in ['train','val','test']};allocation=[];width=config['window_samples']
    for record in selected:
        path=Path(config['source_archive'])/record['export_path']
        assert sha256(path)==record['export_sha256']
        with np.load(path,allow_pickle=False) as raw:
            signal=raw['signals']
            label=int(raw['broken_bars']);load=float(raw['load_nm'])
            assert label==record['broken_bars'] and load==record['load_nm']
            assert int(raw['acquisition'])==record['acquisition'] and int(raw['sampling_hz'])==7600
            part,=[p for p,loads in config['load_allocation'].items() if load in loads]
            group='group_'+secrets.token_hex(10)
            for start in range(0,len(signal)-width+1,width):
                wave=signal[start:start+width]
                assert np.isfinite(wave).all()
                identity=secrets.token_hex(16)
                row=dict(features=features(wave),waveforms=wave.T.copy(),labels=label,
                    groups=group,sample_ids=identity,sampling_hz=7600,operating_load_Nm=load)
                rows[part].append(row)
                allocation.append(dict(sample_id=identity,group=group,label=label,split=part,
                    member=record['member'],load=record['load'],acquisition=record['acquisition'],
                    raw_export_sha256=record['export_sha256'],start=start,length=width))
    sets={part:{x['groups'] for x in values} for part,values in rows.items()}
    assert all(not sets[a]&sets[b] for a,b in [('train','val'),('train','test'),('val','test')])
    random=np.random.default_rng(config['numerical_seed']+block['order'])
    for part,values in rows.items():
        random.shuffle(values)
        assert {x['labels'] for x in values}==set(range(5))
        payload={k:np.asarray([x[k] for x in values]) for k in values[0]
                 if part!='test' or k not in ['labels','groups']}
        payload['dataset']=np.asarray('ROTORBROKENBAR')
        np.savez_compressed(data/(part+'.npz'),**payload)
        with np.load(data/(part+'.npz'),allow_pickle=False) as saved:
            assert all(np.array_equal(saved[k],v) for k,v in payload.items())
        if part=='test':
            save(common/'hidden/labels.json',[dict(sample_id=x['sample_ids'],
                group=x['groups'],label=x['labels']) for x in values])
    info=dict(dataset='ROTORBROKENBAR',seed=config['numerical_seed'],class_names=list(range(5)),
        architectures=['mlp'],source_sha256=config['full_source_ZIP_sha256'],
        source_qualification='Source-qualified complete measured acquisitions;five vibration channels bound to full native MAT7.3 records.',
        group_definition='One complete acquired source record. Complete operating loads are disjoint across fitting/development/final. No independent-motor claim.',
        label_definition='Published original broken-bar count:0healthy,1..4broken bars.',
        channels=['Vib_acpe','Vib_acpi','Vib_axial','Vib_base','Vib_carc'],sampling_hz=7600,
        amplitude_units='Calibration unverified;original measured numerical scale retained',
        operating_load_units='Nm',cv_group_field='operating_load_Nm',
        rows_by_split={p:len(v) for p,v in rows.items()},
        groups_by_split={p:len(v) for p,v in sets.items()},
        train_validation_groups={p:sorted(sets[p]) for p in ['train','val']},
        observation_layout='waveforms[query,channel,2048],features[query,75],sampling_hz[query],operating_load_Nm[query]',
        preprocessing='Existing nonoverlapping2048point complete windows;seven time-domain and eight FFT-band statistics per channel;incomplete tails unused uniformly.No normalization,imputation or augmentation.',
        feature_order='statistic-major then channel',
        split_scope='Complete source records and operating loads disjoint across fitting,development andfinal.',
        identity_policy='Opaque sample/record IDs andprivate order;no label-bearing filenames or source ordinals.',
        labels_in_test_input=False,official_original_benchmark_score_claimed=False)
    save(data/'dataset.json',info)
    save(root/'audit'/(block['id']+'_PRIVATE_ALLOCATION.json'),allocation)
    del rows,payload,signal,wave
    status(root,phase='real_neural_training',block=block['id'])
    audit=train(data,common/'model',epochs=config['reference_epochs'])
    assert audit['optimizer_steps']==audit['nonzero_gradient_steps']>0
    assert audit['changed_parameter_tensors']>0 and not audit['test_labels_accessed']
    checkpoint=torch.load(common/'model/model.pt',weights_only=True,map_location='cpu')
    with np.load(data/'val.npz',allow_pickle=False) as dev:
        value=(dev['features'].astype(np.float32)-checkpoint['mean'].numpy())/checkpoint['scale'].numpy()
        for n in [0,2,4]:
            value=value@checkpoint['state_dict'][f'net.{n}.weight'].numpy().T+checkpoint['state_dict'][f'net.{n}.bias'].numpy()
            if n!=4:value=np.maximum(value,0)
        accuracy=float(np.mean(np.asarray(checkpoint['class_names'])[value.argmax(1)]==dev['labels']))
        assert abs(accuracy-audit['selected_validation_accuracy'])<1e-12
    cases=[x for x in config['cases'] if x['block']==block['id']]
    for case in cases:
        (root/case['cohort']).mkdir(exist_ok=True)
        shutil.copytree(common,root/case['cohort']/'restored/ROTORBROKENBAR',copy_function=os.link)
    first=cases[0]
    status(root,phase='six_actual_fixed_fits',block=block['id'],neural_optimizer_steps=audit['optimizer_steps'])
    controls(root,first['case'],data)
    fixed=json.loads((root/'controls'/first['case']/'FROZEN_SELECTION.json').read_text())
    assert len(fixed['candidates'])==6 and all(x['valid'] for x in fixed['candidates'])
    for case in cases[1:]:
        shutil.copytree(root/'controls'/first['case'],root/'controls'/case['case'],copy_function=os.link)
    status(root,phase='eighteen_actual_whole_load_CV_fits',block=block['id'])
    ref=root/'clients-cache/preparation_reference'/block['id'];ref.mkdir(parents=True)
    workspace=root/'clients-cache/preparation_workspace'/block['id'];workspace.mkdir(parents=True)
    install(root/first['cohort'],dict(dataset='ROTORBROKENBAR'),data,ref,workspace)
    checks=replay_bank(root,block['id'],data,list(range(5)))
    result=dict(block=block['id'],native_records=80,source_acquisitions=block['acquisitions'],
        rows=info['rows_by_split'],groups=info['groups_by_split'],all_groups_disjoint=True,
        reference_optimizer_steps=audit['optimizer_steps'],neural_reference_real_fits=1,
        full_data_fixed_estimator_fits=6,fold_estimator_fits=18,
        development_reference_accuracy=accuracy,independent_numpy_reference_forward=True,
        saved_CV_models_replayed=checks,final_scores_opened=False,
        data_hashes={p:sha256(data/(p+'.npz')) for p in ['train','val','test']})
    save(root/'audit'/(block['id']+'_PREPARATION305.json'),result)
    status(root,phase='block_preparation_complete',block=block['id'],completed_block=result)
    return result


def prepare(root):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert config['iteration']=='A305' and not (root/'PREPARATION305.json').exists()
    archive=Path(config['source_archive']);manifest=archive/'SOURCE_MANIFEST.json'
    assert sha256(manifest)==config['source_manifest_sha256']
    receipt=json.loads((archive/'ARCHIVE_VERIFIED.json').read_text())
    assert receipt['status']=='verified' and receipt['source_cleanup']=='complete'
    index=archive/'audit/FULL_NATIVE_RECORD_INDEX304.json'
    bound,=[x for x in json.loads(manifest.read_text())['files'] if x['path']==str(index.relative_to(archive))]
    assert sha256(index)==bound['sha256']
    qualified=json.loads((archive/'audit/EXISTING_WINDOW_VIEW304.json').read_text())
    assert qualified['task_window_view_qualified'] and qualified['records']==320
    source=json.loads(index.read_text())['records']
    assert not set(config['reserved_acquisitions']) & {a for b in config['blocks'] for a in b['acquisitions']}
    blocks=[prepare_block(root,config,block,source) for block in config['blocks']]
    identities=[]
    for block in blocks:
        rows=json.loads((root/'audit'/(block['block']+'_PRIVATE_ALLOCATION.json')).read_text())
        identities.append({(x['member'],x['load'],x['acquisition']) for x in rows})
    assert not identities[0]&identities[1]
    result=dict(iteration='A305',passed=True,blocks=blocks,actual_neural_training_runs=2,
        actual_fixed_estimator_fits=12,actual_fold_estimator_fits=36,
        saved_fold_models_independently_replayed=36,disjoint_new_native_records=160,
        independent_motors_not_claimed=True,reserved_acquisitions_unscored=config['reserved_acquisitions'],
        new_LLM_calls=0,final_scores_opened=False,protocol_sha256=sha256(root/'CAMPAIGN_CONFIG.json'))
    save(root/'PREPARATION305.json',result)
    save(Path(config['project'])/config['result_directory']/'PREPARATION305.json',result)
    status(root,status='completed',phase='all_source_and_real_training_gates_passed',new_LLM_calls=0)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run.resolve()
    try:prepare(root)
    except BaseException as error:
        status(root,status='failed',phase='preparation_error',error_type=type(error).__name__,error=str(error))
        raise
