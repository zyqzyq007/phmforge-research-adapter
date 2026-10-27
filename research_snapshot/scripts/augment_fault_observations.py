"""Restore omitted observed signals without changing labels or split membership.

Private ID maps and permutations are retained for exact reconstruction. No
final predictions or scores are used to decide these source-based repairs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import secrets
import shutil

import numpy as np
from scipy.io import loadmat

from prepare_fault_extensions import numeric_array
from acquire_forge_pdmb_sources244 import save,sha256


def augment(run,name):
    job={'CWRU':'forge251_pdm_fault_001','MFPT':'forge251_pdm_fault_013','ELECTRICMOTORVIBRATIONS':'forge251_pdm_fault_015'}
    if name=='HUST':
        base=run/'restored/255';source=base/'public/datasets/HUST'
        gold=json.loads((base/'hidden/HUST/labels.json').read_text())
        provenance=json.loads((base/'audit/HUST_SPLIT.json').read_text())['observations']
    else:
        base=run/'restored/252';source=base/'public_data'/job[name]/'datasets'/name
        gold=json.loads((base/'hidden'/name/'labels.json').read_text())
        provenance=(json.loads((run/'restored/246/audit/ElectricMotorVibrations_split.json').read_text())['observations']
                    if name=='ELECTRICMOTORVIBRATIONS' else [])
    old_info=json.loads((source/'dataset.json').read_text())
    old={part:{k:v for k,v in np.load(source/(part+'.npz'),allow_pickle=False).items()}
         for part in ['train','val','test']}
    gold_by_id={x['sample_id']:x for x in gold}
    csv.field_size_limit(256*1024**2)
    extras={};native_checks=[]
    if name=='CWRU':
        path=run/'restored/244/inputs/pdmb_attempt2/CWRU.csv';assert sha256(path)==old_info['source_sha256']
        extras={part:{} for part in old}
        observation_note='Feature-only control: pinnedsourceCSVprovides17numericfeatures,but no raw waveform or originalrecording identity. No waveform invented. Cross-block proxies are not certified physical bearings. Not suitable for claims about raw-signal method choice.'
    elif name=='MFPT':
        path=run/'restored/244/inputs/pdmb_attempt2/MFPT.csv';assert sha256(path)==old_info['source_sha256']
        source_rows=[]
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                raw=numeric_array(row['vibration_raw']);assert raw.shape==(2560,) and np.isfinite(raw).all()
                source_rows.append((raw,numeric_array(row['features']),int(row['label']),float(row['sampling_rate'])))
        for part,data in old.items():
            records=[];rates=[]
            for i,identity in enumerate(data['sample_ids'].astype(str)):
                number=gold_by_id[identity]['original_row'] if part=='test' else int(identity)
                raw,features,label,rate=source_rows[number]
                assert np.array_equal(features.astype(data['features'].dtype),data['features'][i])
                # The original preparer casts the full waveform to float32
                # before averaging. Reproduce that exact recipe, not float64
                # averaging followed by a tolerance chosen after a mismatch.
                assert np.array_equal(raw.astype(np.float32).reshape(256,10).mean(1),data['signals'][i])
                assert label==(gold_by_id[identity]['label'] if part=='test' else data['labels'][i])
                records.append(raw[None,:]);rates.append(rate)
            extras[part]=dict(waveforms=np.asarray(records),sampling_hz=np.asarray(rates))
        observation_note='Complete source2560point vibration waveforms[query,channel,point] supplied in addition to old256point blockmeans; sample rate from each source row. Source load unknown for allIR records; load metadata omitted uniformly,not invented. No calibrated physical amplitude claimed.'
    elif name=='ELECTRICMOTORVIBRATIONS':
        path=run/'restored/244/inputs/pdmb_attempt2/ElectricMotorVibrations.csv';assert sha256(path)==old_info['source_sha256']
        with path.open(newline='') as f:
            signals=[np.stack([numeric_array(row[c]) for c in ['AccX','AccY','AccZ']]) for row in csv.DictReader(f)]
        mapping={x['sample_id']:x for x in provenance}
        for part,data in old.items():
            records=[]
            for identity in data['sample_ids'].astype(str):
                entry=mapping[identity];raw=signals[entry['original_row']][:,entry['start']:entry['start']+entry['length']]
                assert raw.shape==(3,2048) and np.isfinite(raw).all();records.append(raw)
            extras[part]=dict(waveforms=np.asarray(records))
        observation_note='Complete actual source threeaxis2048point windows[query,channel,point],not only45summaries. SamplingHz and calibrated units unresolved,not invented. Torque/ohm metadata not exposed without confirmed observation semantics.'
    else:
        native={};settings={}
        for entry in json.loads((base/'audit/NATIVE_VALUE_CORRESPONDENCE.json').read_text())['native_records']:
            path=base/'source/native_mat'/entry['filename'];assert sha256(path)==entry['sha256']
            mat=loadmat(path);raw=np.asarray(mat['data'],np.float64).reshape(-1)
            native[entry['filename']]=raw;settings[entry['filename']]=entry
            native_checks.append(dict(filename=entry['filename'],sha256=entry['sha256'],samples=len(raw)))
        mapping={x['sample_id']:x for x in provenance}
        for part,data in old.items():
            records=[];families=[];loads=[];shaft=[];means=[];scales=[]
            for i,identity in enumerate(data['sample_ids'].astype(str)):
                entry=mapping[identity];setting=settings[entry['filename']]
                raw=native[entry['filename']][entry['segment_idx']*2560:(entry['segment_idx']+1)*2560]
                assert np.array_equal(((raw-setting['mean'])/setting['std']).astype(np.float32),data['waveforms'][i])
                match=re.fullmatch(r'(?:N|I|O|B|IO|IB|OB)([4-8])0([024])\.mat',entry['filename']);assert match
                records.append(raw[None,:]);families.append('620'+match.group(1));loads.append(int(match.group(2))*100)
                shaft.append(setting['shaft_frequency_hz']);means.append(setting['mean']);scales.append(setting['std'])
            extras[part]=dict(waveforms_native=np.asarray(records),bearing_family=np.asarray(families),
                nominal_load_watts=np.asarray(loads),shaft_frequency_hz=np.asarray(shaft),
                native_record_mean=np.asarray(means),native_record_scale=np.asarray(scales))
        observation_note='Originalfloat64 nativeMAT2560point windows[query,1,point] provided as waveforms_native; old normalizedwaveforms unchanged. Actual bearingfamily6204..6208,nominal0/200/400W, native shaftfrequencyHz, fullrecordmean/std exposed. Sample rate51200Hz; native fs is not sample rate. Offline complete-record classification only; no made-up physicalgain.'
    destination=run/'public/datasets'/name;destination.mkdir(parents=True,exist_ok=False)
    all_groups=set(gold_by_id[i]['group'] for i in gold_by_id)
    for data in old.values():all_groups.update(data.get('groups',[]).tolist() if isinstance(data.get('groups'),np.ndarray) else [])
    aliases={str(g):'group_'+secrets.token_hex(10) for g in sorted(all_groups)}
    maps={};new_gold=[];equivalence=[]
    for part,data in old.items():
        old_ids=data['sample_ids'].astype(str);new_ids=np.asarray([secrets.token_hex(16) for _ in old_ids])
        order=secrets.SystemRandom().sample(range(len(old_ids)),len(old_ids));maps[part]=dict(old_ids=old_ids.tolist(),new_ids=new_ids.tolist(),permutation=order)
        payload={**data,**extras[part], 'sample_ids':new_ids,'dataset':np.asarray(name)}
        if 'groups' in payload:payload['groups']=np.asarray([aliases[str(g)] for g in payload['groups']])
        for key,value in list(payload.items()):
            if value.ndim and len(value)==len(old_ids):payload[key]=value[order]
        np.savez_compressed(destination/(part+'.npz'),**payload)
        inverse=np.argsort(order)
        assert np.array_equal(payload['features'][inverse],data['features'])
        if part!='test':assert np.array_equal(payload['labels'][inverse],data['labels'])
        else:
            assert 'labels' not in payload and 'groups' not in payload
            for index in order:
                row=dict(gold_by_id[old_ids[index]]);row['sample_id']=str(new_ids[index]);row['group']=aliases[str(row['group'])];new_gold.append(row)
        equivalence.append(dict(split=part,rows=len(old_ids),old_feature_vectors_preserved=True,split_membership_and_labels_preserved=True,new_file_sha256=sha256(destination/(part+'.npz'))))
    # The backend needs the numerical training seed. Identity generation and
    # display order use independent private randomness, not this public seed.
    info=dict(old_info)
    info.update(observation_augmentation=observation_note,identity_policy='PrivatecryptographicIDs,privatedisplaypermutationandopaquegroups; no sourceordinalsorclassprefixes exposed',augmentation_is_test_score_independent=True,
        old_feature_vectors_preserved=True,official_benchmark_score_claimed=False)
    # Do not carry population targets or legacy source identities into the
    # observation contract after private IDs and groups have been rebuilt.
    removed_metadata={key:info.pop(key) for key in
                      ['groups_per_class','train_validation_test_groups'] if key in info}
    save(run/'audit'/(name+'_REMOVED_TARGET_METADATA261.json'),removed_metadata)
    info['train_validation_groups']={part:sorted(set(np.load(destination/(part+'.npz'),allow_pickle=False)['groups'].tolist())) for part in ['train','val']}
    if name=='MFPT':info.update(label_mapping={'0':'Normal','1':'Inner-race fault','2':'Outer-race fault'},waveform_layout='waveforms[query,channel,point],sampling_hz[query]')
    save(destination/'dataset.json',info)
    for directory in ['hidden','external_labels']:save(run/directory/name/'labels.json',new_gold)
    save(run/'audit'/(name+'_PRIVATE_ID_MAP261.json'),dict(groups=aliases,splits=maps))
    save(run/'audit'/(name+'_OBSERVATION_REPAIR261.json'),dict(passed=True,dataset=name,source_sha256=old_info['source_sha256'],equivalence=equivalence,observation_note=observation_note,native_files_verified=native_checks,new_model_calls=0,final_scores_used_for_repair=False))
    # Generic backend dependence control is private and never an Agent input.
    changed=dict(np.load(destination/'test.npz',allow_pickle=False));fit=np.load(destination/'train.npz',allow_pickle=False)['features']
    changed['features']=changed['features']+np.maximum(fit.std(0),1e-6)*.25
    np.savez_compressed(destination/'changed_input.npz',**changed)
    print(json.dumps(dict(dataset=name,passed=True,rows=info['rows_by_split'],new_model_calls=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--datasets',nargs='+',default=['MFPT','ELECTRICMOTORVIBRATIONS','HUST'])
    a=p.parse_args()
    for name in a.datasets:augment(a.run.resolve(),name)
