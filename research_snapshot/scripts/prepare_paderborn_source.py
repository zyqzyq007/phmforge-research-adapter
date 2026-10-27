"""Bind all supplied PU windows to original multi-rate MATLAB records.

Fault-code prefixes remain private; all recordings of a bearing stay together.
Per-channel clocks and inverse normalization are preserved as observations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import re
import secrets
import csv

import numpy as np
from scipy.io import loadmat

from acquire_forge_pdmb_sources244 import save,sha256
from native_archive_members import members
from prepare_fault_extensions import features,numeric_array


SEED=20261021257
SOURCE_SHA='0e63a02303e3464e411667c68df62d36811b7f86a487e856d1cad640c13559f4'
SOURCE_COLUMNS=['force_raw','phase_current_1_raw','phase_current_2_raw','speed_raw','torque_raw','vibration_raw']
NATIVE_NAMES=['force','phase_current_1','phase_current_2','speed','torque','vibration_1']
RATES=np.asarray([4000,64000,64000,4000,4000,64000])
PREFIX={'NORMAL':'K','OR':'KA','IR':'KI','OR + IR':'KB'}
TARGET={'NORMAL':0,'IR':1,'OR':2,'OR + IR':3}


def fingerprint(bearing,segment,arrays):
    digest=hashlib.sha256((bearing+':'+str(segment)+'|').encode())
    for a in arrays:digest.update(np.asarray(a,dtype='<f8').tobytes())
    return digest.hexdigest()


def native_index(run):
    directory=run/'source/paderborn_native';paths=sorted(directory.glob('*.rar'));assert len(paths)==32
    def one(path):
        prior=json.loads(path.with_name(path.stem+'.receipt.json').read_text());assert sha256(path)==prior['local_sha256']
        rows=[];failures=[];count=0
        for name,payload in members(path):
            if not name.endswith('.mat'):continue
            count+=1
            try:
                parsed=loadmat(io.BytesIO(payload),simplify_cells=True)
                data=next(v for k,v in parsed.items() if not k.startswith('__'))
                channel=[next(v for v in data['Y'] if v['Name']==key) for key in NATIVE_NAMES]
                y=[np.asarray(v['Data'],np.float64).reshape(-1) for v in channel]
                assert all(np.isfinite(v).all() for v in y)
                mean=np.asarray([v.mean() for v in y]);std=np.asarray([v.std() for v in y]);assert (std>0).all()
                x=[np.asarray(data['X'][int(channel[i]['XIndex'])-1]['Data'],np.float64).reshape(-1) for i in [0,1]]
                assert all(np.all(np.diff(v)>0) for v in x)
                assert all(int(v['XIndex'])==j for v,j in zip(channel,[1,2,2,1,1,2]))
                match=re.search(r'N(\d+)_M(\d+)_F(\d+)_(K[AIB]?\d+)_',name);assert match and match.group(4)==path.stem
                operating=[int(match.group(1))*100,int(match.group(2))/10,int(match.group(3))*100]
                for segment in range(min(map(len,y))//4096):
                    start=segment*4096;end=start+4096
                    arrays=[(v[start:end]-m)/s for v,m,s in zip(y,mean,std)]
                    rows.append(dict(hash=fingerprint(path.stem,segment,arrays),bearing=path.stem,
                        native_member=name,segment=segment,means=mean,scales=std,
                        times=np.stack([v[start:end] for v in x]),operating=operating))
            except Exception as error:
                failures.append(dict(member=name,error_type=type(error).__name__,error=str(error)))
        return rows,failures,count
    with ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(one,paths))
    rows=[v for q,_,_ in reports for v in q]
    np.savez_compressed(run/'audit/PU_NATIVE_WINDOW_INDEX.npz',fingerprints=np.asarray([x['hash'] for x in rows]),
        means=np.asarray([x['means'] for x in rows]),scales=np.asarray([x['scales'] for x in rows]),
        times=np.asarray([x['times'] for x in rows]),operating=np.asarray([x['operating'] for x in rows]))
    save(run/'audit/PU_NATIVE_WINDOW_INDEX.json',dict(rows=[{k:x[k] for k in ['hash','bearing','native_member','segment']} for x in rows],
        native_mat_files=sum(n for _,_,n in reports),native_archives=32,parse_failures=[x for _,f,_ in reports for x in f],
        publisher_whole_sha256_available=False,local_archive_hashes_verified=True,new_LLM_calls=0))
    print(json.dumps(dict(native_mat_files=sum(n for _,_,n in reports),indexed_windows=len(rows),parse_failures=sum(len(f) for _,f,_ in reports))),flush=True)


def prepare(source,run):
    receipt=json.loads(source.with_suffix('.receipt.json').read_text());assert receipt['pinned_hash_verified'] and sha256(source)==SOURCE_SHA
    index=np.load(run/'audit/PU_NATIVE_WINDOW_INDEX.npz',allow_pickle=False)
    metadata=json.loads((run/'audit/PU_NATIVE_WINDOW_INDEX.json').read_text());lookup={}
    for i,key in enumerate(index['fingerprints']):lookup.setdefault(str(key),[]).append(i)
    vectors=[];waveforms=[];native_indices=[];labels=[];identities=[];physical=[];provenance=[];unmatched=[]
    csv.field_size_limit(1024**2)
    with source.open(newline='') as stream:
        for row_number,row in enumerate(csv.DictReader(stream)):
            condition=row['condition'];number=int(row['bearing_num']);bearing=PREFIX[condition]+str(number).zfill(3 if condition=='NORMAL' else 2)
            label=int(row['label']);assert label==TARGET[condition]
            arrays=[numeric_array(row[k]) for k in SOURCE_COLUMNS]
            assert all(a.shape==(4096,) and np.isfinite(a).all() for a in arrays)
            key=fingerprint(bearing,int(row['segment_idx']),arrays)
            if key not in lookup:unmatched.append(dict(source_row=row_number,bearing=bearing,segment=int(row['segment_idx'])));continue
            native=lookup[key][0];ident=secrets.token_hex(16);signal=np.stack(arrays,axis=1)
            vector=features(signal);assert np.isfinite(vector).all()
            vectors.append(vector);waveforms.append(signal.T.astype(np.float32));native_indices.append(native)
            labels.append(label);identities.append(ident);physical.append(bearing)
            provenance.append(dict(sample_id=ident,source_row=row_number,label=label,bearing=bearing,
                native_member=metadata['rows'][native]['native_member'],native_segment=int(row['segment_idx']),
                matched_fingerprint=key,original_bearing_number=number,condition=condition))
    save(run/'audit/PU_UNMATCHED_SOURCE_ROWS.json',unmatched)
    if unmatched:raise ValueError(f'{len(unmatched)} source rows not bound to original measurements; no qualification')
    assert len(provenance)==7679 and len(set(physical))==32
    split={k:[] for k in ['train','val','test']}
    for label in sorted(set(labels)):
        groups=sorted({g for g,l in zip(physical,labels) if l==label});random.Random(str(SEED)+':split:'+str(label)).shuffle(groups)
        n=max(1,len(groups)//5);assert len(groups)>2*n
        split['val']+=groups[:n];split['test']+=groups[n:2*n];split['train']+=groups[2*n:]
    group_alias={g:'bearing_'+secrets.token_hex(12) for g in set(physical)}
    directory=run/'public/datasets/PADERBORN';directory.mkdir(parents=True,exist_ok=False)
    native_indices=np.asarray(native_indices);n=len(provenance)
    arrays=dict(features=np.asarray(vectors,np.float32),waveforms=np.asarray(waveforms),
        sample_ids=np.asarray(identities),native_channel_means=index['means'][native_indices],
        native_channel_scales=index['scales'][native_indices],channel_times_sec=index['times'][native_indices],
        nominal_speed_rpm=index['operating'][native_indices,0],nominal_torque_nm=index['operating'][native_indices,1],
        nominal_radial_force_n=index['operating'][native_indices,2])
    counts={}
    for part,groups in split.items():
        take=np.flatnonzero(np.isin(physical,groups));np.random.default_rng(secrets.randbits(64)).shuffle(take)
        payload={k:v[take] for k,v in arrays.items()};payload['dataset']=np.asarray('PADERBORN')
        if part!='test':payload.update(labels=np.asarray(labels)[take],groups=np.asarray([group_alias[physical[i]] for i in take]))
        np.savez_compressed(directory/(part+'.npz'),**payload);counts[part]=len(take)
        if part=='test':
            save(run/'hidden/PADERBORN/labels.json',[dict(provenance[i],group=physical[i]) for i in take])
            save(run/'external_labels/PADERBORN/labels.json',[dict(provenance[i],group=physical[i]) for i in take])
    save(run/'audit/PU_SPLIT.json',dict(seed=SEED,split=split,group_alias=group_alias,source_rows=provenance))
    info=dict(dataset='PADERBORN',seed=SEED,source_sha256=SOURCE_SHA,source_pin=receipt['pin'],
        class_names=[0,1,2,3],label_mapping={'0':'Normal','1':'Inner-race fault','2':'Outer-race fault','3':'Inner+outer-race fault'},architectures=['mlp'],
        rows_by_split=counts,groups_by_split={k:len(v) for k,v in split.items()},source_rows=n,
        group_definition='Complete published bearing code includingK/KA/KI/KBprefix; all80runs and operatingconditions of each bearing stay together; opaque public aliases',
        preprocessing='15independenttime/FFTstatistics for each of6sourcewindows; no resampling or invented synchronization. Full-record normalized source values exactly bound to publisherMATs.',
        channels=SOURCE_COLUMNS,nominal_channel_sampling_hz=RATES.tolist(),channel_time_axis_index=[0,1,1,0,0,1],
        waveform_layout='waveforms[query,channel,point]; channel_times_sec[query,axis,point] gives actualoriginalclock; channels have different physicalstart/end times even whensegment_idxequals',
        inverse_normalization='Original observed values=waveforms*native_channel_scales[:,:,None]+native_channel_means[:,:,None]. Native waveform float64 correspondence checked before float32publicstorage.',
        operating_metadata='Actual originalfile settings:Nxxspeed*100RPM,Mxxtorque/10Nm,Fxxradialforce*100N; available perquery without faultprefixfilename',
        units='Native sensors; sourceclockandscale retained. Do not assume a common64000Hz for force,speed,torque or treat these channel windows as synchronized.',
        source_corrections='Bare numericbearingIDs collide acrossfaultprefixes; class3isinner+outer,notcage. Actualchannels6,not3. Normalization removed physicalscale but inverse parameters now provided.',
        labels_in_test_input=False,identity_policy='Private cryptographicIDs and private displaypermutation independent of publicmodelseed',
        source_license='PublisherCCBY-NC4.0; HFtagCCBY4.0doesnot override upstreamterms; this is academicnoncommercialresearch',
        official_benchmark_score_claimed=False)
    save(directory/'dataset.json',info)
    changed=dict(np.load(directory/'test.npz',allow_pickle=False));fit=np.load(directory/'train.npz',allow_pickle=False)['features']
    changed['features']=changed['features']+np.maximum(fit.std(0),1e-6)*.25;np.savez_compressed(directory/'changed_input.npz',**changed)
    summary=dict(passed=True,original_short_task='pdm_fault_011',source_rows=7679,all6channel_windows_native_exact_match=True,
        native_archives=32,native_mat_files=metadata['native_mat_files'],rows_by_split=counts,groups_by_split=info['groups_by_split'],
        source_ID_and_clock_conflicts_resolved=True,final_scores_not_read=True,new_LLM_calls=0)
    save(run/'audit/PU_SOURCE_QUALIFICATION.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--phase',choices=['index','prepare'],required=True);a=p.parse_args()
    if a.phase=='index':native_index(a.run)
    else:prepare(a.source,a.run)
