"""Qualify the source identity and prepare measured HUST classification data.

All load recordings with the same fault code and bearing size stay together.
Source names and row order are private; no final labels are scored here.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from prepare_fault_extensions import features, numeric_array


SEED = 20261020255
SOURCE_SHA = 'aaa26b9627dbcd99111bbd393ab28f13d388a832a40f4567e8de0662f366340a'


def acquire_native(run):
    from acquire_forge_pdmb_sources244 import download
    rows=json.loads((run/'source/native_data_files.json').read_text())
    assert len(rows)==99 and sum(x['content_details']['size'] for x in rows)<1024**3
    output=run/'source/native_mat';output.mkdir(exist_ok=True)
    def fetch(row):
        value=row['content_details'];name=row['filename']
        assert re.fullmatch(r'[A-Z]+[4-8]0[024]\.mat',name)
        pin=dict(dataset=name,filename=name,format='mat',bytes=value['size'],
            lfs_sha256=value['sha256_hash'],git_blob_sha1=None,url=value['download_url'],
            publisher='Mendeley dataset DOI10.17632/cbv7jyx4p9.3',user_agent='Mozilla/5.0')
        try:
            result=download(pin,output,False)
            return dict(filename=name,passed=result['pinned_hash_verified'],bytes=value['size'])
        except Exception as error:
            return dict(filename=name,passed=False,error_type=type(error).__name__,error=str(error))
    with ThreadPoolExecutor(max_workers=12) as pool:results=list(pool.map(fetch,rows))
    summary=dict(passed=all(x['passed'] for x in results),files=results,
        publisher_sha256_available=True,new_LLM_calls=0,Agent_observations_changed=False)
    save(run/'audit/NATIVE_ACQUISITION.json',summary)
    print(json.dumps({'passed':summary['passed'],'files':len(results),'failures':[x for x in results if not x['passed']]}),flush=True)
    if not summary['passed']:raise SystemExit(1)


def verify_native(source, run):
    from scipy.io import loadmat
    metadata=json.loads((run/'source/native_data_files.json').read_text())
    normalized={};native_rows=[]
    for row in metadata:
        path=run/'source/native_mat'/row['filename']
        assert path.stat().st_size==row['content_details']['size']
        assert sha256(path)==row['content_details']['sha256_hash']
        data=loadmat(path);signal=np.asarray(data['data'],np.float64).reshape(-1)
        assert np.isfinite(signal).all() and signal.std()>0
        normalized[row['filename']]=(signal-signal.mean())/signal.std()
        native_rows.append(dict(filename=row['filename'],samples=len(signal),mean=float(signal.mean()),
            std=float(signal.std()),sha256=row['content_details']['sha256_hash'],
            shaft_frequency_hz=float(np.asarray(data['fs']).item()),
            fs_is_sampling_frequency=False,sampling_hz_per_original_publication=51200))
    counts=Counter();maximum_error=0.;mismatches=[]
    csv.field_size_limit(128*1024**2)
    with source.open(newline='') as stream:
        for index,row in enumerate(csv.DictReader(stream)):
            supplied=numeric_array(row['vibration_raw']);start=int(row['segment_idx'])*len(supplied)
            original=normalized[row['filename']][start:start+len(supplied)]
            if len(original)!=len(supplied):raise ValueError('HF segment exceeds source observation')
            error=float(np.max(np.abs(original-supplied)));maximum_error=max(maximum_error,error)
            if error>1e-10:mismatches.append(dict(row=index,filename=row['filename'],max_error=error))
            counts[row['filename']]+=1
    result=dict(passed=not mismatches,publisher_native_files=99,publisher_hashes_verified=True,
        supplied_rows_verified=sum(counts.values()),source_files_covered=len(counts),max_abs_error=maximum_error,
        mismatches=mismatches,native_records=native_rows,
        verified_transform='For each native steady recording: subtract full-record mean, divide full-record std, then contiguous nonoverlapping2560point segments indexed by source segment_idx',
        acceleration_physical_amplitude_preserved_in_HF=False,
        consequence='Real measured signals with exactly recovered preprocessing, not synthetic waves; original amplitude unavailable from HF alone. Native raw MATs retained privately for reproduction.',
        data_scope='Offline whole-record fault classification; full-record normalization is not a causal online protocol',
        fs_field='Native MAT fs is shaft frequency inHz,not sample rate. Publication acquisition rate51200Hz.',
        Agent_observations_or_scores_changed=False,final_scores_not_read=True)
    save(run/'audit/NATIVE_VALUE_CORRESPONDENCE.json',result)
    print(json.dumps({k:result[k] for k in ['passed','publisher_native_files','supplied_rows_verified','source_files_covered','max_abs_error']}),flush=True)
    if not result['passed']:raise SystemExit(1)


def audit(source, run, prepare=False):
    receipt = json.loads(source.with_suffix('.receipt.json').read_text())
    assert receipt['pinned_hash_verified'] and sha256(source) == SOURCE_SHA
    labels = defaultdict(Counter)
    recordings = defaultdict(list)
    lengths, means, stds = Counter(), [], []
    bearing_groups = defaultdict(set)
    vectors, signals, identities, targets, group_ids, provenance = [], [], [], [], [], []
    randomizer = random.Random(SEED)
    row_count = 0
    csv.field_size_limit(128 * 1024**2)
    with source.open(newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            match = re.fullmatch(r'(N|I|O|B|IO|IB|OB)([4-8])0([024])\.mat', row['filename'])
            if match is None:
                raise ValueError(f'Unrecognized source identity: {row["filename"]}')
            fault, size, load = match.groups()
            assert row['bearing_type'] == '620'+size
            assert row['load_condition'] == str(int(load)*100)+'W'
            group = fault+size
            label = int(row['label'])
            labels[label][row['fault_type']] += 1
            bearing_groups[label].add(group)
            recordings[row['filename']].append(int(row['segment_idx']))
            signal = numeric_array(row['vibration_raw'])
            if not np.isfinite(signal).all():
                raise ValueError(f'Nonfinite measured values in row {index}')
            lengths[len(signal)] += 1
            means.append(float(signal.mean()))
            stds.append(float(signal.std()))
            row_count += 1
            if prepare:
                if len(signal) != 2560:
                    raise ValueError('This protocol expects complete supplied 2560-point observations')
                vector = features(signal[:,None])
                assert np.isfinite(vector).all()
                ident = '%032x' % randomizer.getrandbits(128)
                vectors.append(vector); signals.append(signal.astype(np.float32))
                identities.append(ident); targets.append(label); group_ids.append(group)
                provenance.append(dict(sample_id=ident,source_row=index,filename=row['filename'],
                    segment_idx=int(row['segment_idx']),group=group,label=label))
    assert all(len(v)==1 for v in labels.values())
    if any(len(v)<3 for v in bearing_groups.values()):
        raise ValueError('Fewer than three bearing proxies for a source class')
    assert all(len(v)==len(set(v)) for v in recordings.values())
    gaps={name:dict(count=len(v),first=min(v),last=max(v),gaps=max(v)-min(v)+1-len(v)) for name,v in recordings.items()}
    summary=dict(source_pin=receipt['pin'],source_sha256=SOURCE_SHA,rows=row_count,
        label_mapping={str(k):dict(v) for k,v in labels.items()},
        recordings=len(recordings),bearing_groups_per_class={str(k):sorted(v) for k,v in bearing_groups.items()},
        recording_segments=gaps,lengths=dict(lengths),
        window_mean_range=[min(means),max(means)],window_std_range=[min(stds),max(stds)],
        per_window_centered_scaled=bool(max(abs(x) for x in means)<1e-5 and max(abs(x-1) for x in stds)<1e-5),
        signal_origin='Published HF segments; real-source link checked separately; not field failure trajectories',
        final_scores_read=False,new_LLM_calls=0)
    save(run/'audit/HUST_SOURCE_AUDIT.json',summary)
    if not prepare:
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        return
    split={part:[] for part in ['train','val','test']}
    for label in sorted(bearing_groups):
        values=sorted(bearing_groups[label]);random.Random(str(SEED)+':'+str(label)).shuffle(values)
        split['val'].append(values[0]);split['test'].append(values[1]);split['train'].extend(values[2:])
    assert all(not set(split[a])&set(split[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    x=np.asarray(vectors,np.float32); raw=np.asarray(signals,np.float32)
    y=np.asarray(targets); g=np.asarray(group_ids); ids=np.asarray(identities)
    directory=run/'public/datasets/HUST';directory.mkdir(parents=True,exist_ok=False)
    split_counts={};split_order={}
    for part,selected in split.items():
        take=np.flatnonzero(np.isin(g,selected));random.Random(str(SEED)+':order:'+part).shuffle(take)
        split_order[part]=take.tolist()
        payload=dict(features=x[take],waveforms=raw[take],sample_ids=ids[take],dataset=np.asarray('HUST'))
        if part!='test':payload.update(labels=y[take],groups=g[take])
        np.savez_compressed(directory/(part+'.npz'),**payload)
        split_counts[part]=len(take)
    save(run/'hidden/HUST/labels.json',[provenance[i] for i in split_order['test']])
    save(run/'audit/HUST_SPLIT.json',dict(seed=SEED,split=split,observations=provenance,split_order=split_order))
    info=dict(dataset='HUST',seed=SEED,source_sha256=SOURCE_SHA,source_pin=receipt['pin'],
        class_names=sorted(labels),label_mapping={str(k):next(iter(v)) for k,v in labels.items()},
        architectures=['mlp'],source_rows=row_count,
        feature_names=['mean','std','rms','peak_abs','peak_to_peak','skewness','kurtosis']+
            [f'log1p_fft_band_{i}' for i in range(8)],
        preprocessing='Supplied 2560-point windows; seven time-domain and eight FFT-band features; waveforms retained as supplied',
        waveform_scale='Supplied per-window standardized scale' if summary['per_window_centered_scaled'] else 'Published supplied scale; physical acceleration units not independently certified',
        group_definition='Fault code plus bearing size; all loads of a bearing proxy kept together; published identity, not independently certified hardware specimen',
        train_validation_groups={k:v for k,v in split.items() if k!='test'},
        rows_by_split=split_counts,groups_by_split={k:len(v) for k,v in split.items()},
        source_recordings=len(recordings),sampling_hz=51200,
        source_observation_duration_seconds=10,window_samples=2560,
        labels_in_test_input=False,test_source_names_absent=True,
        split_scope='Cross-bearing-size proxies within the given fault classes; no temporal prognostics or field-generalization claim')
    save(directory/'dataset.json',info)
    changed=dict(np.load(directory/'test.npz',allow_pickle=False))
    changed['features']=changed['features']+np.maximum(x[np.isin(g,split['train'])].std(0),1e-6)*.25
    np.savez_compressed(run/'audit/changed_input.npz',**changed)
    save(run/'audit/PREPARED.json',dict(info=info,no_final_scores=True))
    print(json.dumps(dict(prepared=True,rows_by_split=split_counts,groups_by_split=info['groups_by_split'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--acquire-native',action='store_true')
    parser.add_argument('--verify-native',action='store_true')
    args=parser.parse_args()
    if args.acquire_native:acquire_native(args.run)
    elif args.verify_native:verify_native(args.source,args.run)
    else:
        assert args.source is not None
        audit(args.source,args.run,args.prepare)
