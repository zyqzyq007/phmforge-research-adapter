"""Bind all79 published records to original WAV/tachometer observations.

Queries are complete one-minute records, not independent bearing windows.
The two physical bearing states do not support a cross-hardware claim.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import secrets
import zipfile

import numpy as np
from scipy.io import wavfile

from acquire_forge_pdmb_sources244 import save,sha256
from prepare_fault_extensions import features


SEED=20261021257
SOURCE_SHA='783aef49db3384d22d211b3c8071638a7e1b826c2aec5ac22b527c29d2918741'
NATIVE_SHA='372b1f23e162a98aa53e051319f7fcea1d9d2adfcd0361a30b55165feb0c16d4'


def prepare(source,run):
    receipt=json.loads(source.with_suffix('.receipt.json').read_text())
    assert receipt['pinned_hash_verified'] and sha256(source)==SOURCE_SHA
    native=run/'source/mendeley_native/native_recordings.zip'
    assert native.stat().st_size==1160365607 and sha256(native)==NATIVE_SHA
    directory=run/'public/datasets/MENDELEY';directory.mkdir(parents=True,exist_ok=False)
    (directory/'records').mkdir()
    values=[];identities=[];targets=[];records=[];paths=[];speed_ranges=[];trends=[]
    csv.field_size_limit(256*1024**2)
    with zipfile.ZipFile(native) as archive,source.open(newline='') as stream:
        assert len(archive.namelist())==158
        for index,row in enumerate(csv.DictReader(stream)):
            number=index+1
            sample=secrets.token_hex(16)
            rate,signal=wavfile.read(io.BytesIO(archive.read(str(number)+'.wav')))
            assert rate==96000 and signal.dtype==np.int32 and signal.shape==(5760000,)
            text=row['features'];assert text.startswith('[') and text.endswith(']')
            supplied=np.fromstring(text[1:-1],sep=',',dtype=np.int32)
            assert len(supplied)==text.count(',')+1 and np.array_equal(supplied,signal)
            target=int(row['label']);assert target==(0 if number<=38 else 1)
            tach=np.loadtxt(io.BytesIO(archive.read(str(number)+'.csv')),delimiter=',',ndmin=2)
            assert tach.shape[1]==2 and np.isfinite(tach).all() and np.all(np.diff(tach[:,0])>0)
            elapsed=tach[:,0]-tach[0,0];rpm=tach[:,1]*30
            name='records/'+sample+'.npz'
            np.savez_compressed(directory/name,signal=signal,tach_time_sec=elapsed,tach_rpm=rpm,
                sampling_hz=np.asarray(rate),dataset=np.asarray('MENDELEY'))
            vector=features(signal.astype(np.float64)[:,None]);assert np.isfinite(vector).all()
            values.append(vector);identities.append(sample);targets.append(target);paths.append(name)
            speed_ranges.append(row['speed']);trends.append(row['speed_trend'])
            records.append(dict(sample_id=sample,source_row=index,native_record=number,label=target,
                group='record_'+sample,signal_samples=len(signal),signal_sha256=hashlib.sha256(signal.tobytes()).hexdigest(),
                speed_range_rpm=row['speed'],speed_trend=row['speed_trend'],tach_rows=len(tach),
                original_tach_start_unix_seconds=float(tach[0,0]),native_wav_crc32=archive.getinfo(str(number)+'.wav').CRC,
                identical_to_native=True))
    assert len(records)==79
    split={k:[] for k in ['train','val','test']}
    for target in [0,1]:
        available=[i for i,x in enumerate(targets) if x==target]
        random.Random(str(SEED)+':split:'+str(target)).shuffle(available);n=len(available)//5
        split['val']+=available[:n];split['test']+=available[n:2*n];split['train']+=available[2*n:]
    arrays=dict(features=np.asarray(values,np.float32),sample_ids=np.asarray(identities),
        record_files=np.asarray(paths),speed_range_rpm=np.asarray(speed_ranges),speed_trend=np.asarray(trends))
    for part,indices in split.items():
        random.Random(str(SEED)+':order:'+part).shuffle(indices)
        payload={k:v[indices] for k,v in arrays.items()};payload['dataset']=np.asarray('MENDELEY')
        if part!='test':payload.update(labels=np.asarray(targets)[indices],groups=np.asarray([records[i]['group'] for i in indices]))
        np.savez_compressed(directory/(part+'.npz'),**payload)
    save(run/'hidden/MENDELEY/labels.json',[records[i] for i in split['test']])
    save(run/'external_labels/MENDELEY/labels.json',[records[i] for i in split['test']])
    save(run/'audit/MENDELEY_SPLIT.json',dict(seed=SEED,split=split,source_records=records))
    counts={k:len(v) for k,v in split.items()}
    info=dict(dataset='MENDELEY',seed=SEED,source_sha256=SOURCE_SHA,source_pin=receipt['pin'],
        class_names=[0,1],label_mapping={'0':'Normal','1':'Outer-ring defect'},architectures=['mlp'],
        source_rows=79,rows_by_split=counts,groups_by_split=counts,
        group_definition='One complete original60secondrecord; no record crosses splits. Healthy/faulty physical prototypes may be shared across runs; no independenthardware generalization claim.',
        preprocessing='15 numerical time/FFT statistics from every complete native int32WAV. Whole waveform and synchronized tachometer retained in opaque record files; no synthetic signals or resampling.',
        feature_names=['mean','std','rms','peak_abs','peak_to_peak','skewness','kurtosis']+[f'log1p_fft_band_{i}' for i in range(8)],
        sampling_hz=96000,duration_seconds=60,signal_units='Native int32PCM representation of accelerometer voltage; physical gain not invented',
        tachometer='tach_rpm=published_speed_samples*30,per originaldoc; contact-bounce outliers preserved,not silently filtered',
        timing='relative time from first tachometer sample; original Unix origins retained privately; no absolute acquisition-order prior exposed',
        input_record_files='record_files entries are relative to this dataset directory. Each npz contains signal,tach_time_sec,tach_rpm,sampling_hz,dataset. No labels or source record numbers.',
        label_source='Original publisher table and documentation; originaldoc states outerring fault. Source filenames and row order stay private.',
        native_DOI='10.17632/6k6fbzc6vv.1',license='CCBY4.0',publisher_native_sha256=NATIVE_SHA,
        all79complete_raw_signals_equal_native=True,labels_in_test_input=False,
        scope='Within-bench complete-record classification across varying0..900RPM profiles,not cross-device or field diagnosis',
        metadata_corrections='PDMBmetadata9600Hz,3seconds,4faultclasses and PHMForge detailedtask009600..1800RPM do not match this source. Originalshorttask014 does not require those incorrect conditions.')
    save(directory/'dataset.json',info)
    changed=dict(np.load(directory/'test.npz',allow_pickle=False));fit=np.load(directory/'train.npz',allow_pickle=False)['features']
    changed['features']=changed['features']+np.maximum(fit.std(0),1e-6)*.25
    np.savez_compressed(directory/'changed_input.npz',**changed)
    result=dict(passed=True,original_task='pdm_fault_014',source_rows=79,complete_signals_matched=79,
        samples_compared=79*5760000,max_sample_difference=0,actual_sampling_hz=96000,
        rows_by_split=counts,source_physical='Real accelerometer measurements; no invented faultgenerationmechanism',
        native_zip_publisher_sha256_verified=True,no_final_scores_read=True,new_LLM_calls=0)
    save(run/'audit/MENDELEY_SOURCE_QUALIFICATION.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.run)
