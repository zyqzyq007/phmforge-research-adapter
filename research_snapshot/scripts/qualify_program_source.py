"""Qualify whole acquired records and make provenance-bound window inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import uuid
import zlib

import numpy as np


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def stats(waves):
    x=np.asarray(waves,dtype=float);centered=x-x.mean(axis=1,keepdims=True)
    std=np.sqrt(np.mean(centered**2,axis=1));rms=np.sqrt(np.mean(x**2,axis=1))
    z=centered/np.maximum(std[:,None,:],1e-12)
    return np.concatenate([x.mean(axis=1),np.log(np.maximum(std,1e-12)),np.log(np.maximum(rms,1e-12)),
        np.ptp(x,axis=1),np.mean(z**3,axis=1),np.mean(z**4,axis=1),
        np.max(np.abs(x),axis=1)/np.maximum(rms,1e-12),np.mean(np.abs(x),axis=1)],axis=1)


def main(root,protocol_name='PROTOCOL336.json',handle_name='HANDLE336.json',tag='336'):
    protocol=json.loads((root/protocol_name).read_text())
    expected=len(protocol['selected_records'])
    assert expected>0 and json.loads((root/handle_name).read_text())['passed']==expected
    directory=root/'qualification';directory.mkdir(exist_ok=False)
    roles=protocol.get('qualified_roles',['train','development','independent_check','final'])
    assert set(roles)=={row['role'] for row in protocol['selected_records']}
    splits={role:[] for role in roles};identities=[]
    for row in protocol['selected_records']:
        receipt=json.loads((root/'records'/row['unit_id']/'RECEIPT.json').read_text())
        assert receipt['passed'] and receipt['CSV_CRC_verified']
        path=Path(receipt['raw_path']);raw=path.read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256']
        assert f'{zlib.crc32(raw):08x}'==row['CRC32']
        delimiter=';' if b';' in raw.splitlines()[0] else ','
        a=np.loadtxt(path,delimiter=delimiter,ndmin=2)
        assert a.shape[1]>=8 and np.isfinite(a).all()
        count=0
        with path.open(newline='') as stream:
            for record in csv.reader(stream,delimiter=delimiter):
                parsed=np.asarray([float(x) for x in record]);assert len(parsed)==a.shape[1] and np.array_equal(parsed,a[count]);count+=1
        assert count==len(a) and np.all(np.diff(a[:,0])>0)
        fs=(len(a)-1)/(a[-1,0]-a[0,0]);grid=a[0,0]+np.arange(len(a))/fs
        assert np.max(np.abs(a[:,0]-grid))<2e-8
        n=protocol['window_points'];nw=len(a)//n;assert nw>=4
        group=uuid.uuid4().hex;ids=[uuid.uuid4().hex for _ in range(nw)]
        waves=a[:nw*n,protocol['sensor_columns_zero_based']].reshape(nw,n,6).copy()
        item=dict(waveforms=waves,features=stats(waves),labels=np.repeat(row['label'],nw),
            sample_ids=np.asarray(ids),groups=np.repeat(group,nw),condition_id=np.repeat(row['target_condition_Hz'],nw),
            rotation_hz=np.repeat(row['recorded_rotation_Hz'],nw),sampling_rate_hz=np.repeat(fs,nw))
        assert np.isfinite(item['features']).all()
        splits[row['role']].append(item)
        identities.append(dict(unit=row['unit_id'],raw_sha256=receipt['sha256'],CRC32=row['CRC32'],role=row['role'],
            actual_shape=list(a.shape),actual_fs_hz=fs,delimiter=delimiter,dual_reader_exact=True,
            group=group,sample_ids=ids,source_class=row['class_name'],label=row['label'],
            window_count=nw,discarded_tail_rows=len(a)-nw*n,unknown_columns_excluded=list(range(8,a.shape[1]))))
    for role,items in splits.items():
        values={k:np.concatenate([x[k] for x in items],axis=0) for k in items[0]}
        np.savez_compressed(directory/(role+'.npz'),**values)
        if role=='independent_check':
            np.savez_compressed(directory/(role+'_observations.npz'),**{k:v for k,v in values.items() if k!='labels'})
    sample_sets=[{i for row in identities if row['role']==role for i in row['sample_ids']} for role in splits]
    group_sets=[{row['group'] for row in identities if row['role']==role} for role in splits]
    assert all(not a&b for i,a in enumerate(sample_sets) for b in sample_sets[i+1:])
    assert all(not a&b for i,a in enumerate(group_sets) for b in group_sets[i+1:])
    save(directory/('IDENTITIES'+tag+'.json'),identities)
    result=dict(passed=True,whole_acquisitions=expected,source_rigs=1,roles={r:dict(records=len(v),windows=sum(len(x['labels']) for x in v)) for r,v in splits.items()},
        record_CRC_and_SHA_verified=True,dual_readers_exact=True,source_metadata_mismatches_retained=True,
        whole_outer_MD5_verified=False,extra_ninth_column_semantics_not_invented=True,
        files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in directory.glob('*.npz')},
        physical_record_and_sample_IDs_disjoint=True,source_filenames_absent_from_observation_arrays=True,
        new_LLM_calls=0,new_PHM_fits=0,new_model_forwards=0,confirmation_scenarios=0)
    save(root/('QUALIFICATION'+tag+'.json'),result);print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--protocol',default='PROTOCOL336.json');p.add_argument('--handle',default='HANDLE336.json');p.add_argument('--tag',default='336');a=p.parse_args();main(a.run.resolve(),a.protocol,a.handle,a.tag)
