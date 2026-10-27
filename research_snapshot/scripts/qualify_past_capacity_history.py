"""Export only already-completed measured capacity prefixes from qualified NPZs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from controlled_feedback import save,sha,utc


read=lambda p:json.loads(Path(p).read_text())
def prefix(path,field,n):
    # Read only the numeric prefix. Never materialize the future values or RUL
    # array; the private NPY length is used solely for an alive-at-query check.
    with zipfile.ZipFile(path) as archive,archive.open(field+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise ValueError('Unsupported NPY encoding')
        assert len(shape)==1 and not fortran and not dtype.hasobject and shape[0]>=n
        body=stream.read(n*dtype.itemsize)
        assert len(body)==n*dtype.itemsize
        return np.frombuffer(body,dtype=dtype).copy()


def run(root):
    start=time.monotonic();cfg=read(root/'PROTOCOL380.json')
    assert not(root/'QUALIFICATION380.json').exists()
    src=Path(cfg['data_archive']);old=Path(cfg['pilot_archive'])
    assert sha(src/'SOURCE_MANIFEST.json')==cfg['data_manifest_sha256']
    assert sha(old/'SOURCE_MANIFEST.json')==cfg['pilot_manifest_sha256']
    qualification=read(src/'QUALIFICATION365.json');items=[i for i in qualification['cells'] if i['role']!='final']
    assert len(items)==55 and sum(i['role']=='train' for i in items)==47
    with np.load(old/'private_inputs/training_queries_private.npz',allow_pickle=False) as z:
        train_groups=z['groups'];train_cap=z['past_cap'];train_rates=z['rates']
    with np.load(old/'private_inputs/development_observations.npz',allow_pickle=False) as z:
        dev_cap=z['past_cap'];dev_rates=z['rates'];dev_groups=z['groups']
    development=[i['cell'] for i in items if i['role']=='development']
    expected_cycles=np.arange(10,500);sparse_cycles=np.arange(473,500,3)
    assert len(expected_cycles)==490 and len(sparse_cycles)==9 and sparse_cycles[-1]==497
    def one(item):
        path=src/'private'/item['cell']/'PRIVATE_CELL.npz'
        assert sha(path)==item['curves_sha256']
        cycles=prefix(path,'cycle',490);q=prefix(path,'dq_mAh',490)
        np.testing.assert_array_equal(cycles,expected_cycles)
        assert np.isfinite(q).all() and np.all(q>0)
        if item['role']=='train':
            i=train_groups.tolist().index(item['cell']);ident=item['cell'];expected=train_cap[i];rates=train_rates[i]
        else:
            i=development.index(item['cell']);ident=str(dev_groups[i]);expected=dev_cap[i];rates=dev_rates[i]
        np.testing.assert_array_equal(q[sparse_cycles-10].astype(np.float32),expected)
        output=root/'public'/item['role']/(ident+'.npz');output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output,cycle=cycles,capacity_mAh=q,rates=rates,
            query_cycle=np.array(500),sample_id=np.asarray(ident))
        with np.load(output,allow_pickle=False) as z:
            assert set(z.files)=={'cycle','capacity_mAh','rates','query_cycle','sample_id'}
            assert np.max(z['cycle'])==499 and z['capacity_mAh'].shape==(490,)
        return dict(cell=item['cell'],public_id=ident,role=item['role'],source_path=str(path),
            source_sha256=item['curves_sha256'],public_path=str(output.relative_to(root)),public_sha256=sha(output),
            measured_completed_cycles=490,existing_capacity_values_matched=9,
            new_capacity_observations=481,last_observed_cycle=499,source_rul_array_opened=False,
            source_future_capacity_values_materialized=False,source_native_capacity_reconstruction_already_qualified=True)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,items))
    assert time.monotonic()-start<cfg['budget']['wall_seconds']
    save(root/'PRIVATE_BINDINGS380.json',rows)
    result=dict(passed=True,utc=utc(),training_cells=47,development_cells=8,final_cells_opened=0,
        public_records=55,actual_measured_prefix_values=55*490,old_sparse_values_matched=55*9,
        newly_available_measured_values=55*481,public_cycle_range=[10,499],query_cycle=500,
        original_sparse_cycle_indices=sparse_cycles.tolist(),current_discharge_capacity_exposed=False,
        future_capacity_values_materialized=False,source_RUL_arrays_opened=False,
        public_future_length_metadata=False,new_fits=0,new_predictions=0,new_LLM_calls=0,
        final22_opened=False,A33872_opened=False,seconds=time.monotonic()-start,
        change='Expanded measured observations, notsameinformationasA375. Anycomparisonmustsupplyidenticalpast-historytoolstoordinaryandnewAgent.',
        meaning='Pastcapacitycanprovideactualhindcasttargets; itdoesnotdirectlyvalidatelifetimeorreliablydefine880mAhendpoint.',
        independent_confirmation=False,novelty_confirmed=False)
    save(root/'QUALIFICATION380.json',result);save(root/'TERMINAL380.json',dict(exit_code=0,utc=utc()))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    run(p.parse_args().run.resolve())
