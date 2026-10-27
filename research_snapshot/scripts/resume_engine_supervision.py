"""Recover only absent official source records, then execute the frozen fit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib

import numpy as np

from acquire_forge_pdmb_sources244 import save,sha256
from qualify_engine_supervision import fit_run,PROJECT


def trend(run):
    """Recover only missing native prefixes; validate retained ranges offline."""
    import h5py
    from probe_remote_hdf5 import RangeReader
    from qualify_engine_trend import state, fit_run as fit_trend
    protocol=json.loads((run/'PROTOCOL313.json').read_text())
    assert not (run/'reference').exists()
    assert json.loads((run/'STATUS313.json').read_text())['status']=='source_incomplete_no_fitting'
    requested={i for row in protocol['selection'] for i in row['source_ids']}
    folder=run/'audit/transport_recovery';folder.mkdir(exist_ok=False)
    present={};checks=[]
    for i in range(4):
        out=run/'outputs'/('native'+str(i))
        receipts=json.loads((out/'ranges/ranges.json').read_text())
        reader=RangeReader(protocol['source_url'],protocol['source_bytes'],out/'ranges',0)
        for receipt in receipts:
            body=(out/'ranges'/receipt['file']).read_bytes()
            assert len(body)==receipt['bytes'] and hashlib.sha256(body).hexdigest()==receipt['sha256']
            reader.cache[(receipt['begin'],receipt['end_exclusive'])]=body
        # A missing key fails the zero-byte budget before any network request.
        with h5py.File(reader,'r') as source:
            ids=[str(x) for x in source['data_ID'][:].reshape(-1)]
            lookup={v:i for i,v in enumerate(ids)}
            for path in sorted(out.glob('*.npy')):
                assert path.stem in requested and path.stem not in present
                a=np.load(path,allow_pickle=False)
                b=source['seq_data'][lookup[path.stem],:60,:]
                assert a.shape==(60,33) and np.isfinite(a).all() and np.array_equal(a,b)
                present[path.stem]=dict(path=str(path),sha256=sha256(path))
                checks.append(dict(source_id=path.stem,source_index=lookup[path.stem],
                    path=str(path),sha256=sha256(path),exact_verified_cached_HDF5_prefix=True))
    missing=sorted(requested-set(present),key=int)
    save(folder/'PRESERVED_PREFIX_VERIFICATION313.json',dict(rows=checks,new_HTTP_requests=0,all_exact=True))
    save(folder/'PROTOCOL.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        original_protocol_sha256=sha256(run/'PROTOCOL313.json'),
        reason='Four bounded source workers reached900seconds;1352 genuine prefixes retained; zero fitting or Agent sessions.',
        preserved_records=present,missing_IDs=missing,no_redownload_of_present_records=True,
        worker_seconds=900,workers=4,per_request_attempts=3,labels_splits_models_unchanged=True))
    for f in ['resume_engine_supervision.py','probe_remote_hdf5.py','qualify_engine_trend.py']:
        shutil.copy2(PROJECT/'scripts'/f,folder/f)
    os.environ.update(TMPDIR=str(run/'runtime/tmp'),TMP=str(run/'runtime/tmp'),TEMP=str(run/'runtime/tmp'),MPLCONFIGDIR=str(run/'runtime/mpl'))
    state(run,status='recovering_only_missing_native_prefixes',preserved_records=len(present),missing_records=len(missing),training_fits=0)
    def worker(i):
        path=folder/('ids'+str(i)+'.json');save(path,missing[i::4])
        command=[sys.executable,'-B',str(folder/'probe_remote_hdf5.py'),'--url',protocol['source_url'],
            '--size',str(protocol['source_bytes']),'--ids',str(path),'--output',str(run/'outputs'/('native_recovery'+str(i))),
            '--max-records','0','--time-rows','60','--transport-attempts','3']
        save(folder/('command'+str(i)+'.json'),command)
        with (folder/('worker'+str(i)+'.log')).open('wb') as log:
            try:code=subprocess.run(command,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
            except subprocess.TimeoutExpired:code='timeout'
        save(folder/('exit'+str(i)+'.json'),dict(exit_code=code));return code
    with ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(worker,range(4)))
    save(folder/'EXIT_CODES.json',codes)
    if any(code!=0 for code in codes):
        state(run,status='bounded_source_recovery_incomplete_no_fitting',recovery_exit_codes=codes)
        raise RuntimeError('Only missing-prefix recovery incomplete; do not fabricate sources')
    for i in range(4):
        out=run/'outputs'/('native_recovery'+str(i))
        receipt=json.loads((out/'RESULT.json').read_text())
        for item in receipt['records']:
            a=np.load(out/(item['id']+'.npy'),allow_pickle=False)
            assert a.shape==(60,33) and hashlib.sha256(a.tobytes()).hexdigest()==item['sha256']
    for value in present.values():assert sha256(Path(value['path']))==value['sha256']
    state(run,recovery_exit_codes=codes)
    return fit_trend(run)


def main(run):
    protocol=json.loads((run/'PROTOCOL309.json').read_text())
    assert not (run/'reference').exists(),'No fitting/recovery repetition after weights exist'
    assert json.loads((run/'STATUS309.json').read_text())['status']=='source_transport_incomplete_no_training'
    requested={v['source_id'] for v in protocol['selection']};present={}
    for path in (run/'outputs').glob('native*/*.npy'):
        assert path.stem in requested and path.stem not in present
        values=np.load(path,allow_pickle=False);assert values.shape==(600,33) and np.isfinite(values).all()
        present[path.stem]=dict(path=str(path),sha256=sha256(path))
    missing=sorted(requested-set(present),key=int)
    folder=run/'audit/transport_recovery';folder.mkdir(exist_ok=False)
    save(folder/'PROTOCOL.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        reason='Two original source workers stopped after RemoteDisconnected / incomplete response; no model was fitted.',
        original_protocol_sha256=sha256(run/'PROTOCOL309.json'),preserved_records=present,missing_IDs=missing,
        no_redownload_of_present_records=True,per_request_attempts=3,worker_wall_limit_seconds=900,
        source_labels_splits_models_unchanged=True,new_Agent_sessions=0,new_final_scores=0))
    for f in ['probe_remote_hdf5.py','qualify_engine_supervision.py','resume_engine_supervision.py']:
        shutil.copy2(PROJECT/'scripts'/f,folder/f)
    os.environ.update(TMPDIR=str(run/'runtime/tmp'),TMP=str(run/'runtime/tmp'),TEMP=str(run/'runtime/tmp'),MPLCONFIGDIR=str(run/'runtime/mpl'))
    state=json.loads((run/'STATUS309.json').read_text());state.update(status='recovering_only_missing_native_records',
        preserved_records=len(present),missing_records=len(missing),updated_utc=datetime.now(timezone.utc).isoformat())
    save(run/'STATUS309.json',state);save(PROJECT/'results/engine310/current_run.json',state)
    def worker(index):
        ids=folder/('ids'+str(index)+'.json');save(ids,missing[index::4])
        command=[sys.executable,'-B',str(PROJECT/'scripts/probe_remote_hdf5.py'),
            '--url',protocol['source_url'],'--size',str(protocol['source_bytes']),'--ids',str(ids),
            '--output',str(run/'outputs'/('native_recovery'+str(index))),'--max-records','0','--transport-attempts','3']
        save(folder/('command'+str(index)+'.json'),command)
        with (folder/('worker'+str(index)+'.log')).open('wb') as log:
            try:code=subprocess.run(command,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
            except subprocess.TimeoutExpired:code='timeout'
        save(folder/('exit'+str(index)+'.json'),dict(exit_code=code));return code
    with ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(worker,range(4)))
    save(folder/'EXIT_CODES.json',codes)
    if any(v!=0 for v in codes):
        state.update(status='bounded_source_recovery_incomplete_no_training',recovery_exit_codes=codes)
        save(run/'STATUS309.json',state);save(PROJECT/'results/engine310/current_run.json',state)
        raise RuntimeError('Bounded recovery incomplete; no fake records or sham fitting')
    paths=list((run/'outputs').glob('native*/*.npy'))
    assert len(paths)==len(requested)==2000 and {p.stem for p in paths}==requested
    for ident,entry in present.items():assert sha256(Path(entry['path']))==entry['sha256']
    fit_run(run,codes)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--trend',action='store_true');a=p.parse_args()
    (trend if a.trend else main)(a.run.resolve())
