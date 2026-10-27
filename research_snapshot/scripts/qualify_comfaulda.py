"""Qualify complete author COMFAULDA records without training or task scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np


def qualify(folder):
    receipt=json.loads((folder/'RECEIPT.json').read_text())
    assert receipt['passed'] and receipt['complete_record_CRC_verified']
    path=folder/'record.csv';raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
    first=raw.splitlines()[0].decode();delimiter=';' if ';' in first else ','
    # Both readers parse the complete original data, including unknown columns.
    a=np.loadtxt(path,delimiter=delimiter,ndmin=2)
    counts=0;maximum_delta=0.;nonfinite=0
    with path.open(newline='') as stream:
        for row in csv.reader(stream,delimiter=delimiter):
            assert len(row)==a.shape[1]
            values=np.asarray([float(x) for x in row])
            nonfinite+=int((~np.isfinite(values)).sum())
            maximum_delta=max(maximum_delta,float(np.max(np.abs(values-a[counts]))))
            counts+=1
    assert counts==a.shape[0] and nonfinite==0 and np.isfinite(a).all() and maximum_delta==0.
    assert a.shape[1]>=8 and a.shape[0]>10000
    dt=np.diff(a[:,0]);assert np.all(dt>0)
    fs=(a.shape[0]-1)/(a[-1,0]-a[0,0])
    expected=a[0,0]+np.arange(a.shape[0])/fs
    time_error=float(np.max(np.abs(a[:,0]-expected)))
    assert time_error<=2e-8
    source_mismatches=[]
    if a.shape[1]!=8:source_mismatches.append('Author table describes8columns;record contains'+str(a.shape[1])+'.Undocumented extra column is not assigned sensor semantics.')
    if a.shape[0]!=250000:source_mismatches.append('Author describes250000rows;actual'+str(a.shape[0])+'.')
    if abs(fs-50000)>1:source_mismatches.append('Author describes50000Hz;stored clock implies'+str(fs)+'Hz.Use actual clock for any future spectra.')
    return dict(passed=True,class_name=folder.parent.name,source_record=receipt['original_member'],raw_sha256=receipt['sha256'],raw_bytes=len(raw),raw_CRC32=f'{zlib.crc32(raw):08x}',delimiter=delimiter,shape=list(a.shape),dual_reader_max_absolute_difference=maximum_delta,nonfinite_entries=nonfinite,time_start=float(a[0,0]),time_end=float(a[-1,0]),sample_rate_from_full_clock=fs,time_grid_max_residual_seconds=time_error,column_min=a.min(axis=0).tolist(),column_max=a.max(axis=0).tolist(),column_std=a.std(axis=0).tolist(),documented_vibration_column_indices_zero_based=[2,3,4,5,6,7],unknown_column_indices_zero_based=list(range(8,a.shape[1])),sensor_mapping_basis='AuthorTable1;extra columns unresolved;no calibrated physical amplitudes or derivedRPMclaimed',source_mismatches=source_mismatches,scope='Selected raw recording is complete finite parseable acquired signal;dataset-level dimensions differ from source prose.',physical_device_count_not_record_count=True,new_agent_sessions=0,new_fits=0,new_final_model_forwards=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();root=args.run.resolve()
    out=root/'qualification';out.mkdir(exist_ok=True)
    rows=[]
    for folder in sorted((root/'records').glob('*/fetch')):
        destination=out/(folder.parent.name+'.json')
        if destination.exists():
            result=json.loads(destination.read_text());assert hashlib.sha256((folder/'record.csv').read_bytes()).hexdigest()==result['raw_sha256']
        elif (folder/'RECEIPT.json').exists() and json.loads((folder/'RECEIPT.json').read_text()).get('passed'):
            result=qualify(folder);destination.write_text(json.dumps(result,indent=2)+'\n')
        else:continue
        rows.append(result)
    summary=dict(qualified_records=len(rows),expected_records=7,all7passed=len(rows)==7 and all(x['passed'] for x in rows),records=rows,new_agent_sessions=0,new_fits=0,new_final_model_forwards=0,whole_outer_ZIP_MD5_verified=False,intermediate_nested_ZIP_CRC_verified=False,selected_CSV_CRC_verified=True,source_mismatches_retained=True,qualification_records_are_confirmation_data=False)
    (root/'QUALIFICATION332.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(qualified_records=len(rows),shapes=[x['shape'] for x in rows],clock_hz=[round(x['sample_rate_from_full_clock'],3) for x in rows],mismatches=[x['source_mismatches'] for x in rows])))


if __name__=='__main__':main()
