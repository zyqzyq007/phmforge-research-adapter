"""Transform official NASA/BEST milling records without imputing measured VB."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from scipy.io import loadmat

CHANNELS = ['smcAC','smcDC','vib_table','vib_spindle','AE_table','AE_spindle']

def scalar(record, field, missing=False):
    x = np.asarray(record[field]).reshape(-1)
    if not x.size and missing:
        return float('nan')
    if x.size != 1:
        raise ValueError('Expected scalar '+field)
    value = float(x[0])
    if not np.isfinite(value) and not missing:
        raise ValueError('Nonfinite metadata '+field)
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.zip) as outer:
        assert outer.testzip() is None
        inner = outer.read('3. Milling/mill.zip')
    with zipfile.ZipFile(io.BytesIO(inner)) as archive:
        assert archive.testzip() is None
        raw = archive.read('mill.mat')
    records = loadmat(io.BytesIO(raw), simplify_cells=True)['mill']
    metadata, features, arrays, hashes = [], [], {}, []
    for index, record in enumerate(records, 1):
        row = {'source_record_id':index}
        for field in ['case','run','time','DOC','feed','material','VB']:
            value = scalar(record, field, missing=field=='VB')
            row[field] = int(value) if field in ['case','run','material'] else value
        row['query_id'] = f"MILL_case{row['case']:02d}_run{row['run']:03d}_source{index:03d}"
        feature = {k:v for k,v in row.items() if k != 'VB'}
        shapes = {}
        for channel in CHANNELS:
            values = np.asarray(record[channel], dtype=np.float64).reshape(-1)
            assert len(values) > 0 and np.isfinite(values).all()
            key = f'record_{index:03d}_{channel}'
            arrays[key] = values
            shapes[channel] = list(values.shape)
            hashes.append({'key':key,'shape':list(values.shape),'dtype':'float64',
                           'canonical_C_array_sha256':hashlib.sha256(values.tobytes()).hexdigest()})
            for name, x in [('full',values),('middle',values[len(values)//3:2*len(values)//3])]:
                prefix = channel+'_'+name+'_'
                feature.update({prefix+'mean':float(np.mean(x)),prefix+'std':float(np.std(x)),
                                prefix+'rms':float(np.sqrt(np.mean(x*x))),
                                prefix+'p10':float(np.quantile(x,.1)),prefix+'p50':float(np.quantile(x,.5)),
                                prefix+'p90':float(np.quantile(x,.9))})
        row['sample_count'] = len(arrays[f'record_{index:03d}_{CHANNELS[0]}'])
        assert len({len(arrays[f'record_{index:03d}_{c}']) for c in CHANNELS}) == 1
        feature['sample_count'] = row['sample_count']
        metadata.append(row)
        features.append(feature)
    table = pd.DataFrame(metadata)
    table.to_csv(args.output/'source_records.csv',index=False)
    pd.DataFrame(features).to_csv(args.output/'same_cut_features.csv',index=False)
    np.savez_compressed(args.output/'sensor_records.npz',**arrays)
    summaries = []
    for case, group in table.groupby('case',sort=True):
        summaries.append({'case':int(case),'records':len(group),'finite_VB':int(group.VB.notna().sum()),
                          'missing_VB':int(group.VB.isna().sum()),
                          'run_ids':group.run.tolist(),'material':group.material.unique().tolist(),
                          'DOC':group.DOC.unique().tolist(),'feed':group.feed.unique().tolist(),
                          'duplicate_case_run':bool(group.duplicated(['case','run']).any())})
    manifest = {'status':'finished','checked_utc':datetime.now(timezone.utc).isoformat(),
                'source_zip_sha256':hashlib.sha256(args.zip.read_bytes()).hexdigest(),
                'nested_zip_sha256':hashlib.sha256(inner).hexdigest(),'MAT_sha256':hashlib.sha256(raw).hexdigest(),
                'records':len(table),'cases':summaries,'sensor_arrays':hashes,
                'missing_policy':'Original absent/NaN VB preserved, no interpolation, no wear cap, no record deletion.',
                'features':'Fixed full-cut and middle-third statistics; same recording only, no labels or other runs used.',
                'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()}}
    (args.output/'TRANSFORM_MANIFEST.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'finished','records':len(table),'case_count':len(summaries),
                      'finite_VB':int(table.VB.notna().sum()),'missing_VB':int(table.VB.isna().sum()),'cases':summaries}))

if __name__ == '__main__':
    main()
