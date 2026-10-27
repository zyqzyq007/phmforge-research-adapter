"""Audit the complete pinned FEMTO CSV before using any lifetime targets."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();assert sha256(a.source)==a.sha256
    csv.field_size_limit(64*1024**2);groups={};bad=[];normalized=Counter();rows=0
    with a.source.open() as f:
        reader=csv.DictReader(f)
        for row in reader:
            rows+=1;name=row['bearing_num'];index=int(row['file_idx'])
            g=groups.setdefault(name,{'rows':0,'operating_metadata':Counter(),'indices':[],
                'segment_indices':[],'rul_percentages':[],'conditions':Counter(),'labels':Counter(),
                'first_raw':{},'last_raw':{}})
            g['rows']+=1;g['indices'].append(index);g['segment_indices'].append(int(row['segment_idx']))
            g['rul_percentages'].append(float(row['rul_percentage']))
            g['operating_metadata'][(row['rpm'],row['load'],row['load_condition'])]+=1
            g['conditions'][row['condition']]+=1;g['labels'][row['label']]+=1
            arrays={k:np.fromstring(row[k].strip().strip('[]'),sep=',')
                    for k in ['horizontal_raw','vertical_raw','features']}
            if any(not len(v) or not np.isfinite(v).all() for v in arrays.values()):bad.append(rows)
            for channel in ['horizontal_raw','vertical_raw']:
                raw=arrays[channel]
                if abs(float(raw.mean()))<1e-8 and abs(float(raw.std())-1)<1e-8:normalized[channel]+=1
                if g['rows']==1:
                    np.save(a.run/'source'/f'{name}_{channel}_first.npy',raw)
                    g['first_raw'][channel]={'samples':len(raw),'mean':float(raw.mean()),
                        'std':float(raw.std()),'sha256_float64':hashlib.sha256(raw.tobytes()).hexdigest()}
                g['last_raw'][channel]={'samples':len(raw),'mean':float(raw.mean()),'std':float(raw.std())}
    output=[]
    for name,g in sorted(groups.items()):
        ids=np.asarray(g.pop('indices'));segment=np.asarray(g.pop('segment_indices'))
        rul=np.asarray(g.pop('rul_percentages'));denom=g['rows']-1
        formula=1-ids/max(denom,1)
        inferred_condition=int(name.split('_')[0].removeprefix('Bearing'))
        expected={1:('1800','4000'),2:('1650','4200'),3:('1500','5000')}.get(inferred_condition)
        operating=[{'rpm':k[0],'load':k[1],'load_condition':k[2],'rows':v} for k,v in g.pop('operating_metadata').items()]
        g.update(bearing=name,operating_metadata=operating,
            expected_from_original_bearing_id=expected,
            operating_matches_original_id=all((x['rpm'],x['load'])==expected for x in operating),
            file_idx_first=int(ids[0]),file_idx_last=int(ids[-1]),file_idx_unique=len(set(ids)),
            file_idx_contiguous=bool(np.array_equal(ids,np.arange(g['rows']))),
            segment_idx_equals_file_idx=bool(np.array_equal(segment,ids)),
            rul_first=float(rul[0]),rul_last=float(rul[-1]),
            max_error_from_fraction_of_available_rows=float(np.max(np.abs(rul-formula))))
        output.append(g)
    result={'source':str(a.source),'source_sha256':a.sha256,'source_hash_verified':True,
        'rows':rows,'bearing_count':len(output),'nonfinite_rows':bad,
        'per_window_zero_mean_unit_std_rows':dict(normalized),'bearings':output,
        'qualification':'pending_original_source_identity_time_and_failure_verification',
        'new_model_calls':0,'new_training_jobs':0}
    save(a.run/'FEMTO_CSV_AUDIT.json',result)
    print(json.dumps({k:result[k] for k in ['rows','bearing_count','per_window_zero_mean_unit_std_rows','nonfinite_rows']}))
    for g in output:
        print(json.dumps({k:g[k] for k in ['bearing','rows','operating_metadata','operating_matches_original_id','file_idx_first','file_idx_last','rul_first','rul_last','max_error_from_fraction_of_available_rows']}))


if __name__=='__main__':main()
