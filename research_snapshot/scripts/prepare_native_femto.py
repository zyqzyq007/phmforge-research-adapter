"""Recover measured PRONOSTIA observations without the PDMB CSV transformations.

Only official learning runs and observed test prefixes enter the public data.
The complete test continuations and challenge labels stay outside that root.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256


LEARNING=['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
TEST=['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
      'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7','Bearing3_3']
CONDITIONS={1:(1800,4000),2:(1650,4200),3:(1500,5000)}


def feature_vector(raw):
    result=[]
    for a in raw.T:
        mean=float(a.mean());centered=a-mean;std=float(a.std());rms=float(np.sqrt(np.mean(a*a)))
        power=np.abs(np.fft.rfft(a))**2;total=float(power.sum());prob=power/max(total,1e-20)
        result.extend([mean,std,rms,float(np.max(np.abs(a))),float(np.ptp(a)),
            float(np.mean(centered**3)/max(std,1e-12)**3),
            float(np.mean(centered**4)/max(std,1e-12)**4),
            float(np.mean(np.diff(np.signbit(a))!=0)),
            float(-(prob*np.log(np.maximum(prob,1e-30))).sum())])
        result.extend(float(np.log1p(b.mean())) for b in np.array_split(power,8))
    return result


def prepare(run):
    public=run/'public/datasets/FEMTO';public.mkdir(parents=True)
    (public/'trajectories').mkdir();units=LEARNING+TEST
    def one(bearing):
        uid=units.index(bearing)+1;scenario=int(bearing[7]);training=bearing in LEARNING
        archive=run/'clients-cache'/('Training_set.zip' if training else 'Test_set.zip')
        with zipfile.ZipFile(archive) as z:
            paths=sorted(x.filename for x in z.infolist() if re.search('/'+bearing+r'/acc_\d+\.csv$',x.filename))
            assert paths
            features=[];time=[];waves=[];receipts=[]
            for path in paths:
                content=z.read(path);delimiter=',' if b',' in content.splitlines()[0] else ';'
                data=np.loadtxt(io.BytesIO(content),delimiter=delimiter)
                assert data.shape==(2560,6) and np.isfinite(data).all(),path
                timestamp=float(data[0,:4]@np.array([3600,60,1,1e-6]))
                raw=data[:,4:];time.append(timestamp);features.append(feature_vector(raw));waves.append(raw.T.astype(np.float32))
                receipts.append({'member':path,'sha256':hashlib.sha256(content).hexdigest(),
                    'CRC32':f'{z.getinfo(path).CRC:08x}','bytes':len(content)})
        logger_elapsed=np.asarray(time)-time[0]
        logger_elapsed[logger_elapsed<0]+=86400
        # Two released learning files contain nonmonotonic logger clocks.
        # Use the documented 10-second acquisition schedule consistently for
        # every run, retain original logger values, and expose the distinction.
        elapsed=np.arange(len(paths),dtype=np.float64)*10
        base=np.asarray(features)
        # Each rolling summary uses the current and preceding observations only.
        causal=[]
        for i,row in enumerate(base):
            history=base[max(0,i-29):i+1]
            causal.append(np.r_[row,history.mean(axis=0),row-history[0],elapsed[i]/3600])
        n=len(paths);identities=np.asarray([f'unit{uid:03d}:point{i+1:05d}' for i in range(n)])
        values={'features':np.asarray(causal,np.float64),'waveforms':np.asarray(waves),
            'sample_ids':identities,'unit_ids':np.full(n,uid,dtype=np.int64),
            'scenario_ids':np.full(n,scenario,dtype=np.int64),'time_sec':elapsed,
            'logger_elapsed_sec':logger_elapsed,'dataset':np.asarray('FEMTO')}
        if training:values['labels']=elapsed[-1]-elapsed
        np.savez_compressed(public/'trajectories'/f'unit{uid:03d}.npz',**values)
        save(run/'audit'/(bearing+'_members.json'),receipts)
        summary={'unit_id':uid,'bearing':bearing,'scenario_id':scenario,'rpm':CONDITIONS[scenario][0],
            'load_N':CONDITIONS[scenario][1],'split':'learning' if training else 'observed_test',
            'observed_points':n,'observed_duration_sec':float(elapsed[-1]),
            'minimum_interval_sec':float(np.diff(elapsed).min()),'maximum_interval_sec':float(np.diff(elapsed).max()),
            'elapsed_time_definition':'documented nominal 10-second acquisition schedule from ordered file identities',
            'nonmonotonic_logger_steps':int((np.diff(logger_elapsed)<=0).sum()),
            'logger_vs_nominal_max_error_sec':float(np.abs(logger_elapsed-elapsed).max()),
            'raw_observations_finite':True,'future_test_data_exposed':False,
            'native_trajectory_sha256':sha256(public/'trajectories'/f'unit{uid:03d}.npz')}
        return summary
    with ThreadPoolExecutor(max_workers=3) as pool:metadata=list(pool.map(one,units))
    def combine(names,queries=False):
        collected={}
        for bearing in names:
            uid=units.index(bearing)+1;data=dict(np.load(public/'trajectories'/f'unit{uid:03d}.npz',allow_pickle=False))
            n=len(data['sample_ids'])
            indices=(np.unique([int((n-1)*f) for f in [.3,.6,.9]]) if bearing in LEARNING else np.array([n-1])) if queries else np.arange(n)
            for key,value in data.items():
                if value.ndim:collected.setdefault(key,[]).append(value[indices])
        return {key:np.concatenate(value) for key,value in collected.items()}|{'dataset':np.asarray('FEMTO')}
    np.savez_compressed(public/'train.npz',**combine(LEARNING))
    np.savez_compressed(public/'val.npz',**combine(LEARNING[1::2],True))
    np.savez_compressed(public/'test.npz',**combine(TEST,True))
    config={'seed':253,'epochs':30,'batch_size':128,'learning_rate':.001,'cpu_threads':2,
        'fit_unit_ids':[1,3,5],'validation_unit_ids':[2,4,6],
        'validation_prefix_fractions':[.3,.6,.9],'feature_count':103,
        'normalization':'fit units only for selection; all learning units only for final refit',
        'target':'seconds to annotated experimental endpoint; never fraction of an available test prefix',
        'checkpoint_selection':'per-scenario minimum development-query MAE, then refit both learning bearings for that epoch count'}
    save(public/'training_config.json',config)
    info={'dataset':'FEMTO','data_origin':'Original PRONOSTIA laboratory measurements redistributed by NASA',
        'source_url':'https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip',
        'source_sha256':sha256(run/'source/NASA_FEMTO.zip'),'sampling_rate_hz':25600,
        'record_duration_sec':.1,'nominal_acquisition_interval_sec':10,
        'channel_names':['horizontal_acceleration','vertical_acceleration'],'acceleration_unit':'g',
        'feature_description':'Per-axis 9 raw-scale time/spectral statistics and8 log-power bands; 30-point causal means and current-minus-first within that causal history; elapsed hours. No per-window normalization.',
        'feature_count':103,'architectures':['mlp'],'label_unit':'seconds','units':metadata,
        'physical_failure_scope':'Experimental termination targets, not field-maintenance outcomes. A single >20g window does not independently certify zero RUL.',
        'validation_limit':'Only two learning bearings per scenario, one retained for development; windows and prefixes are not independent bearings.',
        'evaluation_scope':'Original challenge annotated RUL targets; final labels and continuation data external. Source-label ambiguities are tracked by external evaluation, not invented or tuned by Agent.'}
    save(public/'dataset.json',info)
    seal_labels(run,metadata)
    print(json.dumps({'learning_bearings':6,'test_bearings':11,'scenarios':3,'native_raw_members':sum(x['observed_points'] for x in metadata),'feature_count':103,'prepared':str(public)}))


def seal_labels(run,metadata):
    units=LEARNING+TEST
    paper=(run/'source/IEEEPHM2012-Challenge-Details.txt').read_text()
    # The first occurrence is the table-of-contents entry, not Table3 itself.
    candidates=[{name:float(value) for name,value in re.findall(r'(Bearing\d_\d)\s+(\d+)\s*s',table)}
                for table in re.findall(r'5\.2 Actual RULs to be estimated(.*?)5\.3 Top',paper,re.S)]
    labels=next(x for x in candidates if set(x)==set(TEST))
    gold=[]
    for bearing in TEST:
        uid=units.index(bearing)+1;meta=metadata[uid-1]
        gold.append({'unit_id':uid,'sample_id':f'unit{uid:03d}:point{meta["observed_points"]:05d}',
            'bearing':bearing,'scenario_id':meta['scenario_id'],'actual_rul':labels[bearing],
            'label_source':'Challenge document Table3,seconds','native_continuation_consistency':bearing!='Bearing1_4'})
    save(run/'hidden/FEMTO/labels.json',gold)
    save(run/'NATIVE_PREPARATION_RESULT.json',{'units':metadata,'learning_bearings':6,'test_bearings':11,
        'scenarios':3,'feature_count':103,'all_raw_members_CRC_checked':True,
        'public_test_labels_absent':True,'official_challenge_labels_preserved':True,
        'ambiguity_unit':'Bearing1_4: challenge339s versus native continuation timestamp2890s; neither silently replaced',
        'scientific_Agent_failure_claims_require_consistent_cases':True,'new_model_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--finish-labels',action='store_true');a=p.parse_args()
    if a.finish_labels:
        info=json.loads((a.run/'public/datasets/FEMTO/dataset.json').read_text())
        assert not (a.run/'hidden/FEMTO/labels.json').exists()
        seal_labels(a.run.resolve(),info['units'])
        print(json.dumps({'reused_complete_trajectory_preparation':True,'labels_preserved_from_primary_table':11}))
    else:prepare(a.run.resolve())
