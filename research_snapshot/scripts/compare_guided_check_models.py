"""Frozen 2x2 check-model/aggregation diagnostic using real measured amplitudes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import numpy as np
from evaluate_guided_path_checks import observations, design, kernel, ridge, module
from evaluate_scope_patch_review import read, save, sha


def verify(root):
    cfg=read(root/'PROTOCOL409.json')
    for path,digest in cfg['bindings'].items():assert sha(path)==digest,path
    for path,digest in cfg['code_freeze'].items():assert sha(root/path)==digest,path
    return cfg


def tracked_fit(root,label):
    counts={'started':0,'completed':0}
    path=root/'fit_counts'/(label+'.json')
    def fit(x,y):
        counts['started']+=1;save(path,counts)
        result=ridge(x,y)
        counts['completed']+=1;save(path,counts)
        return result
    return fit


def matrices(xy,pairs,kind,train):
    raw=design(xy,pairs)
    if kind=='reduced':raw=raw[:,list(range(7))+list(range(12,raw.shape[1]))]
    center,scale=raw[train].mean(0),raw[train].std(0)
    center[0],scale[0]=0.,1.;scale[scale<1e-12]=1.
    return (raw-center)/scale


def healthy(root,cfg,feature):
    fit=tracked_fit(root,'healthy')
    case=observations(Path(cfg['observation_archive']))[0]['reference']
    xy,pairs=np.asarray(case['sensor_xy_m']),np.asarray(case['path_pairs'])
    signals=np.asarray(case['signals'])
    if signals.ndim==3:
        assert signals.shape[0]==1,'Only the source-qualified singleton batch is supported'
        signals=signals[0]
    assert signals.ndim==2 and signals.shape[1]==len(pairs)
    y=feature(signals)[1]
    keys=sorted({tuple(sorted(p)) for p in pairs.tolist()})
    order=np.random.default_rng(383).permutation(len(keys))
    assign={keys[int(j)]:i%4 for i,j in enumerate(order)}
    folds=np.array([assign[tuple(sorted(p))] for p in pairs.tolist()])
    rows=[]
    for kind in ['full','reduced']:
        for fold in range(4):
            train,held=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            assert set(tuple(sorted(p)) for p in pairs[train]).isdisjoint(set(tuple(sorted(p)) for p in pairs[held]))
            x=matrices(xy,pairs,kind,train);beta=fit(x[train],y[train])
            rows.append(dict(kind=kind,fold=fold,train=train.tolist(),held=held.tolist(),coefficient=beta.tolist(),
                observed=y[held].tolist(),forecast=(x[held]@beta).tolist(),train_mean=float(y[train].mean()),
                train_MSE=float(np.mean((x[train]@beta-y[train])**2))))
    mse={kind:sum(sum((a-b)**2 for a,b in zip(r['observed'],r['forecast'])) for r in rows if r['kind']==kind)/len(y)
         for kind in ['full','reduced']}
    null=sum(sum((a-r['train_mean'])**2 for a in r['observed']) for r in rows if r['kind']=='full')/len(y)
    chosen=min(['full','reduced'],key=lambda k:mse[k])
    result=dict(rows=rows,held_MSE=mse,constant_mean_MSE=null,selected_kind=chosen,
        selected_beats_constant=mse[chosen]<null,measurement_regressions=8,healthy_paths=len(pairs),
        chosen_without_damaged_query_labels=True)
    save(root/'HEALTHY_CALIBRATION409.json',result)
    return result


def run(root):
    cfg=verify(root);assert not (root/'EVIDENCE409.json').exists()
    start=time.monotonic();parent=Path(cfg['observation_archive'])
    feature=module(parent/'source/transmission.py','measurement409')._path_features
    calibration=healthy(root,cfg,feature)
    cases=observations(parent);old=read(parent/'EVIDENCE383.json')['cases'];family=read(Path(cfg['family_archive'])/'EVIDENCE387.json')
    def one(index):
        fit=tracked_fit(root,'case_'+str(index))
        case=cases[index];g=case['geometry'];xy,pairs=np.asarray(g['sensor_xy_m']),np.asarray(g['path_pairs'])
        y=feature(case['query'])[1];records=[];fits=0
        for kind in ['full','reduced']:
            for f in old[index]['folds']:
                train,held=np.array(f['train_paths']),np.array(f['held_paths'])
                x=matrices(xy,pairs,kind,train);base=fit(x[train],y[train]);fits+=1
                variance=max(float(np.mean((x[train]@base-y[train])**2)),1e-12)
                details={}
                points={n:v['location_m'] for n,v in f['methods'].items()}
                ff=next(v for v in family[index]['folds'] if v['fold']==f['fold'])
                points.update(family_01=ff['methods']['review_01']['location_m'],family_02=ff['methods']['review_02']['location_m'])
                for name,point in points.items():
                    forecasts=[];coefficients=[]
                    for width in f['widths']:
                        aug=np.column_stack([x,kernel(np.array(point),xy,pairs,width)])
                        beta=fit(aug[train],y[train]);fits+=1
                        raw=beta.copy()
                        if beta[-1]>0:beta=np.r_[base,0.]
                        forecasts.append(aug[held]@beta)
                        coefficients.append(dict(raw=raw.tolist(),used=beta.tolist(),width=width))
                    prediction=np.mean(forecasts,axis=0)
                    mse=float(np.mean((prediction-y[held])**2))
                    details[name]=dict(location_m=point,forecast=prediction.tolist(),coefficients=coefficients,
                                       MSE=mse,normalized_MSE=mse/variance)
                    if kind=='full':
                        historical=f['methods'][name] if name in f['methods'] else ff['methods']['review_'+name[-2:]]
                        expected=historical['energy_forecasts'] if name in f['methods'] else historical['forecasts']
                        assert np.allclose(prediction,expected,rtol=0,atol=1e-8),'Original checker replay differs'
                records.append(dict(kind=kind,fold=f['fold'],train=train.tolist(),held=held.tolist(),
                    observed=y[held].tolist(),base_forecast=(x[held]@base).tolist(),base_coefficient=base.tolist(),
                    variance=variance,train_mean=float(y[train].mean()),methods=details))
        out=dict(identity=case['identity'],group=case['group'],folds=records,measurement_regressions=fits)
        save(root/'cases'/(str(index)+'.json'),out)
        return out
    with ThreadPoolExecutor(max_workers=cfg['parallel']) as pool:evidence=list(pool.map(one,range(7)))
    save(root/'EVIDENCE409.json',evidence)
    choices={}
    for panel,spec in cfg['panels'].items():
        rows=[]
        for case in evidence:
            scores={}
            for kind in ['full','reduced']:
                folds=[f for f in case['folds'] if f['kind']==kind]
                scores[kind+'_normalized']={n:sum(f['methods'][p]['normalized_MSE'] for f in folds)/4 for n,p in spec['program_mapping'].items()}
                scores[kind+'_pooled']={n:sum(f['methods'][p]['MSE']*len(f['held']) for f in folds)/sum(len(f['held']) for f in folds)
                                       for n,p in spec['program_mapping'].items()}
            selected={rule:next(p for p in spec['program_mapping'] if values[p]<=min(values.values())+1e-10)
                      for rule,values in scores.items()}
            selected['primary']=selected[calibration['selected_kind']+'_pooled']
            rows.append(dict(identity=case['identity'],group=case['group'],scores=scores,selected=selected))
        choices[panel]=rows
    save(root/'CHOICES409.json',choices)
    save(root/'CHOICE_SEAL409.json',dict(sha256=sha(root/'CHOICES409.json'),healthy_sha256=sha(root/'HEALTHY_CALIBRATION409.json'),damage_labels_read=False))
    save(root/'TERMINAL409.json',dict(complete=True,seconds=time.monotonic()-start,
        measurement_regressions=8+sum(r['measurement_regressions'] for r in evidence),new_LLM_calls=0,
        new_supervised_PHM_fits=0,new_localization_forwards=0,original_check_forecasts_reproduced=True))


def score(root):
    cfg=verify(root);assert not (root/'RESULT409.json').exists()
    assert sha(root/'CHOICES409.json')==read(root/'CHOICE_SEAL409.json')['sha256']
    choices=read(root/'CHOICES409.json');calibration=read(root/'HEALTHY_CALIBRATION409.json');panels={}
    for panel,spec in cfg['panels'].items():
        previous={r['identity']:r for r in read(spec['score'])['rows']};rows=[]
        for row in choices[panel]:
            old=previous[row['identity']]
            error=old['errors_mm'] if panel=='within_family' else {p:r['full_error_mm'] for p,r in old['metrics'].items()}
            selected=dict(row['selected'],keep='transmission',successful_ordinary='review_01')
            rows.append(dict(identity=row['identity'],group=row['group'],selected=selected,
                errors_mm={k:error[p] for k,p in selected.items()},oracle_error_mm=min(error[p] for p in spec['program_mapping'])))
        aggregate={}
        for rule in rows[0]['selected']:
            gain=sum(max(r['errors_mm']['keep']-r['errors_mm'][rule],0) for r in rows)/7
            opportunity=sum(r['errors_mm']['keep']-r['oracle_error_mm'] for r in rows)/7
            aggregate[rule]=dict(mean_error_mm=sum(r['errors_mm'][rule] for r in rows)/7,
                selection_loss_mm=sum(r['errors_mm'][rule]-r['oracle_error_mm'] for r in rows)/7,
                harm_mm=sum(max(r['errors_mm'][rule]-r['errors_mm']['keep'],0) for r in rows)/7,
                gain_mm=gain,gain_capture=gain/opportunity if opportunity else None,
                group_mean_mm={g:sum(r['errors_mm'][rule] for r in rows if r['group']==g)/sum(r['group']==g for r in rows)
                               for g in sorted({r['group'] for r in rows})})
        p,o,a=[aggregate[k] for k in ['primary','successful_ordinary','full_normalized']]
        gates=dict(healthy_predictive_skill=calibration['selected_beats_constant'],lower_error_than_ordinary=p['mean_error_mm']<o['mean_error_mm']-1e-10,
            lower_error_than_original_checker=p['mean_error_mm']<a['mean_error_mm']-1e-10,
            no_extra_harm=p['harm_mm']<=o['harm_mm']+1e-10,preserve_gain=p['gain_mm']>=o['gain_mm']-1e-10,
            both_geometry_means_better=all(p['group_mean_mm'][g]<o['group_mean_mm'][g]-1e-10 for g in p['group_mean_mm']))
        panels[panel]=dict(rows=rows,aggregate=aggregate,gates=gates,gate_passed=all(gates.values()),
            increment_over_plain_pooled=p['mean_error_mm']<aggregate['full_pooled']['mean_error_mm']-1e-10)
    save(root/'RESULT409.json',dict(panels=panels,primary_gate_passed=panels['original_full_pool']['gate_passed'],
        independent_confirmation=False,novelty_established=False,majority_goal_achieved=False,
        actual_cost=read(root/'TERMINAL409.json'),healthy_selected_kind=calibration['selected_kind']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','score'])
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();globals()[args.action](args.run.resolve())
