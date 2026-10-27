"""Frozen measured-waveform forecasting qualification, separate from location scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.signal import hilbert
from evaluate_guided_path_checks import observations, module, save, sha


def read(path):
    return json.loads(Path(path).read_text())


def verify(root):
    cfg = read(root / 'PROTOCOL410.json')
    for path, digest in cfg['bindings'].items():
        assert sha(Path(path)) == digest, path
    for path, digest in cfg['code_freeze'].items():
        assert sha(root / path) == digest, path
    return cfg


def learned_reference(extract, reference):
    q = np.asarray(reference['signals'])
    if q.ndim == 3:
        assert q.shape[0] == 1
        q = q[0]
    assert q.ndim == 2 and np.isfinite(q).all()
    envelope = np.abs(hilbert(q, axis=0))
    j = int(np.argmax(envelope.max(axis=0)))
    lo = int(np.argmax(envelope[:, j])) - extract.TPL_HALF_BACK
    # Reject the helper's synthetic fallback before it can be reached.
    assert np.max(np.abs(q[max(0, lo):min(len(q), lo + extract.TPL_LEN), j])) > 1e-12
    template, a, b, offsets = extract._learn_reference(dict(reference, signals=q))
    return dict(template=template.tolist(), slope=float(a), intercept=float(b),
                offsets={str(k):float(v) for k,v in offsets.items()})


def calibrate(extract, ref, q, xy, pairs):
    template = np.asarray(ref['template'])
    a0, b0 = extract._calibrate_query(q, xy, pairs, template, ref['slope'], ref['intercept'])
    a, b, delays = extract._calibrate(q, xy, pairs, template, a0, b0,
                                    {int(k):v for k,v in ref['offsets'].items()})
    assert np.isfinite([a,b]).all() and a > 0 and np.isfinite(delays).all()
    return dict(slope=float(a), intercept=float(b), delays=np.asarray(delays).tolist())


def design(xy, pairs, panel, template, calibration, n_time, point=None):
    a, b = xy[pairs[:, 0]], xy[pairs[:, 1]]
    delta = b - a
    distance = np.linalg.norm(delta, axis=1)
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    normalized_distance = distance / np.linalg.norm(panel)
    z = np.column_stack([np.ones(len(pairs)), normalized_distance, normalized_distance**2,
        np.cos(2*angle), np.sin(2*angle)] +
        [(pairs[:, k] == j).astype(float) for k in (0,1) for j in range(len(xy)-1)])
    delay = np.asarray(calibration['delays'])
    offset = calibration['intercept'] + delay[pairs[:,0]] + delay[pairs[:,1]]
    quadrature = np.imag(hilbert(template))
    def wave(length, phase):
        arrival = calibration['slope'] * length + offset
        packet = template if phase == 0 else quadrature
        return np.stack([np.interp(np.arange(n_time) - t + 40, np.arange(len(packet)), packet,
                                  left=0., right=0.) for t in arrival])
    columns = [(wave(distance, phase)[:,:,None] * z[:,None,:]).reshape(len(pairs)*n_time, -1)
               for phase in (0,1)]
    # Fixed first-order image-source paths; their amplitudes are shared across paths.
    for axis, boundary in ((0,0.),(0,panel[0]),(1,0.),(1,panel[1])):
        mirror = a.copy()
        mirror[:,axis] = 2*boundary - mirror[:,axis]
        length = np.linalg.norm(mirror - b, axis=1)
        columns += [wave(length, phase).reshape(-1,1) for phase in (0,1)]
    if point is not None:
        incident, outgoing = point - a, b - point
        r1, r2 = np.linalg.norm(incident,axis=1), np.linalg.norm(outgoing,axis=1)
        cosine = np.sum(incident*outgoing,axis=1) / np.maximum(r1*r2, 1e-12)
        cosine = np.clip(cosine, -1., 1.)
        angular = np.column_stack([np.ones(len(pairs)), cosine, 2*cosine*cosine-1])
        angular /= np.sqrt(np.maximum(r1,.01)*np.maximum(r2,.01))[:,None]
        columns += [(wave(r1+r2,phase)[:,:,None] * angular[:,None,:]).reshape(len(pairs)*n_time,-1)
                    for phase in (0,1)]
    return np.column_stack(columns)


def fit(x, y):
    scale = np.sqrt(np.mean(x*x, axis=0))
    scale[scale < 1e-12] = 1.
    standardized = x / scale
    penalty = np.eye(x.shape[1]) * np.sqrt(1e-4)
    beta = np.linalg.lstsq(np.vstack([standardized, penalty]),
                           np.concatenate([y, np.zeros(x.shape[1])]), rcond=None)[0]
    return beta / scale, scale


def worker(root, name):
    cfg = verify(root)
    folder = root / 'workers' / name
    folder.mkdir(exist_ok=False)
    save(folder/'STARTED.json', dict(pid=os.getpid(), unix=time.time()))
    start = time.monotonic()
    source = Path(cfg['observation_archive'])
    cases = observations(source)
    extract = module(source/'source/scattering_extractor.py', 'extract410')
    counter = dict(measured_waveform_regressions_started=0, measured_waveform_regressions_completed=0,
                   numpy_lstsq_calls=0, reference_learning_calls=0, query_calibration_calls=0)
    original = np.linalg.lstsq
    def counted(*args, **kwargs):
        counter['numpy_lstsq_calls'] += 1
        return original(*args, **kwargs)
    np.linalg.lstsq = counted
    try:
        if name == 'reference':
            counter['reference_learning_calls'] += 1
            value = learned_reference(extract, cases[0]['reference'])
            save(folder/'REFERENCE.json', value)
            records = []
        else:
            healthy = name.startswith('healthy_')
            i = int(name.split('_')[-1])
            if healthy:
                ref = cases[0]['reference']
                q = np.asarray(ref['signals'])[0]
                geometry = ref
                saved = read(Path(cfg['check409_archive'])/'HEALTHY_CALIBRATION409.json')
                folds = [f for f in saved['rows'] if f['kind']=='full' and f['fold']==i]
                identity = 'healthy'
            else:
                q, geometry = cases[i]['query'], cases[i]['geometry']
                identity = cases[i]['identity']
                original_case = read(source/'EVIDENCE383.json')['cases'][i]
                family_case = read(Path(cfg['family_archive'])/'EVIDENCE387.json')[i]
                folds = [dict(f, train=f['train_paths'],held=f['held_paths']) for f in original_case['folds']]
            xy, pairs, panel = [np.asarray(geometry[k]) for k in ('sensor_xy_m','path_pairs','panel_size_m')]
            records = []
            for f in folds:
                train, held = np.asarray(f['train']), np.asarray(f['held'])
                assert set(map(tuple,np.sort(pairs[train],axis=1))).isdisjoint(set(map(tuple,np.sort(pairs[held],axis=1))))
                if healthy:
                    counter['reference_learning_calls'] += 1
                    learned = learned_reference(extract, dict(geometry, signals=q[:,train],path_pairs=pairs[train]))
                else:
                    learned = read(root/'workers/reference/REFERENCE.json')
                counter['query_calibration_calls'] += 1
                calibration = calibrate(extract, learned, q[:,train], xy, pairs[train])
                points = {'null':None}
                if not healthy:
                    points.update({n:v['location_m'] for n,v in f['methods'].items()})
                    ff = next(v for v in family_case['folds'] if v['fold']==f['fold'])
                    points.update({f'family_{n[-2:]}':v['location_m'] for n,v in ff['methods'].items() if n!='transmission'})
                predictions, models = {}, {}
                y = q[:,train].T.reshape(-1)
                for program, point in points.items():
                    args = dict(xy=xy,panel=panel,template=np.asarray(learned['template']),calibration=calibration,
                                n_time=len(q),point=None if point is None else np.asarray(point))
                    x = design(pairs=pairs[train], **args)
                    counter['measured_waveform_regressions_started'] += 1
                    save(folder/'COUNTS.json',counter)
                    beta, scale = fit(x,y)
                    counter['measured_waveform_regressions_completed'] += 1
                    forecast = (design(pairs=pairs[held],**args)@beta).reshape(len(held),len(q))
                    predictions[program] = forecast
                    models[program] = dict(point=point,coefficient=beta.tolist(),scale=scale.tolist(),
                                           train_SSE=float(np.sum((x@beta-y)**2)))
                # All models and forecasts are fixed before using held signal values for scoring.
                observed = q[:,held].T
                average = np.broadcast_to(q[:,train].mean(1),observed.shape)
                sse = {p:float(np.sum((pred-observed)**2)) for p,pred in predictions.items()}
                np.savez_compressed(folder/f'fold_{f["fold"]}.npz', observed=observed,average=average,**predictions)
                records.append(dict(identity=identity,fold=f['fold'],train=train.tolist(),held=held.tolist(),
                    learned_reference=learned,calibration=calibration,models=models,SSE=sse,
                    zero_SSE=float(np.sum(observed**2)),average_SSE=float(np.sum((average-observed)**2)),
                    observed_scalars=int(observed.size)))
                save(folder/'EVIDENCE.json',records)
        save(folder/'COUNTS.json',counter)
        save(folder/'TERMINAL.json',dict(exit_code=0,seconds=time.monotonic()-start,counts=counter))
    finally:
        np.linalg.lstsq = original


def run(root):
    cfg = verify(root)
    assert not (root/'HANDLE410.json').exists()
    start = time.monotonic()
    save(root/'HANDLE410.json',dict(pid=os.getpid(),state='running',unix=time.time()))
    steps = []
    def one(name):
        with (root/'logs'/(name+'.log')).open('w') as log:
            try:
                p = subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--run',str(root),'--name',name],
                                   stdout=log,stderr=subprocess.STDOUT,timeout=cfg['budget']['per_worker_seconds'])
                return dict(name=name,exit_code=p.returncode)
            except subprocess.TimeoutExpired:
                return dict(name=name,exit_code=124)
    with ThreadPoolExecutor(max_workers=4) as pool:
        steps += list(pool.map(one,[f'healthy_{i}' for i in range(4)]))
    assert all(x['exit_code']==0 for x in steps), 'Keep failures; no automatic rerun'
    rows = [read(root/'workers'/f'healthy_{i}'/'EVIDENCE.json')[0] for i in range(4)]
    sse = sum(x['SSE']['null'] for x in rows)
    zero, average = [sum(x[k] for x in rows) for k in ('zero_SSE','average_SSE')]
    quality = dict(SSE=sse,zero_SSE=zero,average_SSE=average,relative_zero=sse/zero,
                   relative_average=sse/average,passed=bool(sse<zero and sse<average))
    save(root/'HEALTHY_QUALIFICATION410.json',quality)
    print(json.dumps(dict(phase='healthy',**quality)),flush=True)
    if quality['passed']:
        steps.append(one('reference'))
        assert steps[-1]['exit_code']==0
        with ThreadPoolExecutor(max_workers=4) as pool:
            steps += list(pool.map(one,[f'case_{i}' for i in range(7)]))
        assert all(x['exit_code']==0 for x in steps), 'Keep failures; no automatic rerun'
        choices = {}
        for panel,mapping in cfg['panels'].items():
            selected = []
            for i in range(7):
                evidence = read(root/'workers'/f'case_{i}'/'EVIDENCE.json')
                zero = sum(x['zero_SSE'] for x in evidence)
                scores = {name:sum(x['SSE'][p] for x in evidence)/zero for name,p in mapping.items()}
                pick = next(p for p in mapping if scores[p] <= min(scores.values())+1e-10)
                selected.append(dict(identity=evidence[0]['identity'],scores=scores,selected=pick,
                                     null_relative_SSE=sum(x['SSE']['null'] for x in evidence)/zero))
            choices[panel] = selected
        save(root/'CHOICES410.json',choices)
        save(root/'CHOICE_SEAL410.json',dict(sha256=sha(root/'CHOICES410.json'),location_labels_read=False))
    counts = [read(p) for p in (root/'workers').glob('*/COUNTS.json')]
    totals = {k:sum(x[k] for x in counts) for k in counts[0]}
    save(root/'TERMINAL410.json',dict(exit_code=0,seconds=time.monotonic()-start,healthy_gate_passed=quality['passed'],
                                    counts=totals,steps=steps,new_LLM_calls=0,new_PHM_supervised_fits=0,new_localization_forwards=0))
    print(json.dumps(dict(phase='terminal',seconds=time.monotonic()-start,counts=totals)),flush=True)


def score(root):
    cfg=verify(root)
    assert not (root/'RESULT410.json').exists()
    terminal=read(root/'TERMINAL410.json')
    assert terminal['exit_code']==0
    if not terminal['healthy_gate_passed']:
        save(root/'RESULT410.json',dict(status='healthy_forecasting_qualification_failed',primary_gate_passed=False,
            query_program_choices_evaluated=False,independent_confirmation=False,novelty_established=False,
            majority_goal_achieved=False,actual_cost=terminal))
        return
    assert sha(root/'CHOICES410.json')==read(root/'CHOICE_SEAL410.json')['sha256']
    choices=read(root/'CHOICES410.json')
    panels={}
    for panel,mapping in cfg['panels'].items():
        old={r['identity']:r for r in read(Path(cfg['score_files'][panel]))['rows']}
        rows=[]
        for r in choices[panel]:
            source=old[r['identity']]
            errors=source['errors_mm'] if panel=='within_family' else {p:v['full_error_mm'] for p,v in source['metrics'].items()}
            chosen={'raw_waveform':r['selected'],'keep':'transmission','successful_ordinary':'review_01'}
            rows.append(dict(identity=r['identity'],group=source['group'],selected=chosen,
                errors_mm={k:errors[p] for k,p in chosen.items()},oracle_error_mm=min(errors[p] for p in mapping),
                selected_forecast_beats_null=r['scores'][r['selected']]<r['null_relative_SSE']))
        aggregate={}
        opportunity=sum(r['errors_mm']['keep']-r['oracle_error_mm'] for r in rows)/len(rows)
        for rule in rows[0]['selected']:
            error=sum(r['errors_mm'][rule] for r in rows)/len(rows)
            gain=sum(max(r['errors_mm']['keep']-r['errors_mm'][rule],0) for r in rows)/len(rows)
            aggregate[rule]=dict(mean_error_mm=error,selection_loss_mm=error-sum(r['oracle_error_mm'] for r in rows)/len(rows),
                harm_mm=sum(max(r['errors_mm'][rule]-r['errors_mm']['keep'],0) for r in rows)/len(rows),
                gain_mm=gain,gain_capture=gain/opportunity if opportunity else None,
                group_mean_mm={g:float(np.mean([r['errors_mm'][rule] for r in rows if r['group']==g]))
                               for g in sorted({r['group'] for r in rows})})
        p,o,k=[aggregate[v] for v in ['raw_waveform','successful_ordinary','keep']]
        gates=dict(healthy_forecast_qualified=True,lower_error_than_ordinary=p['mean_error_mm']<o['mean_error_mm']-1e-10,
            lower_error_than_keep=p['mean_error_mm']<k['mean_error_mm']-1e-10,no_extra_harm=p['harm_mm']<=o['harm_mm']+1e-10,
            preserve_gain=p['gain_mm']>=o['gain_mm']-1e-10,
            both_geometry_means_better=all(p['group_mean_mm'][g]<o['group_mean_mm'][g]-1e-10 for g in p['group_mean_mm']))
        panels[panel]=dict(rows=rows,aggregate=aggregate,gates=gates,gate_passed=all(gates.values()))
    save(root/'RESULT410.json',dict(status='qualified_waveforms_program_selection_scored',panels=panels,
        primary_gate_passed=panels['original_full_pool']['gate_passed'],query_program_choices_evaluated=True,
        independent_confirmation=False,novelty_established=False,majority_goal_achieved=False,actual_cost=terminal))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['run','worker','score'])
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--name')
    args=p.parse_args()
    if args.action=='worker':worker(args.run.resolve(),args.name)
    else:globals()[args.action](args.run.resolve())
