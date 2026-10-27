"""A408: passive coordinate changes of measured signals, isolated cold execution.

Keeps original waveforms and path identities. No new faults, fitted PHM model,
or damage labels in workers. Original amplitude rule is the paired control.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from evaluate_guided_path_checks import observations
from evaluate_scope_patch_review import read, save, sha


def verify(root):
    cfg = read(root/'PROTOCOL408.json')
    for path, digest in cfg['bindings'].items():
        assert sha(path) == digest, path
    for path, digest in cfg['code_freeze'].items():
        assert sha(root/path) == digest, path
    return cfg


def change_frame(geometry, axes):
    result = copy.deepcopy(geometry)
    xy = np.asarray(geometry['sensor_xy_m'], float).copy()
    panel = np.asarray(geometry['panel_size_m'], float)
    for axis in axes:
        xy[:, axis] = panel[axis] - xy[:, axis]
    result['sensor_xy_m'] = xy
    return result


def original_frame(point, panel, axes):
    out = list(point)
    for axis in axes:
        out[axis] = float(panel[axis])-out[axis]
    return out


def prepare(root, cfg):
    cases = observations(Path(cfg['observation_archive']))
    qualified = []
    for case in cases:
        for transform, axes in cfg['transforms'].items():
            where = root/'inputs'/str(case['identity'])/transform
            where.mkdir(parents=True)
            geom = change_frame(case['geometry'], axes)
            np.savez_compressed(where/'query.npz', signals=case['query'], **geom)
            with np.load(where/'query.npz', allow_pickle=False) as z:
                assert np.array_equal(z['signals'], case['query'])
                assert np.array_equal(z['path_pairs'], case['geometry']['path_pairs'])
                assert set(z.files) == {'signals','sensor_xy_m','path_pairs','panel_size_m'}
            restored = change_frame(geom, axes)
            assert np.allclose(restored['sensor_xy_m'],case['geometry']['sensor_xy_m'],rtol=0,atol=1e-14)
            a,b = np.asarray(geom['sensor_xy_m']),np.asarray(case['geometry']['sensor_xy_m'])
            assert np.allclose(np.linalg.norm(a[:,None]-a[None,:],axis=2),
                               np.linalg.norm(b[:,None]-b[None,:],axis=2),rtol=0,atol=1e-14)
            qualified.append(dict(identity=case['identity'],group=case['group'],transform=transform,
                path=str(where),query_sha256=sha(where/'query.npz'),panel_size_m=np.asarray(geom['panel_size_m']).tolist(),
                signal_sha256=hashlib.sha256(np.asarray(case['query']).tobytes()).hexdigest()))
    for transform, axes in cfg['transforms'].items():
        where = root/'references'/transform
        where.mkdir(parents=True)
        ref = change_frame(cases[0]['reference'], axes)
        np.savez_compressed(where/'healthy_reference.npz', signals=ref.pop('signals'))
        save(where/'reference_geometry.json',{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in ref.items()})
        with np.load(where/'healthy_reference.npz',allow_pickle=False) as z:
            assert np.array_equal(z['signals'],cases[0]['reference']['signals'])
    save(root/'INPUTS408.json',qualified)
    save(root/'INPUT_AUDIT408.json',dict(passed=True,query_packets=len(qualified),
        original_signal_arrays_unchanged=True,path_pairs_unchanged=True,
        transforms_involutive=True,distances_preserved=True,labels_in_workers=False,
        no_fault_simulation=True))
    return qualified


def run(root):
    cfg = verify(root)
    assert not (root/'HANDLE408.json').exists(), 'No implicit restart'
    save(root/'HANDLE408.json',dict(pid=os.getpid(),status='running',start_unix=time.time()))
    begin = time.monotonic()
    try:
        cases = prepare(root,cfg)
        deadline = begin+cfg['budget']['total_seconds']
        def one(program, packet):
            name, transform = packet['identity'],packet['transform']
            dest = root/'predictions'/name/program/transform
            (dest/'app/.runtime/tmp').mkdir(parents=True)
            (dest/'app/.runtime/mpl').mkdir(parents=True)
            command = ['bwrap','--unshare-all','--die-with-parent','--new-session']
            for path in ['/usr','/lib','/lib64','/etc']:
                if Path(path).exists():command += ['--ro-bind',path,path]
            command += ['--symlink','usr/bin','/bin','--ro-bind',cfg['python_root'],'/opt/python312',
                '--ro-bind',cfg['dependencies'],'/opt/extra','--ro-bind',str(root/'tools'),'/tools',
                '--ro-bind',str(root/'source'/(program+'.py')),'/code/solution.py',
                '--ro-bind',str(root/'references'/transform),'/reference','--ro-bind',packet['path'],'/input',
                '--bind',str(dest/'app'),'/app','--proc','/proc','--dev','/dev','--clearenv']
            env = dict(PATH='/opt/python312/bin:/usr/bin:/bin',HOME='/app',PYTHONPATH='/opt/extra',
                PYTHONDONTWRITEBYTECODE='1',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',
                TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            for key,value in env.items():command += ['--setenv',key,value]
            command += ['--chdir','/app','/opt/python312/bin/python3.12','-B',
                        '/tools/evaluate_scope_patch_review.py','worker']
            save(dest/'COMMAND.json',command)
            start = time.monotonic();code = None;error = None
            try:
                if start>=deadline:raise TimeoutError('Frozen total budget exhausted before dispatch')
                with (dest/'stdout.log').open('w') as out,(dest/'stderr.log').open('w') as err:
                    p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
                    save(dest/'PROCESS.json',dict(pid=p.pid,started_unix=time.time()))
                    try:code=p.wait(timeout=min(cfg['budget']['per_prediction_seconds'],deadline-start))
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid,signal.SIGKILL);code=p.wait();error='prediction_timeout'
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            pred=read(dest/'app/PREDICTION.json') if (dest/'app/PREDICTION.json').exists() else None
            if pred is not None:
                assert pred['source_sha256']==sha(root/'source'/(program+'.py'))
                assert pred['query_sha256']==packet['query_sha256']
            row=dict(program=program,identity=name,group=packet['group'],transform=transform,
                exit_code=code,error=error,prediction=pred,seconds=time.monotonic()-start)
            save(dest/'RESULT.json',row)
            return row
        rows=[]
        with ThreadPoolExecutor(max_workers=cfg['budget']['parallel']) as pool:
            jobs=[pool.submit(one,program,packet) for program in cfg['programs'] for packet in cases]
            for future in as_completed(jobs):
                rows.append(future.result());save(root/'PREDICTIONS408.json',rows)
                save(root/'PROGRESS408.json',dict(done=len(rows),planned=len(jobs),
                    failed=sum(r['exit_code']!=0 or r['prediction'] is None for r in rows)))
                if len(rows)%28==0:print(json.dumps(dict(done=len(rows),planned=len(jobs))),flush=True)
        complete=len(rows)==cfg['budget']['maximum_predictions'] and all(r['exit_code']==0 and r['prediction'] is not None for r in rows)
        save(root/'PREDICTIONS_SEAL408.json',dict(sha256=sha(root/'PREDICTIONS408.json'),complete=complete))
        save(root/'TERMINAL408.json',dict(complete=complete,actual_prediction_processes=len(list((root/'predictions').rglob('PROCESS.json'))),
            seconds=time.monotonic()-begin,new_LLM_calls=0,new_supervised_fits=0,
            numerical_lstsq=sum(r['prediction']['numerical_calls']['numpy_lstsq'] for r in rows if r['prediction'])))
        assert complete, 'Retain incomplete predictions; no automatic recovery'
        save(root/'HANDLE408.json',dict(status='terminal',exit_code=0))
    except BaseException as exc:
        save(root/'HANDLE408.json',dict(status='terminal_error',error=str(exc)))
        raise


def choose(root):
    cfg=verify(root)
    assert not (root/'CHOICES408.json').exists()
    assert read(root/'TERMINAL408.json')['complete']
    assert sha(root/'PREDICTIONS408.json')==read(root/'PREDICTIONS_SEAL408.json')['sha256']
    outputs={(r['identity'],r['program'],r['transform']):r['prediction']['location_m'] for r in read(root/'PREDICTIONS408.json')}
    panels={}
    for panel,spec in cfg['panels'].items():
        evidence=read(spec['evidence']);evidence=evidence['cases'] if isinstance(evidence,dict) else evidence
        rows=[]
        for case in evidence:
            metadata=next(r for r in read(root/'INPUTS408.json') if r['identity']==case['identity'])
            size=metadata['panel_size_m'];tolerance=cfg['tolerance_grid_steps']*math.hypot(*size)/180+1e-8
            scores={}
            for local,program in spec['program_mapping'].items():
                identity=outputs[case['identity'],program,'identity']
                assert math.dist(identity,case['full_predictions'][local])<1e-7, 'Unchanged-input replay differs'
                deviations={t:math.dist(identity,original_frame(outputs[case['identity'],program,t],size,axes))
                            for t,axes in cfg['transforms'].items() if t!='identity'}
                amplitude=sum(f['methods'][local][spec['amplitude_key']] for f in case['folds'])/4
                scores[local]=dict(amplitude=amplitude,maximum_deviation_m=max(deviations.values()),
                    deviations_m=deviations,eligible=max(deviations.values())<=tolerance,identity_prediction_m=identity)
            order=list(spec['program_mapping'])
            eligible=[p for p in order if scores[p]['eligible']]
            low=min(scores[p]['amplitude'] for p in eligible) if eligible else None
            selected=next((p for p in eligible if scores[p]['amplitude']<=low+1e-10),'transmission')
            low_all=min(scores[p]['amplitude'] for p in order)
            amplitude=next(p for p in order if scores[p]['amplitude']<=low_all+1e-10)
            rows.append(dict(identity=case['identity'],group=case['group'],tolerance_m=tolerance,scores=scores,
                selected=selected,amplitude_selected=amplitude,fallback_no_eligible=not eligible))
        panels[panel]=rows
    save(root/'CHOICES408.json',panels)
    save(root/'CHOICE_SEAL408.json',dict(sha256=sha(root/'CHOICES408.json'),damage_labels_read=False))


def score(root):
    cfg=verify(root)
    assert not (root/'RESULT408.json').exists()
    assert sha(root/'CHOICES408.json')==read(root/'CHOICE_SEAL408.json')['sha256']
    choices=read(root/'CHOICES408.json');panels={}
    for panel,spec in cfg['panels'].items():
        old=read(spec['score']);old={r['identity']:r for r in old['rows']}
        rows=[]
        for row in choices[panel]:
            source=old[row['identity']]
            errors=source['errors_mm'] if panel=='within_family' else {p:v['full_error_mm'] for p,v in source['metrics'].items()}
            selected=dict(method=row['selected'],amplitude=row['amplitude_selected'],keep='transmission',successful_ordinary='review_01')
            rows.append(dict(identity=row['identity'],group=row['group'],selected=selected,
                errors_mm={k:errors[p] for k,p in selected.items()},oracle_error_mm=min(errors[p] for p in spec['program_mapping'])))
        aggregate={}
        for rule in rows[0]['selected']:
            gain=sum(max(r['errors_mm']['keep']-r['errors_mm'][rule],0) for r in rows)
            opportunity=sum(r['errors_mm']['keep']-r['oracle_error_mm'] for r in rows)
            aggregate[rule]=dict(mean_error_mm=sum(r['errors_mm'][rule] for r in rows)/7,
                selection_loss_mm=sum(r['errors_mm'][rule]-r['oracle_error_mm'] for r in rows)/7,
                harm_mm=sum(max(r['errors_mm'][rule]-r['errors_mm']['keep'],0) for r in rows)/7,
                gain_mm=gain/7,gain_capture=gain/opportunity if opportunity else None,
                group_mean_mm={g:sum(r['errors_mm'][rule] for r in rows if r['group']==g)/sum(r['group']==g for r in rows)
                               for g in sorted({r['group'] for r in rows})})
        p,o,a=[aggregate[k] for k in ['method','successful_ordinary','amplitude']]
        gates=dict(baseline_all_eligible=all(r['scores']['transmission']['eligible'] for r in choices[panel]),
            lower_error_than_successful_ordinary=p['mean_error_mm']<o['mean_error_mm']-1e-10,
            lower_error_than_amplitude=p['mean_error_mm']<a['mean_error_mm']-1e-10,
            no_extra_harm=p['harm_mm']<=o['harm_mm']+1e-10,preserve_gain=p['gain_mm']>=o['gain_mm']-1e-10,
            both_geometry_means_better=all(p['group_mean_mm'][g]<o['group_mean_mm'][g]-1e-10 for g in p['group_mean_mm']))
        panels[panel]=dict(rows=rows,aggregate=aggregate,gates=gates,gate_passed=all(gates.values()))
    save(root/'RESULT408.json',dict(panels=panels,primary_gate_passed=panels['original_full_pool']['gate_passed'],
        independent_confirmation=False,novelty_established=False,majority_goal_achieved=False,
        actual_cost=read(root/'TERMINAL408.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','choose','score'])
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    globals()[args.action](args.run.resolve())
