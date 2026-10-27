"""One bounded real-data capability control, not an Agent comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from pathlib import Path
import shutil
import time
import traceback
import zlib

import numpy as np

from audit_program_evidence import read, save, sha, utc
from audit_measured_timebase import events_loop, events_vector
from check_harmonic_contract import verify_archive, train_one, evaluate_one
from program_revision_runtime import load

DATA = Path('/data/yyf-zyq/PHM_20261010/20261025T012000Z_source346r1')
OLD = Path('/data/yyf-zyq/PHM_20261010/20261026T011551Z_contract402')
VARIANTS = ['pulse_rate', 'pulse_phase']
ROLES = ['train', 'development', 'independent_check', 'final']


def prepare(root):
    root.mkdir(exist_ok=False)
    for d in ['code', 'common_tools', 'inputs', 'runtime/tmp', 'runtime/mpl']:
        (root / d).mkdir(parents=True)
    project_scripts = Path(__file__).parent
    for name in ['check_pulse_phase.py', 'audit_program_evidence.py',
                 'audit_measured_timebase.py', 'check_harmonic_contract.py',
                 'program_revision_runtime.py', 'controlled_feedback.py']:
        shutil.copy2(project_scripts / name, root / 'code' / name)
    shutil.copy2(project_scripts / 'program_training_entry.py', root / 'common_tools/program_training_entry.py')
    bindings = ['PROTOCOL346.json', 'qualification/IDENTITIES346.json']
    bindings += ['qualification/' + r + '.npz' for r in ROLES]
    verify_archive(DATA, bindings)
    verify_archive(OLD, ['RESULT402.json', 'programs/pair/source/program.py',
                         'programs/original/source/program.py'])
    original = (OLD / 'programs/pair/source/program.py').read_text()
    for name in VARIANTS:
        source = root / 'programs' / name / 'source'
        source.mkdir(parents=True)
        code = original
        if name == 'pulse_rate':
            before = 'rotor = np.asarray(data["rotation_hz"], dtype=float)'
            assert code.count(before) == 1
            code = code.replace(before, 'rotor = np.asarray(data["pulse_hz"], dtype=float)')
        else:
            replacements = {
                'def _one_acquisition(waves, stats, rotor_hz, sample_rate):':
                'def _one_acquisition(waves, stats, rotor_hz, sample_rate, pulse_phase):',
                'phase = 2.0 * np.pi * rotor_hz[row] * np.arange(n_points) / rate':
                'phase = pulse_phase[row] - pulse_phase[row, 0]',
                '_one_acquisition(waves[index], stats[index], rotor[index], rate[index])':
                '_one_acquisition(waves[index], stats[index], rotor[index], rate[index], np.asarray(data["pulse_phase"])[index])',
            }
            for before, after in replacements.items():
                assert code.count(before) == 1, before
                code = code.replace(before, after)
        (source / 'program.py').write_text(code)
        compile(code, str(source / 'program.py'), 'exec')
    cfg = dict(iteration='A439', utc=utc(), source_archive=str(DATA), baseline_archive=str(OLD),
        hypothesis='Measured pulse phase may improve an existing narrow-band complete program whose coordinates use nominal frequency; this is added sensor information, not evidence of an original Agent reasoning error.',
        variants=VARIANTS, primary='pulse_phase', secondary='pulse_rate',
        cached_controls=['original', 'pair'], factors='Only phase coordinate changes relative to the already corrected pair program; seed2701/PCA18/LDA/shrinkage/statistics/geometry/data unchanged.',
        primary_gate='pulse_phase record error strictly below both the original and pair cached controls; paired harm versus original must be zero and gain positive.',
        selection='Minimize mean training-condition CV error; tie lower development error, then fewer coordinate edits, then fixed order original/pair/pulse_rate/pulse_phase. Additional check reported separately.',
        scope='28 already opened real acquisitions, one rig; 8train/8development/4additional/8final. No independent confirmation and no native Agent advantage. Added tachometer must be shared in any future Agent comparison.',
        phase_contract='Pulse count, not verified shaft angle: PPR unknown. Linear phase interpolation between Schmitt events; constant first/last-interval extrapolation only outside endpoints. Each window origin is subtracted. No waveform changes or synthetic labels.',
        event_quantiles=[.01,.99], schmitt_levels=[.2,.8],
        budgets=dict(classifier_fits=6, PCA_fits=6, scaler_fits=6, new_LLM_calls=0,
            cold_prediction_processes=88, workers=2, prepare_workers=4, fit_seconds=120,
            predict_seconds=45, program_seconds=900, disk_bytes=3000000000, memory_bytes=10000000000, GPU_hours=0),
        no_score_driven_recovery=True, held_confirmation_unopened=True,
        bindings={str(p.relative_to(DATA)):sha(p) for p in [DATA/x for x in bindings]},
        baseline_result_sha256=sha(OLD/'RESULT402.json'),
        literature=[dict(url='https://zenodo.org/records/8208454', role='Author identifies tachometer column.'),
            dict(url='https://www.sciencedirect.com/science/article/pii/S0888327018300384',role='Existing phase-time/order-tracking method; no novelty claim.')])
    save(root/'PROTOCOL439.json',cfg)
    save(root/'ENGINE_CONFIG.json',dict(input_root=str(root/'inputs'),common_tools=str(root/'common_tools'),
        fit_seconds=120,predict_seconds=45,program_seconds=900))
    files=[p for d in ['code','common_tools','programs'] for p in (root/d).rglob('*.py')]
    files += [root/'PROTOCOL439.json', root/'ENGINE_CONFIG.json']
    save(root/'FREEZE439.json',dict(utc=utc(),files={str(p.relative_to(root)):sha(p) for p in files}))
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache', shared_dependencies=dict(
        path='/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4', usage='Read-only runtime; source archive SHA-bound.')))
    print(dict(prepared=str(root),fits=6,variants=VARIANTS),flush=True)


def pulse_record(row):
    path=DATA/'records'/row['unit']/'record.csv'
    raw=path.read_bytes()
    assert sha(path)==row['raw_sha256'] and f'{zlib.crc32(raw):08x}'==row['CRC32']
    values=np.loadtxt(io.BytesIO(raw),delimiter=row['delimiter'],usecols=(0,1))
    times,signal=values.T
    assert np.isfinite(values).all() and np.all(np.diff(times)>0)
    lo,hi=np.quantile(signal,[.01,.99]);assert hi>lo
    low,high=lo+.2*(hi-lo),lo+.8*(hi-lo)
    index=events_vector(signal,low,high)
    assert np.array_equal(index,events_loop(signal,low,high)) and len(index)>=3
    t=times[index];cycle=np.arange(len(index),dtype=float)
    phase=np.interp(times,t,cycle)
    left=times<t[0];right=times>t[-1]
    phase[left]=(times[left]-t[0])/(t[1]-t[0])
    phase[right]=cycle[-1]+(times[right]-t[-1])/(t[-1]-t[-2])
    assert np.all(np.diff(phase)>0) and np.allclose(phase[index],cycle,rtol=0,atol=1e-10)
    count=row['window_count'];points=25600
    assert len(row['sample_ids'])==count and count*points<=len(phase)
    pulse_hz=(len(t)-1)/(t[-1]-t[0])
    report=dict(unit=row['unit'],role=row['role'],group=row['group'],source_sha256=row['raw_sha256'],
        pulse_count=len(index),pulse_hz=float(pulse_hz),phase_monotone=True,
        max_event_residual=float(np.max(np.abs(phase[index]-cycle))),
        extrapolated_points=int(left.sum()+right.sum()),PPR_verified=False)
    return row['group'],phase[:count*points].reshape(count,points)*(2*np.pi),pulse_hz,report


def run(root):
    cfg=read(root/'PROTOCOL439.json')
    for name,digest in read(root/'FREEZE439.json')['files'].items():assert sha(root/name)==digest,name
    assert not (root/'STARTED439.json').exists()
    save(root/'STARTED439.json',dict(utc=utc(),status='running'))
    start=time.monotonic()
    identities=read(DATA/'qualification/IDENTITIES346.json')
    with ThreadPoolExecutor(max_workers=4) as executor:rows=list(executor.map(pulse_record,identities))
    lookup={r[0]:r for r in rows};assert len(lookup)==28
    audit=[]
    for role in ROLES:
        data=load(DATA/'qualification'/(role+'.npz'))
        phase=np.empty(data['waveforms'].shape[:2]);rate=np.empty(len(phase))
        for group in sorted(set(data['groups'])):
            mask=data['groups']==group;row=next(r for r in identities if r['group']==group)
            assert data['sample_ids'][mask].tolist()==row['sample_ids']
            _,phase[mask],rate[mask],report=lookup[group];audit.append(report)
        data.update(pulse_phase=phase,pulse_hz=rate)
        np.savez_compressed(root/'inputs'/(role+'.npz'),**data)
    save(root/'INPUT_AUDIT439.json',dict(utc=utc(),records=audit,original_arrays_unchanged=True,
        original_input_excluded_tachometer=True,new_sensor_information=True,
        preparation_read_historically_opened_final_labels=True,no_predictions_or_model_selection_during_preparation=True))
    trained={}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(train_one,root,name) for name in VARIANTS]):
            name,value=future.result();trained[name]=value;save(root/'TRAINING_PROGRESS439.json',trained)
    if not all(r['status']=='complete' for r in trained.values()):
        raise RuntimeError('Incomplete matrix; preserve failures without scientific restart')
    old=read(OLD/'RESULT402.json')
    legal={name:old['legal'][name] for name in ['original','pair']}
    legal.update({name:dict(mean_cv=sum(x['record_error'] for x in v['report']['cv'])/2,
        development=v['report']['development']['record_error'],check=v['check']['record_error']) for name,v in trained.items()})
    order=['original','pair']+VARIANTS
    choice=min(order,key=lambda n:(legal[n]['mean_cv'],legal[n]['development'],order.index(n)))
    save(root/'SEALED_CHOICE439.json',dict(utc=utc(),legal=legal,choice=choice,
        cached_final_historically_known=True,new_final_predictions_generated=False))
    final={name:old['final'][name] for name in ['original','pair']}
    data=load(root/'inputs/final.npz')
    with ThreadPoolExecutor(max_workers=2) as executor:
        final.update(dict(executor.map(lambda n:evaluate_one(root,n,data),VARIANTS)))
    base={r['group']:r for r in final['original']['records']}
    for name in VARIANTS:
        records=final[name]['records']
        final[name]['paired_harm']=sum(base[r['group']]['predicted']==r['actual'] and r['predicted']!=r['actual'] for r in records)/len(records)
        final[name]['paired_gain']=sum(base[r['group']]['predicted']!=r['actual'] and r['predicted']==r['actual'] for r in records)/len(records)
    main=final['pulse_phase']
    gate=(main['record_error']<min(final[n]['record_error'] for n in ['original','pair'])
          and main['paired_harm']==0 and main['paired_gain']>0)
    audits=[read(p) for p in root.glob('programs/*/**/training_audit.json')]
    cost=dict(new_PHM_fit_entries=sum(x['mode']=='fit' for x in audits),
        completed_classifier_fits=sum(x['completed_estimator_fits'] for x in audits),
        cold_predictions=sum(x['mode']=='predict' for x in audits),LLM_calls=0,native_Agent_stages=0)
    result=dict(utc=utc(),status='completed',seconds=time.monotonic()-start,legal=legal,final=final,
        choice=choice,primary_gate_passed=gate,cost=cost,independent_confirmation=False,majority_goal_complete=False,
        decision='Admit only to a new same-information Agent pilot.' if gate else 'Stop this phase-coordinate recipe; no tuning on these opened records.')
    save(root/'RESULT439.json',result)
    save(root/'TERMINAL439.json',dict(utc=utc(),status='completed',cost=cost))
    print(dict(final={k:v['record_accuracy'] for k,v in final.items()},choice=choice,gate=gate,cost=cost),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve()
    if args.action=='prepare':prepare(root)
    else:
        try:run(root)
        except Exception:
            save(root/'TERMINAL439.json',dict(utc=utc(),status='failed',traceback=traceback.format_exc()))
            raise
