"""Replay a completed submission on target-free development observations only.

Remove all train/validation arrays from the replay data mount. This checks the
delivered program, saved artifacts and reported development result without
opening any final-test score or mutating the completed subject workspace.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import namespace, files
from run_full_benchmark_campaign import clean_env


def main(run, ident):
    job=next(j for j in json.loads((run/'PLAN187.json').read_text())['jobs'] if j['id']==ident)
    status=json.loads((run/'primary'/ident/ident/'status.json').read_text())
    assert status['status']=='finished' and not Path('/proc',str(status['cli_pid'])).exists()
    source=run/'public_data'/ident/'datasets'/job['dataset']
    dev=dict(np.load(source/'val.npz',allow_pickle=False))
    target=run/'audit/development_replay'/ident;target.mkdir(parents=True,exist_ok=False)
    output=target/'output';output.mkdir()
    observed={k:v for k,v in dev.items() if k not in ['labels','groups','origin_labels']}
    np.savez_compressed(output/'observations.npz',**observed)
    before=files(run/job['workspace']);app=target/'app';shutil.copytree(run/job['workspace'],app,symlinks=True)
    data=target/'data/datasets'/job['dataset'];data.mkdir(parents=True)
    shutil.copy2(source/'dataset.json',data/'dataset.json')
    mounts=json.loads(Path(job['mounts']).read_text())
    for m in mounts['mounts']:
        if m['destination']=='/data':m['source']=str(target/'data')
    mount_file=target/'mounts.json';save(mount_file,mounts)
    inner=['/opt/python312/bin/python3.12','-B','/app/solution.py','--input','/output/observations.npz','--output','/output/predictions.json']
    cold=dict(job,evaluation_mounts=str(mount_file));command=namespace(run,cold,app,output,inner);save(target/'command.json',command)
    env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',
        PYTHONDONTWRITEBYTECODE='1',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
        MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    try:
        with (target/'stdout.txt').open('wb') as out,(target/'stderr.txt').open('wb') as err:
            code=subprocess.run(command,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,timeout=180).returncode
        assert code==0,'Target-free development replay failed'
        rows=json.loads((output/'predictions.json').read_text())['predictions'];indexed={v['sample_id']:v for v in rows}
        assert len(rows)==len(indexed)==len(dev['sample_ids']) and set(indexed)==set(dev['sample_ids'].tolist())
        if job.get('data_format')=='cmapss_npz':
            p=np.asarray([indexed[str(s)]['predicted_rul'] for s in dev['sample_ids']]);residual=p-dev['labels']
            assert np.isfinite(p).all() and np.min(p)>=0
            metrics=dict(n=len(rows),mae_cycles=float(np.abs(residual).mean()),rmse_cycles=float(np.sqrt(np.square(residual).mean())),bias_cycles=float(residual.mean()))
        else:
            p=[indexed[str(s)]['predicted_label'] for s in dev['sample_ids']]
            metrics=dict(n=len(rows),accuracy=float(np.mean(np.asarray(p)==dev['labels'])))
        result=dict(passed=True,exit_code=code,metrics=metrics)
    except (AssertionError,ValueError,KeyError,TypeError,OSError,subprocess.TimeoutExpired) as e:
        result=dict(passed=False,error_type=type(e).__name__,error=str(e))
    assert before==files(run/job['workspace']),'Root replay changed original subject files'
    result.update(job=ident,task_id=job['task_id'],final_scores_not_read=True,
        input_targets_absent=True,all_training_and_development_arrays_unmounted=True,
        original_submission_unchanged=True,observation_sha256=sha256(output/'observations.npz'))
    save(target/'RESULT.json',result);print(json.dumps(result))


def pointwise(run, ident, protocol_file):
    """Outcome-blind, fixed development subset in the declared singleton context.

    The ordinary full-development replay remains the original validation score.
    This diagnostic never changes a submission or supplies feedback to an Agent.
    """
    protocol_file=Path(protocol_file)
    protocol=json.loads(protocol_file.read_text())
    job=next(j for j in json.loads((run/'PLAN187.json').read_text())['jobs'] if j['id']==ident)
    status=json.loads((run/'primary'/ident/ident/'status.json').read_text())
    assert status['status']=='finished' and not Path('/proc',str(status['cli_pid'])).exists()
    assert protocol['final_scores_not_opened_at_registration'] is True
    source=run/'public_data'/ident/'datasets'/job['dataset']
    assert sha256(source/'val.npz')==protocol['development_npz_sha256']
    dev=dict(np.load(source/'val.npz',allow_pickle=False))
    by_id={str(s):i for i,s in enumerate(dev['sample_ids'])}
    selected=protocol['sample_ids'];assert len(selected)==len(set(selected))==12
    full_dir=run/'audit/development_replay'/ident
    full_result=json.loads((full_dir/'RESULT.json').read_text())
    assert full_result['passed'] is True
    full=json.loads((full_dir/'output/predictions.json').read_text())['predictions']
    full={p['sample_id']:p['predicted_label'] for p in full}
    assert set(full)==set(by_id)
    target=run/'audit/pointwise_development'/ident;target.mkdir(parents=True,exist_ok=False)
    before=files(run/job['workspace'])
    data=target/'data/datasets'/job['dataset'];data.mkdir(parents=True)
    shutil.copy2(source/'dataset.json',data/'dataset.json')
    mounts=json.loads(Path(job['mounts']).read_text())
    for m in mounts['mounts']:
        if m['destination']=='/data':m['source']=str(target/'data')
    mount_file=target/'mounts.json';save(mount_file,mounts)
    cold=dict(job,evaluation_mounts=str(mount_file))
    env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',
        PYTHONDONTWRITEBYTECODE='1',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
        MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    rows=[]
    for ordinal,sid in enumerate(selected):
        i=by_id[sid];dest=target/str(ordinal);output=dest/'output';output.mkdir(parents=True)
        observed={k:(v[i:i+1] if v.ndim and len(v)==len(by_id) else v)
                  for k,v in dev.items() if k not in ['labels','groups','origin_labels']}
        assert observed['sample_ids'].tolist()==[sid]
        np.savez_compressed(output/'observations.npz',**observed)
        app=dest/'app';shutil.copytree(run/job['workspace'],app,symlinks=True)
        inner=['/opt/python312/bin/python3.12','-B','/app/solution.py','--input','/output/observations.npz','--output','/output/predictions.json']
        command=namespace(run,cold,app,output,inner);save(dest/'command.json',command)
        row=dict(sample_id=sid,full_batch_prediction=full[sid],actual_label=str(dev['labels'][i]))
        try:
            with (dest/'stdout.txt').open('wb') as out,(dest/'stderr.txt').open('wb') as err:
                code=subprocess.run(command,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,timeout=180).returncode
            assert code==0,'Singleton program execution failed'
            predicted=json.loads((output/'predictions.json').read_text())['predictions']
            assert len(predicted)==1 and predicted[0]['sample_id']==sid
            row.update(passed=True,singleton_prediction=predicted[0]['predicted_label'])
        except (AssertionError,ValueError,KeyError,TypeError,OSError,subprocess.TimeoutExpired) as e:
            row.update(passed=False,error_type=type(e).__name__,error=str(e))
        row['observation_sha256']=sha256(output/'observations.npz');rows.append(row)
        # Each query used a fresh mutable copy. Retain original sealed files,
        # commands, input hashes and outputs, without duplicating the model 12x.
        shutil.rmtree(app)
    assert before==files(run/job['workspace'])
    good=[r for r in rows if r['passed']]
    result=dict(job=ident,task_id=job['task_id'],rows=rows,n=12,completed=len(good),
        full_batch_subset_accuracy=float(np.mean([r['full_batch_prediction']==r['actual_label'] for r in rows])),
        singleton_subset_accuracy=float(np.mean([r['singleton_prediction']==r['actual_label'] for r in good])) if len(good)==12 else None,
        changed_predictions=sum(r['singleton_prediction']!=r['full_batch_prediction'] for r in good),
        source_submission_unchanged=True,final_scores_not_read=True,new_fits=0,new_LLM_calls=0,
        original_public_fitting_arrays_unmounted=True,current_input_mount_has_one_query=True,
        submitted_model_artifacts_retained=True,
        protocol_sha256=sha256(protocol_file),evaluator_sha256=sha256(Path(__file__)))
    save(target/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--pointwise-protocol',type=Path)
    a=p.parse_args()
    if a.pointwise_protocol:pointwise(a.run.resolve(),a.job,a.pointwise_protocol)
    else:main(a.run.resolve(),a.job)
