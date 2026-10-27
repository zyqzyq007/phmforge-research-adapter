"""Cold replay a closed classifier on its real training/development observations.

The input mount omits labels, other fitting files and final observations, and
network access is blocked. The frozen submission workspace is preserved, so
its cached artifacts and inference behavior still require a choice audit.
A batch-incompatible training replay is not a zero training score.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import terminal, files, namespace
from run_full_benchmark_campaign import clean_env
from summarize_selection_expansion import metrics


def main(run):
    jobs=terminal(run)
    assert json.loads((run/'OBSERVABLE_REVIEW_COMPLETE.json').read_text())['reviewed_before_new_test_scores']
    audit=json.loads((run/'CODING_CHOICE_AUDIT.json').read_text())
    indexed={r['job']:r for r in audit['rows']}
    destination=run/'public_split_evaluation';destination.mkdir(exist_ok=False)
    results=[]
    for job in jobs:
        ws=run/job['workspace']
        assert files(ws)==indexed[job['id']]['files']
        source=run/'public_data'/job['id']/'datasets'/job['dataset']
        for part in ['train','val']:
            dest=destination/job['id']/part
            inputs=dest/'inputs'/'datasets'/job['dataset'];inputs.mkdir(parents=True)
            original=dict(np.load(source/(part+'.npz'),allow_pickle=False))
            payload={k:v for k,v in original.items() if k not in ['labels','groups']}
            np.savez_compressed(inputs/(part+'.npz'),**payload)
            shutil.copy2(source/'dataset.json',inputs/'dataset.json')
            app=dest/'app';shutil.copytree(ws,app,symlinks=True)
            for p in ['.runtime/tmp','.runtime/mpl']:(app/p).mkdir(parents=True,exist_ok=True)
            output=dest/'output';output.mkdir()
            mounts=json.loads(Path(job['mounts']).read_text())
            for m in mounts['mounts']:
                if m['destination']=='/data':m['source']=str(dest/'inputs')
            mount_file=dest/'mounts.json';save(mount_file,mounts)
            cold_job={**job,'evaluation_mounts':str(mount_file)}
            command=namespace(run,cold_job,app,output,[
                '/opt/python312/bin/python3.12','-B','/app/solution.py','--input',
                '/data/datasets/'+job['dataset']+'/'+part+'.npz','--output','/output/predictions.json'])
            save(dest/'command.json',command)
            env=clean_env(run)
            env.update(PYTHONPATH='/opt/phm_python/extra',PYTHONDONTWRITEBYTECODE='1',
                PATH='/opt/python312/bin:/usr/bin:/bin',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',
                TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl',
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            with (dest/'stdout').open('wb') as out,(dest/'stderr').open('wb') as err:
                try:code=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,
                    stdout=out,stderr=err,timeout=180).returncode
                except subprocess.TimeoutExpired:code='timeout'
            row=dict(job=job['id'],split=part,cold_exit_code=code,
                formal_submission=indexed[job['id']]['status']=='finished' and indexed[job['id']]['response_valid'] is True,
                observation_input_excludes_labels_and_groups=True,final_data_accessed=False,
                replay_driver_performs_fitting=False,original_workspace_artifacts_preserved=True,
                subject_inference_behavior_requires_choice_audit=True)
            try:
                assert code==0,'Cold public-split inference did not finish successfully'
                prediction_path=output/'predictions.json'
                predictions=json.loads(prediction_path.read_text())['predictions']
                labels=[dict(sample_id=str(i),label=y.item(),group=str(g))
                    for i,y,g in zip(original['sample_ids'],original['labels'],original['groups'])]
                info=json.loads((source/'dataset.json').read_text())
                assert all(isinstance(p['sample_id'],str) and
                    any(type(p['predicted_label']) is type(c) and p['predicted_label']==c
                        for c in info['class_names']) for p in predictions)
                row.update(valid=True,metrics=metrics(predictions,labels),
                    prediction_sha256=sha256(prediction_path),
                    interpretation='Cold program score on a reused public split; not an independent performance guarantee. Audit batch assumptions before interpreting as supervised fitting error.')
            except (AssertionError,OSError,ValueError,KeyError,TypeError) as error:
                row.update(valid=False,error_type=type(error).__name__,error=str(error),
                    interpretation='Execution/batch compatibility failure,not zero accuracy and not proof of overfitting.')
            assert files(ws)==indexed[job['id']]['files'],'Original submission mutated'
            save(dest/'SCORE.json',row);results.append(row)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=results,
        final_observations_labels_scores_used=False,original_submissions_unchanged=True)
    save(run/'PUBLIC_SPLIT_SCORES.json',result)
    print(json.dumps([dict(job=r['job'],split=r['split'],valid=r['valid'],
        accuracy=r.get('metrics',{}).get('accuracy'),macro_f1=r.get('metrics',{}).get('macro_f1'))
        for r in results]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    main(parser.parse_args().run.resolve())
