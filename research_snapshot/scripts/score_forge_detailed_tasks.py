"""Task-specific external scores for the measured detailed PHMForge tasks.

No fitting, candidate selection, or model calls. Run only after the choices
are audited and frozen. Missing extra outputs remain failures, not imputed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from sklearn.metrics import f1_score

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import score, terminal, files, namespace
from run_full_benchmark_campaign import clean_env
from prepare_forge_detailed_tasks import SPECS, read, arrays


def detailed(predictions, labels, case):
    ids=[r['sample_id'] for r in predictions]
    truth={r['sample_id']:r for r in labels}
    assert len(set(ids))==len(ids)==len(truth) and set(ids)==set(truth)
    correct={r['sample_id']:r['predicted_label']==truth[r['sample_id']]['label'] for r in predictions}
    fields={'h':['label','bearing_type','load_w','shaft_frequency_hz'],
            'p':['label','origin_label'],'m':['label','speed_range_rpm','speed_trend'],
            'f':['label','load_lb']}[case]
    out=dict(n=len(truth),correct=sum(correct.values()),accuracy=sum(correct.values())/len(truth),strata={})
    for field in fields+(['speed_load'] if case=='h' else []):
        buckets=defaultdict(list)
        for ident,row in truth.items():
            value=f"{row['shaft_frequency_hz']}Hz/{row['load_w']}W" if field=='speed_load' else str(row[field])
            buckets[value].append(ident)
        result=[]
        for value,group in sorted(buckets.items()):
            n=len(group);count=sum(correct[i] for i in group)
            result.append(dict(value=value,n=n,correct=count,accuracy=count/n,
                source_groups=len({truth[i]['group'] for i in group})))
        assert sum(v['n'] for v in result)==len(truth)
        assert sum(v['correct'] for v in result)==sum(correct.values())
        out['strata'][field]=result
    groups=defaultdict(list)
    for ident,row in truth.items():groups[row['group']].append(ident)
    out['group_mean_accuracy']=float(np.mean([np.mean([correct[i] for i in ids]) for ids in groups.values()]))
    out['source_group_count']=len(groups)
    if case=='p':
        allowed={'healthy','artificial','accelerated_life'}
        missing=[r['sample_id'] for r in predictions if r.get('predicted_origin') not in allowed]
        out['origin_output_valid']=not missing
        if missing:out['missing_or_invalid_origin_ids']=missing
        else:
            y=[truth[r['sample_id']]['origin_label'] for r in predictions]
            p=[r['predicted_origin'] for r in predictions]
            origin=[a==b for a,b in zip(y,p)]
            faulty=[i for i,r in enumerate(predictions) if truth[r['sample_id']]['label']!=0]
            out.update(origin_accuracy=float(np.mean(origin)),origin_macro_f1=float(f1_score(y,p,average='macro')),
                joint_accuracy=float(np.mean([v and correct[r['sample_id']] for r,v in zip(predictions,origin)])),
                damaged_only_origin_accuracy=float(np.mean([origin[i] for i in faulty])),damaged_only_n=len(faulty))
    return out


def public_validation(cohort,job,case):
    """Evaluate the submitted complete program without public targets mounted."""
    root=cohort/'detailed_evaluation/development';assert not root.exists()
    dataset=job['dataset'];source=cohort/'public_data'/job['id']/'datasets'/dataset
    data=arrays(source/'val.npz')
    observations={k:v for k,v in data.items() if k not in ['labels','origin_labels','groups']}
    target=root/'data/datasets'/dataset;target.mkdir(parents=True)
    np.savez_compressed(target/'val.npz',**observations)
    shutil.copy2(source/'dataset.json',target/'dataset.json')
    for filename in ['dataset_registry.json','tool_catalog.json']:
        shutil.copy2(cohort/'public_data'/job['id']/filename,root/'data'/filename)
    if case=='m':
        for rel in data['record_files']:
            dest=target/str(rel);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/str(rel),dest)
    workspace=cohort/job['workspace'];before=files(workspace)
    app=root/'app';shutil.copytree(workspace,app,symlinks=True)
    output=root/'output';output.mkdir()
    mounts=read(job['mounts'])
    for mount in mounts['mounts']:
        if mount['destination']=='/data':mount['source']=str(root/'data')
    save(root/'mounts.json',mounts)
    query=dict(job,evaluation_mounts=str(root/'mounts.json'))
    command=namespace(cohort,query,app,output,['/opt/python312/bin/python3.12','-B','/app/solution.py',
        '--input','/data/datasets/'+dataset+'/val.npz','--output','/output/predictions.json'])
    save(root/'command.json',command)
    env=clean_env(cohort);env.update(PYTHONPATH='/opt/phm_python/extra',PYTHONDONTWRITEBYTECODE='1',
        PATH='/opt/python312/bin:/usr/bin:/bin',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
        MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (root/'stdout').open('wb') as out,(root/'stderr').open('wb') as err:
        try:code=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=180).returncode
        except subprocess.TimeoutExpired:code='timeout'
    result=dict(exit_code=code,complete_program_replayed=True,labels_groups_and_origin_targets_absent=True,
        reference_training_audit_remains_visible=True,final_data_used=False)
    try:
        assert code==0,'Public validation program execution failed'
        predictions=read(output/'predictions.json')['predictions']
        labels=[]
        for i,ident in enumerate(data['sample_ids']):
            row=dict(sample_id=str(ident),label=int(data['labels'][i]),group=str(data['groups'][i]))
            for key in ['bearing_type','load_w','shaft_frequency_hz','speed_range_rpm','speed_trend','load_lb']:
                if key in data:row[key]=data[key][i].item()
            if 'origin_labels' in data:row['origin_label']=str(data['origin_labels'][i])
            labels.append(row)
        result.update(valid=True,metrics=detailed(predictions,labels,case),prediction_sha256=sha256(output/'predictions.json'))
    except (AssertionError,KeyError,TypeError,ValueError,OSError) as error:
        result.update(valid=False,error_type=type(error).__name__,error=str(error))
    assert files(workspace)==before
    save(root/'SCORE.json',result)
    return result


def main(root):
    assert not (root/'DETAILED_RESULTS306.json').exists()
    for case,_,_ in SPECS:
        cohort=root/case;terminal(cohort)
        assert read(cohort/'OBSERVABLE_REVIEW_COMPLETE.json')['reviewed_before_new_test_scores']
        assert read(cohort/'CODING_CHOICE_AUDIT.json')['new_hidden_test_scores_not_read']
    rows=[]
    for case,name,task in SPECS:
        cohort=root/case;job,=read(cohort/'PLAN187.json')['jobs']
        if not (cohort/'CODING_SCORE.json').exists():score(cohort)
        generic,=read(cohort/'CODING_SCORE.json')['rows']
        row=dict(task_id=task,dataset=name,job=job['id'],status=generic['status'],formal_submission=generic['formal_submission'],
            base_score=generic,original_question_unchanged=True,source_correction=(case=='m'),
            original_official_score_claimed=False)
        try:
            assert generic['final']['valid'],'Invalid fault predictions'
            predictions=read(cohort/'evaluation'/job['id']/'output/predictions.json')['predictions']
            labels=read(cohort/'hidden'/name/'labels.json')
            row['test']=detailed(predictions,labels,case)
            assert abs(row['test']['accuracy']-generic['final']['metrics']['accuracy'])<1e-12
            row['test']['actual_stratification_reconciles_to_overall']=True
        except (AssertionError,KeyError,TypeError,ValueError,OSError) as error:
            row['test']=dict(valid=False,error_type=type(error).__name__,error=str(error))
        row['development']=public_validation(cohort,job,case)
        ws=cohort/job['workspace']
        row['submitted_stratified_validation']=(ws/'stratified_validation.json').is_file()
        row['submitted_report']=(ws/'report.md').is_file()
        save(cohort/'DETAILED_SCORE306.json',row);rows.append(row)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),denominator=4,rows=rows,
        new_original_task_directions=len(rows),new_independent_sources=0,
        direct_causal_overfitting_inferred=False,all_failure_outcomes_retained=True)
    save(root/'DETAILED_RESULTS306.json',result)
    save(Path(__file__).resolve().parents[1]/'results/coverage306/DETAILED_RESULTS306.json',result)
    print(json.dumps([dict(task=x['task_id'],status=x['status'],test=x.get('test',{}).get('accuracy'),
        origin=x.get('test',{}).get('origin_accuracy'),dev=x.get('development',{}).get('metrics',{}).get('accuracy')) for x in rows]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
