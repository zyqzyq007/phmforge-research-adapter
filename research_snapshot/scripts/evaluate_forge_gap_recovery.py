"""Seal A321 submissions, then evaluate genuine saved programs externally."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import numpy as np
from acquire_forge_pdmb_sources244 import save,sha256
from evaluate_forge_coding import terminal,files,audit
from evaluate_xjtu_feedback import cold
from run_validation_transfer import state

PROJECT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text())


def seal(root):
    assert read(root/'CAMPAIGN_STATUS.json')['status']=='gap_recovery_terminal_frozen'
    assert not (root/'CHOICES_SEALED321.json').exists()
    cases=read(root/'CAMPAIGN_CONFIG.json')['cases'];variants=[];seen=set();subjects=[]
    for case in cases:
        run=root/case['cohort'];job=terminal(run)[0];audit(run,reuse_existing=True)
        native=read(run/'primary'/job['id']/job['id']/'status.json');ws=run/job['workspace']
        subjects.append(dict(case,native_status=native['status'],response_valid=native.get('response_valid'),
            elapsed_seconds=native.get('elapsed_seconds'),workspace=job['workspace'],submission_files=files(ws)))
        if native['status']=='finished' and (ws/'solution.py').is_file():
            variants.append(dict(case,id='agent_'+case['cohort'],kind='program',workspace=job['workspace'],entry='solution.py',workspace_files=files(ws),source='unchanged_normal_submission'))
        if case['dataset'] not in seen:
            seen.add(case['dataset']);reference=root/'datasets'/case['dataset']/'references'
            checkpoints=['model.pt']+['candidates/'+v['checkpoint'] for v in read(reference/'candidates/bank_manifest.json')['models']]
            for checkpoint in checkpoints:
                ident='fixed_'+Path(checkpoint).stem;target=run/'fixed_controls'/ident;target.mkdir(parents=True)
                (target/'solution.py').write_text("import argparse,sys\nsys.path.insert(0,'/tools')\nfrom real_gap_rul_model import infer\np=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()\ninfer('/reference/"+checkpoint+"',a.input,a.output)\n")
                variants.append(dict(case,id=ident,kind='program',workspace=str(target.relative_to(run)),entry='solution.py',workspace_files=files(target),source='prespecified_genuinely_fitted_control',checkpoint_sha256=sha256(reference/checkpoint)))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),all_subjects_terminal=True,final_scores_read=False,
        subjects=subjects,variants=variants,excluded_unfinished_submissions=[v['cohort'] for v in subjects if v['native_status']!='finished'])
    save(root/'CHOICES_SEALED321.json',result);save(PROJECT/'results/gap321/CHOICES_SEALED321.json',result)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    print(json.dumps(dict(variants=len(variants),normal_subjects=sum(v['native_status']=='finished' for v in subjects))))


def metrics(rows,gold):
    indexed={r['sample_id']:r for r in rows};assert len(indexed)==len(rows)==len(gold)
    assert set(indexed)=={g['sample_id'] for g in gold}
    events=[];censored=[]
    for g in gold:
        p=indexed[g['sample_id']];assert int(p['unit_id'])==g['unit_id']
        v=float(p['predicted_rul']);assert math.isfinite(v) and v>=0
        row=dict(g,predicted_rul=v)
        if g['event_observed']:
            row['error']=v-g['actual_rul']
            if isinstance(p.get('anomaly_flag'),bool):row['anomaly_flag']=p['anomaly_flag']
            events.append(row)
        else:
            row['lower_bound_deficit']=max(0,g['censor_lower_bound']-v);censored.append(row)
    per_unit=[]
    for u in sorted({r['unit_id'] for r in events}):
        group=[r for r in events if r['unit_id']==u];e=np.asarray([r['error'] for r in group])
        per_unit.append(dict(unit_id=u,queries=len(e),mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),bias=float(e.mean())))
    result=dict(unit='acquisitions',event_bearings=len(per_unit),event_queries=len(events),
        mean_bearing_mae=float(np.mean([v['mae'] for v in per_unit])),pooled_rmse=float(np.sqrt(np.mean([v['error']**2 for v in events]))),
        per_bearing=per_unit,event_predictions=events,censored_queries=len(censored),censored_predictions=censored,
        censor_bound_violations=sum(v['lower_bound_deficit']>0 for v in censored),
        mean_censor_bound_deficit=float(np.mean([v['lower_bound_deficit'] for v in censored])) if censored else None)
    strata={}
    for flag in [False,True]:
        part=[v for v in events if v.get('anomaly_flag') is flag]
        strata[str(flag)]=dict(queries=len(part),mae=float(np.mean([abs(v['error']) for v in part])) if part else None)
    result['model_detected_anomaly_strata']=strata
    latest=[max([v for v in events if v['unit_id']==u],key=lambda v:int(v['sample_id'].split('_q')[1])) for u in {v['unit_id'] for v in events}]
    pairs=[(a,b) for i,a in enumerate(latest) for b in latest[i+1:] if a['actual_rul']!=b['actual_rul']]
    result['latest_query_ranking']=dict(predicted_order=[v['unit_id'] for v in sorted(latest,key=lambda v:v['predicted_rul'])],
        actual_order=[v['unit_id'] for v in sorted(latest,key=lambda v:v['actual_rul'])],
        pairwise_correct=sum((a['predicted_rul']-b['predicted_rul'])*(a['actual_rul']-b['actual_rul'])>0 for a,b in pairs),
        comparable_pairs=len(pairs),scope='Ordering by numerical endpoint distance, not calibrated physical risk.')
    return result


def evaluate(root):
    sealed=read(root/'CHOICES_SEALED321.json');assert sealed['all_subjects_terminal']
    assert not (root/'EXTERNAL_RESULTS321.json').exists()
    jobs=[]
    for v in sealed['variants']:
        base=root/'datasets'/v['dataset']
        for g in read(base/'evaluation_data/labels.json'):
            jobs.append((v,base/g['input_path'],g['sample_id']))
    def run_query(item):
        v,obs,qid=item;run=root/v['cohort']
        return cold(run,v,'final',obs,qid,run/'external321')
    state(root,status='external_predictions_in_progress',final_scoring_started=True,new_model_sessions=len(sealed['subjects']))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(run_query,jobs))
    save(root/'COLD_REPLAY_STATUS321.json',results)
    summaries=[]
    for v in sealed['variants']:
        gold=read(root/'datasets'/v['dataset']/'evaluation_data/labels.json');rows=[];errors=[];sensor_rows={}
        for g in gold:
            folder=root/v['cohort']/'external321'/v['id']/'final'/g['sample_id']
            status=read(folder/'STATUS.json')
            try:
                assert status['exit_code']==0
                value=read(folder/'predictions.json');assert value.get('unit','acquisitions')=='acquisitions'
                rows.extend(value['predictions'])
                for name,predictions in value.get('sensor_predictions',{}).items():
                    sensor_rows.setdefault(name,[]).extend(predictions)
            except Exception as e:errors.append(dict(sample_id=g['sample_id'],error=type(e).__name__,exit_code=status['exit_code']))
        summary={k:v[k] for k in ['id','dataset','cohort','task_id','source']}
        if not errors:
            try:
                summary['metrics']=metrics(rows,gold);summary['numerical_scoring']='complete'
                summary['sensor_metrics']={}
                for name,predictions in sensor_rows.items():
                    try:summary['sensor_metrics'][name]=metrics(predictions,gold)
                    except Exception as e:summary['sensor_metrics'][name]={'error':type(e).__name__}
            except Exception as e:errors.append(dict(error=type(e).__name__,message=str(e)))
        if errors:summary.update(numerical_scoring='invalid_or_execution_failure',errors=errors)
        summaries.append(summary)
    output=dict(updated_utc=datetime.now(timezone.utc).isoformat(),subjects=sealed['subjects'],variants=summaries,
        original_complete_tasks_claimed=0,new_sources=1,source_substitution_tasks=2,endpoint_proxy_tasks=3,
        source_warning='LDM is distinct from original KAt; IMS recording endpoint is a proxy; no physical risk/compliance or new Agent mechanism certified.',
        final_test_used_for_tuning=False,cold_calls=len(results),new_LLM_calls_after_seal=0)
    save(root/'EXTERNAL_RESULTS321.json',output);save(PROJECT/'results/gap321/EXTERNAL_RESULTS321.json',output)
    state(root,status='five_scoped_tasks_terminal_external_results_ready',final_scores_opened=True,final_scoring_complete=True,
        new_model_sessions=len(sealed['subjects']),actual_running_jobs=0,normal_submissions=sum(v['native_status']=='finished' for v in sealed['subjects']),
        new_confirmed_Agent_hard_problems=0)
    print(json.dumps([{k:v.get(k) for k in ['id','dataset','cohort','numerical_scoring']}|{'mae':v.get('metrics',{}).get('mean_bearing_mae')} for v in summaries],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=['seal','evaluate'],required=True)
    a=p.parse_args();(seal if a.phase=='seal' else evaluate)(a.run.resolve())
