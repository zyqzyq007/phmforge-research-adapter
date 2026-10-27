"""External numerical evaluation for frozen real CWRU/NASA coverage subjects."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import terminal, files, namespace, audit as choice_audit, score as classification_score
from run_full_benchmark_campaign import clean_env


def read(path):
    return json.loads(Path(path).read_text())


def rescore_declared_classes(run):
    """Repair the sparse-class evaluator using already frozen cold predictions.

    No program, input, prediction, model or task label is changed. Preserve the
    original evaluator rejection as infrastructure evidence, not Agent failure.
    """
    import warnings
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score
    from verified_benchmark_scoring import classification,aligned
    jobs=terminal(run);score_path=run/'CODING_SCORE.json';old=read(score_path)
    assert len(jobs)==len(old['rows'])==1
    job=jobs[0];row=old['rows'][0];ws=run/job['workspace']
    frozen=read(run/'CODING_CHOICE_AUDIT.json')['rows'][0]
    assert files(ws)==frozen['files']
    assert row['formal_submission'] is True
    assert all(row[k].get('error')=='Unknown predicted class' for k in ['final','reference'])
    dest=run/'audit/declared_class_rescore';dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(score_path,dest/'ORIGINAL_CODING_SCORE.json')
    evaluation=run/'evaluation'/job['id'];shutil.copy2(evaluation/'SCORE.json',dest/'ORIGINAL_SCORE.json')
    info=read(run/'public_data'/job['id']/'datasets'/job['dataset']/'dataset.json')
    gold_path=run/'hidden'/job['dataset']/'labels.json';gold=read(gold_path);prediction_hashes={}
    for kind,filename in [('final','predictions.json'),('reference','reference.json')]:
        path=evaluation/'output'/filename;before=sha256(path);pred=read(path)['predictions']
        assert row['cold_exit_codes'][kind]==0
        m=classification(pred,gold,declared_classes=info['class_names'])
        p,g=aligned(pred,gold,'sample_id');y=[v['label'] for v in g];v=[r['predicted_label'] for r in p]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            independent=dict(accuracy=float(accuracy_score(y,v)),macro_f1=float(f1_score(y,v,average='macro',zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y,v)))
        assert all(abs(m[k]-value)<1e-12 for k,value in independent.items())
        assert m['confusion_matrix']==confusion_matrix(y,v,labels=m['class_order']).tolist()
        m['macro_f1_label_scope']='Observed actual/predicted union; absent-source classes not silently counted as test support.'
        m['declared_class_universe']=info['class_names']
        m['macro_f1_all_declared_classes']=float(f1_score(y,v,labels=info['class_names'],average='macro',zero_division=0))
        row[kind]=dict(valid=True,metrics=m,independent_sklearn_agrees=True,prediction_sha256=before)
        assert before==sha256(path);prediction_hashes[kind]=before
    row['change_from_reference']={k:row['final']['metrics'][k]-row['reference']['metrics'][k] for k in ['accuracy','macro_f1','balanced_accuracy']}
    original_sha=sha256(dest/'ORIGINAL_CODING_SCORE.json')
    correction=dict(utc=datetime.now(timezone.utc).isoformat(),reason='Original evaluator used actual test labels as the valid prediction universe; valid declared classes absent from a three-bearing sample were rejected.',
        original_score_sha256=original_sha,source_metadata_sha256=sha256(run/'public_data'/job['id']/'datasets'/job['dataset']/'dataset.json'),gold_sha256=sha256(gold_path),
        prediction_hashes_unchanged=prediction_hashes,source_submission_unchanged=True,new_fits=0,new_inferences=0,new_LLM_calls=0,
        primary_macro_F1='Observed actual/predicted union, matching sklearn default; all-declared-class macro F1 separately retained.',
        balanced_accuracy='Mean recall over actual classes with support.',not_an_Agent_failure=True,evaluator_sha256=sha256(Path(__file__)),
        scoring_helper_sha256=sha256(Path(__file__).with_name('verified_benchmark_scoring.py')))
    assert files(ws)==frozen['files']
    save(dest/'CORRECTION.json',correction);old['scoring_correction']=correction;save(evaluation/'SCORE.json',row);save(score_path,old)
    return correction


def score_nasa(run):
    jobs=terminal(run);assert not (run/'CODING_SCORE.json').exists()
    assert read(run/'OBSERVABLE_REVIEW_COMPLETE.json')['reviewed_before_new_test_scores']
    audited=read(run/'CODING_CHOICE_AUDIT.json');assert audited['new_hidden_test_scores_not_read']
    indexed={v['job']:v for v in audited['rows']}
    gold=read(run/'hidden/CMAPSS_FD001/labels.json');truth={v['sample_id']:v for v in gold}
    true_top=sorted(gold,key=lambda v:(v['actual_rul'],v['unit_id']))[:10]
    def one(job):
        ws=run/job['workspace'];sealed=indexed[job['id']]
        assert sealed['files']==files(ws),'Submission changed after sealing'
        dest=run/'evaluation'/job['id'];dest.mkdir(parents=True)
        if sealed['status']!='finished' or sealed['response_valid'] is not True:
            row=dict(job=job['id'],task_id=job['task_id'],status=sealed['status'],
                formal_submission=False,final_scores_not_computed=True,
                not_scored_reason='No normal submitted program; preserve transport/termination outcome separately.')
            save(dest/'SCORE.json',row)
            return row
        app=dest/'app';shutil.copytree(ws,app,symlinks=True)
        out=dest/'output';out.mkdir()
        env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PYTHONDONTWRITEBYTECODE='1',
            PATH='/opt/python312/bin:/usr/bin:/bin',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',
            TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        inputs='/data/datasets/CMAPSS_FD001/test.npz'
        commands={k:['/opt/python312/bin/python3.12','-B','/app/solution.py','--input',inputs,'--output','/output/'+k+'.json'] for k in ['final','repeat']}
        commands['reference']=['/opt/python312/bin/python3.12','-B','-c',
            "import sys;sys.path.insert(0,'/tools');from real_rul_model import infer;infer('/reference/model.pt',sys.argv[1],'/output/reference.json')",inputs]
        row=dict(job=job['id'],task_id=job['task_id'],status=sealed['status'],response_valid=sealed['response_valid'],
            formal_submission=sealed['status']=='finished' and sealed['response_valid'] is True,
            partial_decision_contract=job['partial_decision_contract'],cold_exit_codes={})
        predicted={};payloads={}
        for kind,inner in commands.items():
            argv=namespace(run,job,app,out,inner);save(dest/(kind+'_command.json'),argv)
            try:
                with (dest/(kind+'.stdout')).open('wb') as stdout,(dest/(kind+'.stderr')).open('wb') as stderr:
                    code=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=180).returncode
                row['cold_exit_codes'][kind]=code;assert code==0,'Cold execution failed'
                payload=read(out/(kind+'.json'));rows=payload['predictions'];by_id={v['sample_id']:v for v in rows}
                assert len(by_id)==len(rows)==len(gold) and set(by_id)==set(truth),'Incomplete/duplicate final identities'
                for sid,v in by_id.items():
                    assert type(v['unit_id']) is int and v['unit_id']==truth[sid]['unit_id'],'Unit identity mismatch'
                    assert type(v['predicted_rul']) in [float,int] and math.isfinite(v['predicted_rul']) and v['predicted_rul']>=0
                ordered=sorted(truth);residual=np.asarray([by_id[s]['predicted_rul']-truth[s]['actual_rul'] for s in ordered])
                mae=float(np.abs(residual).mean());rmse=float(np.sqrt(np.square(residual).mean()))
                assert abs(mae-math.fsum(abs(float(x)) for x in residual)/len(residual))<1e-10
                assert abs(rmse-math.sqrt(math.fsum(float(x)**2 for x in residual)/len(residual)))<1e-10
                low={v['unit_id'] for v in rows if v['predicted_rul']<20};actual_low={v['unit_id'] for v in gold if v['actual_rul']<20}
                top=sorted(rows,key=lambda v:(v['predicted_rul'],v['unit_id']))[:10]
                metrics=dict(n=len(rows),mae_cycles=mae,rmse_cycles=rmse,bias_cycles=float(residual.mean()),
                    predicted_high_risk_count=len(low),actual_high_risk_count=len(actual_low),
                    threshold20_true_positive=len(low&actual_low),threshold20_false_positive=len(low-actual_low),
                    threshold20_false_negative=len(actual_low-low),top10_overlap=len({v['unit_id'] for v in top}&{v['unit_id'] for v in true_top}),
                    independent_arithmetic_agrees=True)
                row[kind]=dict(valid=True,metrics=metrics,prediction_sha256=sha256(out/(kind+'.json')))
                predicted[kind]=by_id;payloads[kind]=payload
            except (AssertionError,OSError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired) as e:
                row[kind]=dict(valid=False,error_type=type(e).__name__,error=str(e))
        if 'final' in predicted and 'repeat' in predicted:
            row['cold_repeat_max_absolute_difference']=max(abs(predicted['final'][s]['predicted_rul']-predicted['repeat'][s]['predicted_rul']) for s in truth)
            row['cold_repeat_predictions_match']=row['cold_repeat_max_absolute_difference']<1e-6
        if 'final' in predicted:
            payload=payloads['final'];rows=list(predicted['final'].values());low=sorted(v['unit_id'] for v in rows if v['predicted_rul']<20)
            if job['task_id']=='pdm_rul_001':
                expected=[v['unit_id'] for v in sorted(rows,key=lambda v:(v['predicted_rul'],v['unit_id']))[:10]]
                row['top10_output_consistent_with_own_predictions']=payload.get('top_10_risk_units')==expected
            elif job['task_id']=='pdm_cost_benefit_001':
                costs=payload.get('cost_analysis',{})
                row['given_cost_checks']=dict(preventive_expected=5000*len(rows),reactive_expected=50000*len(low),
                    preventive_correct=costs.get('preventive_cost')==5000*len(rows),
                    reactive_correct=costs.get('reactive_cost')==50000*len(low),
                    predicted_failures_correct=sorted(costs.get('predicted_failures',[]))==low)
                row['actual_policy_savings_evaluated']=False
            elif job['task_id']=='pdm_safety_001':
                row['threshold20_output_consistent_with_own_predictions']=sorted(payload.get('high_risk_units',[]))==low
            row['decision_outputs_for_evidence_review']={k:v for k,v in payload.items() if k not in ['predictions','checkpoint_sha256','input_sha256']}
        if 'final' in predicted and 'reference' in predicted:
            row['change_from_reference']={k:row['final']['metrics'][k]-row['reference']['metrics'][k] for k in ['mae_cycles','rmse_cycles','top10_overlap']}
        save(dest/'SCORE.json',row);return row
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,jobs))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),source='Official NASA simulator; allowed by user; not field measurements.',
        original_PHMForge_score=False,real_numerical_evaluation=True,rows=rows,
        physical_failure_risk_or_legal_compliance_certified=False)
    save(run/'CODING_SCORE.json',result);return result


def cost_details(run):
    job,=read(run/'PLAN187.json')['jobs'];out=run/'evaluation'/job['id']/'output'
    payload=read(out/'predictions.json');gold=read(run/'hidden/CWRU/labels.json')
    truth={v['sample_id']:v for v in gold};pred={v['sample_id']:v for v in payload['predictions']}
    units={v['unit_id'] for v in gold};truth_unit={u:{v['label'] for v in gold if v['unit_id']==u} for u in units}
    assert all(len(v)==1 for v in truth_unit.values())
    actual={u:next(iter(v)) for u,v in truth_unit.items()}
    decisions=payload.get('unit_decisions',[]);indexed={v['unit_id']:v for v in decisions}
    maint={'inner_race':3000,'outer_race':4000,'ball_race':5000};fail={k:v*10 for k,v in maint.items()}
    checks=[]
    for unit,v in indexed.items():
        label=v.get('predicted_label');valid=label in maint
        checks.append(dict(unit_id=unit,known_unit=unit in units,label_valid=valid,
            correct_fault=unit in actual and label==actual[unit],
            maintenance_cost_correct=valid and v.get('maintenance_cost')==maint[label],
            failure_cost_correct=valid and v.get('failure_cost')==fail[label],
            conditional_avoided_cost_correct=valid and v.get('avoided_cost_if_failure_prevented')==fail[label]-maint[label],
            benefit_ratio_correct=valid and v.get('cost_benefit_ratio')==fail[label]/maint[label]))
    coverage=len(indexed)==len(decisions)==len(units) and set(indexed)==units
    valid_labels=coverage and all(v.get('predicted_label') in maint for v in decisions)
    expected_predicted_totals=(dict(maintenance=sum(maint[v['predicted_label']] for v in decisions),
        failure=sum(fail[v['predicted_label']] for v in decisions),
        avoided_if_failure_prevented=sum(fail[v['predicted_label']]-maint[v['predicted_label']] for v in decisions))
        if valid_labels else None)
    totals=payload.get('cost_totals',{})
    result=dict(task_id='pdm_cost_benefit_002',unit_coverage_exact=coverage,
        actual_unit_count=len(units),window_count=len(gold),per_unit=checks,
        whole_record_accuracy=sum(v['correct_fault'] for v in checks)/len(units) if coverage else None,
        original_given_cost_total_by_actual_class=dict(maintenance=sum(maint[k] for k in actual.values()),failure=sum(fail[k] for k in actual.values())),
        tied_best_ratio_fault_types=sorted(maint),
        tied_best_ratio_output_correct=sorted(payload.get('best_ratio_fault_types',[]))==sorted(maint),
        priority_tiers_all_tied=len({v.get('priority_tier') for v in decisions})==1 if coverage else False,
        expected_totals_from_predicted_classes=expected_predicted_totals,
        reported_totals_correct=(all(totals.get(k)==v for k,v in expected_predicted_totals.items())
            if expected_predicted_totals is not None else False),
        numerical_given_costs_only=True,actual_expected_maintenance_benefit_evaluated=False)
    save(run/'COST_DETAILS307.json',result);return result


def cold_rul_followup(run,job,observation,destination,kind):
    """Keep the declared fitting-data access; mount one target-free query only."""
    destination.mkdir(parents=True,exist_ok=False)
    ws=run/job['workspace'];app=destination/'app'
    if kind=='final':
        shutil.copytree(ws,app,symlinks=True,ignore=shutil.ignore_patterns('.git','__pycache__'))
    else:app.mkdir()
    for rel in ['.runtime/tmp','.runtime/mpl']:(app/rel).mkdir(parents=True,exist_ok=True)
    output=destination/'output';output.mkdir()
    mounts=read(Path(job['mounts']))
    mounts['mounts'].append(dict(source=str(observation),destination='/input/query.npz'))
    mount_file=destination/'mounts.json';save(mount_file,mounts)
    if kind=='final':command=['/opt/python312/bin/python3.12','-B','/app/solution.py']
    elif job['dataset']=='FEMTO':
        command=['/opt/python312/bin/python3.12','-B','/tools/real_bearing_rul_model.py','infer','--model','/reference/model.pt']
    else:
        command=['/opt/python312/bin/python3.12','-B','/tools/real_xjtu_rul_model.py','--model','/reference/candidates/rf200.joblib']
    argv=namespace(run,dict(job,evaluation_mounts=str(mount_file)),app,output,
        command+['--input','/input/query.npz','--output','/output/predictions.json'])
    save(destination/'command.json',argv)
    env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra:/tools',PATH='/opt/python312/bin:/usr/bin:/bin',
        PYTHONDONTWRITEBYTECODE='1',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',
        MPLCONFIGDIR='/app/.runtime/mpl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (destination/'stdout').open('wb') as out,(destination/'stderr').open('wb') as err:
        try:code=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=180).returncode
        except subprocess.TimeoutExpired:code='timeout'
    result=dict(exit_code=code,input_sha256=sha256(observation),one_final_query_input_only=True,
        fitting_data_access_preserved_as_declared=True,final_labels_mounted=False)
    save(destination/'STATUS.json',result)
    assert code==0,'Cold execution failed: '+str(code)
    payload=read(output/'predictions.json')
    # The exact mutable cold copy is an evaluation cache; the sealed original remains.
    shutil.rmtree(app)
    return payload


def followup_inputs(root,case,split):
    """Return existing final observations, or derive causal development queries."""
    if split=='final':
        if case['dataset']=='FEMTO':
            source=root/'source/femto/rul/source_data'
            return [(source/'public/datasets/FEMTO/test.npz',read(source/'hidden/FEMTO/labels.json'))]
        source=root/'source/xjtu';gold=read(source/'hidden/final_labels.json')
        return [(source/g['input_path'],[g]) for g in gold]
    source=root/'common'/case['dataset']/'public/datasets'/case['dataset']
    dev=dict(np.load(source/'val.npz',allow_pickle=False))
    target=root/'evaluation315/development_inputs'/case['dataset'];target.mkdir(parents=True,exist_ok=True)
    gold=[dict(sample_id=str(s),unit_id=int(u),scenario_id=int(c),actual_rul=float(y))
          for s,u,c,y in zip(dev['sample_ids'],dev['unit_ids'],dev['scenario_ids'],dev['labels'])]
    if case['dataset']=='FEMTO':
        path=target/'batch.npz'
        if not path.exists():np.savez_compressed(path,**{k:v for k,v in dev.items() if k not in ['labels','groups','origin_labels','actual_rul']})
        return [(path,gold)]
    result=[]
    for i,g in enumerate(gold):
        path=target/(g['sample_id']+'.npz')
        if not path.exists():
            observed={k:(v if k=='dataset' else v[i:i+1]) for k,v in dev.items() if k!='labels' and not k.startswith('history_')}
            start,end=map(int,dev['history_offsets'][i:i+2])
            for k in ['history_features','history_elapsed_minutes','history_channel_rms','history_channel_max_abs']:
                observed[k]=dev[k][start:end]
            observed['history_offsets']=np.asarray([0,end-start])
            np.savez_compressed(path,**observed)
        result.append((path,[g]))
    return result


def score_rul_followups(root,mode):
    """Independent actual-program scoring; no manufactured pre-final review gate."""
    from evaluate_forge_coding import bearing_metrics
    from evaluate_xjtu_feedback import numerical_metrics
    from sklearn.metrics import mean_absolute_error
    cases=read(root/'CAMPAIGN_CONFIG.json')['cases']
    if mode=='score':
        assert not (root/'RUL_RESULTS315.json').exists()
        for case in cases:
            run=root/case['cohort'];terminal(run)
            assert read(run/'OBSERVABLE_REVIEW_COMPLETE.json')['reviewed_before_new_test_scores']
    rows=[]
    for case in cases:
        run=root/case['cohort'];job,=read(run/'PLAN187.json')['jobs']
        status=read(run/'primary'/job['id']/job['id']/'status.json')
        if mode=='development' and status['status']!='finished':continue
        terminal(run)
        if not (run/'CODING_CHOICE_AUDIT.json').exists():choice_audit(run)
        frozen=read(run/'CODING_CHOICE_AUDIT.json')['rows'][0]
        assert files(run/job['workspace'])==frozen['files']
        normal=frozen['status']=='finished' and frozen['response_valid'] is True
        if not normal:
            rows.append(dict(case=case,status=frozen['status'],formal_submission=False,final_scores_not_computed=True));continue
        dest=run/('development315' if mode=='development' else 'evaluation315')
        cache=dest/'RESULT.json'
        if cache.exists():
            previous=read(cache);assert previous['choice_audit_sha256']==sha256(run/'CODING_CHOICE_AUDIT.json')
            assert previous['evaluator_sha256']==sha256(Path(__file__))
            rows.append(previous);continue
        row=dict(case=case,status=frozen['status'],formal_submission=True,results={},
            choice_audit_sha256=sha256(run/'CODING_CHOICE_AUDIT.json'),evaluator_sha256=sha256(Path(__file__)),
            fitting_data_visible_at_inference=True,final_labels_hidden=True)
        for kind in (['final'] if mode=='development' else ['final','starting']):
            predictions=[];truth=[];evidence=[]
            try:
                for i,(path,gold) in enumerate(followup_inputs(root,case,'development' if mode=='development' else 'final')):
                    payload=cold_rul_followup(run,job,path,dest/kind/str(i),kind)
                    values=payload['predictions'];lookup={v['sample_id']:v for v in values}
                    assert len(lookup)==len(values)==len(gold) and set(lookup)=={v['sample_id'] for v in gold}
                    for g in gold:
                        p=lookup[g['sample_id']]
                        assert type(p['unit_id']) is int and type(p['scenario_id']) is int
                        assert (p['unit_id'],p['scenario_id'])==(g['unit_id'],g['scenario_id'])
                        assert isinstance(p['predicted_rul'],(float,int)) and not isinstance(p['predicted_rul'],bool)
                        assert math.isfinite(p['predicted_rul']) and p['predicted_rul']>=0
                    predictions.extend(values);truth.extend(gold);evidence.append(payload.get('decision_evidence'))
                if case['dataset']=='FEMTO':
                    metrics=bearing_metrics(predictions,truth)
                    independent=mean_absolute_error([v['actual_rul'] for v in truth],[next(p['predicted_rul'] for p in predictions if p['sample_id']==g['sample_id']) for g in truth])
                    assert abs(metrics['mae']-independent)<1e-8
                    detail=dict(all=metrics,independent_sklearn_agrees=True)
                    if mode=='score':
                        clean=[g for g in truth if g['native_continuation_consistency']];ids={g['sample_id'] for g in clean}
                        detail['consistent10']=bearing_metrics([p for p in predictions if p['sample_id'] in ids],clean)
                else:detail=numerical_metrics(predictions,truth)
                row['results'][kind]=dict(valid=True,metrics=detail,predictions=predictions,decision_evidence=evidence)
            except (AssertionError,ValueError,KeyError,TypeError,OSError,subprocess.TimeoutExpired) as error:
                row['results'][kind]=dict(valid=False,error_type=type(error).__name__,error=str(error))
        assert files(run/job['workspace'])==frozen['files'],'Original submission changed during evaluation'
        row.update(scope='Numerical RUL prediction; no calibrated physical risk, regulatory certification or optimized maintenance effect validated.',
            development_scope=('FEMTO supplied final reference refits all learning bearings; development is not independent.' if case['dataset']=='FEMTO' else 'Three held-out development bearings; program access to permitted public fitting/development data retained.'))
        save(cache,row);rows.append(row)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),mode=mode,planned_original_directions=4,rows=rows,
        final_scores_opened=mode=='score',no_automatic_hard_problem_claim=True,source_and_prior_scores_known_to_researcher=True)
    save(root/('DEVELOPMENT_RESULTS315.json' if mode=='development' else 'RUL_RESULTS315.json'),result)
    print(json.dumps(dict(mode=mode,rows=[dict(case=r['case'],formal_submission=r['formal_submission'],
        metrics={k:v.get('metrics',v) for k,v in r.get('results',{}).items()}) for r in rows]),ensure_ascii=False))


def main(root):
    # Observable review is a separate, mandatory pre-final gate, never silently
    # manufactured from the final scores by this evaluator.
    for c in ['c','n']:
        terminal(root/c)
        assert read(root/c/'OBSERVABLE_REVIEW_COMPLETE.json')['reviewed_before_new_test_scores']
    classification_score(root/'c');cost=cost_details(root/'c');nasa=score_nasa(root/'n')
    save(root/'EXTERNAL_SCORE307.json',dict(utc=datetime.now(timezone.utc).isoformat(),cost=cost,nasa=nasa,
        new_Agent_failure_mechanism_confirmed=False,ordinary_review_performed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--followup-mode',choices=['development','score'])
    p.add_argument('--declared-class-rescore',action='store_true');a=p.parse_args()
    assert not (a.followup_mode and a.declared_class_rescore)
    if a.declared_class_rescore:rescore_declared_classes(a.run.resolve())
    elif a.followup_mode:score_rul_followups(a.run.resolve(),a.followup_mode)
    else:main(a.run.resolve())
