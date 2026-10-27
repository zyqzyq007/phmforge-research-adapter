"""Seal review choices, then re-score hash-bound existing final predictions.

There is no model fitting here. These are outcome-blind selections on previously
evaluated data, not additional independent test cases or newly generated scores.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from acquire_forge_pdmb_sources244 import save,sha256
from evaluate_forge_coding import audit,files,terminal
from evaluate_xjtu_feedback import numerical_metrics
from run_frozen_candidate_review import source_context
from run_validation_transfer import state,utc
from summarize_selection_expansion import metrics,contrast

PROJECT=Path(__file__).resolve().parents[1]


def seal(root):
    assert not (root/'REVIEW_CHOICES_SEALED281.json').exists()
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    manual=json.loads((root/'audit/OBSERVABLE_REVIEW281.json').read_text())
    assert manual.get('all_trajectories_reviewed_before_comparison',manual.get('all_six_trajectories_reviewed_before_comparison',False))
    assert {x['case'] for x in manual['cases']}=={x['id'] for x in config['cases']}
    indexed={x['case']:x for x in manual['cases']}
    rows=[]
    for case in config['cases']:
        run=root/case['id'];job=terminal(run)[0];audit(run,reuse_existing=True)
        ws=run/job['workspace'];catalog=json.loads((ws/'candidates.json').read_text())
        frozen=json.loads((run/'INITIAL_FREEZE187_FIFO_FILES.json').read_text())['files']
        immutable=set(str(ws/p) for p in json.loads((ws/'readonly_inputs.json').read_text()))
        for entry in frozen:
            p=Path(entry['path'])
            if str(p) in immutable or not p.is_relative_to(ws):assert sha256(p)==entry['sha256'],str(p)
        selection=json.loads((ws/'review_selection.json').read_text())
        candidate=next(x for x in catalog['candidates'] if x['id']==selection['candidate_id'])
        status=json.loads((run/'primary'/job['id']/job['id']/'status.json').read_text())
        complete=status['status']=='finished' and status.get('response_valid') is True and all((ws/f).is_file() for f in ['review_report.md','review_validation.json']) and selection.get('reason')!='Starter only; review not completed'
        review=indexed[case['id']]
        rows.append(dict(case,job=job['id'],candidate=candidate,selection=selection,status=status['status'],complete=complete,
            protocol_deviation=review['protocol_deviation'],trace_review=review,workspace_files=files(ws),
            qualification_sha256=sha256(run/'FIXED_CANDIDATE_REPLAY281.json'),catalog_sha256=sha256(ws/'candidates.json')))
    result=dict(utc=utc(),all_terminal=True,all_choices_fixed=True,cases=rows,
        protocol_sha256=sha256(root/'CAMPAIGN_CONFIG.json'),manual_review_sha256=sha256(root/'audit/OBSERVABLE_REVIEW281.json'),
        final_scores_known_to_researcher_before_this_campaign=True,final_scores_hidden_from_subjects=True)
    save(root/'REVIEW_CHOICES_SEALED281.json',result)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    state(root,status='review_choices_sealed_waiting_archived_prediction_comparison')


def independent_classification(predictions,gold,reported):
    lookup={x['sample_id']:x['predicted_label'] for x in predictions}
    truth=Counter(x['label'] for x in gold);guesses=Counter(lookup[x['sample_id']] for x in gold)
    correct=Counter(x['label'] for x in gold if lookup[x['sample_id']]==x['label'])
    accuracy=sum(correct.values())/len(gold)
    classes=set(truth)|set(guesses)
    f1=sum(2*correct[c]/(truth[c]+guesses[c]) for c in classes)/len(classes)
    recall=sum(correct[c]/truth[c] for c in truth)/len(truth)
    assert abs(accuracy-reported['accuracy'])<1e-12 and abs(f1-reported['macro_f1'])<1e-12 and abs(recall-reported['balanced_accuracy'])<1e-12
    return dict(passed=True,accuracy=accuracy,macro_f1=f1,balanced_accuracy=recall)


def score(root):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    sealed=json.loads((root/'REVIEW_CHOICES_SEALED281.json').read_text())
    assert sealed['all_choices_fixed'] and sha256(root/'CAMPAIGN_CONFIG.json')==sealed['protocol_sha256']
    assert not (root/'NUMERICAL_RESULTS281.json').exists(),'No score overwrite'
    results=[]
    for case in sealed['cases']:
        run=root/case['id'];job=json.loads((run/'PLAN187.json').read_text())['jobs'][0]
        assert files(run/job['workspace'])==case['workspace_files']
        source=Path(case['source_archive']);manifest,receipt=source_context(source)
        evidence=[]
        def read(relative):
            entry=manifest[relative];p=source/relative;assert sha256(p)==entry['sha256']
            evidence.append(dict(path=relative,sha256=entry['sha256']))
            return json.loads(p.read_text())
        def prediction(candidate):
            if case['family']=='classification':
                sourcejob=read(case['cohort']+'/PLAN187.json')['jobs'][0]
                if candidate=='original':path=case['cohort']+'/evaluation/'+sourcejob['id']+'/output/predictions.json'
                elif candidate=='reference_mlp':path=case['cohort']+'/evaluation/'+sourcejob['id']+'/output/reference.json'
                elif candidate.startswith('bank_'):path='controls/'+case['source_case']+'/'+candidate[5:]+'_train_external_predictions.json'
                else:path='candidate_replays/'+case['source_case']+'/output/'+candidate+'.json'
                return read(path)['predictions']
            variant='agent_'+case['cohort'] if candidate=='original' else candidate
            prefix='evaluation278/'+variant+'/final/'
            paths=sorted(p for p in manifest if p.startswith(prefix) and p.endswith('/predictions.json'))
            assert len(paths)==15,(variant,len(paths))
            return [row for p in paths for row in read(p)['predictions']]
        chosen=case['candidate']['id']
        original=prediction('original');selected=prediction(chosen)
        if case['family']=='classification':
            gold=read(case['cohort']+'/hidden/'+case['dataset']+'/labels.json')
            before=metrics(original,gold);after=metrics(selected,gold)
            independent=[independent_classification(p,gold,m) for p,m in [(original,before),(selected,after)]]
            difference=contrast(original,selected,gold)
            change=after['macro_f1']-before['macro_f1']
            primary='macro_f1'
        else:
            gold=read('hidden/final_labels.json')
            before=numerical_metrics(original,gold);after=numerical_metrics(selected,gold)
            independent=[dict(passed=m['independent_metric_check']) for m in [before,after]]
            change=before['mean_bearing_mae_minutes']-after['mean_bearing_mae_minutes']
            primary='mean_bearing_mae_minutes'
            difference=dict(improvement_minutes=change,relative_error_change=(after[primary]-before[primary])/before[primary])
        numerical_outcome='repair' if change>1e-9 else ('worse' if change< -1e-9 else 'unchanged')
        outcome='protocol_deviation' if case['protocol_deviation'] else ('incomplete' if not case['complete'] else numerical_outcome)
        # Report the whole declared candidate menu, not only successful choices.
        catalog=json.loads((run/job['workspace']/'candidates.json').read_text())
        all_candidates=[]
        for candidate in catalog['candidates']:
            p=prediction(candidate['id'])
            m=metrics(p,gold) if case['family']=='classification' else numerical_metrics(p,gold)
            if case['family']=='classification':independent_classification(p,gold,m)
            all_candidates.append(dict(candidate=candidate['id'],scope=candidate['scope'],metrics=m))
        best=(max if case['family']=='classification' else min)(all_candidates,key=lambda x:x['metrics'][primary])
        selector=json.loads((root/'audit/FROZEN_DEVELOPMENT_SELECTOR281.json').read_text())
        fixed_id=next(x['candidate'] for x in selector['cases'] if x['case']==case['id'])
        fixed=next(x for x in all_candidates if x['candidate']==fixed_id)
        result=dict(case=case['id'],family=case['family'],dataset=case['dataset'],selected=chosen,
            complete=case['complete'],status=case['status'],protocol_deviation=case['protocol_deviation'],outcome=outcome,
            numerical_outcome=numerical_outcome,primary_metric=primary,original=before,reviewed=after,
            difference=difference,independent_metric_checks=independent,all_frozen_candidates=all_candidates,
            deterministic_development_selector=fixed,
            test_best_candidate_for_diagnostic_only=best['candidate'],test_best_metric=best['metrics'][primary],
            evidence=list({x['path']:x for x in evidence}.values()),source_manifest_sha256=receipt['manifest_sha256'],
            final_predictions_reused_from_verified_original_cold_runs=True,new_fits=0,new_final_inference=0)
        save(root/'comparison281'/(case['id']+'.json'),result);results.append(result)
    result=dict(utc=utc(),cases=results,counts=dict(Counter(x['outcome'] for x in results)),
        new_model_sessions=len(results),physical_allocations=len({c['source_archive'] for c in sealed['cases']}),new_independent_physical_cases=0,
        claims=dict(repeated_feedback_causality_confirmed=False,agent_specificity_confirmed=False,
                    population_frequency_estimated=False,unrestricted_review_difficulty_confirmed=False),
        final_scores_known_to_researcher=True,all_outcomes_retained=True,new_fits=0)
    save(root/'NUMERICAL_RESULTS281.json',result)
    save(PROJECT/config.get('result_directory','results/review281')/'NUMERICAL_RESULTS281.json',result)
    state(root,status='review_comparison_complete_waiting_interpretation_and_archive',review_outcomes=result['counts'],new_fits=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['seal','score']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();(seal if a.mode=='seal' else score)(a.run.resolve())
