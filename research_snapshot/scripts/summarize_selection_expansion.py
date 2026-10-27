"""Summarize sealed paired sessions and prespecified numerical controls.

Run only after choice audit and cold independent scoring of every cohort.
Numerical superiority never automatically establishes an Agent mechanism.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import terminal


def read(path):
    return json.loads(path.read_text())


def metrics(predictions,gold):
    indexed={str(r['sample_id']):r['predicted_label'] for r in predictions}
    assert len(indexed)==len(predictions)==len(gold)
    assert set(indexed)=={r['sample_id'] for r in gold}
    y=np.asarray([r['label'] for r in gold]);p=np.asarray([indexed[r['sample_id']] for r in gold])
    g=np.asarray([r['group'] for r in gold]);classes=sorted(set(y.tolist()))
    by_group=[dict(group=s,n=int(sum(g==s)),accuracy=float(np.mean(y[g==s]==p[g==s])),
        truth_classes=sorted(set(y[g==s].tolist()))) for s in sorted(set(g.tolist()))]
    return dict(accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,p)),
        mean_group_accuracy=float(np.mean([r['accuracy'] for r in by_group])),
        confusion_matrix=confusion_matrix(y,p,labels=classes).tolist(),classes=classes,groups=by_group)


def contrast(left,right,gold):
    a={r['sample_id']:r['predicted_label'] for r in left}
    b={r['sample_id']:r['predicted_label'] for r in right}
    groups={};fixed=broken=0
    for r in gold:
        before=a[r['sample_id']]==r['label'];after=b[r['sample_id']]==r['label']
        v=groups.setdefault(r['group'],dict(n=0,fixed=0,broken=0))
        v['n']+=1;v['fixed']+=int(after and not before);v['broken']+=int(before and not after)
        fixed+=int(after and not before);broken+=int(before and not after)
    ma,mb=metrics(left,gold),metrics(right,gold)
    return dict(delta={k:mb[k]-ma[k] for k in ['accuracy','macro_f1','balanced_accuracy','mean_group_accuracy']},
        fixed=fixed,broken=broken,groups=groups)


def main(root, paired=True):
    config=read(root/'CAMPAIGN_CONFIG.json');tag=config['iteration'].removeprefix('A')
    assert tag.isdigit() and not (root/('SUMMARY'+tag+'.json')).exists()
    recovery_path=root/('RECOVERY'+tag+'.json')
    recovery=read(recovery_path) if recovery_path.exists() else {}
    review_names=({s['case']:recovery.get('review_overrides',{}).get(s['case'],s['cohort']+'_review')
                  for s in config['cases']} if paired else {})
    cohorts={}
    for name in sorted(set(config['cohorts'])|set(review_names.values())):
        run=root/name
        jobs=terminal(run)
        assert read(run/'OBSERVABLE_REVIEW_COMPLETE.json')['reviewed_before_new_test_scores']
        score=read(run/'CODING_SCORE.json')
        cohorts[name]={r['dataset']:(job,r) for r in score['rows'] for job in jobs if job['id']==r['job']}
    results=[]
    for spec in config['cases']:
        case,name,cohort=spec['case'],spec['dataset'],spec['cohort']
        gold=read(root/cohort/'hidden'/name/'labels.json')
        stages={};predictions={}
        stage_runs=[('initial',cohort)]+([('review',review_names[case])] if paired else [])
        for phase,run_name in stage_runs:
            run=root/run_name;job,score=cohorts[run_name][name]
            stages[phase]=score
            out=run/'evaluation'/job['id']/'output'
            for kind,filename in [('final','predictions.json'),('reference','reference.json')]:
                if score[kind]['valid']:
                    rows=read(out/filename)['predictions']
                    score[kind]['group_metrics']=metrics(rows,gold)
                    predictions[phase if kind=='final' else phase+'_reference']=rows
        control=root/'controls'/case;frozen=read(control/'FROZEN_SELECTION.json')
        for file,digest in frozen['model_hashes'].items():assert sha256(control/file)==digest
        first_job=cohorts[cohort][name][0]
        data=root/cohort/'public_data'/first_job['id']/'datasets'/name
        test=dict(np.load(Path(first_job.get('evaluation_test',data/'test.npz')),allow_pickle=False))
        control_rows=[]
        for candidate in frozen['candidates']:
            for scope in config.get('fixed_controls',{}).get('fit_scopes',['train','train_dev']):
                file=control/(candidate['name']+'_'+scope+'.joblib')
                if not candidate['valid'] or not file.exists():
                    control_rows.append(dict(name=candidate['name'],scope=scope,valid=False));continue
                model=joblib.load(file);p=model.predict(test['features'].astype(float))
                rows=[dict(sample_id=str(i),predicted_label=y.item() if hasattr(y,'item') else y)
                    for i,y in zip(test['sample_ids'],p)]
                selected=candidate['name']==frozen['selected']
                control_rows.append(dict(name=candidate['name'],scope=scope,valid=True,
                    selected_from_development=selected,metrics=metrics(rows,gold)))
                save(control/(candidate['name']+'_'+scope+'_external_predictions.json'),dict(predictions=rows))
                if selected:predictions['fixed_'+scope]=rows
        comparisons={}
        for left,right in [('initial_reference','initial'),('initial_reference','review'),
            ('initial','review'),('initial','fixed_train'),('review','fixed_train'),
            ('initial','fixed_train_dev'),('review','fixed_train_dev')]:
            if left in predictions and right in predictions:
                comparisons[left+'_to_'+right]=contrast(predictions[left],predictions[right],gold)
        # These identities were registered from closed traces before scores.
        # Report every candidate, rather than choosing a test-best headline.
        candidate_rows=[]
        replay_names=[('initial',case)]+([('review',case+'_review')] if paired else [])
        for origin_stage, replay_name in replay_names:
            replay=root/'candidate_replays'/replay_name
            if not (replay/'REPLAY_STATUS.json').exists():continue
            replay_status=read(replay/'REPLAY_STATUS.json')
            assert replay_status['exit_code']==0
            assert replay_status['final_labels_read'] is False
            for filename,digest in sorted(replay_status['predictions'].items()):
                path=replay/'output'/filename
                assert sha256(path)==digest
                rows=read(path)['predictions']
                candidate_comparisons={phase:contrast(predictions[phase],rows,gold)
                    for phase in (['initial','review'] if origin_stage=='initial' else ['review'])
                    if phase in predictions}
                flags={}
                for phase,c in candidate_comparisons.items():
                    d=c['delta']
                    flags[phase]=((d['accuracy']>=.05-1e-12 and d['macro_f1']>=-1e-12)
                        or (d['macro_f1']>=.05-1e-12 and d['accuracy']>=-1e-12))
                candidate_rows.append(dict(candidate=filename[:-5],candidate_origin_stage=origin_stage,
                    metrics=metrics(rows,gold),
                    compared_to_submissions=candidate_comparisons,meets_registered_effect_threshold=flags,
                    candidate_provenance='See prescore registration; saved-weight replay and matched refit must be distinguished'))
        results.append(dict(case=case,dataset=name,arm=spec.get('arm'),repeat=spec.get('repeat'),
            test_rows=len(gold),test_groups=len({r['group'] for r in gold}),
            stages=stages,controls=control_rows,development_selected_control=frozen['selected'],comparisons=comparisons,
            registered_agent_candidates=candidate_rows,
            verified_agent_selection_loss=None,mechanism_status='Requires recorded-choice and executable-candidate audit; scores alone insufficient'))
    statuses=Counter(s['status'] for r in results for s in r['stages'].values())
    summary=dict(utc=datetime.now(timezone.utc).isoformat(),cases=results,statuses=dict(statuses),
        cases_planned=len(config['cases']),sources=config['source_count'],sessions=config['planned_sessions'],
        paired_initial_review_stages=paired,original_benchmark_score=False,
        native_attempts=recovery.get('maximum_native_attempts',config['planned_sessions']),
        infrastructure_interrupted_attempts=recovery.get('interrupted_native_attempts',0),
        recovery_context_restart_cases=[r['case'] for r in recovery.get('rows',[]) if r['context_restarted']],
        public_sources_previously_seen=True,unified_failure_mechanism_confirmed=False)
    save(root/('SUMMARY'+tag+'.json'),summary)
    target=Path(__file__).resolve().parents[1]/config['result_directory']/('NUMERICAL_RESULTS'+tag+'.json')
    save(target,summary)
    for row in results:
        print(json.dumps(dict(case=row['case'],stages={k:v['final'] for k,v in row['stages'].items()},
            fixed_selected=row['development_selected_control']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--unpaired',action='store_true',help='One independent stage per configured case; no review cohorts.')
    args=p.parse_args();main(args.run.resolve(),paired=not args.unpaired)
