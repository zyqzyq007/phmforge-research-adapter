"""Seal real ensemble selections and all ordinary reviews before cold scoring.

No estimator fitting or automatic weight search occurs here. Every scored vector
is an actual basis fallback or a registered candidate. Final-best comparisons
are post-hoc diagnostics, never feedback to a live Agent.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np

from controlled_feedback import save,sha,utc
from evaluate_controlled_feedback import cold,independent
from evaluate_forge_coding import terminal,files

PROJECT=Path(__file__).resolve().parents[1]


def matches_submission(path,choice):
    if not path.exists() or choice is None:return False
    try:
        value=json.loads(path.read_text())
        if 'result' in value:value=value['result']
        if 'final_choice' in value:value=value['final_choice']
        return value['candidate_id']==choice['candidate_id'] and value['weights']==choice['weights']
    except (ValueError,KeyError,TypeError):return False


def seal_phase(root):
    assert not (root/'PHASE_SEALED.json').exists(),'No implicit resealing'
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert config['search_mode']=='ensemble_weights'
    frozen=[];rows=[];checks=[]
    for case in config['cases']:
        run=root/case['id'];jobs=terminal(run);assert len(jobs)==1
        job=jobs[0];ws=run/job['workspace']
        status=json.loads((run/'primary'/job['id']/job['id']/'status.json').read_text())
        broker=json.loads((run/'broker/STATE.json').read_text())
        private=json.loads((run/'BROKER_CONFIG.json').read_text())
        assert broker['stage']==config.get('search_stage','search')
        assert sha(private['bank'])==sha(root/'bank/probabilities.npz')
        assert len({tuple(t['weights']) for t in broker['trials']})==len(broker['trials'])
        assert len({t['id'] for t in broker['trials']})==len(broker['trials'])
        new=sum(t['kind']=='ensemble' for t in broker['trials'])
        assert 0<=new<=private['candidate_budget']
        if broker['final_choice'] and broker['stage']=='search':assert new==private['candidate_budget']
        choice=broker['final_choice']
        if choice:
            trial=next(t for t in broker['trials'] if t['id']==choice['candidate_id'])
            assert choice['weights']==trial['weights'] and choice['bank_sha256']==sha(private['bank'])
        delivered=matches_submission(ws/'selection.json',choice) and (ws/'report.md').exists()
        access=run/'broker/ACCESS.jsonl';registrations=[];early_dev_requests=[];previous_count=0
        for line in access.read_text().splitlines():
            event=json.loads(line);response=event['response']
            if not response['ok']:continue
            request=json.loads(event['request'])
            value=response['result'];n=value['ensemble_candidates']
            released=broker['stage']=='review' or case['condition']=='immediate' or n==private['candidate_budget']
            raw=broker['stage']=='review' or n==private['candidate_budget']
            assert value['development_released']==released
            assert bool(value['all_development_probabilities'])==raw
            checks.append(dict(case=case['id'],count=n,passed=True))
            if request['action']=='evaluate':
                assert n==previous_count+len(request['vectors'])
                registrations.append(dict(request_id=event['request_id'],started_utc=event['started_utc'],
                    ended_utc=event['ended_utc'],vectors=len(request['vectors']),previous_count=previous_count,new_count=n,
                    earlier_development_feedback_available=broker['stage']=='review' or case['condition']=='immediate'))
            if request['action']=='evidence' and request['split']=='development' and n<private['candidate_budget']:
                early_dev_requests.append(dict(candidate_id=request['candidate_id'],at_count=n,utc=event['started_utc']))
            previous_count=n
        pool=dict(source_kind='Agent_session',source_phase_terminal=True,candidate_pool_complete=True,
            bank_sha256=sha(private['bank']),classes=config['classes'],basis_names=config['recipe_space']['basis_names'],
            weight_denominator=12,trials=broker['trials'],final_choice=choice,delivered=delivered,status=status['status'])
        save(run/'CANDIDATE_POOL.json',pool)
        inventory=files(ws);save(run/'audit/ENSEMBLE_SUBMISSION_MANIFEST.json',inventory)
        paths=[run/'CANDIDATE_POOL.json',run/'broker/STATE.json',access,run/'BROKER_CONFIG.json',
            run/'primary'/job['id']/job['id']/'status.json',run/'primary'/job['id']/job['id']/'events.jsonl',
            run/'INITIAL_FREEZE187_FIFO_FILES.json',run/'PROTOCOL187.json',run/'PLAN187.json']
        paths += [p for p in (run/'references').rglob('*') if p.is_file() and not p.is_symlink()]
        paths += [ws/x['path'] for x in inventory if 'sha256' in x]
        for p in paths:frozen.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
        rows.append(dict(case,status=status['status'],delivered=delivered,selection=choice,new_vectors=new,
            original_selection=broker.get('prior_choice'),trials=broker['trials'],
            feedback_trace=dict(registration_batches=registrations,early_development_evidence_requests=early_dev_requests,
                interpretation='Availability and explicit evidence requests only. Actual use to change a later vector requires trajectory verification;never inferred from availability alone.')))
    bindings=json.loads((root/'audit/SOURCE_BINDINGS.json').read_text())['files']
    for row in bindings:
        assert sha(root/row['path'])==row['sha256'];frozen.append(dict(path=row['path'],sha256=row['sha256']))
    for p in [root/'CAMPAIGN_CONFIG.json',*sorted((root/'shared_tools').glob('*.py')),*sorted((root/'code').glob('*.py'))]:
        frozen.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
    result=dict(utc=utc(),stage=config.get('search_stage','search'),cases=rows,files=frozen,
        feedback_policy_checks=checks,final_scores_opened=False,all_native_processes_terminal=True)
    save(root/'PHASE_SEALED.json',result)
    return result


def verify_seal(root,sealed):
    for row in sealed['files']:assert sha(root/row['path'])==row['sha256'],row['path']


def prepare_review(root):
    from prepare_feedback_campaign import prepare_bound_source
    sealed=json.loads((root/'PHASE_SEALED.json').read_text());verify_seal(root,sealed)
    assert sealed['stage']=='search' and sealed['final_scores_opened'] is False
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    spec=dict(config,search_stage='review',parent_campaign=str(root),
        candidate_budget_per_case=config['review_candidate_budget'],candidate_fits_per_case=config['review_candidate_budget'],
        purpose='Ordinary full-evidence review before final scoring;retain or register same-space new weights.',
        result_directory=config['result_directory']+'/review')
    # Rebind original immutable source files; hard links are verified against
    # that manifest. Agent workspaces and model channels remain case-private.
    path=root/'audit/REVIEW_SPEC.json';save(path,spec)
    review=root/'review';prepare_bound_source(review,path)
    return review


def seal_all(root):
    assert not (root/'ALL_CHOICES_SEALED.json').exists()
    phase=[]
    for folder in [root,root/'review']:
        p=folder/'PHASE_SEALED.json';sealed=json.loads(p.read_text());verify_seal(folder,sealed)
        for case in sealed['cases']:terminal(folder/case['id'])
        phase.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
    value=dict(utc=utc(),phases=phase,final_scores_opened=False,all_initial_and_review_sessions_terminal=True)
    save(root/'ALL_CHOICES_SEALED.json',value)
    return value


def selected(rows,choice):
    return next((x for x in rows if choice and x['id']==choice['candidate_id']),None)


def evaluate(root):
    assert not (root/'EXTERNAL_SCORING_STARTED.json').exists(),'No implicit repeated scoring'
    seal=json.loads((root/'ALL_CHOICES_SEALED.json').read_text())
    assert seal['all_initial_and_review_sessions_terminal'] and seal['final_scores_opened'] is False
    for p in seal['phases']:assert sha(root/p['path'])==p['sha256']
    phases=[json.loads((folder/'PHASE_SEALED.json').read_text()) for folder in [root,root/'review']]
    for folder,p in zip([root,root/'review'],phases,strict=True):verify_seal(folder,p)
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text());classes=config['classes']
    bank=dict(np.load(root/'bank/probabilities.npz',allow_pickle=False))
    proof=json.loads((root/'BASIS_QUALIFICATION.json').read_text())
    assert proof['passed'] and sha(root/'bank/probabilities.npz')==proof['bank_sha256']
    assert [x['name'] for x in proof['models']]==bank['basis_names'].tolist()
    for model in proof['models']:assert sha(root/model['directory']/'checkpoint.joblib')==model['checkpoint_sha256']
    # Actual training predictions were saved during source-qualified fitting.
    # Retain them separately from out-of-fold CV: a high training score alone
    # must not be confused with adaptive reuse of development feedback.
    training=[]
    for model in proof['models']:
        with np.load(root/model['directory']/'probabilities.npz',allow_pickle=False) as saved:
            assert saved['classes'].tolist()==classes
            assert np.array_equal(saved['train_ids'],bank['cv_ids'])
            values=saved['train'];assert values.shape==(len(bank['cv_labels']),len(classes))
            assert np.isfinite(values).all() and values.min()>=0 and values.max()<=1
            assert np.allclose(values.sum(axis=1),1,rtol=0,atol=1e-8)
            training.append(values)
    training_probabilities=np.stack(training,axis=1)
    save(root/'EXTERNAL_SCORING_STARTED.json',dict(utc=utc(),choices_sha256=sha(root/'ALL_CHOICES_SEALED.json'),
        no_more_Agent_sessions_or_weight_search=True,only_actual_registered_vectors_scored=True))
    data={part:dict(np.load(root/'hidden'/(part+'.npz'),allow_pickle=False)) for part in ['val','test']}
    for part,d in data.items():
        folder=root/'evaluation_inputs'/part;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'observations.npz',waveforms=d['waveforms'],sample_ids=d['sample_ids'])
    outputs={};jobs=[(model,part) for model in proof['models'] for part in ['val','test']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[(m,p,pool.submit(cold,root,'basis',m['name'],p,
            checkpoint=root/m['directory']/'checkpoint.joblib',probability_output=True)) for m,p in jobs]
        for model,part,future in futures:outputs[(model['name'],part)]=future.result()
    probabilities={};cold_checks=[]
    for part,d in data.items():
        arrays=[]
        for model in proof['models']:
            out=outputs[(model['name'],part)];assert out['classes']==classes
            assert len(out['sample_ids'])==len(set(out['sample_ids']))==len(d['sample_ids'])
            positions={str(key):i for i,key in enumerate(out['sample_ids'])}
            assert set(positions)==set(d['sample_ids'].tolist())
            a=np.asarray(out['probabilities'])[[positions[str(key)] for key in d['sample_ids']]]
            assert a.shape==(len(d['sample_ids']),len(classes)) and np.isfinite(a).all()
            assert a.min()>=0 and a.max()<=1 and np.allclose(a.sum(1),1,rtol=0,atol=1e-8)
            arrays.append(a);cold_checks.append(dict(model=model['name'],split=part,passed=True))
        probabilities[part]=np.stack(arrays,axis=1)
    assert np.array_equal(data['val']['sample_ids'],bank['development_ids'])
    assert np.array_equal(data['val']['labels'],bank['development_labels'])
    assert np.allclose(probabilities['val'],bank['development'],rtol=0,atol=1e-12)
    rows=[];checks=[]
    for phase in phases:
        for case in phase['cases']:
            candidates=[]
            for trial in case['trials']:
                w=np.asarray(trial['weights'],dtype=np.float64)
                values={}
                for part in ['train','cv','development','final']:
                    a=training_probabilities if part=='train' else bank[part] if part!='final' else probabilities['test']
                    y=bank['cv_labels'] if part=='train' else bank[part+'_labels'] if part!='final' else data['test']['labels']
                    p=np.asarray(classes)[(np.einsum('nbc,b->nc',a,w)/12).argmax(axis=1)]
                    value=independent(y,p,classes)
                    if part in ['cv','development']:
                        assert value['confusion']==trial[part]['confusion']
                        assert abs(value['macro_f1']-trial[part]['macro_f1'])<1e-12
                    keys=bank.get('cv_conditions' if part=='train' else part+'_conditions') if part!='final' else data['test'].get('condition_id')
                    if keys is not None:value['by_condition']={str(c):independent(y[keys==c],p[keys==c],classes) for c in np.unique(keys)}
                    values[part]=value
                    checks.append(dict(stage=phase['stage'],case=case['id'],candidate=trial['id'],split=part,passed=True))
                candidates.append(dict(id=trial['id'],kind=trial['kind'],weights=trial['weights'],**values))
            chosen=selected(candidates,case['selection'])
            best=max(candidates,key=lambda x:(x['final']['macro_f1'],x['final']['accuracy']))
            fixed={split:max(enumerate(candidates),key=lambda x:(x[1][split]['macro_f1'],x[1][split]['accuracy'],-x[0]))[1]['id'] for split in ['cv','development']}
            fixed_outcomes={split:next(x for x in candidates if x['id']==ident) for split,ident in fixed.items()}
            regret=best['final']['macro_f1']-chosen['final']['macro_f1'] if chosen else None
            rows.append(dict(stage=phase['stage'],**{k:case[k] for k in ['id','condition','replicate','status','delivered','selection','new_vectors','feedback_trace']},
                chosen=chosen,candidates=candidates,posthoc_actual_pool_best=best['id'],macro_f1_regret=regret,
                major_actual_candidate_loss=bool(chosen and regret>=.05 and best['final']['accuracy']>=chosen['final']['accuracy']),
                fixed_selectors_same_actual_pool=fixed,
                fixed_selector_outcomes={split:dict(candidate_id=x['id'],final=x['final'],
                    Agent_minus_fixed_final_macro_f1=chosen['final']['macro_f1']-x['final']['macro_f1'] if chosen else None) for split,x in fixed_outcomes.items()}))
    groups={};effects={};pairs=[]
    for stage in ['search','review']:
        groups[stage]={}
        for arm in ['immediate','delayed']:
            planned=[r for r in rows if r['stage']==stage and r['condition']==arm]
            valid=[r for r in planned if r['delivered'] and r['chosen']]
            group=dict(planned=len(planned),delivered=len(valid),native_status_counts={s:sum(r['status']==s for r in planned) for s in {r['status'] for r in planned}})
            for metric,values in [('final_macro_f1',[r['chosen']['final']['macro_f1'] for r in valid]),
                ('training_macro_f1',[r['chosen']['train']['macro_f1'] for r in valid]),
                ('cv_macro_f1',[r['chosen']['cv']['macro_f1'] for r in valid]),
                ('final_accuracy',[r['chosen']['final']['accuracy'] for r in valid]),
                ('development_macro_f1',[r['chosen']['development']['macro_f1'] for r in valid]),
                ('optimism_gap',[r['chosen']['development']['macro_f1']-r['chosen']['final']['macro_f1'] for r in valid])]:
                group['mean_'+metric]=float(np.mean(values)) if values else None
            groups[stage][arm]=group
        a,b=groups[stage]['immediate'],groups[stage]['delayed']
        effects[stage]={k:a[k]-b[k] for k in a if k.startswith('mean_')} if all(g['delivered']==g['planned'] for g in [a,b]) else None
    for initial in [r for r in rows if r['stage']=='search']:
        review=next(r for r in rows if r['stage']=='review' and r['id']==initial['id'])
        pairs.append(dict(case=initial['id'],initial_delivered=initial['delivered'],review_delivered=review['delivered'],
            final_macro_f1_change=review['chosen']['final']['macro_f1']-initial['chosen']['final']['macro_f1'] if initial['chosen'] and review['chosen'] else None,
            initial_major_loss=initial['major_actual_candidate_loss'],review_major_loss=review['major_actual_candidate_loss']))
    for folder,p in zip([root,root/'review'],phases,strict=True):verify_seal(folder,p)
    result=dict(utc=utc(),cases=rows,groups=groups,immediate_minus_delayed=effects,ordinary_review_pairs=pairs,
        independent_metric_checks=checks,cold_probability_checks=cold_checks,cold_processes=len(jobs),
        basis_wrapper_fits=24,new_estimator_fits_during_Agent=0,
        training_metrics_source='Immutable actual full-training basis probabilities,independently checked during real source fitting;distinct from out-of-fold CV.',
        final_scores_opened=True,goal_status='active',
        mechanism_confirmed=False,interpretation='Descriptive timing contrast on one gear and one fixed split;not an automatic causal/population conclusion.',limitations=config['limitations'])
    save(root/'ENSEMBLE_RESULTS.json',result);save(PROJECT/config['result_directory']/'ENSEMBLE_RESULTS.json',result)
    print(json.dumps(dict(groups=groups,immediate_minus_delayed=effects,ordinary_review_pairs=pairs,cold_processes=len(jobs)),indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--mode',choices=['seal-phase','seal-all','evaluate'],required=True);a=parser.parse_args()
    {'seal-phase':seal_phase,'seal-all':seal_all,'evaluate':evaluate}[a.mode](a.run.resolve())
