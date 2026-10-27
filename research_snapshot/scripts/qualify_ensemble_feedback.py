"""Qualify cached probability feedback and real complete-program ensembles.

The historical modes replay genuine probability arrays. Complete-program modes
add isolated cold inference. Neither mode fits models,searches weights,reads
new final data,or reports a new Agent outcome.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from controlled_ensemble_feedback import EnsembleBroker
from controlled_feedback import save,sha,utc


NAMES=['et_band_both_sqrt','et_bands_only_shape','et_base_sqrt',
       'et_points_both_01','et_points_shape_sqrt','svc_band_both']


def independent_metrics(y,p,classes):
    table=np.zeros((len(classes),len(classes)),dtype=np.int64)
    positions={v:i for i,v in enumerate(classes)}
    for a,b in zip(y,p,strict=True):table[positions[a],positions[b]]+=1
    denominator=table.sum(axis=0)+table.sum(axis=1)
    f1=np.divide(2*np.diag(table),denominator,out=np.zeros(len(classes)),where=denominator!=0)
    return dict(accuracy=float(np.trace(table)/table.sum()),macro_f1=float(f1.mean()),confusion_matrix=table.tolist())


def qualify(root):
    cfg=json.loads((root/'PROTOCOL290.json').read_text());source=Path(cfg['source_archive'])
    assert cfg['new_fits']==cfg['new_LLM_calls']==cfg['new_final_scores']==0
    manifest=source/'SOURCE_MANIFEST.json';rows={x['path']:x for x in json.loads(manifest.read_text())['files']}
    probabilities='rotor_c/workspaces/s284_rotor_c/historical_validation/candidate_probabilities.npz'
    references=[probabilities]+['rotor_c/public_data/datasets/ROTORBROKENBAR/'+s+'.npz' for s in ['train','val']]
    bindings=[]
    for relative in references:
        path=source/relative;row=rows[relative]
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
        bindings.append(dict(path=str(path),bytes=row['bytes'],sha256=row['sha256']))
    fields={}
    with np.load(source/probabilities,allow_pickle=False) as old:
        for part,split,prefix,labelkey in [('cv','train','oof__','y'),('development','val','dev__','y_dev')]:
            with np.load(source/('rotor_c/public_data/datasets/ROTORBROKENBAR/'+split+'.npz'),allow_pickle=False) as data:
                assert np.array_equal(data['labels'],old[labelkey])
                fields[part]=np.stack([old[prefix+n] for n in NAMES],axis=1)
                for suffix,key in [('labels','labels'),('ids','sample_ids'),('conditions','operating_load_Nm'),('groups','groups')]:
                    fields[part+'_'+suffix]=data[key]
    fields.update(classes=np.unique(fields['cv_labels']),basis_names=np.asarray(NAMES))
    bank=root/'inputs/historical_probability_bank.npz'
    with bank.open('xb') as f:np.savez_compressed(f,**fields)
    save(root/'audit/SOURCE_BINDINGS290.json',dict(utc=utc(),source_manifest_sha256=sha(manifest),files=bindings,
        bank_sha256=sha(bank),real_historical_predictions_only=True,final_data_read=False))
    checks={};manual=[];brokers={}
    for condition in ['immediate','delayed']:
        run=root/condition;run.mkdir(exist_ok=False)
        save(run/'BROKER_CONFIG.json',dict(condition=condition,bank=str(bank),candidate_budget=1,weight_denominator=12,
             request_directory='app/.runtime/feedback_requests'))
        b=EnsembleBroker(run);brokers[condition]=b
        catalog=json.loads((run/'feedback/catalog.json').read_text())
        checks[condition+'_six_real_fallbacks']=len(catalog['candidates'])==6
        checks[condition+'_initial_feedback_timing']=all(isinstance(x['development'],dict)==(condition=='immediate') for x in catalog['candidates'])
        checks[condition+'_raw_dev_initially_absent']=not (run/'feedback/development_probabilities.npz').exists()
        cv=np.load(run/'feedback/training_cv_probabilities.npz',allow_pickle=False)
        checks[condition+'_public_cv_no_dev_fields']=not any('development' in k for k in cv.files)
        invalid=[dict(action='evaluate',vectors=[[True,1,2,2,2,5]]),
                 dict(action='evaluate',vectors=[[2.0]*6]),dict(action='evaluate',vectors=[[-1,1,2,2,2,6]]),
                 dict(action='evaluate',vectors=[[1]*6]),dict(action='evaluate',vectors=[[2]*6,[2]*6]),
                 dict(action='evaluate',vectors=[[12,0,0,0,0,0]]),dict(action='read_file',path=str(bank)),
                 dict(action='status',path=str(bank)),dict(action='select',candidate_id='base_1',reason='early')]
        if condition=='delayed':invalid.append(dict(action='evidence',candidate_id='base_1',split='development'))
        rejected=[]
        for request in invalid:
            before=json.dumps(b.state,sort_keys=True)
            try:b.dispatch(request);accepted=True
            except (AssertionError,ValueError,StopIteration):accepted=False
            rejected.append(not accepted and before==json.dumps(b.state,sort_keys=True))
        checks[condition+'_invalid_requests_rejected_without_mutation']=all(rejected)
        if condition=='immediate':
            b.dispatch(dict(action='evidence',candidate_id='base_1',split='development'))
            evidence=json.loads((run/'feedback/evidence/base_1/development.json').read_text())
            checks['early_evidence_no_soft_probabilities']=all(set(x)=={'sample_id','actual','predicted','conditions','groups'} for x in evidence['rows'])
        b.dispatch(dict(action='evaluate',vectors=[[2]*6]))
        checks[condition+'_budget_counted']=b.count()==1 and len(b.state['trials'])==7
        checks[condition+'_all_dev_released']=all(isinstance(x['development'],dict) for x in json.loads((run/'feedback/catalog.json').read_text())['candidates'])
        with np.load(run/'feedback/development_probabilities.npz',allow_pickle=False) as public:
            checks[condition+'_dev_bundle_actual']=np.array_equal(public['probabilities'],fields['development']) and np.array_equal(public['development_ids'],fields['development_ids'])
        for row in b.state['trials']:
            for part in ['cv','development']:
                # Independent aggregation route, separate confusion-table arithmetic.
                prediction=fields['classes'][(fields[part]*np.asarray(row['weights'])[None,:,None]).sum(axis=1).argmax(axis=1)]
                expected=independent_metrics(fields[part+'_labels'],prediction,fields['classes'].tolist())
                b.dispatch(dict(action='evidence',candidate_id=row['id'],split=part))
                evidence=json.loads((run/'feedback/evidence'/row['id']/(part+'.json')).read_text())
                assert np.array_equal(prediction,[x['predicted'] for x in evidence['rows']])
                ok=all(np.isclose(expected[k],row[part][k],rtol=0,atol=1e-12) for k in ['accuracy','macro_f1'])
                manual.append(dict(condition=condition,candidate=row['id'],split=part,passed=bool(ok),metrics=expected))
        checks[condition+'_original_equal_cv']=abs(b.state['trials'][-1]['cv']['macro_f1']-.9787682283473469)<1e-12
        checks[condition+'_original_equal_dev']=abs(b.state['trials'][-1]['development']['macro_f1']-.9837087498457361)<1e-12
        # Predetermined fixture selection: this is not an Agent decision.
        b.dispatch(dict(action='select',candidate_id='ensemble_1',reason='Fixed historical arithmetic fixture; no new model choice.'))
        final_before=json.dumps(b.state,sort_keys=True);denied=[]
        for req in [dict(action='evaluate',vectors=[[2]*6]),dict(action='select',candidate_id='base_1',reason='late')]:
            try:b.dispatch(req);denied.append(False)
            except AssertionError:denied.append(final_before==json.dumps(b.state,sort_keys=True))
        checks[condition+'_sealed_once']=all(denied)
    checks['both_conditions_identical_candidate_scores']=all(
        a['weights']==b['weights'] and a['cv']==b['cv'] and a['development']==b['development']
        for a,b in zip(brokers['immediate'].state['trials'],brokers['delayed'].state['trials'],strict=True))
    checks['independent_metric_recomputations']=all(x['passed'] for x in manual)
    for name in ['controlled_feedback.py','controlled_ensemble_feedback.py','qualify_ensemble_feedback.py']:
        src=Path(__file__).with_name(name);dst=root/'code'/name
        if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    save(root/'audit/INDEPENDENT_RECOMPUTATIONS290.json',manual)
    result=dict(passed=all(checks.values()),checks=checks,utc=utc(),new_fits=0,new_LLM_calls=0,new_model_inference=0,
        new_final_scores=0,new_Agent_cases=0,historical_component_and_equal_vectors_only=True,
        native_cli_qualified=False,numerical_feedback_contract_qualified=all(checks.values()))
    save(root/'NUMERICAL_GATE290.json',result);print(json.dumps(result));assert result['passed']


def qualify_review(root):
    """Use only the seven genuine historical vectors;controls are not Agents."""
    protocol=json.loads((root/'PROTOCOL294.json').read_text());source=Path(protocol['source_archive'])
    manifest=source/'SOURCE_MANIFEST.json';assert sha(manifest)==protocol['source_manifest_sha256']
    index={x['path']:x for x in json.loads(manifest.read_text())['files']};bindings=[]
    for relative in ['inputs/historical_probability_bank.npz','immediate/broker/STATE.json']:
        path=source/relative;row=index[relative]
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
        bindings.append(dict(path=str(path),bytes=row['bytes'],sha256=row['sha256']))
    bank=root/'inputs/historical_probability_bank.npz';shutil.copy2(source/'inputs/historical_probability_bank.npz',bank)
    original=json.loads((source/'immediate/broker/STATE.json').read_text())
    allowed={tuple(x['weights']) for x in original['trials']};assert len(allowed)==7
    values=np.load(bank,allow_pickle=False);checks={};brokers=[];search={}
    def rejected(b,request):
        before=json.dumps(b.state,sort_keys=True)
        try:b.dispatch(request);return False
        except (AssertionError,ValueError,StopIteration):return before==json.dumps(b.state,sort_keys=True)
    def make(name,condition,stage='search',prior=None):
        folder=root/name;folder.mkdir(exist_ok=False)
        config=dict(condition=condition,bank=str(bank),candidate_budget=6 if stage=='review' else 1,
                    weight_denominator=12,stage=stage,request_directory='app/.runtime/feedback_requests')
        if prior is not None:
            save(folder/'references/prior_pool.json',prior);config['prior_pool']='references/prior_pool.json'
        save(folder/'BROKER_CONFIG.json',config);broker=EnsembleBroker(folder);brokers.append(broker);return broker
    for condition in ['immediate','delayed']:
        b=make('search_'+condition,condition);search[condition]=b
        rows=json.loads((b.run/'feedback/catalog.json').read_text())['candidates']
        checks[condition+'_original_metric_timing']=all(isinstance(x['development'],dict)==(condition=='immediate') for x in rows)
        checks[condition+'_raw_development_initially_withheld']=not (b.run/'feedback/development_probabilities.npz').exists()
        checks[condition+'_early_choice_denied']=rejected(b,dict(action='select',candidate_id='base_3',reason='early fixture'))
        checks[condition+'_API_stage_escape_denied']=rejected(b,dict(action='status',stage='review'))
        b.dispatch(dict(action='evaluate',vectors=[[2]*6]))
        checks[condition+'_release_at_exact_budget']=(b.run/'feedback/development_probabilities.npz').exists() and b.count()==1
        b.dispatch(dict(action='select',candidate_id='base_3',reason='Predetermined numeric fixture;not an Agent decision.'))
    def prior_pool(trials,choice):
        return dict(source_kind='numeric_fixture',source_phase_terminal=True,candidate_pool_complete=True,
            source_description='Predetermined control metadata over actual archived components;not a claimed Agent trajectory.',
            bank_sha256=sha(bank),classes=values['classes'].tolist(),basis_names=values['basis_names'].tolist(),
            weight_denominator=12,trials=trials,final_choice=choice)
    selected=dict(candidate_id='base_3',weights=[0,0,12,0,0,0],weight_denominator=12,bank_sha256=sha(bank),
                  reason='Predetermined component fixture,not the original Agent selection.')
    component_pool=prior_pool([x for x in original['trials'] if x['kind']=='basis_anchor'],selected)
    save(root/'inputs/COMPONENT_FIXTURE_POOL294.json',component_pool)
    b=make('review_retain','delayed','review',component_pool)
    checks['review_raw_development_available_before_new_candidate']=(b.run/'feedback/development_probabilities.npz').exists()
    checks['review_zero_new_candidates_initially']=b.count()==0 and len(b.state['trials'])==6
    b.dispatch(dict(action='evidence',candidate_id='base_3',split='development'))
    b.dispatch(dict(action='select',candidate_id='base_3',reason='Retain predetermined component fixture without forced queries.'))
    checks['retain_without_new_candidate_or_budget_exhaustion']=b.count()==0 and b.state['final_choice']['candidate_id']=='base_3'
    checks['sealed_review_cannot_change']=rejected(b,dict(action='evaluate',vectors=[[2]*6]))
    b=make('review_add','immediate','review',component_pool)
    checks['prior_component_duplicate_denied']=rejected(b,dict(action='evaluate',vectors=[[0,0,12,0,0,0]]))
    b.dispatch(dict(action='evaluate',vectors=[[2]*6]))
    expected=next(x for x in original['trials'] if tuple(x['weights'])==(2,)*6)
    checks['new_registration_outside_prior_pool_allowed']=b.count()==1 and b.state['trials'][-1]['weights']==[2]*6
    checks['actual_historical_scores_equal']=all(b.state['trials'][-1][p]==expected[p] for p in ['cv','development'])
    b.dispatch(dict(action='select',candidate_id='ensemble_1',reason='Predetermined historical equal-vector fixture.'))
    checks['choose_new_weight_before_optional_budget_exhaustion']=b.state['final_choice']['weights']==[2]*6 and b.count()<b.budget
    complete=prior_pool(search['delayed'].state['trials'],search['delayed'].state['final_choice'])
    b=make('review_inherited','delayed','review',complete)
    checks['all_original_candidates_preserved']=len(b.state['trials'])==7 and b.count()==0
    inherited=next(x for x in b.state['trials'] if x['kind']=='prior_ensemble')
    checks['inherited_scores_recomputed']=all(inherited[p]==expected[p] for p in ['cv','development'])
    b.dispatch(dict(action='select',candidate_id=inherited['id'],reason='Retain an imported historical ensemble fixture.'))
    checks['imported_ensemble_selectable_without_queries']=b.state['final_choice']['weights']==[2]*6 and b.count()==0
    no_delivery=prior_pool(complete['trials'],None)
    no_delivery['delivered']=False
    b=make('review_missing_initial_delivery','immediate','review',no_delivery)
    checks['missing_initial_choice_not_fabricated']=b.state['prior_choice'] is None and b.state['prior_delivery'] is False
    b.dispatch(dict(action='select',candidate_id='base_3',reason='Predetermined recovery fixture;original non-delivery remains recorded.'))
    checks['missing_initial_delivery_pool_reviewable']=len(b.state['trials'])==7 and b.count()==0 and b.state['final_choice']['candidate_id']=='base_3'
    invalid=[]
    for label,stage,pool in [('stage_missing_prior','review',None),('search_cannot_import','search',component_pool),
                             ('unclosed_prior','review',dict(component_pool,source_phase_terminal=False)),
                             ('bank_mismatch','review',dict(component_pool,bank_sha256='0'*64))]:
        folder=root/label;folder.mkdir()
        config=dict(condition='delayed',stage=stage,bank=str(bank),candidate_budget=6,weight_denominator=12)
        if pool is not None:
            save(folder/'references/prior_pool.json',pool);config['prior_pool']='references/prior_pool.json'
        save(folder/'BROKER_CONFIG.json',config)
        try:EnsembleBroker(folder);denied=False
        except AssertionError:denied=True
        invalid.append(denied and not (folder/'broker/STATE.json').exists() and not (folder/'feedback/development_probabilities.npz').exists())
    checks['invalid_stage_or_prior_denied_before_publication']=all(invalid)
    checks['only_seven_preexisting_actual_vectors_recomputed']=all(tuple(x['weights']) in allowed for b in brokers for x in b.state['trials'])
    for b in brokers:
        with np.load(b.run/'feedback/development_probabilities.npz',allow_pickle=False) as public:
            assert np.array_equal(public['probabilities'],values['development'])
    checks['published_development_arrays_are_genuine']=True
    save(root/'audit/SOURCE_BINDINGS294.json',dict(source_manifest_sha256=sha(manifest),files=bindings))
    for name in ['controlled_feedback.py','controlled_ensemble_feedback.py','qualify_ensemble_feedback.py','qualify_feedback_native.py','isolated_model_gateway.py']:
        src=Path(__file__).with_name(name);dst=root/'code'/name
        if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    result=dict(passed=all(checks.values()),checks=checks,utc=utc(),new_fits=0,new_LLM_calls=0,new_model_inference=0,
        new_final_scores=0,new_Agent_cases=0,new_weight_vectors_searched=0,
        scope='Source-verified old probabilities,seven previously used weights,and explicit numeric control states;not Agent behavior or new PHM outcomes.')
    save(root/'NUMERICAL_REVIEW_GATE294.json',result);print(json.dumps(result));assert result['passed']


def prepare_complete_programs(root):
    """Freeze a genuine complete-program inference qualification,not a search."""
    import hashlib
    project=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010');archives=Path('/data/yyf-zyq/PHM_20261010')
    if root.exists():
        assert {p.name for p in root.iterdir()}=={'runtime'} and not any(p.is_file() for p in root.rglob('*'))
    else:root.mkdir()
    for name in ['runtime/tmp','runtime/mpl','code','common_tools','assets','private_inputs','audit']:(root/name).mkdir(parents=True,exist_ok=True)
    bindings=[];indexes={}
    def bind(archive,relative,destination=None):
        if str(archive) not in indexes:
            receipt=json.loads((archive/'ARCHIVE_VERIFIED.json').read_text());assert receipt['source_cleanup']=='complete'
            assert sha(archive/'SOURCE_MANIFEST.json')==receipt['manifest_sha256']
            indexes[str(archive)]={x['path']:x for x in json.loads((archive/'SOURCE_MANIFEST.json').read_text())['files']}
        path=archive/relative;entry=indexes[str(archive)][relative]
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
        bindings.append(dict(archive=str(archive),path=relative,sha256=entry['sha256'],bytes=entry['bytes']))
        if destination is not None:destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        return path
    def read_bound(archive,relative):return json.loads(bind(archive,relative).read_text())
    archive397=archives/'20261025T230000Z_ensemble397';cache=read_bound(archive397,'RESULT397.json')
    assert read_bound(archive397,'INDEPENDENT_AUDIT397.json')['passed']
    source=archives/'20261025T012000Z_source346r1';q=read_bound(source,'QUALIFICATION346.json')
    bind(source,'qualification/development.npz',root/'private_inputs/development.npz')
    assert sha(root/'private_inputs/development.npz')==q['files']['development.npz']['sha256']
    selected=[('A396','20261025T215000Z_submit396','ordinary_coding_1'),
              ('A396','20261025T215000Z_submit396','submission_review_1'),
              ('A347','20261025T012600Z_local347','local_revision_1')]
    pools=[]
    for batch,name,subject in selected:
        archive=archives/name;state=read_bound(archive,'subjects/'+subject+'/broker/STATE.json');members={};reports={}
        for ident,entry in state['programs'].items():
            parts=Path(entry['path']).parts;relative=str(Path(*parts[parts.index(name)+1:]))
            report=read_bound(archive,relative+'/REPORT.json')
            required={k:v for k,v in report['artifacts'].items() if k.startswith(('source/','full_fit/artifacts/'))}
            assert any(k.startswith('source/') for k in required) and any(k.startswith('full_fit/artifacts/') for k in required)
            signature=hashlib.sha256(json.dumps(required,sort_keys=True).encode()).hexdigest();asset=root/'assets'/signature
            for member,digest in required.items():
                file=bind(archive,relative+'/'+member,None if (asset/member).exists() else asset/member)
                assert sha(file)==digest and sha(asset/member)==digest
            members[ident]=dict(source=str(asset/'source'),artifacts=str(asset/'full_fit/artifacts'),signature=signature)
            reports[ident]=report
        historical=next(p for p in cache['pools'] if p['batch']==batch and p['subject']==subject)
        assert set(historical['members'])==set(members)
        pools.append(dict(id=batch+'_'+subject,batch=batch,subject=subject,members=members,reports=reports,
                          historical_development={k:v['dev'] for k,v in historical['members'].items()}))
    source396=archives/'20261025T215000Z_submit396'
    for file in sorted((source396/'common_tools').glob('*.py')):bind(source396,str(file.relative_to(source396)),root/'common_tools'/file.name)
    for name in ['qualify_ensemble_feedback.py','program_revision_runtime.py','controlled_feedback.py','controlled_ensemble_feedback.py']:
        shutil.copy2(project/'scripts'/name,root/'code'/name)
    engine=dict(input_root=str(root/'private_inputs'),common_tools=str(root/'common_tools'),fit_seconds=120,predict_seconds=45,program_seconds=450)
    save(root/'ENGINE_CONFIG.json',engine)
    cfg=dict(iteration='A398',utc=utc(),scope='Engineeringqualification of fixed A397 whole-record voting with genuine complete programs;not a new Agent comparison',
        pools=pools,maximum_parallel_pools=3,replays=['forward','renamed_cold'],incumbent='p00',
        expected_prediction_processes=2*8*sum(len(p['members']) for p in pools),
        source_records=8,physical_sources=1,new_physical_records=0,new_LLM_calls=0,new_fits=0,final_data_read=False,
        budgets=dict(wall_seconds=450,per_record_all_members_seconds=45,live_disk_GiB=1),
        acceptance='All real member calls successful;per-window repeat and renamed identity equivalence;member record votes match original reports;whole-record ensemble equals frozenA397rule;0fit/optimizer in prediction;input label isolation;all source/weight hashes unchanged;record time and whole wall limits satisfied.',
        fixture_selection='5memberordinary,4memberreview,and2membercounterexample;chosen as explicit software/cost fixtures,not as independent successful scientific cases.',
        prior_negative='A298 real six-model probability-ensemble search already showed validation-optimal mixtures can fail. Do not revive oldG2weightsearch or claim ensemble novelty.',
        no_weight_or_rule_change=True,protected_confirmation_opened=False,
        frozen_code={str(p.relative_to(root)):sha(p) for d in ['code','common_tools'] for p in (root/d).glob('*.py')})
    save(root/'PROTOCOL398.json',cfg);save(root/'SOURCE_BINDINGS398.json',bindings)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4',usage='Pinned read-only scientific packages'),
        retain=['All measuredinputs,member sources/weights,original bindings,cold predictions,audits,time and failures;no model or raw data synthesized']))
    output=project/'results/coldensemble398';output.mkdir(exist_ok=True)
    save(output/'PROTOCOL398.json',cfg);save(output/'current_run.json',dict(utc=utc(),run=str(root),status='prepared_not_started',new_LLM_calls=0,goal_complete=False))
    print(json.dumps(dict(status='frozen',run=str(root),pools=len(pools),expected_prediction_processes=cfg['expected_prediction_processes'],new_fits=0,new_LLM_calls=0)))


def run_complete_programs(root):
    import time
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from program_revision_runtime import Runtime,load,score
    project=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010');cfg=json.loads((root/'PROTOCOL398.json').read_text())
    assert not (root/'STARTED398.json').exists()
    for relative,digest in cfg['frozen_code'].items():assert sha(root/relative)==digest
    started=time.monotonic();deadline=started+cfg['budgets']['wall_seconds'];data=load(root/'private_inputs/development.npz')
    assert len(set(data['groups'].tolist()))==8
    save(root/'STARTED398.json',dict(utc=utc(),new_fits=0,new_LLM_calls=0))
    def worker(pool):
        runtime=Runtime(root);runtime.deadline=deadline;target=root/'execution'/pool['id'];target.mkdir(parents=True);replays=[]
        try:
            for replay in cfg['replays']:
                prediction,audits=runtime.predict_ensemble(target/replay,pool['members'],data,incumbent='p00',rename=replay=='renamed_cold')
                replays.append(dict(replay=replay,score=score(data,prediction),audits=audits,
                    ensemble_report=json.loads((target/replay/'ENSEMBLE_REPORT.json').read_text())))
            answer=dict(id=pool['id'],status='complete',replays=replays)
        except Exception as error:
            answer=dict(id=pool['id'],status='error',replays=replays,error_type=type(error).__name__,error=str(error))
        save(target/'RESULT.json',answer);print(json.dumps(dict(pool=pool['id'],status=answer['status'],finished_replays=len(replays))),flush=True)
        return answer
    rows=[]
    with ThreadPoolExecutor(max_workers=cfg['maximum_parallel_pools']) as executor:
        futures=[executor.submit(worker,pool) for pool in cfg['pools']]
        for future in as_completed(futures):rows.append(future.result())
    result=dict(utc=utc(),status='complete' if all(r['status']=='complete' for r in rows) else 'completed_with_failures',
        rows=sorted(rows,key=lambda r:r['id']),elapsed_seconds=time.monotonic()-started,new_fits=0,new_LLM_calls=0,
        cold_prediction_audits=len(list((root/'execution').rglob('training_audit.json'))),final_data_read=False,protected_confirmation_opened=False,
        independent_qualification_audit_pending=True,goal_complete=False)
    save(root/'RESULT398.json',result);save(project/'results/coldensemble398/RESULT398.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


def audit_complete_programs(root):
    from collections import Counter
    from program_revision_runtime import load
    project=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010');cfg=json.loads((root/'PROTOCOL398.json').read_text());result=json.loads((root/'RESULT398.json').read_text())
    checks={};bindings=json.loads((root/'SOURCE_BINDINGS398.json').read_text());data=load(root/'private_inputs/development.npz');truth=dict(zip(data['sample_ids'],data['labels']));total_calls=0;per_record_times=[]
    for n,b in enumerate(bindings):checks['source_binding_'+str(n)]=sha(Path(b['archive'])/b['path'])==b['sha256']
    for relative,digest in cfg['frozen_code'].items():checks['frozen_'+relative]=sha(root/relative)==digest
    for pool in cfg['pools']:
        prefix=pool['id'];target=root/'execution'/prefix;observed=[];window_outputs=[]
        for ident,member in pool['members'].items():
            base=Path(member['source']).parent
            for relative,digest in pool['reports'][ident]['artifacts'].items():
                if relative.startswith(('source/','full_fit/artifacts/')):checks[prefix+'/'+ident+'/'+relative]=sha(base/relative)==digest
        for replay in cfg['replays']:
            folder=target/replay;report_path=folder/'ENSEMBLE_REPORT.json';checks[prefix+'/'+replay+'/report_exists']=report_path.exists()
            if not report_path.exists():continue
            report=json.loads(report_path.read_text());predictions={};member_outputs={}
            for index,group in enumerate(sorted(set(data['groups'].tolist()))):
                path=folder/('record_'+str(index));observations=load(path/'observations.npz');original=data['sample_ids'][data['groups']==group].tolist()
                key=prefix+'/'+replay+'/'+str(index);checks[key+'/labels_absent']='labels' not in observations
                checks[key+'/exact_observed_waveform']=all(np.array_equal(observations[k],data[k][data['groups']==group]) for k in data if k not in ['labels','sample_ids','groups'])
                votes=[];by_id={}
                for ident in sorted(pool['members']):
                    proc=json.loads((path/ident/'PROCESS.json').read_text());audit=json.loads((path/ident/'training_audit.json').read_text())
                    checks[key+'/'+ident+'/real_process']=proc['returncode']==0 and proc['mode']=='predict' and proc['command'][0]=='bwrap'
                    checks[key+'/'+ident+'/isolation']='--unshare-net' in proc['command'] and '--ro-bind' in proc['command']
                    checks[key+'/'+ident+'/no_fitting']=audit['mode']=='predict' and audit['status']=='completed' and audit['completed_estimator_fits']==audit['optimizer_steps']==0
                    rows=json.loads((path/ident/'predictions.json').read_text())['predictions'];mapping={r['sample_id']:r['predicted_label'] for r in rows}
                    checks[key+'/'+ident+'/all_ids']=len(rows)==len(mapping)==len(original) and set(mapping)==set(observations['sample_ids'].tolist())
                    labels=[mapping[s] for s in observations['sample_ids']];counts=Counter(labels);pred=min(counts,key=lambda k:(-counts[k],k))
                    by_id[ident]=pred;votes.append(pred);member_outputs[ident+'/'+str(group)]=dict(zip(original,labels))
                    expected=pool['historical_development'][ident][str(group)]
                    checks[key+'/'+ident+'/historical_vote']=pred==expected[1] and all(int(truth[s])==expected[0] for s in original)
                    total_calls+=1
                counts=Counter(votes);tied=[k for k,n in counts.items() if n==max(counts.values())]
                chosen=by_id['p00'] if by_id['p00'] in tied else min(tied);predictions[str(group)]=chosen
                stored=json.loads((path/'ENSEMBLE.json').read_text());checks[key+'/independent_vote']=stored['predicted']==chosen and stored['members']==by_id
                checks[key+'/record_time_budget']=stored['seconds']<=cfg['budgets']['per_record_all_members_seconds'];per_record_times.append(stored['seconds'])
            with np.load(folder/'PREDICTIONS.npz',allow_pickle=False) as z:
                checks[prefix+'/'+replay+'/output_ids']=np.array_equal(z['sample_ids'],data['sample_ids'])
                checks[prefix+'/'+replay+'/output_values']=z['prediction'].tolist()==[predictions[str(g)] for g in data['groups']]
            observed.append(predictions);window_outputs.append(member_outputs)
        checks[prefix+'/two_cold_ensembles_identical']=len(observed)==2 and observed[0]==observed[1]
        checks[prefix+'/all_member_windows_identity_invariant']=len(window_outputs)==2 and window_outputs[0]==window_outputs[1]
    checks['actual_calls_complete']=total_calls==cfg['expected_prediction_processes']==result['cold_prediction_audits']
    checks['whole_wall_budget']=result['elapsed_seconds']<=cfg['budgets']['wall_seconds']
    checks['all_pools_finished']=len(result['rows'])==3 and all(r['status']=='complete' for r in result['rows'])
    out=dict(utc=utc(),passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        actual_cold_prediction_processes=len(list((root/'execution').rglob('training_audit.json'))),fully_recounted_processes=total_calls,
        record_calls=len(per_record_times),planned_record_calls=48,new_fits=0,new_LLM_calls=0,
        seconds=result['elapsed_seconds'],maximum_record_seconds=max(per_record_times,default=None),
        mean_record_seconds=sum(per_record_times)/len(per_record_times) if per_record_times else None,
        raw_records=8,whole_program_pools=3,independent_confirmation=False,final_data_read=False,
        caveat='Operational qualification only;not method novelty,held-out improvement,or a new Agent experiment.8records shared by all3pools;member windows agree under identity renaming.')
    save(root/'INDEPENDENT_AUDIT398.json',out);save(project/'results/coldensemble398/INDEPENDENT_AUDIT398.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='checks'}));assert out['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--review',action='store_true')
    p.add_argument('--complete-programs',choices=['prepare','run','audit'])
    a=p.parse_args()
    if a.complete_programs:{'prepare':prepare_complete_programs,'run':run_complete_programs,'audit':audit_complete_programs}[a.complete_programs](a.run.resolve())
    else:(qualify_review if a.review else qualify)(a.run.resolve())
