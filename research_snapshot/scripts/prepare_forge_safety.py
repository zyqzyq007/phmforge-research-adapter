"""Three original safety directions: genuine diagnosis and bounded claims."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import secrets

import numpy as np

from acquire_forge_pdmb_sources244 import save,sha256
from prepare_verified_benchmarks import extract_bound
from run_validation_transfer import prepare,freeze,metadata_gate,execute_wave,state

PROJECT=Path(__file__).resolve().parents[1]
ARCHIVE=Path('/data/yyf-zyq/PHM_20261010')


def read(p):return json.loads(Path(p).read_text())


def terminal_xjtu(root,run):
    """Only the recorded terminal observation has a teardown diagnostic label."""
    from real_fault_model import train
    allocation=read(ARCHIVE/'20261022T081740Z_feedback278/hidden/ALLOCATION278.json')
    parent=read(ARCHIVE/'20261022T081740Z_feedback278/CAMPAIGN_CONFIG.json')
    source=Path(parent['source_archive']);qualification=read(source/'audit/XJTU_SOURCE_QUALIFICATION277.json')
    assert qualification['passed'];records={v['bearing']:v for v in qualification['bearings']}
    labels={name:'+'.join(sorted(v['terminal_damage_locations'])) for name,v in records.items()}
    classes=sorted(set(labels.values()));target=run/'restored/XJTU_TERMINAL';data=target/'data';data.mkdir(parents=True)
    truth=[];bindings=[]
    for part in ['train','val','test']:
        rows=[]
        for bearing in allocation['splits'][part]:
            item=records[bearing];path=source/item['path'];assert sha256(path)==item['sha256']
            arrays=dict(np.load(path,allow_pickle=False));x=arrays['features'][-1].astype(np.float32)
            assert x.shape==(30,) and np.isfinite(x).all()
            sid=secrets.token_hex(16);group=secrets.token_hex(12)
            rows.append(dict(sample_id=sid,features=x,label=labels[bearing],group=group))
            bindings.append(dict(part=part,bearing=bearing,path=str(path),sha256=sha256(path),observation='Last recorded measured two-axis feature vector only',terminal_label=labels[bearing]))
            if part=='test':truth.append(dict(sample_id=sid,label=labels[bearing],group=group))
        values=dict(features=np.asarray([r['features'] for r in rows]),sample_ids=np.asarray([r['sample_id'] for r in rows]),dataset=np.asarray('XJTU_TERMINAL'))
        if part!='test':values.update(labels=np.asarray([r['label'] for r in rows]),groups=np.asarray([r['group'] for r in rows]))
        np.savez_compressed(data/(part+'.npz'),**values)
    save(data/'dataset.json',dict(dataset='XJTU_TERMINAL',class_names=classes,architectures=['mlp'],seed=2026102318,
        source_sha256=sha256(source/'audit/XJTU_SOURCE_QUALIFICATION277.json'),
        source_sha256_scope='Verified source qualification; native-feature arrays individually hash bound.',
        rows_by_split=dict(train=9,val=3,test=3),feature_count=30,
        group_definition='One original bearing terminal observation; frozen prior9/3/3 allocation,all bearings disjoint.',
        data_origin='Measured XJTU-SY terminal acquisition statistics with publisher teardown damage locations.',
        target_scope='Terminal diagnostic labels only; no observed early-stage fault labels or severity bins.',
        limitations=['Fifteen terminal observations are a very small diagnostic sample.','The three source conditions remain shared.','Severity-stratified and early-versus-late accuracy cannot be certified.']))
    save(target/'hidden/labels.json',truth);audit=train(data,target/'reference','mlp',50)
    assert audit['optimizer_steps']>0 and audit['nonzero_gradient_steps']>0 and audit['changed_parameter_tensors']>0
    save(root/'audit/XJTU_TERMINAL_SOURCE318.json',dict(rows=bindings,new_real_reference_fits=1,
        initial_reference_updates=audit['optimizer_steps'],unchanged_prior_bearing_allocation=True,
        early_fault_or_severity_labels_not_fabricated=True,final_predictions_not_created=True))


def build(root,diagnostic_extras=False,cwru_diagnostic=False):
    assert not (diagnostic_extras and cwru_diagnostic)
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    for f in ['audit','source','code','logs','runtime/tmp','runtime/mpl','clients-cache']:(root/f).mkdir(parents=True,exist_ok=True)
    prior=read(PROJECT/'results/enginecoding312/current_run.json')
    online=read(Path(prior['run'])/'audit/CLI_ONLINE_CHECK.json')
    url='https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-research'}),timeout=40) as response:
        body=response.read();http=response.status
    release=json.loads(body);assert release['tag_name']=='rust-v'+online['latest_stable_version']
    assert sha256(Path(online['CLI_path']))==online['CLI_sha256']
    online.update(checked_utc=datetime.now(timezone.utc).isoformat(),http_status=http,latest_response_sha256=hashlib.sha256(body).hexdigest())
    save(root/'audit/CLI_ONLINE_CHECK.json',online);(root/'audit/CLI_LATEST_RELEASE.json').write_bytes(body)
    cases=[dict(cohort='c',dataset='CWRU',task_id='pdm_safety_002'),
           dict(cohort='m',dataset='ELECTRICMOTORVIBRATIONS',task_id='pdm_safety_003'),
           dict(cohort='r',dataset='ROTORBROKENBAR',task_id='pdm_safety_004')]
    if diagnostic_extras:
        cases=[dict(cohort='m',dataset='ELECTRICMOTORVIBRATIONS',task_id='pdm_fault_003'),
               dict(cohort='x',dataset='XJTU_TERMINAL',task_id='pdm_fault_002')]
    if cwru_diagnostic:
        cases=[dict(cohort='c',dataset='CWRU',task_id='pdm_fault_001')]
    config=dict(iteration='A314',result_directory='results/safety314',planned_sessions=3,
        purpose='Three original safety directions: actual measured-signal diagnosis and evidence limits; no fabricated physical risks, standards certificates or maintenance effects.',
        authorization='用户：把PHMforge能真实实验的全做了，并以此实验数据进行分析并验证问题。',
        test_inputs_visible=False,candidate_bank_from_training_cv=False,cases=cases,
        cohorts={c['cohort']:dict(family='classification',specs=[dict(id='safety314_'+c['cohort'],dataset=c['dataset'],task_id=c['task_id'])]) for c in cases},
        session_budget=dict(subject_timeout_seconds=3600,max_model_requests_per_subject=180,max_total_input_tokens_per_subject=18000000,
            max_output_tokens_per_request=32768,max_total_output_tokens_per_subject=262144,mcp_tool_timeout_seconds=180,
            max_concurrency=8,local_max_concurrency=1,artifact_budget_bytes=4*1024**3),
        prior_source_final_scores_known_to_researcher=True,new_independent_sources=0,
        supported_subtask='Classify source-observed laboratory classes; assess what follows and what remains unknown.',
        physical_risk_calibration_or_regulatory_certification_qualified=False,
        no_automatic_final_scoring=True,no_automatic_review_or_retry=True)
    if diagnostic_extras:
        config.update(iteration='A318',result_directory='results/diagnostic318',planned_sessions=2,cases=cases,
            purpose='Two additional original diagnostic directions with real observable labels: vibration-only motor classification and verified XJTU terminal diagnosis. Unsupported modality/severity comparisons stay unknown.',
            cohorts={c['cohort']:dict(family='classification',specs=[dict(id='diagnostic318_'+c['cohort'],dataset=c['dataset'],task_id=c['task_id'])]) for c in cases},
            supported_subtask='Original source diagnostic label prediction; requested acoustic fusion and early/late severity comparisons remain unqualified.',
            full_original_auxiliary_comparison_qualified=False)
    if cwru_diagnostic:
        config.update(iteration='A319',result_directory='results/cwrudiagnostic319',planned_sessions=1,cases=cases,
            purpose='One current qualified CWRU diagnostic baseline: repair historical task001 source-identity and metadata qualification gaps, not a retry selected for low scores.',
            cohorts={'c':dict(family='classification',specs=[dict(id='cwrudiagnostic319_c',dataset='CWRU',task_id='pdm_fault_001')])},
            supported_subtask='Original inner/outer/ball diagnostic labels on verified native CWRU recordings; public fault-size split, not a new source or unseen-outcome confirmation.',
            new_original_task_directions=0,historical_original_task_attempts_retained=2,
            reason_for_current_qualified_baseline='The historical two task001 attempts had source-block identity and test metadata limitations; neither was a normal final submission. Current native source and hidden-test metadata gates are independently qualified.')
    addendum='''
Safety-task evidence contract: class predictions can be independently evaluated against source laboratory fault labels. The original request's physical failure probabilities, fire or electrical hazard magnitudes, certified vibration-limit compliance and immediate-shutdown thresholds are not supplied by these data. Generic safety/calibration tools correctly report unavailable. Do not manufacture those quantities or certificates. Separate a measured/predicted class, a possible hazard requiring expert assessment, and a verified safe/unsafe action. State what machine ratings, calibrated units, operating limits, standard edition and field evidence would be needed. Conditional general recommendations may be given but do not claim measured maintenance benefit or legal compliance.

Include safety_evidence in the output JSON with diagnostic_scope, calibrated_failure_probability (null when unmeasured), compliance_status (unknown unless an applicable verified contract exists), shutdown_threshold (null when not supplied), and limitations. Retain per-sample predictions in the original source label representation. Field conformance and evidence sufficiency are separate from classification accuracy; plausible prose is not calibrated physical risk. The original task remains visible.
'''
    if diagnostic_extras:
        addendum='''
Diagnostic evidence contract: retain the original requested source-label prediction, and distinguish its supported part from missing comparison data. ElectricMotorVibrations supplies measured three-axis vibration only; acoustic-only and acoustic-plus-vibration scores cannot be produced. Do not relabel an acceleration axis as a microphone or invent audio. XJTU_TERMINAL supplies one verified terminal observation per bearing with publisher teardown damage locations. Train9/development3/final3 bearings follow a previously frozen allocation; this is a tiny endpoint diagnostic study. Do not propagate teardown labels into earlier trajectory rows, invent a healthy early stage, or claim severity-stratified accuracy. No per-time severity ground truth exists here. Explicitly mark these requested comparisons unavailable. The diagnostic target labels themselves are unchanged source labels.
'''
    if cwru_diagnostic:
        addendum='''
Diagnostic evidence contract: predict the original inner-race, outer-race and ball fault labels from verified measured CWRU vibration. These nine public laboratory recordings use fault diameter .007 inch for fitting, .014 for development and .021 for external evaluation; this is a small fault-size transport study, not a new physical source. Preserve the provided input/output interface and fit/inference access rules. There are no calibrated physical-risk labels, actual maintenance action outcomes or regulatory certification targets. No final observations, class counts, final labels, previous Agent submissions or previous scores are supplied during development. Retaining the genuine starting program is allowed. Report source-group and window validation separately when available and do not treat many windows as many independent machines.
'''
    config['prompt_addendum']=addendum;save(root/'CAMPAIGN_CONFIG.json',config)
    source_specs={
        'c':('20261023T145500Z_review311','source/previous/c/restored/CWRU/'),
        'm':('20261021T174126Z_choice270','a/restored/ELECTRICMOTORVIBRATIONS/'),
        'r':('20261023T052522Z_rotor305','a/restored/ROTORBROKENBAR/')}
    bindings=[]
    for case in cases:
        c=root/case['cohort']
        if diagnostic_extras and case['cohort']=='x':terminal_xjtu(root,c)
        else:
            rid,prefix=source_specs[case['cohort']]
            restored=root/'source'/case['cohort'];bound=extract_bound(ARCHIVE/rid,restored,[prefix]);bindings.extend(bound)
            origin=restored/prefix;target=c/'restored'/case['dataset'];shutil.copytree(origin,target)
        if case['cohort']=='r':
            p=target/'data/dataset.json';v=read(p);v['known_deployment_loads_Nm']=[3.5,4.0];v['data_scope']='Acquisitions3/4; fit loads0.5..2.5Nm,development3.0Nm,external3.5/4.0Nm. Same public source used previously,not unseen-source confirmation.';save(p,v)
        prepare(root,c,'classification','initial')
        gate=read(c/'TASK_QUALIFICATION251.json');gate.update(supported_original_subtask=config['supported_subtask'],full_physical_safety_contract_qualified=False,
            full_auxiliary_comparison_qualified=cwru_diagnostic)
        save(c/'TASK_QUALIFICATION251.json',gate);metadata_gate(c);freeze(c)
    save(root/'audit/SOURCE_BINDINGS314.json',bindings)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared',retention='Pinned dependencies; original measured inputs and genuine reference fits retained with source hashes.')))
    state(root,status='cwru_diagnostic_prepared' if cwru_diagnostic else 'diagnostic_extras_prepared' if diagnostic_extras else 'three_original_safety_directions_prepared',
        new_model_sessions=0,new_original_task_directions_planned=0 if cwru_diagnostic else len(cases),
        actual_physical_safety_certification=False,final_scores_opened=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=['build','run'],required=True)
    p.add_argument('--diagnostic-extras',action='store_true')
    p.add_argument('--cwru-diagnostic',action='store_true')
    a=p.parse_args();root=a.run.resolve()
    if a.phase=='build':build(root,a.diagnostic_extras,a.cwru_diagnostic)
    else:execute_wave(root,'cwru_diagnostic' if a.cwru_diagnostic else 'diagnostic_extras' if a.diagnostic_extras else 'three_original_safety_directions',
        [root/c['cohort'] for c in read(root/'CAMPAIGN_CONFIG.json')['cases']])
