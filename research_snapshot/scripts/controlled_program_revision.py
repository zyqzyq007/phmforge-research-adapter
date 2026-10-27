"""Prospective R4 development pilot:controlled code edits plus crossed selection.

Four actual native builders precede four native selectors on the SAME four
proposal pools. This separates candidate-generation effects from selection
effects. All PHM fits and predictions are real. Final data stay outside Actors.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import time
import urllib.request

import numpy as np

from controlled_feedback import save,sha,utc
from native_validation_comparison import closure
from program_revision_runtime import Runtime,load,score
from evaluate_free_program_comparison import independent_score
from run_frozen_candidate_review import source_context
import free_program_comparison as coding
import screen_program_updates as selection

PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261024T203500Z_source341')
LIVE_SOURCE=Path('/srv/yyf-zyq/PHM_20261010/20261024T203500Z_source341')
TEMPLATE=Path('/data/yyf-zyq/PHM_20261010/20261024T202500Z_decision340')
OUT=PROJECT/'results/controlled342'

BUILD='''Improve or retain a genuinely trained four-class PHM diagnosis program using a native coding Agent. Read /data/metadata.json,/reference/source/program.py,/reference/source/recipe.json and /reference/REPORT.json,then phmforge info. You may write arbitrary actual feature extraction,classifiers and inference code,or use the optional modular scaffold. Real train/development observations and labels are supplied;independent_check_observations.npz has no labels. There are8training acquisitions,not72independent devices. Raw measured waveforms and source-supplied statistical features are available.

Register complete source with register_program(files,claim,rival). Every accepted version really fits two complete-condition leave-out models and one full training checkpoint,then cold-evaluates development and opaque-ID invariance. All supervised fitting for candidate programs must use this tool;no unregistered fits. At most4new versions besides p00. Source and fitted artifacts are immutable once registered;an edit needs a new registration. Use supported audited sklearn classifiers or genuine Torch optimizer updates,fixed seeds and at most2numeric threads. No fabricated waveforms,fault labels,mock training,constant results or batch-class-count tricks. Prediction receives one acquisition at a time,without labels,with read-only fitted artifacts. Do not depend on sample/group identifiers. Whole-acquisition error by9window majority vote is primary;future conditions are unknown.

This is the BUILD stage of a common two-stage experiment. The controller will freeze the entire actual program bank after this stage and cold-evaluate every complete program on one permitted independent validation condition. probe is deferred until then;neither arm receives its labels during building. A separate fresh decision session selects from the frozen bank using real CV/development/check evidence. Your submit(program_id,reason) is a PRELIMINARY handoff,not the final scientific choice. Preserve useful candidates even if you do not prefer them now. Save report.md describing actual modifications and remaining uncertainties. No validation view is guaranteed to transfer.

The optional common scaffold uses recipe.json flags feature_mode/model_mode0or1. Zero means supplied48statistics and StandardScaler/LDA. A new feature module provides feature_impl.new_features(data);a new model module provides model_impl.new_estimator(). Both arms may use this scaffold,choose any actual feature/estimator implementations,or replace it. All helper files must be registered. Do not dump sensor arrays through shell output. Final data,labels,conditions,scores,other runs and other Agents are inaccessible. No network,other model APIs or agent spawning. Store scratch under /app/.runtime,never /tmp.

Build budget:1500seconds,30model requests,2000000cumulative input admission threshold,65536output total,32768output/request,4new versions;each version450seconds,fit120seconds,predict45seconds. Numeric remaining budget is in info/register responses. Deliver a preliminary handoff before this stage ends. The common controller separately reserves300seconds for independent checks and600seconds/20requests/1000000input/65536output for fresh selection. Total per paired build/choice route at most2400seconds/50requests/3000000input admission thresholds/131072output. Actual input is counted after replies and may overshoot by one admitted reply;the same rule applies to both arms.
'''

EDIT='''
Additional R4 controlled-edit workflow under test:separate an intended feature change F from an intended classifier/decision change M. State a concrete expected benefit and a rival explanation in edit_plan.json BEFORE registering the changed programs. Use the supplied modular program.py unchanged to obtain three complete recipes:10(new features,old classifier),01(old features,new classifier),11(both new). Use the SAME new feature_impl.py and model_impl.py bytes across these three cells;only recipe.json flags differ. p00 is the00anchor;its zero branches never import changed helpers. Choose actual new implementations yourself;this is not a fixed model bank. Freeze these three cells before spending the optional fourth registration on a targeted repair. Complete the three controlled comparisons before preferring a composite edit.

Compare actual per-condition full-program outcomes. If a partial change helps but the combination loses that benefit,retain the partial program instead of presuming the combined edit is best. Compute e11-e10-e01+e00 where all cells exist;this is an empirical program-edit interaction on those records,not physical fault causality or a guarantee about unseen conditions. Record negative and inconclusive results. Extra roles,claims or a low score alone are not evidence. No deterministic worst-condition gate is imposed. This workflow is an exploratory hypothesis,not an established novel or superior method.
'''

CHOOSE='''Choose one immutable actually trained PHM program for EACH of the four supplied related proposal pools. All four pools share one physical train/development/check/final split;these are not independent datasets. Read /data/contract.json,/data/cases.json and phmforge evidence for every case. Every pool has original incumbent p00. You may analyze actual source and acquisition-level predictions in /data/rows/,compare paired gains/harms,and quantify uncertainty. No fitting,new predictors,mixtures,network or other Agents. Final observations,labels,conditions,scores and historical conclusions are absent.

CV uses single-other-condition trained checkpoints;development and independent check use the unchanged full-training checkpoint that would deploy. The check is a permitted validation condition,not the final evaluation. All rows are genuine,and fitting/evaluation contexts are explicit. The metric is whole-acquisition classification error,9window majority vote;windows are not independent devices. Minimize future error while preserving useful improvements and avoiding harmful replacements. No validation view or decision rule is guaranteed to transfer. Candidate descriptions and any controlled-edit ledger are the SAME for both selector styles. Missing factorial cells do not justify invented results.

Submit exactly4(case,incumbent,selected,reason)rows,incumbent p00,once via phmforge select. Read evidence for every case first. Save selection.json and report.md. File fallback:python3 /tools/program_update_tools.py client --request followed by the JSON action. Successful immutable submission is the deliverable;do not postpone it for a lengthy report. Budget600seconds/20requests/1000000cumulative input admission threshold/65536output/16384perreply/2numeric threads. Remaining numeric budget is in cases/evidence. All scratch under /app/.runtime,never /tmp. External final scoring starts only after all builder and selector choices seal.
'''

CONTRAST='''
Additional R4 decision workflow:before choosing a combined revision,inspect any verified controlled-edit cells and paired acquisition outcomes. Distinguish which change repaired each observed failure,which new mistakes it introduced,and whether the composite lost a benefit of a partial program. Write the observed interaction and the evidence scope in your reason/report. Do not carry an explanation across different recipes,weights or evaluation conditions. If the proposed explanation is contradicted,retain the counterevidence and evaluate the actual partial alternatives. Incomplete control cells remain incomplete evidence. Use full-trained-model legal results as a strong comparison;do not treat a perfect fold score as automatic proof of deployment quality. No fixed keep rule,absolute minimax gate or mandatory preference for combined edits is imposed. Choose the actual candidate supported by the evidence and disclose unresolved uncertainty.
'''


def read(path):return json.loads(Path(path).read_text())


def status(root,**changes):
    path=root/'STATUS342.json';v=read(path) if path.exists() else {};v.update(updated_utc=utc(),**changes);save(path,v);save(OUT/'current_run.json',dict(v,run=str(root)))


def freeze(root,budget_profile='standard',reuse_baseline_from=None,result_key='controlled342'):
    root.mkdir(exist_ok=False)
    for name in ['code','common_tools','logs','audit','reference/p00/source','private_inputs','public_data','runtime/tmp','runtime/mpl']:(root/name).mkdir(parents=True)
    source_protocol=(SOURCE if SOURCE.exists() else LIVE_SOURCE)/'PROTOCOL341.json';assert source_protocol.exists()
    inventory,_=source_context(TEMPLATE);prior=read(TEMPLATE/'CAMPAIGN_CONFIG.json');assert sha(TEMPLATE/'CAMPAIGN_CONFIG.json')==inventory['CAMPAIGN_CONFIG.json']['sha256']
    url='https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-research-provenance'}),timeout=30) as response:release=json.load(response)
    assert release['tag_name']=='rust-v0.164.0' and sha(Path(prior['cli']))==prior['cli_sha256']
    save(root/'audit/CLI_SOURCE342.json',dict(url=url,checked_utc=utc(),tag=release['tag_name'],published_at=release['published_at']))
    cfg=dict(iteration='A342',method_version='R4controlled-feature-model-edits',frozen_utc=utc(),goal='program_update_iteration_goal330',
        model='deepseek-flash',cli=prior['cli'],cli_sha256=prior['cli_sha256'],source_archive=str(SOURCE),source_protocol_sha256=sha(source_protocol),
        method='Controlled actual feature/model edit cells10,01,11 plus00anchor;crossed ordinary/contrast-aware native selection on identical proposal banks.',
        mechanism_hypothesis='Composite program changes can obscure which modification earns a validation gain and which introduces errors. Real controlled partial programs may preserve a useful edit that ordinary combined search misses.',
        prediction='R4must improve final task loss relative to ordinary under equal budgets,and its gain must exceed full-checkpoint validation/keep controls. Crossed selectors expose whether benefits come from selection or candidate generation.',
        falsification='No changed useful candidates,ordinary/fixed controls match or beat R4,interaction evidence unused/unreliable,or gain only from never updating or incomplete builds. Retain every planned attempt,no outcome-based reruns.',
        novelty_status='Not established.2x2DOE,interventions,pairwise risk and holdout acceptance are prior art;test the complete PHMprogram-edit workflow increment rather than naming those ideas novel.',
        related_work=[dict(title='DoVer',venue='ICLR2026',url='https://proceedings.iclr.cc/paper_files/paper/2026/hash/2f5fdce256ff0d7fb774da76e5e63209-Abstract-Conference.html'),
            dict(title='Model Assessment and Selection under Temporal Distribution Shift',venue='ICML2024',url='https://proceedings.mlr.press/v235/han24b.html'),
            dict(title='GTDD',venue='2026preprint',url='https://arxiv.org/abs/2610.02952')],
        arms=['ordinary_coding','controlled_edits'],selector_conditions=['ordinary','controlled'],repeats=2,
        planned_builder_sessions=4,planned_selector_sessions=4,maximum_parallel_agents=4,
        stage_budget=dict(build=dict(seconds=1500,requests=30,input_tokens=2000000,output_tokens=65536,per_reply=32768),
            independent_check=dict(seconds=300,model_requests=0),choice=dict(seconds=600,requests=20,input_tokens=1000000,output_tokens=65536,per_reply=16384)),
        max_registered_programs=4,program_fit_seconds=120,program_predict_seconds=45,program_total_seconds=450,
        numerical_threads=2,maximum_new_model_requests=200,maximum_new_fit_entries=54,maximum_disk_GiB=8,
        source_role_contract_unchanged=True,source_prior_or_reserved_overlap=0,whole_physical_development_cohorts=1,
        independent_confirmation_scenarios=0,confirmation_experiment=False,reserved338_not_used=True,
        primary='Final whole-acquisition loss from common starting program. Generator/selector2x2effects descriptive on this single new cohort;all repetitions and pools retained.',
        common_bank_metric='Compare ordinary and controlled selectors on exactly the same four banks using selection loss,harm and useful gain capture. Do not compare own-pool oracle regrets across generators as improvement.',
        weighting='Average two selector repeats per pool,then two builder repeats per generator. All share one physical split;no claim of4or8independent scenarios.',
        final_failures='Keep missing selectors in denominator,report incomplete primary comparison;fallback p00 diagnostic never scored as method gain. Builder noncompletion reported separately;all registered complete programs remain eligible.',
        all_choices_before_final_scores=True,synthetic_data_forbidden=True,new_semantic_fault_labels_forbidden=True)
    cfg.update(result_key=result_key,budget_profile=budget_profile)
    build_prompt=BUILD
    if budget_profile=='adequacy':
        assert reuse_baseline_from is not None and not (Path(reuse_baseline_from)/'FINAL_EVALUATION_STARTED342.json').exists()
        cfg.update(iteration='A342-L2',budget_profile='adequacy',reuse_baseline_from=str(Path(reuse_baseline_from).resolve()),
            max_registered_programs=4,maximum_new_model_requests=320,maximum_new_fit_entries=48,
            source_shared_with_original_budget_layer=True,source_records_new_to_layer=False,source_current_layer_overlap=20,
            source_prior_original_and_reserved_overlap=0,
            preregistration_status='Exploratory budget sensitivity specified after two ordinary empty-bank30requestfailures,but before any final predictions. Original layer retained,not replaced.',
            joint_parent_choice_seal=str(Path(reuse_baseline_from).resolve()/'ALL_CHOICES_SEALED342.json'))
        cfg['stage_budget']['build'].update(requests=60,input_tokens=4000000)
        build_prompt=BUILD.replace('1500seconds,30model requests,2000000cumulative','1500seconds,60model requests,4000000cumulative')
        build_prompt=build_prompt.replace('2400seconds/50requests/3000000input','2400seconds/80requests/5000000input')
    save(root/'PROTOCOL342.json',cfg);save(OUT/'PROTOCOL342.json',cfg)
    build=dict(cfg,stage='build',episode_seconds=1500,episode_requests=30,episode_input_tokens=2000000,episode_output_tokens=65536,output_per_request=32768,
        prompt_file='BUILD_PROMPTS342.json',build_stage_only=True,defer_independent_check=True)
    build.update(episode_requests=cfg['stage_budget']['build']['requests'],episode_input_tokens=cfg['stage_budget']['build']['input_tokens'])
    save(root/'PROTOCOL337.json',build);save(root/'BUILD_PROMPTS342.json',dict(ordinary_coding=build_prompt,controlled_edits=build_prompt+EDIT))
    choice=dict(cfg,stage='choice',episode_seconds=600,episode_requests=20,episode_input_tokens=1000000,episode_output_tokens=65536,output_per_request=16384,
        preparation_file='PREPARATION342.json',qualify_submission_chain=True)
    save(root/'CAMPAIGN_CONFIG.json',choice);save(root/'PROMPT_FREEZE.json',dict(ordinary=CHOOSE,controlled=CHOOSE+CONTRAST))
    names=closure(['controlled_program_revision','free_program_comparison','screen_program_updates','program_revision_tools',
        'program_revision_runtime','program_training_entry','phm_modular_program','program_update_tools','qualify_native_cli_mcp'])
    for name in names:shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    for name in ['program_revision_tools','program_training_entry','phm_program_features','program_update_tools','controlled_feedback','prepare_fault_extensions','acquire_forge_pdmb_sources244']:
        shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'common_tools'/(name+'.py'))
    for name in ['base_cli_config.toml','model_catalog187.json']:
        assert sha(TEMPLATE/name)==inventory[name]['sha256'];shutil.copy2(TEMPLATE/name,root/name)
    shutil.copy2(PROJECT/'scripts/phm_modular_program.py',root/'reference/p00/source/program.py')
    save(root/'reference/p00/source/recipe.json',dict(feature_mode=0,model_mode=0))
    save(root/'CODE_FREEZE.json',{p.name:sha(p) for p in (root/'code').glob('*.py')})
    shutil.copy2(TEMPLATE/'retention_policy.json',root/'retention_policy.json')
    save(root/'OWNERSHIP.json',dict(run=str(root),archive='/data/yyf-zyq/PHM_20261010/'+root.name,user_request='围绕错误替换有效PHM程序迭代有机制依据的真实方法对照',created_utc=utc()))
    status(root,status='protocol_frozen',source_not_scored=True,new_model_calls=0,goal_complete=False)


def prepare(root):
    cfg=read(root/'PROTOCOL342.json');assert not (root/'PREPARATION337.json').exists()
    inventory,receipt=source_context(SOURCE);assert sha(SOURCE/'PROTOCOL341.json')==cfg['source_protocol_sha256']
    qualification=read(SOURCE/'QUALIFICATION341.json');assert qualification['passed']
    audit=[];rng=np.random.default_rng(342)
    for role in ['train','development','independent_check']:
        path=SOURCE/'qualification'/(role+'.npz');assert sha(path)==qualification['files'][path.name]['sha256']
        values=load(path);groups=np.unique(values['groups']);rng.shuffle(groups)
        order=np.concatenate([np.flatnonzero(values['groups']==group) for group in groups]);values={k:v[order] for k,v in values.items()}
        private=root/'private_inputs'/(role+'.npz');np.savez_compressed(private,**values)
        if role!='independent_check':shutil.copy2(private,root/'public_data'/private.name)
        else:np.savez_compressed(root/'public_data/independent_check_observations.npz',**{k:v for k,v in values.items() if k!='labels'})
        audit.append(dict(role=role,source_sha256=sha(path),private_sha256=sha(private),shuffled_record_blocks=True,within_record_chronology_preserved=True))
    metadata=dict(task='Whole-acquisition diagnosis on future unknown conditions from one measured rotating-machine rig.',
        classes=dict(normal=0,unbalance=1,horizontal_misalignment=2,vertical_misalignment=3),
        training_acquisitions=8,development_acquisitions=4,windows_per_acquisition=9,window_points=25600,
        sensor_columns='Six documented vibration channels;unknown ninth raw column excluded.',sampling_rate_hz=51200,
        source='COMFAULDA author-acquired laboratory measurements,not simulation.',
        train_conditions_hz=[18,28],development_condition_hz=38,independent_validation='Observations legal;labels only returned by common post-commit checker. Not final evaluation.',
        primary='Whole-acquisition majority-vote classification error. Do not infer independent sample size or real deployment prevalence from windows/class-balanced collection.',
        optional_scaffold='Source program.py and recipe.json in /reference;mode0suppliedstatistics/StandardScaler-LDA,mode1Agent-writtenfeature_impl.new_features(data)/model_impl.new_estimator().',
        fit_policy='Train-only,complete-condition CV,fullcheckpointtrainedontrainingonly;no development refit.',
        budgets=cfg['stage_budget'],future_conditions_unknown=True,final_data_unavailable=True)
    save(root/'public_data/metadata.json',metadata)
    save(root/'ENGINE_CONFIG.json',dict(input_root=str(root/'private_inputs'),common_tools=str(root/'common_tools'),fit_seconds=120,predict_seconds=45,program_seconds=450))
    reused=cfg.get('reuse_baseline_from')
    if reused:
        origin=Path(reused);binding=read(origin/'SOURCE_BINDING342.json');assert binding['source_qualification_sha256']==sha(SOURCE/'QUALIFICATION341.json')
        for role in ['train','development','independent_check']:
            old=load(origin/'private_inputs'/(role+'.npz'));current=load(root/'private_inputs'/(role+'.npz'))
            assert set(old)==set(current) and all(np.array_equal(old[k],current[k]) for k in old)
        report=read(origin/'reference/p00/REPORT.json')
        for relative,digest in report['artifacts'].items():assert sha(origin/'reference/p00'/relative)==digest
        assert sha(origin/'reference/p00/source/program.py')==sha(root/'reference/p00/source/program.py')
        shutil.copytree(origin/'reference/p00',root/'reference/p00',dirs_exist_ok=True)
        preflight=read(origin/'REGISTRATION_PREFLIGHT337.json');assert preflight['passed']
        for name in ['program_revision_tools.py','program_training_entry.py','program_revision_runtime.py','controlled_feedback.py']:
            assert sha(root/'code'/name)==sha(origin/'code'/name),name
        save(root/'REUSED_PREFLIGHT342.json',dict(passed=True,source=str(origin),source_preflight_sha256=sha(origin/'REGISTRATION_PREFLIGHT337.json'),
            baseline_report_sha256=sha(origin/'reference/p00/REPORT.json'),all_legal_arrays_exact=True,reused_baseline_fit_entries=3,reused_qualification_fit_entries=3,new_fit_entries=0))
    else:report=Runtime(root).fit_program(root/'reference/p00')
    assert report['fit_entry_executions']==3 and report['identity_invariance_passed']
    coding.OUT=OUT;coding.prepare(root)
    # Both runners consume the same frozen tool files;selection has no access
    # to private waveforms or final arrays.
    shutil.copytree(root/'common_tools',root/'shared_tools')
    save(root/'SOURCE_BINDING342.json',dict(source_archive=receipt,source_qualification_sha256=sha(SOURCE/'QUALIFICATION341.json'),legal_array_packaging=audit,final_arrays_not_read=True))
    save(root/'CODE_FREEZE.json',{p.name:sha(p) for p in (root/'code').glob('*.py')})
    status(root,status='prepared',new_baseline_fit_entries=0 if reused else 3,reused_baseline_fit_entries=3 if reused else 0,final_data_read=False)


def factorial(root,programs):
    cells={'00':'p00'};hashes={};errors=[];expected=sha(root/'reference/p00/source/program.py')
    for ident,value in programs.items():
        if ident=='p00':continue
        source=Path(value['path'])/'source';p=source/'recipe.json'
        if not p.exists() or sha(source/'program.py')!=expected:continue
        r=read(p)
        if set(r)!={'feature_mode','model_mode'} or any(type(v) is not int or v not in [0,1] for v in r.values()):continue
        cell=str(r['feature_mode'])+str(r['model_mode'])
        if cell not in ['10','01','11'] or cell in cells:continue
        cells[cell]=ident;hashes[cell]={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file() and p.name!='recipe.json'}
    complete=set(cells)=={'00','10','01','11'} and all('feature_impl.py' in h and 'model_impl.py' in h for h in hashes.values()) and len({json.dumps(h,sort_keys=True) for h in hashes.values()})==1
    return dict(cells=cells,complete_structural_control=complete,shared_nonrecipe_hashes=hashes,
        interpretation='Controlled complete-program recipe changes on these real records;not physical fault causal identification.00never imports changed helpers.',errors=errors)


def materialize(root):
    assert (root/'BUILDERS_SEALED342.json').exists();cfg=read(root/'PROTOCOL342.json');cases=[];fixed={m:[] for m in ['keep','mean_cv','full_checkpoint_validation','absolute_minimax']};mapping={}
    (root/'public/rows').mkdir(parents=True)
    sessions=read(root/'SESSIONS337.json')
    def independent_check(row):
        run=Path(row['path']);state=read(run/'broker/STATE.json');programs=state['programs'];packet=run/'common_check';packet.mkdir()
        save(packet/'PRECOMMIT.json',dict(utc=utc(),code_and_weights={i:v['report']['artifacts'] for i,v in programs.items()},builder_has_ended=True))
        engine=Runtime(run);engine.config['program_seconds']=cfg['stage_budget']['independent_check']['seconds']
        return row['id'],engine.independent({i:Path(v['path']) for i,v in programs.items()},packet)
    packets={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(independent_check,row) for row in sessions]):
            ident,checked=f.result();packets[ident]=checked
    for number,row in enumerate(sessions,1):
        run=Path(row['path']);programs=read(run/'broker/STATE.json')['programs'];checked=packets[row['id']]
        case_id='pool_'+str(number);mapping[case_id]=row;ids=[i for i in programs if checked[i]['status']=='complete'];assert 'p00' in ids
        design=factorial(root,programs);risks=[];details={};descriptions=[]
        for ident in ids:
            p=Path(programs[ident]['path']);report=programs[ident]['report'];views=[*report['cv'],report['development'],checked[ident]]
            for view in views:assert view['record_error']==sum(r['actual']!=r['predicted'] for r in view['records'])/len(view['records'])
            risks.append([v['record_error'] for v in views]);details[ident]=dict(cv=report['cv'],development=report['development'],independent_check=checked[ident])
            for source in (p/'source').rglob('*'):
                if source.is_file():
                    target=root/'public/rows'/case_id/'source'/ident/source.relative_to(p/'source');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            claim=read(p/'CLAIM.json') if (p/'CLAIM.json').exists() else dict(claim='Common actually fitted starting program',rival='No optimality claim')
            descriptions.append(dict(id=ident,claim=claim['claim'],rival=claim['rival'],source='/data/rows/'+case_id+'/source/'+ident))
        if design['complete_structural_control'] and all(i in ids for i in design['cells'].values()):
            values={cell:risks[ids.index(ident)] for cell,ident in design['cells'].items()}
            design['error_interaction']=[values['11'][j]-values['10'][j]-values['01'][j]+values['00'][j] for j in range(4)]
            design['combination_worse_than_best_partial']=[values['11'][j]>min(values['10'][j],values['01'][j]) for j in range(4)]
        case=dict(id=case_id,candidate_ids=ids,incumbent_ids=['p00'],original_incumbent='p00',metric='whole-acquisition classification error',
            groups=['CV/train28/evaluate18','CV/train18/evaluate28','full/train18+28/development38','full/train18+28/independent_check54'],
            group_losses=risks,full_checkpoint_validation_columns=[2,3],candidate_descriptions=descriptions,controlled_edit_evidence=design,
            context='Two fold-trained vs two full-checkpoint views;each evaluation condition has4wholeacquisitions. Four banks share one physical split.')
        cases.append(case);save(root/'public/rows'/(case_id+'.json'),details)
        tie=lambda ident:(ident!='p00',ids.index(ident))
        choices=dict(keep='p00',mean_cv=min(ids,key=lambda i:(sum(risks[ids.index(i)][:2])/2,*tie(i))),
            full_checkpoint_validation=min(ids,key=lambda i:(sum(risks[ids.index(i)][2:])/2,*tie(i))),
            absolute_minimax=min(ids,key=lambda i:(max(risks[ids.index(i)]),sum(risks[ids.index(i)])/4,*tie(i))))
        for method,ident in choices.items():fixed[method].append(dict(case=case_id,incumbent='p00',selected=ident))
    save(root/'public/cases.json',cases);save(root/'public/contract.json',dict(task='Select frozen complete PHM programs;one common original incumbent per real proposal bank.',
        metric='Whole-acquisition error from9windowmajorityvote.',relationships='All4banks share one measured physical split;not4independent scenes.',
        permission='Onlylegalvalidationrows,source,andlabels;no raw final arrays,source filenames or history. No new fitting.',
        evidence='Each row from genuine code/weights;checkobservationsandlabelsarelegalindependentvalidation,notfinal. Exact training/evaluation conditions in group names.',
        control='Any structural interaction ledger is identically visible to both selector conditions;missing cells remain missing.'))
    save(root/'POOL_MAPPING342.json',mapping);save(root/'FIXED_CHOICES.json',dict(utc=utc(),choices=fixed,final_scores_joined=False))
    save(root/'PREPARATION342.json',dict(passed=True,utc=utc(),builders=4,related_pools=4,source_program_members=sum(len(c['candidate_ids']) for c in cases),
        complete_factorials=sum(c['controlled_edit_evidence']['complete_structural_control'] for c in cases),final_data_read=False))


def run(root):
    assert not (root/'BUILDERS_SEALED342.json').exists();coding.OUT=OUT
    cfg=read(root/'PROTOCOL342.json')
    if cfg.get('joint_parent_choice_seal'):assert Path(cfg['joint_parent_choice_seal']).exists(),'Original layer must seal before new layer starts'
    if cfg.get('reuse_baseline_from'):assert read(root/'REUSED_PREFLIGHT342.json')['passed']
    else:coding.preflight(root)
    coding.qualify(root)
    for name,digest in read(root/'CODE_FREEZE.json').items():assert sha(root/'code'/name)==digest
    status(root,status='actual_native_builders_running',controller_status='running',planned_builders=4,completed_builders=0)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(coding.execute,root,row) for row in read(root/'SESSIONS337.json')]):
            results.append(f.result());save(root/'BUILDER_PROGRESS342.json',results);status(root,status='actual_native_builders_running',completed_builders=len(results))
    save(root/'BUILDERS_SEALED342.json',dict(utc=utc(),episodes=results,final_scores_joined=False))
    status(root,status='common_independent_checks',completed_builders=4);materialize(root)
    selection.OUT=OUT;selection.agents(root)
    save(root/'ALL_CHOICES_SEALED342.json',dict(utc=utc(),builders_sha256=sha(root/'BUILDERS_SEALED342.json'),
        selectors_sha256=sha(root/'ALL_CHOICES_SEALED.json'),protocol_sha256=sha(root/'PROTOCOL342.json'),code_freeze_sha256=sha(root/'CODE_FREEZE.json'),final_scores_joined=False))
    status(root,status='all_choices_sealed',controller_status='terminal',final_data_read=False)


def evaluate(root):
    assert (root/'ALL_CHOICES_SEALED342.json').exists() and not (root/'RESULT342.json').exists()
    cfg=read(root/'PROTOCOL342.json');seal=read(root/'ALL_CHOICES_SEALED342.json');assert sha(root/'PROTOCOL342.json')==seal['protocol_sha256']
    if cfg.get('joint_parent_choice_seal'):assert Path(cfg['joint_parent_choice_seal']).exists(),'Both budget layers must seal before any final evaluation'
    for name,digest in read(root/'CODE_FREEZE.json').items():assert sha(root/'code'/name)==digest
    qualification=read(SOURCE/'QUALIFICATION341.json');path=SOURCE/'qualification/final.npz';assert sha(path)==qualification['files']['final.npz']['sha256']
    save(root/'FINAL_EVALUATION_STARTED342.json',dict(utc=utc(),all_choices_sha256=sha(root/'ALL_CHOICES_SEALED342.json'),final_source_sha256=sha(path)))
    data=load(path);mapping=read(root/'POOL_MAPPING342.json');catalog={'common_p00':root/'reference/p00'}
    for case,row in mapping.items():
        for ident,value in read(Path(row['path'])/'broker/STATE.json')['programs'].items():
            if ident!='p00':catalog[case+'__'+ident]=Path(value['path'])
    def forward(item):
        key,program=item;manifest=read(program/'REPORT.json')
        for relative,digest in manifest['artifacts'].items():assert sha(program/relative)==digest
        engine=Runtime(root);engine.deadline=time.monotonic()+450;folder=root/'evaluation'/key
        try:
            a,_=engine.predict(folder/'forward',program/'source',program/'full_fit/artifacts',data)
            b,_=engine.predict(folder/'repeat',program/'source',program/'full_fit/artifacts',data);assert np.array_equal(a,b)
            result=score(data,a);independent=independent_score(data,a);assert result['confusion']==independent['confusion'] and result['record_error']==independent['record_error']
            result.update(status='complete',cold_repeat_identical=True,independent_vote_verified=True,actual_prediction_processes=2*len(result['records']))
        except Exception as error:result=dict(status='evaluation_error',error_type=type(error).__name__,error=str(error))
        save(folder/'RESULT.json',result);return key,result
    metrics={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(forward,item) for item in catalog.items()]):
            key,value=f.result();metrics[key]=value;save(root/'EVALUATION_PROGRESS342.json',metrics)
    assert metrics['common_p00']['status']=='complete';baseline=metrics['common_p00']['record_error'];cases=read(root/'public/cases.json');bycase={c['id']:c for c in cases}
    def measure(rows):
        results=[]
        for row in rows:
            get=lambda i:metrics['common_p00' if i=='p00' else row['case']+'__'+i]
            value=get(row['selected']);valid=[get(i)['record_error'] for i in bycase[row['case']]['candidate_ids'] if get(i)['status']=='complete'];oracle=min(valid)
            v=dict(row,evaluation_status=value['status'])
            if value['status']=='complete':
                loss=value['record_error'];v.update(record_error=loss,selection_loss=loss-oracle,harm=max(0,loss-baseline),gain_capture=max(0,baseline-loss)/(baseline-oracle) if baseline>oracle else None)
            results.append(v)
        return results
    selector_results=[]
    for episode in read(root/'AGENT_CHOICES.json')['episodes']:
        selector_results.append(dict(id=episode['id'],condition=episode['condition'],completed=episode['completed'],
            rows=measure(episode['choice']['choices']) if episode.get('choice') else None,seconds=episode['seconds']))
    fixed={name:measure(rows) for name,rows in read(root/'FIXED_CHOICES.json')['choices'].items()}
    matrix={}
    for generator in cfg['arms']:
        matrix[generator]={}
        ids={case for case,row in mapping.items() if row['arm']==generator}
        for selector in cfg['selector_conditions']:
            episodes=[e for e in selector_results if e['condition']==selector];rows=[r for e in episodes if e['completed'] for r in e['rows'] if r['case'] in ids]
            complete=len(rows)==4 and all(r['evaluation_status']=='complete' for r in rows)
            matrix[generator][selector]=dict(complete=complete,mean_record_error=float(np.mean([r['record_error'] for r in rows])) if complete else None,
                common_pool_mean_selection_loss=float(np.mean([r['selection_loss'] for r in rows])) if complete else None,
                mean_harm=float(np.mean([r['harm'] for r in rows])) if complete else None)
    receipts=[read(p) for p in root.glob('subjects/*/model_gateway/*/*_receipt.json')]+[read(p) for p in root.glob('episodes/*/model_gateway/*/*_receipt.json')]
    fit_audits=[read(p) for p in root.rglob('training_audit.json') if read(p).get('mode')=='fit']
    result=dict(iteration='A342',utc=utc(),status='complete',physical_development_cohorts=1,independent_confirmation_scenarios=0,
        baseline_record_error=baseline,generator_selector_matrix=matrix,selectors=selector_results,fixed_controls=fixed,
        source_qualification='Author-acquired raw CSV,CRC/SHA/full dual-reader/timegrid verified;whole source ZIP MD5 not verified.',
        own_pool_regrets_not_cross_generator_improvement=True,related_pool_count=4,all_program_results=metrics,
        cost=dict(requests=len(receipts),usage_complete=all(r.get('usage') for r in receipts),HTTP200=sum(r.get('upstream_status')==200 for r in receipts),
            input_tokens=sum((r.get('usage') or {}).get('input_tokens',0) for r in receipts),output_tokens=sum((r.get('usage') or {}).get('output_tokens',0) for r in receipts),
            actual_fit_entries=len(fit_audits),reused_baseline_fit_entry_records=3 if cfg.get('reuse_baseline_from') else 0,
            new_fit_entry_executions=len(fit_audits)-(3 if cfg.get('reuse_baseline_from') else 0),
            completed_estimator_fits=sum(a['completed_estimator_fits'] for a in fit_audits),
            actual_optimizer_steps=sum(a['optimizer_steps'] for a in fit_audits),final_prediction_processes=sum(v.get('actual_prediction_processes',0) for v in metrics.values())),
        novelty_established=False,majority_goal_complete=False,reserved338_performance_unopened=True)
    save(root/'RESULT342.json',result);save(OUT/'RESULT342.json',result);status(root,status='evaluation_complete',controller_status='terminal',goal_complete=False)
    print(json.dumps(dict(matrix=matrix,cost=result['cost'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','run','evaluate']);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--budget-profile',choices=['standard','adequacy'],default='standard');p.add_argument('--reuse-baseline-from',type=Path);p.add_argument('--result-key',default='controlled342')
    a=p.parse_args();root=a.run.resolve()
    key=a.result_key if a.action=='freeze' else read(root/'PROTOCOL342.json').get('result_key','controlled342')
    assert key.replace('_','').isalnum();OUT=PROJECT/'results'/key
    if a.action=='freeze':freeze(root,a.budget_profile,a.reuse_baseline_from,key)
    else:globals()[a.action](root)
