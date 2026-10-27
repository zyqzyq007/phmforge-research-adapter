"""Real source-bound numerical adaptations of18 PHMForge Engine questions.

Original questions are retained; the unverified named summaries are explicitly
distinguished from actual source sequences. Evaluation is publisher annotation
agreement, never certified physical causality. All original source IDs are
excluded from numerical fitting and development.
"""
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import re
import secrets
import shutil

import numpy as np

from acquire_forge_pdmb_sources244 import save,sha256
from prepare_verified_benchmarks import extract_bound
from prepare_forge_coding_baseline import RUNTIME
from real_engine_health_model import HEADS,fit_heads,infer
from run_validation_transfer import prepare,freeze,metadata_gate,execute_wave,state

PROJECT=Path(__file__).resolve().parents[1]
ARCHIVES=Path('/data/yyf-zyq/PHM_20261010')
ORIGINAL=ARCHIVES/'20261023T105942Z_coverage306/source/original/source/official/ReActXen/src/reactxen/demo/intent_implementation_demo/scenarios/phm_scenarios.json'


def read(path):return json.loads(Path(path).read_text())


def annotations(path,keep):
    result={}
    for line in Path(path).open():
        row=json.loads(line)
        if not isinstance(row['id'],str) or row['id'] not in keep:continue
        labels={};health=None;cause=None;cs=row['conversations']
        for i,c in enumerate(cs[:-1]):
            if c['from']!='human' or c.get('stage')!='2' or cs[i+1]['from']!='gpt':continue
            q=c['value'];ans=cs[i+1]['value'].strip();opts={a:b.strip() for a,b in re.findall(r'([a-z]):\s*([^\n]+)',q)}
            if 'health status' in q.lower():
                h=next(h for h in HEADS if re.search(r'\b'+h+r'\b',q,re.I))
                assert ans in ['a','b'];labels[h]=int(ans=='b')
            elif 'potential health state' in q.lower():health=opts.get(ans) if ans else None
            elif 'timing issue' in q.lower() and ans:
                tokens=[s.strip() for s in ans.split(',')];assert all(v in opts for v in tokens)
                cause='|'.join(opts[v] for v in tokens)
        assert set(labels)==set(HEADS)
        result[row['id']]=dict(bits=''.join(str(labels[h]) for h in HEADS),health_state=health,timing_causes=cause)
    assert set(result)==keep
    return result


def build(root,source):
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    for f in ['audit','logs','code','runtime/tmp','runtime/mpl','clients-cache']:(root/f).mkdir(parents=True,exist_ok=True)
    matrix=read(PROJECT/'results/forgesources244_qa_correction/SOURCE_MATRIX244.json')['engine_task_matrix']
    selected=[v for v in matrix if len(v['source_ids'])==1];assert len(selected)==18
    plan=read(source/'PROTOCOL309.json');assert len(plan['selection'])==2000
    reserved={str(v) for row in matrix for v in row['source_ids']};fit_ids={v['source_id'] for v in plan['selection']}
    assert fit_ids.isdisjoint(reserved)
    cases=[dict(cohort=f'e{i:02d}',task_id=v['task_id'],dataset='ENGINE_QA_HEALTH',source_id=v['source_ids'][0]) for i,v in enumerate(selected,1)]
    config=dict(iteration='A312',result_directory='results/enginecoding312',planned_sessions=18,
        purpose='18 original Engine question directions with actual N-CMAPSS-derived arrays, real supervised tools and external QA annotation agreement; disclosed source-input correction, not full official reproduction.',
        authorization='用户：把PHMforge能真实实验的全做了；允许 NASA 官方仿真数据，禁止占位返回和伪训练。',
        test_inputs_visible=False,candidate_bank_from_training_cv=False,cases=cases,
        cohorts={v['cohort']:dict(family='classification',specs=[dict(id='engine312_'+v['cohort'],dataset=v['dataset'],task_id=v['task_id'])]) for v in cases},
        session_budget=dict(subject_timeout_seconds=1200,max_model_requests_per_subject=60,
            max_total_input_tokens_per_subject=8000000,max_output_tokens_per_request=32768,
            max_total_output_tokens_per_subject=262144,mcp_tool_timeout_seconds=180,
            max_concurrency=8,local_max_concurrency=1,artifact_budget_bytes=2*1024**3),
        no_automatic_review_or_retry=True,all_choices_sealed_before_final_evaluation=True,
        independent_physical_engine_generalization_claimed=False,source_qa_not_physical_probability=True,
        source_qualification_protocol_sha256=sha256(source/'PROTOCOL309.json'),
        resource_estimate=dict(max_requests=1080,gpu_hours=0,numerical_threads_per_active_subject=2,
            max_wall_after_preparation_seconds=4500,shared_raw_sequence_bytes=160000000,
            artifact_cap_all_cohorts_bytes=36*1024**3,price='Provider monetary rate not assumed; token/request caps enforced.'),
        annotation_metrics=['five_component_agreement','joint_health_accuracy','overall_state_agreement','timing_QA_annotation_agreement','coverage'],
        warning='Named summaries were author-written; do not pretend they are verified channel decodings. Physical causal conclusions remain unqualified.')
    save(root/'CAMPAIGN_CONFIG.json',config)
    # A freshly checked CLI record is supplied before this build is invoked.
    assert read(root/'audit/CLI_ONLINE_CHECK.json')['latest_stable_version']=='0.164.0'
    data=root/'source/shared_public/datasets/ENGINE_QA_HEALTH';data.mkdir(parents=True)
    for f in ['train.npz','val.npz']:shutil.copy2(source/'public/datasets/ENGINE_QA_HEALTH'/f,data/f)
    ann=annotations(source/'source/official_train_qa.jsonl',fit_ids|{v['source_id'] for v in cases})
    sup={}
    for part in ['train','val']:
        arrays=dict(np.load(data/(part+'.npz'),allow_pickle=False))
        for sid,y in zip(arrays['sample_ids'],arrays['labels']):
            ident=str(sid).removeprefix('engine_qa_');assert str(y)==ann[ident]['bits']
            sup[str(sid)]=dict(ann[ident],part=part)
    assert len(sup)==2000
    save(data/'supervision.json',sup)
    identities=[];sequences=[]
    for part in ['train','val']:
        with np.load(data/(part+'.npz'),allow_pickle=False) as arrays:
            for sid in arrays['sample_ids']:
                ident=str(sid).removeprefix('engine_qa_');paths=list((source/'outputs').glob('native*/'+ident+'.npy'));assert len(paths)==1
                raw=np.load(paths[0],allow_pickle=False);assert raw.shape==(600,33) and np.isfinite(raw).all()
                identities.append(str(sid));sequences.append(raw.astype(np.float32))
    np.savez_compressed(data/'sequences.npz',sample_ids=np.asarray(identities),sequences=np.asarray(sequences))
    info=read(source/'public/datasets/ENGINE_QA_HEALTH/dataset.json')
    info.update(class_names=[''.join(v) for v in itertools.product('01',repeat=5)],
        rows_by_split=dict(train=1600,val=400,test=1),data_origin='Official N-CMAPSS-derived EngineMT-QA, not field measurements.',
        annotation_head_order=HEADS,source_pin='ea2051a15512212976c22c1a075ec549c7d48eef',
        feature_definition='198 values:33 anonymous channel means,std,min,max,first,last over600 original rows.',
        raw_sequence_file='sequences.npz; float32 views of publisher float64 arrays; no generated signals.',
        supervision='supervision.json contains only supplied fitting/development annotations. Missing cause/state labels remain null.',
        hidden_target_policy='All114 original-task source records excluded from fitting/development. Test labels and observations are external.',
        physical_channel_order_verified=False,physical_engine_grouping_verified=False,
        original_named_summary_source_bound=False,physical_causality_or_risk_certified=False,
        original_benchmark_score_claimed=False)
    save(data/'dataset.json',info)
    reference=root/'source/shared_reference';reference.mkdir()
    for f in ['model.pt','training_audit.json']:shutil.copy2(source/'reference'/f,reference/f)
    audit=fit_heads(data,data/'supervision.json',source/'reference/ordinary_extra_trees.joblib',reference/'engine_health.joblib')
    audit.update(source_component_joint_development_accuracy=0.81,
        supervised_source_IDs_exclude_all_originals=True,physical_engine_grouping_qualified=False,
        classifier_outputs_are_QA_annotations=True,reference_fits_reused_not18_independent_fits=True)
    save(reference/'engine_training_audit.json',audit)
    save(root/'audit/ENGINE_REFERENCE_QUALIFICATION312.json',audit)
    held=root/'source/held';prefixes=['outputs/engine_signals_all/'+v['source_id']+'.npy' for v in cases]
    bindings=extract_bound(ARCHIVES/'20261020T133326Z_forgesources244',held,prefixes);assert len(bindings)==18
    save(root/'audit/HELD_NATIVE_BINDINGS312.json',bindings)
    shared=root/'source/shared_public'
    # Genuine numerical MCP tools stay available. Unsupported physical health
    # calibration tools still return unavailable; Python may use fitted QA models.
    from prepare_forge_coding_baseline import RUNTIME
    save(shared/'dataset_registry.json',{'ENGINE_QA_HEALTH':dict(task='classification',path='datasets/ENGINE_QA_HEALTH',
        architectures=['mlp'],available_splits=['train','val'],test_inputs_visible=False,data_origin=info['data_origin'])})
    shutil.copy2(RUNTIME/'public/tool_catalog.json',shared/'tool_catalog.json')
    for case in cases:
        c=root/case['cohort'];target=c/'restored/ENGINE_QA_HEALTH';d=target/'data';d.mkdir(parents=True)
        for f in ['train.npz','val.npz','dataset.json']:shutil.copy2(data/f,d/f)
        (target/'reference').mkdir()
        for f in ['model.pt','training_audit.json']:shutil.copy2(reference/f,target/'reference'/f)
        raw=np.load(held/'outputs/engine_signals_all'/(case['source_id']+'.npy'),allow_pickle=False)
        feature=np.concatenate([raw.mean(0),raw.std(0),raw.min(0),raw.max(0),raw[0],raw[-1]]).astype(np.float32)
        sid=secrets.token_hex(16);np.savez_compressed(d/'test.npz',sample_ids=np.asarray([sid]),features=feature[None],
            sequences=raw.astype(np.float32)[None],dataset=np.asarray('ENGINE_QA_HEALTH'))
        label=ann[case['source_id']]
        core=next(v['ground_truth_core'] for v in selected if v['task_id']==case['task_id'])
        assert {h:('Fault' if b=='1' else 'Healthy') for h,b in zip(HEADS,label['bits'])}==core['component_health']
        assert label['health_state']==core['overall_health_state']
        save(target/'hidden/labels.json',[dict(sample_id=sid,label=label['bits'],group='held_'+secrets.token_hex(12),
            health_state=label['health_state'],timing_causes=label['timing_causes'],source_id=case['source_id'])])
        prepare(root,c,'classification','initial')
        job,=read(c/'PLAN187.json')['jobs'];ws=c/job['workspace']
        shutil.copy2(PROJECT/'scripts/real_engine_health_model.py',c/'public_tools/real_engine_health_model.py')
        (ws/'solution.py').write_text('''import argparse,sys
from pathlib import Path
sys.path.insert(0,'/tools')
from real_engine_health_model import infer
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();infer(Path('/reference/engine_health.joblib'),a.input,a.output)
''')
        prompt=Path(job['prompt_file'])
        with prompt.open('a') as stream:
            stream.write('''\nEngine source correction: the publisher's original question contains an author-written named-sensor summary, which is not an authenticated decoding of the33 columns. For this explicitly source-bound research adaptation, predict the five component health annotations, overall health-state annotation and timing-cause annotation from the actual source observations. Do not invent a channel-name map or certify physical causality. Original question is preserved as context; disagreement with its unbound summary must be disclosed. Evaluation is annotation agreement with the publisher QA for the actual record, not legal compliance, calibrated failure probability or proven physical cause.\n
The read-only /reference/engine_health.joblib contains genuinely trained ExtraTrees predictors. The five-component model achieved joint81percent and per-component91.5..94.25percent on400 held-out source records. Separate overall-state and timing-annotation model audits are in /reference/engine_training_audit.json. /tools/real_engine_health_model.py contains actual inference and head-training code. The starting solution uses this real fitted bundle; retain it or fit/code another predictor if justified. The MLP /reference/model.pt is a weaker genuinely trained numerical reference, not the starting selection. No new fitting is required merely to produce a result. Genuine MCP dataset/model/metric tools remain available; unavailable physical-calibration tools must not be replaced by invented readings.\n
The33 anonymous channels and198 computed features are real. Optional sequences.npz supplies all2000 fitting/development raw sequence views, keyed by sample_ids, and supervision.json supplies only those allowed annotations. IDs are not predictors. Head order for the joint binary predicted_label is Fan,LPC,HPC,HPT,LPT;0=Healthy,1=Fault. Also include component_health, faulty_components, overall_health_state and timing_cause_annotation for every observation. Timing labels describe publisher annotations, not established physical causality. Missing training annotations must remain missing rather than being fabricated. Predictions, checkpoints, development validation and an evidence-bounded report are required. The actual final source observation and all final annotations remain sealed until your saved program is frozen.\n''')
        for key in ['mounts']:
            path=Path(job[key]);mounts=read(path)
            for m in mounts['mounts']:
                if m['destination']=='/data':m['source']=str(shared)
                if m['destination']=='/reference':m['source']=str(reference)
            save(path,mounts)
        # Evaluation retains this task's single observation and the shared fit.
        path=Path(job['evaluation_mounts']);mounts=read(path)
        for m in mounts['mounts']:
            if m['destination']=='/reference':m['source']=str(reference)
        save(path,mounts)
        gate=read(c/'TASK_QUALIFICATION251.json');gate.update(source_type='official_NASA_simulator_derived',
            source_input_correction_disclosed=True,annotation_task_qualified=True,physical_causal_contract_qualified=False,
            actual_strong_starting_candidate='engine_health.joblib',starting_component_joint_development_accuracy=0.81)
        save(c/'TASK_QUALIFICATION251.json',gate)
        metadata_gate(c);freeze(c)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',
        shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared',retention='Pinned runtime/CLI retained; common raw training/reference files retained once inside this archive.')))
    state(root,status='18_source_bound_original_directions_prepared',planned_model_sessions=18,new_model_sessions=0,
        final_scores_opened=False,not_full_official_or_physical_causality_benchmark=True)


def build_trend(root,source):
    """Ten original trend directions; categorical QA agreement, not risk calibration."""
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    for f in ['audit','logs','code','runtime/tmp','runtime/mpl','clients-cache']:(root/f).mkdir(parents=True,exist_ok=True)
    qualification=read(source/'QUALIFICATION313.json');assert qualification['numerical_pipeline_qualified']
    original=read(ORIGINAL)['pdm_scenarios']
    cases=[dict(cohort=f't{i:02d}',task_id=f'engine_health_combined_{i+12:03d}',dataset='ENGINE_QA_TREND') for i in range(1,11)]
    cases=[dict(c,source_ids=next(r['source_engine_ids'] for r in original if r['task_id']==c['task_id'])) for c in cases]
    config=dict(iteration='A316',result_directory='results/enginetrendcoding316',planned_sessions=10,cases=cases,
        purpose='Ten original multi-cycle Engine directions with actual native prefixes and genuine models; publisher condition/probability-bin annotation agreement only.',
        authorization='用户：把PHMforge能真实实验的全做了；允许NASA官方仿真，禁止占位返回和伪训练。',
        test_inputs_visible=False,candidate_bank_from_training_cv=False,
        cohorts={c['cohort']:dict(family='classification',specs=[dict(id='enginetrend316_'+c['cohort'],dataset=c['dataset'],task_id=c['task_id'])]) for c in cases},
        session_budget=dict(subject_timeout_seconds=3600,max_model_requests_per_subject=180,max_total_input_tokens_per_subject=18000000,
            max_output_tokens_per_request=32768,max_total_output_tokens_per_subject=262144,mcp_tool_timeout_seconds=180,
            max_concurrency=8,local_max_concurrency=1,artifact_budget_bytes=4*1024**3),
        source_protocol_sha256=sha256(source/'PROTOCOL313.json'),physical_failure_probability_calibrated=False,
        source_input_correction_disclosed=True,reference_development=qualification,
        supplied_reference_is_weak=True,no_strong_solver_or_Agent_hardness_claim=True,
        no_automatic_final_scoring=True,no_automatic_review_or_retry=True)
    save(root/'CAMPAIGN_CONFIG.json',config)
    online=read(Path(read(PROJECT/'results/rulfollow315/current_run.json')['run'])/'audit/CLI_ONLINE_CHECK.json')
    assert sha256(Path(online['CLI_path']))==online['CLI_sha256'];save(root/'audit/CLI_ONLINE_CHECK.json',online)
    public=root/'source/shared_public';data=public/'datasets/ENGINE_QA_TREND';data.mkdir(parents=True)
    for f in ['train.npz','val.npz','dataset.json']:shutil.copy2(source/'public/datasets/ENGINE_QA_TREND'/f,data/f)
    info=read(data/'dataset.json');info['rows_by_split']['test']=1
    info['source_sha256']=sha256(source/'PROTOCOL313.json')
    info['source_sha256_scope']='Qualification and derivation protocol; not a claimed hash of the complete remote HDF5 file.'
    info.update(physical_probability_calibration=False,original_named_summary_not_verified=True,
        source_group_scope='No overlapping source-ID tuple graph groups across train/dev or original target groups; physical engine IDs unresolved.')
    save(data/'dataset.json',info)
    selection=read(source/'PROTOCOL313.json')['selection'];ids=sorted({i for r in selection for i in r['source_ids']},key=int)
    aliases={v:'prefix_'+__import__('hashlib').sha256(('A316:'+v).encode()).hexdigest()[:24] for v in ids}
    arrays=[]
    for v in ids:
        paths=list((source/'outputs').glob('native*/'+v+'.npy'));assert len(paths)==1
        a=np.load(paths[0],allow_pickle=False);assert a.shape==(60,33) and np.isfinite(a).all();arrays.append(a.astype(np.float32))
    np.savez_compressed(data/'cycle_prefixes.npz',sample_ids=np.asarray([aliases[v] for v in ids]),prefixes=np.asarray(arrays))
    save(data/'tuple_members.json',{r['sample_id']:[aliases[i] for i in r['source_ids']] for r in selection})
    save(public/'dataset_registry.json',{'ENGINE_QA_TREND':dict(task='classification',path='datasets/ENGINE_QA_TREND',architectures=['mlp'],available_splits=['train','val'],test_inputs_visible=False)})
    shutil.copy2(RUNTIME/'public/tool_catalog.json',public/'tool_catalog.json')
    reference=root/'source/shared_reference';shutil.copytree(source/'reference',reference)
    held=root/'source/held'
    held_ids=sorted({i for c in cases for i in c['source_ids']},key=int)
    if not (root/'audit/HELD_SOURCE_BINDINGS316.json').exists():
        source_bindings=extract_bound(ARCHIVES/'20261020T133326Z_forgesources244',held,
            ['outputs/engine_signals_all/'+i+'.npy' for i in held_ids])
        save(root/'audit/HELD_SOURCE_BINDINGS316.json',source_bindings)
    heldroot=held/'outputs/engine_signals_all'
    bindings=[]
    for case in cases:
        run=root/case['cohort'];d=run/'restored/ENGINE_QA_TREND/data';d.mkdir(parents=True)
        for f in ['train.npz','val.npz','dataset.json']:shutil.copy2(data/f,d/f)
        ref=d.parent/'reference';ref.mkdir()
        for f in ['model.pt','training_audit.json']:shutil.copy2(reference/f,ref/f)
        raw=[]
        for ident in case['source_ids']:
            path=heldroot/(ident+'.npy');a=np.load(path,allow_pickle=False)
            assert a.shape==(600,33) and np.isfinite(a).all()
            raw.append(a[:60]);bindings.append(dict(source_id=ident,path=str(path),sha256=sha256(path)))
        feature=np.concatenate([np.concatenate([a.mean(0),a.std(0),a.min(0),a.max(0),a[0],a[-1]]) for a in raw]).astype(np.float32)
        sid=secrets.token_hex(16)
        np.savez_compressed(d/'test.npz',sample_ids=np.asarray([sid]),features=feature[None],sequences=np.concatenate(raw).astype(np.float32)[None],dataset=np.asarray('ENGINE_QA_TREND'))
        gold=next(r['ground_truth'] for r in original if r['task_id']==case['task_id'])
        label=gold['qualitative_condition']+'|'+gold['failure_probability_range'];assert label in info['class_names']
        save(d.parent/'hidden/labels.json',[dict(sample_id=sid,label=label,group='held_'+secrets.token_hex(12),
            condition=gold['qualitative_condition'],probability_bin=gold['failure_probability_range'])])
        prepare(root,run,'classification','initial');job,=read(run/'PLAN187.json')['jobs'];ws=run/job['workspace']
        (ws/'solution.py').write_text("import argparse,json,sys\nfrom pathlib import Path\nsys.path.insert(0,'/tools')\nfrom frozen_classifier import infer\np=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)\na=p.parse_args();v=infer('/reference/ordinary_extra_trees.joblib',a.input,a.output)\nfor r in v['predictions']:\n r['qualitative_condition'],r['failure_probability_bin_annotation']=r['predicted_label'].split('|');r['calibrated_failure_probability']=None\nv['scope']='Publisher categorical QA annotations,not calibrated physical failure risk';a.output.write_text(json.dumps(v,indent=2)+'\\n')\n")
        with Path(job['prompt_file']).open('a') as stream:
            stream.write("""
Engine trend source correction: the original named-sensor text summary is author-written and is not an authenticated decoding of the33 anonymous channels. For this source-bound adaptation predict the actual source tuple's publisher qualitative-condition and probability-bin annotations. A bin annotation is not a calibrated physical failure probability. Do not claim established physical causality or certified maintenance benefit.
There are5381 fitting tuples and1985 development tuples built from1720 real60x33 prefixes. Overlapping source-ID tuple graph components are separated; original-task connected groups are excluded. Physical engine grouping remains unresolved. Each tuple has1980 actual per-cycle statistics; cycle_prefixes.npz and tuple_members.json expose permitted raw prefix views and their order. IDs are not predictors.
The supplied real ExtraTrees starting predictor has only36.83percent joint development annotation accuracy; the genuine50epoch MLP has17.28percent. These are weak numerical references. You may keep one, fit/code another real predictor, or disclose that the available evidence does not support reliable generalization. No strong-solver or physical-risk claim is implied.
The predicted_label is the exact combined string condition|probability_bin from dataset.json. Also include qualitative_condition, failure_probability_bin_annotation and calibrated_failure_probability(null when unmeasured). Preserve saved fitted code and evidence. Report annotation agreement; probability calibration and verified optimal maintenance are outside the available contract.
""")
        for key in ['mounts','evaluation_mounts']:
            path=Path(job[key]);mounts=read(path)
            for m in mounts['mounts']:
                if m['destination']=='/reference':m['source']=str(reference)
                if key=='mounts' and m['destination']=='/data':m['source']=str(public)
            save(path,mounts)
        gate=read(run/'TASK_QUALIFICATION251.json');gate.update(annotation_task_qualified=True,physical_risk_calibration_qualified=False,supplied_reference_is_weak=True)
        save(run/'TASK_QUALIFICATION251.json',gate);metadata_gate(run);freeze(run)
    save(root/'audit/HELD_BINDINGS316.json',bindings);shutil.copy2(__file__,root/'code'/Path(__file__).name)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared',retention='Pinned runtime; actual shared learning prefixes, QA targets and fitted models retained.')))
    state(root,status='ten_multi_cycle_original_directions_prepared',new_model_sessions=0,final_scores_opened=False,supplied_reference_is_weak=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--phase',choices=['build','run'],required=True);p.add_argument('--kind',choices=['single','trend'],default='single')
    a=p.parse_args();root=a.run.resolve()
    if a.phase=='build':(build_trend if a.kind=='trend' else build)(root,a.source.resolve())
    else:execute_wave(root,'multi_cycle_engine_directions' if a.kind=='trend' else '18_original_engine_directions',
        [root/c['cohort'] for c in read(root/'CAMPAIGN_CONFIG.json')['cases']])
