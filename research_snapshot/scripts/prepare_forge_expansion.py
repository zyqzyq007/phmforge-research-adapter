"""Expand PHMForge coverage using genuine CWRU and official NASA observations.

Reuse the isolated native coding executor and numerical backends. No final
scoring occurs in this controller; all programs are sealed together first.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import shutil

import numpy as np
from scipy.io import loadmat

from acquire_forge_pdmb_sources244 import save, sha256
from prepare_fault_extensions import features
from prepare_forge_coding_baseline import PYTHON, RUNTIME
from prepare_verified_benchmarks import extract_bound
from real_fault_model import train as train_fault
import real_rul_model as rul
from run_validation_transfer import prepare, bootstrap, freeze, metadata_gate, execute_wave, state

PROJECT = Path(__file__).resolve().parents[1]
ARCHIVES = Path('/data/yyf-zyq/PHM_20261010')
ORIGINAL = ARCHIVES/'20261023T105942Z_coverage306/source/original/source/official/ReActXen/src/reactxen/demo/intent_implementation_demo/scenarios/phm_scenarios.json'
NASA_TASKS = ['pdm_rul_001', 'pdm_cost_benefit_001', 'pdm_safety_001', 'pdm_safety_006']


def read(path):
    return json.loads(Path(path).read_text())


def configure(root):
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    config = dict(iteration='A307', result_directory='results/expansion307', planned_sessions=5,
        purpose='New original task coverage with real fitting, inference and bounded independent evaluation; no assumed Agent failure.',
        authorization='用户：把PHMforge能真实实验的全做了；允许 NASA 官方仿真数据，禁止占位返回和伪训练。',
        test_inputs_visible=False, candidate_bank_from_training_cv=False,
        cohorts={'c':dict(family='classification',specs=[dict(id='expand307_cost002',dataset='CWRU',task_id='pdm_cost_benefit_002')]),
                 'n':dict(family='cmapss',specs=[dict(id='expand307_'+t,task_id=t,dataset='CMAPSS_FD001') for t in NASA_TASKS])},
        cases=[dict(cohort='c',task_id='pdm_cost_benefit_002',dataset='CWRU')]+[
            dict(cohort='n',task_id=t,dataset='CMAPSS_FD001') for t in NASA_TASKS],
        session_budget=dict(subject_timeout_seconds=3600,mcp_tool_timeout_seconds=180,
            max_model_requests_per_subject=180,max_total_input_tokens_per_subject=32000000,
            max_output_tokens_per_request=32768,max_total_output_tokens_per_subject=524288,
            max_concurrency=5,local_max_concurrency=4,artifact_budget_bytes=40*1024**3),
        prior_source_results_known_to_researcher=True, independent_new_sources=False,
        stage_plan='One initial session per task; closed-choice audit; stop before final scores.',
        estimation=dict(max_requests=900,gpu_hours=0,numerical_threads_per_subject=2,
            max_wall_after_preparation_seconds=4200,artifact_cap_bytes=40*1024**3,
            api_cost='Bounded token/session caps; no unverified provider price assumed.'),
        scope='CMAPSS costs and safety are partial contracts, not certified optimal decisions or legal compliance.',
        metrics=['ID_coverage','MAE_cycles','RMSE_cycles','top10_recall','threshold20_detection',
                 'window_and_whole_record_accuracy','given_cost_arithmetic','unsupported_certification_claims'],
        no_preassumed_failure=True,no_automatic_review_or_retry=True)
    save(root/'CAMPAIGN_CONFIG.json',config)
    save(root/'audit/PREPARATION_PROTOCOL307.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        configuration_sha256=sha256(root/'CAMPAIGN_CONFIG.json'),no_new_final_scores_read=True,
        measured_and_official_simulation_sources_separate=True))
    shutil.copy2(ORIGINAL,root/'source/original75.json')
    shutil.copy2(PROJECT/'docs/phmforge_expansion307.md',root/'audit/PROTOCOL307.md')


def cwru(root):
    source=root/'source/cwru'
    numbers=[105,118,130,169,185,197,209,222,234]
    bindings=extract_bound(ARCHIVES/'20261023T015909Z_cwrunative299',source,
        ['source/snapshot/cwru_'+str(n)+'.mat' for n in numbers])
    assert len(bindings)==9
    save(root/'audit/CWRU_SOURCE_BINDINGS.json',bindings)
    data=root/'c/restored/CWRU/data';data.mkdir(parents=True)
    parts={k:[] for k in ['train','val','test']};native=[]
    for n,kind,part in zip(numbers,['inner_race','ball_race','outer_race']*3,['train']*3+['val']*3+['test']*3):
        path=source/'source/snapshot'/('cwru_'+str(n)+'.mat')
        mat=loadmat(path);keys=[k for k in mat if k.endswith('_DE_time')];assert len(keys)==1
        wave=np.asarray(mat[keys[0]],np.float64).ravel();assert np.isfinite(wave).all()
        unit=sha256(path)[:24]
        native.append(dict(number=n,label=kind,part=part,unit_id=unit,array_key=keys[0],
            samples=len(wave),file_sha256=sha256(path)))
        for start in range(0,len(wave)-2048+1,2048):
            x=wave[start:start+2048]
            parts[part].append(dict(sample_id=secrets.token_hex(16),group=unit,unit_id=unit,
                label=kind,features=features(x[:,None]),waveforms=x.astype(np.float32),start_sample=start))
    for part,rows in parts.items():
        np.random.default_rng(2026102307).shuffle(rows)
        values={k:np.asarray([x[k] for x in rows]) for k in ['features','waveforms','unit_id','start_sample']}
        values['unit_ids']=values.pop('unit_id');values['sample_ids']=np.asarray([x['sample_id'] for x in rows])
        values['dataset']=np.asarray('CWRU')
        if part!='test':values.update(labels=np.asarray([x['label'] for x in rows]),groups=np.asarray([x['group'] for x in rows]))
        np.savez_compressed(data/(part+'.npz'),**values)
    save(data.parent/'hidden/labels.json',[{k:v for k,v in x.items() if k not in ['features','waveforms']} for x in parts['test']])
    info=dict(dataset='CWRU',seed=2026102307,class_names=['inner_race','outer_race','ball_race'],
        architectures=['mlp'],source_sha256=sha256(root/'audit/CWRU_SOURCE_BINDINGS.json'),
        rows_by_split={k:len(v) for k,v in parts.items()},
        group_definition='Whole official recording; all windows of a recording remain together. Not certified independent physical hardware.',
        sampling_hz=12000,channels=['native drive-end acceleration'],
        preprocessing='Nonoverlapping2048samples; seven time and eight FFT features; no full-record normalization.',
        split_rule='0HP records: 0.007inch fitting,0.014inch development,0.021inch final. Fault diameter shift known before fitting.',
        deployment_fault_diameter_inches=0.021,source_url='https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data',
        feature_names=['mean','std','rms','peak_abs','peak_to_peak','skewness','kurtosis']+[f'log1p_fft_band_{i}' for i in range(8)],
        maintenance_cost_by_fault={'inner_race':3000,'outer_race':4000,'ball_race':5000},
        failure_cost_by_fault={'inner_race':30000,'outer_race':40000,'ball_race':50000},
        cost_provenance='Amounts explicitly given by original task procedure; not measured bills or causal maintenance savings.',
        original_benchmark_score_claimed=False)
    save(data/'dataset.json',info);save(root/'audit/CWRU_NATIVE_RECORDS.json',native)
    audit=train_fault(data,data.parent/'reference','mlp',50)
    assert audit['optimizer_steps']>0 and audit['nonzero_gradient_steps']>0 and audit['changed_parameter_tensors']>0
    return dict(dataset='CWRU',optimizer_steps=audit['optimizer_steps'],rows=info['rows_by_split'])


def nasa(root):
    source=root/'source/nasa';data=source/'datasets/CMAPSS_FD001';data.mkdir(parents=True)
    shared=RUNTIME/'public/datasets/CMAPSS_FD001'
    for f in ['train.txt','training_config.json']:shutil.copy2(shared/f,data/f)
    assert sha256(data/'train.txt')=='963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8'
    assert sha256(shared/'test.txt')=='3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851'
    cfg=read(data/'training_config.json');cfg['epochs']=50;save(data/'training_config.json',cfg)
    raw=rul.read_series(data/'train.txt');split_counts={}
    for part,ids in [('train',cfg['train_engine_ids']),('val',cfg['validation_engine_ids'])]:
        windows,labels,records=rul.development_windows(raw,ids,cfg,validation=part=='val')
        values=dict(windows=windows,labels=labels,unit_ids=np.asarray([x['unit_id'] for x in records]),
            endpoint_cycles=np.asarray([x['endpoint_cycle'] for x in records]),
            groups=np.asarray(['train_engine_'+str(x['unit_id']) for x in records]),
            sample_ids=np.asarray([f"{part}:{x['unit_id']}:{x['endpoint_cycle']}" for x in records]),dataset=np.asarray('CMAPSS_FD001'))
        np.savez_compressed(data/(part+'.npz'),**values);split_counts[part]=len(records)
    info=dict(dataset='CMAPSS_FD001',format='cmapss_npz',data_origin='Official NASA C-MAPSS simulator; not field measurements.',
        source_url='https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/',
        architectures=['mlp','lstm','transformer'],group_definition='Complete simulated engine; fitting80 engines and development20 engines disjoint.',
        train_sha256=sha256(data/'train.txt'),window=cfg['window'],sensor_numbers=rul.SENSORS,
        window_units='Native NASA sensors; target is uncapped remaining operating cycles.',
        normalization='Fit only80 fitting engines. Training target capped125; final target uncapped.',
        fitting_engine_ids=cfg['train_engine_ids'],development_engine_ids=cfg['validation_engine_ids'],
        rows_by_split=split_counts,final_test_units=100,public_test_labels=False,
        row_identity='sample_id names partition, original unit_id and causal endpoint_cycle.',
        source_known_to_researcher=True,not_new_independent_confirmation=True,
        original_benchmark_score_claimed=False)
    save(data/'dataset.json',info)
    registry={'CMAPSS_FD001':dict(task='rul',format='cmapss_npz',path='datasets/CMAPSS_FD001',
        architectures=info['architectures'],available_splits=['train','val'],test_inputs_visible=False,
        data_origin=info['data_origin'])}
    save(source/'dataset_registry.json',registry);shutil.copy2(RUNTIME/'public/tool_catalog.json',source/'tool_catalog.json')
    final=root/'source/nasa_evaluation/datasets/CMAPSS_FD001';final.mkdir(parents=True)
    test=rul.read_series(shared/'test.txt');units=sorted(np.unique(test[:,0]).astype(int))
    endpoints=[int(np.count_nonzero(test[:,0]==u)) for u in units]
    windows=np.asarray([rul.window_at(test[test[:,0]==u],n,cfg['window']) for u,n in zip(units,endpoints)],np.float32)
    sample_ids=np.asarray([f'test:{u}:{n}' for u,n in zip(units,endpoints)])
    np.savez_compressed(final/'test.npz',windows=windows,unit_ids=np.asarray(units),endpoint_cycles=np.asarray(endpoints),
        sample_ids=sample_ids,dataset=np.asarray('CMAPSS_FD001'))
    shutil.copy2(data/'dataset.json',final/'dataset.json')
    eval_root=final.parents[1];registry['CMAPSS_FD001'].update(available_splits=['test'],test_inputs_visible=True)
    save(eval_root/'dataset_registry.json',registry);shutil.copy2(source/'tool_catalog.json',eval_root/'tool_catalog.json')
    hidden=root/'n/hidden/CMAPSS_FD001';hidden.mkdir(parents=True)
    bindings=extract_bound(ARCHIVES/'20261018T170243Z_fullbench187',root/'source/gold', ['runtime/forge_data/RUL_FD001.txt'])
    assert len(bindings)==1;labels_path=Path(bindings[0]['destination'])
    assert sha256(labels_path)=='a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca'
    labels=np.loadtxt(labels_path);assert len(labels)==len(units)==100 and np.isfinite(labels).all()
    save(hidden/'labels.json',[dict(sample_id=str(s),unit_id=int(u),label=float(y),actual_rul=float(y)) for s,u,y in zip(sample_ids,units,labels)])
    save(root/'audit/CMAPSS_SOURCE_BINDINGS.json',dict(train_sha256=sha256(data/'train.txt'),
        test_sha256=sha256(shared/'test.txt'),label_bindings=bindings,config_sha256=sha256(data/'training_config.json'),
        selection_source='Fixed prior source-qualified configuration; no tuning from final scores.',
        independent_new_source=False))
    # The trainer receives only the public data path, never any hidden-label path.
    audit=rul.train(data,root/'source/nasa_reference','transformer',epochs_override=50)
    assert audit['optimizer_steps']>0 and audit['nonzero_gradient_steps']>0 and audit['changed_parameter_tensors']>0
    return dict(dataset='CMAPSS_FD001',optimizer_steps=audit['optimizer_steps'],training_engines=80,development_engines=20,final_units=100)


def install_cost(root):
    cohort=root/'c';job,=read(cohort/'PLAN187.json')['jobs'];ws=cohort/job['workspace']
    prompt=Path(job['prompt_file']);text=prompt.read_text()
    text+='''
Cost-task contract: one physical test recording is one task bearing, identified by observation unit_ids. Windows are repeated observations; do not charge each window as another bearing. Preserve every original window prediction and aggregate a unit decision. dataset.json contains the original task's maintenance/failure dollar amounts by fault type. These are given scenario amounts, not measured bills. For the requested cost-benefit ratio use failure_cost/maintenance_cost; equal ratios form a tied priority tier and must not be declared a unique winner. Other prioritization criteria may be discussed separately with explicit assumptions. Real failure probabilities, intervention effects, downtime and a decision horizon are unavailable, so actual expected ROI and globally optimal scheduling are unidentifiable.
Also include unit_decisions in the output JSON: one entry per unique unit_id, with predicted_label, maintenance_cost, failure_cost, avoided_cost_if_failure_prevented, cost_benefit_ratio and priority_tier. Include cost_totals and best_ratio_fault_types. The external evaluator scores whole-record classifications and independently checks given-cost arithmetic and ties. Do not invent RUL labels for CWRU or copy the original scenario's unrelated RUL answer template. Validate the full submitted program, including aggregation and costs, on allowed development observations.
'''
    prompt.write_text(text)
    (ws/'solution.py').write_text('''import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,'/tools')
from real_fault_model import infer
MAINT={'inner_race':3000,'outer_race':4000,'ball_race':5000}
FAIL={'inner_race':30000,'outer_race':40000,'ball_race':50000}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 value=infer(Path('/reference/model.pt'),a.input,a.output)
 with np.load(a.input,allow_pickle=False) as data:
  units=data['unit_ids'].astype(str);ids=data['sample_ids'].astype(str)
 pred={r['sample_id']:r['predicted_label'] for r in value['predictions']};dec=[]
 for u in sorted(set(units)):
  votes=Counter(pred[s] for s,v in zip(ids,units) if v==u);label=sorted(votes,key=lambda c:(-votes[c],c))[0]
  dec.append(dict(unit_id=u,predicted_label=label,maintenance_cost=MAINT[label],failure_cost=FAIL[label],avoided_cost_if_failure_prevented=FAIL[label]-MAINT[label],cost_benefit_ratio=FAIL[label]/MAINT[label],priority_tier=1))
 value.update(unit_decisions=dec,cost_totals={'maintenance':sum(x['maintenance_cost'] for x in dec),'failure':sum(x['failure_cost'] for x in dec)},best_ratio_fault_types=sorted(MAINT),cost_scope='Given costs, conditional avoided cost if a failure is prevented; no expected ROI or real optimal policy claimed.')
 a.output.write_text(json.dumps(value)+'\\n')
''')
    metadata_gate(cohort);freeze(cohort)


def install_nasa(root):
    cohort=root/'n';bootstrap(root,cohort,'cmapss','initial')
    template=root/'c';protocol=read(template/'PROTOCOL187.json');protocol.update(read(cohort/'BOOTSTRAP_PROTOCOL.json'))
    protocol.update(run_id=cohort.name,max_new_model_sessions=4,local_max_concurrency=4,cli=read(root/'audit/CLI_ONLINE_CHECK.json')['CLI_path'],
        no_scores_until_all_submissions_frozen=True)
    save(cohort/'PROTOCOL187.json',protocol)
    for f in ['cli_config187.toml','model_catalog187.json']:shutil.copy2(template/f,cohort/f)
    shutil.copytree(template/'public_tools',cohort/'public_tools')
    tasks={x['task_id']:x for x in read(ORIGINAL)['pdm_scenarios']};jobs=[];routes=[]
    common='''
This is a source-bound PHMForge research adaptation using official NASA C-MAPSS simulator observations, not field data. All numerical tools train and infer genuinely. Original question above is preserved. The independent evaluator, rather than the Agent, receives final observations/labels after all submissions are sealed. No final scores or labels are available now. Do not fabricate test_predictions.json or final MAE/RMSE. Source and interface details are in /data/datasets/CMAPSS_FD001/dataset.json and training_config.json.
The MCP server provides genuine loading, train_rul_model (MLP/LSTM/Transformer), predict_rul and identity-aligned development verification. /reference/model.pt is a genuinely trained50-epoch Transformer candidate, with actual loss/gradient/parameter audit. You may retain, retrain or replace it with another genuinely fitted predictor. Preserve80 fitting engines and20 disjoint development engines; fit preprocessing and predictors only on the fitting IDs. Raw train.txt includes complete allowed learning histories; it contains both groups and must respect that split. Development labels support selection, never a train+development refit. Final inference has no training data. The train/val NPZ files provide causal30x14 sensor windows, unit_ids, endpoint_cycles and unique sample_ids; train/val labels are uncapped remaining cycles, and train-only normalization/training cap125 are documented. Do not use validation labels as model inputs.
Submit solution.py runnable as python3 solution.py --input OBSERVATION_NPZ --output OUTPUT_JSON. Each input row must produce exactly one prediction with sample_id, unit_id and finite nonnegative predicted_rul in cycles. Read observation windows (no future or targets); load saved fitted artifacts. New fitting on final observations is prohibited. Save all code/checkpoints, validation.json and report.md with real held-out-engine development MAE/RMSE, exact program validation and evidence limits. The evaluator reports all100 final engine errors, MAE/RMSE, bias, top10 overlap and any explicitly requested threshold checks. Original expected error ranges are context, not a reason to reject smaller errors. No arbitrary positive lower error bound.
Use /opt/python312/bin/python3.12 with NumPy,SciPy,scikit-learn,PyTorch. /tools contains genuine numerical source; /data and /reference are read-only. All new files under /app, temporary files only /app/.runtime, never /tmp. No network, package installation, hidden files, other tasks, or additional model APIs. Budget3600seconds,180model requests,32768output/request,524288total output,32000000completed input tokens,CPU2threads. Save a runnable result before deadline.
'''
    addenda={
        'pdm_rul_001':'''Also output top_10_risk_units, the10 unit IDs sorted by (predicted_rul,unit_id). All final metrics are computed externally. Do not guess actual RUL or report development error as final test error.''',
        'pdm_cost_benefit_001':'''The original procedure supplies maintenance_cost_per_unit=5000, failure_cost_per_unit=50000 and a stated RUL threshold20. Output cost_analysis with preventive_cost=5000 times the number of units and reactive_cost=50000 times the number with predicted_rul<20, plus the predicted_failures IDs. Label these as original scenario arithmetic, not validated physical failure probability or equal-horizon policy costs. No decision horizon, action-dependent failure distribution or intervention efficacy is supplied. If a cost-optimal threshold or optimized savings cannot be identified, return null and explain the missing assumptions instead of inventing an optimizer result. A nominal20 threshold can be reported as a given rule, not a proven optimum.''',
        'pdm_safety_001':'''Output high_risk_units containing exactly predicted_rul<20, and risk_assessment/compliance_status/safety_recommendations with evidence scope. The20-cycle rule is specified by the question. No FAA/EASA clauses, equipment certification documents, operating context, catastrophic-failure probabilities or environmental consequence model are provided. Unsupported compliance and calibrated physical risk must remain unknown/null, not arbitrary booleans or invented four-level truth. Qualitative precautionary advice must be distinguished from a certified maintenance decision.''',
        'pdm_safety_006':'''The short original question gives no unique risk threshold, four-level risk calibration or applicable compliance clauses. Supply real RUL predictions and document conditional risk indicators and evidence needed. Do not assert compliant/noncompliant equipment without actual auditable evidence. Keep compliance_status unknown/null where unsupported and explain safety recommendation limits.'''}
    for task in NASA_TASKS:
        ident='expand307_'+task;ws=cohort/'workspaces'/ident
        for f in ['.runtime/tmp','.runtime/mpl','.runtime/xdg','artifacts']:(ws/f).mkdir(parents=True,exist_ok=True)
        (ws/'TASK.txt').write_text(tasks[task]['input_question']+'\n')
        (ws/'AGENTS.md').write_text('Use /app/.runtime for all temporary files, never /tmp. Fit only listed fitting engines; development for selection. No final observations/labels/scores, network, packages or other model APIs. Save a fitted runnable solution.\n')
        (ws/'solution.py').write_text('''import argparse,sys
from pathlib import Path
sys.path.insert(0,'/tools')
from real_rul_model import infer
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();infer(Path('/reference/model.pt'),a.input,a.output)
''')
        reference=cohort/'references'/ident;reference.mkdir(parents=True)
        for f in ['model.pt','training_audit.json']:shutil.copy2(root/'source/nasa_reference'/f,reference/f)
        public=cohort/'public_data'/ident;public.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(root/'source/nasa',public)
        evaluation=cohort/'evaluation_data'/ident;evaluation.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(root/'source/nasa_evaluation',evaluation)
        prompt=cohort/'plans'/(ident+'.txt');prompt.parent.mkdir(parents=True,exist_ok=True)
        prompt.write_text(tasks[task]['input_question']+'\n'+common+'\n'+addenda[task]+'\n')
        mounts=cohort/'plans'/(ident+'_mounts.json')
        values={'mounts':[dict(source=str(PYTHON),destination='/opt/python312'),dict(source=str(RUNTIME/'deps'),destination='/opt/phm_python/extra'),
            dict(source=str(cohort/'public_tools'),destination='/tools'),dict(source=str(public),destination='/data'),dict(source=str(reference),destination='/reference')]}
        save(mounts,values)
        eval_mounts=cohort/'plans'/(ident+'_evaluation_mounts.json')
        values['mounts'][3]['source']=str(evaluation);save(eval_mounts,values)
        native=dict(id=ident,family='forge_real_rul_coding',task_id=task,condition='coding_baseline',repeat=1,
            workspace=str(ws),prompt_file=str(prompt),answer_mode='text')
        plan=cohort/'plans'/(ident+'.json');save(plan,{'jobs':[native]})
        socket=cohort/'ipc'/('s'+str(len(jobs))+'.sock')
        jobs.append(dict(native,workspace=str(ws.relative_to(cohort)),plan=str(plan),mounts=str(mounts),
            evaluation_mounts=str(eval_mounts),evaluation_test=str(evaluation/'datasets/CMAPSS_FD001/test.npz'),
            model_gateway_socket=str(socket),dataset='CMAPSS_FD001',data_format='cmapss_npz',
            test_inputs_visible_during_development=False,partial_decision_contract=task!='pdm_rul_001'))
        routes.append(dict(job=ident,socket=str(socket)))
    save(cohort/'PLAN187.json',{'jobs':jobs})
    gateway=read(template/'MODEL_GATEWAY_CONFIG.json');gateway['routes']=routes;save(cohort/'MODEL_GATEWAY_CONFIG.json',gateway)
    audit=read(root/'source/nasa_reference/training_audit.json')
    save(cohort/'TASK_QUALIFICATION251.json',dict(passed=True,real_optimizer_steps=audit['optimizer_steps'],
        real_nonzero_gradient_steps=audit['nonzero_gradient_steps'],source_simulator='Official NASA C-MAPSS FD001',
        source_authorization='允许 NASA 官方仿真数据，禁止占位返回和伪训练',
        tasks=NASA_TASKS,final_labels_external=True,full_decision_or_compliance_contract_qualified=False,
        claim_scope='Real numerical RUL and specified arithmetic/thresholds; no physical policy optimum or legal certification.',
        original_benchmark_score_claimed=False))
    save(cohort/'audit/AUTHORIZATION251.json',dict(user_reply=read(root/'CAMPAIGN_CONFIG.json')['authorization'],max_new_primary=4))
    save(cohort/'OWNERSHIP.json',dict(owner='root',writers_stopped=True,status='prepared_waiting_isolation'))
    metadata_gate(cohort);freeze(cohort)


def build(root):
    configure(root)
    # Separate processes keep Torch/NumPy random states independent.
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(cwru,root),pool.submit(nasa,root)]
        rows=[f.result() for f in futures]
    prepare(root,root/'c','classification','initial');install_cost(root);install_nasa(root)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    save(root/'PREPARATION307.json',dict(passed=True,rows=rows,planned_sessions=5,
        no_model_sessions_yet=True,no_new_final_scores_read=True))
    state(root,status='five_original_task_subjects_prepared',planned_model_sessions=5,new_model_sessions=0,
        original_denominator=75,measured_original_directions_before=13,
        source_policy='Official NASA simulation allowed; placeholders and sham training forbidden.')


def run(root):
    assert read(root/'PREPARATION307.json')['passed']
    execute_wave(root,'new_original_tasks',[root/'c',root/'n'])
    state(root,status='five_subjects_terminal_waiting_observable_review',completed_model_sessions=5,
        no_new_final_scores_read=True,next_action='Audit actual program choices and development checks; freeze before external numerical scoring.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--phase',required=True,choices=['build','run'])
    a=p.parse_args();(build if a.phase=='build' else run)(a.run.resolve())
