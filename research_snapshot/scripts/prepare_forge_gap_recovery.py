"""Recover A321 measured-source gaps using the existing isolated coding runner.

Two declared source/target adaptations; five original questions. No equivalence
to a full original PHMForge evaluation, and no automatic final-score access.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import numpy as np
import pandas as pd
from acquire_forge_pdmb_sources244 import save, sha256
from prepare_forge_coding_baseline import PYTHON, RUNTIME
from prepare_verified_benchmarks import extract_bound
from run_validation_transfer import freeze, metadata_gate, execute_wave, state
from real_gap_rul_model import train, fit_bank

PROJECT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text())


def make_rows(name, trajectories, selection, endpoints):
    features=[];labels=[];units=[];sample_ids=[];history=[];offsets=[0];steps=[]
    for unit, indices in selection.items():
        a=trajectories[unit]
        for i in indices:
            features.append(a[i]);labels.append(len(a)-1-i);units.append(unit)
            sample_ids.append('u%03d_q%06d'%(unit,i));steps.append(i)
            if endpoints:history.append(a[:i+1]);offsets.append(offsets[-1]+i+1)
    value=dict(dataset=np.asarray(name),features=np.asarray(features,dtype=np.float32),
        labels=np.asarray(labels,dtype=float),unit_ids=np.asarray(units),
        sample_ids=np.asarray(sample_ids),scenario_ids=np.ones(len(units),dtype=int),
        elapsed_acquisitions=np.asarray(steps,dtype=int))
    if endpoints:value.update(history_features=np.concatenate(history),history_offsets=np.asarray(offsets))
    return value


def prepare_sources(root):
    assert not (root/'SPLIT_AND_TARGET_FREEZE321.json').exists()
    eligible=[f'B{i:02}' for i in range(1,18) if i not in [1,5,16]]
    order=np.random.default_rng(20261024).permutation(eligible).tolist()
    split=dict(fit=order[:7],development=order[7:9],final=order[9:])
    declaration=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),LDM=split,
        source_commit='d7167c927aad5b7f197b1d735b21339e3e17844e',
        source_substitution='Paderborn KAt diagnostic data -> distinct Paderborn LDM2024 measured lifetime data',
        exclusions={'B01':'no predetermined failure threshold','B05':'deliberately interrupted','B16':'threshold absent in source table'},
        LDM_target='Remaining acquisition count to source-documented vibration/temperature threshold termination; not hours or physical safety probability.',
        IMS_fit='1st acquisition bearing3 channel5',IMS_development='1st acquisition bearing4 channel7 (same acquisition; correlated development)',
        IMS_final='Independent 2nd acquisition all four bearings; only bearing1 has documented terminal damage. Others right-censored.',
        IMS_target='Remaining acquisitions to recording end, explicit proxy; terminal damage time is not precisely measured.',
        IMS_excluded='3rd archive/4th folder endpoint dispute; no arbitrary truncation.',
        query_fractions=[.2,.5,.8],fractions_hidden_from_subject=True,
        primary_metric='Mean bearing MAE in acquisitions over event-bearing queries; RMSE/bias per bearing, censor-bound violations separately.',
        top5_scope='LDM has five final bearings; order of their latest query predictions can be audited. IMS has four; do not invent a fifth.',
        sensor_scope='LDM has measured FFT features, temperatures, loads and speed; no motor-current or torque channels. Original sensor request only partially covered.',
        maximum_new_subjects=5,automatic_review=False,automatic_retry=False,new_final_scores_opened=False,
        training_subsample='At most250 evenly spaced actual training observations per bearing; complete learning trajectories remain available.',
        comparison='Fixed HGB starting predictor, all genuine fixed bank candidates, final submitted programs; no choosing winner using final scores.')
    save(root/'SPLIT_AND_TARGET_FREEZE321.json',declaration)
    source=root/'source/ldm_author/data/LDM';trajectories={};mapping={}
    for k,native in enumerate(eligible,1):
        a=np.load(source/'fft'/(native+'.npy'),allow_pickle=False)
        op=pd.read_csv(source/'op'/(native+'.csv')).iloc[:,1:].to_numpy(float)
        temp=pd.read_csv(source/'temp'/(native+'.csv')).iloc[:,1:].to_numpy(float)
        assert len(a)==len(op)==len(temp)
        x=np.column_stack((a.reshape(len(a),20,10).max(2),op,temp,np.arange(len(a))))
        assert np.isfinite(x).all();trajectories[k]=x;mapping[native]=k
    datasets={'LDM_RTF':dict(trajectories=trajectories,
        fit=[mapping[x] for x in split['fit']],development=[mapping[x] for x in split['development']],
        final=[mapping[x] for x in split['final']],event_final=[mapping[x] for x in split['final']],
        target=declaration['LDM_target'],source='Author-released measured FFT features/operating conditions/temperature; raw152GB not redownloaded or independently recomputed.',
        features=[f'fft_bucket_{j:02}' for j in range(20)]+['dynamic_load_set_N','dynamic_load_peak_N','static_load_set_N','static_load_measured_N','speed_set_rpm','speed_measured_rpm','temp_T1_C','temp_T2_C','ambient_C','elapsed_acquisitions'],
        validation_scope='Two whole held-out bearings; no overlap with seven fitting bearings.')}
    archive=Path('/data/yyf-zyq/PHM_20261010/20261022T110459Z_source280')
    bindings=extract_bound(archive,root/'source/ims',['qualified_ims/1st_test.npz','qualified_ims/2nd_test.npz'])
    save(root/'audit/IMS_SOURCE_BINDINGS321.json',bindings)
    first=np.load(root/'source/ims/qualified_ims/1st_test.npz',allow_pickle=False)
    second=np.load(root/'source/ims/qualified_ims/2nd_test.npz',allow_pickle=False)
    trajectories={1:first['per_channel_features'][:,4,:],2:first['per_channel_features'][:,6,:]}
    trajectories.update({i+3:second['per_channel_features'][:,i,:] for i in range(4)})
    trajectories={u:np.column_stack((a,np.arange(len(a)))) for u,a in trajectories.items()}
    assert all(np.isfinite(a).all() for a in trajectories.values())
    datasets['IMS_ENDPOINT']=dict(trajectories=trajectories,fit=[1],development=[2],final=[3,4,5,6],event_final=[3],
        target=declaration['IMS_target'],source='Full NASA original archive previously CRC/hash verified in A280; no manufactured failure-time or anomaly labels.',
        features=[f'measured_signal_feature_{i:02}' for i in range(15)]+['elapsed_acquisitions'],
        validation_scope='Different damaged bearing but same first acquisition as training; this is correlated development, only final acquisition independent.')
    for name,s in datasets.items():
        base=root/'datasets'/name;data=base/'public/datasets'/name;data.mkdir(parents=True)
        selection={u:range(len(s['trajectories'][u])) for u in s['fit']}
        raw=make_rows(name,s['trajectories'],selection,False);np.savez_compressed(data/'train.npz',**raw)
        q=lambda u:np.unique((np.asarray(declaration['query_fractions'])*(len(s['trajectories'][u])-1)).astype(int))
        dev=make_rows(name,s['trajectories'],{u:q(u) for u in s['development']},True)
        np.savez_compressed(data/'val.npz',**dev)
        info=dict(dataset=name,prediction_unit='acquisitions',target_definition=s['target'],source=s['source'],
            validation_scope=s['validation_scope'],features=s['features'],fit_unit_ids=s['fit'],development_unit_ids=s['development'],
            history_contract='Only own causal prefix at inference. No full final trajectory, final labels, future sensor values or score access.',
            inference_contract='Final prediction runs without /data; save all fitted dependencies in /app or use /reference.',
            task_scope='Explicit research adaptation; not complete original PHMForge task. No physical safety, compliance or optimal maintenance certification.')
        save(data/'dataset.json',info)
        save(data/'training_config.json',dict(fit_unit_ids=s['fit'],epochs=80,cpu_threads=2,seed=321,
            learning_rate=.001,batch_size=128,target_definition=s['target']))
        save(base/'public/dataset_registry.json',{name:dict(task='rul',format='measured_endpoint_npz',path='datasets/'+name,
            architectures=['mlp'],available_splits=['train','val'],test_inputs_visible=False)})
        shutil.copy2(RUNTIME/'public/tool_catalog.json',base/'public/tool_catalog.json')
        final=base/'evaluation_data';final.mkdir();gold=[]
        for u in s['final']:
            for i in q(u):
                obs=make_rows(name,s['trajectories'],{u:[i]},True);sid=str(obs['sample_ids'][0]);label=float(obs.pop('labels')[0])
                np.savez_compressed(final/(sid+'.npz'),**obs)
                gold.append(dict(sample_id=sid,unit_id=u,scenario_id=1,event_observed=u in s['event_final'],
                    actual_rul=label if u in s['event_final'] else None,censor_lower_bound=label if u not in s['event_final'] else None,
                    input_path=str((final/(sid+'.npz')).relative_to(base))))
        save(final/'labels.json',gold)
        audit=train(data,base/'references');bank=fit_bank(data,base/'references/candidates')
        save(base/'SOURCE_AND_TRAINING_QUALIFICATION.json',dict(passed=audit['optimizer_steps']>0 and audit['nonzero_gradient_steps']>0,
            measured_input=True,actual_fit=True,source_substitution=name=='LDM_RTF',proxy_target=name=='IMS_ENDPOINT',
            independent_fit_bearings=len(s['fit']),development_bearings=len(s['development']),final_bearings=len(s['final']),
            final_event_bearings=len(s['event_final']),native_to_unit=mapping if name=='LDM_RTF' else {},
            optimizer_steps=audit['optimizer_steps'],changed_parameters=audit['changed_parameter_tensors'],bank=bank))
    save(root/'SOURCE_STATUS321.json',dict(status='two_explicit_measured_adaptations_qualified',new_subjects=0,final_scores_read=False))


COMMON="""
Complete the original request within the explicit measured-data scope below. Required unavailable quantities must be reported as unknown. No placeholder training, invented labels or fabricated physical risk conclusions. This is a research adaptation, not a claim to reproduce the complete original benchmark.
Inspect /data/datasets/DATASET/dataset.json and training_config.json. Train and val contain actual measured inputs and allowed target labels; final observations and labels are absent. Fit preprocessing on training units; assess complete saved inference programs on target-free development inputs. Whole-bearing validation is required, with its stated dependence limitations. You may retain the useful initial HGB predictor, select supplied genuinely fitted alternatives, or train/code your own. The supplied MLP has real gradient-update logs. Actual MCP loading/training/inference is supported for this dataset. All candidates and honest validation evidence must be retained.
Save solution.py, validation.json, report.md and fitted dependencies. Invocation: python3 solution.py --input INPUT_NPZ --output OUTPUT_JSON. Output predictions must contain one row per input sample_id, its unit_id and scenario_id, and finite nonnegative predicted_rul, unit acquisitions. For anomaly comparison include anomaly_flag and describe the training-fitted IsolationForest rule. If requested, add sensor-group predictions under sensor_predictions using the same identity-bearing record format; each group's model must be actually fitted. Do not invent missing modalities.
The evaluator invokes each causal prefix in a fresh process. Own history_features/history_offsets are available; no other final queries, future data or labels are mounted. /data is absent in inference; preserve fitted objects in /app or use read-only /reference. Only /app is writable. /tools contains genuine tools; no network, outside tasks, secret files or other model APIs. Temporary data only /app/.runtime, never /tmp. Budget3600s,180 requests,18000000input tokens,32768output/request,262144totaloutput,2numericalthreads. No compulsory model change.
"""


def prepare_agents(root):
    assert read(root/'SOURCE_STATUS321.json')['status']=='two_explicit_measured_adaptations_qualified'
    prior=Path('/data/yyf-zyq/PHM_20261010/20261023T154500Z_safety314/c')
    online=read(prior/'audit/CLI_ONLINE_CHECK.json')
    with urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/openai/codex/releases/latest',headers={'User-Agent':'PHM-source-recovery'}),timeout=40) as f:raw=f.read()
    release=json.loads(raw);assert release['tag_name']=='rust-v'+online['latest_stable_version']
    assert sha256(Path(online['CLI_path']))==online['CLI_sha256']
    online.update(checked_utc=datetime.now(timezone.utc).isoformat(),latest_response_sha256=hashlib.sha256(raw).hexdigest())
    save(root/'audit/CLI_ONLINE_CHECK.json',online);(root/'audit/CLI_LATEST_RELEASE.json').write_bytes(raw)
    cases=[dict(cohort='l1',dataset='LDM_RTF',task_id='pdm_rul_014'),dict(cohort='l2',dataset='LDM_RTF',task_id='pdm_rul_008'),
        dict(cohort='i1',dataset='IMS_ENDPOINT',task_id='pdm_rul_005'),dict(cohort='i2',dataset='IMS_ENDPOINT',task_id='pdm_rul_011'),
        dict(cohort='i3',dataset='IMS_ENDPOINT',task_id='pdm_safety_010')]
    budget=dict(subject_timeout_seconds=3600,max_model_requests_per_subject=180,max_total_input_tokens_per_subject=18000000,
        max_output_tokens_per_request=32768,max_total_output_tokens_per_subject=262144,mcp_tool_timeout_seconds=180,
        max_concurrency=8,local_max_concurrency=1,artifact_budget_bytes=4*1024**3)
    save(root/'CAMPAIGN_CONFIG.json',dict(iteration='A321',result_directory='results/gap321',cases=cases,planned_sessions=5,session_budget=budget,
        purpose='Recover16 data/contract gaps;5 genuine measured-data coding tasks under explicit source/endpoint adaptations.',
        authorization='用户：16个缺少数据或评价依据，缺少什么呢，你自己找一下数据然后实验。前序允许研究适配。',
        no_automatic_final_scoring=True,no_automatic_retry=True,source_or_target_adaptation=True,original_complete_tasks_claimed=0))
    originals={v['task_id']:v for v in read(root/'source/ORIGINAL_16.json')}
    tools=root/'public_tools';tools.mkdir()
    for name in ['verified_phmforge.py','verified_benchmark_worker.py','verified_benchmark_scoring.py','real_fault_model.py',
                 'real_rul_model.py','real_bearing_rul_model.py','real_xjtu_rul_model.py','real_gap_rul_model.py','isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name,tools/name)
    for c in cases:
        run=root/c['cohort'];name=c['dataset'];ident='gap321_'+c['cohort'];base=root/'datasets'/name
        for folder in ['audit','logs','code','runtime/tmp','runtime/mpl','plans','workspaces']:(run/folder).mkdir(parents=True)
        ws=run/'workspaces'/ident;ws.mkdir()
        for folder in ['.runtime/tmp','.runtime/mpl','.runtime/xdg','artifacts']:(ws/folder).mkdir(parents=True)
        (ws/'solution.py').write_text("import argparse,sys\nfrom pathlib import Path\nsys.path.insert(0,'/tools')\nfrom real_gap_rul_model import infer\np=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)\na=p.parse_args();infer('/reference/candidates/hgb100.joblib',a.input,a.output)\n")
        (ws/'AGENTS.md').write_text('Respect measured-source and target limits, whole-unit splits and causal prefixes. No final inputs/labels during coding. Save genuine fitted artifacts. Temporary files /app/.runtime only.\n')
        question=originals[c['task_id']]['input_question'];(ws/'TASK.txt').write_text(question)
        detail=('Paderborn substitution: LDM2024 is a distinct measured run-to-failure dataset, not KAt2016. Fourteen records have documented threshold endpoints; no current or torque sensor exists. Seven whole bearings fit, two develop, five final. Target remaining acquisitions, never hours. Source FFT buckets and measured temperatures/loads/speed are supplied. Evaluate available sensor groups honestly, with same learning data; current/torque comparisons unknown.' if name=='LDM_RTF' else
            'IMS target is remaining acquisitions to published recording end, a proxy rather than exact physical failure time. Training bearing3 and development bearing4 share acquisition1. Independent acquisition2 has four monitored bearings, only bearing1 has documented terminal damage; others are right-censored. Third acquisition excluded for disputed endpoint. Model-detected anomaly flags are not anomaly ground truth; fit IsolationForest on training only and report errors for its detected strata. Five distinct final units are unavailable; rank available units and disclose limits. No calibrated physical risk or compliance standard is provided.')
        prompt=run/'plans'/(ident+'.txt');prompt.write_text(question+'\n'+COMMON.replace('DATASET',name)+'\n'+detail)
        job=dict(id=ident,family='forge_measured_rul',task_id=c['task_id'],condition='coding_baseline',repeat=1,workspace=str(ws),prompt_file=str(prompt),answer_mode='text')
        plan=run/'plans'/(ident+'.json');save(plan,dict(jobs=[job]))
        mounts=run/'plans'/(ident+'_mounts.json');save(mounts,dict(mounts=[dict(source=str(a),destination=b) for a,b in
            [(PYTHON,'/opt/python312'),(RUNTIME/'deps','/opt/phm_python/extra'),(tools,'/tools'),(base/'public','/data'),(base/'references','/reference')]]))
        socket=run/'ipc/s0.sock';save(run/'PLAN187.json',dict(jobs=[dict(job,workspace=str(ws.relative_to(run)),plan=str(plan),mounts=str(mounts),model_gateway_socket=str(socket),dataset=name,data_format='measured_endpoint_npz',test_inputs_visible_during_development=False)]))
        save(run/'PROTOCOL187.json',dict(budget,model='deepseek-flash',cli=online['CLI_path'],cli_sha256=online['CLI_sha256'],
            max_new_model_sessions=1,max_new_primary_rolling24h=400,training_gate='TASK_QUALIFICATION251.json',environment_gate='AGENT_ISOLATION_GATE251.json',
            authorization_record='audit/AUTHORIZATION251.json',controller_status_name='CONTROLLER_STATUS251.json',external_model_gateway=True,
            test_inputs_visible=False,original_benchmark_score_claimed=False,
            additional_gates=['PUBLIC_METADATA_GATE267.json','NATIVE_CLI_MCP_QUALIFICATION.json','OUTPUT_BUDGET_QUALIFICATION.json']))
        save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=ident,socket=str(socket))],base_url='https://api.deepseek.com/',model='deepseek-flash',max_requests=180,max_input_tokens=18000000,max_output_tokens=262144,max_output_per_request=32768))
        for file in ['cli_config187.toml','model_catalog187.json']:shutil.copy2(prior/file,run/file)
        shutil.copy2(root/'audit/CLI_ONLINE_CHECK.json',run/'audit/CLI_ONLINE_CHECK.json')
        source_qualification=read(base/'SOURCE_AND_TRAINING_QUALIFICATION.json')
        save(run/'TASK_QUALIFICATION251.json',dict(passed=source_qualification['passed'],research_adaptation=True,actual_measured_inputs=True,
            genuine_fit=True,proxy_target=name=='IMS_ENDPOINT',source_substitution=name=='LDM_RTF',full_original_task_qualified=False))
        save(run/'audit/AUTHORIZATION251.json',dict(user_reply=read(root/'CAMPAIGN_CONFIG.json')['authorization'],scope=detail))
        save(run/'OWNERSHIP.json',dict(owner='root',writers_stopped=True,status='prepared_waiting_isolation'))
        (run/'public_tools').symlink_to(tools,target_is_directory=True)
        (run/'references').symlink_to(base/'references',target_is_directory=True)
        metadata_gate(run);freeze(run)
        frozen=read(run/'INITIAL_FREEZE187_FIFO_FILES.json');known={v['path'] for v in frozen['files']}
        for tree in [tools,base/'references']:
            for f in tree.rglob('*'):
                if f.is_file() and str(f) not in known:frozen['files'].append(dict(path=str(f),sha256=sha256(f)))
        save(run/'INITIAL_FREEZE187_FIFO_FILES.json',frozen)
    for file in ['prepare_forge_gap_recovery.py','real_gap_rul_model.py','verified_phmforge.py','qualify_coding_agent_isolation.py']:
        shutil.copy2(PROJECT/'scripts'/file,root/'code'/file)
    state(root,status='five_scoped_subjects_prepared',new_model_sessions=0,final_scores_opened=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=['source','prepare','run'],required=True)
    a=p.parse_args();root=a.run.resolve()
    if a.phase=='source':prepare_sources(root)
    elif a.phase=='prepare':prepare_agents(root)
    else:execute_wave(root,'gap_recovery',[root/c['cohort'] for c in read(root/'CAMPAIGN_CONFIG.json')['cases']])
