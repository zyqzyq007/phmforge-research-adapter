"""Prepare and run four native coding trajectories with frozen real RUL data.

Stops after all choices are sealed; never computes final scores automatically.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat

from acquire_forge_pdmb_sources244 import save,sha256
from prepare_forge_coding_baseline import CLI,CLI_SHA,PYTHON,RUNTIME
from run_validation_transfer import execute_wave,freeze,metadata_gate,state,utc
from run_full_benchmark_campaign import verify
from evaluate_real_training_reviews import alive


PROJECT=Path(__file__).resolve().parents[1]


COMMON='''
Complete this PHMForge RUL task using the supplied actual measured bearing data.
This is a research adaptation: the original XJTU question/configuration conflict is corrected to XJTU measurements, original terminal endpoints, and nominal minutes. Do not apply engine-cycle accuracy bands. No synthetic records, simulation, label modification, or LLM training is permitted.

Data: /data/datasets/XJTU/train.npz has nine complete fitting bearings, labels and opaque unit IDs. val.npz has one labeled causal observed prefix from each of three separate development bearings. The descriptor documents 30 raw-scale signal statistics, current age, known operating conditions and causal feature histories. It is a feature-interface study, not access to the full waveform. All target values are nominal remaining minutes to the original terminal acquisition. Terminal damage categories are available only to the external evaluator; do not infer current fault state from eventual damage labels.

Final observations, future histories, final labels and final scores are absent. Each final invocation receives only ONE query and that bearing's history up to the query time; there is no cross-query state or access to other future prefixes. Final source names, lifetimes, sampling fractions and other queries are not provided. Do not infer a lifetime from an assumed fraction of life or from the fact a prefix ends here. No maintenance-cost or field-safety claim is supported by this laboratory task.

The six real train-only candidates in /reference/candidates were fitted before development scores were calculated. manifest.json includes whole-bearing training CV and development results; training_cv_folds.json specifies the folds. /reference/model.pt is a separate actual gradient-trained MLP with its audit. The MCP tools provide real dataset loading, MLP fitting, inference and arithmetic verification; their source is /tools. You may inspect and execute Python, compare actual evidence and retain a good existing model. Baseline selection is not a requirement to change anything. No final scores or prior research conclusions are supplied.

Fit supervised models and all learned preprocessing only on train.npz. Development information may guide assessment and selection; do not refit on development labels or append those examples to training. Preserve whole-bearing groups in CV and use only causal histories. Report actual mean-per-bearing MAE and RMSE in MINUTES, distinguish fitting, CV and development evidence, and explain limitations. Keep useful tested alternatives with code/checkpoints and record why you retained/rejected each. Lower MAE is better, without a positive minimum error bound. The final endpoint and fault-type evaluation will be external after every submission is sealed.

Save selection.json, validation.json and report.md. selection.json must contain {"candidate": "NAME"}, where NAME is a bank name or "custom" when allowed. Submit a saved fitted, runnable solution.py with interface:
  python3 solution.py --input OBSERVATION_NPZ --output PREDICTION_JSON
The output is {"unit":"minutes","predictions":[{"sample_id":string,"unit_id":integer,"scenario_id":integer,"predicted_rul":finite_nonnegative_number},...]}, with exactly every provided identity. For custom deployment include every necessary code/checkpoint outside .runtime; data and fitting labels will not be mounted during final inference. If choosing a bank name, external evaluation invokes that exact immutable bank artifact, and your wrapper cannot alter its predictions. Do not fabricate final predictions or test metrics. Test the real interface on label-free development observations.

Python3.12,NumPy,SciPy,scikit-learn,PyTorch and MCP are available. There is no Internet, package installation, other-task data, additional model API or access to evaluator files. /data,/reference,/tools are read-only. Put new work under /app; all temporary files and caches under /app/.runtime, never /tmp. You have3600seconds,180 model requests,32768 output tokens/request,524288 total output tokens,32000000 completed input tokens and two numerical threads. Preserve a useful runnable submission before the budget ends.
'''


def prepare(root):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert json.loads((root/'audit/NUMERICAL_QUALIFICATION278.json').read_text())['passed']
    assert sha256(CLI)==CLI_SHA
    template=Path('/data/yyf-zyq/PHM_20261010/20261022T052053Z_pipeline276/a')
    source=Path(config['source_archive'])
    original=json.loads((source/'audit/ORIGINAL_XJTU_TASKS277.json').read_text())
    # The source audit may store records under a wrapper; retain its full form.
    shutil.copy2(source/'audit/ORIGINAL_XJTU_TASKS277.json',root/'source/ORIGINAL_XJTU_TASKS277.json')
    question='The XJTU dataset has bearings with five different fault types, and I want to see if our RUL predictions are more accurate for certain fault types. Can you predict remaining life for all test bearings, give me overall accuracy numbers, then break it down by fault type so I can see which fault types we predict best? This will help us prioritize which failure modes to focus our maintenance efforts on.'
    assert question in json.dumps(original)
    tools=root/'public_tools';tools.mkdir()
    for name in ['verified_phmforge.py','verified_benchmark_worker.py','verified_benchmark_scoring.py',
        'real_fault_model.py','real_rul_model.py','real_bearing_rul_model.py','real_xjtu_rul_model.py','isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name,tools/name)
    shutil.copy2(RUNTIME/'public/tool_catalog.json',root/'public_data/tool_catalog.json')
    shutil.copy2(root/'references/mlp/model.pt',root/'references/model.pt')
    shutil.copy2(root/'references/mlp/training_audit.json',root/'references/training_audit.json')
    starter='''import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/tools')
from real_xjtu_rul_model import infer
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();name=json.loads(Path('selection.json').read_text())['candidate']
if name=='custom':raise ValueError('Provide a saved custom prediction program')
infer(Path('/reference/candidates')/(name+'.joblib'),a.input,a.output)
'''
    for case in config['cases']:
        run=root/case['cohort']
        assert not run.exists()
        for folder in ['audit','code','logs','runtime/tmp','runtime/mpl','plans','workspaces']:
            (run/folder).mkdir(parents=True,exist_ok=True)
        ident='s278_'+case['case'];ws=run/'workspaces'/ident;ws.mkdir()
        for folder in ['.runtime/tmp','.runtime/mpl','.runtime/xdg','artifacts']:(ws/folder).mkdir(parents=True)
        (ws/'solution.py').write_text(starter);save(ws/'selection.json',dict(candidate='rf200'))
        (ws/'AGENTS.md').write_text('Use actual measured data and real fitting. Keep temporary files in /app/.runtime. Respect train/development/final and causal-time boundaries. No final test access.\n')
        arm=('For this condition, choose exactly one of the six immutable bank candidates. You may investigate and reproduce evidence, but new fitted predictors, mixtures, calibration, rules, or altered bank weights are not admissible final submissions. selection.json must name one bank candidate.' if case['arm']=='frozen' else
             'For this condition, you may choose an unchanged bank candidate OR modify/train a new predictor based on the permitted training/development evidence. A new model, mixture, calibration or rule is a custom program; set candidate="custom" and preserve the exact fitted deployment program. Keeping a bank candidate is acceptable.')
        prompt=run/'plans'/(ident+'.txt');prompt.write_text(question+'\n'+COMMON+'\n'+arm+'\n')
        job=dict(id=ident,family='forge_measured_rul',task_id='pdm_rul_004',condition=case['arm'],
            repeat=case['repeat'],workspace=str(ws),prompt_file=str(prompt),answer_mode='text')
        plan=run/'plans'/(ident+'.json');save(plan,dict(jobs=[job]))
        mounts=run/'plans'/(ident+'_mounts.json');save(mounts,dict(mounts=[
            dict(source=str(PYTHON),destination='/opt/python312'),
            dict(source=str(RUNTIME/'deps'),destination='/opt/phm_python/extra'),
            dict(source=str(tools),destination='/tools'),
            dict(source=str(root/'public_data'),destination='/data'),
            dict(source=str(root/'references'),destination='/reference')]))
        socket=run/'ipc/s0.sock'
        save(run/'PLAN187.json',dict(jobs=[dict(job,workspace=str(ws.relative_to(run)),
            plan=str(plan),mounts=str(mounts),model_gateway_socket=str(socket),dataset='XJTU',
            data_format='xjtu_npz',test_inputs_visible_during_development=False)]))
        protocol=dict(config['session_budget'],model='deepseek-flash',cli=str(CLI),cli_sha256=CLI_SHA,
            max_new_model_sessions=1,max_new_primary_rolling24h=400,local_max_concurrency=1,
            training_gate='TASK_QUALIFICATION251.json',environment_gate='AGENT_ISOLATION_GATE251.json',
            authorization_record='audit/AUTHORIZATION251.json',controller_status_name='CONTROLLER_STATUS251.json',
            external_model_gateway=True,test_inputs_visible=False,original_benchmark_score_claimed=False,
            additional_gates=['PUBLIC_METADATA_GATE267.json','NATIVE_CLI_MCP_QUALIFICATION.json','OUTPUT_BUDGET_QUALIFICATION.json'])
        save(run/'PROTOCOL187.json',protocol)
        save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=ident,socket=str(socket))],
            base_url='https://api.deepseek.com/',model='deepseek-flash',
            max_requests=protocol['max_model_requests_per_subject'],
            max_input_tokens=protocol['max_total_input_tokens_per_subject'],
            max_output_tokens=protocol['max_total_output_tokens_per_subject'],
            max_output_per_request=protocol['max_output_tokens_per_request']))
        for name in ['cli_config187.toml','model_catalog187.json']:shutil.copy2(template/name,run/name)
        shutil.copy2(root/'audit/CLI_ONLINE_CHECK.json',run/'audit/CLI_ONLINE_CHECK.json')
        save(run/'TASK_QUALIFICATION251.json',dict(passed=True,
            numerical_gate=str(root/'audit/NUMERICAL_QUALIFICATION278.json'),
            source_archive=config['source_archive'],real_measured_source=True,research_adaptation=True,
            scope='Original XJTU RUL objective, feature interface, measured minute targets and external final scoring'))
        save(run/'audit/AUTHORIZATION251.json',dict(user_reply=config['authorization'],scope='Four frozen A278 native coding trajectories under the active goal'))
        save(run/'OWNERSHIP.json',dict(owner='root',writers_stopped=True,status='prepared_waiting_isolation'))
        metadata_gate(run);freeze(run)
        manifest=json.loads((run/'INITIAL_FREEZE187_FIFO_FILES.json').read_text())
        known={x['path'] for x in manifest['files']}
        for tree in [root/'references',tools]:
            for p in tree.rglob('*'):
                if p.is_file() and str(p) not in known:manifest['files'].append(dict(path=str(p),sha256=sha256(p)))
        save(run/'INITIAL_FREEZE187_FIFO_FILES.json',manifest)
    for name in ['run_xjtu_feedback.py','qualify_xjtu_feedback.py','verified_phmforge.py','qualify_coding_agent_isolation.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'code'/name)
    save(root/'audit/AGENT_PREPARATION278.json',dict(utc=utc(),passed=True,planned_sessions=4,
        only_intended_arm_difference=True,common_public_data=str(root/'public_data'),
        common_readonly_reference=str(root/'references'),no_final_observations_mounted=True))


def main(root, prepare_only=False, resume_qualified=False):
    if resume_qualified:
        previous=json.loads((root/'CAMPAIGN_STATUS.json').read_text())
        assert previous['status']=='controller_error_inspect_before_resume'
        assert not alive(previous['supervisor_pid'])
        attempt=1+len(list((root/'audit').glob('pre_admission_recovery*')))
        recovery=root/'audit'/('pre_admission_recovery'+str(attempt));recovery.mkdir()
        shutil.copy2(root/'CAMPAIGN_STATUS.json',recovery/'ORIGINAL_CAMPAIGN_STATUS.json')
        views=[]
        for name in ['a','b','c','d']:
            run=root/name
            assert not list((run/'primary').glob('*'))
            assert not list((run/'model_gateway').glob('*/*_receipt.json'))
            if (run/'MODEL_GATEWAY_READY.json').exists():
                ready=json.loads((run/'MODEL_GATEWAY_READY.json').read_text())
                assert not alive(ready['pid'])
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():continue
                try:argv=(proc/'cmdline').read_bytes().split(b'\0')
                except OSError:continue
                assert not (str(run).encode() in argv and any(x.endswith(b'isolated_model_gateway.py') for x in argv)), 'A gateway is still running'
            for gate in ['NATIVE_CLI_MCP_QUALIFICATION.json','OUTPUT_BUDGET_QUALIFICATION.json']:
                assert json.loads((run/'audit'/gate).read_text())['passed']
                shutil.copy2(run/'audit'/gate,run/gate)
            job=json.loads((run/'PLAN187.json').read_text())['jobs'][0]
            verify(run,job)
            mounts=json.loads(Path(job['mounts']).read_text())['mounts']
            views.append(dict(mounts=mounts,solution=sha256(run/job['workspace']/'solution.py'),
                initial_selection=sha256(run/job['workspace']/'selection.json'),
                common_prompt=Path(job['prompt_file']).read_text().split('For this condition,')[0]))
            old=recovery/name;old.mkdir()
            for filename in ['MODEL_GATEWAY_READY.json','MODEL_GATEWAY_STOP','model_gateway',
                             'CONTROLLER_STATUS251.json','OWNERSHIP.json']:
                if (run/filename).exists():shutil.move(str(run/filename),str(old/filename))
            save(run/'OWNERSHIP.json',dict(owner='root',writers_stopped=True,
                status='pre_admission_recovered_without_native_launch'))
            inactive=[]
            sockets=Path('/proc/net/unix').read_text()
            for path in (run/'ipc').glob('*.sock'):
                assert str(path) not in sockets and stat.S_ISSOCK(path.stat().st_mode)
                inactive.append(str(path));path.unlink()
            save(old/'INACTIVE_SOCKET_CLEANUP.json',dict(removed=inactive,live_socket_removed=False))
        assert all(x==views[0] for x in views)
        save(recovery/'RECOVERY.json',dict(utc=utc(),passed=True,
            reason=previous.get('error'),
            all_initial_scientific_files_unchanged=True,common_views_equal=True,
            prior_LLM_calls=0,repeated_benchmark_launches=0,all_four_offline_budget_checks_passed=True))
        shutil.copy2(PROJECT/'scripts/qualify_model_output_budget.py',root/'code/qualify_model_output_budget.py')
        shutil.copy2(__file__,root/'code/run_xjtu_feedback_recovery.py')
    else:
        assert not (root/'CAMPAIGN_STATUS.json').exists(), 'No automatic campaign rerun'
        prepare(root)
    if prepare_only:return
    state(root,status='prepared_real_RUL_feedback_contrast',supervisor_pid=os.getpid(),
          planned_model_sessions=4,physical_allocations=1,final_scores_read=False,
          error=None,error_type=None)
    try:
        execute_wave(root,'comparison',[root/name for name in ['a','b','c','d']],
                     reuse_pre_admission_qualification=resume_qualified)
        state(root,status='all_sessions_terminal_waiting_choice_review_and_external_scores',
            completed_model_sessions=4,final_scores_read=False,
            next_action='Register actual choices and alternatives, then isolated one-query cold predictions and independent metrics.')
    except BaseException as error:
        state(root,status='controller_error_inspect_before_resume',error_type=type(error).__name__,error=str(error),automatic_retries=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume-qualified',action='store_true')
    a=p.parse_args();assert not(a.prepare_only and a.resume_qualified)
    main(a.run.resolve(),a.prepare_only,a.resume_qualified)
