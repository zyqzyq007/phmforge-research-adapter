"""Configure bounded native coding repetitions on the qualified measured source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from acquire_forge_pdmb_sources244 import save,sha256
from prepare_forge_coding_baseline import CLI,CLI_SHA,PYTHON,RUNTIME
from qualify_native_cli_mcp import mcp_config


def prepare(run,source):
    assert not (run/'PLAN187.json').exists()
    result=json.loads((source/'qualification_output/MCP_QUALIFICATION.json').read_text())
    assert result['passed']
    info=json.loads((source/'public/datasets/FEMTO/dataset.json').read_text())
    assert len(info['units'])==17 and info['feature_count']==103
    assert sha256(CLI)==CLI_SHA
    bootstrap=json.loads((run/'BOOTSTRAP_PROTOCOL.json').read_text()) if (run/'BOOTSTRAP_PROTOCOL.json').exists() else {}
    if bootstrap:
        assert json.loads((run/'audit/CLI_ONLINE_CHECK.json').read_text())['latest_stable_version']=='0.163.0'
    for folder in ['audit','source','logs','plans','workspaces','references','public_tools','runtime/tmp','runtime/mpl','code']:
        (run/folder).mkdir(parents=True,exist_ok=True)
    prior=Path('/data/yyf-zyq/PHM_20261010/20261020T183545Z_forgecoding252/RUN_CONTENTS.tar.gz')
    wanted={'cli_config187.toml','model_catalog187.json'}
    with tarfile.open(prior,'r:gz') as archive:
        for member in archive:
            if member.name in wanted:
                (run/member.name).write_bytes(archive.extractfile(member).read());wanted.remove(member.name)
            if not wanted:break
    assert not wanted
    (run/'cli_config187.toml').write_text(mcp_config((run/'cli_config187.toml').read_text()))
    original_archive=Path('/data/yyf-zyq/PHM_20261010/20261020T133326Z_forgesources244/RUN_CONTENTS.tar.gz')
    with tarfile.open(original_archive,'r:gz') as archive:
        for member in archive:
            if member.name=='source/official/ReActXen/src/reactxen/demo/intent_implementation_demo/scenarios/phm_scenarios.json':
                scenarios=json.load(archive.extractfile(member))['pdm_scenarios'];break
    task=next(x for x in scenarios if x['task_id']=='pdm_rul_003')
    (run/'source/ORIGINAL_QUESTION.txt').write_text(task['input_question']+'\n')
    names=['verified_phmforge.py','verified_benchmark_worker.py','verified_benchmark_scoring.py',
        'real_fault_model.py','real_rul_model.py','real_bearing_rul_model.py','isolated_model_gateway.py']
    for name in names:shutil.copy2(source/'public_tools'/name,run/'public_tools'/name)
    # The source run predates the gateway output-budget fix; use the current
    # verified transport, keeping the source-specific numerical backend.
    shutil.copy2(Path(__file__).with_name('isolated_model_gateway.py'),run/'public_tools/isolated_model_gateway.py')
    checkpoint=source/'qualification_output'/Path(result['model_path']).relative_to('/output')
    jobs=[];routes=[];frozen=[]
    common='''
This is a measured-data research adaptation of the original PHMForge task. Use the supplied original PRONOSTIA learning runs and observed test prefixes; the transformed PDMB CSV is not used. The target is annotated remaining life in seconds. The three scenarios are 1800rpm/4000N,1650rpm/4200N,1500rpm/5000N. There are two complete learning bearings per scenario and 5/5/1 observed test bearings. This small laboratory sample does not establish field-maintenance benefits.

The MCP server phmforge provides actual loading, CPU model training, inference and development arithmetic verification. The genuine reference contains three separately fitted scenario models. You may retain it, retrain it or write another genuinely fitted predictor. Its selection audit includes held-out learning-bearing prefix metrics, followed by a final refit using all six learning bearings. Scores computed with that final model on those fitting bearings are not independent validation scores. Preserve device groups for model selection and fit normalization only on allowed fitting data.

The data directory /data/datasets/FEMTO contains train.npz (complete six learning trajectories, targets available), val.npz (nine labeled development queries at30/60/90percent of three held-out learning bearings), test.npz (eleven unlabeled endpoint queries), and trajectories/unitNNN.npz (learning histories and the observed test prefixes). Each query contains causal103-dimensional features, current measured two-axis waveforms, original sample_id, unit_id, scenario_id and elapsed time. The descriptor explains the declared10-second acquisition schedule and original logger anomalies. Sensor amplitudes are preserved; no per-window normalization was applied. For a historical query use only observations at or before its query time. Complete learning trajectories may be used to refit after model selection; future test continuations, final labels, prior test predictions and scores are absent.

Independent final evaluation is added here: final test labels and scores are external, unlike the original benchmark's Agent-side final truth verification. Report actual development metrics, label their selection/refit scope, and mark final test metrics externally pending. Do not invent scores. Some source-annotation uncertainty is tracked externally; it must not be silently fixed by guessing. A single large acceleration peak does not by itself define zero remaining life for the supplied annotated benchmark target.

Submit solution.py with interface:
  python3 solution.py --input OBSERVATION_NPZ --output PREDICTION_JSON
The program must load saved fitted artifacts and predict each query from observations without reading target labels. Output {"predictions":[{"sample_id":string,"unit_id":integer,"scenario_id":integer,"predicted_rul":finite_nonnegative_seconds}, ...]}, covering every input identity exactly once and preserving units and scenarios. Save all required code/checkpoints, test_predictions.json produced by this exact program on test.npz, validation.json and report.md. The report should explain real data/fitting, per-scenario development MAE/RMSE, model choice, uncertainty and what the evidence permits concerning maintenance. Final external evaluation computes all11 bearing errors, per-scenario MAE/RMSE, bias, overestimation and underestimation, the challenge asymmetric score, and source-uncertainty sensitivity. There is no arbitrary positive lower error bound.

You have1200seconds,CPU-only with two numerical threads,at most80 model requests,32768 output tokens per request,262144 cumulative output tokens and6000000 completed input tokens (then the next request is denied). Save a runnable submission before the deadline. Python3.12,NumPy,SciPy,scikit-learn,PyTorch and MCP are installed. /data and /reference are read-only. All output belongs under /app; temporary files/caches only under /app/.runtime,never /tmp. No Internet,package installation,other tasks,hidden labels or additional model APIs.
'''
    starter='''import argparse,sys
from pathlib import Path
sys.path.insert(0,'/tools')
from real_bearing_rul_model import infer
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();infer(Path('/reference/model.pt'),a.input,a.output)
'''
    repeats=bootstrap.get('repeats',[1,2]);prefix=bootstrap.get('job_prefix','forge254')
    common=common.replace('You have1200seconds','You have'+str(bootstrap.get('subject_timeout_seconds',1200))+'seconds').replace(
        'at most80 model requests','at most'+str(bootstrap.get('max_model_requests_per_subject',80))+' model requests').replace(
        'and6000000 completed input tokens','and'+str(bootstrap.get('max_total_input_tokens_per_subject',6000000))+' completed input tokens')
    for repeat in repeats:
        ident=f'{prefix}_pdm_rul_003_r{repeat}';ws=run/'workspaces'/ident;ws.mkdir()
        for folder in ['.runtime/tmp','.runtime/mpl','artifacts']:(ws/folder).mkdir(parents=True)
        (ws/'solution.py').write_text(starter);(ws/'TASK.txt').write_text(task['input_question']+'\n')
        (ws/'AGENTS.md').write_text('Use only /app/.runtime for temporary files. Preserve source data and model/scenario identities. No final labels,other tasks,Internet or additional model APIs. Save a fitted runnable submission within the declared budget.\n')
        reference=run/'references'/ident;reference.mkdir()
        shutil.copy2(checkpoint,reference/'model.pt');shutil.copy2(checkpoint.with_name('training_audit.json'),reference/'training_audit.json')
        prompt=run/'plans'/(ident+'.txt');prompt.write_text(task['input_question']+'\n'+common)
        mounts=run/'plans'/(ident+'_mounts.json');save(mounts,{'mounts':[
            {'source':str(PYTHON),'destination':'/opt/python312'},
            {'source':str(RUNTIME/'deps'),'destination':'/opt/phm_python/extra'},
            {'source':str(run/'public_tools'),'destination':'/tools'},
            {'source':str(source/'public'),'destination':'/data'},
            {'source':str(reference),'destination':'/reference'}]})
        job={'id':ident,'family':'forge_measured_rul','task_id':'pdm_rul_003','condition':'coding_baseline',
            'repeat':repeat,'workspace':str(ws),'prompt_file':str(prompt),'answer_mode':'text'}
        plan=run/'plans'/(ident+'.json');save(plan,{'jobs':[job]})
        socket=run/'ipc'/(f'rul003_r{repeat}.sock');routes.append({'job':ident,'socket':str(socket)})
        jobs.append(dict(job,workspace=str(ws.relative_to(run)),plan=str(plan),mounts=str(mounts),
            model_gateway_socket=str(socket),dataset='FEMTO',data_format='femto_npz'))
        for p in list(ws.rglob('*'))+list(reference.rglob('*'))+[prompt,plan,mounts]:
            if p.is_file():frozen.append({'path':str(p),'sha256':sha256(p)})
    protocol={'run_id':run.name,'model':'deepseek-flash','cli':str(CLI),'cli_sha256':CLI_SHA,
        'max_new_model_sessions':2,'max_new_primary_rolling24h':400,'local_max_concurrency':2,'max_concurrency':8,
        'subject_timeout_seconds':1200,'max_model_requests_per_subject':80,
        'max_total_input_tokens_per_subject':6000000,'max_output_tokens_per_request':32768,'max_total_output_tokens_per_subject':262144,
        'artifact_budget_bytes':8*1024**3,'training_gate':'TASK_QUALIFICATION251.json','environment_gate':'AGENT_ISOLATION_GATE251.json',
        'authorization_record':'audit/AUTHORIZATION251.json','controller_status_name':'CONTROLLER_STATUS251.json',
        'external_model_gateway':True,'original_benchmark_score_claimed':False,'source_run':str(source),
        'label_policy':'Keep all11 official labels; exclude unresolved Bearing1_4 target ambiguity from Agent-failure attribution; report all11 plus consistent10 subset.',
        'no_scores_until_both_submissions_frozen':True,'no_preassumed_failure':True}
    protocol.update(bootstrap)
    protocol.update(max_new_model_sessions=len(jobs),local_max_concurrency=min(bootstrap.get('local_max_concurrency',2),len(jobs)))
    save(run/'PROTOCOL187.json',protocol);save(run/'PLAN187.json',{'jobs':jobs})
    save(run/'MODEL_GATEWAY_CONFIG.json',{'routes':routes,'base_url':'https://api.deepseek.com/',
        'model':'deepseek-flash','max_requests':protocol['max_model_requests_per_subject'],
        'max_input_tokens':protocol['max_total_input_tokens_per_subject'],
        'max_output_tokens':protocol['max_total_output_tokens_per_subject'],
        'max_output_per_request':protocol['max_output_tokens_per_request']})
    save(run/'TASK_QUALIFICATION251.json',{'passed':True,'real_MCP_source_gate':str(source/'qualification_output/MCP_QUALIFICATION.json'),
        'original_task_id':'pdm_rul_003','original_question_preserved':True,'research_adaptation':True,
        'measured_training_bearings':6,'observed_test_bearings':11,'scenarios':3,'label_ambiguities':1,
        'all11_annotations_physically_certified':False,'new_Agent_sessions':0})
    save(run/'audit/AUTHORIZATION251.json',{'user_reply':'设立简洁清晰目标：核查权限、尽可能真实、coding Agent baseline、分析问题；继续执行该活动目标',
        'scope':'bounded coding repetitions with corrected native MCP on measured FEMTO; same source and prior scores known only to researcher; within current active goal,not the old333-task grant'})
    save(run/'OWNERSHIP.json',{'owner':'root','writers_stopped':True,'status':'prepared_waiting_isolation'})
    for p in list((source/'public').rglob('*'))+list((run/'public_tools').glob('*.py'))+[run/x for x in ['PROTOCOL187.json','MODEL_GATEWAY_CONFIG.json','cli_config187.toml','model_catalog187.json','TASK_QUALIFICATION251.json']]:
        if p.is_file():frozen.append({'path':str(p),'sha256':sha256(p)})
    save(run/'INITIAL_FREEZE187_FIFO_FILES.json',{'files':frozen})
    print(json.dumps({'tasks':1,'sessions':len(jobs),'source':str(source),'prepared':str(run)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();prepare(a.run.resolve(),a.source.resolve())
