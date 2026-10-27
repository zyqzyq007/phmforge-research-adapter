"""Four same-budget native coding chains on real CMAPSS training and prefix checks."""
import argparse,ast,json,os,pickle,shutil,socket,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from controlled_feedback import save,sha,utc
from native_validation_comparison import closure
from free_program_comparison import arguments,environment,execute
from replay_diagnostic_cli import isolate_filesystem
from qualify_native_cli_mcp import probe
from cmapss_program_runtime import Runtime
PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
PYTHON=Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS=Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')
read=lambda p:json.loads(Path(p).read_text())

COMMON="""Improve or retain complete PHM remaining-useful-life programs. Predict RUL in cycles at a current observation time, using only settings and sensors observed at or before that time. Minimize mean absolute error on unseen engines. NASA official CMAPSS FD001 is physical simulation data, not field measurements. Actual training, inference and external numerical evaluation are required. Read /data/metadata.json and the three original program sources and trained artifacts under /reference. These are actual earlier Agent programs, not fabricated failures; no candidate is guaranteed correct or optimal.

You have 80 complete training engines and 20 separate complete development engines. Full development RUL labels are legal for validation and model selection, never for fitting supervised parameters. You may further split training engines if useful. Do not random-split overlapping rows across training and validation. Original cached models used development early stopping; development is adaptive evidence, not independent certification. Raw train and development NPZ files and column/role metadata are under /data. Final observations, labels, scores and other Agents are unavailable. The final metric uses unchanged uncapped RUL; capped125 metrics are secondary. Complete historical development trajectories are available for analysis, but a prospective prediction cannot use observations later than its query time.

You may write arbitrary feature extraction and supported sklearn/PyTorch training and inference code. All supervised candidate fitting must use register_program(files,claim,rival); up to four new complete versions. Interface: program.py fit --train PATH --model-dir DIR; program.py predict --input PATH --model-dir DIR --output JSON. The fit NPZ contains train_values, development_values and columns. Fit only train_values; development_values may select versions/early stopping. Prediction NPZ contains values and columns for ONE engine through its observation time, with no labels. Return {"rul":[nonnegative finite predictions for each input row in original order]}. Fitted artifacts are read-only during prediction; no fitting during inference. Source is frozen by registration; subsequent edits need a new version. Actual estimator fits and optimizer updates are recorded. Use supplied source as a working interface example; its fit template reuses corresponding estimator settings, while original_agent.py preserves the full earlier authored script.

Tools: info; register_program; check(program_id); submit(program_id,reason). Registration returns full-trajectory development error. Optional check executes the exact registered program on real truncated development trajectories at three observation times per engine, comparing these predictions with full-context predictions at identical times. It returns actual per-engine errors and context changes. Both evidence types can imperfectly predict unknown engines. You may run numerical/source checks on legal data without fitting outside registration. Keep previous complete versions and decide whether to retain or replace. Save report.md with changes, actual checks, choice and uncertainty, then submit any complete registered program, including a reference. Submission is immutable.

No network, hidden-source access, other models, Agent delegation, invented measurements, mock fits or test-derived labels. Scratch under /app/.runtime, never /tmp. Budget:1800 seconds,60 model requests,4000000 input tokens,131072 output tokens,32768 output tokens/request,four registrations,two numeric threads. Each registration/check <=450 seconds,fit <=180 seconds,prediction process <=45 seconds. Failed registrations count; tools consume episode time. If discovery fails, python3 /tools/cmapss_program_tools.py client --request accepts the same action fields. A prose score is not a result.
"""
METHOD="""
Additional workflow under test: treat the validity of a program change as a falsifiable claim. Before selecting a replacement, identify what information or computation produced its apparent validation gain and a rival explanation. Use a concrete legal intervention that separates the explanations, rather than more verbal review. When a context/availability check exposes an invalid computation, localize that computation and construct a minimal corrected program preserving other potentially useful changes. Retrain and check the entire corrected program; evidence for one immutable version does not certify another. Compare actual benefits and regressions against the common initial program and uncorrected candidate. Keep negative findings. Decide whether to retain, repair, replace or stop. No specific algorithm, feature correction, threshold or outcome is prescribed.
"""
TEMPLATE='''import argparse,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import features
SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())
def matrix(values,columns):
    df=pd.DataFrame(values,columns=columns)
    df['engine_id']=df['engine_id'].astype(int);df['cycle']=df['cycle'].astype(int)
    result=getattr(features,SPEC['feature_function'])(df)
    return result[0],result[1],result[-1]
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','predict']);p.add_argument('--train');p.add_argument('--input');p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output');a=p.parse_args()
    if a.mode=='fit':
        z=np.load(a.train,allow_pickle=False);x,y,_=matrix(z['train_values'],z['columns'].tolist())
        dx,dy,_=matrix(z['development_values'],z['columns'].tolist());dy=np.minimum(dy,125)
        models=[];counts=[]
        for params in SPEC['estimator_parameters']:
            model=HistGradientBoostingRegressor(**params).fit(x,y);best=(float('inf'),0)
            for i,pred in enumerate(model.staged_predict(dx),1):
                loss=float(np.mean(np.abs(np.clip(pred,0,125)-dy)))
                if loss<best[0]:best=(loss,i)
            models.append(model);counts.append(best[1])
        a.model_dir.mkdir(parents=True,exist_ok=True)
        (a.model_dir/'fitted.pkl').write_bytes(pickle.dumps(models));(a.model_dir/'iterations.json').write_text(json.dumps(counts))
    else:
        z=np.load(a.input,allow_pickle=False);x,_,rows=matrix(z['values'],z['columns'].tolist())
        models=pickle.loads((a.model_dir/'fitted.pkl').read_bytes());counts=json.loads((a.model_dir/'iterations.json').read_text());preds=[]
        for model,count in zip(models,counts):
            for i,pred in enumerate(model.staged_predict(x),1):
                if i==count:preds.append(np.clip(pred,0,125));break
        assert len(preds)==len(models)
        pred=np.mean(preds,axis=0)[np.argsort(rows)]
        Path(a.output).write_text(json.dumps({'rul':pred.tolist()}))
if __name__=='__main__':main()
'''

def prepare(root):
    assert not (root/'PROTOCOL337.json').exists()
    prior=read(PROJECT/'results/contrast_transfer427/PROTOCOL427.json')
    cfg=dict(iteration='A435',frozen_utc=utc(),cli=prior['cli'],cli_sha256=prior['cli_sha256'],cli_version='0.164.0',
        model='deepseek-flash',arms=['ordinary','causal_repair'],repeats=2,episode_seconds=1800,episode_requests=60,
        episode_input_tokens=4000000,episode_output_tokens=131072,output_per_request=32768,max_registered_programs=4,
        build_stage_only=True,broker_script='cmapss_program_tools.py',stage='single_context_coding',maximum_parallel_sessions=4,
        maximum_disk_GiB=16,source_clusters=1,primary='All four normal; method mean final uncapped MAE below ordinary AND keep; harm no greater and gain no smaller than ordinary. Different pools not compared by own-oracle regret.',
        controls=['keep','original full-development minimum','original prefix minimum','fixed causal repair A433/A434','per-bank development/prefix selectors'],
        advancement='Pilot success necessary not sufficient; fixed repair comparison before confirmation admission.',
        no_outcome_based_reruns=True,confirmation_COMFAULDA72_HUST22='unopened',independent_confirmation_scenarios=0,
        novelty_established=False,source_scope='historically opened FD001 development pilot, not independent confirmation',
        factor='Shared programs/data/tools/budget; intervention and local repair workflow sole arm addendum.')
    assert sha(Path(cfg['cli']))==cfg['cli_sha256']
    save(root/'PROTOCOL337.json',cfg)
    save(root/'CLI_SOURCE435.json',dict(tag='rust-v0.164.0',checked_utc=utc(),released='2026-10-23',url='https://github.com/openai/codex/releases/tag/rust-v0.164.0',latest_url='https://github.com/openai/codex/releases/latest'))
    for name in closure(['run_cmapss_coding_pilot','cmapss_program_runtime','cmapss_program_tools','program_training_entry']):
        shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    for name in closure(['cmapss_program_tools','program_training_entry']):
        shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'common_tools'/(name+'.py'))
    shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',root/'common_tools/isolated_model_gateway.py')
    (root/'scripts').mkdir(exist_ok=True)
    shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',root/'scripts/isolated_model_gateway.py')
    source=Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks240/public/datasets/CMAPSS_FD001')
    raw=np.loadtxt(source/'train.txt');assert sha(source/'train.txt')=='963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8'
    dev_ids=np.array([8,10,23,29,41,44,51,52,53,55,71,72,73,77,83,89,90,93,95,97])
    train=raw[~np.isin(raw[:,0],dev_ids)];dev=raw[np.isin(raw[:,0],dev_ids)]
    columns=np.asarray(read(root/'restored/models/initial/case.json')['columns'])
    for name,values in [('train',train),('development',dev)]:np.savez_compressed(root/'inputs'/(name+'.npz'),values=values,columns=columns)
    np.savez_compressed(root/'inputs/fit_input.npz',train_values=train,development_values=dev,columns=columns)
    save(root/'inputs/metadata.json',dict(task='Prospective RUL cycles, current and past only',source='NASA official CMAPSS FD001 simulation',
        train_engines=80,development_engines=20,primary_metric='Uncapped official RUL MAE; equal weight per final engine',
        truth='Historical max cycle minus current cycle; development labels legal for selection only.',
        columns=columns.tolist(),train_engine_ids=sorted(set(train[:,0].astype(int).tolist())),development_engine_ids=dev_ids.tolist(),
        raw_train_sha256=sha(source/'train.txt'),query_information='One engine through query; no future or labels.',
        program_interface='See /reference/p00/source/program.py. Any complete compatible program allowed.',
        evidence_scope='Full-trajectory validation retrospective; optional real prefix checks prospective, adaptively reused.',
        cached_reference='Actual A431/A432 models; adapter transports exact features/iterations. Full original code retained. No repair outcome supplied.'))
    engine=dict(input_root=str(root/'inputs'),common_tools=str(root/'common_tools'),fit_seconds=180,predict_seconds=45,program_seconds=450,prediction_workers=4,prefix_fractions=[.25,.5,.75])
    save(root/'ENGINE_CONFIG.json',engine);initial={}
    for ident,name in [('p00','initial'),('ref_selected','selected'),('ref_rejected','rejected')]:
        folder=root/'reference'/ident;src=folder/'source';art=folder/'full_fit/artifacts';src.mkdir(parents=True);art.mkdir(parents=True)
        original=(root/'restored/models'/name/'source_program.py').read_text();spec=read(root/'restored/models'/name/'case.json')
        fn_name='build_features' if name=='rejected' else 'feats'
        node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name==fn_name)
        header='import numpy as np\nimport pandas as pd\nID="engine_id"\nCYC="cycle"\nRUL="RUL"\nCAP=125\nSENS='+repr(spec['sensor_columns'])+'\n'
        (src/'features.py').write_text(header+ast.get_source_segment(original,node)+'\n')
        (src/'original_agent.py').write_text(original);(src/'program.py').write_text(TEMPLATE)
        models=pickle.loads((root/'restored/models'/name/'fitted.pkl').read_bytes())
        spec.update(feature_function=fn_name,estimator_parameters=[m.get_params() for m in models]);save(src/'spec.json',spec)
        shutil.copy2(root/'restored/models'/name/'fitted.pkl',art/'fitted.pkl');save(art/'iterations.json',spec['iterations']);initial[ident]=str(folder)
    def check_initial(item):
        ident,path=item;folder=Path(path)
        result=dict(status='complete',program_id=ident,development=Runtime(root).development(folder),fit_entry_executions=0,
            confirmed_estimator_fits=0,actually_trained_reference_reuse=True,source_run='A432 refit, frozen transport A434',final_data_read=False,
            artifacts={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and ('source' in p.parts or 'artifacts' in p.parts)})
        save(folder/'REPORT.json',result);return result
    with ThreadPoolExecutor(max_workers=3) as pool:reports=list(pool.map(check_initial,initial.items()))
    finish_preparation(root,cfg,initial,reports)

def finish_preparation(root,cfg,initial,reports):
    for report in reports:
        name={'p00':'initial','ref_selected':'selected','ref_rejected':'rejected'}[report['program_id']]
        spec=read(root/'restored/models'/name/'case.json')
        difference=report['development']['capped125_MAE']-spec['source_dev_mae']
        report['transport_absolute_MAE_delta']=abs(difference)
        # Cold single-engine execution is not bitwise-equivalent to aggregate
        # pandas feature execution. Preserve the observed discrepancy explicitly.
        assert abs(difference)<1e-5,(name,report['development']['capped125_MAE'],spec['source_dev_mae'])
    save(root/'REFERENCE_QUALIFICATION435.json',dict(passed=True,reports=reports,new_fits=0,cold_processes=60))
    prompts=dict(ordinary=COMMON,causal_repair=COMMON+METHOD);save(root/'PROMPTS435.json',prompts);rows=[]
    for arm in cfg['arms']:
        for repeat in [1,2]:
            run=root/'subjects'/(arm+'_'+str(repeat))
            for d in ['app/.runtime/tmp','app/.runtime/mpl','runtime/tmp','runtime/mpl','logs','plans','audit','broker','primary','feedback','ipc','scripts']:(run/d).mkdir(parents=True)
            (run/'app/AGENTS.md').write_text('Real train-only fitting via register_program. Current/past-only prediction. No network or delegation. Scratch /app/.runtime.\n')
            shutil.copy2(root/'ENGINE_CONFIG.json',run/'ENGINE_CONFIG.json')
            shutil.copy2(root/'restored/model_catalog187.json',run/'model_catalog187.json')
            shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',run/'scripts/isolated_model_gateway.py')
            mcp='''[mcp_servers.phmforge]
command = "/opt/python312/bin/python3.12"
args = ["-B", "/tools/cmapss_program_tools.py", "mcp"]
required = true
env = { PYTHONPATH = "/opt/phm_python/extra:/tools", PYTHONDONTWRITEBYTECODE = "1", TMPDIR = "/app/.runtime/tmp", TMP = "/app/.runtime/tmp", TEMP = "/app/.runtime/tmp", MPLCONFIGDIR = "/app/.runtime/mpl", OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "2", MKL_NUM_THREADS = "2" }
startup_timeout_sec = 60
tool_timeout_sec = 600
'''
            (run/'cli_config187.toml').write_text((root/'restored/base_cli_config.toml').read_text()+mcp)
            save(run/'mounts.json',dict(mounts=[dict(source=str(p),destination=q) for p,q in [(PYTHON,'/opt/python312'),(DEPS,'/opt/phm_python/extra'),(root/'common_tools','/tools'),(root/'inputs','/data'),(root/'reference','/reference'),(run/'feedback','/feedback')]]))
            save(run/'BROKER_CONFIG.json',dict(initial_programs=initial,contract=str(root/'inputs/metadata.json'),max_programs=4,request_directory='app/.runtime/feedback_requests'))
            save(run/'PROTOCOL187.json',dict(cli=cfg['cli'],expected_mcp_tools=['mcp__phmforge__'+n for n in ['info','register_program','check','submit']]))
            save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=run.name,socket=str(run/'ipc/s.sock'))],base_url='https://api.deepseek.com/',model='deepseek-flash',max_requests=60,max_input_tokens=4000000,max_output_tokens=131072,max_output_per_request=32768,max_inflight=1,denied_tool_namespaces=['collaboration'],denied_tool_names=['spawn_agent','followup_task','send_message','list_agents'],buffer_verified_responses=True))
            (run/'plans/prompt.txt').write_text(prompts[arm]);rows.append(dict(id=run.name,arm=arm,repeat=repeat,path=str(run)))
    save(root/'SESSIONS337.json',rows)
    save(root/'EFFECTIVE_CODE_FREEZE337.json',{str(p.relative_to(root)):sha(p) for d in ['code','common_tools','inputs','reference'] for p in (root/d).rglob('*') if p.is_file()})
    save(root/'PREPARATION435.json',dict(passed=True,utc=utc(),same_information_tools_budget=True,method_only_prompt_addendum=True,new_LLM_calls=0,new_fits=0,official_test_mounted=False))

def qualify_one(root,row):
    run=Path(row['path']);ns=arguments(root,run);fixture=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);fixture.bind(str(run/'ipc/s.sock'));fixture.listen(1)
    folder=run/'audit/isolation_checked';folder.mkdir()
    check="""import os,socket,json
from pathlib import Path
c=dict(private_absent=not any(Path(p).exists() for p in ['/home/jupyter-yyf-zyq/.bashrc','/data/yyf-zyq/PHM_20261010/20261026T192330Z_transport434','/srv/yyf-zyq/PHM_20261010/_shared','/home/jupyter-yyf-zyq/Bo_2/PHM_20261010']),secret_absent=not any('API_KEY'in k or 'SECRET'in k for k in os.environ),tmp_legal=os.environ['TMPDIR'].startswith('/app/.runtime/'),public_allowlist={p.name for p in Path('/data').iterdir()}=={'train.npz','development.npz','fit_input.npz','metadata.json'},references=len(list(Path('/reference').iterdir()))==3)
try:s=socket.create_connection(('1.1.1.1',443),timeout=1);s.close();c['network_blocked']=False
except OSError:c['network_blocked']=True
print(json.dumps(c));assert all(c.values())
"""
    env=environment(root);env.update(PATH='/opt/python312/bin:/usr/bin:/bin',PYTHONPATH='/opt/phm_python/extra:/tools',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp')
    def isolated(code,where):
        command=isolate_filesystem([str(ns.cli),'sandbox','-P','iot-offline','-C','/app','--','/opt/python312/bin/python3.12','-B','-c',code],ns,run/'app',where,None)
        p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
        save(where/'RESULT.json',dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,p.stderr[-1600:]
    try:isolated(check,folder)
    finally:fixture.close();(run/'ipc/s.sock').unlink()
    result=probe(run,run,dict(id=run.name,mounts=str(run/'mounts.json')),'fixed_discovery',(run/'cli_config187.toml').read_text());assert result['passed']
    worker=None;fixture=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:
        with (run/'logs/transport.log').open('w') as log:worker=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B',str(root/'code/cmapss_program_tools.py'),'serve','--run',str(run)],env=environment(root),stdout=log,stderr=subprocess.STDOUT)
        for _ in range(200):
            if (run/'BROKER_READY.json').exists():break
            assert worker.poll() is None;time.sleep(.1)
        assert (run/'BROKER_READY.json').exists()
        fixture.bind(str(run/'ipc/s.sock'));fixture.listen(1)
        folder=run/'audit/transport';folder.mkdir()
        isolated("from cmapss_program_tools import client;r=client({'action':'info'});assert r['ok'] and len(r['result']['programs'])==3;print('actual_transport_passed')",folder)
    finally:
        if worker is not None:worker.terminate();worker.wait(timeout=10)
        fixture.close();(run/'ipc/s.sock').unlink(missing_ok=True);(run/'BROKER_READY.json').unlink(missing_ok=True)
    return dict(id=row['id'],passed=True,real_LLM_calls=0)

def qualify(root):
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda row:qualify_one(root,row),read(root/'SESSIONS337.json')))
    save(root/'NATIVE_QUALIFICATION435.json',dict(passed=True,rows=rows,utc=utc()))

def agents(root):
    assert read(root/'NATIVE_QUALIFICATION435.json')['passed'];assert not (root/'HANDLE435.json').exists(),'Do not restart native episodes'
    for name,digest in read(root/'EFFECTIVE_CODE_FREEZE337.json').items():assert sha(root/name)==digest,name
    save(root/'HANDLE435.json',dict(pid=os.getpid(),utc=utc(),state='running'));results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(execute,root,row) for row in read(root/'SESSIONS337.json')]):
            results.append(f.result());save(root/'PROGRESS435.json',results)
    save(root/'ALL_CHOICES_SEALED435.json',dict(utc=utc(),results=results,official_test_scores_opened=False))
    save(root/'TERMINAL435.json',dict(utc=utc(),sessions=len(results),normal=sum(r['completed'] for r in results),exit_code=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','qualify','agents']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.run.resolve())
