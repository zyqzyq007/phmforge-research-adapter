"""Execute an outcome-blind feedback-timing probe on real UoC observations.

This is a constrained mechanism experiment using the native coding CLI, not an
unrestricted coding benchmark. Original acquisition identities are unavailable;
the estimand is within the released collection, never cross-device performance.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

from controlled_feedback import SPACE, client, save, sha, utc
from prepare_forge_coding_baseline import CLI, CLI_SHA, PYTHON, RUNTIME
from replay_diagnostic_cli import isolate_filesystem
from run_full_benchmark_campaign import clean_env
from run_frozen_candidate_review import source_context
from run_validation_transfer import execute_wave, freeze, metadata_gate, start_gateway, state, command

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261022T050024Z_source275')
TEMPLATE = Path('/data/yyf-zyq/PHM_20261010/20261022T130911Z_review281/rotor_a')

PROMPT = '''
Classify measured gearbox vibration observations into the nine documented UoC input-pinion states. This is an explicitly adapted PHMForge pdm_fault_010 task, not a claim that the original question's component/race labels were valid.

This session is a controlled model-search experiment. Use the same permitted recipe space and exactly six distinct candidate requests. All models are really fitted on training observations. Every request also performs three-fold training-only CV; normalization is fitted inside each fold. The only experimental restriction is that candidate model families, feature choices, parameters and fitting opportunities are fixed by /app/recipe_space.json. This is not unrestricted algorithm development.

Read /data/datasets/UOC/dataset.json and /app/recipe_space.json. The public train.npz contains waveforms[observation,3600], labels, opaque observation IDs and one observation ID per group. That group field is NOT a physical recording or hardware identity: the publisher does not provide these. Data are real measured signals, already processed by the publisher, with no fabricated faults or synthetic waveforms. The coordinate axis must not be assumed to have a verified physical sample rate. Final conclusions are limited to the published collection. The precomputed Stage0/PCA representation is excluded.

The private numerical service can be called from Python or shell:
  python3 /tools/controlled_feedback.py client --request '{"action":"status"}'
  python3 /tools/controlled_feedback.py client --request '{"action":"fit","recipe":{"features":"stats_fft64","model":"rf","params":{"max_depth":null,"min_samples_leaf":1}}}'
  python3 /tools/controlled_feedback.py client --request '{"action":"select","candidate_id":"candidate_1","reason":"Explain the observed evidence and limitations"}'

Each accepted distinct recipe consumes one opportunity and fits three CV models plus one saved deployment model. The six model families have the exact parameter grids in recipe_space.json. A rejected invalid request does not consume a fit, but a numerical failure or timeout does. At most one fit runs at a time within this session; a fit may take up to 180 seconds. Do not request duplicate recipes. The service returns real training/CV results immediately. DEVELOPMENT_TIMING

You may use the native coding and MCP tools to read training data, inspect the feature/model implementation, write analysis code, compare recorded results and reason about model selection. The generic MCP fitting tools are disabled to preserve the six-fit budget: use the service for all fitting. Do not fit extra estimators in /app, change the service, tune new parameters or construct ensembles outside the recipe space. Train-only CV is the same in both conditions. Development observations and labels are not mounted. They are used only inside the service, never in model fitting. Final observations and labels are absent from both the service and your workspace.

After six requests, inspect all available development and CV results. Review your selection before sealing it; retaining an earlier candidate is fine. Call select exactly once, then copy the returned selection into selection.json and write report.md explaining checks, reasons for retaining/rejecting candidates, uncertainty and source limitations. The external evaluator deploys that exact saved checkpoint; it does not score an invented answer. Do not fabricate final test scores. If fitting fails or you cannot finish, record that explicitly.

No Internet, other cases, extra LLM APIs or final labels are available. All temporary files belong in /app/.runtime, never /tmp. Budget:2400 seconds,120 model requests,32768 output tokens/request,262144 cumulative output tokens,16000000 completed input tokens,two numerical threads. Both successful and unsuccessful cases are retained.
'''

PROBE = r'''
import json,os,socket,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/tools')
from verified_phmforge import VerifiedForge
from controlled_feedback import client
checks={}
spec=json.loads(sys.argv[2])
dataset=spec['dataset']
private_paths=[sys.argv[1],str(Path(sys.argv[1]).parent/'source'),str(Path(sys.argv[1]).parent/'immediate_1/broker'),
 '/home/jupyter-yyf-zyq/.bashrc','/home/jupyter-yyf-zyq/Bo_2/PHM_20261010/AGNETS.md',
 '/data/yyf-zyq/PHM_20261010','/srv/yyf-zyq/PHM_20261010/_shared']
checks['private_paths_absent']=not any(Path(p).exists() for p in private_paths)
checks['credentials_absent']=not any(any(q in k.upper() for q in ['API_KEY','ACCESS_TOKEN','PASSWORD','SECRET']) for k in os.environ)
checks['parent_credentials_absent']=b'API_KEY=' not in Path('/proc/1/environ').read_bytes()
checks['no_development_or_final_files']=set(p.name for p in (Path('/data/datasets')/dataset).iterdir())=={'train.npz','dataset.json'}
train=np.load('/data/datasets/'+dataset+'/train.npz',allow_pickle=False)
checks['real_train_available']=train['waveforms'].shape==tuple(spec['train_shape']) and np.isfinite(train['waveforms']).all()
checks['safe_temp']=os.environ['TMPDIR'].startswith('/app/.runtime')
s=socket.socket();s.settimeout(1)
try:s.connect(('1.1.1.1',443));checks['network_blocked']=False
except OSError:checks['network_blocked']=True
finally:s.close()
api=VerifiedForge('/data','/app/artifacts/preflight')
for split in ['val','test']:
 for tool,arg in [('load_dataset','split'),('load_ground_truth','file')]:
  checks[tool+'_'+split+'_denied']=api.call(tool,{'dataset_name':dataset,arg:split})['status']=='unavailable'
checks['uncounted_fitting_disabled']=api.call('train_fault_classifier',{'dataset_name':dataset})['training_performed'] is False
r=client({'action':'status'})
if spec.get('search_mode')=='ensemble_weights':
 from controlled_ensemble_feedback import client
 value=r['result']
 checks['broker_connected_initial_pool']=r['ok'] and value['remaining']==spec['candidate_budget'] and value['ensemble_candidates']==0 and value['basis_anchors']==6
 checks['raw_development_policy']=(Path('/feedback/development_probabilities.npz').exists()==(spec.get('search_stage')=='review'))
 checks['training_probabilities_available']=Path('/feedback/training_cv_probabilities.npz').exists()
else:checks['broker_connected_no_trials']=r['ok'] and r['result']['remaining']==spec['candidate_budget'] and not r['result']['trials']
r=client({'action':'read_file','path':'/hidden'})
checks['broker_arbitrary_path_denied']=not r['ok']
checks={k:bool(v) for k,v in checks.items()}
print(json.dumps({'passed':all(checks.values()),'checks':checks}))
raise SystemExit(0 if all(checks.values()) else 1)
'''


def numerical_env(root):
    env = clean_env(root)
    env.update(PYTHONPATH=str(RUNTIME/'deps')+':'+str(PROJECT/'scripts'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return env


def recover_transport(root,previous):
    """Fresh subjects on exactly the same allocation after a zero-fit transport failure."""
    old=json.loads((previous/'CAMPAIGN_CONFIG.json').read_text())
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    for c in old['cases']:
        prior_state=json.loads((previous/c['id']/'broker/STATE.json').read_text())
        assert not prior_state['trials'] and prior_state['final_choice'] is None
        for p in (previous/c['id']/'primary').glob('*/*/status.json'):
            assert json.loads(p.read_text())['status']!='running'
    for name in ['audit','code','logs','runtime/tmp','runtime/mpl','clients-cache','shared_tools']:(root/name).mkdir(parents=True,exist_ok=True)
    config=dict(old,run_id=root.name,created_utc=utc(),runtime=str(root),recovery_attempt=1,
        infrastructure_attempt_archive='/data/yyf-zyq/PHM_20261010/'+previous.name,
        recovery='Native command AF_UNIX denial prevented all candidate fits. Restore only initial inputs,keep exact allocation and budgets,replace transport with scoped file queue;do not reuse failed Agent outputs.',
        total_native_attempts_including_infrastructure=12,maximum_total_API_requests_including_observed_infrastructure=930)
    save(root/'CAMPAIGN_CONFIG.json',config)
    bindings=[]
    for folder in ['source','hidden']:
        shutil.copytree(previous/folder,root/folder)
        for p in (root/folder).rglob('*'):
            if p.is_file():
                relative=p.relative_to(root);assert sha(p)==sha(previous/relative)
                bindings.append(dict(path=str(relative),sha256=sha(p)))
    for name in ['UOC_QUALIFICATION275.json','UOC_NATIVE_COLUMN_BINDINGS275.json','PRIVATE_ALLOCATION282.json','qualification_train.npz','qualification_dev.npz','qualification_recipe.json','CLI_ONLINE_CHECK.json']:
        shutil.copy2(previous/'audit'/name,root/'audit'/name)
    shutil.copytree(previous/'audit/qualification_fit',root/'audit/qualification_fit')
    shutil.copy2(previous/'retention_policy.json',root/'retention_policy.json')
    for p in (previous/'shared_tools').glob('*.py'):
        shutil.copy2(PROJECT/'scripts'/p.name,root/'shared_tools'/p.name)
    for c in config['cases']:
        before=previous/c['id'];run=root/c['id'];job=json.loads((before/'PLAN187.json').read_text())['jobs'][0]
        for name in ['audit','code','logs','plans','references','feedback','broker','runtime/tmp','runtime/mpl']:(run/name).mkdir(parents=True,exist_ok=True)
        shutil.copytree(before/'public_data',run/'public_data')
        ws=run/job['workspace']
        for name in ['artifacts','.runtime/tmp','.runtime/mpl','.runtime/xdg']:(ws/name).mkdir(parents=True,exist_ok=True)
        frozen={x['path']:x['sha256'] for x in json.loads((before/'INITIAL_FREEZE187_FIFO_FILES.json').read_text())['files']}
        for name in ['AGENTS.md','recipe_space.json']:
            p=before/job['workspace']/name;assert sha(p)==frozen[str(p)];shutil.copy2(p,ws/name)
        for p in (before/'plans').iterdir():
            (run/'plans'/p.name).write_text(p.read_text().replace(str(previous),str(root)))
        (run/'PLAN187.json').write_text((before/'PLAN187.json').read_text().replace(str(previous),str(root)))
        for name in ['cli_config187.toml','model_catalog187.json','MODEL_GATEWAY_CONFIG.json','TASK_QUALIFICATION251.json']:
            (run/name).write_text((before/name).read_text().replace(str(previous),str(root)))
        protocol=json.loads((before/'PROTOCOL187.json').read_text());protocol['additional_gates'].append('NATIVE_FEEDBACK_GATE282.json');save(run/'PROTOCOL187.json',protocol)
        shutil.copy2(root/'audit/CLI_ONLINE_CHECK.json',run/'audit/CLI_ONLINE_CHECK.json')
        save(run/'audit/AUTHORIZATION251.json',dict(user_reply=config['authorization'],scope=config['purpose'],infrastructure_recovery_only=True))
        save(run/'BROKER_CONFIG.json',dict(condition=c['condition'],train=str(root/'hidden/train.npz'),development=str(root/'hidden/val.npz'),request_directory=str(ws.relative_to(run)/'.runtime/feedback_requests')))
        save(run/'OWNERSHIP.json',dict(owner='root',writers_stopped=True,status='recovery_prepared_waiting_qualification'))
        metadata_gate(run);freeze(run)
        initial=json.loads((run/'INITIAL_FREEZE187_FIFO_FILES.json').read_text())
        initial['files'] += [dict(path=str(p),sha256=sha(p)) for p in [*sorted((root/'shared_tools').glob('*.py')),run/'BROKER_CONFIG.json',PROJECT/'scripts/controlled_feedback.py']]
        save(run/'INITIAL_FREEZE187_FIFO_FILES.json',initial)
    save(root/'audit/RECOVERY_INPUT_BINDINGS282.json',dict(utc=utc(),previous_attempt=str(previous),all_input_bytes_equal=True,files=bindings))
    for name in ['controlled_feedback.py','run_controlled_feedback.py','qualify_feedback_native.py','evaluate_controlled_feedback.py','verified_phmforge.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'code'/name)
    dest=PROJECT/'results/feedback282/PROTOCOL282.json'
    if not (dest.parent/'PROTOCOL282_INITIAL.json').exists():shutil.copy2(dest,dest.parent/'PROTOCOL282_INITIAL.json')
    save(dest,config)
    state(root,status='transport_recovery_prepared_waiting_native_tool_fit_gate',planned_native_sessions=6,previous_infrastructure_native_attempts=6,previous_requests=210,new_formal_fits=0,final_scores_opened=False)


def prepare(root):
    import numpy as np
    from scipy.io import loadmat
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    for name in ['audit','source','hidden','shared_tools','code','logs','runtime/tmp','runtime/mpl','clients-cache']:
        (root/name).mkdir(parents=True,exist_ok=True)
    cases = [dict(id=f'{condition}_{i}', condition=condition, replicate=i) for i in range(1,4) for condition in ['immediate','delayed']]
    config = dict(iteration='A282', run_id=root.name, created_utc=utc(), result_directory='results/feedback282',
        purpose='Equal-space,equal-fit-budget development-feedback-timing contrast on unused real UoC published observations; constrained mechanism probe.',
        authorization='用户：持续迭代对PHMforge进行真实实验（不要模拟数据），继续验证agent容易对训练数据过拟合的问题。',
        goal_status='active', previous_goal_turn='progress:A281 six real fixed-candidate reviews completed,226 model requests,positive cases and counterevidence archived.',
        scope_revision='A275 rejected guessed original-record groups, not a declared within-collection control. Use whole published columns without claiming independent acquisitions. This probe does not satisfy cross-device confirmation.',
        hypothesis='Immediate reused development feedback may improve apparent development F1 while worsening final F1 relative to delayed feedback at equal candidate-search opportunity.',
        primary_estimands=['mean final macro-F1 immediate minus delayed','mean development-minus-final macro-F1 gap immediate minus delayed'],
        secondary=['accuracy','all candidate CV/development/final metrics','retained-candidate regret','delivered choices and all failure states'],
        interpretation='Three stochastic trajectories per arm share one allocation. Descriptive contrast only; no powered population test,independent-device claim,or requirement to find a failure.',
        falsification='No additional gap or better final performance under immediate feedback is counterevidence in this scope; generic low performance is not causal proof.',
        major_loss_rule='>=0.05 final macro-F1 loss with no accuracy gain versus an actually fitted available candidate; other differences remain descriptive.',
        split=dict(seed=282104, train_per_class=62, development_per_class=21, final_per_class=21, unit='whole published matrix column', original_acquisition_groups_unknown=True),
        cases=cases, planned_sessions=6, candidate_fits_per_case=6, models_per_candidate=4, maximum_formal_estimator_fits=144,
        recipe_space=SPACE, numerical_seed=282, only_changed_factor='development feedback released immediately or after all six candidates have been committed and fitted',
        session_budget=dict(subject_timeout_seconds=2400,mcp_tool_timeout_seconds=180,max_model_requests_per_subject=120,
            max_total_input_tokens_per_subject=16000000,max_total_output_tokens_per_subject=262144,max_output_tokens_per_request=32768),
        parallel_sessions=6, artifact_budget_bytes=8*1024**3, expected_API_requests='100-300;hard cap720',
        API_price_not_asserted=True, expected_wall_minutes='15-35 after qualification', runtime='/srv/yyf-zyq/PHM_20261010/'+root.name,
        final_scores_unopened=True, no_adaptive_retry=True, no_tmp=True, original_benchmark_score_claimed=False)
    save(root/'CAMPAIGN_CONFIG.json',config)
    save(PROJECT/'results/feedback282/PROTOCOL282.json',config)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',shared_dependencies=dict(path=str(RUNTIME.parent),retention='Pinned dependencies and hash-bound source archive retained.')))
    manifest, receipt = source_context(SOURCE)
    required = ['source/DataForClassification_TimeDomain.mat','source/uoc_metadata.json','audit/UOC_QUALIFICATION275.json','audit/UOC_NATIVE_COLUMN_BINDINGS275.json']
    for rel in required:
        src=SOURCE/rel;row=manifest[rel]
        assert src.stat().st_size==row['bytes'] and sha(src)==row['sha256']
        dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    q=json.loads((root/'audit/UOC_QUALIFICATION275.json').read_text())
    assert q['source_qualified_as_measured'] and not q['physical_hardware_or_acquisition_ids_available']
    waves=loadmat(root/'source/DataForClassification_TimeDomain.mat')['AccTimeDomain'].T
    labels=np.repeat(np.arange(9),104)
    bindings=json.loads((root/'audit/UOC_NATIVE_COLUMN_BINDINGS275.json').read_text())
    assert waves.shape==(936,3600)
    assert all(hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()==r['signal_sha256'] for x,r in zip(waves,bindings))
    rng=np.random.default_rng(282104);splits={'train':[],'val':[],'test':[]}
    for label in range(9):
        order=rng.permutation(np.where(labels==label)[0])
        for part,indices in zip(splits,[order[:62],order[62:83],order[83:]]):splits[part].extend(indices.tolist())
    ids=np.asarray([secrets.token_hex(16) for _ in labels]);mapping=[]
    for part,indices in splits.items():
        rng.shuffle(indices);ix=np.asarray(indices)
        np.savez_compressed(root/'hidden'/(part+'.npz'),waveforms=waves[ix],labels=labels[ix],sample_ids=ids[ix],groups=ids[ix],dataset=np.asarray('UOC'))
        mapping += [dict(sample_id=str(ids[i]),native_column=int(i),label=int(labels[i]),split=part) for i in ix]
    assert len({x['native_column'] for x in mapping})==936
    save(root/'audit/PRIVATE_ALLOCATION282.json',mapping)
    # Qualify numerical fitting with training-only observations; do not open formal development metrics.
    train=np.load(root/'hidden/train.npz',allow_pickle=False)
    qi=np.concatenate([np.where(train['labels']==c)[0][:12] for c in range(9)])
    vi=np.concatenate([np.where(train['labels']==c)[0][12:16] for c in range(9)])
    for name,index in [('train',qi),('dev',vi)]:
        np.savez_compressed(root/'audit'/('qualification_'+name+'.npz'),**{k:(train[k] if train[k].ndim==0 else train[k][index]) for k in train.files})
    save(root/'audit/qualification_recipe.json',dict(features='stats_fft64',model='rf',params={'max_depth':None,'min_samples_leaf':1}))
    invocation=[str(PYTHON/'bin/python3.12'),'-B',str(PROJECT/'scripts/controlled_feedback.py'),'fit','--recipe',str(root/'audit/qualification_recipe.json'),
        '--train',str(root/'audit/qualification_train.npz'),'--dev',str(root/'audit/qualification_dev.npz'),'--output',str(root/'audit/qualification_fit')]
    save(root/'audit/NUMERICAL_QUALIFICATION_COMMAND.json',invocation)
    result=subprocess.run(invocation,env=numerical_env(root),capture_output=True,text=True,timeout=180)
    (root/'audit/numerical.stdout').write_text(result.stdout);(root/'audit/numerical.stderr').write_text(result.stderr)
    assert result.returncode==0,result.stderr[-2000:]
    real=json.loads((root/'audit/qualification_fit/fit_report.json').read_text());assert real['real_fits']==4 and real['cold_reload_equal']
    url='https://registry.npmjs.org/@openai/codex/latest'
    with urllib.request.urlopen(url,timeout=30) as response: latest=json.load(response)
    save(root/'audit/npm_codex_latest.json',latest)
    assert latest['version']=='0.163.0' and sha(CLI)==CLI_SHA
    save(root/'audit/CLI_ONLINE_CHECK.json',dict(utc=utc(),url=url,latest_stable_version=latest['version'],selected_version='0.163.0',cli_sha256=CLI_SHA))
    for name in ['controlled_feedback.py','prepare_fault_extensions.py','acquire_forge_pdmb_sources244.py','verified_phmforge.py','verified_benchmark_worker.py','verified_benchmark_scoring.py','real_fault_model.py','real_rul_model.py','isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'shared_tools'/name)
    for case in cases:
        run=root/case['id'];ident='s282_'+case['id'];ws=run/'workspaces'/ident
        for name in ['audit','code','logs','plans','references','public_data/datasets/UOC','feedback','broker','runtime/tmp','runtime/mpl']:(run/name).mkdir(parents=True,exist_ok=True)
        for name in ['artifacts','.runtime/tmp','.runtime/mpl','.runtime/xdg']:(ws/name).mkdir(parents=True,exist_ok=True)
        public=run/'public_data';ds=public/'datasets/UOC'
        shutil.copy2(root/'hidden/train.npz',ds/'train.npz')
        info=dict(dataset='UOC',class_names=list(range(9)),class_meanings=['healthy','missing','crack','spall','chip5a','chip4a','chip3a','chip2a','chip1a'],
            source_sha256=q['native_sha256'],source_DOI=q['publisher_DOI'],labels='Publisher column ranges; opaque IDs contain no source ordering.',
            axis='Published 3600-point observation; physical clock after preprocessing not certified.',
            group_definition='Each published column is indivisible. No certified original acquisition/hardware identity; no cross-device claim.',
            fitting_scope='Training only; same collection train-only CV; no Stage0 or development refit.',
            information_scope='Only training arrays visible. The private broker controls development feedback. Final observations/answers not mounted.',
            available_splits=['train'],official_original_benchmark_score_claimed=False)
        save(ds/'dataset.json',info)
        save(public/'dataset_registry.json',{'UOC':dict(task='classification',path='datasets/UOC',architectures=['mlp'],test_inputs_visible=False,available_splits=['train'])})
        shutil.copy2(TEMPLATE/'public_data/tool_catalog.json',public/'tool_catalog.json')
        save(public/'execution_policy.json',dict(allow_new_fitting=False,reason='Controlled fitting is available through the six-candidate numerical broker. Generic uncounted fitting is disabled.'))
        save(ws/'recipe_space.json',SPACE)
        (ws/'AGENTS.md').write_text('Use the six-candidate private fitting service and the declared recipe space for all model fitting. Training data are public; development feedback timing is enforced. No extra estimators, ensembles or fitted transforms. All files/caches under /app/.runtime, never /tmp. Final data and scores are unavailable.\n')
        timing=('Each completed candidate immediately receives its development metrics; later recipes may use this evidence.' if case['condition']=='immediate' else
                'Development metrics are withheld until all six candidate recipes have been committed and fitted. Earlier results contain training/CV only. After the sixth fit, all six development metrics become available for final selection.')
        prompt=run/'plans'/(ident+'.txt');prompt.write_text(PROMPT.replace('DEVELOPMENT_TIMING',timing))
        mounts=run/'plans'/(ident+'_mounts.json')
        save(mounts,dict(mounts=[dict(source=str(PYTHON),destination='/opt/python312'),dict(source=str(RUNTIME/'deps'),destination='/opt/phm_python/extra'),
            dict(source=str(root/'shared_tools'),destination='/tools'),dict(source=str(public),destination='/data'),dict(source=str(run/'references'),destination='/reference'),
            dict(source=str(run/'feedback'),destination='/feedback')]))
        native=dict(id=ident,family='forge_real_coding',task_id='pdm_fault_010',condition=case['condition'],repeat=case['replicate'],workspace=str(ws),prompt_file=str(prompt),answer_mode='text')
        plan=run/'plans'/(ident+'.json');save(plan,dict(jobs=[native]))
        sock=run/'ipc/s0.sock';assert len(str(sock))<108
        job=dict(native,workspace=str(ws.relative_to(run)),plan=str(plan),mounts=str(mounts),model_gateway_socket=str(sock),dataset='UOC',test_inputs_visible_during_development=False)
        save(run/'PLAN187.json',dict(jobs=[job]))
        protocol=dict(config['session_budget'],model='deepseek-flash',cli=str(CLI),cli_sha256=CLI_SHA,max_new_model_sessions=1,
            max_new_primary_rolling24h=400,local_max_concurrency=1,max_concurrency=8,training_gate='TASK_QUALIFICATION251.json',
            environment_gate='AGENT_ISOLATION_GATE251.json',authorization_record='audit/AUTHORIZATION251.json',controller_status_name='CONTROLLER_STATUS251.json',
            external_model_gateway=True,test_inputs_visible=False,original_benchmark_score_claimed=False,
            additional_gates=['PUBLIC_METADATA_GATE267.json','NATIVE_CLI_MCP_QUALIFICATION.json','OUTPUT_BUDGET_QUALIFICATION.json'])
        save(run/'PROTOCOL187.json',protocol)
        save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=ident,socket=str(sock))],base_url='https://api.deepseek.com/',model='deepseek-flash',
            max_requests=120,max_input_tokens=16000000,max_output_tokens=262144,max_output_per_request=32768))
        for name in ['cli_config187.toml','model_catalog187.json']:shutil.copy2(TEMPLATE/name,run/name)
        shutil.copy2(root/'audit/CLI_ONLINE_CHECK.json',run/'audit/CLI_ONLINE_CHECK.json')
        save(run/'TASK_QUALIFICATION251.json',dict(passed=True,measured_source=True,source_qualification=q,real_numerical_fits_verified=4,
            independent_recording_generalization_qualified=False,scope='Explicit within-published-collection controlled experiment'))
        save(run/'audit/AUTHORIZATION251.json',dict(user_reply=config['authorization'],scope=config['purpose']))
        save(run/'BROKER_CONFIG.json',dict(condition=case['condition'],train=str(root/'hidden/train.npz'),development=str(root/'hidden/val.npz'),request_directory=str(ws.relative_to(run)/'.runtime/feedback_requests')))
        save(run/'OWNERSHIP.json',dict(owner='root',writers_stopped=True,status='prepared_waiting_qualification'))
        metadata_gate(run);freeze(run)
        initial=json.loads((run/'INITIAL_FREEZE187_FIFO_FILES.json').read_text())
        initial['files'] += [dict(path=str(p),sha256=sha(p)) for p in [*sorted((root/'shared_tools').glob('*.py')),run/'BROKER_CONFIG.json',PROJECT/'scripts/controlled_feedback.py']]
        save(run/'INITIAL_FREEZE187_FIFO_FILES.json',initial)
    for name in ['run_controlled_feedback.py','controlled_feedback.py','verified_phmforge.py']:shutil.copy2(PROJECT/'scripts'/name,root/'code'/name)
    save(root/'PREPARED282.json',dict(utc=utc(),passed=True,source_archive=str(SOURCE),source_manifest_sha256=receipt['manifest_sha256'],
        train=558,development=189,final=189,qualification_real_fits=4,formal_fits=0,LLM_calls=0,final_scores_opened=False))
    state(root,status='prepared_waiting_native_qualification',planned_sessions=6,new_formal_fits=0)


def qualify(root, run):
    job=json.loads((run/'PLAN187.json').read_text())['jobs'][0];ws=run/job['workspace']
    gateway=start_gateway(run)
    try:
        cache=run/'clients-cache'/('isolation_'+str(len(list((run/'clients-cache').glob('isolation*')))+1))
        ns=SimpleNamespace(cli=CLI,model_gateway_socket=Path(job['model_gateway_socket']),runtime_mounts_file=Path(job['mounts']),credential_env_file=None,
            isolation_config=run/'cli_config187.toml',model_catalog=run/'model_catalog187.json',permission_profile='iot-offline',workspace_alias='/app')
        config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
        probe_spec=dict(dataset=config.get('dataset','UOC'),train_shape=config.get('train_shape',[558,3600]),candidate_budget=config['candidate_fits_per_case'],
            search_mode=config.get('search_mode'),search_stage=config.get('search_stage','search'))
        invocation=isolate_filesystem(['/opt/python312/bin/python3.12','-B','-c',PROBE,str(root/'hidden'),json.dumps(probe_spec)],ns,ws,cache,None)
        save(run/'audit/ISOLATION_COMMAND282.json',invocation)
        env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',PATH='/opt/python312/bin:/usr/bin:/bin',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl')
        r=subprocess.run(invocation,env=env,capture_output=True,text=True,timeout=90)
        (run/'audit/isolation.stdout').write_text(r.stdout);(run/'audit/isolation.stderr').write_text(r.stderr)
        assert r.returncode==0,r.stderr+r.stdout
        result=json.loads(r.stdout.strip().splitlines()[-1]);assert result['passed']
        save(run/'AGENT_ISOLATION_GATE251.json',result)
        command(run,'qualify_native_cli_mcp.py','--run',run,'--template',run,'--conditions','fixed_discovery')
        command(run,'qualify_model_output_budget.py','--run',run,'--template',run)
        for name in ['NATIVE_CLI_MCP_QUALIFICATION.json','OUTPUT_BUDGET_QUALIFICATION.json']:shutil.copy2(run/'audit'/name,run/name)
    finally:
        (run/'MODEL_GATEWAY_STOP').touch();gateway.wait(timeout=30)
        # The established gateway retains closed Unix socket nodes on shutdown.
        # Remove only this stopped process's endpoints before real admission.
        for name in ['MODEL_GATEWAY_STOP','MODEL_GATEWAY_READY.json']:
            path=run/name
            if path.exists():path.unlink()
        path=Path(job['model_gateway_socket'])
        if path.is_socket():path.unlink()


def execute(root,resume=False):
    os.environ.setdefault('PHM_CAPACITY_CACHE','/srv/yyf-zyq/PHM_20261010/_shared/verified_capacity_cache')
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    if resume:
        assert all(not (root/c['id']/'primary').exists() for c in config['cases']), 'Recovery is only before any native subject'
        assert (root/'audit/PRE_ADMISSION_RECOVERY282.json').exists()
    else:assert not (root/'MODEL_EXECUTION_STARTED282.json').exists()
    brokers=[]
    try:
        if config.get('native_fit_gate_required') and not resume:
            gate=root/'NATIVE_FEEDBACK_GATE282.json'
            if gate.exists():
                assert json.loads(gate.read_text())['passed'], 'A failed native fitting gate needs explicit recovery'
            else:
                arguments=['--run',root]
                if config.get('search_mode')=='ensemble_weights':
                    arguments+=['--ensemble-bank',root/'bank/probabilities.npz','--gate-file','NATIVE_FEEDBACK_GATE282.json']
                    if config.get('search_stage')=='review':
                        arguments+=['--ensemble-stage','review','--prior-pool',root/config['cases'][0]['id']/'references/prior_pool.json']
                command(root,'qualify_feedback_native.py',*arguments)
            for c in config['cases']:
                shutil.copy2(root/'NATIVE_FEEDBACK_GATE282.json',root/c['id']/'NATIVE_FEEDBACK_GATE282.json')
        for c in config['cases']:
            run=root/c['id']
            with (run/'logs/broker.log').open('ab') as log:
                broker='controlled_ensemble_feedback.py' if config.get('search_mode')=='ensemble_weights' else 'controlled_feedback.py'
                p=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B',str(PROJECT/'scripts'/broker),'serve','--run',str(run)],
                    env=numerical_env(run),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            brokers.append(p)
            # Ordinary review rechecks hundreds of real inherited candidates.
            # This is pre-subject numerical import,not the Agent time budget.
            ready_deadline=time.monotonic()+120
            while time.monotonic()<ready_deadline:
                if (run/'BROKER_READY.json').exists():break
                assert p.poll() is None,'Broker startup failed'
                time.sleep(.05)
            else:raise TimeoutError('Broker not ready')
        if not resume:
            qualify(root,root/config['cases'][0]['id'])
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(qualify,root,root/c['id']) for c in config['cases'][1:]]
                for f in futures:f.result()
            save(root/'MODEL_EXECUTION_STARTED282.json',dict(utc=utc(),max_sessions=len(config['cases']),final_scores_unopened=True))
        execute_wave(root,'feedback_timing',[root/c['id'] for c in config['cases']],reuse_pre_admission_qualification=True)
        state(root,status='all_sessions_terminal_waiting_choice_audit',final_scores_opened=False)
    except BaseException as e:
        state(root,status='execution_error_inspect_no_automatic_retry',error=repr(e));raise
    finally:
        for p in brokers:
            if p.poll() is None:p.terminate()
        for p in brokers:p.wait(timeout=30)


def run_with_review(root):
    from evaluate_ensemble_feedback import seal_phase,prepare_review,seal_all
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert config['search_mode']=='ensemble_weights' and config.get('search_stage','search')=='search'
    execute(root)
    seal_phase(root)
    review=prepare_review(root)
    execute(review)
    seal_phase(review)
    seal_all(root)
    state(root,status='initial_and_ordinary_review_sealed_waiting_external_evaluation',final_scores_opened=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--mode',choices=['prepare','run','run-with-review','resume-before-model'],required=True)
    p.add_argument('--spec',type=Path)
    p.add_argument('--ensemble-source',type=Path);p.add_argument('--iteration');p.add_argument('--result-directory')
    a=p.parse_args()
    if a.mode=='prepare' and a.ensemble_source:
        from prepare_feedback_campaign import prepare_bound_source,write_ensemble_spec
        assert a.iteration and a.result_directory and not a.spec
        root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
        spec=root/'SPEC.json';write_ensemble_spec(a.ensemble_source.resolve(),spec,a.iteration,a.result_directory)
        prepare_bound_source(root,spec)
    elif a.mode=='prepare' and a.spec:
        from prepare_feedback_campaign import prepare_bound_source
        prepare_bound_source(a.run.resolve(),a.spec.resolve())
    elif a.mode=='prepare':prepare(a.run.resolve())
    elif a.mode=='run-with-review':run_with_review(a.run.resolve())
    else:execute(a.run.resolve(),resume=a.mode=='resume-before-model')
