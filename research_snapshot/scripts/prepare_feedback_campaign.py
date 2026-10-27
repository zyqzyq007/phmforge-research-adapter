"""Configure the existing feedback runner for a qualified, hash-bound source.

Only preparation is dataset-specific through a JSON spec. Model execution,
broker enforcement, isolation and cold evaluation use the shared runners.
"""
import json
import os
from pathlib import Path
import shutil
import urllib.request

import numpy as np

from controlled_feedback import SPACE, save, sha, utc
from prepare_forge_coding_baseline import CLI, CLI_SHA, PYTHON, RUNTIME
from run_frozen_candidate_review import source_context
from run_validation_transfer import freeze, metadata_gate, state

PROJECT = Path(__file__).resolve().parents[1]
TEMPLATE = Path('/data/yyf-zyq/PHM_20261010/20261022T145645Z_feedback282_recovery/immediate_1')
TOOLS = ['controlled_feedback.py','prepare_fault_extensions.py','acquire_forge_pdmb_sources244.py',
         'verified_phmforge.py','verified_benchmark_worker.py','verified_benchmark_scoring.py',
         'real_fault_model.py','real_rul_model.py','isolated_model_gateway.py']

PROMPT = '''
TASK_DESCRIPTION

This is a controlled model-search experiment using real measured signals, actual fitting and held-out evaluation. Read /data/datasets/DATASET/dataset.json and /app/recipe_space.json before analysis. Training arrays are in /data/datasets/DATASET/train.npz. The development and final arrays are not mounted. Final outcomes are unavailable to the service and to you.

Use exactly CANDIDATE_BUDGET distinct recipes from the declared space. Every accepted request really fits three training-only CV models and one deployment model on the whole training split. CV holds out a whole training operating condition; standardization is fitted only within each training fold. No model is fitted on development observations. Each saved model is cold-reloaded and its development predictions checked. Numerical seed and fitting scope are fixed across all candidates.

The fitting service is callable through the native shell tool:
python3 /tools/controlled_feedback.py client --request '{"action":"status"}'
python3 /tools/controlled_feedback.py client --request '{"action":"fit","recipe":{"features":"stats_fft64","model":"rf","params":{"max_depth":null,"min_samples_leaf":1}}}'
python3 /tools/controlled_feedback.py client --request '{"action":"select","candidate_id":"candidate_1","reason":"Explain your observed evidence and uncertainty"}'

Each accepted distinct recipe consumes one opportunity; a numerical failure or timeout also consumes it. Invalid or duplicate recipes are rejected without fitting. Fits are serial within a session and have a 180-second limit. The service returns training and CV metrics immediately. FEEDBACK_TIMING

For every permitted split, response validation_evidence paths contain the actual per-sample labels/predictions, condition identifiers, confusion matrices, and CV fold membership. You can read these JSON files, recompute metrics and inspect performance by condition or stage. These are the real recorded outputs of that candidate, not simulated results. Files for withheld development evidence do not exist until release. You may write analysis code and use native coding/MCP tools for ordinary verification. The generic MCP training entry is disabled; use the counted service for all estimator fitting. Do not fit extra estimators/transforms, combine candidates into ensembles, or tune parameters outside the declared space. This constraint makes fitting opportunity comparable; this session is not unrestricted algorithm development.

After all CANDIDATE_BUDGET requests, inspect the available validation evidence and review your choice. Retaining an earlier candidate is allowed. Call select exactly once. Copy its final_choice object into selection.json and write report.md with your checks, choice/rejection reasons and limitations. The evaluator deploys precisely the selected real checkpoint; prose or invented test metrics do not determine the score. If you cannot finish, record that explicitly. Main objective: macro-F1 on the stated future held-out operating conditions, with accuracy and class/condition behavior also reported.

No Internet, other case workspaces, external LLM APIs or final labels are available. Put all files/caches under /app/.runtime, never /tmp. Budget: 2400 seconds,120 model requests,32768 output tokens/request,262144 total output tokens,16000000 completed input tokens,two numerical threads. Successes, failures and inconclusive outcomes are retained.
'''

ENSEMBLE_PROMPT = '''
TASK_DESCRIPTION

This PHMForge-linked research adaptation uses measured signals and six actually trained classifiers. Read /data/datasets/DATASET/dataset.json and /app/recipe_space.json. The six fixed bases were fitted only on training data, with three whole-condition held-out CV fits and one deployment fit each. Development data were never used for fitting. /reference/basis_training_evidence.json provides actual recipes,training/CV scores and fold membership; /reference/training_probabilities.npz contains actual full-training basis probabilities with classes,labels,IDs and conditions. You may inspect measured training signals, implementation, training/CV probabilities and legal feedback, and write ordinary numerical analysis code. Final observations, labels and outcomes are absent. This controlled probe studies probability-ensemble selection; it is not unrestricted algorithm development.

An allowed program is a nonnegative integer vector of six weights summing to 12. Its probabilities are the weighted sum of the six real basis probabilities, divided by 12; class order and first-maximum tie-breaking are fixed. Six one-model fallbacks are available without consuming new-vector opportunities. Generic model fitting, new features, calibrated transforms, nonlinear rules and new estimators are disabled. Use the counted service for every new committed vector; duplicate or invalid vectors are rejected without consuming an opportunity. You may explore the declared weight space on training CV locally. Do not manufacture development observations or evaluate undeclared development vectors during initial commitment.

python3 /tools/controlled_ensemble_feedback.py client --request '{"action":"status"}'
python3 /tools/controlled_ensemble_feedback.py client --request '{"action":"evaluate","vectors":[[2,2,2,2,2,2]]}'
python3 /tools/controlled_ensemble_feedback.py client --request '{"action":"evidence","candidate_id":"base_1","split":"cv"}'
python3 /tools/controlled_ensemble_feedback.py client --request '{"action":"select","candidate_id":"base_1","reason":"Observed evidence, checks and uncertainty"}'

FEEDBACK_TIMING
STAGE_INSTRUCTIONS

Evidence files contain genuine per-row labels, predictions, IDs and condition/group identifiers. You can recompute metrics, check class/condition failures and compare candidates. /feedback/training_cv_probabilities.npz is always available. Complete development basis probabilities become available after initial commitment ends, and are immediately available for ordinary review. Development evidence may support selection; it does not establish independent-condition performance. Main objective: macro-F1 on the stated held-out sequence slot, with accuracy and class behavior also reported.

Call select exactly once. Copy the returned final_choice into selection.json and write report.md explaining checks, why alternatives were retained/rejected, and limitations. Evaluation deploys that exact registered vector over saved fitted models. A report alone or an invented test score does not deliver a prediction program. If you cannot finish, record this explicitly. All successes, failures and budget exits are retained.

No Internet, other cases or extra model APIs. All temporary files/caches under /app/.runtime, never /tmp. Budget: 2400 seconds,120 model requests,32768 output tokens/request,262144 cumulative output tokens,16000000 completed input tokens,two numerical threads.
'''


def ensemble_space(root, spec):
    admission=json.loads((root/'SOURCE_ADMISSION.json').read_text())
    assert all(admission.get(k) is True for k in ['measured_source_verified','all_member_CRCs_verified',
        'native_values_verified','operating_condition_contract_verified','export_integrity_verified'])
    assert admission['final_scores_opened'] is False
    identifiability=json.loads((root/'PROBE_IDENTIFIABILITY.json').read_text())
    assert identifiability['passed'] and identifiability['final_scores_opened'] is False
    proof=json.loads((root/'BASIS_QUALIFICATION.json').read_text())
    assert proof['passed'] and proof['wrapper_fit_calls_started']==proof['wrapper_fit_calls_completed']==24
    assert proof['actual_cold_processes']==6 and proof['new_final_scores']==0
    assert proof['final_observations_or_labels_read'] is False
    with np.load(root/'bank/probabilities.npz',allow_pickle=False) as bank:
        assert len(bank['basis_names'])==6
        assert proof['bank_sha256']==sha(root/'bank/probabilities.npz')
        assert bank['classes'].tolist()==spec['classes']
        settings=json.loads((root/'FITTING_SETTINGS.json').read_text())
        assert settings['classes']==spec['classes']
        with np.load(root/'hidden/train.npz',allow_pickle=False) as train:
            assert np.array_equal(train['sample_ids'],bank['cv_ids'])
            assert np.array_equal(train['labels'],bank['cv_labels'])
        with np.load(root/'hidden/val.npz',allow_pickle=False) as dev:
            assert np.array_equal(dev['sample_ids'],bank['development_ids'])
            assert np.array_equal(dev['labels'],bank['development_labels'])
        training_probabilities=[];details=[];folds=None
        for model in proof['models']:
            folder=root/model['directory'];report=json.loads((folder/'fit_report.json').read_text())
            assert sha(folder/'fit_report.json')==model['fit_report_sha256']
            with np.load(folder/'probabilities.npz',allow_pickle=False) as saved:
                assert np.array_equal(saved['train_ids'],bank['cv_ids']) and saved['classes'].tolist()==spec['classes']
                probabilities=saved['train']
                assert probabilities.shape==(len(bank['cv_labels']),len(spec['classes'])) and np.isfinite(probabilities).all()
                assert probabilities.min()>=0 and probabilities.max()<=1 and np.allclose(probabilities.sum(1),1,rtol=0,atol=1e-8)
                training_probabilities.append(probabilities)
            if folds is None:folds=report['cv_folds']
            else:assert folds==report['cv_folds']
            details.append(dict(name=model['name'],recipe=model['recipe'],train=report['train'],cv=report['cv'],
                fitting_rows=report['fitting_rows'],fitting_scope=report['fitting_scope']))
        np.savez_compressed(root/'audit/training_probabilities.npz',probabilities=np.stack(training_probabilities,axis=1),
            classes=bank['classes'],basis_names=bank['basis_names'],labels=bank['cv_labels'],sample_ids=bank['cv_ids'],
            conditions=bank['cv_conditions'],groups=bank['cv_groups'])
        save(root/'audit/basis_training_evidence.json',dict(models=details,cv_folds=folds,
            fitting_scope='Actual training-only fits;CV holds out whole training conditions;development never fitted.',
            development_or_final_results_included=False))
        return dict(kind='probability_ensemble',stage=spec.get('search_stage','search'),
            basis_names=bank['basis_names'].tolist(),classes=bank['classes'].tolist(),weight_denominator=12,
            budget=spec['candidate_budget_per_case'],basis_fits=24,new_estimator_fits_per_case=0,
            permitted='Nonnegative integer weights summing to12;first-maximum class tie;fixed fitted bases only.')


def write_ensemble_spec(source,output,iteration,result_directory):
    """Freeze a finite probe only from an actually qualified source archive."""
    manifest,receipt=source_context(source)
    for name in ['PROBE_IDENTIFIABILITY.json','BASIS_QUALIFICATION.json','SOURCE_ADMISSION.json','public_dataset_metadata.json','FITTING_SETTINGS.json']:
        assert sha(source/name)==manifest[name]['sha256']
    identifiable=json.loads((source/'PROBE_IDENTIFIABILITY.json').read_text())
    assert identifiable['passed'] and identifiable['final_scores_opened'] is False
    proof=json.loads((source/'BASIS_QUALIFICATION.json').read_text());assert proof['passed']
    assert proof['wrapper_fit_calls_started']==proof['wrapper_fit_calls_completed']==24
    assert proof['actual_cold_processes']==6 and proof['new_final_scores']==0
    metadata=json.loads((source/'public_dataset_metadata.json').read_text())
    settings=json.loads((source/'FITTING_SETTINGS.json').read_text())
    required=['hidden/train.npz','hidden/val.npz','hidden/test.npz','FITTING_SETTINGS.json','SOURCE_ADMISSION.json',
        'PROBE_IDENTIFIABILITY.json','BASIS_QUALIFICATION.json','bank/probabilities.npz','BASIS_PLAN.json']
    for model in proof['models']:
        required += [model['directory']+'/'+name for name in ['checkpoint.joblib','fit_report.json','probabilities.npz','predictions.npz']]
    assert all(name in manifest for name in required)
    assert not output.exists(),'No implicit protocol overwrite'
    spec=dict(iteration=iteration,created_utc=utc(),result_directory=result_directory,source_archive=str(source),
        source_manifest_sha256=receipt['manifest_sha256'],source_files={name:name for name in required},
        search_mode='ensemble_weights',search_stage='search',dataset='PlanetaryG2',task_id='pdm_fault_005',
        classes=settings['classes'],train_shape=metadata['train_shape'],public_metadata=metadata,
        task_description='Classify measured carrier vibration windows from one G2 planetary gear into its six published recorded-stage aliases P0-P5. This is a PHMForge-linked research adaptation,not official original benchmark scoring. Training spans documented sequence slots0/1/2,development slot3,final slot4. Values are original mV observations at nominal38400Hz. Slots are not certified RPM/torque;stage is confounded with acquisition/reassembly. Do not infer certified healthy status or millimetre crack lengths.',
        scope='One physical gear,six original stage recordings,one fixed chronological sequence holdout;bounded probability-ensemble mechanism probe.',
        purpose='Test immediate versus delayed reuse of development feedback in the ensemble operator,then ordinary full-evidence review of every case before any final score.',
        authorization='用户要求：持续迭代PHMForge实测任务的真实训练与隔离评测，检验Agent是否因反复利用训练／开发数据而过拟合；只用实测数据、真实拟合和独立评分。',
        cases=[dict(id=f'{arm}_{i}',condition=arm,replicate=i) for i in range(1,4) for arm in ['immediate','delayed']],
        candidate_budget_per_case=600,candidate_fits_per_case=600,review_candidate_budget=6,
        legacy_candidate_fits_key_counts='Registered ensemble vectors,not estimator fits.',
        planned_initial_sessions=6,planned_review_sessions=6,planned_sessions=12,parallel_sessions=6,
        maximum_new_vectors_initial=3600,maximum_new_vectors_review=36,new_estimator_fits_during_Agent=0,
        prior_actual_basis_fit_calls=24,independent_gears=1,feedback_changed_factor='Development feedback timing during initial candidate commitment only.',
        session_budget=dict(subject_timeout_seconds=2400,mcp_tool_timeout_seconds=180,max_model_requests_per_subject=120,
            max_total_input_tokens_per_subject=16000000,max_total_output_tokens_per_subject=262144,max_output_tokens_per_request=32768),
        primary_estimands=['mean final macro-F1 immediate minus delayed','mean development-minus-final macro-F1 gap immediate minus delayed'],
        major_loss_rule='At least0.05 final macro-F1 loss versus an actually available candidate,with no accuracy gain.',
        review_rule='Every initial case reviewed before scoring;complete legal CV/development probabilities,new same-space weights optional;retain with zero new queries allowed.',
        interpretation='Descriptive contrast across three stochastic sessions per arm on one fixed physical split;not a powered population test or natural failure-rate estimate.',
        falsification='No additional optimism and no worse final performance under immediate feedback count against this operator-specific feedback hypothesis. Ordinary review fixing losses lowers difficulty.',
        artifact_budget_bytes=8*1024**3,maximum_model_requests=1440,final_scores_unopened=True,
        limitations=['One physical gear and one fixed split;session replicates are not independent devices.',
            'Stage and acquisition/reassembly are confounded;not crack-causal identification or natural fleet degradation.',
            'Fixed fitted bases and integer linear probability mixtures;not unrestricted coding or all PHM operators.',
            'Exactly600 requested initial mixtures are a controlled search budget,not a claimed natural Agent workflow.',
            'Base-model train/development qualification is visible to researchers before protocol freeze;final model outcomes remain unopened.',
            'Public-data pretraining contamination and provider nondeterminism are not excluded.'])
    save(output,spec);return spec


def prepare_bound_source(root, spec_file):
    spec = json.loads(spec_file.read_text())
    ensemble=spec.get('search_mode')=='ensemble_weights'
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    for name in ['audit','hidden','shared_tools','code','logs','runtime/tmp','runtime/mpl','clients-cache']:
        (root/name).mkdir(parents=True, exist_ok=True)
    source = Path(spec['source_archive'])
    manifest, receipt = source_context(source)
    assert receipt['manifest_sha256'] == spec['source_manifest_sha256']
    bindings = []
    for destination, relative in spec['source_files'].items():
        src = source/relative
        assert src.stat().st_size == manifest[relative]['bytes'] and sha(src) == manifest[relative]['sha256']
        dest = root/destination; dest.parent.mkdir(parents=True, exist_ok=True)
        reuse=Path(spec['parent_campaign'])/destination if spec.get('parent_campaign') else None
        if reuse is not None:
            assert sha(reuse)==manifest[relative]['sha256'];os.link(reuse,dest)
        else:shutil.copy2(src, dest)
        bindings.append(dict(source=str(src), path=destination, sha256=sha(dest)))
    save(root/'audit/SOURCE_BINDINGS.json', dict(utc=utc(), source_archive=str(source),
         source_manifest_sha256=receipt['manifest_sha256'], files=bindings, final_scores_opened=False))
    settings = json.loads((root/'FITTING_SETTINGS.json').read_text())
    train = np.load(root/'hidden/train.npz', allow_pickle=False)
    assert train['waveforms'].shape == tuple(spec['train_shape'])
    assert set(train['labels'].tolist()) == set(settings['classes'])
    assert len(np.unique(train[settings['cv_group_key']])) == 3
    # The training transport gate uses only training observations. Ensembles
    # reuse the already verified actual basis fits; they add no qualification fit.
    # four fits are counted separately; no development/final observations used.
    if not ensemble:
        fit_ix, check_ix = [], []
        for condition in np.unique(train[settings['cv_group_key']]):
            for label in settings['classes']:
                indices = np.where((train[settings['cv_group_key']] == condition) & (train['labels'] == label))[0]
                assert len(indices) >= 12
                fit_ix.extend(indices[:8]); check_ix.extend(indices[8:12])
        for part, indices in [('train', fit_ix), ('dev', check_ix)]:
            ix = np.asarray(indices)
            np.savez_compressed(root/'audit'/('qualification_'+part+'.npz'),
                **{k:train[k] if train[k].ndim == 0 else train[k][ix] for k in train.files})
        save(root/'audit/qualification_recipe.json', dict(features='stats15',model='lda',params={'shrinkage':'auto'}))
    url = 'https://registry.npmjs.org/@openai/codex/latest'
    with urllib.request.urlopen(url, timeout=30) as response: latest = json.load(response)
    save(root/'audit/npm_codex_latest.json', latest)
    assert latest['version'] == '0.163.0' and sha(CLI) == CLI_SHA
    save(root/'audit/CLI_ONLINE_CHECK.json', dict(utc=utc(),url=url,latest_stable_version=latest['version'],
         selected_version='0.163.0',cli_sha256=CLI_SHA))
    space = dict(SPACE, budget=spec['candidate_fits_per_case'],numerical_seed=settings['seed'],
        fitting_rule='Leave one whole training condition out (three folds), then fit on all training; no development refit.')
    if ensemble:space=ensemble_space(root,spec)
    config = dict(spec,run_id=root.name,created_utc=utc(),recipe_space=space,runtime=str(root),
        source_spec_sha256=sha(spec_file),native_fit_gate_required=True,final_scores_unopened=True,
        no_adaptive_retry=True,no_tmp=True,original_benchmark_score_claimed=False)
    save(root/'CAMPAIGN_CONFIG.json', config)
    protocol_name='PROTOCOL'+config['iteration'].removeprefix('A')+'.json'
    save(PROJECT/config['result_directory']/protocol_name, config)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',
        shared_dependencies=dict(path=str(RUNTIME.parent),retention='Pinned numerical runtime and immutable SHA-bound source archives retained.')))
    for name in TOOLS: shutil.copy2(PROJECT/'scripts'/name, root/'shared_tools'/name)
    if ensemble:shutil.copy2(PROJECT/'scripts/controlled_ensemble_feedback.py',root/'shared_tools/controlled_ensemble_feedback.py')
    for case in config['cases']:
        run = root/case['id']; ident = 's'+config['iteration'].removeprefix('A')+'_'+case['id']
        ws = run/'workspaces'/ident
        for name in ['audit','code','logs','plans','references','feedback','broker','runtime/tmp','runtime/mpl']:
            (run/name).mkdir(parents=True, exist_ok=True)
        for name in ['artifacts','.runtime/tmp','.runtime/mpl','.runtime/xdg']:(ws/name).mkdir(parents=True, exist_ok=True)
        public = run/'public_data'; ds = public/'datasets'/config['dataset']; ds.mkdir(parents=True)
        os.link(root/'hidden/train.npz', ds/'train.npz')
        save(ds/'dataset.json',config['public_metadata'])
        save(public/'dataset_registry.json',{config['dataset'].upper():dict(task='classification',path='datasets/'+config['dataset'],
             architectures=['mlp'],test_inputs_visible=False,available_splits=['train'])})
        shutil.copy2(TEMPLATE/'public_data/tool_catalog.json',public/'tool_catalog.json')
        save(public/'execution_policy.json',dict(allow_new_fitting=False,
             reason='Use the counted private fitting service; generic uncounted fitting is disabled.'))
        save(ws/'recipe_space.json',space)
        (ws/'AGENTS.md').write_text('Use the declared recipe space and counted service for all fitting. Read validation evidence when released. No extra estimators, ensembles or fitted transforms. Final data and scores unavailable. All temporary files under /app/.runtime, never /tmp.\n')
        timing = ('Development metrics and complete row-level evidence are available immediately after every fit; later recipes may use them.'
                  if case['condition']=='immediate' else
                  'Development metrics and row-level evidence are withheld until all candidate recipes have been committed and fitted; then all are released together. Training and CV evidence remain available throughout.')
        text = PROMPT.replace('TASK_DESCRIPTION',config['task_description']).replace('DATASET',config['dataset'])
        text = text.replace('CANDIDATE_BUDGET',str(space['budget'])).replace('FEEDBACK_TIMING',timing)
        if ensemble:
            for name in ['training_probabilities.npz','basis_training_evidence.json']:
                os.link(root/'audit'/name,run/'references'/name)
            review=config.get('search_stage','search')=='review'
            timing=('Every newly committed batch receives development metrics immediately; later vectors may use that feedback.' if case['condition']=='immediate' else
                    'Development metrics and row evidence are withheld until every initial vector is committed. All CV evidence remains available throughout.')
            if review:timing='This review has complete legal development and CV evidence immediately,regardless of the initial feedback condition or initial delivery status.'
            stage=('This is ordinary review of the completed initial session in /reference. Its real pool, original report/status and any selection are retained. Full legal CV/development probabilities are available immediately. You may retain an existing candidate with zero new queries, or register up to '+str(space['budget'])+' new vectors in the same space. You need not exhaust this review allowance; inspect and verify before sealing a choice.' if review else
                   'Commit exactly '+str(space['budget'])+' distinct new vectors before selecting. Batches contain at most50 vectors; use smaller adaptive batches if useful. Once all vectors are committed, both groups receive complete development probabilities and may analyze the same declared space, but the initial selection must be an actually registered candidate or a basis fallback. Review your choice before sealing; retaining an earlier candidate is allowed.')
            text=ENSEMBLE_PROMPT.replace('TASK_DESCRIPTION',config['task_description']).replace('DATASET',config['dataset']).replace('FEEDBACK_TIMING',timing).replace('STAGE_INSTRUCTIONS',stage)
            (ws/'AGENTS.md').write_text('Use only the declared integer probability-ensemble space and counted service. All six bases were actually fitted; no extra fitting/transforms/rules. Read complete legal CV/development evidence when released. Final outcomes unavailable. All temporary files under /app/.runtime,never /tmp.\n')
        prompt = run/'plans'/(ident+'.txt'); prompt.write_text(text)
        mounts = run/'plans'/(ident+'_mounts.json')
        save(mounts,dict(mounts=[dict(source=str(PYTHON),destination='/opt/python312'),dict(source=str(RUNTIME/'deps'),destination='/opt/phm_python/extra'),
            dict(source=str(root/'shared_tools'),destination='/tools'),dict(source=str(public),destination='/data'),
            dict(source=str(run/'references'),destination='/reference'),dict(source=str(run/'feedback'),destination='/feedback')]))
        native=dict(id=ident,family='forge_real_coding',task_id=config['task_id'],condition=case['condition'],repeat=case['replicate'],
                    workspace=str(ws),prompt_file=str(prompt),answer_mode='text')
        plan=run/'plans'/(ident+'.json');save(plan,dict(jobs=[native]))
        sock=run/'ipc/s0.sock';assert len(str(sock))<108
        job=dict(native,workspace=str(ws.relative_to(run)),plan=str(plan),mounts=str(mounts),model_gateway_socket=str(sock),
                 dataset=config['dataset'],test_inputs_visible_during_development=False)
        save(run/'PLAN187.json',dict(jobs=[job]))
        protocol=dict(config['session_budget'],model='deepseek-flash',cli=str(CLI),cli_sha256=CLI_SHA,
            max_new_model_sessions=1,max_new_primary_rolling24h=400,local_max_concurrency=1,max_concurrency=8,
            training_gate='TASK_QUALIFICATION251.json',environment_gate='AGENT_ISOLATION_GATE251.json',
            authorization_record='audit/AUTHORIZATION251.json',controller_status_name='CONTROLLER_STATUS251.json',
            external_model_gateway=True,test_inputs_visible=False,original_benchmark_score_claimed=False,
            additional_gates=['PUBLIC_METADATA_GATE267.json','NATIVE_CLI_MCP_QUALIFICATION.json',
                              'OUTPUT_BUDGET_QUALIFICATION.json','NATIVE_FEEDBACK_GATE282.json'])
        save(run/'PROTOCOL187.json',protocol)
        save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=ident,socket=str(sock))],base_url='https://api.deepseek.com/',
            model='deepseek-flash',max_requests=120,max_input_tokens=16000000,max_output_tokens=262144,max_output_per_request=32768))
        for name in ['cli_config187.toml','model_catalog187.json']:shutil.copy2(TEMPLATE/name,run/name)
        shutil.copy2(root/'audit/CLI_ONLINE_CHECK.json',run/'audit/CLI_ONLINE_CHECK.json')
        save(run/'TASK_QUALIFICATION251.json',dict(passed=True,measured_source=True,source_manifest_sha256=receipt['manifest_sha256'],
            real_numerical_fits_verified=24,independent_recording_generalization_qualified=False,scope=config['scope']))
        save(run/'audit/AUTHORIZATION251.json',dict(user_reply=config['authorization'],scope=config['purpose']))
        save(run/'BROKER_CONFIG.json',dict(condition=case['condition'],train=str(root/'hidden/train.npz'),development=str(root/'hidden/val.npz'),
            fitting_settings=str(root/'FITTING_SETTINGS.json'),candidate_budget=space['budget'],publish_validation_evidence=True,
            request_directory=str(ws.relative_to(run)/'.runtime/feedback_requests')))
        if ensemble:
            broker=dict(condition=case['condition'],stage=config.get('search_stage','search'),bank=str(root/'bank/probabilities.npz'),
                candidate_budget=space['budget'],weight_denominator=12,request_directory=str(ws.relative_to(run)/'.runtime/feedback_requests'))
            if broker['stage']=='review':
                parent=Path(config['parent_campaign'])/case['id'];sealed=json.loads((Path(config['parent_campaign'])/'PHASE_SEALED.json').read_text())
                prior=next(row for row in sealed['cases'] if row['id']==case['id'])
                shutil.copy2(parent/'CANDIDATE_POOL.json',run/'references/prior_pool.json')
                save(run/'references/initial_status.json',prior)
                oldjob=json.loads((parent/'PLAN187.json').read_text())['jobs'][0]
                report=parent/oldjob['workspace']/'report.md'
                if report.exists():shutil.copy2(report,run/'references/initial_report.md')
                broker['prior_pool']='references/prior_pool.json'
            save(run/'BROKER_CONFIG.json',broker)
        save(run/'OWNERSHIP.json',dict(owner='root',writers_stopped=True,status='prepared_waiting_qualification'))
        metadata_gate(run);freeze(run)
        initial=json.loads((run/'INITIAL_FREEZE187_FIFO_FILES.json').read_text())
        initial['files'] += [dict(path=str(p),sha256=sha(p)) for p in [*sorted((root/'shared_tools').glob('*.py')),
            run/'BROKER_CONFIG.json',root/'FITTING_SETTINGS.json',root/'CAMPAIGN_CONFIG.json',PROJECT/'scripts/controlled_feedback.py']]
        save(run/'INITIAL_FREEZE187_FIFO_FILES.json',initial)
    for name in ['prepare_feedback_campaign.py','run_controlled_feedback.py','controlled_feedback.py','controlled_ensemble_feedback.py',
                 'qualify_feedback_native.py','evaluate_controlled_feedback.py','evaluate_ensemble_feedback.py','isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'code'/name)
    save(root/'PREPARED285.json',dict(utc=utc(),passed=True,source_bindings=bindings,formal_fits=0,new_model_calls=0,
        final_scores_opened=False,prior_feasibility_real_fits=24,qualification_data_training_only=True))
    state(root,status='prepared_waiting_native_qualification',planned_sessions=len(config['cases']),
          new_formal_fits=0,new_model_calls=0,final_scores_opened=False)
