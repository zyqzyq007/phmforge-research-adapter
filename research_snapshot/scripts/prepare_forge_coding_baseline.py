"""Freeze source-qualified PHMForge tasks for native coding Agent execution.

This reuses the established native runner, numerical backends, and scorer.
Original task text is retained; independent final-label evaluation is an
explicit research adaptation of PHMForge's self-verification workflow.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from acquire_forge_pdmb_sources244 import save, sha256


TASKS=[('pdm_fault_001','CWRU'),('pdm_fault_013','MFPT'),
       ('pdm_fault_015','ELECTRICMOTORVIBRATIONS')]
RUNTIME=Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4')
CLI=Path('/data/yyf-zyq/PHM_20261010/20261015T215024Z_independent95/runtime/codex0163/package_files/package/vendor/x86_64-unknown-linux-musl/bin/codex')
CLI_SHA='399ff69c431c066e40de318fba45c2b07f45a76b19939d247ee4828fc70c4215'
PYTHON=Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')


def prepare(run, project, task_specs=None):
    if (run/'PLAN187.json').exists():
        raise FileExistsError('No overwrite of prepared task cohort')
    for name in ['audit','source','code','logs','plans','workspaces','outputs',
                 'references','hidden','runtime/tmp','runtime/mpl']:
        (run/name).mkdir(parents=True,exist_ok=True)
    online=json.loads((run/'audit/CLI_ONLINE_CHECK.json').read_text())
    cli=Path(online.get('CLI_path',str(CLI)))
    cli_sha=online.get('CLI_sha256',CLI_SHA)
    assert sha256(cli)==cli_sha
    if cli != CLI:
        assert online['CLI_version_output']=='codex-cli '+online['latest_stable_version']
        assert online['qualified_local_binary_version']
    else:
        assert online['latest_stable_version']=='0.163.0'
    # Read the original task JSON from the already pinned local reconstruction.
    from prepare_verified_benchmarks import extract_bound
    archive=Path('/data/yyf-zyq/PHM_20261010/20261020T133326Z_forgesources244')
    prefix='source/official/ReActXen/src/reactxen/demo/intent_implementation_demo/scenarios/phm_scenarios.json'
    bindings=extract_bound(archive,run/'original',[prefix])
    scenarios=json.loads((run/'original'/prefix).read_text())['pdm_scenarios']
    originals={row['task_id']:row for row in scenarios}
    tools=run/'public_tools';tools.mkdir()
    names=['verified_phmforge.py','verified_benchmark_worker.py','verified_benchmark_scoring.py',
           'real_fault_model.py','real_rul_model.py','isolated_model_gateway.py','frozen_classifier.py']
    for name in names:shutil.copy2(project/'scripts'/name,tools/name)
    protocol=json.loads((run/'BOOTSTRAP_PROTOCOL.json').read_text())
    sealed_test=protocol.get('test_inputs_visible',True) is False
    common='''
Complete the original PHMForge request above using the supplied actual observations and genuine numerical tools.

This is a research adaptation with independent final evaluation. The original question remains the task objective. Training and development labels are available; test inputs are available but their labels and final scores are withheld. Report actual development metrics and mark final test metrics as externally pending, rather than inventing them. This added restriction differs from the original benchmark's agent-side final-ground-truth verification.

The MCP server phmforge provides actual dataset loading, model training, model inference and development verification. Its training functions fit CPU PyTorch models; their returned paths are real files. Tool implementation source is available under /tools. You may use those tools, write and execute Python, fit another suitable PHM classifier, or use the genuine development-selected reference checkpoint in /reference. You are not required to keep it or to change it. No test scores, prior Agent submissions, research conclusions or original answer JSON are provided.

Choose preprocessing, fitting and model settings from training/development information only. Preserve source-record grouping: never mix windows from one record across train and validation. Fit normalization on the fitting data. Do not use evaluation labels, other cases, Internet downloads or additional model APIs. The per-case source limitations are in /data/datasets/DATASET/dataset.json; a published recording group is not certified independent physical hardware. The data are laboratory-benchmark-derived observations, not field-maintenance outcomes.

Submit solution.py with this runnable interface:
  python3 solution.py --input PATH_TO_OBSERVATIONS_NPZ --output OUTPUT_JSON
It must load saved fitted artifacts and predict each supplied sample_id, without reading labels from the input. OUTPUT_JSON contains {"predictions":[{"sample_id":string,"predicted_label":original_source_label}, ...]}, with every input identity exactly once. Retain the original source label representation listed in dataset.json. External evaluation uses exact ID alignment, accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1 and the full confusion matrix. More accurate predictions do not fail an arbitrary upper accuracy bound.

Generate test_predictions.json for /data/datasets/DATASET/test.npz with that final program, and save validation.json and report.md explaining your actual data, training/loading, checks, model choice and remaining limitations. Include all code and checkpoints needed to reproduce your submission. Check the final program before finishing. Files under /data and /reference are read-only; all new artifacts belong under /app. You have 1200 seconds total, CPU-only computation with two numerical threads. Python 3.12, NumPy, SciPy, scikit-learn, PyTorch and MCP are installed. Use /app/.runtime for all temporary files and caches; never /tmp. Save a runnable submission before the deadline.
'''
    if sealed_test:
        replacements={
            'Training and development labels are available; test inputs are available but their labels and final scores are withheld.':
                'Training and development observations and labels are available. Final test observations, labels and scores are absent during development and review; only the external evaluator receives test observations after your program and checkpoints are frozen.',
            'Generate test_predictions.json for /data/datasets/DATASET/test.npz with that final program, and save':
                'The external evaluator will generate test_predictions.json from the withheld input after freezing. Do not fabricate that file or try to recover the test observations. Save',
        }
        for old,new in replacements.items():
            assert common.count(old)==1
            common=common.replace(old,new)
        common+='\nFor this controlled experiment, fit supervised models and saved preprocessing/normalization only on train.npz. Development labels may guide assessment and selection, but may not enter a train-plus-development refit. Final inference receives saved artifacts and observation-only input; fitting/development files are not present then. Any data-dependent inference rule must be fixed before final observations are released and its scope must be disclosed. You may write new code and train new models: the supplied candidates are available references, not a menu restriction.\n'
    jobs=[];frozen=[];gates=[];routes=[]
    specifications = task_specs or [dict(task_id=task,dataset=name) for task,name in TASKS]
    for spec in specifications:
        task,name=spec['task_id'],spec['dataset']
        root=run/'restored'/name
        metadata=next(root.rglob('dataset.json'));info=json.loads(metadata.read_text())
        source=metadata.parent
        checkpoint=next(p for p in root.rglob('model.pt') if 'untrained' not in p.name)
        audit=checkpoint.with_name('training_audit.json')
        # A249/A250 reconstructed outputs did not always retain this audit;
        # restore the exact corresponding audit from A246 when needed.
        if not audit.exists():
            prior=Path('/data/yyf-zyq/PHM_20261010/20261020T143845Z_forgerepair246')
            manifest=json.loads((prior/'SOURCE_MANIFEST.json').read_text())
            matches=[row['path'] for row in manifest['files']
                     if row['path'].startswith('outputs/'+name+'/') and row['path'].endswith('training_audit.json')]
            assert len(matches)==1
            restored=run/'reference_audits'/name
            extract_bound(prior,restored,matches)
            audit=restored/matches[0]
        training=json.loads(audit.read_text())
        assert training['optimizer_steps']>0 and training['nonzero_gradient_steps']>0
        assert training['changed_parameter_tensors']>0 and not training['test_labels_accessed']
        fit=np.load(source/'train.npz',allow_pickle=False)
        dev=np.load(source/'val.npz',allow_pickle=False)
        test=np.load(source/'test.npz',allow_pickle=False)
        assert 'labels' not in test and 'groups' not in test
        assert not set(fit['groups']) & set(dev['groups'])
        all_ids=[set(data['sample_ids'].tolist()) for data in [fit,dev,test]]
        assert all(not all_ids[a]&all_ids[b] for a,b in [(0,1),(0,2),(1,2)])
        assert all(np.isfinite(data['features']).all() for data in [fit,dev,test])
        job_id=spec.get('id','forge251_'+task)
        ws=run/'workspaces'/job_id;ws.mkdir()
        for folder in ['.runtime/tmp','.runtime/mpl','.runtime/xdg','artifacts']:(ws/folder).mkdir(parents=True,exist_ok=True)
        data_root=run/'public_data'/job_id
        dataset=data_root/'datasets'/name
        if sealed_test:
            dataset.mkdir(parents=True)
            for filename in ['train.npz','val.npz','dataset.json']:
                shutil.copy2(source/filename,dataset/filename)
        else:
            shutil.copytree(source,dataset)
        # Population label totals can reveal the withheld split's class mix
        # after subtracting training/development counts. Legacy source group
        # IDs also defeat the private group aliases. Retain both only in the
        # host-side provenance; they are not observations for this subject.
        private_keys = ['groups_per_class', 'train_validation_test_groups']
        removed = {key: info[key] for key in private_keys if key in info}
        public_info = {key: value for key, value in info.items()
                       if key not in private_keys}
        save(dataset/'dataset.json', public_info)
        save(run/'audit'/(job_id+'_REMOVED_TARGET_METADATA.json'), removed)
        # Keep only data required by this subject; perturbation qualification
        # files are not evidence about real task performance.
        (dataset/'changed_input.npz').unlink(missing_ok=True)
        for filename in ['dataset_registry.json','tool_catalog.json']:
            if filename=='dataset_registry.json':
                value={name:dict(task='classification',path='datasets/'+name,
                    architectures=info['architectures'],group_definition=info['group_definition'],
                    rows_by_split=info['rows_by_split'],data_origin='Pinned published laboratory benchmark-derived observations')}
                if sealed_test:value[name].update(test_inputs_visible=False,available_splits=['train','val'])
                save(data_root/filename,value)
            else:shutil.copy2(RUNTIME/'public'/filename,data_root/filename)
        reference=run/'references'/job_id;reference.mkdir(parents=True)
        shutil.copy2(checkpoint,reference/'model.pt');shutil.copy2(audit,reference/'training_audit.json')
        golds=[p for p in root.rglob('*.json') if p.name in ['test_labels248.json','labels.json']]
        assert len(golds)==1
        gold=run/'hidden'/name/'labels.json';gold.parent.mkdir(parents=True,exist_ok=True)
        if gold.exists():
            assert sha256(gold)==sha256(golds[0]),'Repeated dataset has different hidden labels'
        else:
            shutil.copy2(golds[0],gold)
        gold_rows=json.loads(gold.read_text())
        assert {row['sample_id'] for row in gold_rows}==all_ids[2]
        assert len({row['sample_id'] for row in gold_rows})==len(gold_rows)==len(test['sample_ids'])
        assert not (set(row['group'] for row in gold_rows)&set(fit['groups']))
        assert not (set(row['group'] for row in gold_rows)&set(dev['groups']))
        evaluation_root=None
        if sealed_test:
            evaluation_root=run/'evaluation_data'/job_id
            evaluation_dataset=evaluation_root/'datasets'/name
            evaluation_dataset.mkdir(parents=True)
            shutil.copy2(source/'test.npz',evaluation_dataset/'test.npz')
            shutil.copy2(dataset/'dataset.json',evaluation_dataset/'dataset.json')
            for filename in ['dataset_registry.json','tool_catalog.json']:
                shutil.copy2(data_root/filename,evaluation_root/filename)
            registry=json.loads((evaluation_root/'dataset_registry.json').read_text())
            registry[name].update(test_inputs_visible=True,available_splits=['test'])
            save(evaluation_root/'dataset_registry.json',registry)
        initial='''"""Genuine fitted reference candidate; may be retained or replaced."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,'/tools')
from real_fault_model import infer
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();infer(Path('/reference/model.pt'),a.input,a.output)
'''
        (ws/'solution.py').write_text(initial)
        (ws/'AGENTS.md').write_text('Use /app/.runtime for all temporary files and caches; never /tmp. Preserve read-only source data. Model fitting is allowed on training/development data. Final labels and scores are unavailable. No Internet, packages, other cases or additional model API access.\n')
        if sealed_test:
            (ws/'AGENTS.md').write_text('Use /app/.runtime for temporary files and caches; never /tmp. Fit supervised models and saved preprocessing on train.npz only. Use development labels for assessment and selection, without a final train+development refit. Test observations/labels/scores are absent until external evaluation. New coding and training are allowed. Preserve read-only source data; no Internet, packages, other cases or additional model APIs.\n')
        if protocol.get('candidate_bank_from_training_cv'):
            from prepare_classifier_bank import install
            bank=install(run,spec,source,reference,ws)
        (ws/'TASK.txt').write_text(originals[task]['input_question']+'\n')
        prompt=originals[task]['input_question']+'\n'+common.replace('DATASET',name).replace(
            'You have 1200 seconds total','You have '+str(protocol['subject_timeout_seconds'])+' seconds total')
        if protocol.get('candidate_bank_from_training_cv'):
            prompt+='\nA genuinely fitted candidate bank is available read-only at /reference/candidates/manifest.json. The starting solution uses '+bank['starting_model']+'. Its actual selection rule is: '+bank['starting_choice_rule']+'. The bank lists actual train-only checkpoints, training CV and development evidence; training recipes are available in /tools/frozen_classifier.py. The genuine neural reference remains at /reference/model.pt. Review these as useful candidates, and retain, replace or modify the solution with executable evidence.\n'
        prompt_path=run/'plans'/(job_id+'.txt');prompt_path.write_text(prompt)
        mounts=run/'plans'/(job_id+'_mounts.json')
        save(mounts,{'mounts':[
            {'source':str(PYTHON),'destination':'/opt/python312'},
            {'source':str(RUNTIME/'deps'),'destination':'/opt/phm_python/extra'},
            {'source':str(tools),'destination':'/tools'},
            {'source':str(data_root),'destination':'/data'},
            {'source':str(reference),'destination':'/reference'}]})
        socket_path=run/'ipc'/(job_id+'.sock')
        native=dict(id=job_id,family='forge_real_coding',task_id=task,condition='coding_baseline',
                    repeat=spec.get('repeat',1),workspace=str(ws),prompt_file=str(prompt_path),answer_mode='text')
        plan=run/'plans'/(job_id+'.json');save(plan,{'jobs':[native]})
        job=dict(native,workspace=str(ws.relative_to(run)),plan=str(plan),mounts=str(mounts),
                 model_gateway_socket=str(socket_path),dataset=name,
                 source_label_representation=info['class_names'])
        if evaluation_root is not None:
            evaluation_mounts=run/'plans'/(job_id+'_evaluation_mounts.json')
            values=json.loads(mounts.read_text())
            for mount in values['mounts']:
                if mount['destination']=='/data':mount['source']=str(evaluation_root)
            save(evaluation_mounts,values)
            job.update(evaluation_mounts=str(evaluation_mounts),
                evaluation_test=str(evaluation_root/'datasets'/name/'test.npz'),
                test_inputs_visible_during_development=False)
        jobs.append(job);routes.append(dict(job=job_id,socket=str(socket_path)))
        gates.append(dict(task_id=task,dataset=name,source_data_sha256=info['source_sha256'],
                          checkpoint_sha256=sha256(checkpoint),real_parameter_updates=training['optimizer_steps'],
                          train_rows=len(fit['labels']),validation_rows=len(dev['labels']),test_rows=len(test['sample_ids']),
                          labels_absent_from_test_input=True,groups_disjoint=True,identities_disjoint=True,
                          class_names=info['class_names'],group_limit=info['group_definition'],passed=True))
        for path in list(ws.rglob('*'))+list(data_root.rglob('*'))+list(reference.rglob('*'))+[plan,mounts,prompt_path]:
            if path.is_file():frozen.append(dict(path=str(path),sha256=sha256(path)))
    config='''default_permissions = "iot-offline"
model_catalog_json = "/home/jupyter-yyf-zyq/.codex/models.json"
model = "deepseek-flash"
model_provider = "isolated_deepseek"
model_reasoning_effort = "max"
web_search = "disabled"
approval_policy = "never"
[model_providers.isolated_deepseek]
name = "DeepSeek via task-scoped relay"
base_url = "http://127.0.0.1:47777/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
stream_idle_timeout_ms = 620000
[features]
apps = false
multi_agent = false
[shell_environment_policy]
exclude = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
[permissions.iot-offline.filesystem]
":root" = "write"
":tmpdir" = "write"
[permissions.iot-offline.filesystem.":workspace_roots"]
"." = "write"
[permissions.iot-offline.network]
enabled = false
[mcp_servers.phmforge]
command = "/opt/python312/bin/python3.12"
args = ["-B", "/tools/verified_benchmark_worker.py", "--family", "forge", "--data", "/data", "--output", "/app/artifacts", "--transport", "mcp"]
required = true
env = { PYTHONPATH = "/opt/phm_python/extra", PYTHONDONTWRITEBYTECODE = "1", OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "2", MKL_NUM_THREADS = "2", TMPDIR = "/app/.runtime/tmp", TMP = "/app/.runtime/tmp", TEMP = "/app/.runtime/tmp", MPLCONFIGDIR = "/app/.runtime/mpl" }
startup_timeout_sec = 60
tool_timeout_sec = 180
'''
    (run/'cli_config187.toml').write_text(config)
    protocol.update(cli=str(cli),cli_sha256=cli_sha,cli_version_online=online['latest_stable_version'],model='deepseek-flash',
                    max_new_primary_rolling24h=protocol.get('max_new_primary_rolling24h',400),local_max_concurrency=min(protocol.get('local_max_concurrency',3),len(jobs)),
                    training_gate='TASK_QUALIFICATION251.json',environment_gate='AGENT_ISOLATION_GATE251.json',
                    authorization_record='audit/AUTHORIZATION251.json',controller_status_name='CONTROLLER_STATUS251.json',
                    external_model_gateway=True,phase='prepared_not_launched')
    save(run/'PROTOCOL187.json',protocol)
    save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=routes,base_url='https://api.deepseek.com/',
         max_requests=protocol.get('max_model_requests_per_subject',40),
         max_input_tokens=protocol.get('max_total_input_tokens_per_subject',600000),
         max_output_tokens=protocol.get('max_total_output_tokens_per_subject',32768),
         max_output_per_request=protocol.get('max_output_tokens_per_request',8192),
         input_budget_limit='stop after completed usage crosses limit; next request denied',
         credential_source='/home/jupyter-yyf-zyq/.bashrc',model='deepseek-flash'))
    save(run/'PLAN187.json',{'jobs':jobs})
    save(run/'TASK_QUALIFICATION251.json',dict(passed=all(x['passed'] for x in gates),tasks=gates,
         original_benchmark_score_claimed=False,official_test_scores_visible_to_agent=False,
         prior_holdout_results_known_to_researcher=protocol.get('prior_holdout_results_known_to_researcher',True),
         claim_scope='exploratory original-task research adaptations'))
    save(run/'audit/AUTHORIZATION251.json',dict(user_reply='设立简洁清晰目标：1. 检查PHMforge对测试集合权限是否有规定；2. 对PHMforge再次检查确保尽可能真实不用模拟数据；3. 在确保真实场景中再次以coding agent作为baseline运行；4. 分析结果分析出问题',
         scope='new task-qualified primary coding sessions under the active goal; no recovery of old quota tasks',max_new_primary=len(jobs)))
    save(run/'OWNERSHIP.json',dict(owner='root',controller_pid=None,writers_stopped=True,status='prepared_waiting_isolation_gate'))
    for path in list(tools.glob('*.py'))+[run/name for name in ['PROTOCOL187.json','MODEL_GATEWAY_CONFIG.json','cli_config187.toml','model_catalog187.json','TASK_QUALIFICATION251.json']]+[project/'scripts'/name for name in ['replay_diagnostic_cli.py','run_full_benchmark_campaign.py','run_qualified_reviews.py','isolated_model_gateway.py']]:
        frozen.append(dict(path=str(path),sha256=sha256(path)))
    save(run/'INITIAL_FREEZE187_FIFO_FILES.json',{'files':frozen})
    print(json.dumps(dict(tasks=len(jobs),source_gates_passed=True,new_model_sessions=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--task-specs',type=Path)
    args=parser.parse_args();prepare(args.run.resolve(),Path(__file__).resolve().parents[1],
        json.loads(args.task_specs.read_text()) if args.task_specs else None)
