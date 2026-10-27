"""Review unchanged real fitted PHM candidates with the established native runner.

Only training/development inputs are mounted. This phase never fits estimators
or selects parameters, and it stops before any final-score comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from acquire_forge_pdmb_sources244 import save, sha256
from prepare_forge_coding_baseline import CLI, CLI_SHA, PYTHON, RUNTIME
from run_validation_transfer import execute_wave, freeze, metadata_gate, state, utc

PROJECT = Path(__file__).resolve().parents[1]

HELPER = r'''"""Run one already fitted candidate; no training or new mixtures."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--list',action='store_true')
p.add_argument('--candidate')
p.add_argument('--input',type=Path)
p.add_argument('--output',type=Path)
a=p.parse_args()
catalog=json.loads(Path('/app/candidates.json').read_text())
if a.list:
 print(json.dumps(catalog,indent=2));raise SystemExit(0)
row=next(x for x in catalog['candidates'] if x['id']==a.candidate)
assert a.input is not None and a.output is not None
source=a.input.resolve();output=a.output.resolve()
assert output.is_relative_to('/app') and not output.is_relative_to('/app/inputs')
work=Path(tempfile.mkdtemp(prefix='candidate_',dir='/app/.runtime/tmp'))
try:
 destination=work/'predictions'
 command=[sys.executable,'-B']+row['command']+['--input',str(source),'--output',str(destination)]
 result=subprocess.run(command,cwd=row.get('cwd','/app'),timeout=170)
 if result.returncode:raise RuntimeError('Candidate inference failed: '+str(result.returncode))
 if row.get('output_member'):destination=destination/row['output_member']
 value=json.loads(destination.read_text())
 output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'candidate':row['id'],'output':str(output),'predictions':len(value['predictions']),'new_fitting':False}))
finally:shutil.rmtree(work)
'''

COMMON = '''
Review the existing PHM predictor using real measured observations and genuinely fitted, unchanged checkpoints. The original PHMForge task remains the context; this phase is a bounded fixed-candidate review, not new model development.

You have public training/development observations and labels under /data. Final observations, labels, prior final scores, and other review cases are absent. The researcher has evaluated the original submissions before; you are not given those results. Do not infer that the current submission must be wrong. Retaining it is acceptable.

The immutable original program is /app/solution.py. Its original report and validation are under /app/inputs when present. /app/candidates.json lists every eligible exact candidate for this case and distinguishes complete original programs, fitted bank models, actual prior alternatives, and components of a saved model. A fitted component is not automatically a previously discarded complete program. Prior reconstructed artifacts are explicitly marked. Candidate availability differs between original trajectories; do not invent missing evidence.

Run `python3 review_candidates.py --list` to inspect the candidate catalog. To obtain actual predictions, use `python3 review_candidates.py --candidate ID --input OBSERVATION_NPZ --output /app/artifacts/FILE.json`. Create observation-only inputs by removing labels/groups from the allowed training or development file. All model files and original code are read-only. You may inspect the code and existing training/CV evidence, compute metrics, examine per-group/per-condition errors, and check whether the deployed program behaves as its report says. The MCP tools support real reading, inference and arithmetic checks; training tools explicitly deny new fitting in this phase.

Choose exactly one catalog ID. Do not refit models, train new models, alter features, tune thresholds, tune weights, construct new mixtures, or replace a candidate program. Those operations are outside this frozen review. Existing frozen transformations executed during inference remain part of the recorded candidate. You may write analysis code and run unchanged candidates on allowed observations; do not access the Internet, other model services, external research files, final data or final scores.

Submit review_selection.json containing {"candidate_id":"EXACT_ID","reason":"evidence-based explanation"}, review_validation.json with actual checks and metric scope, and review_report.md explaining the retained/rejected alternatives and remaining uncertainty. Save the choice before the deadline. A starter choice retains the original; merely leaving it untouched without a completed review is not a successful review. The external evaluator deploys the frozen candidate matching your chosen ID, not newly generated predictor code.

All new files and caches belong under /app; use /app/.runtime/tmp, never /tmp. You have 2400 seconds, 120 model requests, 32768 output tokens/request, 262144 total output tokens, 16000000 completed input tokens and two numerical threads. No LLM training occurs. Do not fabricate final test metrics or maintenance benefits.
'''


def source_context(source):
    receipt = json.loads((source/'ARCHIVE_VERIFIED.json').read_text())
    manifest = source/receipt['manifest_filename']
    assert sha256(manifest) == receipt['manifest_sha256']
    assert not receipt.get('archive_format'), 'This entry expects already extracted verified source archives'
    return {r['path']: r for r in json.loads(manifest.read_text())['files']}, receipt


def prepare(root):
    assert not (root/'PREPARED281.json').exists(), 'No implicit overwrite or task restart'
    config = json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    assert config['new_fit_budget'] == 0 and sha256(CLI) == CLI_SHA
    tools = root/'shared_tools'
    for name in ['verified_phmforge.py','verified_benchmark_worker.py','verified_benchmark_scoring.py',
                 'real_fault_model.py','real_rul_model.py','real_bearing_rul_model.py',
                 'real_xjtu_rul_model.py','frozen_classifier.py','isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name, tools/name)
    lineage, contexts = [], {}
    for case in config['cases']:
        source = Path(case['source_archive'])
        if source not in contexts: contexts[source] = source_context(source)
        manifest, receipt = contexts[source]
        old = source/case['cohort']
        oldjob = json.loads((old/'PLAN187.json').read_text())['jobs'][0]
        oldplanpath = oldjob['mounts'].split(source.name+'/',1)[1]
        oldmounts = json.loads((source/oldplanpath).read_text())['mounts']
        run = root/case['id']; run.mkdir(exist_ok=False)
        ident = 's'+str(config.get('iteration_number',281))+'_'+case['id']
        ws = run/'workspaces'/ident
        for directory in ['audit','logs','plans','references','public_data','runtime/tmp','runtime/mpl','code']:
            (run/directory).mkdir(parents=True,exist_ok=True)
        for directory in ['artifacts','inputs','.runtime/tmp','.runtime/mpl','.runtime/xdg']:
            (ws/directory).mkdir(parents=True,exist_ok=True)
        bound = []

        def copy_file(relative, destination):
            relative = str(relative)
            row = manifest[relative]
            src = source/relative
            assert src.stat().st_size == row['bytes'] and sha256(src) == row['sha256'], relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            assert not destination.exists(), str(destination)
            shutil.copy2(src,destination)
            assert sha256(destination) == row['sha256']
            bound.append(dict(source=relative,destination=str(destination.relative_to(root)),sha256=row['sha256']))

        def copy_tree(prefix, destination, include=lambda rel:True):
            prefix = str(prefix).rstrip('/')+'/'
            selected = [k for k in manifest if k.startswith(prefix) and include(k[len(prefix):])]
            assert selected, prefix
            for name in selected: copy_file(name,destination/name[len(prefix):])

        def source_relative(absolute):
            return absolute.split(source.name+'/',1)[1]

        for destination in ['/data','/reference']:
            src = next(m['source'] for m in oldmounts if m['destination']==destination)
            copy_tree(source_relative(src),run/('public_data' if destination=='/data' else 'references'))
        # Never import original final inputs just because an archived mount grew.
        assert not list((run/'public_data').rglob('test.npz'))
        save(run/'public_data/execution_policy.json',dict(allow_new_fitting=False,phase='fixed_candidate_review'))
        original_workspace = case['cohort']+'/'+oldjob['workspace']
        readonly = []
        prefix = original_workspace+'/'
        for name in sorted(k for k in manifest if k.startswith(prefix)):
            relative = name[len(prefix):]
            if relative.startswith(('.runtime/','.git/','__pycache__/')) or relative in ['AGENTS.md','TASK.txt']:
                continue
            if relative in ['report.md','validation.json','selection.json','development_predictions.json']:
                relative = 'inputs/original_'+relative
            copy_file(name,ws/relative); readonly.append(relative)
        candidates = [dict(id='original',command=['/app/solution.py'],
                           scope='Exact original submitted program and fitted checkpoints')]
        bank = json.loads((run/'references/candidates/manifest.json').read_text())
        backend = 'real_xjtu_rul_model.py' if case['family']=='rul' else 'frozen_classifier.py'
        for row in bank['models']:
            candidates.append(dict(id='bank_'+row['name'],command=['/tools/'+backend,'--model','/reference/candidates/'+row['checkpoint']],
                                   scope='Previously provided, actually fitted train-only bank model'))
        mlpbackend = 'real_xjtu_rul_model.py' if case['family']=='rul' else 'real_fault_model.py'
        # The existing backend CLI is not uniform for neural checkpoints; a tiny
        # inference-only adapter keeps the saved numerical behavior unchanged.
        reference_script = "from pathlib import Path\nimport argparse,sys\nsys.path.insert(0,'/tools')\nfrom "+mlpbackend[:-3]+" import infer\np=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');a=p.parse_args();infer('/reference/model.pt',a.input,a.output)\n"
        (run/'references/reference_infer.py').write_text(reference_script)
        candidates.append(dict(id='reference_mlp',command=['/reference/reference_infer.py'],scope='Original genuinely gradient-trained reference'))
        if case['family']=='classification':
            replay = 'candidate_replays/'+case['source_case']
            copy_file(replay+'/replay.py',run/'references/registered_replay.py')
            label = case['cohort']
            if label=='a': variants=['subset_0','subset_1','subset_2']
            elif label=='b': variants=['component_'+str(i) for i in range(5)]
            elif label=='c':
                declaration = json.loads((source/replay/'CANDIDATE_REGISTRATION276.json').read_text())
                variants = [v for v in declaration['variants'] if v.startswith('component_')]+['equal_original_recipe','optimized_original_recipe']
                copy_tree(replay+'/replay_assets',run/'references/replay_assets')
            else:
                declaration = json.loads((source/replay/'CANDIDATE_REGISTRATION276.json').read_text())
                variants = declaration['variants']
                cacheprefix = replay+'/app/.runtime/'
                names = [k for k in manifest if k.startswith(cacheprefix) and (k.endswith('_train.joblib') or k.endswith('/candidate_ensemble.joblib'))]
                assert len(names)==14
                for name in names:
                    dest = '.runtime/'+name[len(cacheprefix):]
                    copy_file(name,ws/dest);readonly.append(dest)
            for variant in variants:
                scope = ('Actual saved raw-feature alternative or recorded ensemble' if label=='d' else
                         'Existing fitted component; not automatically a prior complete submission')
                if variant.endswith('original_recipe'):scope='Previously compared recipe reconstructed and sealed before original final scoring'
                candidates.append(dict(id=variant,command=['/reference/registered_replay.py'],output_member=variant+'.json',scope=scope))
        else:
            if case['cohort']=='b':
                copy_tree('replays/b_components',run/'references/alternatives/b_components')
                copy_tree('replays/b_prior',run/'references/alternatives/b_prior')
                for component in ['tree','mlp','mlp035']:
                    candidates.append(dict(id='b_'+component,command=['/reference/alternatives/b_components/entry.py','--component',component],
                                           scope='Actual evaluated saved component, unchanged weights'))
                candidates.append(dict(id='b_prior_90_10',command=['/reference/alternatives/b_prior/solution.py'],
                    scope='Earlier actual program reconstructed before original final scoring; serialization is not the exact old checkpoint, development reproduction was verified'))
            else:
                for number in [100,200]:
                    folder='d'+str(number)
                    copy_tree('replays/'+folder,run/'references/alternatives'/folder)
                    candidates.append(dict(id=folder+'_minimum',command=['/reference/alternatives/'+folder+'/entry.py',
                        '--component','minimum','--checkpoint','/reference/alternatives/'+folder+'/checkpoint.joblib'],
                        scope='Previously fitted and compared complete alternative, unchanged weights'))
        assert len({x['id'] for x in candidates})==len(candidates)
        save(ws/'candidates.json',dict(dataset=case['dataset'],current_candidate='original',candidates=candidates))
        (ws/'review_candidates.py').write_text(HELPER)
        readonly += ['candidates.json','review_candidates.py','readonly_inputs.json']
        historical = case.get('historical_evidence',[])
        if historical:
            evidence_audit=json.loads((root/'audit/HISTORICAL_EVIDENCE284.json').read_text())
            evidence=next(x for x in evidence_audit['cases'] if x['case']==case['id'])
            index=[]
            for relative in historical:
                assert relative.startswith('.runtime/') and Path(relative).name==relative.split('/',1)[1]
                dest='historical_validation/'+Path(relative).name
                copy_file(prefix+relative,ws/dest);readonly.append(dest)
                item=next(x for x in evidence['files'] if x['relative_path']==relative)
                index.append(dict(path=Path(relative).name,sha256=item['sha256'],pre_final_recorded=True,
                    arrays=[{k:a[k] for k in ['key','shape','dtype','split','matches'] if k in a} for a in item.get('arrays',[])]))
            save(ws/'historical_validation/INDEX.json',dict(files=index,
                train_sha256=evidence['train_sha256'],development_sha256=evidence['development_sha256'],
                row_mapping='OOF arrays follow the exact train.npz order; development arrays follow val.npz order. Forward-load arrays follow the train.npz subset operating_load_Nm==the named load, without reordering.',
                limitations='Only actually retained historical files are provided. Missing CV probabilities are not fabricated. Do not execute model-fitting scripts; recompute from saved arrays and unchanged candidates.'))
            readonly.append('historical_validation/INDEX.json')
        save(ws/'readonly_inputs.json',readonly)
        save(ws/'review_selection.json',dict(candidate_id='original',reason='Starter only; review not completed'))
        (ws/'AGENTS.md').write_text('Use real existing model artifacts. Fixed-candidate review only: no new fitting or parameter search. Original artifacts are read-only. Final data/labels absent. Write analysis and selection under /app; temporary files only /app/.runtime.\n')
        detail = ('This is five-label broken-rotor-bar severity classification. Use macro-F1 as primary, and report accuracy, per-class, per-source-record and per-load results. All windows from a recording are dependent; disclosed inference normalizations may use a whole unlabeled load batch. Do not treat window count as independent devices.' if case['family']=='classification' else
                  'This is XJTU remaining-life prediction in nominal MINUTES to original terminal acquisition. Primary score is mean-per-bearing MAE; report RMSE and bearing/condition results. Training has nine full bearings; development has three separate causal prefixes. Final evaluation uses one query/past prefix per fresh process. No future prefixes, endpoints or final damage categories are provided.')
        extra=('\nAdditional historical training/CV/development evidence originally recorded before final scoring is under /app/historical_validation. Read INDEX.json for provenance, files and exact row mappings. You may recompute per-class/per-record/per-condition results from these existing arrays. Some scripts describe prior fitting; inspect them but do not run new fitting. No missing probability arrays are invented. These are the original training/development observations, not extra data.\n' if historical else '')
        prompt = run/'plans'/(ident+'.txt');prompt.write_text(COMMON+'\n'+detail+'\n'+extra)
        mounts = run/'plans'/(ident+'_mounts.json')
        save(mounts,dict(mounts=[dict(source=str(PYTHON),destination='/opt/python312'),dict(source=str(RUNTIME/'deps'),destination='/opt/phm_python/extra'),
            dict(source=str(tools),destination='/tools'),dict(source=str(run/'public_data'),destination='/data'),dict(source=str(run/'references'),destination='/reference')]))
        job=dict(id=ident,family='forge_real_coding',task_id=oldjob['task_id'],condition='fixed_candidate_review',repeat=1,
                 workspace=str(ws),prompt_file=str(prompt),answer_mode='text')
        plan=run/'plans'/(ident+'.json');save(plan,dict(jobs=[job]))
        socket=run/'ipc/s0.sock'
        job.update(workspace=str(ws.relative_to(run)),plan=str(plan),mounts=str(mounts),model_gateway_socket=str(socket),
                   dataset=case['dataset'],test_inputs_visible_during_development=False)
        save(run/'PLAN187.json',dict(jobs=[job]))
        protocol=dict(config['session_budget'],model='deepseek-flash',cli=str(CLI),cli_sha256=CLI_SHA,
            max_new_model_sessions=1,max_new_primary_rolling24h=400,local_max_concurrency=1,max_concurrency=8,
            training_gate='TASK_QUALIFICATION251.json',environment_gate='AGENT_ISOLATION_GATE251.json',
            authorization_record='audit/AUTHORIZATION251.json',controller_status_name='CONTROLLER_STATUS251.json',
            external_model_gateway=True,test_inputs_visible=False,original_benchmark_score_claimed=False,
            additional_gates=['PUBLIC_METADATA_GATE267.json','NATIVE_CLI_MCP_QUALIFICATION.json','OUTPUT_BUDGET_QUALIFICATION.json','FIXED_CANDIDATE_REPLAY281.json'])
        save(run/'PROTOCOL187.json',protocol)
        save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=ident,socket=str(socket))],base_url='https://api.deepseek.com/',model='deepseek-flash',
            max_requests=protocol['max_model_requests_per_subject'],max_input_tokens=protocol['max_total_input_tokens_per_subject'],
            max_output_tokens=protocol['max_total_output_tokens_per_subject'],max_output_per_request=protocol['max_output_tokens_per_request']))
        for name in ['cli_config187.toml','model_catalog187.json']:shutil.copy2(old/name,run/name)
        shutil.copy2(root/'audit/CLI_ONLINE_CHECK.json',run/'audit/CLI_ONLINE_CHECK.json')
        save(run/'TASK_QUALIFICATION251.json',dict(passed=True,phase='review_real_previously_fitted_artifacts',
            source_archive=str(source),source_archive_manifest_sha256=receipt['manifest_sha256'],real_training_reused=True,new_fitting_allowed=False,
            source_numerical_qualification='Inherited source-qualified measured data and original verified numerical fits; current development replay required before admission'))
        save(run/'audit/AUTHORIZATION251.json',dict(user_reply=config['authorization'],scope=config['purpose'],max_campaign_model_sessions=len(config['cases'])))
        save(run/'OWNERSHIP.json',dict(owner='root',writers_stopped=True,status='prepared_waiting_qualification'))
        save(run/'audit/BOUND_SOURCE_ARTIFACTS281.json',dict(source_archive=str(source),source_manifest_sha256=receipt['manifest_sha256'],files=bound))
        metadata_gate(run)
        freeze(run)
        frozen_path=run/'INITIAL_FREEZE187_FIFO_FILES.json'
        frozen=json.loads(frozen_path.read_text())
        frozen['files'] += [dict(path=str(p),sha256=sha256(p)) for p in sorted(tools.glob('*.py'))]
        save(frozen_path,frozen)
        lineage.append(dict(case,job=ident,candidates=len(candidates),source_workspace=original_workspace,
                            candidate_ids=[x['id'] for x in candidates],bound_files=len(bound)))
    save(root/'PREPARED281.json',dict(utc=utc(),cases=lineage,new_fits=0,final_inputs_exposed=False,model_sessions_started=0))
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    state(root,status='cases_prepared_waiting_candidate_replay',planned_model_sessions=len(config['cases']),model_sessions_started=0,
          new_fits=0,final_scores_read=False)


def main(root, mode):
    if mode=='prepare':return prepare(root)
    assert json.loads((root/'CANDIDATE_REPLAY_GATE281.json').read_text())['passed']
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    resume=mode=='resume-before-model'
    if resume:
        assert json.loads((root/'audit/PRE_ADMISSION_RECOVERY281.json').read_text())['zero_model_subjects_started']
        assert not (root/'PRE_ADMISSION_RESUMED281.json').exists()
        save(root/'PRE_ADMISSION_RESUMED281.json',dict(utc=utc(),model_reruns=0))
    else:
        assert not (root/'MODEL_EXECUTION_STARTED281.json').exists(), 'No automatic model rerun'
        save(root/'MODEL_EXECUTION_STARTED281.json',dict(utc=utc(),cases=[x['id'] for x in config['cases']]))
    os.environ.setdefault('PHM_CAPACITY_CACHE','/srv/yyf-zyq/PHM_20261010/_shared/verified_capacity_cache')
    try:
        execute_wave(root,'fixed_candidate_review',[root/x['id'] for x in config['cases']],reuse_pre_admission_qualification=resume)
    except BaseException as error:
        state(root,status='controller_error_inspect_before_resume',error=repr(error),automatic_retries=False)
        raise
    state(root,status='all_reviews_terminal_waiting_choice_audit',completed_model_sessions=len(config['cases']),new_fits=0,
          next_action='Validate completed choices and unchanged candidate hashes before consulting archived final predictions.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--mode',choices=['prepare','run','resume-before-model'],required=True)
    args=parser.parse_args();main(args.run.resolve(),args.mode)
