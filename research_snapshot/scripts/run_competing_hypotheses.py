"""Bounded B-versus-ordinary pilot reusing the native CLI and isolation harness."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from controlled_feedback import save, sha, utc
from controlled_program_checks import CANDIDATES, compact, accepted_choice
from replay_diagnostic_cli import replay, isolate_filesystem
from run_full_benchmark_campaign import clean_env
from run_validation_transfer import start_gateway
from qualify_native_cli_mcp import probe

PROJECT=Path(__file__).resolve().parents[1]
PYTHON=Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS=Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')

COMMON='''You are working on a measured PHM broken-rotor-bar diagnosis task. Choose a complete prediction program for held-out records. Classes are the documented counts 0,1,2,3,4. Read /reference/catalog.json and /data/metadata.json. Raw training/development measurements, labels, record groups and load metadata are legal. Final measurements, labels, class counts, target operating conditions and prior final results are unavailable. No independent-motor generalization is promised.

This is a FIXED-CANDIDATE program-selection experiment, not unrestricted algorithm invention. All five candidate recipes were genuinely fitted on these training records, including five whole-load CV fits per learned component and one full-training deployment fit. Fold-local learned preprocessing is included. No model was fitted on development data. All fitted artifacts were cold-reloaded and checked. The service now executes real cold replays and reports complete-program evidence; it does not fabricate scores or claim another fit occurred.

You may inspect the recipe source in /tools, write and execute analysis code, and use the native MCP tools under phmforge: status, check, hypotheses, plan, select. If native discovery is delayed, the exact same tools are available with:
python3 /tools/controlled_program_checks.py client --request '{"action":"status"}'
python3 /tools/controlled_program_checks.py client --request '{"action":"check","candidate_id":"P00"}'
Tools write genuine per-window and per-record CV/development evidence under /feedback/<candidate>/. The summary names those files. Do not fit extra models, alter features or thresholds, invent labels, create extra mixtures, access the Internet or use other model APIs. The evaluator deploys the registered checkpoint recipe, not arbitrary prose or a new local script.

P00 is the current program; P11B is the proposed combined update. P10,P01,P11 are controlled intermediate recipes. The common five-recipe catalog is available to both systems. Initial aggregate evidence for P00/P11B is in /reference/initial_evidence.json. Every candidate you select must have an executed check, except the predeclared system fallback P00 after failure. At most five distinct candidate checks are allowed; duplicate checks return the same bound evidence. Keep all contrary evidence and unknown explanations.

Objective: minimize final whole-record classification error; majority of window predictions determines each record class, ties choose the smaller class. Window macro-F1/accuracy and class behavior are secondary. No test outcome is available to guide your choice. The COMMON acceptance gate permits retaining P00. A replacement must have strictly lower mean(CV whole-record error, development whole-record error) than P00, measured on its entire pipeline. Otherwise the system keeps P00. Passing that empirical gate does not establish unknown-condition performance. You can retain P00 despite a candidate passing the gate if evidence is insufficient. The final candidate and reason must be submitted using select; write report.md with evidence citations and uncertainty.

The total budget for this entire selection episode, across every role if split, is 1800 seconds, 120 model requests, 12000000 input tokens and 262144 output tokens; maximum32768 output tokens/request. CPU tools use two numerical threads. Role handoffs consume this same budget. Store all scratch/cache files under /app/.runtime; never /tmp. External evaluation of these decisions occurs only after every episode has sealed its choice. All successes, failures, timeouts and fallbacks count.
'''

ROLES={
 'ordinary':'''Independently review the supplied old and proposed complete programs. Use the common tools and real evidence as useful, investigate alternative explanations, and keep or replace the current program. You may inspect all five candidates. You have the full single-session episode budget. Call select exactly once with candidate_id and a nonempty reason; copy its result to selection.json. Explain actual checks and limitations in report.md.''',
 'builder_proposal':'''You are the BUILD role in a three-role process. Inspect the actual recipes and execute checks for P00 and P11B. This stage may check only those two programs; intermediate checks are designed later. Write proposal.md describing the actual changes and legal observed outcomes without declaring a cause or final choice. Retain conflicting evidence. Do not call select. Stage maximum300seconds; preserve budget for later roles.''',
 'hypothesis':'''You are the HYPOTHESIS role. Read proposal.md and actual available evidence. Propose competing explanations with different observable predictions; explanations may coexist. Include insufficient evidence or instability when applicable. No final facts are available. Register 2-5 substantive explanations with hypotheses tool: each row has id,explanation,prediction,contradiction. Do not invent hypotheses to meet a count; explicitly report if fewer are defensible. Write hypotheses.md explaining how legal observations could distinguish them. This stage cannot run new candidate checks or select. Stage maximum240seconds.''',
 'experiment':'''You are the EXPERIMENT DESIGN role. Read proposal.md,hypotheses.md and registered evidence. Design a small controlled comparison within the five common recipes. Identify changes, fixed factors, what would distinguish each explanation, interactions you cannot identify and remaining data limitations. Do not merely rerun an aggregate score if hypotheses predict the same outcome. Register your plan with plan tool: candidate_ids,contrasts,fixed_factors,interpretation. The last three fields are text. The builder will execute these IDs next; include P00 and P11B if needed plus useful intermediates, up to all five. Write plan.md. No new checks or select in this stage. Stage maximum240seconds.''',
 'builder_checks':'''You are the BUILD role returning to execute the registered controlled checks. Read plan.md and call check for the registered candidate IDs; each really replays the fitted entire pipeline. You may write analysis code to compare group/record behavior, main effects or interactions as supported by the plan. Verify the actual factors correspond to the intended contrasts. Record measured evidence and failures in check_results.md; do not replace missing evidence with simulated values or claim unknown conditions certified. Do not call select. Stage maximum780seconds.''',
 'hypothesis_decision':'''You are the HYPOTHESIS role returning after actual checks. Read all handoffs and real evidence. For each proposed explanation report supported in observed scope, contradicted prediction, inconclusive, or not validly executed; multiple explanations may remain. Use actual measured comparisons to choose a checked complete program or retain P00. No new checks in this stage. Call select exactly once with candidate_id and reason,copy the returned final_choice to selection.json, and write report.md with evidence paths, unresolved explanations and scope. A good explanation alone does not justify replacement. Stage maximum240seconds.'''
}

ROLES.update({
 'reviewer_plan':'''You are the REVIEW role in a two-role process. Read proposal.md and the actual initial evidence. Propose competing explanations with different observable predictions, then design controlled checks that can distinguish them using the common recipes. These are your combined hypothesis and experiment-design responsibilities. Register 2-5 defensible explanations with hypotheses: id,explanation,prediction,contradiction. Explanations may coexist; report insufficient evidence instead of inventing a cause. Register a plan with plan: candidate_ids,contrasts,fixed_factors,interpretation (last three are text). Identify actual changed and fixed factors, distinguishable predictions, interactions not identified, and remaining limitations. Write hypotheses.md and plan.md. The builder will execute the plan. No new checks or select in this stage.''',
 'reviewer_decision':'''You are the REVIEW role returning after the builder executed your plan. Read proposal.md,hypotheses.md,plan.md,check_results.md and actual evidence. Update each explanation as supported in observed scope, contradicted prediction, inconclusive, or not validly executed. Explanations may coexist. Choose a checked complete program or retain P00 under the common acceptance rule. A good explanation alone does not justify replacement. No new checks. Call select exactly once with candidate_id and reason; copy final_choice to selection.json. Write report.md with actual evidence paths and remaining uncertainty.''',
 'coordinator_dispatch':'''You are the COORDINATOR. Read proposal.md and actual initial evidence. Assign three independent investigation branches that can help distinguish explanations of the observed program difference. Focus on relevant changed factors or evidence limitations, not advocates for predetermined answers. Write /app/dispatch.json as a JSON list of exactly three objects with id (branch_1,branch_2,branch_3 respectively) and focus (a substantive text instruction). Each branch must propose competing explanations, different observable predictions, and checks within the common recipes, including ways its explanation could fail. The HOST CONTROLLER, not you, launches the three isolated investigation sessions AFTER this stage exits. Do not spawn, message, wait for or poll other agents, and do not start another CLI or model API client. Your only handoff is the dispatch file; end the stage after writing and validating it. Branches will receive your dispatch and proposal plus the same legal data and evidence, but cannot read each other's work until they all finish. No new checks, hypothesis/plan registration or selection in this dispatch stage. Write /app/dispatch.md if helpful. Branches share this episode's budget.''',
 'investigation':'''You are one INDEPENDENT INVESTIGATOR, already launched by the host controller. Read /app/inputs/proposal.md and /app/inputs/dispatch.json. Your branch ID is specified below: perform that focus. Do not spawn, contact or wait for other agents, or launch another CLI or model API client. You have the same legal data, catalog and initial evidence as the other roles. Do not guess their conclusions. Propose 1-3 substantive competing explanations or explicit insufficiency; include observable predictions and contradiction conditions. Write the FINAL HANDOFF to exactly /app/hypotheses.json (a JSON list with id,explanation,prediction,contradiction text fields) and /app/investigation.md (proposed contrasts, fixed factors, interpretation and limitations). These two deliverables belong at /app root; only scratch belongs in /app/.runtime. Tools may report status but you cannot run new candidate checks, register changes or select in this stage. The coordinator later merges independent reports and registers the common plan. Do not fabricate evidence. Analyses of already available evidence and legal measurements are allowed. Complete the two handoff files and end the stage within its budget.''',
 'coordinator_plan':'''You are the COORDINATOR returning after three independent investigations. Read proposal.md,dispatch.json and the frozen reports under /app/inputs/branches/. Preserve disagreement and contrary evidence. Merge duplicated explanations and controlled comparisons; identify cross-branch interactions not covered. Register 2-6 defensible explanations using hypotheses: id,explanation,prediction,contradiction. Register plan with candidate_ids,contrasts,fixed_factors,interpretation (last three are text). Prioritize execution validity, then minimal comparisons that distinguish unresolved explanations, with lower-cost comparisons as a tie-breaker. Do not invent numerical hypothesis probabilities, take a majority vote, or assemble untested components. Only the common five complete programs are selectable. Write hypotheses.md and plan.md. No new checks or select; the builder next executes this plan.''',
 'coordinator_decision':'''You are the COORDINATOR returning after the actual controlled checks. Read all handoffs, independent reports and measured results. Update explanations as supported in observed scope, contradicted prediction, inconclusive, or not validly executed. Explain remaining conflicts and limitations. Multiple causes may remain; a majority vote or plausible explanation is not sufficient for a program update. Choose a checked complete program or retain P00 using the common acceptance rule. No new checks. Call select exactly once with candidate_id and reason, copy final_choice to selection.json, and write report.md with actual evidence paths and uncertainty.'''
})


def update(root,**changes):
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    path=root/('STATUS'+config.get('iteration','A325')[1:]+'.json');state=json.loads(path.read_text()) if path.exists() else {}
    state.update(updated_utc=utc(),**changes);save(path,state)
    save(PROJECT/config['result_directory']/'current_run.json',dict(state,run=str(root)))


def prepare_episode(root,config,block,arm,repeat):
    name=f'{block}_{arm}_{repeat}';run=root/'episodes'/name
    for d in ['app/.runtime/tmp','app/.runtime/mpl','app/artifacts','runtime/tmp','runtime/mpl','audit','logs','broker','feedback','reference','public','plans','primary']:
        (run/d).mkdir(parents=True,exist_ok=True)
    for part in ['train','val']:os.link(root/'hidden'/block/(part+'.npz'),run/'public'/(part+'.npz'))
    metadata=dict(task='PHMForge pdm_fault_004 research adaptation',measured_source=True,channels=5,sampling_hz=7600,
        amplitude_units='Original values;physical calibration unverified',classes=list(range(5)),
        window_samples=2048,record_definition='One complete measured acquisition at a source fault state and load',
        train_and_development_fields=['waveforms','features','labels','groups','sample_ids','operating_load_Nm'],
        final_observations_and_condition_range_available=False,one_laboratory_motor=True)
    save(run/'public/metadata.json',metadata)
    catalog=dict(candidates=[dict(id=c,features=('spectral' if c in ['P10','P11','P11B'] else 'raw_plus_spatial_plus_spectral'),
        tree_parameters=config['tree_parameters']['old' if c in ['P00','P10'] else 'new'],
        blend=('.9tree+.1standardized_raw_logistic_C10' if c=='P11B' else 'none')) for c in CANDIDATES],
        numerical_seed=config['numerical_seed'],class_weight='balanced',training_scope='train_only',
        folds='Leave each of the five observed training loads out, refit all learned components',
        starting_candidate='P00',proposed_update='P11B',selection_rule=config['selection_rule'])
    save(run/'reference/catalog.json',catalog)
    reports=json.loads((root/'bank'/block/'VALIDATION_REPORTS.json').read_text())
    save(run/'reference/initial_evidence.json',{c:compact(reports[c]) for c in ['P00','P11B']})
    (run/'app/AGENTS.md').write_text('Use only the common five complete-program recipes and counted real check service. Training/development evidence is permitted;final data unavailable. Do not fit extra models. Preserve measurements and failures. Scratch under /app/.runtime only.\n')
    base=(root/'base_cli_config.toml').read_text()
    base+='''[mcp_servers.phmforge]
command = "/opt/python312/bin/python3.12"
args = ["-B", "/tools/controlled_program_checks.py", "mcp"]
required = true
env = { PYTHONPATH = "/opt/phm_python/extra", PYTHONDONTWRITEBYTECODE = "1", TMPDIR = "/app/.runtime/tmp", TMP = "/app/.runtime/tmp", TEMP = "/app/.runtime/tmp", OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "2", MKL_NUM_THREADS = "2" }
startup_timeout_sec = 60
tool_timeout_sec = 180
'''
    (run/'cli_config187.toml').write_text(base)
    shutil.copy2(root/'model_catalog187.json',run/'model_catalog187.json')
    save(run/'mounts.json',{'mounts':[{'source':str(src),'destination':dst} for src,dst in [
        (PYTHON,'/opt/python312'),(DEPS,'/opt/phm_python/extra'),(root/'shared_tools','/tools'),
        (run/'public','/data'),(run/'reference','/reference'),(run/'feedback','/feedback')]]})
    budget=config['episode_budget'];socket=run/'ipc/s.sock';assert len(str(socket))<108
    save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=name,socket=str(socket))],
        base_url='https://api.deepseek.com/',model='deepseek-flash',max_requests=budget['model_requests'],
        max_input_tokens=budget['input_tokens'],max_output_tokens=budget['output_tokens'],max_output_per_request=budget['output_per_request'],max_inflight=3 if arm=='C' else 1))
    if config.get('enforce_external_roles'):
        gateway_config=json.loads((run/'MODEL_GATEWAY_CONFIG.json').read_text())
        gateway_config.update(denied_tool_namespaces=['collaboration'],
            denied_tool_names=['spawn_agent','followup_task','interrupt_agent','list_agents','send_message','wait_agent'],
            buffer_verified_responses=True)
        save(run/'MODEL_GATEWAY_CONFIG.json',gateway_config)
    save(run/'BROKER_CONFIG.json',dict(campaign=str(root),block=block,arm=arm,request_directory='app/.runtime/feedback_requests'))
    if arm=='C':
        queues=['app/.runtime/feedback_requests']
        for index in range(1,4):
            app=run/'branches'/f'branch_{index}'/'app'
            for d in ['.runtime/tmp','.runtime/mpl','inputs']: (app/d).mkdir(parents=True)
            shutil.copy2(run/'app/AGENTS.md',app/'AGENTS.md')
            queues.append(str((app/'.runtime/feedback_requests').relative_to(run)))
        save(run/'BROKER_CONFIG.json',dict(campaign=str(root),block=block,arm=arm,
            request_directory=queues[0],request_directories=queues))
    save(run/'PHASE.json',dict(role='qualification'))
    save(run/'PROTOCOL187.json',dict(cli=config['cli'],expected_mcp_tools=['mcp__phmforge__'+s for s in ['status','check','hypotheses','plan','select']]))
    return dict(id=name,run=run,block=block,arm=arm,repeat=repeat,socket=socket)


def args_for(row,stage,seconds):
    run=row['run'];config=json.loads((Path(row['run']).parents[1]/'CAMPAIGN_CONFIG.json').read_text())
    return SimpleNamespace(cli=Path(config['cli']),model='deepseek-flash',output=run/'primary',plan=run/'plans/unused.json',
        timeout=seconds,analysis_tools=True,permission_profile='iot-offline',isolation_config=run/'cli_config187.toml',
        model_gateway_socket=row['socket'],model_catalog=run/'model_catalog187.json',runtime_mounts_file=run/'mounts.json',
        workspace_alias='/app',credential_env_file=None,credential_env_name='DEEPSEEK_API_KEY',snapshot_generated_source=True,
        cleanup_runtime_cache=True)


ISOLATION_PROBE='''import json,os,socket
from pathlib import Path
checks={}
checks['cwd']=str(Path.cwd())=='/app'
checks['private_paths_absent']=not any(Path(x).exists() for x in ['/home/jupyter-yyf-zyq/.bashrc','/home/jupyter-yyf-zyq/.codex/auth.json','/data/yyf-zyq/PHM_20261010','/home/jupyter-yyf-zyq/Bo_2/PHM_20261010','/srv/yyf-zyq/PHM_20261010/_shared'])
checks['keys_absent']=not any('API_KEY' in k or 'SECRET' in k for k in os.environ)
checks['data_allowlist']={p.name for p in Path('/data').iterdir()}=={'train.npz','val.npz','metadata.json'}
checks['temporary_path']=os.environ['TMPDIR'].startswith('/app/.runtime/')
try:
 s=socket.create_connection(('1.1.1.1',443),timeout=1);s.close();checks['network_blocked']=False
except OSError:checks['network_blocked']=True
print(json.dumps({'checks':checks,'passed':all(checks.values())}))
raise SystemExit(0 if all(checks.values()) else 1)
'''


def qualify(row,workspace=None,branch_name=None):
    run=row['run'];ns=args_for(row,'qualification',90)
    folder=run/'audit'/('isolation_'+branch_name if branch_name else 'isolation');folder.mkdir()
    command=isolate_filesystem([str(ns.cli),'sandbox','-P','iot-offline','-C','/app','--',
        '/opt/python312/bin/python3.12','-B','-c',ISOLATION_PROBE],ns,workspace or run/'app',folder,None)
    env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',PATH='/opt/python312/bin:/usr/bin:/bin')
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
    (folder/'stdout').write_text(result.stdout);(folder/'stderr').write_text(result.stderr)
    assert result.returncode==0, 'Isolation failed: '+result.stderr[-1000:]
    check=json.loads(result.stdout.strip().splitlines()[-1]);assert check['passed']
    if branch_name:
        save(folder/'RESULT.json',dict(passed=True,isolation=check,new_model_calls=0))
        return
    job=dict(id=row['id'],mounts=str(run/'mounts.json'))
    native=probe(run,run,job,'fixed_discovery',(run/'cli_config187.toml').read_text())
    assert native['passed'],'Native CLI did not expose every controlled-check MCP tool'
    save(run/'QUALIFICATION.json',dict(passed=True,isolation=check,native_MCP=native,new_model_calls=0))


def run_episode(root,row,stop,gateway):
    run=row['run'];config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    broker=None;stages=[];started=time.monotonic()
    try:
        env=clean_env(run);env.update(PYTHONPATH=str(DEPS)+':'+str(root/'code'),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',JOBLIB_TEMP_FOLDER=str(run/'runtime/tmp'))
        with (run/'logs/broker.log').open('w') as log:
            broker=subprocess.Popen([str(PYTHON/'bin/python3.12'),'-B','-u',str(root/'code/controlled_program_checks.py'),
                'serve','--run',str(run)],env=env,stdout=log,stderr=subprocess.STDOUT)
        for _ in range(200):
            if (run/'BROKER_READY.json').exists():break
            assert broker.poll() is None,'Broker exited before ready'
            time.sleep(.1)
        assert (run/'BROKER_READY.json').exists()
        schedule=([dict(role='ordinary',seconds=1800)] if row['arm']=='ordinary' else config[row['arm']+'_stages'])
        started=time.monotonic()
        def execute(i,role,seconds,workspace=None,extra=''):
            remaining=1800-(time.monotonic()-started)
            assert remaining>=5,'Episode wall budget exhausted'
            prompt=run/'plans'/(str(i)+'_'+role+extra+'.txt')
            role_text=ROLES[role]
            if role=='builder_proposal':role_text=role_text.replace('three-role',row['arm']+'-route')
            if role=='builder_checks':role_text=role_text.replace('Stage maximum780seconds.','')
            prompt.write_text(COMMON+'\nCURRENT ROLE\n'+role_text+f'\nStage maximum {seconds} seconds; remaining episode wall time {int(remaining)} seconds.\n'+('Your branch ID: '+extra if extra else ''))
            job=dict(id=f'{row["id"]}_{i}_{role}'+extra,condition=row['arm'],repeat=row['repeat'],
                workspace=str(workspace or run/'app'),prompt_file=str(prompt),answer_mode='text')
            result=replay(job,args_for(row,role,min(seconds,int(remaining))))
            result['role']=role;result['branch_id']=extra or None
            return result
        for i,stage in enumerate(schedule):
            if stop.is_set():raise RuntimeError('Provider account unavailable; no further stage admitted')
            remaining=1800-(time.monotonic()-started)
            if remaining<5:break
            role=stage['role'];save(run/'PHASE.json',dict(role=role,started_utc=utc(),remaining_seconds=remaining))
            if role=='investigation':
                assignment=json.loads((run/'app/dispatch.json').read_text())
                assert isinstance(assignment,list) and len(assignment)==3
                assert [a['id'] for a in assignment]==['branch_1','branch_2','branch_3']
                assert all(isinstance(a['focus'],str) and a['focus'].strip() for a in assignment)
                apps=[run/'branches'/a['id']/'app' for a in assignment]
                for app in apps:
                    for file in ['proposal.md','dispatch.json']:shutil.copy2(run/'app'/file,app/'inputs'/file)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures=[pool.submit(execute,i,role,stage['seconds'],app,a['id']) for app,a in zip(apps,assignment)]
                    branch_results=[f.result() for f in futures]
                stages.extend(branch_results)
                if not all(x['status']=='finished' for x in branch_results):break
                frozen=[]
                for app,a in zip(apps,assignment):
                    hypotheses=json.loads((app/'hypotheses.json').read_text())
                    assert isinstance(hypotheses,list) and 1<=len(hypotheses)<=3
                    assert all(all(isinstance(h.get(k),str) and h[k].strip() for k in ['id','explanation','prediction','contradiction']) for h in hypotheses)
                    out=run/'app/inputs/branches'/a['id'];out.mkdir(parents=True)
                    for file in ['hypotheses.json','investigation.md']:
                        shutil.copy2(app/file,out/file)
                        frozen.append(dict(branch=a['id'],file=file,sha256=sha(app/file)))
                save(run/'audit/INDEPENDENT_BRANCH_HANDOFFS.json',dict(utc=utc(),files=frozen,
                    peer_workspaces_unmounted=True,shared_feedback_during_investigation='initial P00/P11B only',all_branches_finished_before_merge=True))
            else:
                result=execute(i,role,stage['seconds']);stages.append(result)
                if result['status']!='finished':break
            for receipt in (run/'model_gateway').glob('*/*_receipt.json'):
                if json.loads(receipt.read_text()).get('upstream_status')==402:stop.set()
            state=json.loads((run/'broker/STATE.json').read_text())
            if role=='hypothesis' and not state['hypotheses']:break
            if role=='experiment' and not state['plans']:break
            if role in ['reviewer_plan','coordinator_plan'] and not (state['hypotheses'] and state['plans']):break
        state=json.loads((run/'broker/STATE.json').read_text())
        choice=state['final_choice']
        expected_stages=sum(3 if s['role']=='investigation' else 1 for s in schedule)
        complete=choice is not None and bool(stages) and all(s['status']=='finished' for s in stages) and len(stages)==expected_stages
        result=dict(id=row['id'],block=row['block'],arm=row['arm'],repeat=row['repeat'],stages=stages,
            selected=choice['selected'] if choice else 'P00',choice=choice,completed=complete,
            fallback=choice is None,checks=list(state['checks']),hypotheses=state['hypotheses'],plans=state['plans'],
            elapsed_seconds=time.monotonic()-started,final_scores_opened=False)
    except Exception as exc:
        result=dict(id=row['id'],block=row['block'],arm=row['arm'],repeat=row['repeat'],stages=stages,
            selected='P00',completed=False,fallback=True,error_type=type(exc).__name__,error=str(exc),final_scores_opened=False)
    finally:
        if broker is not None:
            broker.terminate()
            try:broker.wait(timeout=10)
            except subprocess.TimeoutExpired:broker.kill();broker.wait()
        if gateway is not None:
            (run/'MODEL_GATEWAY_STOP').touch()
            gateway.wait(timeout=650)
    save(run/'EPISODE_RESULT.json',result)
    return result


def main(root):
    from threading import Event
    config=json.loads((root/'CAMPAIGN_CONFIG.json').read_text())
    suffix=config.get('iteration','A325')[1:]
    prep=json.loads((root/('PREPARATION'+suffix+'.json')).read_text())
    assert prep['passed'] and prep['actual_component_fits']==60
    if config.get('reuse_existing_bank'):assert prep['actual_new_fits']==0 and prep['researcher_previously_knows_final_scores']
    else:assert not prep['final_scores_opened']
    if config.get('reuse_existing_bank'):
        assert json.loads((root/'audit/ENGINEERING_AUDIT.json').read_text())['passed']
        for name,digest in prep['code_hashes'].items():
            assert sha(PROJECT/'scripts'/name)==digest and sha(root/'code'/name)==digest, 'Execution code changed after freeze: '+name
    assert prep['protocol_sha256']==sha(root/'CAMPAIGN_CONFIG.json')
    assert not (root/'ALL_CHOICES_SEALED.json').exists()
    episodes=[prepare_episode(root,config,b['id'],arm,rep) for rep in [1,2] for b in config['blocks'] for arm in config['arms']]
    # All public data and common starting evidence are identical within a block.
    for block in config['blocks']:
        hashes=[]
        for row in episodes:
            if row['block']==block['id']:
                hashes.append({str(p.relative_to(row['run'])):sha(p) for folder in ['public','reference'] for p in sorted((row['run']/folder).glob('*'))})
        assert all(h==hashes[0] for h in hashes)
    save(root/'audit/PAIRED_INFORMATION_GATE.json',dict(passed=True,episodes=len(episodes),test_inputs_unmounted=True))
    update(root,status='running',controller_pid=os.getpid(),planned_episodes=len(episodes),completed_episodes=0,
        planned_maximum_stage_sessions=config['maximum_cli_stage_sessions'],actual_PHM_component_fits=60,final_scores_opened=False)
    results=[];stop=Event();gateways={}
    try:
        for row in episodes:
            gateway=start_gateway(row['run']);gateways[row['id']]=gateway
            qualify(row)
            if row['arm']=='C':
                for j in range(1,4):qualify(row,row['run']/'branches'/f'branch_{j}'/'app',f'branch_{j}')
    except BaseException as error:
        for row in episodes:
            if row['id'] in gateways:
                (row['run']/'MODEL_GATEWAY_STOP').touch();gateways[row['id']].wait(timeout=30)
        update(root,status='pre_subject_qualification_failed',error=type(error).__name__+': '+str(error),
               new_agent_sessions=0,final_scores_opened=False)
        raise
    save(root/'audit/ALL_NATIVE_GATES.json',dict(passed=True,episodes=len(episodes),new_model_calls=0))
    with ThreadPoolExecutor(max_workers=config['maximum_parallel_episodes']) as pool:
        futures={pool.submit(run_episode,root,row,stop,gateways[row['id']]):row['id'] for row in episodes}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            update(root,status='running',completed_episodes=len(results),last_completed={k:result.get(k) for k in ['id','completed','selected','error']})
    reports={b['id']:json.loads((root/'bank'/b['id']/'VALIDATION_REPORTS.json').read_text()) for b in config['blocks']}
    fixed={}
    for block,bank in reports.items():
        dev=lambda c:bank[c]['development']['record_error'];cv=lambda c:bank[c]['cv']['record_error']
        proposed=min(CANDIDATES,key=lambda c:((dev(c)+cv(c))/2,CANDIDATES.index(c)))
        accepted,reason=accepted_choice(proposed,bank,CANDIDATES)
        fixed[block]=dict(keep_P00='P00',development_best=min(CANDIDATES,key=lambda c:(dev(c),cv(c),CANDIDATES.index(c))),
            CV_best=min(CANDIDATES,key=lambda c:(cv(c),dev(c),CANDIDATES.index(c))),complete_fixed_grid=accepted,
            fixed_grid_proposed=proposed,fixed_grid_gate_reason=reason)
    freeze=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in sorted(root.rglob('*'))
        if p.is_file() and not p.is_symlink() and ('clients-cache' not in p.parts) and ('runtime' not in p.parts)
        and p.name!=('STATUS'+suffix+'.json')]
    seal=dict(utc=utc(),protocol_sha256=sha(root/'CAMPAIGN_CONFIG.json'),episodes=sorted(results,key=lambda r:r['id']),
        fixed_controls=fixed,files=freeze,final_scores_opened=False)
    save(root/'ALL_CHOICES_SEALED.json',seal)
    save(PROJECT/config['result_directory']/('CHOICES'+suffix+'.json'),{k:v for k,v in seal.items() if k!='files'})
    update(root,status='all_terminal_choices_sealed_waiting_external_evaluation',completed_episodes=len(episodes),
        normal_episodes=sum(r['completed'] for r in results),final_scores_opened=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    main(parser.parse_args().run.resolve())
