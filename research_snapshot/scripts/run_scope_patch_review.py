"""A392: same-budget native coding versus a separate, edit-level review.

This is a developmental qualification, not a novel-method or independent-data
claim. No query or location label is exposed during either coding stage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import time
from types import SimpleNamespace
import urllib.request

from controlled_feedback import save, sha, utc
from native_validation_comparison import closure
from replay_diagnostic_cli import replay, isolate_filesystem
from run_full_benchmark_campaign import clean_env
from run_validation_transfer import start_gateway

PROJECT = Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE = Path('/data/yyf-zyq/PHM_20261010/20261019T174700Z_scopereview227')
SIGNALS = Path('/data/yyf-zyq/PHM_20261010/20261025T162500Z_path383')
TEMPLATE = Path('/data/yyf-zyq/PHM_20261010/20261024T202500Z_decision340')
CLI = Path('/data/yyf-zyq/PHM_20261010/20261023T052522Z_rotor305/runtime/codex0164/bin/codex')
PYTHON = Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
DEPS = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')
OUT = PROJECT/'results/patch392'
read = lambda p: json.loads(Path(p).read_text())


def output_directory(root):
    return PROJECT/('results/patch392r1' if root.name.endswith('patch392r1') else 'results/patch392')

COMMON = '''Review and improve the actual guided-wave localization PROGRAM in /app/solution.py. Read /app/instruction.md for the original task, legal measurements, physical units and predict interface. The present review episode supersedes the original long task budget. You may write and execute arbitrary numerical Python and modify the entire predictor, not merely choose a fixed candidate. /reference/incumbent.py is the immutable starting implementation. /reference/candidate_transmission.py and candidate_waveform.py are two pre-existing numerical tools; neither is certified correct and no historical performance is provided. Retaining an effective implementation or making a useful change are both valid. Do not assume a more complex program is better.

Only the supplied healthy measurement and its geometry are available during development. No damaged query, damage label, final condition, evaluation result or other Agent workspace is accessible. Do not access the internet, spawn agents or call other model services. Do not create synthetic waveforms, synthetic damage labels or simulated faults. Real signal analysis and numerical parameter fitting to allowed measurements are permitted; there are no supervised damage labels. Software checks, mathematical arguments and measurement checks have different scopes. A signal fit or executable program alone does not establish damage-location accuracy. Avoid both harmful replacement and unnecessary rejection of useful improvements.

Submit a self-contained /app/solution.py with predict(healthy_reference,query,query_geometry), and /app/report.md with actual changes, checks, contrary evidence and uncertainty. Original third-party dependencies and the 120-second per-prediction limit remain. Final query evaluation happens externally only after every episode has finished. Do not invent a final score. Use at most two numerical threads and /app/.runtime for scratch; never /tmp. New files may be written under /app only. The immutable inputs and incumbent remain readable for comparisons.
'''

ORDINARY = '''You are an ordinary coding Agent. Independently inspect, test, improve or retain the solution using the common inputs. You may organize your work, revise earlier choices, inspect code differences and critically review your changes as useful. Do not finish until you have checked the submitted program and explained the actual evidence and its limits. Total budget:1800seconds,60model requests,4000000input tokens,131072output tokens,16384output tokens/request. No extra review or retry is supplied after submission.'''

PROPOSE = '''This is stage1 of an edit-level review workflow. Develop a useful proposed program, preserving the immutable incumbent. Prefer separable changes when they represent distinct hypotheses; arbitrary algorithm changes are allowed. Write the proposed final code to solution.py. In proposal.json list each substantive changed computation, its intended benefit, a rival explanation, the actual check files and what each check does and does not establish. Include an explanation of the strongest reason to KEEP each useful change, so the next reviewer cannot equate caution with rejecting everything. Do not fabricate a discriminating observation when available data cannot provide one. A later independent context will review the exact code and artifacts, without your internal conversation. Stage budget:1000seconds,35model requests,2500000input tokens,81920output tokens,16384output tokens/request. Complete the proposal and report within this budget.'''

REVIEW = '''This is stage2 in a fresh review context. /prior is a frozen copy of stage1's proposed code, reports and executed check artifacts. The immutable original is /reference/incumbent.py. Review the ACTUAL code delta, not just the previous narrative. Decompose coupled changes by the behavior or physical assumption they alter. For each change, check its claimed evidence against the actual input, executed computation and output, and state a rival explanation. A claim about a component, a healthy propagation fit or numerical execution must not silently become proof about damage localization or a new condition. At the same time, uncertainty is not proof that the incumbent is correct: explicitly examine the cost of removing a plausible useful change and any independently verifiable task constraint or numerical improvement.

Use the remaining budget to execute discriminating checks on allowed real measurements or verify mathematical/code properties. Do not invent damaged examples. Retain, remove or repair changes separately where technically coherent; inspect dependencies and run the resulting COMPLETE program. You are free to retain all, accept all, or write another justified program. No fixed candidate, algorithm family, score threshold or keep preference is imposed. Save review_manifest.json with each reviewed change, actual affected computation, evidence paths plus SHA256, scope, strongest counterevidence, decision and reason. Missing evidence must be marked missing. The final solution.py and report.md are the deliverables. A checklist is not proof of correctness. Stage budget:800seconds,25model requests,1500000input tokens,49152output tokens,16384output tokens/request. These are the remaining resources of the SAME1800second/60request/4M-input episode, not an additional budget.'''


def status(root, **changes):
    p=root/'STATUS392.json'; value=read(p) if p.exists() else {}
    value.update(utc=utc(), **changes);save(p,value);save(output_directory(root)/'current_run.json',dict(value,run=str(root)))


def bound_copy(base, relative, target, bindings):
    receipt=read(base/'ARCHIVE_VERIFIED.json')
    manifest=base/receipt.get('manifest_filename','SOURCE_MANIFEST.json')
    assert sha(manifest)==receipt['manifest_sha256']
    index={r['path']:r for r in read(manifest)['files']}
    source=base/relative;digest=sha(source);assert digest==index[relative]['sha256']
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    bindings.append(dict(archive=str(base),member=relative,sha256=digest,target=str(target)))


def prepare(root, recover_from=None):
    assert not root.exists(), 'Unique run required; no overwrite or automatic restart'
    for d in ['runtime/tmp','runtime/mpl','audit','code','tools','scripts','public/data','reference','episodes','sealed']:
        (root/d).mkdir(parents=True,exist_ok=True)
    output_directory(root).mkdir(exist_ok=True)
    bindings=[]
    receipt=read(SOURCE/'ARCHIVE_VERIFIED.json');manifest=SOURCE/receipt.get('manifest_filename','SOURCE_MANIFEST.json')
    assert sha(manifest)==receipt['manifest_sha256']
    index={r['path']:r for r in read(manifest)['files']}
    members={
        'workspaces/sreview227_01/instruction.md':'public/instruction.md',
        'primary/source_history/sreview227_01/b3face277af4e4b97e162ea1f8e0a0f109a6ece38209240a0aaaa7a47096ad0b.py':'reference/incumbent.py',
    }
    with tarfile.open(SOURCE/'RUN_CONTENTS.tar.gz','r:gz') as archive:
        for name,dest in members.items():
            data=archive.extractfile(name).read();digest=hashlib.sha256(data).hexdigest()
            assert digest==index[name]['sha256'];(root/dest).write_bytes(data)
            bindings.append(dict(archive=str(SOURCE),member=name,sha256=digest,target=dest))
    for name in ['healthy_reference.npz','reference_geometry.json']:
        bound_copy(SIGNALS,'inputs/'+name,root/'public/data'/name,bindings)
    for name in ['transmission','waveform']:
        bound_copy(SIGNALS,'source/'+name+'.py',root/'reference'/('candidate_'+name+'.py'),bindings)
    for name in ['base_cli_config.toml','model_catalog187.json','retention_policy.json']:
        bound_copy(TEMPLATE,name,root/name,bindings)
    url='https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-research'}),timeout=30) as response:
        release=json.load(response)
    assert release['tag_name']=='rust-v0.164.0'
    assert sha(CLI)=='b93f6811d3f653f0fb83be4050d857fb9abbb111dfa2ce160563327daff7f306'
    save(root/'audit/CLI_SOURCE392.json',dict(url=url,tag=release['tag_name'],published_at=release['published_at'],checked_utc=utc(),cli=str(CLI),sha256=sha(CLI)))
    names=closure(['run_scope_patch_review'])
    for name in names:shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',root/'tools/isolated_model_gateway.py')
    shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',root/'scripts/isolated_model_gateway.py')
    protocol=dict(id='A392',frozen_utc=utc(),model='deepseek-flash',cli=str(CLI),cli_sha256=sha(CLI),
        arms=['ordinary','edit_review'],repeats=2,episodes=4,native_sessions=6,maximum_concurrent_episodes=4,
        episode_seconds=1800,episode_requests=60,episode_input_tokens=4000000,episode_output_tokens=131072,
        output_per_request=16384,numerical_threads=2,maximum_total_requests=240,maximum_disk_GiB=2,
        method_stages=[dict(seconds=1000,requests=35,input_tokens=2500000,output_tokens=81920),dict(seconds=800,requests=25,input_tokens=1500000,output_tokens=49152)],
        factor='Fresh-context review of individual executable changes and their actual evidence scope; free final coding. No forced physical-family retention or outcome-derived rule.',
        prediction='The independent edit-level audit will undo unsupported harmful changes without discarding useful numerical or task-constraint improvements.',
        rivals=['A fresh context alone helps','Keeping the incumbent explains any gain','Split compute budget hurts','The legal data cannot distinguish changes','Ordinary review already sufficient'],
        admission='New automation qualification of a flexible full-coding workflow. A386 fixed-family gate remains failed; no native expansion of that fixed rule. Fresh review is known prior art, novelty not established.',
        common_information='Exact same incumbent, two numerical tools, healthy measurement, task contract, numerical libraries. Method adds only a staged process and an edit manifest; ordinary can perform identical checks.',
        independent_confirmation_scenarios=0,previously_opened_queries=7,source_tasks=1,query_geometries=2,
        scores_not_visible_to_actors=True,query_waveforms_not_visible_during_coding=True,real_observations_only=True,synthetic_damage_forbidden=True,
        primary='Mean Euclidean localization error/mm. Average7queries within episode, then2repeats per arm. Four episodes are NOT four independent scenarios.',
        secondary=['fraction<=20mm','mean positive degradation vs original incumbent','mean positive improvement vs incumbent','runtime and actual usage','method edit-manifest compliance'],
        gate='All4episodes valid, method mean error strictly below ordinary and keep, harm no greater and positive improvement no smaller than ordinary. Joint gate, not majority-goal completion.',
        fixed_controls=['immutable incumbent','unchanged waveform tool','historical successful ordinary program reported separately, not a contemporaneous equal-budget control'],
        failure_policy='Retain all missing submissions, runtime failures and budget truncations. No outcome-based reruns. Missing outputs separately reported with keep fallback but cannot pass gate.',
        supervised_fit_budget=0,numerical_signal_fits_allowed=True,per_prediction_seconds=120,
        external_evaluation='All output sources sealed before reading query arrays/labels. Fresh isolated prediction process for each query; labels never mounted.',
        fixed_rule_failure386_preserved=True,protected_battery22_opened=False,protected_A33872_opened=False,novelty_confirmed=False,
        literature=[dict(url='https://arxiv.org/html/2607.24604v3',scope='StateSeal: state-bound evidence/version protection already exists; workshop, not main-conference novelty.'),dict(url='https://arxiv.org/html/2609.09133',scope='ExecCritic independent checking already exists; this study tests PHM edit-scope behavior and does not claim an original critic architecture.')])
    if recover_from is not None:
        previous=read(recover_from/'PROTOCOL392.json')
        assert previous['id']=='A392' and not (recover_from/'RESULT392.json').exists()
        # This engineering recovery is selected from budget receipts only,
        # before inspecting any proposed program, prediction or task score.
        meta=read(recover_from/'episodes/edit_review_2/propose/model_gateway/edit_review_2_propose/status.json')
        assert meta['input_tokens']>2500000 and not meta['active']
        for repeat in [1,2]:
            prior=read(recover_from/'episodes'/f'ordinary_{repeat}'/'TERMINAL.json')
            assert prior['valid'] and prior['arm']=='ordinary'
        protocol.update(id='A392R1',recovery_of=str(recover_from),recovery_attempt_limit=1,
            recovery_trigger='Fixed stage1 input allocation exhausted while the whole episode still has unused input budget. Only numeric receipts/status were inspected; no task scores or submitted code read.',
            change='Pool the SAME4M input budget across the two method stages; wall time, request counts, output budgets, scientific prompts/data unchanged except accurate input-budget wording.',
            method_input_budget_pooled=True,reuse_all_original_ordinary_repeats=True,
            native_sessions=4,comparison_native_sessions_including_cached_ordinary=6,
            maximum_concurrent_episodes=2,maximum_total_requests=120,
            baseline_reuse='Both original ordinary repeats completed with exactly the current common information and1800s/60requests/4M input budget. Reuse BOTH without reading/choosing scores; do not recount their model usage.',
            original_failure_retained=True,no_further_recovery_after_this=True)
        for b in protocol['method_stages']:b['input_tokens']=4000000
        save(root/'RECOVERY_DECISION392.json',dict(frozen_utc=utc(),source_protocol_sha256=sha(recover_from/'PROTOCOL392.json'),
            source_budget_receipt=meta,all_final_scores_unopened=True,all_source_code_contents_uninspected=True,
            mechanism_unchanged=True,total_budget_unchanged=True,rerun_both_method_repeats=True,reuse_both_ordinary_repeats=True))
    save(root/'PROTOCOL392.json',protocol);save(root/'PROMPTS392.json',dict(common=COMMON,ordinary=ORDINARY,propose=PROPOSE,review=REVIEW))
    rows=[]
    for repeat in [1,2]:
        for arm in (['ordinary','edit_review'] if repeat==1 else ['edit_review','ordinary']):
            episode=root/'episodes'/f'{arm}_{repeat}';episode.mkdir()
            reuse=recover_from is not None and arm=='ordinary'
            stages=[] if reuse else (['ordinary'] if arm=='ordinary' else ['propose','review'])
            for stage in stages:
                run=episode/stage
                for d in ['app/.runtime/tmp','app/.runtime/mpl','runtime/tmp','runtime/mpl','logs','primary','plans','audit','ipc','scripts']:
                    (run/d).mkdir(parents=True)
                shutil.copytree(root/'public/data',run/'app/data')
                shutil.copy2(root/'public/instruction.md',run/'app/instruction.md')
                shutil.copy2(root/'reference/incumbent.py',run/'app/solution.py')
                save(run/'app/readonly_inputs.json',['data','instruction.md'])
                (run/'app/AGENTS.md').write_text('Follow the frozen task and phase budget. Real observations only. No network or model delegation. No final data. Scratch: /app/.runtime, never /tmp.\n')
                for src,dst in [('base_cli_config.toml','cli_config187.toml'),('model_catalog187.json','model_catalog187.json')]:
                    shutil.copy2(root/src,run/dst)
                shutil.copy2(root/'tools/isolated_model_gateway.py',run/'scripts/isolated_model_gateway.py')
                mounts=[(PYTHON,'/opt/python312'),(DEPS,'/opt/phm_python/extra'),(root/'tools','/tools'),(root/'reference','/reference')]
                save(run/'mounts.json',dict(mounts=[dict(source=str(a),destination=b) for a,b in mounts]))
                budget=dict(seconds=1800,requests=60,input_tokens=4000000,output_tokens=131072) if stage=='ordinary' else protocol['method_stages'][0 if stage=='propose' else 1]
                save(run/'STAGE_BUDGET.json',budget)
                save(run/'MODEL_GATEWAY_CONFIG.json',dict(routes=[dict(job=run.parent.name+'_'+stage,socket=str(run/'ipc/s.sock'))],base_url='https://api.deepseek.com/',model='deepseek-flash',max_requests=budget['requests'],max_input_tokens=budget['input_tokens'],max_output_tokens=budget['output_tokens'],max_output_per_request=16384,max_inflight=1,denied_tool_namespaces=['collaboration'],denied_tool_names=['spawn_agent','followup_task','interrupt_agent','list_agents','send_message','wait_agent'],buffer_verified_responses=True))
                prompt=COMMON+'\n'+dict(ordinary=ORDINARY,propose=PROPOSE,review=REVIEW)[stage]
                if recover_from is not None:
                    prompt=prompt.replace('2500000input tokens','4000000input tokens shared across BOTH stages').replace('1500000input tokens','REMAINING_SHARED_INPUTinput tokens')
                    prompt+='\nThe TWO stages share one4000000-input-token allowance. Any stage1 input usage is deducted before stage2 starts; the episode total is identical to ordinary coding. Other stage limits remain as stated.\n'
                (run/'plans/prompt.txt').write_text(prompt)
            row=dict(id=episode.name,arm=arm,repeat=repeat,stages=stages,path=str(episode))
            if reuse:
                source=recover_from/'episodes'/episode.name
                prior=read(source/'TERMINAL.json')
                # Reuse only sealed complete artifacts, by byte hash, without
                # reading their scientific contents or any score.
                destination=root/'sealed'/episode.name;destination.mkdir()
                for name,digest in prior['files'].items():
                    p=recover_from/'sealed'/episode.name/name;assert sha(p)==digest;shutil.copy2(p,destination/name)
                row['reuse_terminal']=dict(prior,reused_from=str(source),new_model_sessions=0,new_requests=0)
            rows.append(row)
    save(root/'SESSIONS392.json',rows);save(root/'SOURCE_BINDINGS392.json',bindings)
    frozen=[root/'PROTOCOL392.json',root/'PROMPTS392.json',root/'SESSIONS392.json',root/'SOURCE_BINDINGS392.json']+list((root/'code').glob('*.py'))+list((root/'tools').glob('*.py'))
    frozen += [p for directory in ['reference','public','sealed'] for p in (root/directory).rglob('*') if p.is_file()]
    frozen += [p for row in rows for stage in row['stages'] for p in (Path(row['path'])/stage).rglob('*') if p.is_file()]
    save(root/'CODE_INPUT_FREEZE392.json',{str(p.relative_to(root)):sha(p) for p in frozen})
    save(output_directory(root)/'PROTOCOL392.json',protocol)
    status(root,state='prepared_not_started',planned_episodes=4,planned_native_sessions=protocol['native_sessions'],completed_episodes=0,actual_running_jobs=0,final_scores_opened=False)


def arguments(run, timeout):
    return SimpleNamespace(cli=CLI,model='deepseek-flash',output=run/'primary',plan=run/'plans/unused.json',timeout=timeout,
        analysis_tools=True,permission_profile='iot-offline',isolation_config=run/'cli_config187.toml',model_gateway_socket=run/'ipc/s.sock',
        model_catalog=run/'model_catalog187.json',runtime_mounts_file=run/'mounts.json',workspace_alias='/app',credential_env_file=None,
        credential_env_name='DEEPSEEK_API_KEY',snapshot_generated_source=True,cleanup_runtime_cache=True)


def qualify(root):
    assert not (root/'QUALIFICATION392.json').exists()
    check='''import json,os,socket,importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
assert not any(Path(p).exists() for p in ['/home/jupyter-yyf-zyq/.bashrc','/home/jupyter-yyf-zyq/Bo_2/PHM_20261010','/data/yyf-zyq/PHM_20261010/20261025T162500Z_path383','/prior'])
assert not any('API_KEY' in k or 'SECRET' in k for k in os.environ)
assert not Path('/tmp').exists()
assert os.environ['TMPDIR']=='/app/.runtime/tmp'
try: socket.create_connection(('1.1.1.1',443),1); raise AssertionError('network available')
except OSError: pass
g=json.loads(Path('/app/data/reference_geometry.json').read_text())
z=np.load('/app/data/healthy_reference.npz'); r=dict(g,signals=z['signals'])
s=importlib.util.spec_from_file_location('p','/reference/incumbent.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
out=m.predict(r,z['signals'][0],g);assert set(out)=={'location_m'} and np.isfinite(out['location_m']).all()
try: Path('/app/data/reference_geometry.json').write_text('bad');raise AssertionError('writable input')
except OSError:pass
print(json.dumps(dict(passed=True,healthy_call_only=True,localization_accuracy_not_tested=True,new_model_calls=0)))
'''
    results=[]
    for row in read(root/'SESSIONS392.json'):
        for stage in row['stages']:
            run=Path(row['path'])/stage;ns=arguments(run,120);folder=run/'audit/qualification';folder.mkdir()
            fixture=socket.socket(socket.AF_UNIX);fixture.bind(str(run/'ipc/s.sock'));fixture.listen(1)
            command=isolate_filesystem([str(CLI),'sandbox','-P','iot-offline','-C','/app','--','/opt/python312/bin/python3.12','-B','-c',check],ns,run/'app',folder,None)
            env=clean_env(run);env.update(PYTHONPATH='/opt/phm_python/extra',TMPDIR='/app/.runtime/tmp',TMP='/app/.runtime/tmp',TEMP='/app/.runtime/tmp',MPLCONFIGDIR='/app/.runtime/mpl')
            try:p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
            finally:fixture.close();(run/'ipc/s.sock').unlink()
            save(folder/'RESULT.json',dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,command=command))
            results.append(dict(episode=row['id'],stage=stage,exit_code=p.returncode));assert p.returncode==0,p.stderr[-1500:]
    save(root/'QUALIFICATION392.json',dict(passed=True,rows=results,model_calls=0,healthy_smoke_calls=len(results),damage_predictions=0))
    status(root,state='qualified_not_started',actual_running_jobs=0)


def stage_run(root, row, stage, deadline):
    run=Path(row['path'])/stage;budget=read(run/'STAGE_BUDGET.json');gateway=None
    cfg=read(root/'PROTOCOL392.json')
    if cfg.get('method_input_budget_pooled') and stage=='review':
        used=read(Path(row['path'])/'propose/model_gateway'/(row['id']+'_propose')/'status.json')['input_tokens']
        remaining=cfg['episode_input_tokens']-used
        save(run/'SHARED_BUDGET_RECEIPT.json',dict(total=cfg['episode_input_tokens'],previous_actual_input=used,remaining=remaining))
        if remaining<=0:
            result=dict(status='shared_input_budget_exhausted',previous_actual_input=used)
            save(run/'STAGE_TERMINAL.json',result);return result
        gateway_cfg=read(run/'MODEL_GATEWAY_CONFIG.json');gateway_cfg['max_input_tokens']=remaining;save(run/'MODEL_GATEWAY_CONFIG.json',gateway_cfg)
        prompt=run/'plans/prompt.txt';prompt.write_text(prompt.read_text().replace('REMAINING_SHARED_INPUT',str(remaining)))
        save(run/'EFFECTIVE_STAGE_INPUT_FREEZE.json',dict(prompt_sha256=sha(prompt),gateway_config_sha256=sha(run/'MODEL_GATEWAY_CONFIG.json')))
    available=min(budget['seconds'],max(1,deadline-time.monotonic()))
    save(run/'EPISODE_BUDGET.json',dict(start_epoch=time.time(),deadline_epoch=time.time()+available,stage=stage))
    try:
        gateway=start_gateway(run)
        job=dict(id=row['id']+'_'+stage,condition=row['arm'],repeat=row['repeat'],workspace=str(run/'app'),prompt_file=str(run/'plans/prompt.txt'),answer_mode='text')
        result=replay(job,arguments(run,max(1,available-2)))
    except Exception as e:
        result=dict(status='controller_error',error_type=type(e).__name__,error=str(e))
    finally:
        if gateway is not None:
            (run/'MODEL_GATEWAY_STOP').touch()
            try:gateway.wait(timeout=30)
            except subprocess.TimeoutExpired:gateway.terminate();gateway.wait(timeout=10)
    save(run/'STAGE_TERMINAL.json',result)
    return result


def execute(root,row):
    episode=Path(row['path']);assert not (episode/'STARTED.json').exists()
    if 'reuse_terminal' in row:
        save(episode/'TERMINAL.json',row['reuse_terminal']);return row['reuse_terminal']
    start=time.monotonic();save(episode/'STARTED.json',dict(pid=os.getpid(),utc=utc()))
    states=[]
    for stage in row['stages']:
        if stage=='review':
            prior=episode/'propose/app';current=episode/'review/app'
            frozen=episode/'review/frozen_proposal';frozen.mkdir()
            copied=[]
            # Preserve actual visible artifacts, excluding runtime caches and immutable public inputs.
            for p in sorted(prior.rglob('*')):
                relative=p.relative_to(prior)
                if p.is_symlink() or not p.is_file() or relative.parts[0] in ['.runtime','data','AGENTS.md','instruction.md','readonly_inputs.json']:
                    continue
                assert p.stat().st_size < 64*1024**2, 'Unexpected artifact size; retain failure rather than silently truncate'
                for target in [frozen/relative,current/relative]:
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
                copied.append(dict(path=str(relative),sha256=sha(p),bytes=p.stat().st_size))
            save(episode/'HANDOFF.json',dict(files=copied,first_stage_finished=states[-1].get('status')=='finished',hidden_feedback=False))
            mounts=read(episode/'review/mounts.json');mounts['mounts'].append(dict(source=str(frozen),destination='/prior'));save(episode/'review/mounts.json',mounts)
        if time.monotonic()>=start+1800:break
        states.append(stage_run(root,row,stage,start+1800))
    final=episode/row['stages'][-1]/'app';sealed=root/'sealed'/row['id'];sealed.mkdir()
    for name in ['solution.py','report.md','proposal.json','review_manifest.json']:
        if (final/name).is_file():shutil.copy2(final/name,sealed/name)
    valid=len(states)==len(row['stages']) and all(s.get('status')=='finished' for s in states) and all((sealed/n).is_file() for n in ['solution.py','report.md'])
    result=dict(id=row['id'],arm=row['arm'],repeat=row['repeat'],valid=valid,stages=states,elapsed_seconds=time.monotonic()-start,files={p.name:sha(p) for p in sealed.iterdir()},final_queries_read=False)
    save(episode/'TERMINAL.json',result);return result


def run(root):
    assert read(root/'QUALIFICATION392.json')['passed'];assert not (root/'HANDLE392.json').exists()
    for path,digest in read(root/'CODE_INPUT_FREEZE392.json').items():assert sha(root/path)==digest,path
    save(root/'HANDLE392.json',dict(pid=os.getpid(),utc=utc(),state='running'))
    rows=read(root/'SESSIONS392.json');results=[]
    concurrency=read(root/'PROTOCOL392.json')['maximum_concurrent_episodes']
    status(root,state='native_coding_running',controller_pid=os.getpid(),actual_running_jobs=concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures={pool.submit(execute,root,row):row for row in rows}
        for f in as_completed(futures):
            row=futures[f]
            try:result=f.result()
            except Exception as e:result=dict(id=row['id'],arm=row['arm'],repeat=row['repeat'],valid=False,error_type=type(e).__name__,error=str(e))
            results.append(result);save(root/'PROGRESS392.json',results)
            remaining=sum('reuse_terminal' not in j and j['id'] not in {r['id'] for r in results} for j in rows)
            status(root,state='native_coding_running',completed_episodes=len(results),actual_running_jobs=min(concurrency,remaining))
            print(json.dumps(dict(completed=len(results),id=result['id'],valid=result['valid'])),flush=True)
    save(root/'ALL_SUBMISSIONS_SEALED392.json',dict(utc=utc(),results=results,protocol_sha256=sha(root/'PROTOCOL392.json'),final_queries_read=False))
    save(root/'TERMINAL392.json',dict(exit_code=0,episodes=4,valid=sum(r['valid'] for r in results)))
    status(root,state='sealed_ready_for_external_evaluation',completed_episodes=4,actual_running_jobs=0,final_scores_opened=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','qualify','run']);p.add_argument('--run',type=Path,required=True);p.add_argument('--recover-from',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.run,a.recover_from)
    else:globals()[a.action](a.run)
