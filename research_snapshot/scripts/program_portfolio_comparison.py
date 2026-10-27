"""Bounded independent coding branches, reusing the qualified PHM runner.

The experiment changes generation context, not data or the composition rule.
All model and training work is charged to the complete chain. This is a
development pilot; the independent confirmation panel stays closed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

import numpy as np

from controlled_feedback import save,sha,utc
from native_validation_comparison import closure
from program_revision_tools import Broker
from program_revision_runtime import Runtime,load,score
from run_submission_review import bound_copy
import free_program_comparison as coding
import local_program_revision as evaluator

PROJECT=coding.PROJECT
OUT=PROJECT/'results/portfolio399'
SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261025T215000Z_submit396')
read=lambda p:json.loads(Path(p).read_text())

COMMON='''Improve or retain the supplied actually trained four-class PHM diagnosis PROGRAM for unseen operating conditions. Read /data/metadata.json, /reference/source/program.py and phmforge info. Use measured training/development acquisitions and the optional labeled development-check tool. These are few acquisitions from one laboratory rig, not independent machines. Final observations, conditions, counts, labels, scores and other studies are unavailable.

Write arbitrary genuine feature extraction and supervised classification programs. All supervised candidate fitting MUST use register_program(files,claim,rival); every edit requires a new registration. The service freezes source, actually fits TWO whole-condition leave-out models and ONE full model, cold-evaluates development and opaque-ID invariance. No unregistered fitting, synthetic observations/labels, constant or mock learning. Each prediction receives one acquisition without labels and with read-only fitted weights. No sample/group-ID or class-count rules. Use fixed seeds and <=2 numerical threads; source may use the optional /tools/phm_program_features.py or your own genuine features. Match the starting fit/predict interface.

The shared tools info,register_program,probe,paired_evidence,review_update,compose_programs,submit are available to both workflows. Probe reuses the same legal check records on exact registered programs; adaptive development evidence is not independent certification. Paired evidence/review reports complete-program repairs AND harms. Fold models and the full checkpoint are different models. compose_programs can combine 2-5 distinct trained base programs by equal whole-acquisition vote, with ties preferring your declared tie_anchor member then smallest class. It reuses genuine weights with zero new fitting; it cold-executes development twice and binds actual out-of-fold predictions. Nested compositions are unavailable. All members share the SAME45second per-acquisition prediction budget. Choose a single program or a real composition; neither is guaranteed better.

Consider the same competing explanations: apparent improvement may be condition-specific amplitude discrimination; a changed representation may capture transferable fault structure; or the small validation cohort may be insufficient to decide. Explain discriminating legal evidence and contrary outcomes without forcing a hypothesis. Minimize unseen whole-acquisition classification error, avoid harmful replacement, and preserve useful improvements. Keeping p00 is allowed but not automatically best. Submit a registered ID with a reason; do not claim a final score. Write report.md describing actual evidence and uncertainty. Only the immutable submit response seals the decision. No network, other model APIs or agent spawning. Scratch is /app/.runtime, never/tmp. The fallback client python3 /tools/program_revision_tools.py client --request accepts the same action JSON.

Whole-chain upper limits are1800summed stage seconds,60model requests,4000000input admission threshold,131072output tokens,32768output per request,4new supervised registrations and2composition attempts. Each program/composition<=450seconds; each fit<=120seconds; each acquisition prediction<=45seconds. Input usage is known after each reply; the final admitted reply can exceed the input admission threshold. All closing model calls count. Reserve time to submit before polishing prose. Numeric remaining budgets and stage-specific limits below are authoritative.
'''

ROLES={
 'ordinary':'''You are the ordinary single-context coding baseline. Organize your own investigation, generate up to4new complete programs, optionally compose up to2times, review and submit. Time1800seconds and the complete-chain model budget are yours. No review is mandatory.''',
 'branch_a':'''You are an independent complete-program builder, already launched by the controller. Work from the common initial evidence without seeing another branch. Up to2new registrations,675seconds; no composition attempts in this stage. Investigate the common competing explanations and produce genuinely useful complete programs. Submit your preferred registered base program and write report.md. A subsequent selector receives all actual programs and evidence. Both builders share50requests,3000000input admission threshold and98304output tokens in total; these are not per-builder allowances. Do not wait for or contact another branch.''',
 'branch_b':'''You are an independent complete-program builder, already launched by the controller. Work from the common initial evidence without seeing another branch. Up to2new registrations,675seconds; no composition attempts in this stage. Investigate the common competing explanations and produce genuinely useful complete programs. Submit your preferred registered base program and write report.md. A subsequent selector receives all actual programs and evidence. Both builders share50requests,3000000input admission threshold and98304output tokens in total; these are not per-builder allowances. Do not wait for or contact another branch.''',
 'selector':'''You are the final selector of the complete independent-branch chain. Read /app/handoff.json and the two /app/inputs branch reports/source snapshots, then info. Both builders have terminated. All actually registered base programs are available under the IDs in info. No new supervised registrations in this stage; up to2real compositions are allowed. You may probe/review and submit any registered single program or composition. There is no fixed scoring veto. Stage450seconds,at most10requests/1000000input admission threshold/32768output tokens, reduced if necessary to respect the total chain budget. Submit before polishing report.md.'''
}


def prepare(root):
    assert not (root/'PROTOCOL399.json').exists()
    assert not root.exists() or set(p.name for p in root.iterdir())<={'runtime'}
    for d in ['runtime/tmp','runtime/mpl','code','common_tools','logs','scripts','audit','public_data','private_inputs','reference/p00']:(root/d).mkdir(parents=True,exist_ok=True)
    bindings=[];old=read(SOURCE/'PROTOCOL337.json')
    for name in ['base_cli_config.toml','model_catalog187.json','retention_policy.json']:
        bound_copy(SOURCE,name,root/name,bindings)
    for role in ['train','development','independent_check']:
        bound_copy(SOURCE,'private_inputs/'+role+'.npz',root/'private_inputs'/(role+'.npz'),bindings)
    for name in ['train.npz','development.npz','independent_check_observations.npz','metadata.json']:
        bound_copy(SOURCE,'public_data/'+name,root/'public_data'/name,bindings)
    baseline=read(SOURCE/'reference/p00/REPORT.json')
    for name in ['REPORT.json',*baseline['artifacts']]:bound_copy(SOURCE,'reference/p00/'+name,root/'reference/p00'/name,bindings)
    for p in (SOURCE/'common_tools').glob('*.py'):shutil.copy2(PROJECT/'scripts'/p.name,root/'common_tools'/p.name)
    url='https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-research'}),timeout=30) as response:release=json.load(response)
    assert release['tag_name']=='rust-v0.164.0' and sha(Path(old['cli']))==old['cli_sha256']
    save(root/'audit/CLI_SOURCE399.json',dict(url=url,official_page=release['html_url'],latest_tag=release['tag_name'],published_at=release['published_at'],checked_utc=utc(),cli=old['cli'],sha256=old['cli_sha256']))
    cfg=dict(iteration='A399',frozen_utc=utc(),cli=old['cli'],cli_sha256=old['cli_sha256'],model='deepseek-flash',
        arms=['ordinary','branch_a','branch_b','selector'],repeats=2,planned_native_stages=8,planned_complete_chains=4,
        episode_seconds=1800,episode_requests=60,episode_input_tokens=4000000,episode_output_tokens=131072,output_per_request=32768,
        max_registered_programs=4,program_seconds=450,fit_seconds=120,predict_seconds=45,numerical_threads=2,
        prompt_file='PROMPTS399.json',enable_paired_evidence=True,probe_reusable=True,build_stage_only=True,defer_independent_check=False,
        source_archive=old['source_archive'],source_version='Same qualified A346 measured cohort;previously opened developmental material,not independent confirmation.',
        source_record_roles_Hz=old['source_record_roles_Hz'],max_compositions_per_chain=2,maximum_new_requests=240,maximum_new_fit_entries=48,
        maximum_disk_GiB=24,maximum_parallel_native_stages=6,maximum_parallel_upstream_requests=4,
        stage_seconds=dict(ordinary=1800,branch_a=675,branch_b=675,selector=450),
        shared_builder_model_budget=dict(requests=50,input_tokens=3000000,output_tokens=98304,max_inflight=1),
        selector_model_budget=dict(requests=10,input_tokens=1000000,output_tokens=32768),
        budget_policy='Two builders share ONE serial-accounted gateway. Selector input allowance=min(1M,4M-actual builder input);if exhausted,chain remains incomplete. At most one final admitted reply can overshoot the chain threshold. All closing calls counted;no recovery.',
        factor='Independent complete coding contexts with equal chain resources and shared legal information/tools.',
        mechanism='Potential reduction of correlated candidate errors caused by continuous editing;unproven.',
        primary='All4chains complete;method mean final acquisition error strictly below ordinary and keep,record harms no greater and record gains no lower.',
        attribution='Additionally beat ordinary-generated bank aggregated by the identical equal record-vote rule, ties prefer ordinary sealed prediction if tied. This fixed control uses exact cold individual outputs;not an additional Agent or fitting run.',
        fixed_controls=['keep','development','mean_cv','equal_CV_development','full_checkpoint_validation','same_bank_equal_vote'],
        failures='Keep every timeout,code error and missing submission. No outcome-based rerun,quota rescue or exclusion. Failures make primary gate fail.',
        source_clusters=1,known_development_cohorts=1,independent_confirmation_scenarios=0,
        confirmation338_performance_unopened=True,HUST_final22_unopened=True,novelty_established=False,
        novelty_note='Independent candidate generation and ensembling have prior art;this is a mechanism pilot,not a novelty claim.',
        composite_protocol='Host-sealed relocatable manifest binds real base sources/checkpoints;0newfits.2-5members,non-nested,shared45second record cap.',
        no_further_quota_recovery=True)
    save(root/'PROTOCOL399.json',cfg);save(root/'PROTOCOL337.json',cfg)
    save(root/'PROMPTS399.json',{arm:COMMON+'\nCURRENT STAGE\n'+ROLES[arm] for arm in cfg['arms']})
    save(root/'ENGINE_CONFIG.json',dict(input_root=str(root/'private_inputs'),common_tools=str(root/'common_tools'),fit_seconds=120,predict_seconds=45,program_seconds=450))
    coding.OUT=OUT;coding.prepare(root)
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path']);role=row['arm'];bc=read(run/'BROKER_CONFIG.json')
        bc.update(enable_submit_review=True,submission_review_required=False,enable_composition=True,
            max_compositions=0 if role.startswith('branch_') else 2,max_programs=2 if role.startswith('branch_') else 0 if role=='selector' else 4)
        save(run/'BROKER_CONFIG.json',bc)
        p=run/'cli_config187.toml';p.write_text(p.read_text().replace('PHM_PAIRED_EVIDENCE = "1"','PHM_PAIRED_EVIDENCE = "1", PHM_SUBMIT_REVIEW = "1", PHM_COMPOSITION = "1"'))
        protocol=read(run/'PROTOCOL187.json');protocol['expected_mcp_tools']+=['mcp__phmforge__review_update','mcp__phmforge__compose_programs'];save(run/'PROTOCOL187.json',protocol)
    for repeat in [1,2]:
        chain=root/'chains'/('portfolio_'+str(repeat))
        for d in ['ipc','logs','runtime/tmp','runtime/mpl','scripts']:(chain/d).mkdir(parents=True)
        gateway=read(root/'subjects'/('ordinary_'+str(repeat))/'MODEL_GATEWAY_CONFIG.json')
        gateway.update(routes=[dict(job='builders',socket=str(chain/'ipc/s.sock'))],max_requests=50,max_input_tokens=3000000,max_output_tokens=98304,max_inflight=1,concurrent_queue_seconds=650)
        save(chain/'MODEL_GATEWAY_CONFIG.json',gateway)
        shutil.copy2(PROJECT/'scripts/isolated_model_gateway.py',chain/'scripts/isolated_model_gateway.py')
        for role in ['branch_a','branch_b']:
            run=root/'subjects'/(role+'_'+str(repeat));bc=read(run/'BROKER_CONFIG.json');bc['budget_root']=str(chain);save(run/'BROKER_CONFIG.json',bc)
        run=root/'subjects'/('selector_'+str(repeat));g=read(run/'MODEL_GATEWAY_CONFIG.json');g.update(max_requests=10,max_input_tokens=1000000,max_output_tokens=32768);save(run/'MODEL_GATEWAY_CONFIG.json',g)
    for name in closure(['program_portfolio_comparison']):shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    save(root/'SOURCE_BINDINGS399.json',bindings)
    save(root/'OWNERSHIP.json',dict(run=str(root),archive='/data/yyf-zyq/PHM_20261010/'+root.name,created_utc=utc(),goal='A330',scope='A399developmentpilot'))
    save(root/'EFFECTIVE_CODE_FREEZE337.json',{str(p.relative_to(root)):sha(p) for d in ['code','common_tools','scripts'] for p in (root/d).glob('*.py')})
    save(root/'PREPARATION399.json',dict(utc=utc(),passed=True,actual_new_fits=0,actual_new_LLM_calls=0,reused_real_baseline_fits=3,source_bindings=len(bindings)))
    save(OUT/'PROTOCOL399.json',cfg);save(OUT/'current_run.json',dict(run=str(root),status='prepared_not_started',planned_native_stages=8,new_LLM_calls=0,goal_complete=False))
    print(json.dumps(read(root/'PREPARATION399.json')),flush=True)


def preflight(root):
    assert not (root/'PREFLIGHT399.json').exists();run=root/'composition_qualification'
    for d in ['broker','app/.runtime/tmp','runtime/tmp','runtime/mpl']:(run/d).mkdir(parents=True)
    shutil.copy2(root/'ENGINE_CONFIG.json',run/'ENGINE_CONFIG.json')
    save(run/'BROKER_CONFIG.json',dict(baseline=str(root/'reference/p00'),contract=str(root/'public_data/metadata.json'),max_programs=0,
        enable_composition=True,max_compositions=1,enable_paired_evidence=True,enable_submit_review=True,submission_review_required=False))
    b=Broker(run);bindings=[]
    for ident in ['p01','p02']:
        relative='subjects/ordinary_coding_1/programs/'+ident;report=read(SOURCE/relative/'REPORT.json');target=run/'parents'/ident
        for member in ['REPORT.json',*report['artifacts']]:bound_copy(SOURCE,relative+'/'+member,target/member,bindings)
        b.state['programs'][ident]=dict(path=str(target),report=report)
    b.persist();save(run/'SOURCE_BINDINGS.json',bindings)
    result=b.dispatch(dict(action='compose_programs',program_ids=['p00','p01','p02'],tie_anchor='p01',reason='Real qualified member composition,interface check only.'))
    assert result['status']=='complete',result
    assert result['fit_entry_executions']==0 and result['actual_cold_prediction_processes']==48
    joined=b.dispatch(dict(action='paired_evidence',incumbent='p00',challenger='e01'));assert len(joined['views'])==3
    reviewed=b.dispatch(dict(action='review_update',program_id='e01'));assert reviewed['program_id']=='e01'
    submitted=b.dispatch(dict(action='submit',program_id='e01',reason='Qualification only,not an Agent decision.'));assert submitted['program_id']=='e01'
    marker=run/'programs/e01/COMPOSITION.json';original=marker.read_bytes()
    try:
        marker.write_bytes(original+b' ')
        try:b.engine.verified_members(marker.parent)
        except AssertionError:tamper_rejected=True
        else:raise AssertionError('Unbound composition accepted')
    finally:marker.write_bytes(original)
    b.engine.verified_members(marker.parent)
    audits=[read(p) for p in (run/'programs/e01').rglob('training_audit.json')]
    assert len(audits)==48 and all(a['mode']=='predict' and a['completed_estimator_fits']==a['optimizer_steps']==0 for a in audits)
    value=dict(passed=True,utc=utc(),actual_new_fits=0,new_LLM_calls=0,actual_prediction_processes=48,
        member_report_bound_CV=True,cold_development_and_identity=True,paired_review_submit_chain=True,tamper_rejected=tamper_rejected,final_data_read=False)
    save(root/'PREFLIGHT399.json',value);save(OUT/'PREFLIGHT399.json',value);print(json.dumps(value),flush=True)


def qualify(root):
    assert read(root/'PREFLIGHT399.json')['passed'];coding.OUT=OUT;coding.qualify(root)
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path']);saved=run/'audit/pre_scientific_transport';saved.mkdir()
        for rel in ['broker/STATE.json','feedback/responses','app/.runtime/feedback_requests']:
            p=run/rel
            if p.exists():
                shutil.move(str(p),str(saved/rel.replace('/','_')))
                if p.suffix!='.json':p.mkdir(parents=True)
    paths=[root/'PROTOCOL399.json',root/'PROTOCOL337.json',root/'PROMPTS399.json',root/'EFFECTIVE_CODE_FREEZE337.json',root/'SOURCE_BINDINGS399.json']
    paths += [p for row in read(root/'SESSIONS337.json') for p in Path(row['path']).glob('*.json') if p.name!='BROKER_READY.json']
    paths += [p for p in (root/'subjects').glob('*/cli_config187.toml')]+list((root/'chains').glob('*/MODEL_GATEWAY_CONFIG.json'))
    save(root/'ADMISSION_FREEZE399.json',{str(p.relative_to(root)):sha(p) for p in paths})
    save(root/'QUALIFIED399.json',dict(passed=True,utc=utc(),native_stages=8,same7MCPtools=True,real_composition_passed=True,independent_workspaces=True,actual_model_calls=0))
    print(json.dumps(read(root/'QUALIFIED399.json')),flush=True)


def stop_gateway(run,gateway):
    (run/'MODEL_GATEWAY_STOP').touch()
    try:gateway.wait(timeout=650)
    except subprocess.TimeoutExpired:gateway.terminate();gateway.wait(timeout=10);raise


def stage(root,row,gateway_socket=None):
    run=Path(row['path']);role=row['arm'];cfg=read(root/'PROTOCOL399.json');broker=None;gateway=None;started=time.monotonic()
    result=dict(id=row['id'],arm=role,repeat=row['repeat'],completed=False)
    try:
        with (run/'logs/broker.log').open('w') as log:
            broker=subprocess.Popen([str(coding.PYTHON/'bin/python3.12'),'-B','-u',str(root/'code/program_revision_tools.py'),'serve','--run',str(run)],env=coding.environment(root),stdout=log,stderr=subprocess.STDOUT)
        for _ in range(200):
            if (run/'BROKER_READY.json').exists():break
            assert broker.poll() is None;time.sleep(.1)
        assert (run/'BROKER_READY.json').exists()
        seconds=cfg['stage_seconds'][role];save(run/'EPISODE_BUDGET.json',dict(start_epoch=time.time(),deadline_epoch=time.time()+seconds,stage=role))
        if gateway_socket is None:gateway=coding.start_gateway(run)
        ns=coding.arguments(root,run);ns.timeout=seconds
        if gateway_socket is not None:ns.model_gateway_socket=gateway_socket
        job=dict(id=row['id'],condition=role,repeat=row['repeat'],workspace=str(run/'app'),prompt_file=str(run/'plans/prompt.txt'),answer_mode='text')
        result['native']=coding.replay(job,ns)
    except Exception as error:result.update(error_type=type(error).__name__,error=str(error))
    finally:
        if broker is not None:
            broker.terminate()
            try:broker.wait(timeout=10)
            except subprocess.TimeoutExpired:broker.kill();broker.wait()
        if gateway is not None:stop_gateway(run,gateway)
    state=read(run/'broker/STATE.json');submission=state.get('submission')
    result.update(submission=submission,actor_submission_valid=submission is not None,
        completed=result.get('native',{}).get('status')=='finished' and submission is not None,
        seconds=time.monotonic()-started,final_scores_joined=False,registered_programs=sorted(state['programs']))
    save(run/'STAGE_RESULT399.json',result);return result


def portfolio(root,repeat,rows):
    chain=root/'chains'/('portfolio_'+str(repeat));gateway=coding.start_gateway(chain);started=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda role:stage(root,rows[role],chain/'ipc/s.sock'),['branch_a','branch_b']))
    finally:stop_gateway(chain,gateway)
    selector=Path(rows['selector']['path']);b=Broker(selector);handoff=[];checks={};all_builder_checks=True
    for role,result in zip(['branch_a','branch_b'],results):
        run=Path(rows[role]['path']);state=read(run/'broker/STATE.json');mapping={'p00':'p00'}
        for old,v in state['programs'].items():
            if old=='p00':continue
            ident=('a_' if role=='branch_a' else 'b_')+old;mapping[old]=ident;b.state['programs'][ident]=v
            source=selector/'app/inputs'/role/ident/'source';shutil.copytree(Path(v['path'])/'source',source)
        for name in ['report.md']:
            p=run/'app'/name
            if p.exists():q=selector/'app/inputs'/role/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        packet=state.get('probe')
        if packet and packet['status']=='complete':checks.update({mapping[k]:v for k,v in packet['reports'].items()})
        else:all_builder_checks=False
        handoff.append(dict(branch=role,completed=result['completed'],registered_ids=mapping,preferred_id=mapping.get((state.get('submission')or{}).get('program_id'))))
    if checks:b.state['probe']=dict(status='complete',program_ids=sorted(checks),reports=checks,adaptive_validation_reuse=True,utc=utc())
    b.persist();save(selector/'app/handoff.json',dict(branches=handoff,all_branches_terminal=True,peer_workspaces_unmounted_during_generation=True,
        failed_branches_not_dropped=True,full_programs_and_real_reports_merged=True))
    usage=read(chain/'model_gateway/builders/status.json');g=read(selector/'MODEL_GATEWAY_CONFIG.json')
    g['max_input_tokens']=max(0,min(1000000,4000000-usage['input_tokens']));save(selector/'MODEL_GATEWAY_CONFIG.json',g)
    save(chain/'SELECTOR_RESIDUAL_BUDGET.json',dict(utc=utc(),builder_usage=usage,selector_budget=g,derived_by_frozen_rule=True))
    final=stage(root,rows['selector']);all_stages=results+[final]
    entry=dict(final,id=rows['selector']['id'],arm='portfolio',completed=all(r['completed'] for r in all_stages),stages=all_stages,
        seconds=time.monotonic()-started,summed_stage_seconds=sum(r['seconds'] for r in all_stages),builder_budget_shared=True)
    save(chain/'CHAIN_RESULT399.json',entry);return entry


def agents(root):
    assert read(root/'QUALIFIED399.json')['passed'];assert not (root/'ALL_CHOICES_SEALED337.json').exists()
    for rel,digest in read(root/'ADMISSION_FREEZE399.json').items():assert sha(root/rel)==digest,rel
    for rel,digest in read(root/'EFFECTIVE_CODE_FREEZE337.json').items():assert sha(root/rel)==digest,rel
    save(root/'HANDLE337.json',dict(pid=os.getpid(),status='running',phase='native_coding',utc=utc()))
    rows=read(root/'SESSIONS337.json');results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[]
        for repeat in [1,2]:
            rr={r['arm']:r for r in rows if r['repeat']==repeat}
            futures.append(pool.submit(stage,root,rr['ordinary']))
            futures.append(pool.submit(portfolio,root,repeat,rr))
        for f in as_completed(futures):
            results.append(f.result());save(root/'PROGRESS399.json',results)
            print(json.dumps(dict(utc=utc(),ended_chains=len(results),latest_id=results[-1]['id'],completed=results[-1]['completed'])),flush=True)
    save(root/'ALL_CHOICES_SEALED337.json',dict(utc=utc(),episodes=results,final_scores_joined=False,code_freeze_sha256=sha(root/'EFFECTIVE_CODE_FREEZE337.json'),protocol_sha256=sha(root/'PROTOCOL399.json')))
    save(root/'HANDLE337.json',dict(pid=os.getpid(),status='terminal',phase='choices_sealed',utc=utc()))
    save(OUT/'current_run.json',dict(run=str(root),status='choices_sealed',completed_chains=sum(r['completed'] for r in results),goal_complete=False))


def evaluate(root, iteration='399', arms=None):
    arms = ['ordinary', 'portfolio'] if arms is None else arms
    assert len(arms) == 2
    assert not (root/('RESULT'+iteration+'.json')).exists()
    cfg=read(root/('PROTOCOL'+iteration+'.json'));evaluation_cfg=dict(cfg,arms=arms);save(root/'LOCAL_REVISION_PROTOCOL347.json',evaluation_cfg)
    evaluator.OUT=OUT;evaluator.SOURCE=Path(cfg['source_archive']);evaluator.evaluate(root)
    result=read(root/'RESULT347.json');values=result['all_program_results'];base={r['group']:r for r in values['common_p00']['records']}
    union_best=min(v['record_error'] for v in values.values() if v['status']=='complete')
    for episode in result['episodes']:
        run=root/'subjects'/episode['id'];state=read(run/'broker/STATE.json')
        lookup=lambda k:values['common_p00' if k=='p00' else episode['id']+'__'+k]
        if 'equal_CV_development' in cfg.get('controls', []):
            combined = min(state['programs'], key=lambda k: (
                (sum(x['record_error'] for x in state['programs'][k]['report']['cv']) /
                 len(state['programs'][k]['report']['cv']) + state['programs'][k]['report']['development']['record_error']) / 2,
                k != 'p00', k))
            value = lookup(combined)
            choice = dict(program_id=combined, evaluation_status=value['status'])
            if value['status'] == 'complete':
                choice.update(record_error=value['record_error'], harm=max(0., value['record_error'] - result['baseline_error']), gain=result['baseline_error'] - value['record_error'])
            episode['methods']['equal_CV_development'] = choice
        for choice in episode['methods'].values():
            if choice['evaluation_status']!='complete':continue
            records=lookup(choice['program_id'])['records'];n=len(records)
            choice.update(record_harm=sum(base[r['group']]['actual']==base[r['group']]['predicted'] and r['actual']!=r['predicted'] for r in records)/n,
                record_gain=sum(base[r['group']]['actual']!=base[r['group']]['predicted'] and r['actual']==r['predicted'] for r in records)/n,
                common_union_selection_loss=choice['record_error']-union_best)
        ids=[i for i,v in state['programs'].items() if v['report'].get('program_kind')!='equal_record_vote']
        actor=episode['methods'].get('actor');control=dict(status='unavailable',source='Frozen equal vote of actual cold base-program outputs,not an extra Agent run.')
        if actor and actor['evaluation_status']=='complete' and all(lookup(i)['status']=='complete' for i in ids):
            member_maps={i:{r['group']:r['predicted'] for r in lookup(i)['records']} for i in ids}
            anchor={r['group']:r['predicted'] for r in lookup(actor['program_id'])['records']};rows=[]
            for group,truth in base.items():
                counts=np.bincount([m[group] for m in member_maps.values()],minlength=4);tied=np.flatnonzero(counts==counts.max()).tolist()
                guessed=anchor[group] if anchor[group] in tied else min(tied)
                rows.append(dict(group=group,actual=truth['actual'],predicted=int(guessed)))
            control.update(status='complete',record_error=sum(r['actual']!=r['predicted'] for r in rows)/len(rows),records=rows,
                base_programs=ids,anchor=actor['program_id'],new_fits=0,new_prediction_processes=0,additional_full_bundle_latency_not_measured=True)
        episode['same_bank_equal_vote']=control
    primary={}
    for arm in arms:
        rr=[r for r in result['episodes'] if r['arm']==arm];valid=all(r['completed'] and r['methods'].get('actor',{}).get('evaluation_status')=='complete' for r in rr)
        primary[arm]=dict(planned=len(rr),completed=sum(r['completed'] for r in rr),valid=valid,
            **{key:sum(r['methods']['actor'][key] for r in rr)/len(rr) if valid else None for key in ['record_error','record_harm','record_gain','common_union_selection_loss']},
            same_bank_vote_error=sum(r['same_bank_equal_vote']['record_error'] for r in rr)/len(rr) if all(r['same_bank_equal_vote']['status']=='complete' for r in rr) else None)
    a,b=(primary[arm] for arm in arms);all_scored=all(v['status']=='complete' for v in values.values())
    gate=bool(all_scored and a['valid'] and b['valid'] and b['record_error']<min(a['record_error'],result['baseline_error']) and b['record_harm']<=a['record_harm'] and b['record_gain']>=a['record_gain'])
    attribution=bool(gate and a['same_bank_vote_error'] is not None and b['record_error']<a['same_bank_vote_error'])
    receipts=[read(p) for p in root.glob('subjects/*/model_gateway/*/*_receipt.json')]+[read(p) for p in root.glob('chains/*/model_gateway/*/*_receipt.json')]
    requests=list(root.glob('subjects/*/model_gateway/*/*_request.json'))+list(root.glob('chains/*/model_gateway/*/*_request.json'))
    fits=[read(p) for p in root.glob('subjects/*/programs/*/**/training_audit.json') if read(p).get('mode')=='fit']
    result.update(iteration='A'+iteration,primary=primary,joint_pilot_gate_passed=gate,context_attribution_gate_passed=attribution,
        common_union_best_error=union_best,all_actual_registered_candidates_scored=all_scored,independent_confirmation_scenarios=0,goal_complete=False,
        cost=dict(actual_requests=len(requests),receipts=len(receipts),HTTP200=sum(r.get('upstream_status')==200 for r in receipts),
            returned_model_ids=sorted({m for r in receipts for m in r.get('returned_model_ids',[])}),
            input_tokens=sum(r.get('usage',{}).get('input_tokens',0) for r in receipts),output_tokens=sum(r.get('usage',{}).get('output_tokens',0) for r in receipts),
            actual_new_fit_entries=len(fits),actual_estimator_fits=sum(r['completed_estimator_fits'] for r in fits),
            actual_optimizer_steps=sum(r['optimizer_steps'] for r in fits),cold_external_prediction_processes=sum(v.get('actual_prediction_processes',0) for v in values.values())))
    save(root/('RESULT'+iteration+'.json'),result);save(OUT/('RESULT'+iteration+'.json'),result);save(root/('TERMINAL'+iteration+'.json'),dict(utc=utc(),status='scored',joint_gate=gate,attribution_gate=attribution))
    print(json.dumps(dict(primary=primary,joint_gate=gate,attribution_gate=attribution,cost=result['cost'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','qualify','agents','evaluate']);p.add_argument('--run',type=Path,required=True);a=p.parse_args();globals()[a.action](a.run.resolve())
