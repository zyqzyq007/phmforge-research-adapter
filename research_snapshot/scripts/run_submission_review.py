"""Single-context, submission-triggered counterevidence; bounded native pilot.

Uses the established real-training runtime and isolated native CLI. No new
scoring shortcut, outcome veto, synthetic data, or second model context.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import uuid

from controlled_feedback import save, sha, utc
from native_validation_comparison import closure
import free_program_comparison as coding
import local_program_revision as prior

PROJECT=Path('/home/jupyter-yyf-zyq/Bo_2/PHM_20261010')
SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261025T012600Z_local347')
OUT=PROJECT/'results/submission396'
read=lambda p:json.loads(Path(p).read_text())

COMMON='''Improve or retain the supplied genuinely trained PHM diagnosis PROGRAM for unseen operating conditions. You have a single ordinary coding-Agent context and may write arbitrary complete feature extraction, classifier, and input-observable decision logic. Read /data/metadata.json, /reference/source/program.py and phmforge info. Public train and development recordings are measured laboratory data. All supervised fitting, including learned gates, must occur through register_program; do not fit unregistered models locally. The tool freezes the whole declared source, genuinely trains two complete-condition held-out models plus the full model, and cold-evaluates the complete program, including its final prediction rules. Four authored registrations are available, in addition to p00. Every changed version must be registered separately. No synthetic waveforms, artificial fault labels, constant/mock training or hidden-label access.

The shared optional probe evaluates all not-yet-checked registered versions on the same four legal check acquisitions. Reuse is adaptive development evidence, not independent certification. The same paired_evidence and review_update tools are available to both policies. They expose actual per-acquisition harms and repairs, exact source/checkpoint identities, and other candidates that may offer useful gains. A fold checkpoint and the full deployment checkpoint are different validation objects. Neither visible validation performance nor program execution guarantees accuracy on unknown conditions. Avoid harmful replacement while retaining useful improvements; retaining p00 is allowed but is not automatically best.

Final arrays, conditions, labels, scores, other Agent workspaces and the Internet are inaccessible. No delegation or other model calls. Each prediction receives one acquisition with no label and read-only fitted artifacts; no fitting during prediction or routing by sample/group identity or batch class counts. Use reproducible seeds and at most two numerical threads. Write under /app and use /app/.runtime for scratch, never /tmp. Submit one registered program with phmforge submit and save report.md with actual choices, evidence and uncertainty. A review_required response is NOT a sealed submission. The current info response states whether acknowledgement of a review receipt is mandatory. You may revise or retain any candidate after review; there is no score-based veto or forced keep rule. A new registration/check invalidates a prior review receipt. The same tools are callable directly using python3 /tools/program_revision_tools.py client --request if MCP discovery fails.

Total budget:1800seconds,60model requests,4000000input tokens,131072output tokens,32768output tokens per request. Up to4new registrations,each<=450seconds with each fit<=120seconds and each record prediction<=45seconds. All tool work and any review acknowledgement share this total budget. Complete the registration work early enough to leave at least the final5requests for deciding, writing report.md and confirming the sealed submit response. Do not start optional additional work when the remaining budget is needed to finish. No extra context, quota recovery or post-submission model call is supplied.
'''
POLICIES={
 'ordinary_coding':'Policy: ordinary coding. Organize your own investigation and review. You may request review_update at any time but no submission review is compulsory.',
 'submission_review':'Policy: submission-triggered evidence review. Before an actual seal, submit returns a machine-produced comparison of the chosen complete program against p00 and a common-validation alternative. Read the real repairs AND harms, separate fold and full-model evidence, and check the cost of rejecting an improvement. In this SAME context, decide to keep the choice or revise/register a complete program. Final submit must acknowledge the current review_id and explain the decision; the tool does not dictate which candidate wins.'}


def bound_copy(base,relative,target,bindings):
    receipt=read(base/'ARCHIVE_VERIFIED.json');manifest=base/'SOURCE_MANIFEST.json'
    assert sha(manifest)==receipt['manifest_sha256']
    index={r['path']:r for r in read(manifest)['files']}
    source=base/relative;assert sha(source)==index[relative]['sha256']
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    bindings.append(dict(archive=str(base),member=relative,sha256=sha(target),target=str(target)))


def prepare(root):
    assert not root.exists(),'Unique run required'
    for d in ['runtime/tmp','runtime/mpl','audit','code','common_tools','scripts','logs','public_data','private_inputs','reference/p00']:(root/d).mkdir(parents=True)
    old=read(SOURCE/'PROTOCOL337.json');bindings=[]
    for name in ['base_cli_config.toml','model_catalog187.json','retention_policy.json']:
        bound_copy(SOURCE,name,root/name,bindings)
    for role in ['train','development','independent_check']:
        bound_copy(SOURCE,'private_inputs/'+role+'.npz',root/'private_inputs'/(role+'.npz'),bindings)
    for name in ['train.npz','development.npz','independent_check_observations.npz','metadata.json']:
        bound_copy(SOURCE,'public_data/'+name,root/'public_data'/name,bindings)
    report=read(SOURCE/'reference/p00/REPORT.json')
    for member in ['REPORT.json',*report['artifacts']]:
        bound_copy(SOURCE,'reference/p00/'+member,root/'reference/p00'/member,bindings)
    for name in [p.name for p in (SOURCE/'common_tools').glob('*.py')]:
        shutil.copy2(PROJECT/'scripts'/name,root/'common_tools'/name)
    url='https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-research'}),timeout=30) as response:release=json.load(response)
    assert release['tag_name']=='rust-v0.164.0' and sha(Path(old['cli']))==old['cli_sha256']
    save(root/'audit/CLI_SOURCE396.json',dict(url=url,latest_tag=release['tag_name'],published_at=release['published_at'],checked_utc=utc(),cli=old['cli'],sha256=old['cli_sha256']))
    cfg=dict(old,iteration='A396',method_version='submission-triggered complete-program counterevidence',
        purpose='Test whether an enforced same-context evidence challenge at the actual submission decision improves complete-program choice.',
        arms=['ordinary_coding','submission_review'],repeats=2,planned_native_sessions=4,maximum_parallel_sessions=4,
        episode_seconds=1800,episode_requests=60,episode_input_tokens=4000000,episode_output_tokens=131072,output_per_request=32768,
        max_registered_programs=4,prompt_file='PROMPTS396.json',build_stage_only=True,stage='single_context_final_choice',
        enable_paired_evidence=True,probe_reusable=True,maximum_new_request_budget=240,maximum_fit_entry_budget=51,
        maximum_disk_GiB=6,mechanism='A submit attempt triggers actual paired counterevidence and a current-bank receipt acknowledgement in the SAME native context; no outcome veto.',
        primary='Complete four episodes, then lower mean final acquisition error and no greater mean per-acquisition harm/no lower mean per-acquisition gain than ordinary; also strictly beat keep.',
        secondary=['same-common-union-bank descriptive selection loss','gain capture on common union','native completion','input/output/time cost','receipt acknowledgement and evidence use'],
        factor='Mandatory submit-time challenge;both arms share the exact same callable tool,legal data,initial program,training limits and total budget.',
        rivals=['Same information was already sufficient without a forced challenge','Extra interaction hurts completion','A challenge only causes conservative retention','Candidate generation variability rather than choice explains differences'],
        intervention='Mandatory current-bank evidence acknowledgement at submit;identical tool remains voluntarily callable in ordinary coding.',
        rival='Ordinary complete validation already suffices;forced attention may add cost without changing a correct choice.',
        design_file='docs/submission_review396_protocol.md',maximum_final_cold_prediction_processes=272,
        novelty='Submission validation gates are prior art,including SIGA. Potential PHM effect of simultaneous harm/opportunity and complete-program evidence is unproven;not a novelty claim.',
        pilot_only=True,independent_confirmation_scenarios=0,known_development_cohorts=1,
        no_outcome_based_rerun=True,no_further_quota_recovery=True,baseline_reference_reused_real_fit_entries=3,
        post_submission_model_calls=0,all_final_evaluation_after_all4choices_sealed_and_controller_terminal=True)
    save(root/'PROTOCOL337.json',cfg);save(root/'PROTOCOL396.json',cfg)
    save(root/'LOCAL_REVISION_PROTOCOL347.json',cfg)
    save(root/'PROMPTS396.json',{arm:COMMON+'\n'+POLICIES[arm] for arm in cfg['arms']})
    engine=read(SOURCE/'ENGINE_CONFIG.json');engine.update(input_root=str(root/'private_inputs'),common_tools=str(root/'common_tools'))
    save(root/'ENGINE_CONFIG.json',engine)
    coding.OUT=OUT;coding.prepare(root)
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path']);bc=read(run/'BROKER_CONFIG.json')
        bc.update(enable_submit_review=True,submission_review_required=row['arm']=='submission_review');save(run/'BROKER_CONFIG.json',bc)
        config=run/'cli_config187.toml';text=config.read_text()
        assert 'PHM_PAIRED_EVIDENCE = "1"' in text
        config.write_text(text.replace('PHM_PAIRED_EVIDENCE = "1"','PHM_PAIRED_EVIDENCE = "1", PHM_SUBMIT_REVIEW = "1"'))
        discovery=read(run/'PROTOCOL187.json');discovery['expected_mcp_tools'].append('mcp__phmforge__review_update');save(run/'PROTOCOL187.json',discovery)
    for name in closure(['run_submission_review','local_program_revision']):shutil.copy2(PROJECT/'scripts'/(name+'.py'),root/'code'/(name+'.py'))
    # The compiler copied the current broker; freeze again after all additions.
    for folder in ['code','common_tools']:
        assert sha(root/folder/'program_revision_tools.py')==sha(PROJECT/'scripts/program_revision_tools.py')
    save(root/'EFFECTIVE_CODE_FREEZE337.json',{str(p.relative_to(root)):sha(p) for d in ['code','common_tools','scripts'] for p in (root/d).glob('*.py')})
    save(root/'SOURCE_BINDINGS396.json',bindings)
    # Reserved sources are identified by old manifests and role definitions;
    # neither their observations nor their task performance is read here.
    cm=read(PROJECT/'results/confirmsource338/RECOVERY_ARCHIVE_VERIFIED.json')
    bat=read(PROJECT/'results/data365/ARCHIVE_VERIFIED.json')
    panel=dict(frozen_utc=utc(),method_version='A396 qualified single-context intervention,exact code hash fixed separately',
        admission='Execute only after a complete developmental pilot and an explicit unchanged-method admission record; otherwise keep sealed.',
        cohorts=[dict(id='COMFAULDA_'+k,source_archive=cm['archive_path'],manifest_sha256=cm['manifest_sha256'],roles='A338 unchanged train/development/check/final',physical_cluster='COMFAULDA_rig') for k in ['c1','c2','c3']]
                +[dict(id='HUST_battery_official22',source_archive=bat['archive_path'],manifest_sha256=bat['manifest_sha256'],roles='A36547train/8development/22officialtest unchanged;cycle500',physical_cluster='HUST_battery_cohort')],
        source_clusters=2,cohorts_count=4,cluster_aggregation='Equal mean of3 mechanical cohorts is one cluster;22 battery cells aggregate into one cluster. Repeats and windows never add wins.',
        repeats_per_arm=2,arms=['ordinary_coding','submission_review'],future_native_sessions=16,
        primary='Strictly win more than50percent of the2predeclared source clusters (therefore2/2),plus joint harm/gain and common-bank selection criteria. Four cohort counts also reported;cannot claim broad population superiority from2clusters.',
        failure_policy='No score-based exclusions or substitutions;all incomplete/missing tasks remain. A failed pilot does not authorize opening confirmation scores.',
        final_data_read=False,performance_opened=False,confirmation_not_started=True,
        limitations='Two source clusters are a bounded confirmation panel,not proof over allPHMdomains;novelty and mechanism controls still required for goal completion.')
    save(root/'CONFIRMATION_PANEL396.json',panel);save(PROJECT/'results/program_update_goal330/CONFIRMATION_PANEL396.json',panel)
    save(root/'LITERATURE396.json',dict(utc=utc(),sources=[
        dict(url='https://arxiv.org/html/2606.09774v2',title='Auto-Configuring Scientific Simulators with Lightweight Coding-Agent Adapters',scope='June25v2:agent-callablevalidationandvalidation-gatedtermination alreadypriorart;scientificconfiguration notPHM predictivequality.'),
        dict(url='https://arxiv.org/html/2610.03969v1',title='How Do Coding Agents Optimize Software and Report Performance Validation?',scope='October2preprint:validationpresence doesnotestablishintendedperformanceeffect;not a PHM intervention result.'),
        dict(url='https://arxiv.org/html/2607.28871',scope='Cross-state validation evidence is already studied;our mandatory decision challenge not claimed first.')],novelty_established=False))
    freeze=[root/'PROTOCOL396.json',root/'PROMPTS396.json',root/'CONFIRMATION_PANEL396.json',root/'SOURCE_BINDINGS396.json',root/'EFFECTIVE_CODE_FREEZE337.json']
    freeze += [p for row in read(root/'SESSIONS337.json') for p in Path(row['path']).rglob('*') if p.is_file()]
    save(root/'ADMISSION_FREEZE396.json',{str(p.relative_to(root)):sha(p) for p in freeze})
    save(OUT/'PROTOCOL396.json',cfg);save(OUT/'current_run.json',dict(run=str(root),status='prepared_not_started',new_model_calls=0,planned_native_sessions=4,goal_complete=False))
    print(json.dumps(dict(prepared=True,source_bindings=len(bindings),native_sessions=4,confirmation_clusters_reserved=2,confirmation_data_read=False)))


def preflight(root):
    """Actual source/checkpoint replay, fresh fitting and receipt invalidation."""
    from program_revision_tools import Broker
    assert not (root/'PREFLIGHT396.json').exists()
    run=root/'qualification_subject'
    for d in ['app/.runtime/tmp','app/.runtime/feedback_requests','feedback','broker','logs']:(run/d).mkdir(parents=True)
    shutil.copy2(root/'ENGINE_CONFIG.json',run/'ENGINE_CONFIG.json')
    save(run/'BROKER_CONFIG.json',dict(baseline=str(root/'reference/p00'),contract=str(root/'public_data/metadata.json'),max_programs=1,
        enable_paired_evidence=True,probe_reusable=True,enable_submit_review=True,submission_review_required=True))
    broker=Broker(run)
    first=broker.dispatch(dict(action='submit',program_id='p00',reason='Real tool qualification only'))
    assert first['status']=='review_required' and broker.state['submission'] is None
    old_id=first['review']['review_id']
    for p in (root/'reference/p00/source').rglob('*'):
        if p.is_file():q=run/'app'/p.relative_to(root/'reference/p00/source');q.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,q)
    files=[str(p.relative_to(root/'reference/p00/source')) for p in (root/'reference/p00/source').rglob('*') if p.is_file()]
    fitted=broker.dispatch(dict(action='register_program',files=files,claim='Exact real baseline copy,qualification only',rival='End-to-end source/report wiring could be broken'))
    assert fitted['status']=='complete' and fitted['confirmed_estimator_fits']==3
    stale=broker.dispatch(dict(action='submit',program_id='p00',review_id=old_id,reason='Check stale receipt rejection'))
    assert stale['status']=='review_required' and broker.state['submission'] is None
    new_id=stale['review']['review_id']
    actual=broker.dispatch(dict(action='probe'));assert set(actual['results'])=={'p00','p01'}
    changed=broker.dispatch(dict(action='submit',program_id='p00',review_id=new_id,reason='Check new evidence invalidates receipt'))
    assert changed['status']=='review_required' and broker.state['submission'] is None
    current=broker.dispatch(dict(action='review_update',program_id='p01'))
    assert all(v['repairs']==v['harms']==0 for p in current['comparisons'] for v in p['views'])
    accepted=broker.dispatch(dict(action='submit',program_id='p01',review_id=current['review_id'],reason='Actual clone is valid;qualification verifies no forced keep policy'))
    assert accepted['program_id']=='p01' and broker.state['submission'] is not None
    # Also verify nontrivial repair/harm arithmetic from a genuine historical
    # complete program. No historical final score or code enters Actor inputs.
    q=root/'qualification_counterevidence';(q/'broker').mkdir(parents=True);(q/'app').mkdir()
    shutil.copy2(root/'ENGINE_CONFIG.json',q/'ENGINE_CONFIG.json');shutil.copy2(run/'BROKER_CONFIG.json',q/'BROKER_CONFIG.json')
    b=Broker(q);historic='subjects/ordinary_coding_2/programs/p04';target=q/'historical/p02'
    report=read(SOURCE/historic/'REPORT.json');bindings=[]
    for member in ['REPORT.json',*report['artifacts']]:bound_copy(SOURCE,historic+'/'+member,target/member,bindings)
    b.state['programs']['p02']=dict(path=str(target),report=report);b.persist()
    challenge=b.dispatch(dict(action='review_update',program_id='p02'))
    changes=sum(v['repairs']+v['harms'] for p in challenge['comparisons'] for v in p['views'])
    assert changes>0
    save(q/'SOURCE_BINDINGS.json',bindings)
    # The ordinary arm has the same review tool but can submit directly.
    b.config['submission_review_required']=False
    direct=b.dispatch(dict(action='submit',program_id='p00',reason='Ordinary policy transport qualification'))
    assert direct['program_id']=='p00'
    audits=[read(p) for p in run.rglob('training_audit.json')]
    result=dict(passed=True,utc=utc(),real_fit_entries=sum(x['mode']=='fit' for x in audits),actual_prediction_processes=sum(x['mode']=='predict' for x in audits),
        actual_estimator_fits=sum(x['completed_estimator_fits'] for x in audits),first_submit_unsealed=True,
        registration_and_probe_invalidate_receipt=True,accepting_new_program_permitted=True,retention_receives_opportunity_check=True,
        genuine_historical_changed_outcomes=changes,ordinary_direct_submit_permitted=True,new_LLM_calls=0,final_data_read=False)
    save(root/'PREFLIGHT396.json',result);save(OUT/'PREFLIGHT396.json',result);print(json.dumps(result))


def qualify(root):
    assert read(root/'PREFLIGHT396.json')['passed']
    coding.OUT=OUT;coding.qualify(root)
    for row in read(root/'SESSIONS337.json'):
        run=Path(row['path']);saved=run/'audit/pre_scientific_transport';saved.mkdir()
        for relative in ['broker/STATE.json','feedback/responses','app/.runtime/feedback_requests']:
            p=run/relative
            if p.exists():
                shutil.move(str(p),str(saved/relative.replace('/','_')))
                if p.suffix!='.json':p.mkdir(parents=True)
    save(root/'QUALIFIED396.json',dict(passed=True,utc=utc(),all4isolation_passed=True,same6MCPtools=True,real_training_and_receipt_chain=True,scientific_state_clean=True,new_LLM_calls=0))
    print(json.dumps(read(root/'QUALIFIED396.json')))


def agents(root):
    assert read(root/'QUALIFIED396.json')['passed']
    for relative,digest in read(root/'ADMISSION_FREEZE396.json').items():
        # Native qualification writes only its own audit/transport artifacts.
        assert sha(root/relative)==digest,(relative,'changed before admission')
    coding.OUT=OUT;coding.agents(root)


def evaluate(root):
    prior.OUT=OUT;prior.SOURCE=Path(read(root/'PROTOCOL396.json')['source_archive'])
    prior.evaluate(root)
    result=read(root/'RESULT347.json');result['iteration']='A396';result['scope']='One previously opened development cohort;new native programs,not independentconfirmation.'
    base=result['all_program_results']['common_p00'];base_records={r['group']:r for r in base['records']}
    all_scored=all(v['status']=='complete' for v in result['all_program_results'].values())
    common_best=min(v['record_error'] for v in result['all_program_results'].values() if v['status']=='complete')
    common_opportunity=result['baseline_error']-common_best
    for episode in result['episodes']:
        state=read(root/'subjects'/episode['id']/'broker/STATE.json')
        ids=list(state['programs'])
        combined=min(ids,key=lambda k:((sum(v['record_error'] for v in state['programs'][k]['report']['cv'])/len(state['programs'][k]['report']['cv'])+state['programs'][k]['report']['development']['record_error'])/2,k!='p00',k))
        value=result['all_program_results']['common_p00' if combined=='p00' else episode['id']+'__'+combined]
        episode['methods']['equal_CV_development']=dict(program_id=combined,evaluation_status=value['status'])
        if value['status']=='complete':episode['methods']['equal_CV_development'].update(record_error=value['record_error'],harm=max(0.,value['record_error']-result['baseline_error']),gain=result['baseline_error']-value['record_error'])
        for choice in episode['methods'].values():
            if choice['evaluation_status']!='complete':continue
            name='common_p00' if choice['program_id']=='p00' else episode['id']+'__'+choice['program_id']
            rows=result['all_program_results'][name]['records'];n=len(rows)
            choice['record_harm']=sum(base_records[r['group']]['actual']==base_records[r['group']]['predicted'] and r['actual']!=r['predicted'] for r in rows)/n
            choice['record_gain']=sum(base_records[r['group']]['actual']!=base_records[r['group']]['predicted'] and r['actual']==r['predicted'] for r in rows)/n
            choice['common_union_selection_loss']=choice['record_error']-common_best
            choice['common_union_gain_capture']=max(0.,result['baseline_error']-choice['record_error'])/common_opportunity if common_opportunity>0 else None
        episode['submission_review_receipts']=len(list((root/'subjects'/episode['id']/'broker/submission_reviews').glob('*.json')))
        episode['acknowledged_review']=bool((state.get('submission')or{}).get('review_id'))
    joint={}
    for arm in read(root/'PROTOCOL396.json')['arms']:
        rows=[r for r in result['episodes'] if r['arm']==arm];valid=all(r['completed'] and r['methods'].get('actor',{}).get('evaluation_status')=='complete' for r in rows)
        joint[arm]=dict(valid=valid,planned=len(rows),mean_error=sum(r['methods']['actor']['record_error'] for r in rows)/len(rows) if valid else None,
            mean_record_harm=sum(r['methods']['actor']['record_harm'] for r in rows)/len(rows) if valid else None,
            mean_record_gain=sum(r['methods']['actor']['record_gain'] for r in rows)/len(rows) if valid else None,
            mean_common_union_selection_loss=sum(r['methods']['actor']['common_union_selection_loss'] for r in rows)/len(rows) if valid else None,
            mean_common_union_gain_capture=sum(r['methods']['actor']['common_union_gain_capture'] for r in rows)/len(rows) if valid and common_opportunity>0 else None)
    a,b=joint['ordinary_coding'],joint['submission_review']
    passed=bool(all_scored and a['valid'] and b['valid'] and b['mean_error']<a['mean_error'] and b['mean_error']<result['baseline_error'] and b['mean_record_harm']<=a['mean_record_harm'] and b['mean_record_gain']>=a['mean_record_gain'])
    result.update(joint_comparison=joint,joint_pilot_gate_passed=passed,confirmation_data_opened=False,goal_complete=False,
                  common_union_best_error=common_best,common_union_opportunity=common_opportunity,all_actual_candidates_scored=all_scored,
                  union_scope='Descriptive shared evaluator reference only;Actors did not exchange their generated candidates. Primary comparison remains same-start final loss/harm/gain.')
    save(root/'RESULT396.json',result);save(OUT/'RESULT396.json',result);save(root/'TERMINAL396.json',dict(utc=utc(),status='scored',joint_gate=passed))
    print(json.dumps(dict(joint_comparison=joint,joint_pilot_gate=passed,cost=result['cost'])))


def finish(root):
    """Connect a live frozen pilot to its evaluator without a manual idle gap.

    This coordinator cannot change the native run, retry it, or launch a model.
    It executes the already frozen scientific evaluator and audit exactly once.
    """
    import os
    import sys
    marker=root/'FINISH_HANDLE396.json'
    with marker.open('x') as stream:
        json.dump(dict(pid=os.getpid(),status='waiting_for_native_terminal',utc=utc()),stream)
    try:
        assert not (root/'RESULT396.json').exists()
        plan=read(root/'AUDIT_PLAN396.json');audit=Path(plan['script'])
        assert sha(audit)==plan['sha256']
        while True:
            handle=read(root/'HANDLE337.json')
            alive=Path('/proc',str(handle['pid'])).exists()
            if handle['status']=='terminal' and not alive:break
            if not alive and handle['status']!='terminal':raise RuntimeError('Native controller exited without a terminal seal')
            time.sleep(3)
        sealed=read(root/'ALL_CHOICES_SEALED337.json');assert len(sealed['episodes'])==4
        assert not (root/'FINAL_STARTED347.json').exists(), 'Do not duplicate external evaluation'
        save(marker,dict(pid=os.getpid(),status='external_evaluation',utc=utc()))
        print(json.dumps(dict(phase='external_evaluation',utc=utc())),flush=True)
        subprocess.run([sys.executable,'-B','-u',str(root/'code/run_submission_review.py'),'evaluate','--run',str(root)],check=True)
        save(marker,dict(pid=os.getpid(),status='independent_recount',utc=utc()))
        assert sha(audit)==plan['sha256']
        subprocess.run([sys.executable,'-B','-u',str(audit),'submission','--run',str(root)],check=True)
        save(marker,dict(pid=os.getpid(),status='terminal',utc=utc(),manual_trace_audit_pending=True))
    except BaseException as error:
        save(marker,dict(pid=os.getpid(),status='error',utc=utc(),error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','qualify','agents','evaluate','finish']);p.add_argument('--run',required=True,type=Path);a=p.parse_args();globals()[a.action](a.run.resolve())
