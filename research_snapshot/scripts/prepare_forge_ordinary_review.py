"""Outcome-blind ordinary review of one preserved, outcome-selected case.

Researcher already knows its failure. The reviewing Agent sees only the
original public inputs/submission and development evidence. This is an
exploratory same-record review, never an independent confirmation experiment.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import urllib.request

from acquire_forge_pdmb_sources244 import save,sha256
from prepare_verified_benchmarks import extract_bound
from run_validation_transfer import prepare,freeze,execute_wave,state

PROJECT=Path(__file__).resolve().parents[1]


def build(root,archive):
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    for f in ['audit','logs','code','runtime/tmp','runtime/mpl','clients-cache']:(root/f).mkdir(parents=True,exist_ok=True)
    bindings=extract_bound(archive,root/'source/previous',['c/'])
    previous=root/'source/previous/c'
    old=json.loads((previous/'audit/CLI_ONLINE_CHECK.json').read_text())
    url='https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-baseline-check'}),timeout=40) as r:
        body=r.read();http=r.status
    release=json.loads(body);assert not release['prerelease']
    assert release['tag_name']=='rust-v'+old['latest_stable_version'],'New release requires qualification before this subject'
    assert sha256(Path(old['CLI_path']))==old['CLI_sha256']
    old.update(checked_utc=datetime.now(timezone.utc).isoformat(),http_status=http,latest_response_sha256=__import__('hashlib').sha256(body).hexdigest())
    save(root/'audit/CLI_ONLINE_CHECK.json',old);(root/'audit/CLI_LATEST_RELEASE.json').write_bytes(body)
    oldjob,=json.loads((previous/'PLAN187.json').read_text())['jobs']
    oldprompt=next((previous/'plans').glob(oldjob['id']+'.txt')).read_text()
    assert 'Cost-task contract:' in oldprompt
    addendum='\nCost-task contract:'+oldprompt.split('Cost-task contract:',1)[1]
    config=dict(iteration='A311',result_directory='results/review311',planned_sessions=1,
        purpose='One outcome-blind ordinary review of the CWRU cost002 submission; outcome-selected and same-record exploratory, not new-source confirmation.',
        authorization='用户持续要求真实PHMForge实验、分析问题及保留反证；本例普通复核是候选验证所需的小规模对照。',
        test_inputs_visible=False,candidate_bank_from_training_cv=False,
        cohorts={'c':dict(family='classification',specs=[dict(id='review311_cost002',dataset='CWRU',task_id='pdm_cost_benefit_002')])},
        cases=[dict(cohort='c',dataset='CWRU',task_id='pdm_cost_benefit_002',arm='ordinary')],
        arm_prompt_addenda={'ordinary':addendum},
        session_budget=dict(subject_timeout_seconds=1800,max_model_requests_per_subject=80,
            max_total_input_tokens_per_subject=14000000,max_output_tokens_per_request=32768,
            max_total_output_tokens_per_subject=262144,mcp_tool_timeout_seconds=180,
            max_concurrency=1,local_max_concurrency=1,artifact_budget_bytes=8*1024**3),
        researcher_knows_prior_final_scores=True,final_scores_or_test_observations_provided_to_review_Agent=False,
        new_original_task_directions=0,independent_confirmation=False,
        no_automatic_retry=True,no_automatic_final_scoring=True)
    save(root/'CAMPAIGN_CONFIG.json',config);save(root/'audit/SOURCE_BINDINGS311.json',bindings)
    prepare(root,root/'c','classification','ordinary_review',previous=previous)
    alias=root/'c/restored';assert alias.is_symlink();alias.unlink()
    shutil.copytree(previous/'restored',alias)
    freeze(root/'c')
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',
        shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared',retention='Verified runtime/CLI retained; source inputs and original public submission preserved.')))
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    state(root,status='prepared_outcome_blind_ordinary_review',planned_model_sessions=1,new_model_sessions=0,
        new_original_task_directions=0,prior_case_outcome_known_only_to_researcher=True)


def build_engine_completion(root,previous):
    """One bounded completion pass for every unfinished initial Engine subject."""
    from evaluate_forge_coding import terminal,files
    from run_validation_transfer import metadata_gate
    assert not (root/'CAMPAIGN_CONFIG.json').exists()
    config=json.loads((previous/'CAMPAIGN_CONFIG.json').read_text())
    cases=[]
    for case in config['cases']:
        old=previous/case['cohort'];job,=terminal(old)
        frozen=json.loads((old/'CODING_CHOICE_AUDIT.json').read_text())['rows'][0]
        assert not (old/'CODING_SCORE.json').exists()
        assert files(old/job['workspace'])==frozen['files']
        if frozen['status']!='finished' or frozen['response_valid'] is not True:cases.append(case)
    assert cases
    for d in ['audit','logs','code','runtime/tmp','runtime/mpl','clients-cache']:(root/d).mkdir(parents=True,exist_ok=True)
    online=json.loads((previous/'audit/CLI_ONLINE_CHECK.json').read_text())
    url='https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-research'}),timeout=40) as response:
        raw=response.read();status=response.status
    assert json.loads(raw)['tag_name']=='rust-v'+online['latest_stable_version']
    assert sha256(Path(online['CLI_path']))==online['CLI_sha256']
    online.update(checked_utc=datetime.now(timezone.utc).isoformat(),http_status=status,latest_response_sha256=__import__('hashlib').sha256(raw).hexdigest())
    save(root/'audit/CLI_ONLINE_CHECK.json',online);(root/'audit/CLI_LATEST_RELEASE.json').write_bytes(raw)
    completion=dict(iteration='A317',result_directory='results/enginecompletion317',planned_sessions=len(cases),cases=cases,
        purpose='Single bounded completion of every unfinished A312 Engine session before any new final score is opened; preserve original timeouts and do not count new original directions.',
        authorization='用户要求全部可真实执行PHMForge任务；仅补齐本批未交付结果，不改变原题、数据、答案或已完成会话。',
        test_inputs_visible=False,candidate_bank_from_training_cv=False,
        cohorts={c['cohort']:dict(family='classification',specs=[dict(id='enginecompletion317_'+c['cohort'],dataset=c['dataset'],task_id=c['task_id'])]) for c in cases},
        session_budget=dict(subject_timeout_seconds=3600,max_model_requests_per_subject=180,max_total_input_tokens_per_subject=18000000,
            max_output_tokens_per_request=32768,max_total_output_tokens_per_subject=262144,mcp_tool_timeout_seconds=180,
            max_concurrency=8,local_max_concurrency=1,artifact_budget_bytes=4*1024**3),
        new_original_directions=0,independent_confirmation=False,earlier_budget_seconds=1200,
        additional_budget_seconds=3600,completion_not_a_normal_post_submission_review=True,
        all_completion_choices_before_any_initial_or_completion_final_score=True,
        no_automatic_further_retry=True,no_automatic_final_scoring=True)
    save(root/'CAMPAIGN_CONFIG.json',completion)
    for tree in ['shared_public','shared_reference']:shutil.copytree(previous/'source'/tree,root/'source'/tree)
    for case in cases:
        run=root/case['cohort'];old=previous/case['cohort'];oldjob,=terminal(old)
        oldprompt=Path(oldjob['prompt_file']).read_text();assert 'Engine source correction:' in oldprompt
        prepare(root,run,'classification','completion',previous=old)
        # Generic classification preparation omits this Engine-only helper.
        # Preserve the declared callable reference as well as its checkpoint.
        helper=old/'public_tools/real_engine_health_model.py'
        assert helper.is_file()
        shutil.copy2(helper,run/'public_tools'/helper.name)
        assert sha256(helper)==sha256(run/'public_tools'/helper.name)
        alias=run/'restored';assert alias.is_symlink();alias.unlink();shutil.copytree(old/'restored',alias)
        job,=json.loads((run/'PLAN187.json').read_text())['jobs']
        with Path(job['prompt_file']).open('a') as stream:
            stream.write('\nEngine source correction:'+oldprompt.split('Engine source correction:',1)[1])
            stream.write('\nThe earlier session reached its frozen1200second deadline before normal completion. Continue its saved work and finish the requested runnable program, validation and report. No new final predictions or scores have been computed or supplied. Retaining a justified working predictor is acceptable; additional model search is optional. This is completion of partial work on the same record,not an independent experiment. The extra budget is3600seconds and180 requests; original timeout and consumed budget remain recorded.\n')
        for key in ['mounts','evaluation_mounts']:
            path=Path(job[key]);mounts=json.loads(path.read_text())
            for item in mounts['mounts']:
                if item['destination']=='/reference':item['source']=str(root/'source/shared_reference')
                if key=='mounts' and item['destination']=='/data':item['source']=str(root/'source/shared_public')
            save(path,mounts)
        save(run/'audit/COMPLETION_LINEAGE317.json',dict(previous_run=str(old),previous_job=oldjob['id'],
            original_case=case,original_status='timeout',final_scores_read=False,new_original_direction=False))
        metadata_gate(run);freeze(run)
    shutil.copy2(__file__,root/'code'/Path(__file__).name)
    save(root/'retention_policy.json',dict(excluded_rebuildable_directory='clients-cache',shared_dependencies=dict(path='/srv/yyf-zyq/PHM_20261010/_shared',retention='Pinned runtime; same original source inputs, partial public work and genuine references retained.')))
    state(root,status='all_unfinished_initial_subjects_prepared_for_one_bounded_completion',planned_model_sessions=len(cases),
        new_model_sessions=0,new_original_directions=0,final_scores_opened=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--phase',choices=['build','run'],required=True);p.add_argument('--engine-completion',action='store_true')
    a=p.parse_args();root=a.run.resolve()
    if a.phase=='build':(build_engine_completion if a.engine_completion else build)(root,a.archive.resolve())
    else:execute_wave(root,'engine_partial_completion' if a.engine_completion else 'ordinary_review',
        [root/c['cohort'] for c in json.loads((root/'CAMPAIGN_CONFIG.json').read_text())['cases']])
