"""Freeze A/C architecture comparison; reuse exactly the verified A325 bank."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import urllib.request

from controlled_feedback import save, sha, utc
from run_competing_hypotheses import PROJECT, COMMON, ROLES

SOURCE=Path('/data/yyf-zyq/PHM_20261010/20261024T103209Z_hypotheses325')
MANIFEST_SHA='ec2b43dc89d0b779156ad65425718531eccdfccab9d8c146ae701d27bce1ecb4'


def main(root,recovery=False):
    root.mkdir(exist_ok=False)
    for d in ['runtime/tmp','runtime/mpl','logs','audit','code','shared_tools','hidden']:(root/d).mkdir(parents=True)
    assert sha(SOURCE/'SOURCE_RECOVERY_MANIFEST.json')==MANIFEST_SHA
    manifest=json.loads((SOURCE/'SOURCE_RECOVERY_MANIFEST.json').read_text())
    files={r['path']:r for r in manifest['files']}
    old=json.loads((SOURCE/'CAMPAIGN_CONFIG.json').read_text())
    config=dict(old,iteration='A326',frozen_utc=utc(),run=str(root),result_directory='results/architectures326',
        authorization='用户要求：尝试AC路线并结合B路线的结果总结一下。',arms=['A','C'],
        scope='Retrospective fixed-candidate architecture comparison on already opened A325 data; agents remain blinded.',
        reuse_existing_bank=True,reuse_source=str(SOURCE),reuse_source_manifest_sha256=MANIFEST_SHA,
        researcher_previously_knows_final_scores=True,confirmation_experiment=False,
        all_choices_sealed_before_final=False,all_new_choices_sealed_before_score_join=True,
        maximum_cli_stage_sessions=48,maximum_parallel_episodes=2,
        A_stages=[dict(role=r,seconds=s) for r,s in [('builder_proposal',300),('reviewer_plan',480),('builder_checks',780),('reviewer_decision',240)]],
        C_stages=[dict(role=r,seconds=s) for r,s in [('builder_proposal',300),('coordinator_dispatch',120),('investigation',240),('coordinator_plan',240),('builder_checks',660),('coordinator_decision',240)]],
        C_parallel_branches=3,C_shared_model_inflight_limit=3,actual_new_PHM_fits=0,
        evaluation_policy='After all8 choices are sealed, independently rescore the hash-bound original final predictions. No new fitting or final model forwards.',
        input_token_budget_policy='Same cumulative provider-usage admission gate as A325; up to3 in-flight requests in C. Report actual usage and any overrun; output tokens reserved atomically.',
        comparison_kind='Architecture package comparison, not a single-component causal ablation.',
        scheduler_caveat='A/C max2 simultaneous episodes vs A325 max4; wall time descriptive, API usage more comparable.',
        failure_policy='All8 episodes retained including missing handoffs, timeouts and budget failures; no adaptive retries.',
        interpretation='One measured motor, two related acquisition blocks, two choices per block/arm. Final results already known to researcher; architecture-only exploratory reuse, not new confirmation.')
    if recovery:
        config.update(iteration='A327',result_directory='results/architectures327',arms=['C'],episodes=4,
            maximum_cli_stage_sessions=32,enforce_external_roles=True,
            comparison_parent_run='/srv/yyf-zyq/PHM_20261010/20261024T114710Z_routes326',
            comparison_parent_archive='/data/yyf-zyq/PHM_20261010/20261024T114710Z_routes326',
            engineering_recovery=True,
            recovery_reason='A326 native collaboration remained exposed despite config/CLI disabling, allowing duplicate branch spawning. Preserve all A326 C failures/cancellations. Block nested delegation at gateway, explicitly delegate dispatch to host, and use exact handoff paths. No PHM choice rule or evidence hint changes.',
            C_stages=[dict(role=r,seconds=s) for r,s in [('builder_proposal',300),('coordinator_dispatch',120),('investigation',480),('coordinator_plan',240),('builder_checks',420),('coordinator_decision',240)]],
            failure_policy='All4 recovery episodes retained, plus all A326 original C attempts separately. No further adaptive retries.',
            input_token_budget_policy='Shared cumulative admission gate; up to3 in-flight requests. Output atomically reserved. Buffer responses for prohibited-call checking and preserve provider usage after client timeout.')
    suffix=config['iteration'][1:]
    url='https://api.github.com/repos/openai/codex/releases/latest'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PHM-research-provenance'}),timeout=30) as response:
        release=json.load(response)
    assert release['tag_name']=='rust-v0.164.0','New stable CLI found; resolve before freezing comparison'
    assert sha(Path(config['cli']))==config['cli_sha256']
    save(root/'audit/CLI_SOURCE.json',dict(url=url,checked_utc=utc(),tag=release['tag_name'],
        published_at=release['published_at'],html_url=release['html_url'],cli_sha256=config['cli_sha256']))
    save(root/'CAMPAIGN_CONFIG.json',config)
    # All reuse inputs must match the prior verified immutable archive.
    prefixes=('bank/','hidden/block78/train.npz','hidden/block78/val.npz','hidden/block910/train.npz','hidden/block910/val.npz')
    selected=[name for name in files if name.startswith(prefixes)]
    selected+=['base_cli_config.toml','model_catalog187.json','shared_tools/rotor_feature_recipe.py']
    copied=[]
    for name in selected:
        src=SOURCE/name;dest=root/name
        assert sha(src)==files[name]['sha256']
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        assert sha(dest)==files[name]['sha256']
        copied.append(dict(path=name,sha256=files[name]['sha256'],bytes=dest.stat().st_size))
    assert len(list((root/'bank').glob('*/*/fold_*.joblib')))==60
    assert not list((root/'hidden').glob('*/test.npz'))
    # These numerical functions and the acceptance gate are identical to B.
    def funcs(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    before=funcs(SOURCE/'code/controlled_program_checks.py');after=funcs(PROJECT/'scripts/controlled_program_checks.py')
    invariant=['accepted_choice','replay_check','metrics','paired','compact','designs','estimator','feature_kind']
    assert all(before[k]==after[k] for k in invariant)
    code=['run_competing_hypotheses.py','controlled_program_checks.py','controlled_feedback.py','isolated_model_gateway.py',
        'replay_diagnostic_cli.py','run_validation_transfer.py','run_full_benchmark_campaign.py','qualify_native_cli_mcp.py',
        'prepare_competing_architectures.py','audit_competing_architectures.py','evaluate_competing_hypotheses.py']
    for name in code:shutil.copy2(PROJECT/'scripts'/name,root/'code'/name)
    for name in ['controlled_program_checks.py','controlled_feedback.py','isolated_model_gateway.py']:
        shutil.copy2(PROJECT/'scripts'/name,root/'shared_tools'/name)
    save(root/'PROMPT_FREEZE.json',dict(common=COMMON,roles={k:ROLES[k] for k in ['builder_proposal','builder_checks','reviewer_plan','reviewer_decision',
        'coordinator_dispatch','investigation','coordinator_plan','coordinator_decision']},derived_from='Pre-existing A/B/C architecture design',no_final_outcome_hints_added=True))
    proof=dict(passed=True,protocol_sha256=sha(root/'CAMPAIGN_CONFIG.json'),actual_component_fits=60,
        actual_new_fits=0,actual_new_agent_sessions=0,researcher_previously_knows_final_scores=True,
        agent_final_data_available=False,source=str(SOURCE),manifest_sha256=MANIFEST_SHA,
        copied=copied,unchanged_numerical_functions=invariant,code_hashes={name:sha(root/'code'/name) for name in code},
        all60_checkpoint_bytes_identical=True)
    save(root/('PREPARATION'+suffix+'.json'),proof)
    policy=json.loads((SOURCE/'retention_policy.json').read_text())
    policy['final_results_not_yet_opened_at_policy_creation']=False
    policy['socket_cleanup']='After all handlers stop, remove inactive Unix socket nodes, record every removal; no regular evidence deleted.'
    save(root/'retention_policy.json',policy)
    save(root/'OWNERSHIP.json',dict(user_request=config['authorization'],created_utc=utc(),run=str(root),archive='/data/yyf-zyq/PHM_20261010/'+root.name))
    out=PROJECT/config['result_directory'];out.mkdir(parents=True,exist_ok=True)
    save(out/('PROTOCOL'+suffix+'.json'),config);save(out/('PREPARATION'+suffix+'.json'),proof)
    save(out/'current_run.json',dict(status='prepared_pending_engineering_audit',run=str(root),new_agent_sessions=0))
    print(json.dumps(dict(run=str(root),verified_files=len(copied),new_fits=0,planned_episodes=config['episodes'],maximum_sessions=config['maximum_cli_stage_sessions'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--recovery-C',action='store_true',dest='recovery')
    a=p.parse_args();main(a.run.resolve(),a.recovery)
