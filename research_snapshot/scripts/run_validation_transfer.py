"""Bounded paired PHM development/review campaign using existing native runners.

This controller never reads final scores. It starts three real-data subjects,
then three ordinary reviews, and stops for choice audit before external scoring.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from acquire_forge_pdmb_sources244 import save, sha256
from evaluate_forge_coding import audit, terminal, files
from evaluate_real_training_reviews import alive
from phmforge_coding252 import MODEL_CATALOG
from prepare_forge_coding_baseline import prepare as prepare_classification
from prepare_femto_coding import prepare as prepare_rul
from run_full_benchmark_campaign import clean_env

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / 'scripts'
AUTHORIZATION = '用户2026-10-21明确要求：开始真实验证！此前已讨论电机、轴承诊断和寿命预测的有界扩展。'


def utc():
    return datetime.now(timezone.utc).isoformat()


def campaign(root):
    path = root / 'CAMPAIGN_CONFIG.json'
    return json.loads(path.read_text()) if path.exists() else {}


def state(root, **changes):
    path = root / 'CAMPAIGN_STATUS.json'
    value = json.loads(path.read_text()) if path.exists() else {}
    value.update(updated_utc=utc(), **changes)
    save(path, value)
    target = campaign(root).get('result_directory', 'results/validation267')
    save(PROJECT / target / 'current_run.json', dict(value, run=str(root)))


def freeze(run):
    paths = set()
    for name in ['workspaces', 'public_data', 'references', 'public_tools', 'plans', 'evaluation_data', 'hidden']:
        paths.update(p for p in (run / name).rglob('*') if p.is_file() and not p.is_symlink())
    plan = json.loads((run / 'PLAN187.json').read_text())['jobs']
    for job in plan:
        for mount in json.loads(Path(job['mounts']).read_text())['mounts']:
            if mount['destination'] == '/data':
                paths.update(p for p in Path(mount['source']).rglob('*') if p.is_file())
    paths.update(run / name for name in ['PLAN187.json','PROTOCOL187.json', 'MODEL_GATEWAY_CONFIG.json',
        'cli_config187.toml', 'model_catalog187.json', 'TASK_QUALIFICATION251.json'])
    paths.update(SCRIPTS / name for name in ['replay_diagnostic_cli.py',
        'run_full_benchmark_campaign.py', 'run_qualified_reviews.py', 'isolated_model_gateway.py',
        'run_validation_transfer.py', 'evaluate_forge_coding.py'])
    save(run / 'INITIAL_FREEZE187_FIFO_FILES.json', {
        'frozen_utc': utc(), 'files': [{'path': str(p), 'sha256': sha256(p)} for p in sorted(paths)]})


def bootstrap(root, run, family, stage):
    for folder in ['audit', 'runtime/tmp', 'runtime/mpl', 'logs', 'code']:
        (run / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / 'audit/CLI_ONLINE_CHECK.json', run / 'audit/CLI_ONLINE_CHECK.json')
    shutil.copy2(MODEL_CATALOG, run / 'model_catalog187.json')
    configuration = dict(run_id=run.name, max_new_model_sessions=2 if family == 'classification' else 1,
        max_concurrency=8, local_max_concurrency=2 if family == 'classification' else 1,
        subject_timeout_seconds=2400, mcp_tool_timeout_seconds=180,
        max_model_requests_per_subject=120, max_total_input_tokens_per_subject=12000000,
        max_output_tokens_per_request=32768, max_total_output_tokens_per_subject=262144,
        artifact_budget_bytes=24 * 1024**3, prior_holdout_results_known_to_researcher=True,
        purpose='Prespecified development and outcome-blind ordinary review on three real PHM sources; exploratory transport test, not unseen confirmation.',
        stage=stage, no_preassumed_failure=True, original_benchmark_score_claimed=False,
        additional_gates=['OUTPUT_BUDGET_QUALIFICATION.json', 'NATIVE_CLI_MCP_QUALIFICATION.json',
                          'PUBLIC_METADATA_GATE267.json'])
    if family == 'rul':
        configuration.update(repeats=[1], job_prefix='transfer267_' + stage)
    configuration.update(campaign(root).get('session_budget', {}))
    for key in ['test_inputs_visible','candidate_bank_from_training_cv','fitting_scope']:
        if key in campaign(root):configuration[key]=campaign(root)[key]
    if campaign(root):
        count = len(campaign(root)['cohorts'][run.name.removesuffix('_review')]['specs'])
        configuration.update(max_new_model_sessions=count, local_max_concurrency=count,
            purpose=campaign(root)['purpose'])
    save(run / 'BOOTSTRAP_PROTOCOL.json', configuration)


def metadata_gate(run):
    """Reject target priors in both source descriptors and reference audits."""
    banned = {'groups_per_class', 'train_validation_test_groups', 'test_class_counts',
              'test_class_distribution', 'test_labels', 'actual_rul'}
    violations = []
    def scan(value, location):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in banned:
                    violations.append(location + ':' + key)
                scan(child, location + '/' + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan(child, location + '/' + str(index))
    checked = []
    for job in json.loads((run / 'PLAN187.json').read_text())['jobs']:
        mounts = json.loads(Path(job['mounts']).read_text())['mounts']
        for mount in mounts:
            if mount['destination'] in ['/data', '/reference']:
                for p in Path(mount['source']).rglob('*.json'):
                    scan(json.loads(p.read_text()), str(p))
                    checked.append({'path': str(p), 'sha256': sha256(p)})
    result = dict(passed=not violations, checked=checked, violations=violations,
                  whole_population_class_totals_disallowed=True, known_source_names_and_test_observations_allowed=True)
    save(run / 'PUBLIC_METADATA_GATE267.json', result)
    assert result['passed'], 'Visible target-prior metadata: ' + repr(violations)


def prepare(root, run, family, stage, previous=None):
    bootstrap(root, run, family, stage)
    if family == 'classification':
        specs = [dict(task_id='pdm_fault_015', dataset='ELECTRICMOTORVIBRATIONS',
                      id='transfer267_motor_' + stage),
                 dict(task_id='pdm_fault_011', dataset='PADERBORN',
                      id='transfer267_paderborn_' + stage)]
        if campaign(root):
            specs = [dict(row, id=row['id'] + '_' + stage) for row in
                     campaign(root)['cohorts'][run.name.removesuffix('_review')]['specs']]
        if previous:
            # Reuse only already hash-bound source inputs, never previous scores.
            (run / 'restored').symlink_to(previous / 'restored', target_is_directory=True)
        prepare_classification(run, PROJECT, specs)
    else:
        prepare_rul(run, root / 'rul/source_data')
    jobs = json.loads((run / 'PLAN187.json').read_text())['jobs']
    # AF_UNIX names are limited to 108 bytes; nested campaign paths need short IDs.
    gateway_config = json.loads((run / 'MODEL_GATEWAY_CONFIG.json').read_text())
    routes = []
    for index, job in enumerate(jobs):
        socket = run / 'ipc' / ('s' + str(index) + '.sock')
        assert len(os.fsencode(socket)) < 108
        job['model_gateway_socket'] = str(socket)
        routes.append({'job': job['id'], 'socket': str(socket)})
    gateway_config['routes'] = routes
    save(run / 'MODEL_GATEWAY_CONFIG.json', gateway_config)
    save(run / 'PLAN187.json', {'jobs': jobs})
    previous_jobs = {j['dataset']: j for j in terminal(previous)} if previous else {}
    lineage = []
    for job in jobs:
        ws = run / job['workspace']
        if previous:
            old = previous_jobs[job['dataset']]
            old_ws = previous / old['workspace']
            shutil.copytree(old_ws, ws, dirs_exist_ok=True, symlinks=True)
            old_output = previous / 'primary' / old['id'] / old['id']
            response = old_output / 'response.txt'
            if response.exists():
                shutil.copy2(response, ws / 'prior_response.txt')
            prior_status = json.loads((old_output / 'status.json').read_text())['status']
            lineage.append(dict(original_job=old['id'], review_job=job['id'],
                prior_status=prior_status, cloned_public_files=files(ws),
                final_scores_provided=False, scope='ordinary review of public submission; completion of partial work counted separately'))
        prompt = Path(job['prompt_file'])
        text = prompt.read_text()
        if previous:
            text=text.replace('No test scores, prior Agent submissions, research conclusions or original answer JSON are provided.',
                'No test scores, research conclusions or original answer JSON are provided. The prior submission for ordinary review is provided below.')
            text += '\nAn earlier Agent has left its program, artifacts and report in /app. Independently review the program and its validation using the allowed training/development information. Keep it if justified, or correct issues you find. You have no final test scores or labels. The original task and required submission interface remain unchanged. Explain your checks and any decision to retain or modify the program.\n'
        budget = json.loads((run / 'PROTOCOL187.json').read_text())
        budget_text = ('\nThe frozen session budget is {subject_timeout_seconds} seconds, at most '
                 '{max_model_requests_per_subject} model requests, {max_output_tokens_per_request} '
                 'output tokens per request, {max_total_output_tokens_per_subject} cumulative output '
                 'tokens and {max_total_input_tokens_per_subject} completed input tokens. '
                 'All comparisons and parameter selection must use allowed training/development '
                 'evidence. Test observations are readable; final labels and final scores are '
                 'unavailable. If using unlabeled batch structure, disclose that scope. Preserve '
                 'useful candidate code/artifacts and describe your actual choice; retaining '
                 'the supplied program is acceptable.\n').format(**budget)
        if not budget.get('test_inputs_visible',True):
            budget_text=budget_text.replace('Test observations are readable;',
                'Test observations are absent during development/review and are released only to the external evaluator;')
        text += budget_text
        if campaign(root):
            text += campaign(root).get('prompt_addendum', '')
            case_specs=[c for c in campaign(root).get('cases',[])
                if c['cohort']==run.name.removesuffix('_review') and c['dataset']==job['dataset']]
            if len(case_specs)==1:
                text += campaign(root).get('arm_prompt_addenda',{}).get(case_specs[0].get('arm'), '')
        prompt.write_text(text)
    if previous:
        save(run / 'audit/PAIRED_REVIEW_LINEAGE267.json', {'utc': utc(), 'rows': lineage,
            'no_new_test_scores_read': True, 'all_first_stage_terminal_before_review': True})
    save(run / 'audit/AUTHORIZATION251.json', dict(
        user_reply=campaign(root).get('authorization', AUTHORIZATION),
        scope=campaign(root).get('purpose', 'Exactly three development sessions and three ordinary reviews; no automatic retries.'),
        campaign=str(root), max_campaign_model_sessions=campaign(root).get('planned_sessions', 6)))
    metadata_gate(run)
    for name in ['run_validation_transfer.py', 'prepare_forge_coding_baseline.py',
        'prepare_femto_coding.py', 'qualify_coding_agent_isolation.py', 'qualify_native_cli_mcp.py',
        'qualify_model_output_budget.py', 'run_qualified_reviews.py', 'run_full_benchmark_campaign.py',
        'replay_diagnostic_cli.py', 'isolated_model_gateway.py', 'evaluate_forge_coding.py']:
        shutil.copy2(SCRIPTS / name, run / 'code' / name)
    for name in ['prepare_classifier_bank.py','frozen_classifier.py']:
        shutil.copy2(SCRIPTS/name,run/'code'/name)
    freeze(run)


def command(run, script, *arguments):
    with (run / 'logs' / (script + '.log')).open('ab') as output:
        result = subprocess.run([sys.executable, '-B', '-u', str(SCRIPTS / script), *map(str, arguments)],
            cwd=PROJECT, env=clean_env(run), stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(script + ' failed; inspect ' + str(run / 'logs' / (script + '.log')))


def start_gateway(run):
    with (run / 'logs/gateway.log').open('ab') as output:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(SCRIPTS / 'isolated_model_gateway.py'),
            'serve', '--run', str(run)], cwd=PROJECT, env=clean_env(run), stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(200):
        if process.poll() is not None:
            raise RuntimeError('Model gateway exited before ready')
        if (run / 'MODEL_GATEWAY_READY.json').exists():
            return process
        time.sleep(.1)
    process.terminate()
    raise TimeoutError('Gateway did not publish readiness')


def qualify(run):
    gateway = start_gateway(run)
    try:
        command(run, 'qualify_coding_agent_isolation.py', '--run', run)
        command(run, 'qualify_native_cli_mcp.py', '--run', run, '--template', run,
                '--conditions', 'fixed_discovery')
        command(run, 'qualify_model_output_budget.py', '--run', run, '--template', run)
        for name in ['NATIVE_CLI_MCP_QUALIFICATION.json', 'OUTPUT_BUDGET_QUALIFICATION.json']:
            shutil.copy2(run / 'audit' / name, run / name)
        return gateway
    except BaseException:
        (run / 'MODEL_GATEWAY_STOP').touch()
        gateway.wait(timeout=30)
        raise


def provider_account_gate(root, stage, cohorts):
    """Stop later stages after a real provider balance rejection; never retry it."""
    failures=[]
    for run in cohorts:
        for path in (run/'model_gateway').glob('*/*_receipt.json'):
            receipt=json.loads(path.read_text())
            if receipt.get('upstream_status')==402:
                failures.append(dict(cohort=run.name,receipt=str(path),upstream_status=402))
    if failures:
        save(root/'audit/PROVIDER_ACCOUNT_GATE.json',dict(utc=utc(),passed=False,
            stage=stage,failures=failures,new_model_calls_for_check=0,
            action='Preserve current attempts and stop before subsequent stages; await provider account recovery.'))
        state(root,provider_account_unavailable=True,
              blocked_condition='provider_HTTP402_insufficient_balance',
              later_stages_held_after_provider_rejection=True)
        raise RuntimeError('Provider HTTP402: account balance required; subsequent stages held, no automatic retry')


def execute_wave(root, stage, cohorts, reuse_pre_admission_qualification=False):
    state(root, status='qualifying_' + stage, current_stage=stage,
          cohorts=[str(r) for r in cohorts], final_scoring_started=False)
    gateways, controllers = [], []
    def qualified_gateway(run):
        if not reuse_pre_admission_qualification:
            return qualify(run)
        # Manual recovery is permitted only before any model subject starts.
        jobs=json.loads((run/'PLAN187.json').read_text())['jobs']
        assert all(not (run/'primary'/j['id']).exists() for j in jobs)
        for gate in ['TASK_QUALIFICATION251.json','AGENT_ISOLATION_GATE251.json',
                     'NATIVE_CLI_MCP_QUALIFICATION.json','OUTPUT_BUDGET_QUALIFICATION.json',
                     'PUBLIC_METADATA_GATE267.json']:
            assert json.loads((run/gate).read_text())['passed'],gate
        return start_gateway(run)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(qualified_gateway, run) for run in cohorts]
            errors = []
            for future in futures:
                try: gateways.append(future.result())
                except Exception as error: errors.append(str(error))
            if errors: raise RuntimeError('; '.join(errors))
        for run in cohorts:
            with (run / 'logs/controller.log').open('ab') as output:
                controllers.append(subprocess.Popen([sys.executable, '-B', '-u',
                    str(SCRIPTS / 'run_qualified_reviews.py'), '--run', str(run)], cwd=PROJECT,
                    env=clean_env(run), stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True))
        state(root, status='running_' + stage, controller_pids=[p.pid for p in controllers])
        while any(p.poll() is None for p in controllers):
            progress = []
            for run in cohorts:
                path = run / 'CONTROLLER_STATUS251.json'
                if path.exists():
                    row = json.loads(path.read_text())
                    progress.append({k: row.get(k) for k in ['planned', 'pending', 'active_jobs', 'completed', 'waiting', 'status']})
            state(root, stage_progress=progress)
            time.sleep(5)
        assert all(p.returncode == 0 for p in controllers), 'A cohort controller failed; no automatic rerun'
        for run in cohorts:
            terminal(run)
            audit(run, reuse_existing=True)
        state(root, status=stage + '_terminal_frozen', completed_stage=stage)
        provider_account_gate(root,stage,cohorts)
    finally:
        # Never close a channel while its authorized subjects are still running.
        if not controllers or all(p.poll() is not None for p in controllers):
            # A failed controller may leave already-launched CLI children alive.
            # Their own frozen deadlines still apply; preserve the model channel.
            while any(alive(json.loads(path.read_text()).get('cli_pid'))
                      for run in cohorts for path in (run / 'primary').glob('*/*/status.json')
                      if json.loads(path.read_text()).get('status') == 'running'):
                state(root, status='controller_stopped_waiting_for_authorized_subjects',
                      gateway_kept_open_for_live_subjects=True)
                time.sleep(5)
            for run in cohorts:
                (run / 'MODEL_GATEWAY_STOP').touch()
            for gateway in gateways:
                gateway.wait(timeout=30)


def main(root):
    config = campaign(root)
    sessions = config.get('planned_sessions', 6)
    state(root, status='preparing_initial', supervisor_pid=os.getpid(), planned_model_sessions=sessions,
          task_count=sessions // 2, automatic_retries=False, hidden_scores_read=False)
    families = ['classification', 'rul']
    first = [root / 'classification', root / 'rul']
    if config:
        first = [root / name for name in config['cohorts']]
        families = [row['family'] for row in config['cohorts'].values()]
    second = [root / (r.name + '_review') for r in first]
    try:
        for run, family in zip(first, families):
            prepare(root, run, family, 'initial')
        execute_wave(root, 'initial', first)
        # All initial choices are fixed before any ordinary-review subject starts.
        for run, family, prior in zip(second, families, first):
            prepare(root, run, family, 'review', prior)
        execute_wave(root, 'review', second)
        state(root, status='all_sessions_terminal_waiting_choice_review_and_external_scores',
              completed_model_sessions=sessions, no_final_scores_opened=True,
              next_action='Read frozen choices, register mechanism contrasts, then score stages externally and archive.')
    except BaseException as error:
        state(root, status='controller_error_inspect_before_resume', error_type=type(error).__name__,
              error=str(error), automatic_retries=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    assert not (root / 'CAMPAIGN_STATUS.json').exists(), 'No automatic restart of a started campaign'
    main(root)
