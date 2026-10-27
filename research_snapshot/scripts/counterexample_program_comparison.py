"""A404: single-context error-directed generation, equal native budgets.

Reuses the actual training, sandbox, MCP, composition, and evaluation runners.
This is a developmental mechanism trial, not a claimed novel Agent method.
"""
import argparse
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import urllib.error
import urllib.request

from controlled_feedback import save, sha, utc
from native_validation_comparison import closure
from run_submission_review import bound_copy
import free_program_comparison as coding
import program_portfolio_comparison as shared

PROJECT = coding.PROJECT
OUT = PROJECT / 'results/counterexample404'
SOURCE = shared.SOURCE
QUALIFIED = Path('/data/yyf-zyq/PHM_20261010/20261025T235000Z_portfolio399')
read = lambda p: json.loads(Path(p).read_text())
ARMS = ['ordinary', 'error_directed']
NUMBER = '404'
EXPERIMENT = {}
POLICY = '''You use an error-directed generation policy in ONE coding context. The code space and final acceptance decision are unrestricted, as in the ordinary policy.

Before each new registration, write a short numbered entry in experiment_plan.md. Bind it to the current registered parent ID. Identify concrete wrong predictions in its existing lawful per-acquisition reports, and at least one previously correct group that the change could harm. Distinguish fold-model evidence from full-checkpoint evidence and condition groups from independent machines. State a physical/statistical explanation, a rival, the one representation or modeling assumption to change, and a predicted observable contrast. If no labeled error is visible, state that and use a discriminating hypothesis instead of inventing errors. Do not use group IDs or label counts as prediction rules.

Put the parent ID, named legal counterexample groups and predicted contrast in the register_program claim argument, with the competing explanation in rival. These arguments are captured before fitting returns; a journal written only after seeing a result does not demonstrate a prospective prediction.

After registration, compare actual whole-program repairs AND harms using the shared tools. Update the entry with the immutable new ID and observed result. When successive programs retain the same wrong predictions on the same legal view, do not spend the next registration on a cosmetic or parameter-only variant of that failed assumption: test a different feature or modeling hypothesis that could correct the persistent error while preserving other decisions. Different algorithm names alone are not evidence of useful diversity. You may stop if no defensible alternative remains.

This policy is a hypothesis under test. Do not force validation gains, retrain on development labels, or regard fitting the selected error cases as proof of transport. Final selection can keep any registered program or use the shared real composition tool. Compare the whole-program evidence, useful improvements, and harms; no forced keep, vote, or predetermined score rule. Keep the journal concise and preserve the same total budget for implementation and sealed submission.'''


def configure(root, specification=None):
    """Allow a new policy trial without copying the qualified execution harness."""
    global NUMBER, EXPERIMENT, OUT, ARMS, POLICY
    source = specification or root / 'EXPERIMENT_SPEC.json'
    if not source.exists():
        return
    spec = read(source)
    assert re.fullmatch(r'[0-9]+', spec['number'])
    assert re.fullmatch(r'[a-z][a-z0-9_]*', spec['method_id'])
    assert re.fullmatch(r'[a-z][a-z0-9_]*', spec['result_directory'])
    assert set(spec['protocol_updates']) <= {'factor', 'mechanism', 'rivals', 'novelty_boundary'}
    assert spec['policy'].strip() and spec['method_id'] != 'ordinary'
    if 'shared_evidence' in spec:
        evidence = spec['shared_evidence']
        assert evidence['target_source'] == 'COMFAULDA'
        assert evidence['cards'] and all(c['source'] != evidence['target_source'] for c in evidence['cards'])
        assert len(json.dumps(evidence).encode()) <= 32768
    NUMBER, EXPERIMENT = spec['number'], spec
    OUT = PROJECT / 'results' / spec['result_directory']
    ARMS, POLICY = ['ordinary', spec['method_id']], spec['policy']


def prepare(root):
    recovery = root / f'PREPARATION_FAILURE{NUMBER}.json'
    if root.exists() and set(p.name for p in root.iterdir()) != {'runtime'}:
        assert recovery.exists() and not (root / 'PROTOCOL337.json').exists() and not (root / 'subjects').exists(), 'Use a unique run; only documented pre-protocol recovery is allowed'
        assert read(recovery)['new_model_requests'] == 0 and read(recovery)['new_fits'] == 0
    for d in ['runtime/tmp', 'runtime/mpl', 'audit', 'code', 'common_tools', 'scripts', 'logs', 'public_data', 'private_inputs', 'reference/p00']:
        (root / d).mkdir(parents=True, exist_ok=True)
    if EXPERIMENT:
        save(root / 'EXPERIMENT_SPEC.json', EXPERIMENT)
    bindings = []
    for name in ['base_cli_config.toml', 'model_catalog187.json', 'retention_policy.json']:
        bound_copy(SOURCE, name, root / name, bindings)
    for role in ['train', 'development', 'independent_check']:
        name = 'private_inputs/' + role + '.npz'; bound_copy(SOURCE, name, root / name, bindings)
    for name in ['train.npz', 'development.npz', 'independent_check_observations.npz', 'metadata.json']:
        bound_copy(SOURCE, 'public_data/' + name, root / 'public_data' / name, bindings)
    baseline = read(SOURCE / 'reference/p00/REPORT.json')
    for name in ['REPORT.json', *baseline['artifacts']]:
        bound_copy(SOURCE, 'reference/p00/' + name, root / 'reference/p00' / name, bindings)
    for p in (SOURCE / 'common_tools').glob('*.py'):
        shutil.copy2(PROJECT / 'scripts' / p.name, root / 'common_tools' / p.name)
    old = read(SOURCE / 'PROTOCOL337.json')
    url = 'https://api.github.com/repos/openai/codex/releases/latest'
    precheck = root / f'audit/CLI_PRECHECK{NUMBER}.json'
    if precheck.exists():
        evidence = read(precheck)
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(evidence['checked_utc'])).total_seconds()
        assert 0 <= age <= 3600 and evidence['http_status'] == 200
        assert evidence['url'] == 'https://github.com/openai/codex/releases/latest'
        assert evidence['official_page'] == 'https://github.com/openai/codex/releases/tag/' + evidence['latest_tag']
        assert sha(root / f'audit/CLI_RELEASE_PAGE{NUMBER}.html') == evidence['page_sha256']
        assert evidence['cli_sha256'] == old['cli_sha256']
        release = dict(tag_name=evidence['latest_tag'], html_url=evidence['official_page'], published_at=evidence['published_at'])
        source_detail = dict(method='same_run_recent_official_HTML_precheck', precheck_sha256=sha(precheck), page_sha256=evidence['page_sha256'], checked_online_utc=evidence['checked_utc'])
    else:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'PHM-research'}), timeout=30) as response:
                release = json.load(response)
            source_detail = dict(method='official_GitHub_release_API')
        except urllib.error.HTTPError as error:
            if error.code not in (403, 429):
                raise
            fallback = 'https://github.com/openai/codex/releases/latest'
            with urllib.request.urlopen(urllib.request.Request(fallback, headers={'User-Agent': 'PHM-research'}), timeout=30) as response:
                page, final_url = response.read(), response.geturl()
            match = re.fullmatch(r'https://github.com/openai/codex/releases/tag/(rust-v[0-9.]+)', final_url)
            assert match, 'Latest stable release redirect must resolve to the official repository'
            page_path = root / f'audit/CLI_RELEASE_PAGE{NUMBER}.html'
            page_path.write_bytes(page)
            release = dict(tag_name=match.group(1), html_url=final_url, published_at=None)
            source_detail = dict(method='official_latest_release_redirect', api_http_status=error.code, fallback_url=fallback, page_sha256=sha(page_path))
    assert release['tag_name'] == 'rust-v0.164.0' and sha(Path(old['cli'])) == old['cli_sha256']
    save(root / f'audit/CLI_SOURCE{NUMBER}.json', dict(url=url, official_page=release['html_url'], latest_tag=release['tag_name'], published_at=release['published_at'], checked_utc=utc(), cli_sha256=old['cli_sha256'], **source_detail))
    cfg = dict(iteration='A'+NUMBER, frozen_utc=utc(), cli=old['cli'], cli_sha256=old['cli_sha256'], model='deepseek-flash',
        arms=ARMS, repeats=2, planned_native_sessions=4, maximum_parallel_sessions=4,
        episode_seconds=1800, episode_requests=60, episode_input_tokens=4000000, episode_output_tokens=131072, output_per_request=32768,
        max_registered_programs=4, max_compositions_per_chain=2, program_seconds=450, fit_seconds=120, predict_seconds=45, numerical_threads=2,
        prompt_file=f'PROMPTS{NUMBER}.json', enable_paired_evidence=True, probe_reusable=True, build_stage_only=True, defer_independent_check=False,
        stage='single_context_final_choice', source_archive=old['source_archive'], source_record_roles_Hz=old['source_record_roles_Hz'],
        maximum_new_requests=240, maximum_fit_entry_attempts=48, maximum_disk_GiB=24,
        factor='Error-directed generation and evidence journal versus unrestricted ordinary single-context coding; same seven tools, data, source, model and budget.',
        mechanism='Persistent legal errors trigger a different modeling assumption rather than another variant with unchanged error coverage. Transfer is unproven.',
        rivals=['Current validation errors are unrepresentative; targeting them worsens transport.', 'Ordinary already performs the same investigation.', 'Journaling reduces useful coding budget.', 'Apparent benefit is only fixed ensembling.'],
        primary='All4chains complete and all candidates scored; method mean error strictly below ordinary and keep, no higher harm and no lower gain.',
        attribution='Additionally beat ordinary-generated programs under the identical uniform vote with ties anchored to the ordinary sealed program.',
        controls=['keep', 'development', 'mean_cv', 'equal_CV_development', 'full_checkpoint_validation', 'same_bank_equal_vote'],
        no_outcome_based_rerun=True, no_further_quota_recovery=True, independent_confirmation_scenarios=0, source_clusters=1, known_development_cohorts=1,
        confirmation338_performance_unopened=True, HUST_final22_unopened=True, novelty_established=False,
        novelty_boundary='MLE-STAR and SAGE-Loop already use feedback/refinement/ensembles; EEMA uses feedback-routed search. This pilot tests an explicit error-preservation contrast, not a novelty claim.',
        all_final_evaluation_after_all_choices_sealed=True)
    cfg.update(EXPERIMENT.get('protocol_updates', {}))
    for name in ['PROTOCOL337.json', f'PROTOCOL{NUMBER}.json']: save(root / name, cfg)
    common = shared.COMMON
    if EXPERIMENT.get('shared_evidence'):
        save(root / 'public_evidence/contrasts.json', EXPERIMENT['shared_evidence'])
        common += '\nBoth workflows have the identical read-only /evidence/contrasts.json: completed contrasts from OTHER physical sources. These are historical observations and counterexamples, not target answers, transferable guarantees, or an instruction to keep every old feature. Read this short dossier. You may use or reject its relevance; all actual current-task choices must use the same lawful tools and real measurements.\n'
    save(root / f'PROMPTS{NUMBER}.json', {ARMS[0]:common + '\n' + shared.ROLES['ordinary'], ARMS[1]:common + '\n' + POLICY})
    save(root / 'ENGINE_CONFIG.json', dict(input_root=str(root / 'private_inputs'), common_tools=str(root / 'common_tools'), fit_seconds=120, predict_seconds=45, program_seconds=450))
    coding.OUT = OUT; coding.prepare(root)
    for row in read(root / 'SESSIONS337.json'):
        run = Path(row['path']); cfg_b = read(run / 'BROKER_CONFIG.json')
        cfg_b.update(enable_submit_review=True, submission_review_required=False, enable_composition=True, max_compositions=2)
        save(run / 'BROKER_CONFIG.json', cfg_b)
        p = run / 'cli_config187.toml'; p.write_text(p.read_text().replace('PHM_PAIRED_EVIDENCE = "1"', 'PHM_PAIRED_EVIDENCE = "1", PHM_SUBMIT_REVIEW = "1", PHM_COMPOSITION = "1"'))
        discovery = read(run / 'PROTOCOL187.json'); discovery['expected_mcp_tools'] += ['mcp__phmforge__review_update', 'mcp__phmforge__compose_programs']; save(run / 'PROTOCOL187.json', discovery)
        if EXPERIMENT.get('shared_evidence'):
            mounts = read(run / 'mounts.json')
            mounts['mounts'].append(dict(source=str(root / 'public_evidence'), destination='/evidence'))
            save(run / 'mounts.json', mounts)
    for name in closure(['counterexample_program_comparison', 'audit_native_program_comparison']):
        shutil.copy2(PROJECT / 'scripts' / (name + '.py'), root / 'code' / (name + '.py'))
    save(root / f'SOURCE_BINDINGS{NUMBER}.json', bindings)
    save(root / 'EFFECTIVE_CODE_FREEZE337.json', {str(p.relative_to(root)): sha(p) for d in ['code', 'common_tools', 'scripts'] for p in (root / d).glob('*.py')})
    save(root / 'OWNERSHIP.json', dict(run=str(root), archive='/data/yyf-zyq/PHM_20261010/' + root.name, created_utc=utc(), goal='A330', scope='A'+NUMBER+'developmentpilot'))
    save(OUT / f'PROTOCOL{NUMBER}.json', cfg)
    print(json.dumps(dict(prepared=True, new_fits=0, new_LLM_calls=0, source_bindings=len(bindings))))


def qualify(root):
    # Reuse the real-training qualification only if the exact engine/tool code,
    # input arrays, baseline and numerical caps are unchanged. No fake retraining.
    bindings = []; receipt = read(QUALIFIED / 'ARCHIVE_VERIFIED.json')
    for name in ['PREFLIGHT399.json', 'ENGINE_CONFIG.json']:
        bound_copy(QUALIFIED, name, root / 'audit/previous_qualification' / name, bindings)
    assert read(root / 'audit/previous_qualification/PREFLIGHT399.json')['passed']
    for p in (root / 'common_tools').glob('*.py'):
        relative = 'common_tools/' + p.name
        bound_copy(QUALIFIED, relative, root / 'audit/previous_qualification' / relative, bindings)
        assert sha(p) == sha(root / 'audit/previous_qualification' / relative), p.name
    for name in ['private_inputs/train.npz', 'private_inputs/development.npz', 'private_inputs/independent_check.npz', 'public_data/metadata.json', 'reference/p00/REPORT.json']:
        bound_copy(QUALIFIED, name, root / 'audit/previous_qualification' / name, bindings)
        assert sha(root / name) == sha(root / 'audit/previous_qualification' / name), name
    previous = read(root / 'audit/previous_qualification/ENGINE_CONFIG.json'); current = read(root / 'ENGINE_CONFIG.json')
    assert {k: v for k, v in previous.items() if k not in ['input_root', 'common_tools']} == {k: v for k, v in current.items() if k not in ['input_root', 'common_tools']}
    save(root / f'REUSED_QUALIFICATION{NUMBER}.json', dict(utc=utc(), passed=True, source=str(QUALIFIED), manifest_sha256=receipt['manifest_sha256'], bindings=bindings, actual_new_training=0, actual_new_forward=0, qualification_reused=True))
    coding.OUT = OUT
    native_receipt = root / 'NATIVE_QUALIFICATION337.json'
    if native_receipt.exists():
        assert (root / f'QUALIFICATION_RECOVERY{NUMBER}.json').exists()
        prior = read(native_receipt)
        assert prior['passed'] and prior['actual_model_calls'] == 0
        assert {r['id'] for r in prior['rows']} == {r['id'] for r in read(root / 'SESSIONS337.json')}
        assert all(r['isolation'] and r['mcp_catalog'] and r['file_transport'] for r in prior['rows'])
    else:
        coding.qualify(root)
    if EXPERIMENT.get('shared_evidence'):
        digest = sha(root / 'public_evidence/contrasts.json')
        evidence_checks = []
        for row in read(root / 'SESSIONS337.json'):
            run = Path(row['path']); folder = run / 'audit/shared_evidence'; folder.mkdir()
            probe = """import hashlib,json
from pathlib import Path
p=Path('/evidence/contrasts.json');b=p.read_bytes();d=json.loads(b)
assert hashlib.sha256(b).hexdigest()==%r
assert d['target_source']=='COMFAULDA' and all(c['source']!='COMFAULDA' for c in d['cards'])
try:p.write_bytes(b)
except OSError:readonly=True
else:readonly=False
assert readonly
print(json.dumps(dict(passed=True,readonly=readonly,cards=len(d['cards']))))
""" % digest
            ns = coding.arguments(root, run)
            command = coding.isolate_filesystem([str(ns.cli), 'sandbox', '-P', 'iot-offline', '-C', '/app', '--', '/opt/python312/bin/python3.12', '-B', '-c', probe], ns, run / 'app', folder, None)
            env = coding.clean_env(run)
            env.update(PYTHONPATH='/opt/phm_python/extra:/tools', PATH='/opt/python312/bin:/usr/bin:/bin', TMPDIR='/app/.runtime/tmp', TMP='/app/.runtime/tmp', TEMP='/app/.runtime/tmp')
            fixture = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            fixture.bind(str(run / 'ipc/s.sock')); fixture.listen(1)
            try:
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
            finally:
                fixture.close(); (run / 'ipc/s.sock').unlink(missing_ok=True)
            save(folder / 'RESULT.json', dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, sha256=digest))
            assert result.returncode == 0, row['id']
            evidence_checks.append(dict(id=row['id'], sha256=digest, readonly=True, target_source_excluded=True))
        save(root / f'EVIDENCE_QUALIFICATION{NUMBER}.json', dict(passed=True, checks=evidence_checks, new_model_calls=0, new_fits=0))
    for row in read(root / 'SESSIONS337.json'):
        run = Path(row['path']); saved = run / 'audit/pre_scientific_transport'; saved.mkdir()
        for relative in ['broker/STATE.json', 'feedback/responses', 'app/.runtime/feedback_requests']:
            p = run / relative
            if p.exists():
                shutil.move(str(p), str(saved / relative.replace('/', '_')))
                if p.suffix != '.json': p.mkdir(parents=True)
    frozen = [root / name for name in [f'PROTOCOL{NUMBER}.json', 'PROTOCOL337.json', f'PROMPTS{NUMBER}.json', 'EFFECTIVE_CODE_FREEZE337.json', f'SOURCE_BINDINGS{NUMBER}.json', f'REUSED_QUALIFICATION{NUMBER}.json']]
    if EXPERIMENT:
        frozen.append(root / 'EXPERIMENT_SPEC.json')
    if EXPERIMENT.get('shared_evidence'):
        frozen += [root / 'public_evidence/contrasts.json', root / f'EVIDENCE_QUALIFICATION{NUMBER}.json']
    frozen += [p for row in read(root / 'SESSIONS337.json') for p in Path(row['path']).glob('*.json') if p.name != 'BROKER_READY.json']
    frozen += list((root / 'subjects').glob('*/cli_config187.toml'))
    save(root / f'ADMISSION_FREEZE{NUMBER}.json', {str(p.relative_to(root)): sha(p) for p in frozen})
    save(root / f'QUALIFIED{NUMBER}.json', dict(utc=utc(), passed=True, native_workspaces=4, same7MCPtools=True, new_model_calls=0, real_training_qualification='exact-source reuse', fresh_native_isolation_and_transport=True))
    print(json.dumps(read(root / f'QUALIFIED{NUMBER}.json')))


def agents(root):
    assert read(root / f'QUALIFIED{NUMBER}.json')['passed']
    for relative, digest in read(root / f'ADMISSION_FREEZE{NUMBER}.json').items(): assert sha(root / relative) == digest, relative
    coding.OUT = OUT; coding.agents(root)


def evaluate(root):
    shared.OUT = OUT; shared.evaluate(root, iteration=NUMBER, arms=ARMS)


def pipeline(root):
    marker = root / f'PIPELINE{NUMBER}.json'
    assert not marker.exists(), 'Never restart a started pipeline without inspecting its actual child'
    script = root / 'code/counterexample_program_comparison.py'
    python = str(coding.PYTHON / 'bin/python3.12')
    commands = [
        ('agents', [python, '-B', str(script), 'agents', '--run', str(root)]),
        ('evaluate', [python, '-B', str(script), 'evaluate', '--run', str(root)]),
        ('audit', [python, '-B', str(root / 'code/audit_native_program_comparison.py'), '--run', str(root), '--iteration', NUMBER, '--output', str(root / f'INDEPENDENT_AUDIT{NUMBER}.json')])]
    for step, command in commands:
        if step != 'agents':
            assert read(root / 'HANDLE337.json')['status'] == 'terminal'
            assert (root / 'ALL_CHOICES_SEALED337.json').exists()
        with (root / 'logs' / ('pipeline_' + step + '.stdout.log')).open('w') as stdout, (root / 'logs' / ('pipeline_' + step + '.stderr.log')).open('w') as stderr:
            child = subprocess.Popen(command, env=coding.environment(root), stdout=stdout, stderr=stderr)
            save(marker, dict(utc=utc(), status='running', controller_pid=os.getpid(), step=step, child_pid=child.pid))
            code = child.wait()
        if code:
            save(marker, dict(utc=utc(), status='failed', controller_pid=os.getpid(), step=step, child_pid=child.pid, child_exit_code=code))
            raise SystemExit(code)
        print(json.dumps(dict(utc=utc(), step=step, status='terminal', child_exit_code=code)), flush=True)
    save(marker, dict(utc=utc(), status='terminal_audited', controller_pid=os.getpid(), child_pid=None, behavioral_audit_and_archive_pending=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'qualify', 'agents', 'evaluate', 'pipeline']); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--spec', type=Path)
    a = p.parse_args()
    assert a.spec is None or a.action == 'prepare', 'An existing run loads its own frozen specification'
    configure(a.run.resolve(), a.spec)
    globals()[a.action](a.run.resolve())
