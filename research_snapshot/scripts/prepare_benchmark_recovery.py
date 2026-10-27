"""Prepare exactly the three IPC-contaminated originals, without model calls."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import uuid

from prepare_full_benchmark_campaign import forge_command


JOBS = ('forge_0005', 'forge_0023', 'forge_0024')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prep-run', type=Path, required=True)
    args = parser.parse_args()
    prep = args.prep_run.resolve()
    settings = json.loads((prep / 'PROTOCOL222.json').read_text())
    source = Path(settings['source_archive'])
    project = Path(__file__).resolve().parents[1]
    manifest_data = (source / 'SOURCE_MANIFEST.json').read_bytes()
    assert digest(manifest_data) == settings['source_manifest_sha256']
    manifest = json.loads(manifest_data)
    cache = Path(settings['shared_cache'])
    if cache.exists():
        raise FileExistsError('Refuse to replace an existing shared restore: ' + str(cache))
    cache.mkdir()
    shared_prefixes = ['runtime/forge_public/', 'runtime/forge_deps/', 'runtime/forge_data/']
    selected = {r['path']: r for r in manifest['files'] if any(r['path'].startswith(p) for p in shared_prefixes)}
    metadata = {'PROTOCOL187.json', 'PLAN187.json', 'cli_config187.toml', 'model_catalog187.json'}
    for job in JOBS:
        metadata |= {f'workspaces/{job}/inputs/question.txt', f'workspaces/{job}/instruction.txt'}
    metadata_index = {r['path']: r for r in manifest['files'] if r['path'] in metadata}
    assert len(metadata_index) == 10
    assert sum(r['bytes'] for r in selected.values()) < settings['restore_logical_budget_bytes']
    restored, metadata_seen = [], []
    with tarfile.open(source / 'RUN_CONTENTS.tar.gz', 'r|gz') as tar:
        for member in tar:
            if member.name in selected:
                assert member.isfile() and '..' not in Path(member.name).parts
                expected = selected[member.name]
                target = cache / Path(member.name).relative_to('runtime')
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with target.open('wb') as output, tar.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(block)
                        output.write(block)
                assert target.stat().st_size == expected['bytes'] and h.hexdigest() == expected['sha256']
                target.chmod(0o555 if member.mode & 0o111 else 0o444)
                restored.append(expected | {'cache_relative_path': str(target.relative_to(cache))})
            elif member.name in metadata_index:
                data = tar.extractfile(member).read()
                expected = metadata_index[member.name]
                assert len(data) == expected['bytes'] and digest(data) == expected['sha256']
                target = prep / 'evidence' / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                metadata_seen.append(expected)
            if len(restored) == len(selected) and len(metadata_seen) == len(metadata):
                break
    assert len(restored) == len(selected) and len(metadata_seen) == len(metadata)
    cache_manifest = {'source_archive': str(source), 'source_manifest_sha256': digest(manifest_data),
                      'files': restored, 'files_verified': len(restored),
                      'bytes_verified': sum(r['bytes'] for r in restored)}
    save(cache / 'CACHE_VERIFIED.json', cache_manifest)
    save(prep / 'CACHE_RESTORE222.json', cache_manifest)
    save(prep / 'SOURCE_METADATA222.json', metadata_seen)
    for directory in sorted((p for p in cache.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        directory.chmod(0o555)
    cache.chmod(0o555)

    execution = Path(settings['execution_run'])
    execution.mkdir(exist_ok=False)
    for name in ['runtime/tmp', 'runtime/mpl', 'plans', 'workspaces', 'primary', 'logs', 'traces', 'ipc', 'audit']:
        (execution / name).mkdir(parents=True, exist_ok=True)
    old_protocol = json.loads((prep / 'evidence/PROTOCOL187.json').read_text())
    binary = Path(old_protocol['cli'])
    assert binary.is_file() and digest(binary.read_bytes()) == old_protocol['cli_sha256']
    for name in ['cli_config187.toml', 'model_catalog187.json']:
        shutil.copyfile(prep / 'evidence' / name, execution / name)
    old_jobs = {r['id']: r for r in json.loads((prep / 'evidence/PLAN187.json').read_text())['jobs']}
    routes, jobs, frozen = {}, [], []
    for job_id in JOBS:
        workspace = execution / 'workspaces' / job_id
        for name in ['inputs', '.runtime/tooltmp', '.runtime/mpl', '.runtime/cache', '.runtime/hf', '.runtime/tool_requests']:
            (workspace / name).mkdir(parents=True, exist_ok=True)
        original = prep / 'evidence/workspaces' / job_id
        for name in ['inputs/question.txt', 'instruction.txt']:
            shutil.copyfile(original / name, workspace / name)
        (workspace / 'tool_client.py').write_bytes((project / 'scripts/tool_bridge_client.py').read_bytes())
        save(workspace / 'readonly_inputs.json', ['tool_client.py'])
        capability = uuid.uuid4().hex
        save(workspace / 'inputs/tool_connection.json', {'route': capability})
        command = forge_command(execution, workspace, project)
        for folder in ['forge_public', 'forge_deps', 'forge_data']:
            command = [str(cache / folder) if part == str(execution / 'runtime' / folder) else part for part in command]
        routes[capability] = dict(job=job_id, family='forge', command=command, cwd=str(execution),
                                  environment={}, request_root=str(workspace / '.runtime/tool_requests'))
        old = old_jobs[job_id]
        job = {k: old[k] for k in ['id', 'family', 'task_id', 'category', 'condition', 'repeat', 'answer_mode', 'numerical_threads']}
        job.update(workspace=str(workspace.relative_to(execution)), prompt_file=str((workspace / 'instruction.txt').relative_to(execution)))
        native = {**job, 'workspace': str(workspace), 'prompt_file': str(workspace / 'instruction.txt')}
        plan = execution / 'plans' / (job_id + '.json')
        mounts = execution / 'plans' / (job_id + '_mounts.json')
        save(plan, {'jobs': [native]})
        save(mounts, {'mounts': [{'source': str(execution / 'ipc/fifo_files' / job_id), 'destination': '/toolbridge'}]})
        jobs.append({**job, 'plan': str(plan), 'mounts': str(mounts)})
        for path in [workspace / 'inputs/question.txt', workspace / 'instruction.txt', workspace / 'tool_client.py',
                     workspace / 'inputs/tool_connection.json', workspace / 'readonly_inputs.json', plan, mounts]:
            frozen.append({'path': str(path), 'sha256': digest(path.read_bytes())})
    for path in [execution / 'cli_config187.toml', execution / 'model_catalog187.json',
                 project / 'scripts/benchmark_tool_bridge.py', project / 'scripts/replay_diagnostic_cli.py',
                 project / 'scripts/run_full_benchmark_campaign.py', project / 'scripts/tool_bridge_client.py']:
        frozen.append({'path': str(path), 'sha256': digest(path.read_bytes())})
    save(execution / 'PLAN187.json', {'jobs': jobs})
    save(execution / 'BROKER187_FIFO_FILES.json', dict(transport='fifo-files', fifo_root=str(execution / 'ipc/fifo_files'),
            socket=str(execution / 'ipc/not_used.sock'), traces=str(execution / 'traces'), routes=routes))
    protocol = dict(old_protocol, published_questions=3, max_concurrency=3, max_new_primary_rolling24h=400,
                    subject_timeout_seconds=1200, original_tool_timeout_seconds=180,
                    recovery_only_original_ids=[old_jobs[j]['task_id'] for j in JOBS],
                    scope='Three IPC-contaminated originals only; no grade-based retries or altered source data.',
                    authorization_status='PENDING_EXPLICIT_SCOPED_EXTENSION; old400grantcovered231remainingonly',
                    no_model_calls_during_preparation=True)
    save(execution / 'PROTOCOL187.json', protocol)
    save(execution / 'INITIAL_FREEZE187_FIFO_FILES.json', {'files': frozen})
    save(execution / 'RECOVERY_SCOPE.json', {'original_jobs': list(JOBS), 'original_run': source.name,
        'original_task_ids': protocol['recovery_only_original_ids'], 'proof': str(project / 'results/fullbench187/confirmed_rpc_misassociation221.json'),
        'cache_path': str(cache), 'cache_manifest_sha256': digest((cache / 'CACHE_VERIFIED.json').read_bytes()),
        'new_model_sessions_max': 3, 'old_answers_or_scores_provided_to_subject': False})
    save(execution / 'OWNERSHIP.json', {'owner': 'root', 'takeover_allowed': False,
        'lease_expires_utc': (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        'status': 'prepared_pending_user_scope_extension_no_primary_started', 'writers_stopped': True})
    save(execution / 'retention_policy.json', {'excluded_rebuildable_directory': 'clients-cache',
        'shared_dependencies': {'path': str(cache.parent), 'retention': 'Byte-identical A187sharedsource/cache, full manifest reference retained.'}})
    result = {'status': 'prepared_no_models', 'execution_run': str(execution), 'original_jobs': list(JOBS),
              'shared_files_verified': len(restored), 'shared_bytes_verified': cache_manifest['bytes_verified'],
              'CLI_sha256_verified': old_protocol['cli_sha256'], 'new_Agent_sessions': 0,
              'approval_required_before_any_primary': True, 'next': 'actual original-tool/native-offline qualification and controller refusal gate'}
    save(prep / 'PREPARED222.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
