"""Bounded independent CLI replays of frozen diagnostic task batches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shlex
import shutil
import subprocess
from threading import Event, Thread
import time


def capture_source_versions(workspace, destination, stop, record):
    """Retain observed code/result versions outside the evaluated filesystem."""
    destination.mkdir(parents=True, exist_ok=False)
    previous = {}
    record.update(observed_versions=0, skipped_large_files=[], read_errors=0)
    with (destination / 'observations.jsonl').open('w') as log:
        while True:
            try:
                paths = sorted(workspace.rglob('*'))
            except OSError:
                # The native CLI may remove a transient runtime directory during
                # traversal. Keep observing subsequent scientific file versions.
                record['directory_scan_errors'] = record.get('directory_scan_errors', 0) + 1
                if stop.wait(0.2):
                    break
                continue
            for path in paths:
                if path.relative_to(workspace).parts[0] in ('inputs', '.runtime'):
                    continue
                if path.is_symlink() or not path.is_file() or path.suffix not in {'.py', '.json', '.md', '.csv'}:
                    continue
                try:
                    before = path.stat()
                    if before.st_size > 4 * 1024 * 1024:
                        name = str(path.relative_to(workspace))
                        if name not in record['skipped_large_files']:
                            record['skipped_large_files'].append(name)
                        continue
                    data = path.read_bytes()
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        continue
                    digest = hashlib.sha256(data).hexdigest()
                    name = str(path.relative_to(workspace))
                    if previous.get(name) == digest:
                        continue
                    saved = destination / (digest + path.suffix)
                    if not saved.exists():
                        saved.write_bytes(data)
                    log.write(json.dumps(dict(path=name, sha256=digest, bytes=len(data),
                              snapshot=saved.name, observed_utc=datetime.now(timezone.utc).isoformat())) + '\n')
                    log.flush()
                    previous[name] = digest
                    record['observed_versions'] += 1
                except OSError:
                    record['read_errors'] += 1
            if stop.wait(0.2):
                break


def isolate_filesystem(command, args, workspace, folder, schema):
    """Expose task files and interpreter libraries, not research archives."""
    visible_workspace = Path(getattr(args, 'workspace_alias', None) or workspace)
    mount = ['bwrap', '--unshare-user', '--unshare-pid', '--unshare-ipc',
             '--unshare-uts', '--die-with-parent', '--new-session']
    gateway_socket = getattr(args, 'model_gateway_socket', None)
    if gateway_socket:
        mount += ['--unshare-net', '--ro-bind', str(gateway_socket), '/model_gateway.sock']
    for path in [Path('/usr'), Path('/bin'), Path('/sbin'), Path('/lib'), Path('/lib64'), Path('/etc'),
                 Path.home() / '.local/lib', args.cli.parent.parent]:
        if path.exists():
            mount.extend(['--ro-bind', str(path), str(path)])
    if gateway_socket:
        # Task data occupy /data, which can shadow an archived CLI below /data.
        mount += ['--ro-bind', str(args.cli.parent.parent), '/opt/codex']
        command = ['/opt/codex/bin/codex' if x == str(args.cli) else x for x in command]
    resolver = Path('/etc/resolv.conf').resolve()
    if resolver != Path('/etc/resolv.conf') and resolver.is_file():
        mount.extend(['--ro-bind', str(resolver), str(resolver)])
    if args.runtime_mounts_file:
        for entry in json.loads(args.runtime_mounts_file.read_text())['mounts']:
            mount.extend(['--ro-bind', entry['source'], entry['destination']])
    auth = Path.home() / '.codex/auth.json'
    if not args.credential_env_file and not gateway_socket and not auth.is_file():
        raise FileNotFoundError('Expected existing CLI authentication file')
    cache = args.isolation_config.parent / 'clients-cache' / folder.name
    cache.mkdir(parents=True, exist_ok=False)
    for name in ['models_cache.json', 'installation_id']:
        runtime_file = Path.home() / '.codex' / name
        if runtime_file.is_file():
            shutil.copy2(runtime_file, cache / name)
    mount.extend(['--bind', str(cache), str(Path.home() / '.codex')])
    if args.model_catalog:
        mount.extend(['--ro-bind', str(args.model_catalog),
                      str(Path.home() / '.codex/models.json')])
    if not args.credential_env_file and not gateway_socket:
        runtime_env = Path.home() / '.codex/.env'
        if runtime_env.is_file():
            mount.extend(['--ro-bind', str(runtime_env), str(runtime_env)])
        mount.extend(['--ro-bind', str(auth), str(auth)])
    mount.extend(['--ro-bind', str(args.isolation_config), str(Path.home() / '.codex/config.toml'),
                  '--bind', str(workspace), str(visible_workspace),
                  '--bind', str(folder), str(folder),
                  '--proc', '/proc', '--dev', '/dev',
                  '--chdir', str(visible_workspace)])
    if schema is not None:
        mount.extend(['--ro-bind', str(schema), str(schema)])
    # Source inputs are immutable; generated analyses remain writable.
    if (workspace / 'inputs').is_dir():
        mount.extend(['--ro-bind', str(workspace / 'inputs'), str(visible_workspace / 'inputs')])
    readonly_inputs = workspace / 'readonly_inputs.json'
    if readonly_inputs.is_file():
        # Older legitimate workflows keep data beside their analysis script.
        # Mount only this job's declared inputs, never other task workspaces.
        for relative in json.loads(readonly_inputs.read_text()):
            source = workspace / relative
            if not source.resolve().is_relative_to(workspace.resolve()) or not source.exists():
                raise ValueError('Read-only input must exist inside the task workspace')
            mount.extend(['--ro-bind', str(source), str(visible_workspace / relative)])
        mount.extend(['--ro-bind', str(readonly_inputs), str(visible_workspace / 'readonly_inputs.json')])
    if getattr(args, 'permission_profile', None):
        # The outer namespace supplies filesystem isolation; the CLI profile
        # still enforces code-tool network policy. Keep its root immutable.
        mount.extend(['--remount-ro', '/'])
    if gateway_socket:
        command = ['/opt/python312/bin/python3.12', '-B', '/tools/isolated_model_gateway.py',
                   'relay', '--socket', '/model_gateway.sock', '--port', '47777', '--'] + command
    return mount + command


def replay(job, args):
    folder = args.output / job['id']
    folder.mkdir()
    prompt = (args.plan.parent / job['prompt_file']).read_text()
    workspace = args.plan.parent / job['workspace']
    visible_workspace = Path(getattr(args, 'workspace_alias', None) or workspace)
    schema = args.plan.parent / job['schema_file'] if job.get('schema_file') else None
    answer_name = 'response.txt' if job.get('answer_mode') == 'text' else 'response.json'
    runtime = workspace / '.runtime'
    for name in ('tmp', 'mpl'):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    command = [str(args.cli), 'exec', '--disable', 'apps',
               '--ephemeral', '--skip-git-repo-check',
               '--json',
               '-m', args.model,
               '-c', 'model_reasoning_effort="max"', '-c', 'web_search="disabled"',
               '-c', 'approval_policy="never"',
               '-C', str(visible_workspace), '-o', str(folder / answer_name), '-']
    if getattr(args, 'permission_profile', None):
        command[-1:-1] = ['-c', 'default_permissions=' + json.dumps(args.permission_profile)]
    else:
        command[-1:-1] = ['--sandbox', 'workspace-write' if args.analysis_tools else 'read-only',
                         '-c', 'sandbox_workspace_write.network_access=false']
    if schema is not None:
        command[-1:-1] = ['--output-schema', str(schema)]
    if not args.analysis_tools:
        command[4:4] = ['--disable', 'shell_tool', '--disable', 'unified_exec']
    if args.isolation_config:
        command = isolate_filesystem(command, args, workspace, folder, schema)
    status = dict(job_id=job['id'], condition=job['condition'], repeat=job['repeat'],
                  status='running', model_requested=args.model, command=command,
                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                  timeout_seconds=args.timeout,
                  network_route='process_proxy_vars_removed; mounted_CLI_runtime_may_supply_route'
                                if args.isolation_config else 'process_proxy_vars_removed',
                  tools_disabled=['apps', 'web_search'] +
                                 ([] if args.analysis_tools else ['shell_tool', 'unified_exec']),
                  analysis_tools_enabled=args.analysis_tools,
                  permission_profile=getattr(args, 'permission_profile', None),
                  filesystem_isolated=bool(args.isolation_config),
                  workspace_alias=getattr(args, 'workspace_alias', None),
                  started_utc=datetime.now(timezone.utc).isoformat())
    if getattr(args, 'model_gateway_socket', None):
        status.update(network_route='Private network namespace; scoped Unix model gateway only',
                      upstream_credentials_visible_to_agent=False,
                      model_gateway_socket=str(args.model_gateway_socket))
    status_path = folder / 'status.json'
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    env = {k:v for k,v in os.environ.items() if not any(w in k.upper() for w in ['KEY','TOKEN','SECRET','PASSWORD'])}
    visible_runtime = visible_workspace / '.runtime'
    env.update(TMPDIR=str(visible_runtime / 'tmp'), TMP=str(visible_runtime / 'tmp'),
               TEMP=str(visible_runtime / 'tmp'), MPLCONFIGDIR=str(visible_runtime / 'mpl'))
    if args.credential_env_file:
        pattern = re.compile(r'^\s*(?:export\s+)?' + re.escape(args.credential_env_name) + r'\s*=\s*(.*)$')
        values = [m.group(1) for line in args.credential_env_file.read_text().splitlines()
                  if (m := pattern.match(line))]
        if len(values) != 1:
            raise ValueError('Expected exactly one requested credential assignment')
        parts = shlex.split(values[0], comments=True)
        if len(parts) != 1 or not parts[0] or any(c in parts[0] for c in ['$', '`', '\n']):
            raise ValueError('Credential assignment must contain one literal value')
        env[args.credential_env_name] = parts[0]
        status['credential_source'] = str(args.credential_env_file)
        status['credential_env_name'] = args.credential_env_name
    if args.analysis_tools:
        env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
        if args.runtime_mounts_file:
            env['PYTHONPATH'] = '/opt/phm_python/jax:/opt/phm_python/base:/opt/phm_python/extra:' + str(visible_workspace/'inputs/impact_current') + ':' + str(visible_workspace/'inputs/runtime')
    if getattr(args, 'model_gateway_socket', None):
        env['PATH'] = '/opt/python312/bin:' + env.get('PATH','/usr/bin:/bin')
    for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
        env.pop(key, None)
    start = time.monotonic()
    with (folder / 'events.jsonl').open('w') as out, (folder / 'stderr.txt').open('w') as err:
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                             text=True, env=env, start_new_session=True)
        status['cli_pid'] = p.pid
        status_path.write_text(json.dumps(status, indent=2) + '\n')
        capture_stop, capture_thread = Event(), None
        if args.snapshot_generated_source:
            history = args.output.parent / 'source_history' / job['id']
            status['source_history'] = dict(path=str(history), poll_seconds=0.2,
                                           limitation='observed versions; changes faster than polling may be missed')
            capture_thread = Thread(target=capture_source_versions,
                                    args=(workspace, history, capture_stop, status['source_history']))
            capture_thread.start()
        try:
            p.communicate(prompt, timeout=args.timeout)
            status.update(status='finished' if p.returncode == 0 else 'execution_error', exit_code=p.returncode)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            status.update(status='timeout', exit_code=p.returncode)
        finally:
            capture_stop.set()
            if capture_thread:
                capture_thread.join()
    status.update(elapsed_seconds=time.monotonic()-start, ended_utc=datetime.now(timezone.utc).isoformat())
    calls, usage = [], None
    # Preserve JSONL records containing valid Unicode line separators in
    # command output; they are not physical newline record boundaries.
    for line in (folder / 'events.jsonl').open():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        item = event.get('item', {})
        if item.get('type') not in [None, 'agent_message', 'reasoning']:
            calls.append(item.get('type'))
        if event.get('type') == 'turn.completed':
            usage = event.get('usage')
    status.update(observed_non_message_item_types=sorted(set(calls)), usage=usage)
    answer = folder / answer_name
    if status['status'] == 'finished':
        try:
            raw_answer = answer.read_text()
            if job.get('answer_mode') == 'text':
                if not raw_answer.strip():
                    raise ValueError('Empty benchmark answer')
            else:
                response = json.loads(raw_answer)
                expected = set(job['case_ids'])
                actual = [x['case_id'] for x in response['results']]
                if len(actual) != len(expected) or set(actual) != expected:
                    raise ValueError('Case inventory mismatch')
            status['response_valid'] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            status.update(response_valid=False, response_error_type=type(error).__name__)
    if args.cleanup_runtime_cache and args.isolation_config:
        cache = args.isolation_config.parent / 'clients-cache' / folder.name
        if p.poll() is None:
            raise RuntimeError('Runtime cache cleanup requires a terminal CLI process')
        files = [x for x in cache.rglob('*') if x.is_file() and not x.is_symlink()]
        status['excluded_runtime_cache_cleanup'] = dict(
            path=str(cache), file_count=len(files),
            logical_bytes=sum(x.lstat().st_size for x in files),
            allocated_bytes=sum(x.lstat().st_blocks * 512 for x in files),
            process_terminal=True,
            exclusion='predeclared rebuildable plugin catalogs and ephemeral CLI state')
        shutil.rmtree(cache)
        status['excluded_runtime_cache_cleanup']['completed'] = not cache.exists()
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps({k:status.get(k) for k in ['job_id','status','response_valid','elapsed_seconds']}), flush=True)
    return status


def validate_execution_budget(workers, timeout, jobs, task_config=None):
    """Keep the legacy limit; allow one explicit native task budget up to 8h."""
    if not 1 <= workers <= 16 or not 1 <= timeout <= 28800:
        raise ValueError('workers must be 1..16; timeout must be 1..28800 seconds')
    if timeout <= 3600:
        return None
    if task_config is None or workers != 1 or len(jobs) != 1:
        raise ValueError('timeout >3600 requires --task-budget-config, one worker and one job')
    import math
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # Existing dependency on Python3.10.
    raw = task_config.read_bytes()
    native = tomllib.loads(raw.decode('utf-8')).get('agent', {}).get('timeout_sec')
    if (isinstance(native, bool) or not isinstance(native, (int, float)) or
            not math.isfinite(native) or native <= 0 or timeout > native):
        raise ValueError('requested timeout exceeds or lacks a finite native agent budget')
    return dict(task_config=str(task_config), task_config_sha256=hashlib.sha256(raw).hexdigest(),
                native_timeout_seconds=native, requested_timeout_seconds=timeout,
                maximum_jobs=1, maximum_workers=1, absolute_cap_seconds=28800)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--task-budget-config', type=Path,
                        help='Original task.toml required for a single-job timeout above3600s (absolute cap28800s)')
    parser.add_argument('--analysis-tools', action='store_true',
                        help='Enable CLI code tools with workspace-write sandbox')
    parser.add_argument('--permission-profile',
                        help='Named CLI code permission profile; outer namespace root stays read-only')
    parser.add_argument('--isolation-config', type=Path,
                        help='Minimal task config to mount in a separate filesystem namespace')
    parser.add_argument('--cleanup-runtime-cache', action='store_true',
                        help='Delete only excluded per-job CLI cache after its process exits')
    parser.add_argument('--credential-env-file', type=Path,
                        help='Read one literal credential assignment; do not mount the file or OpenAI credentials')
    parser.add_argument('--credential-env-name', default='DEEPSEEK_API_KEY',
                        help='Environment variable supplied to the CLI; exclude it from code tools in config')
    parser.add_argument('--model-gateway-socket', type=Path,
                        help='Scoped external model relay; no provider credential or host network in the Agent namespace')
    parser.add_argument('--model-catalog', type=Path,
                        help='Read-only official custom model catalog mounted for the native CLI')
    parser.add_argument('--runtime-mounts-file', type=Path)
    parser.add_argument('--workspace-alias', choices=['/app'],
                        help='Mount each isolated task at the original benchmark /app path')
    parser.add_argument('--snapshot-generated-source', action='store_true',
                        help='Retain observed source/result versions outside the evaluated workspace')
    args = parser.parse_args()
    # Namespace mount destinations and native -C paths must be absolute. Keep
    # the CLI symlink unresolved so its node_modules dependency mount is intact.
    for name in ['plan', 'cli', 'output', 'isolation_config',
                 'credential_env_file', 'model_catalog', 'runtime_mounts_file', 'task_budget_config',
                 'model_gateway_socket']:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(os.path.abspath(value)))
    jobs = json.loads(args.plan.read_text())['jobs']
    try:
        budget_evidence = validate_execution_budget(args.workers, args.timeout, jobs, args.task_budget_config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    if budget_evidence is not None:
        (args.output / 'native_budget_evidence.json').write_text(json.dumps(budget_evidence, indent=2) + '\n')
    if args.permission_profile and not args.isolation_config:
        parser.error('Permission-profile experiments require the outer filesystem namespace')
    if args.workspace_alias and not args.isolation_config:
        parser.error('Workspace alias requires the outer filesystem namespace')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        statuses = list(pool.map(lambda job: replay(job, args), jobs))
    (args.output / 'batch_status.json').write_text(json.dumps(statuses, indent=2) + '\n')


if __name__ == '__main__':
    main()
