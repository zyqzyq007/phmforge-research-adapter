"""Quota-aware scheduler for a frozen full pass; never reads live answers."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from tool_bridge_client import fifo_request


def now():
    return datetime.now(timezone.utc)


def save(path, value):
    temp = path.with_name(path.name + '.new')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def clean_env(run):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('CODEX_') and
           not any(x in k.upper() for x in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
    # A background parent once passed a home that was archived and removed.
    # Use the native CLI's existing isolated home binding, without changing it.
    env.pop('CODEX_HOME', None)
    env.update(TMPDIR=str(run / 'runtime/tmp'), TMP=str(run / 'runtime/tmp'),
               TEMP=str(run / 'runtime/tmp'), MPLCONFIGDIR=str(run / 'runtime/mpl'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    return env


def status_path(run, job):
    return run / 'primary' / job['id'] / job['id'] / 'status.json'


def close_worker(run, job):
    config_path = run / 'BROKER187_FIFO_FILES.json'
    if not config_path.exists():
        config_path = run / 'BROKER187_FIFO.json'
    if not config_path.exists():
        config_path = run / 'BROKER187.json'
    config = json.loads(config_path.read_text())
    cap = next(k for k, v in config['routes'].items() if v['job'] == job['id'])
    if config.get('transport') == 'fifo-files':
        result = fifo_request(Path(config['fifo_root']) / job['id'], cap,
                              'close_worker', {}, timeout=40,
                              request_root=config['routes'][cap]['request_root'])
        if not result.get('transport_ok'):
            raise RuntimeError(result)
        return
    if config.get('transport') == 'fifo':
        folder = Path(config['fifo_root']) / job['id']
        with (folder / 'request.fifo').open('w') as writer:
            writer.write(json.dumps(dict(route=cap, tool='close_worker', args={})) + '\n')
        with (folder / 'response.fifo').open() as reader:
            result = json.loads(reader.readline())
        if not result.get('transport_ok'):
            raise RuntimeError(result)
        return
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as handle:
        handle.settimeout(210)
        handle.connect(config['socket'])
        handle.sendall((json.dumps(dict(route=cap, tool='close_worker', args={})) + '\n').encode())
        result = json.loads(handle.makefile('rb').readline(1024 * 1024))
    if not result.get('transport_ok'):
        raise RuntimeError(result)


def verify(run, job):
    freeze = run / 'INITIAL_FREEZE187_FIFO_FILES.json'
    if not freeze.exists():
        freeze = run / 'INITIAL_FREEZE187_FIFO.json'
    if not freeze.exists():
        freeze = run / 'INITIAL_FREEZE187.json'
    targets = json.loads(freeze.read_text())['files']
    amendment = run / 'audit/PATH_AMENDMENT187.json'
    if amendment.exists() and not (run / 'INITIAL_FREEZE187_FIFO.json').exists():
        targets += json.loads(amendment.read_text())['files']
    own_workspace = run / job['workspace']
    for item in targets:
        path = Path(item['path'])
        if path.is_relative_to(run / 'workspaces') and not path.is_relative_to(own_workspace):
            continue
        if path.is_relative_to(run / 'workspaces') or not path.name.endswith('_mounts.json'):
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise RuntimeError('Frozen source or input changed: ' + str(path))


def launch(run, project, protocol, job, serial, environment):
    lock = (project / 'logs/background_research.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None, dict(reason='shared_scheduler_lock_busy')
    try:
        if (project / 'docs/BACKGROUND_STOP').exists():
            return None, dict(reason='user_stop_file')
        output = run / 'primary' / job['id']
        if output.exists():
            raise FileExistsError('No automatic rerun: ' + str(output))
        verify(run, job)
        capacity_path = run / 'audit' / ('CAPACITY_LAUNCH_' + str(serial).zfill(4) + '.json')
        result = subprocess.run([sys.executable, '-B', str(project / 'scripts/recount_subject_capacity.py'),
                                 '--output', str(capacity_path)], capture_output=True, text=True,
                                timeout=120, env=environment)
        if result.returncode:
            raise RuntimeError('Capacity audit failed: ' + result.stderr[-1000:])
        capacity = json.loads(capacity_path.read_text())
        if capacity['errors']:
            raise RuntimeError('Capacity audit has unresolved records')
        counted = [x for x in capacity['rows'] if x['counted_for_capacity']]
        running = []
        for row in counted:
            if row['status'] == 'running':
                try:
                    os.kill(row['pid'], 0)
                    running.append(row)
                except ProcessLookupError:
                    pass
        if len(counted) >= protocol['max_new_primary_rolling24h']:
            release = min(datetime.fromisoformat(x['started_utc'].replace('Z', '+00:00'))
                          + timedelta(hours=24) for x in counted)
            return None, dict(reason='rolling24h_capacity', counted=len(counted),
                              next_release_utc=release.isoformat())
        if len(running) >= protocol['max_concurrency']:
            return None, dict(reason='concurrent_capacity', actual_running=len(running))
        command = ['/usr/bin/taskset', '-c', '0-15', sys.executable, '-B', '-u',
                   str(project / 'scripts/replay_diagnostic_cli.py'),
                   '--plan', job['plan'], '--cli', protocol['cli'],
                   '--model', protocol['model'], '--output', str(output),
                   '--workers', '1', '--timeout', str(protocol['subject_timeout_seconds']),
                   '--analysis-tools', '--permission-profile', 'iot-offline',
                   '--isolation-config', str(run / 'cli_config187.toml'),
                   '--model-catalog', str(run / 'model_catalog187.json'),
                   '--runtime-mounts-file', job['mounts'], '--workspace-alias', '/app',
                   '--snapshot-generated-source', '--cleanup-runtime-cache']
        if protocol.get('external_model_gateway'):
            gateway = json.loads((run / 'MODEL_GATEWAY_READY.json').read_text())
            os.kill(gateway['pid'], 0)
            command += ['--model-gateway-socket', job['model_gateway_socket']]
        else:
            command += ['--credential-env-file', '/home/jupyter-yyf-zyq/.bashrc',
                        '--credential-env-name', 'DEEPSEEK_API_KEY']
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=(run / 'logs' / (job['id'] + '.stdout')).open('ab'),
                                   stderr=(run / 'logs' / (job['id'] + '.stderr')).open('ab'),
                                   env=environment, cwd=project, start_new_session=True)
        record = dict(job=job['id'], family=job['family'], task_id=job['task_id'],
                      runner_pid=process.pid, started_utc=now().isoformat(), command=command,
                      capacity_before=len(counted), reservation_included=len(counted) + 1,
                      actual_other_running=len(running))
        save(run / 'audit' / ('LAUNCH_' + job['id'] + '.json'), record)
        # Hold the shared launch lock until the native runner publishes its
        # reservation, or the child exits. Never launch a second invisible job.
        deadline = time.monotonic() + 15
        while not status_path(run, job).exists() and process.poll() is None:
            if time.monotonic() > deadline:
                raise RuntimeError('Native runner has not published a reservation; inspect child')
            time.sleep(0.1)
        return process, record
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--first-only', action='store_true')
    args = ap.parse_args()
    run = args.run.resolve()
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads((run / 'PROTOCOL187.json').read_text())
    plan = run / 'PLAN187_LAUNCH.json'
    if not plan.exists():
        plan = run / 'PLAN187.json'
    all_jobs = json.loads(plan.read_text())['jobs']
    env = clean_env(run)
    first = next(x for x in all_jobs if x['family'] == 'asset' and x['task_id'] == '1')
    if args.first_only:
        jobs = [first]
    else:
        gate = json.loads((run / 'FIRST_SUBJECT_GATE187.json').read_text())
        if not gate.get('passed'):
            raise RuntimeError('First original subject has not passed operational qualification')
        jobs = [x for x in all_jobs if not status_path(run, x).exists()]
    progress = dict(controller_pid=os.getpid(), started_utc=now().isoformat(),
                    first_only=args.first_only, planned=len(all_jobs),
                    selected_this_controller=len(jobs), launched=[], completed=[], pending=len(jobs))
    pending = list(jobs)
    active = {}
    serial = len(list((run / 'audit').glob('CAPACITY_LAUNCH_*.json')))
    while pending or active:
        for job_id, (process, job) in list(active.items()):
            if process.poll() is None:
                continue
            try:
                close_worker(run, job)
            except Exception as exc:
                # A cleanup problem must not freeze every other subject's
                # admission/heartbeat. Preserve it for runtime validity audit.
                save(run / 'audit' / ('CLEANUP_ERROR_' + job_id + '.json'),
                     dict(utc=now().isoformat(), error_type=type(exc).__name__,
                          error=str(exc), terminal_subject=True))
            p = status_path(run, job)
            row = json.loads(p.read_text()) if p.exists() else dict(status='runner_error_no_status')
            progress['completed'].append(dict(job=job_id, status=row['status'],
                                               runner_exit_code=process.returncode,
                                               ended_utc=now().isoformat()))
            del active[job_id]
        waiting = None
        while pending and len(active) < protocol['max_concurrency']:
            serial += 1
            process, admission = launch(run, project, protocol, pending[0], serial, env)
            if process is None:
                waiting = admission
                break
            job = pending.pop(0)
            active[job['id']] = (process, job)
            progress['launched'].append(admission)
            if args.first_only:
                break
        progress.update(updated_utc=now().isoformat(), pending=len(pending),
                        active_jobs=list(active), waiting=waiting,
                        terminal_counts=dict(Counter(x['status'] for x in progress['completed'])))
        save(run / ('FIRST_CONTROLLER_STATUS187.json' if args.first_only else 'CONTROLLER_STATUS187.json'), progress)
        ownership = json.loads((run / 'OWNERSHIP.json').read_text())
        ownership.update(lease_expires_utc=(now() + timedelta(hours=1)).isoformat(),
                         controller_pid=os.getpid(), writers_stopped=False,
                         status='original_subjects_running_or_quota_wait', heartbeat_utc=now().isoformat())
        save(run / 'OWNERSHIP.json', ownership)
        if pending or active:
            time.sleep(15)
    progress.update(ended_utc=now().isoformat(), status='all_selected_subjects_terminal_no_scoring')
    save(run / ('FIRST_CONTROLLER_STATUS187.json' if args.first_only else 'CONTROLLER_STATUS187.json'), progress)
    print(json.dumps(dict(status=progress['status'], launched=len(progress['launched']),
                          terminal_counts=progress['terminal_counts'])), flush=True)


if __name__ == '__main__':
    main()
