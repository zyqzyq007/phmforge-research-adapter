"""Bounded recovery of exactly three contaminated original benchmark tasks."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_full_benchmark_campaign import clean_env, close_worker, launch, now, save, status_path, verify


ALLOWED = {'forge_0005', 'forge_0023', 'forge_0024'}


def check(run):
    protocol = json.loads((run / 'PROTOCOL187.json').read_text())
    scope = json.loads((run / 'RECOVERY_SCOPE.json').read_text())
    jobs = json.loads((run / 'PLAN187.json').read_text())['jobs']
    if set(scope['original_jobs']) != ALLOWED or {r['id'] for r in jobs} != ALLOWED or len(jobs) != 3:
        raise RuntimeError('Recovery scope must be exactly the three confirmed contaminated originals')
    if (protocol['model'] != 'deepseek-flash' or protocol['subject_timeout_seconds'] != 1200 or
            protocol['original_tool_timeout_seconds'] != 180 or protocol['max_concurrency'] != 3 or
            protocol['max_new_primary_rolling24h'] != 400):
        raise RuntimeError('Frozen model, budget or concurrency changed')
    for job in jobs:
        verify(run, job)
    binary = Path(protocol['cli'])
    if hashlib.sha256(binary.read_bytes()).hexdigest() != protocol['cli_sha256']:
        raise RuntimeError('Frozen original CLI changed')
    gate = run / 'RECOVERY_QUALIFICATION_GATE.json'
    if not gate.exists() or json.loads(gate.read_text()).get('passed') is not True:
        raise PermissionError('Actual source-tool/native-offline qualification has not passed')
    qualification = json.loads(gate.read_text())
    cache_record = Path(scope['cache_path']) / 'CACHE_VERIFIED.json'
    cache_digest = hashlib.sha256(cache_record.read_bytes()).hexdigest()
    if (cache_digest != scope['cache_manifest_sha256'] or
            cache_digest != qualification['source_cache_manifest_sha256']):
        raise RuntimeError('Qualified source/cache identity changed')
    permit = run / 'APPROVAL_RECOVERY.json'
    if not permit.exists():
        raise PermissionError('Missing explicit user extension of the old231-only quota scope')
    approval = json.loads(permit.read_text())
    if (approval.get('source') != 'explicit_user' or not approval.get('user_reply') or
            approval.get('run_id') != run.name or set(approval.get('jobs', [])) != ALLOWED or
            approval.get('max_new_model_sessions') != 3 or approval.get('rolling_ceiling') != 400):
        raise PermissionError('Approval does not authorize exactly these three recovery sessions')
    if any((run / 'primary' / j['id']).exists() for j in jobs):
        raise RuntimeError('Recovery already started; inspect the existing controller, never auto-restart')
    for item in json.loads(cache_record.read_text())['files']:
        path = Path(scope['cache_path']) / item['cache_relative_path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Byte-frozen original dependency or data changed: ' + item['cache_relative_path'])
    return protocol, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    try:
        protocol, jobs = check(run)
    except PermissionError as error:
        print(json.dumps({'status': 'blocked_no_model_started', 'reason': str(error), 'model_sessions_started': 0}))
        raise SystemExit(2)
    if args.check_only:
        print(json.dumps({'status': 'authorized_ready', 'original_jobs': sorted(ALLOWED)}))
        return
    if (run / 'CONTROLLER_STATUS_RECOVERY.json').exists():
        raise FileExistsError('Controller already recorded; no automatic restart')
    project = Path(__file__).resolve().parents[1]
    env = clean_env(run)
    broker = subprocess.Popen([sys.executable, '-B', str(project / 'scripts/benchmark_tool_bridge.py'),
        'broker', '--config', str(run / 'BROKER187_FIFO_FILES.json')], stdin=subprocess.DEVNULL,
        stdout=(run / 'logs/broker.stdout').open('ab'), stderr=(run / 'logs/broker.stderr').open('ab'),
        env=env, start_new_session=True)
    save(run / 'BROKER_START_RECOVERY.json', {'pid': broker.pid, 'started_utc': now().isoformat()})
    deadline = time.monotonic() + 30
    while not all((run / 'ipc/fifo_files' / j['id'] / 'request.fifo').exists() for j in jobs):
        if broker.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('Recovery broker did not become ready; inspect it, do not relaunch subjects')
        time.sleep(0.1)
    state = {'controller_pid': os.getpid(), 'started_utc': now().isoformat(), 'planned': 3,
             'launched': [], 'completed': [], 'pending': 3, 'active_jobs': []}
    pending, active, serial = list(jobs), {}, 0
    while pending or active:
        for job_id, (process, job) in list(active.items()):
            if process.poll() is None:
                continue
            try:
                close_worker(run, job)
            except Exception as error:
                save(run / 'audit' / ('CLEANUP_ERROR_' + job_id + '.json'), {'error': str(error)})
            status = json.loads(status_path(run, job).read_text()) if status_path(run, job).exists() else {'status': 'runner_error_no_status'}
            state['completed'].append({'job': job_id, 'status': status['status'], 'runner_exit_code': process.returncode})
            del active[job_id]
        while pending and len(active) < 3:
            serial += 1
            process, admission = launch(run, project, protocol, pending[0], serial, env)
            state['waiting'] = admission if process is None else None
            if process is None:
                break
            job = pending.pop(0)
            active[job['id']] = (process, job)
            state['launched'].append(admission)
        state.update(updated_utc=now().isoformat(), pending=len(pending), active_jobs=list(active))
        save(run / 'CONTROLLER_STATUS_RECOVERY.json', state)
        owner = json.loads((run / 'OWNERSHIP.json').read_text())
        owner.update(controller_pid=os.getpid(), lease_expires_utc=(now() + timedelta(hours=1)).isoformat(),
                     writers_stopped=False, status='recovery_originals_running_or_capacity_wait')
        save(run / 'OWNERSHIP.json', owner)
        if pending or active:
            time.sleep(5)
    broker.terminate()
    broker.wait(timeout=15)
    state.update(status='all3_terminal_unscored', ended_utc=now().isoformat(), broker_exit_code=broker.returncode)
    save(run / 'CONTROLLER_STATUS_RECOVERY.json', state)
    print(json.dumps({'status': state['status'], 'completed': state['completed']}))


if __name__ == '__main__':
    main()
