"""Freeze native artifacts, then replay unchanged Python in isolated clean folders."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def freeze(root, jobs, native_directory='formal'):
    inventories = {}
    for job in jobs:
        status = json.loads((root / native_directory / job['id'] / 'status.json').read_text())
        assert status['status'] in ('finished', 'execution_error', 'timeout')
        workspace = root / job['workspace']
        inventories[job['id']] = {
            str(p.relative_to(workspace)): {'sha256': digest(p), 'bytes': p.stat().st_size}
            for p in sorted(workspace.rglob('*'))
            if p.is_file() and not p.is_symlink()
            and p.relative_to(workspace).parts[0] not in ('inputs', '__pycache__', '.runtime')
        }
    target = root / 'native_artifact_manifest.json'
    assert not target.exists(), 'Do not replace frozen original evidence'
    save(target, {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'jobs': inventories})
    return inventories


def replay(root, job, inventory, timeout, runtime_mounts_file=None):
    original = root / job['workspace']
    alias = job.get('workspace_alias')
    if alias is not None and alias != '/app':
        raise ValueError('Only the documented /app workspace alias is supported')
    visible = Path(alias) if alias else original
    destination = root / 'scientific_replay' / job['id']
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('tmp', 'mpl'):
        (destination / '.runtime' / name).mkdir(parents=True, exist_ok=True)
    if (original / 'inputs').is_dir():
        shutil.copytree(original / 'inputs', destination / 'inputs')
    else:
        (destination / 'inputs').mkdir()
    copied = []
    for relative, entry in inventory.items():
        assert digest(original / relative) == entry['sha256']
        if Path(relative).suffix == '.py':
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original / relative, target)
            copied.append(relative)
    logs = root / 'replay_logs' / job['id']
    logs.mkdir(parents=True, exist_ok=False)
    command = ['bwrap', '--unshare-user', '--unshare-pid', '--unshare-ipc',
               '--unshare-uts', '--unshare-net', '--die-with-parent', '--new-session']
    for name in ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc', str(Path.home() / '.local/lib')]:
        if Path(name).exists():
            command += ['--ro-bind', name, name]
    if runtime_mounts_file:
        for entry in json.loads(runtime_mounts_file.read_text())['mounts']:
            command += ['--ro-bind', entry['source'], entry['destination']]
    command += ['--bind', str(destination), str(visible), '--ro-bind',
                str(destination / 'inputs'), str(visible / 'inputs')]
    # Reuse large immutable flat-layout data without copying them into replays.
    # Only the current job's declared inputs are exposed at their original paths.
    for relative in job.get('readonly_input_paths', []):
        source = original / relative
        if not source.resolve().is_relative_to(original.resolve()) or not source.exists():
            raise ValueError('Replay input must exist inside the original job')
        command += ['--ro-bind', str(source), str(visible / relative)]
    command += ['--proc', '/proc', '--dev', '/dev',
                '--chdir', str(visible), job.get('python_command', 'python3'),
                job.get('entry_script', 'analysis.py')]
    command.extend(job.get('entry_args', []))
    env = {k: v for k, v in os.environ.items()
           if not any(word in k.upper() for word in ['KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PROXY'])}
    threads = str(job.get('numerical_threads', 4))
    env.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(visible / '.runtime' / 'tmp'),
               TMP=str(visible / '.runtime' / 'tmp'), TEMP=str(visible / '.runtime' / 'tmp'),
               MPLCONFIGDIR=str(visible / '.runtime' / 'mpl'))
    if runtime_mounts_file:
        env['PYTHONPATH'] = '/opt/phm_python/jax:/opt/phm_python/base:/opt/phm_python/extra:' + str(visible / 'inputs/runtime')
    status = {'job_id': job['id'], 'status': 'running', 'command': command,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'copied_source_files': copied, 'generated_outputs_copied': False,
              'timeout_seconds': timeout, 'network': False, 'LLM_calls': 0,
              'workspace_alias': alias}
    save(logs / 'status.json', status)
    start = time.monotonic()
    with (logs / 'stdout.txt').open('w') as out, (logs / 'stderr.txt').open('w') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
        status['pid'] = process.pid
        save(logs / 'status.json', status)
        try:
            process.wait(timeout=timeout)
            status.update(status='finished' if process.returncode == 0 else 'execution_error',
                          exit_code=process.returncode)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            status.update(status='timeout', exit_code=process.returncode)
    status.update(elapsed_seconds=time.monotonic() - start,
                  ended_utc=datetime.now(timezone.utc).isoformat(),
                  original_artifacts_unchanged=all(digest(original / p) == e['sha256']
                                                   for p, e in inventory.items()),
                  replay_source_unchanged=all(digest(destination / p) == inventory[p]['sha256']
                                              for p in copied),
                  artifact_byte_equality={p: (destination / p).is_file() and
                                          digest(destination / p) == inventory[p]['sha256']
                                          for p in job.get('replay_artifacts', ['results.json', 'report.md'])
                                          if p in inventory})
    save(logs / 'status.json', status)
    print(json.dumps({k: status[k] for k in ['job_id', 'status', 'elapsed_seconds']}), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'replay'])
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--native-directory', choices=['formal', 'native', 'native_runs'], default='formal',
                        help='Directory holding the frozen native CLI status files')
    parser.add_argument('--runtime-mounts-file', type=Path,
                        help='Read-only shared inputs available at identical native and replay paths')
    args = parser.parse_args()
    root = args.run.resolve()
    jobs = json.loads((root / args.plan).read_text())['jobs']
    if args.mode == 'freeze':
        print(json.dumps({'jobs_frozen': list(freeze(root, jobs, args.native_directory))}))
    else:
        inventory = json.loads((root / 'native_artifact_manifest.json').read_text())['jobs']
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(lambda j: replay(root, j, inventory[j['id']], args.timeout,
                                                     args.runtime_mounts_file), jobs))
        save(root / 'replay_summary.json', statuses)
