"""Cold replay a saved coding submission, retaining legitimate dependencies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--inputs', default='test.json')
    p.add_argument('--runtime-mounts-file', type=Path)
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output/'workspace'
    def ignored(directory, names):
        return {n for n in names if n == '__pycache__' or
                Path(directory) == source and n in {'predictions.json', 'REPORT.md'}}
    # Shared readonly input trees may be mounted outside /app and reached by
    # symlink. Preserve those references rather than dereferencing a path that
    # only exists inside the isolated namespace or copying the data per replay.
    shutil.copytree(source, work, ignore=ignored, symlinks=True)
    relative_inputs = Path(a.inputs)
    if relative_inputs.is_absolute() or '..' in relative_inputs.parts:
        raise ValueError('Inputs must use a relative workspace path')
    shared_mounts = (json.loads(a.runtime_mounts_file.read_text())['mounts']
                     if a.runtime_mounts_file else [])
    resolved_inputs = (work/relative_inputs).resolve()
    if not resolved_inputs.is_relative_to(work) and not any(
            resolved_inputs.is_relative_to(Path(entry['destination'])) for entry in shared_mounts):
        raise ValueError('Inputs must be in the workspace or an explicitly declared readonly mount')
    runtime = work/'.runtime'
    for name in ['tmp', 'mpl']:
        (runtime/name).mkdir(parents=True, exist_ok=True)
    command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session']
    for path in ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc', str(Path.home()/'.local/lib')]:
        if Path(path).exists():
            command += ['--ro-bind', path, path]
    for entry in shared_mounts:
        command += ['--ro-bind', entry['source'], entry['destination']]
    command += ['--bind', str(work), '/app']
    for name in json.loads((source/'readonly_inputs.json').read_text()):
        command += ['--ro-bind', str(source/name), '/app/'+name]
    command += ['--proc', '/proc', '--dev', '/dev', '--chdir', '/app', '--remount-ro', '/',
                'python3', 'extract.py', '--inputs', a.inputs, '--out', 'predictions.json']
    env = {k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['KEY','TOKEN','SECRET','PASSWORD','PROXY'])}
    env.update(TMPDIR='/app/.runtime/tmp', TMP='/app/.runtime/tmp', TEMP='/app/.runtime/tmp',
               MPLCONFIGDIR='/app/.runtime/mpl', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    if a.runtime_mounts_file:
        env['PYTHONPATH'] = '/opt/phm_python/jax:/opt/phm_python/base:/opt/phm_python/extra'
    status = dict(status='running', command=command, timeout=a.timeout,
                  started_utc=datetime.now(timezone.utc).isoformat(), model_calls=0,
                  output_answers_copied=False, other_dependencies_preserved=True)
    path = output/'status.json'
    path.write_text(json.dumps(status, indent=2)+'\n')
    start = time.monotonic()
    with (output/'stdout.txt').open('w') as stdout, (output/'stderr.txt').open('w') as stderr:
        child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            child.wait(a.timeout)
            status.update(status='finished' if child.returncode == 0 else 'execution_error', exit_code=child.returncode)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            status.update(status='timeout', exit_code=child.returncode)
    target = work/'predictions.json'
    original = source/'predictions.json'
    status.update(elapsed_seconds=time.monotonic()-start,
        ended_utc=datetime.now(timezone.utc).isoformat(),
        predictions_present=target.is_file(),
        predictions_byte_equal=target.is_file() and original.is_file() and target.read_bytes()==original.read_bytes())
    path.write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
