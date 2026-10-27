"""Schedule original SmartGrid tasks through existing CLI and AaT entry points."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

from replay_diagnostic_cli import replay


def sdk_replay(job, args):
    folder = args.output / job['id']
    folder.mkdir()
    workspace = args.plan.parent / job['workspace']
    scratch = workspace / '.runtime'
    for name in ('tmp', 'mpl'):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    command = [str(args.python), str(args.sdk_runner), '--source', str(args.source),
               '--prompt', str(args.plan.parent / job['prompt_file']),
               '--output', str(folder / 'trajectory.json'),
               '--server-python', str(args.server_python),
               '--credential-file', str(args.credential_file),
               '--max-turns', str(args.max_turns)]
    state = dict(job_id=job['id'], scenario_id=job['scenario_id'],
                 condition=job['condition'], repeat=job['repeat'], status='running',
                 model_requested='deepseek-flash', command=command,
                 timeout_seconds=args.timeout, max_turns=args.max_turns,
                 started_utc=datetime.now(timezone.utc).isoformat())
    path = folder / 'status.json'
    path.write_text(json.dumps(state, indent=2) + '\n')
    env = {k: v for k, v in os.environ.items()
           if not any(w in k.upper() for w in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
    env.update(TMPDIR=str(scratch / 'tmp'), TMP=str(scratch / 'tmp'),
               TEMP=str(scratch / 'tmp'), MPLCONFIGDIR=str(scratch / 'mpl'),
               PYTHONDONTWRITEBYTECODE='1', OPENAI_AGENTS_DISABLE_TRACING='1',
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    started = time.monotonic()
    with (folder / 'stdout.txt').open('w') as out, (folder / 'stderr.txt').open('w') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, env=env,
                                 start_new_session=True)
        state['pid'] = child.pid
        path.write_text(json.dumps(state, indent=2) + '\n')
        try:
            child.wait(timeout=args.timeout)
            state.update(status='finished' if child.returncode == 0 else 'execution_error',
                         exit_code=child.returncode)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            state.update(status='timeout', exit_code=child.returncode)
    state.update(elapsed_seconds=time.monotonic() - started,
                 ended_utc=datetime.now(timezone.utc).isoformat())
    if (folder / 'trajectory.json').exists():
        result = json.loads((folder / 'trajectory.json').read_text())
        state.update(response_valid=bool(result.get('answer')), usage=result.get('usage'),
                     tool_call_count=result.get('tool_call_count'), error=result.get('error'))
    path.write_text(json.dumps(state, indent=2) + '\n')
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = SimpleNamespace(**json.loads(parser.parse_args().config.read_text()))
    for name in ('plan', 'output', 'cli', 'isolation_config', 'runtime_mounts_file',
                 'credential_env_file', 'model_catalog', 'python', 'sdk_runner',
                 'source', 'server_python', 'credential_file'):
        setattr(args, name, Path(getattr(args, name)))
    if args.model != 'deepseek-flash':
        parser.error('Only deepseek-flash is authorized')
    if args.output.resolve().is_relative_to(Path('/tmp')):
        parser.error('Task work under /tmp is forbidden')
    args.output.mkdir(exist_ok=False)
    jobs = json.loads(args.plan.read_text())['jobs']

    def run(job):
        try:
            state = sdk_replay(job, args) if job['condition'] == 'official_aat' else replay(job, args)
        except Exception as exc:
            state = dict(job_id=job['id'], scenario_id=job['scenario_id'],
                         condition=job['condition'], status='scheduler_error',
                         error=f'{type(exc).__name__}: {exc}')
            folder = args.output / job['id']
            folder.mkdir(exist_ok=True)
            (folder / 'scheduler_error.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps({k: state.get(k) for k in
                         ('job_id', 'status', 'response_valid', 'elapsed_seconds')}), flush=True)
        return state

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        states = list(pool.map(run, jobs))
    (args.output / 'batch_status.json').write_text(json.dumps(states, indent=2) + '\n')
    print(json.dumps({'planned': len(jobs), 'terminal': len(states)}), flush=True)


if __name__ == '__main__':
    main()
