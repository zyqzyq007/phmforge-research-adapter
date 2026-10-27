"""Execute one frozen discovery probe and retain all CLI events and status."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True, type=Path)
args = parser.parse_args()
run = args.run.resolve()
cli = run / 'clients' / 'node_modules' / '.bin' / 'codex'
command = [str(cli), 'exec', '--ephemeral', '--skip-git-repo-check', '--json',
           '--sandbox', 'read-only', '-m', 'gpt-6.1-sol',
           '-c', 'model_reasoning_effort="max"', '-c', 'approval_policy="never"',
           '-C', str(run / 'probe_inputs'), '--output-schema', str(run / 'probe_schema.json'),
           '-o', str(run / 'probe_response.json'), '-']
env = os.environ.copy()
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
    env.pop(key, None)
status = {'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
          'network_route': 'direct', 'timeout_seconds': 300, 'status': 'running',
          'input_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted((run / 'probe_inputs').glob('*.json'))},
          'prompt_sha256': hashlib.sha256((run / 'probe_prompt.txt').read_bytes()).hexdigest()}
status_path = run / 'probe_status.json'
status_path.write_text(json.dumps(status, indent=2) + '\n')
start = time.monotonic()
with (run / 'probe_events.jsonl').open('w') as events, (run / 'probe_stderr.txt').open('w') as errors:
    process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=events,
                               stderr=errors, text=True, start_new_session=True)
    try:
        process.communicate((run / 'probe_prompt.txt').read_text(), timeout=300)
        status.update(exit_code=process.returncode,
                      status='finished' if process.returncode == 0 else 'execution_error')
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        status.update(status='timeout', exit_code=process.returncode)
status.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start)
status_path.write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps({'status': status['status'], 'exit_code': status['exit_code'],
                  'elapsed_seconds': round(status['elapsed_seconds'], 2)}), flush=True)
