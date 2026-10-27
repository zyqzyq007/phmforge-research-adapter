"""Run a coding agent with training-only data and a task-scoped model relay."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

from .data import file_sha256, write_json
from .runner import _ready, evaluate, train_reference, workspace


def run_agent(data_root, run_dir, cli, base_url, epochs=1, timeout=1200,
              max_requests=40, max_input_tokens=4000000, max_output_tokens=65536,
              max_output_per_request=8192, smoke=False):
    """Develop a submission without mounting holdout data; then cold-score it.

    The Responses-compatible provider and credentials remain in the outer
    process. The agent can use only the task-scoped relay in its network namespace.
    """
    root, meta = _ready(data_root)
    run, cli = Path(run_dir).resolve(), Path(cli).resolve()
    if run.exists():
        raise FileExistsError(run)
    if run.is_relative_to(root) or root.is_relative_to(run):
        raise ValueError('Run directory and evaluation data root must be separate')
    url = urlsplit(base_url)
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Model base URL must be HTTPS and contain no credentials, query, or fragment')
    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise ValueError('Set DEEPSEEK_API_KEY in the environment of the outer process')
    if not cli.is_file() or not os.access(cli, os.X_OK):
        raise FileNotFoundError('Supply the native executable of a compatible Codex CLI')
    if min(epochs, timeout, max_requests, max_input_tokens, max_output_tokens, max_output_per_request) < 1:
        raise ValueError('All training, time, and model budgets must be positive')
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise FileNotFoundError('bubblewrap is required for isolated agent execution')
    version = subprocess.run([str(cli), '--version'], capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    if not version.startswith('codex-cli '):
        raise ValueError('Expected a native Codex CLI executable')
    run.mkdir(parents=True)
    ws = run/'workspace'
    socket_path = run/'model.sock'
    if len(os.fsencode(socket_path)) >= 104:
        raise ValueError('Run path is too long for a Unix-domain model gateway socket')
    for name in ['client-home', 'gateway', 'logs']:
        (run/name).mkdir()
    meta_protocol = {'task_id':meta['task_id'], 'dataset':meta['dataset'],
        'model_requested':'deepseek-flash', 'provider_base_url':base_url.rstrip('/'),
        'cli_version':version, 'cli_sha256':file_sha256(cli), 'timeout_seconds':timeout,
        'max_requests':max_requests, 'max_total_input_tokens_admission_threshold':max_input_tokens,
        'max_total_output_tokens':max_output_tokens, 'max_output_per_request':max_output_per_request,
        'network':'isolated namespace with task-scoped Responses relay',
        'development_data':'training and development only', 'final_observations_and_labels_in_development':False,
        'starter':'genuinely fitted reference, independently replaceable', 'starter_epochs':epochs,
        'mode':'engineering smoke test' if smoke else 'open research task',
        'actual_returned_model_ids':'see model_gateway status and receipts',
        'scope':'engineering / research adaptation, not an official PHMForge score',
        'created_utc':datetime.now(timezone.utc).isoformat()}
    write_json(run/'PROTOCOL.json', meta_protocol)
    workspace(root, ws)
    starter = train_reference(root, ws/'submission', epochs)
    for name in ['tmp', 'mpl', 'xdg']:
        (ws/'.runtime'/name).mkdir(parents=True)
    (ws/'AGENTS.md').write_text('Use only the provided training and development files. Keep all temporary files in /app/.runtime. Final inputs, labels and scores are unavailable. Submit executable code and any model files under /app/submission.\n')
    prompt = ('Read /app/TASK.md. A genuinely trained initial solution is in /app/submission. '
              'Inspect the task, training and development data, and use the PHMForge MCP tools or local code as appropriate. '
              'You may retain, modify or replace the initial submission. Save a complete executable submission in '
              '/app/submission/solution.py and any required model files in that directory. '
              f'The whole session is limited to {timeout} seconds, including preparation and submission; '
              'choose bounded analyses and leave enough time to submit. '
              'Do not use final data or infer physical safety claims. Report what you did and its limitations.')
    if smoke:
        prompt += (' This is only an engineering smoke test, not a scientific comparison. '
                   'Inspect the task and the initial submission, call an MCP read tool, and then retain '
                   'the existing executable submission. Do not run broad model searches or long training jobs. '
                   'Finish promptly and report that this was an infrastructure check.')
    (run/'PROMPT.txt').write_text(prompt+'\n')
    write_json(run/'MODEL_GATEWAY_CONFIG.json', {'routes':[{'job':'agent','socket':str(socket_path)}],
        'base_url':base_url.rstrip('/'), 'max_requests':max_requests, 'max_input_tokens':max_input_tokens,
        'max_output_tokens':max_output_tokens, 'max_output_per_request':max_output_per_request,
        'model':'deepseek-flash', 'credential_source':'outer environment, never agent namespace',
        'input_budget_limit':'stop admitting new requests after completed usage crosses threshold'})
    tools = Path(__file__).resolve().parents[1]
    gateway = Path(__file__).resolve().with_name('gateway.py')
    python = Path(os.path.abspath(sys.executable))
    extra = []
    for val in os.environ.get('PYTHONPATH','').split(os.pathsep):
        if val:
            p = Path(val).resolve()
            if not p.is_dir() or p == Path('/') or root.is_relative_to(p) or p.is_relative_to(root):
                raise ValueError('Unsafe PYTHONPATH entry in agent runtime')
            extra.append(p)
    py_path = ':'.join(['/tools']+[str(p) for p in extra])
    config = f'''default_permissions = "phm-offline"
model_catalog_json = "/tools/phmforge_bench/resources/deepseek_model_catalog.json"
model = "deepseek-flash"
model_provider = "isolated_deepseek"
model_reasoning_effort = "max"
web_search = "disabled"
approval_policy = "never"
[model_providers.isolated_deepseek]
name = "Task-scoped model relay"
base_url = "http://127.0.0.1:47777/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
stream_idle_timeout_ms = 620000
[features]
apps = false
multi_agent = false
[shell_environment_policy]
exclude = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
[permissions.phm-offline.filesystem]
":root" = "write"
":tmpdir" = "write"
[permissions.phm-offline.filesystem.":workspace_roots"]
"." = "write"
[permissions.phm-offline.network]
enabled = false
[mcp_servers.phmforge]
command = {json.dumps(str(python))}
args = ["-B", "/tools/verified_benchmark_worker.py", "--family", "forge", "--data", "/data", "--output", "/app/artifacts", "--transport", "mcp"]
required = true
env = {{ PYTHONPATH = {json.dumps(py_path)}, PYTHONDONTWRITEBYTECODE = "1", OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "1", TMPDIR = "/app/.runtime/tmp", TMP = "/app/.runtime/tmp", TEMP = "/app/.runtime/tmp", MPLCONFIGDIR = "/app/.runtime/mpl", XDG_CACHE_HOME = "/app/.runtime/xdg" }}
startup_timeout_sec = 60
tool_timeout_sec = 180
'''
    (run/'client-home/config.toml').write_text(config)
    args = [bwrap, '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts', '--unshare-net',
            '--die-with-parent', '--new-session', '--clearenv']
    mounted = set()
    def bind(path):
        path = Path(os.path.abspath(path)); resolved = path.resolve()
        if str(path) in mounted:
            return
        if resolved == Path('/') or root.is_relative_to(resolved) or resolved.is_relative_to(root):
            raise ValueError('Runtime bind would expose the evaluation data root')
        args.extend(['--ro-bind',str(path),str(path)])
        mounted.add(str(path))
    for path in [Path(p) for p in ['/usr','/bin','/sbin','/lib','/lib64','/etc'] if Path(p).exists()]: bind(path)
    for path in [Path(sys.prefix),Path(sys.base_prefix),*extra]: bind(path)
    args += ['--dir','/opt', '--dir','/home', '--dir','/home/agent',
        '--ro-bind',str(cli),'/opt/codex', '--ro-bind',str(tools),'/tools',
        '--ro-bind',str(socket_path),'/model_gateway.sock',
        '--bind',str(ws),'/app', '--ro-bind',str(ws/'data'),'/app/data',
        '--ro-bind',str(ws/'data'),'/data', '--bind',str(run/'client-home'),'/home/agent/.codex',
        '--proc','/proc', '--dev','/dev', '--chdir','/app', '--remount-ro','/']
    env = {'HOME':'/home/agent', 'PATH':str(python.parent)+':/usr/bin:/bin', 'PYTHONPATH':py_path,
        'PYTHONDONTWRITEBYTECODE':'1', 'TMPDIR':'/app/.runtime/tmp', 'TMP':'/app/.runtime/tmp',
        'TEMP':'/app/.runtime/tmp', 'MPLCONFIGDIR':'/app/.runtime/mpl', 'XDG_CACHE_HOME':'/app/.runtime/xdg',
        'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'2'}
    for k,v in env.items(): args += ['--setenv',k,v]
    args += [str(python),'-B','/tools/phmforge_bench/gateway.py','relay','--socket','/model_gateway.sock','--port','47777','--',
             '/opt/codex','exec','--disable','apps','--ephemeral','--skip-git-repo-check','--json',
             '-m','deepseek-flash','-C','/app','-o','/app/last_message.txt','-']
    write_json(run/'ISOLATION.json', {'arguments':args, 'agent_environment':env, 'credential_in_agent':False})
    gateway_env = {k:v for k,v in os.environ.items() if k in ['PATH','DEEPSEEK_API_KEY','TMPDIR','TMP','TEMP','MPLCONFIGDIR','XDG_CACHE_HOME']}
    gateway_process = None
    started = time.monotonic()
    try:
        with (run/'logs/gateway_stdout.log').open('w') as gout, (run/'logs/gateway_stderr.log').open('w') as gerr:
            gateway_process = subprocess.Popen([str(python),'-B',str(gateway),'serve','--run',str(run)],
                                               stdout=gout, stderr=gerr, env=gateway_env)
            deadline = time.monotonic()+20
            while not socket_path.exists():
                if gateway_process.poll() is not None: raise RuntimeError('Model gateway exited before readiness; inspect gateway logs')
                if time.monotonic()>=deadline: raise TimeoutError('Model gateway readiness timed out')
                time.sleep(.1)
            with (run/'logs/agent_events.jsonl').open('w') as out, (run/'logs/agent_stderr.log').open('w') as err:
                result = subprocess.run(args, input=prompt, text=True, stdout=out, stderr=err, timeout=timeout, check=False)
            if result.returncode:
                raise RuntimeError(f'Coding agent exited with status {result.returncode}; inspect agent logs')
            state = json.loads((run/'model_gateway/agent/status.json').read_text())
            if not state['requests'] or not state['returned_model_ids']:
                raise RuntimeError('No completed model response was recorded; agent run is not valid')
            scored = evaluate(root, ws/'submission', run/'evaluation', timeout=120)
            report = {'status':'completed','task_id':meta['task_id'], 'smoke_mode':smoke, 'starter_training':starter,
                'score':scored,'agent_elapsed_seconds':time.monotonic()-started,
                'returned_models_and_usage':'model_gateway/agent/status.json'}
            write_json(run/'RUN.json',report)
            return report
    except Exception as exc:
        write_json(run/'RUN.json',{'status':'failed','task_id':meta['task_id'],
            'error_type':type(exc).__name__,'error':str(exc),'elapsed_seconds':time.monotonic()-started})
        raise
    finally:
        if gateway_process is not None:
            (run/'MODEL_GATEWAY_STOP').touch()
            try: gateway_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                gateway_process.terminate()
                try: gateway_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    gateway_process.kill(); gateway_process.wait()
                write_json(run/'logs/gateway_cleanup.json',{'forced_stop':True,'note':'Inspect receipts for any incomplete request'})
