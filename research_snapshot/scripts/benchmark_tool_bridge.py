"""Transport original PHM benchmark tools through isolated persistent workers.

No LLM credentials, task answers, scorer, or research history belong in workers.
Worker commands and per-task capabilities are frozen in the broker config.
"""
import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import select
import socketserver
import subprocess
import sys
import threading
import time


def clean_environment(extra=None):
    env = {k: v for k, v in os.environ.items()
           if not any(x in k.upper() for x in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
    env.update(extra or {})
    return env


def json_value(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if isinstance(value, (tuple, list)):
        return [json_value(x) for x in value]
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    return value


class ForgeTools:
    def __init__(self):
        if os.environ.get("PHM_ALLOW_ORIGINAL_SIMULATED_TOOLS") != "1":
            raise RuntimeError(
                "Original PHMForge model tools are simulated/label-dependent. "
                "Use verified_benchmark_worker.py for new experiments. "
                "Set PHM_ALLOW_ORIGINAL_SIMULATED_TOOLS=1 only for explicitly labelled source reproduction.")
        from tools.data_tools import LoadDatasetTool, LoadGroundTruthTool
        from tools.model_tools import TrainRULModelTool, PredictRULTool, TrainFaultClassifierTool, ClassifyFaultsTool
        from tools.metric_tools import CalculateMAETool, CalculateRMSETool, VerifyGroundTruthTool, CalculateAccuracyTool, VerifyClassificationTool
        from tools.analysis_tools import AnalyzeEngineSignalsTool, AssessComponentHealthTool, DiagnoseTimingIssuesTool, DetectDegradationTrendTool, CalculateMaintenanceCostTool, CalculateFailureCostTool, OptimizeMaintenanceScheduleTool, AssessSafetyRiskTool, CheckComplianceTool, GenerateSafetyRecommendationsTool
        from tools.web_search_tool import WebSearchTool
        classes = [LoadDatasetTool, LoadGroundTruthTool, TrainRULModelTool, PredictRULTool,
                   TrainFaultClassifierTool, ClassifyFaultsTool, CalculateMAETool, CalculateRMSETool,
                   VerifyGroundTruthTool, CalculateAccuracyTool, VerifyClassificationTool,
                   AnalyzeEngineSignalsTool, AssessComponentHealthTool, DiagnoseTimingIssuesTool,
                   DetectDegradationTrendTool, CalculateMaintenanceCostTool, CalculateFailureCostTool,
                   OptimizeMaintenanceScheduleTool, AssessSafetyRiskTool, CheckComplianceTool,
                   GenerateSafetyRecommendationsTool, WebSearchTool]
        self.tools = {t.name: t for cls in classes for t in [cls()]}

    def catalog(self):
        return {name: dict(description=t.description,
                           inputSchema=t.args_schema.model_json_schema())
                for name, t in self.tools.items()}

    def call(self, name, arguments):
        return self.tools[name].invoke(arguments)


class AssetTools:
    def __init__(self):
        if os.environ.get("PHM_ALLOW_UNVERIFIED_ORIGINAL_TOOLS") != "1":
            raise RuntimeError(
                "Use the qualified AssetOps profile via verified_benchmark_worker.py. "
                "Original-source replay requires PHM_ALLOW_UNVERIFIED_ORIGINAL_TOOLS=1.")
        self.loop = asyncio.new_event_loop()
        self.servers = {}
        self.schemas = {}
        for name in ['wo', 'iot', 'tsfm', 'utilities', 'fmsr', 'vibration']:
            module = importlib.import_module('servers.' + name + '.main')
            self.servers[name] = module.mcp
            for tool in self.loop.run_until_complete(module.mcp.list_tools()):
                self.schemas[name + '.' + tool.name] = tool.model_dump(mode='json')

    def catalog(self):
        return self.schemas

    def call(self, name, arguments):
        server, tool = name.split('.', 1)
        return json_value(self.loop.run_until_complete(self.servers[server].call_tool(tool, arguments)))


def worker(family):
    with contextlib.redirect_stdout(sys.stderr):
        tools = ForgeTools() if family == 'forge' else AssetTools()
    for line in sys.stdin:
        started = time.monotonic()
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request['tool'] == 'list_tools':
                    catalog = tools.catalog()
                    prefix = request.get('args', {}).get('prefix', '')
                    result = {k: dict(description=v.get('description', '')[:500])
                              for k, v in catalog.items() if k.startswith(prefix)}
                elif request['tool'] == 'tool_schema':
                    result = tools.catalog()[request['args']['name']]
                elif request['tool'] == 'full_catalog':
                    result = tools.catalog()
                else:
                    result = tools.call(request['tool'], request.get('args', {}))
            out = dict(transport_ok=True, result=json_value(result),
                       elapsed_seconds=time.monotonic() - started)
        except Exception as exc:
            out = dict(transport_ok=False, error_type=type(exc).__name__,
                       error=str(exc)[:2000], elapsed_seconds=time.monotonic() - started)
        print(json.dumps(out, default=str), flush=True)


class ToolProcess:
    def __init__(self, route, root):
        self.route, self.root = route, root
        self.lock = threading.Lock()
        self.process = None
        self.stderr = None

    def request(self, request):
        if self.process is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.stderr = (self.root / 'worker.stderr').open('ab')
            self.process = subprocess.Popen(
                self.route['command'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, text=True, bufsize=1, start_new_session=True,
                cwd=self.route['cwd'], env=clean_environment(self.route.get('environment')))
            (self.root / 'worker_status.json').write_text(json.dumps(dict(
                pid=self.process.pid, started_utc=datetime.now(timezone.utc).isoformat(),
                command=self.route['command'], family=self.route['family']), indent=2) + '\n')
        if self.process.poll() is not None:
            raise RuntimeError('Original tool worker exited with code ' + str(self.process.returncode))
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], 180)
        if not ready:
            self.close()
            raise TimeoutError('Original tool call exceeded180 seconds; worker closed')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Original tool worker produced no JSON result')
        return json.loads(line)

    def close(self):
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            # A caller can abandon a request after the worker has already exited.
            # Reaping it must still finish and leave a closure receipt.
            pass
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.stderr:
            self.stderr.close()
        self.process.stdout.close()
        (self.root / 'worker_closed.json').write_text(json.dumps(dict(
            pid=self.process.pid, exit_code=self.process.returncode,
            closed_utc=datetime.now(timezone.utc).isoformat()), indent=2) + '\n')
        self.process = None


def broker(config_path):
    config = json.loads(config_path.read_text())
    socket_path = Path(config['socket'])
    if config.get('transport') not in ('fifo', 'fifo-files') and socket_path.exists():
        raise FileExistsError('Refusing to replace existing broker socket')
    processes = {cap: ToolProcess(route, Path(config['traces']) / route['job'])
                 for cap, route in config['routes'].items()}

    def invoke(cap, request):
        process = processes[cap]
        with process.lock:
            if request['tool'] == 'close_worker':
                process.close()
                out = dict(transport_ok=True, result='closed')
            else:
                out = process.request(request)
            process.root.mkdir(parents=True, exist_ok=True)
            with (process.root / 'calls.jsonl').open('a') as log:
                log.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                           tool=request['tool'], arguments=request.get('args', {}),
                           request_id=request.get('request_id'),
                           response=out), default=str) + '\n')
        return out

    if config.get('transport') in ('fifo', 'fifo-files'):
        def serve_pipe(cap, directory):
            # Hold the read side open across clients. Each client serializes its
            # own requests with a workspace-local lock, including large arrays.
            fd = os.open(directory / 'request.fifo', os.O_RDWR)
            with os.fdopen(fd, 'r') as reader:
                for line in reader:
                    request_id = None
                    try:
                        request = json.loads(line)
                        if config['transport'] == 'fifo-files':
                            request_id = request.get('request_id')
                            if not isinstance(request_id, str) or not re.fullmatch('[0-9a-f]{32}', request_id):
                                raise ValueError('Invalid request identifier')
                            if (directory / ('response_' + request_id + '.json')).exists():
                                request_id = None
                                raise ValueError('Repeated request identifier')
                            body = Path(config['routes'][cap]['request_root']) / ('request_' + request_id + '.json')
                            original = json.loads(body.read_text())
                            if original.get('request_id') != request_id or original.get('route') != cap:
                                raise ValueError('Tool request body identifier mismatch')
                            request = original
                        if request.pop('route') != cap:
                            raise ValueError('FIFO capability mismatch')
                        out = invoke(cap, request)
                    except Exception as exc:
                        out = dict(transport_ok=False, error_type=type(exc).__name__, error=str(exc)[:1000])
                    if config['transport'] == 'fifo-files':
                        if request_id is None or not re.fullmatch('[0-9a-f]{32}', request_id):
                            continue
                        response = directory / ('response_' + request_id + '.json')
                        # A closed client cannot break this route or consume the
                        # next client's result. Keep late responses as evidence.
                        temporary = response.with_suffix('.new')
                        temporary.write_text(json.dumps(dict(request_id=request_id, response=out), default=str) + '\n')
                        os.replace(temporary, response)
                    else:
                        try:
                            with (directory / 'response.fifo').open('w') as writer:
                                writer.write(json.dumps(out, default=str) + '\n')
                        except BrokenPipeError:
                            continue
        for cap, route in config['routes'].items():
            directory = Path(config['fifo_root']) / route['job']
            directory.mkdir(parents=True, exist_ok=False)
            names = ['request.fifo'] if config['transport'] == 'fifo-files' else ['request.fifo', 'response.fifo']
            for name in names:
                os.mkfifo(directory / name, 0o666)
                (directory / name).chmod(0o666)
            threading.Thread(target=serve_pipe, args=(cap, directory), daemon=True).start()
        try:
            threading.Event().wait()
        finally:
            for process in processes.values():
                process.close()
        return

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            cap = None
            try:
                request = json.loads(self.rfile.readline(4 * 1024 * 1024))
                cap = request.pop('route')
                out = invoke(cap, request)
            except Exception as exc:
                out = dict(transport_ok=False, error_type=type(exc).__name__, error=str(exc)[:1000])
            self.wfile.write(json.dumps(out, default=str).encode() + b'\n')

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

    try:
        with Server(str(socket_path), Handler) as server:
            socket_path.chmod(0o666)
            server.serve_forever(poll_interval=0.2)
    finally:
        for process in processes.values():
            process.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['worker', 'broker'])
    ap.add_argument('--family', choices=['forge', 'asset'])
    ap.add_argument('--config', type=Path)
    args = ap.parse_args()
    if args.mode == 'worker':
        worker(args.family)
    else:
        broker(args.config)


if __name__ == '__main__':
    main()
