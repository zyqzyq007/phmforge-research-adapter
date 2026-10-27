"""Narrow I/O adapters for the pinned official MLE-STAR ADK implementation.

The author workflow and prompts remain unchanged. A budgeted Responses socket
replaces Gemini; real web-search requests use a file broker; generated Python
runs in the project's existing network-isolated numerical environment.
"""
import asyncio
import ast
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid


PYTHON = Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
NUMERICAL = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.new')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def schema(value):
    if isinstance(value, list):
        return [schema(x) for x in value]
    if not isinstance(value, dict):
        return value
    out = {k: schema(v) for k, v in value.items() if k != 'property_ordering'}
    if isinstance(out.get('type'), str):
        out['type'] = out['type'].lower()
    return out


def responses_payload(request):
    """Convert ADK text/function history without flattening tool results."""
    config = request.config
    instruction = config.system_instruction
    if instruction is not None and not isinstance(instruction, str):
        instruction = '\n'.join(p.text for p in instruction.parts if p.text)
    inputs, pending = [], {}
    for content in request.contents:
        role = 'assistant' if content.role == 'model' else 'user'
        for part in content.parts or []:
            if part.text:
                inputs.append({'role': role, 'content': part.text})
            if part.function_call:
                fc = part.function_call
                call_id = fc.id or f'call_{len(inputs)}_{fc.name}'
                pending.setdefault(fc.name, []).append(call_id)
                inputs.append({'type': 'function_call', 'call_id': call_id,
                               'name': fc.name, 'arguments': json.dumps(fc.args or {})})
            if part.function_response:
                fr = part.function_response
                queue = pending.get(fr.name, [])
                call_id = fr.id or (queue[0] if queue else None)
                if call_id is None:
                    raise ValueError('Function response has no preceding call identity')
                if call_id in queue:
                    queue.remove(call_id)
                inputs.append({'type': 'function_call_output', 'call_id': call_id,
                               'output': json.dumps(fr.response, ensure_ascii=False)})
            if part.inline_data or part.file_data:
                raise ValueError('This PHM adapter admits text and function calls only')
    tools = []
    for tool in config.tools or []:
        declarations = tool.function_declarations or []
        if not declarations:
            raise ValueError('Replace provider-specific tools before model invocation')
        for declaration in declarations:
            params = declaration.parameters_json_schema
            if params is None and declaration.parameters is not None:
                params = declaration.parameters.model_dump(mode='json', exclude_none=True)
            tools.append({'type': 'function', 'name': declaration.name,
                          'description': declaration.description or '',
                          'parameters': schema(params or {'type': 'object', 'properties': {}})})
    payload = {'model': 'deepseek-flash', 'input': inputs, 'tools': tools,
               'max_output_tokens': config.max_output_tokens or 8192, 'stream': False}
    if instruction:
        payload['instructions'] = instruction
    if config.temperature is not None:
        payload['temperature'] = config.temperature
    return payload


def post_socket(socket_path, payload):
    class UnixConnection(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(str(socket_path))
    connection = UnixConnection('localhost', timeout=610)
    try:
        connection.request('POST', '/responses', json.dumps(payload).encode(),
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f'Budgeted model relay HTTP {response.status}: {raw[:400]!r}')
        value = json.loads(raw)
        if value.get('status') != 'completed':
            raise RuntimeError(f'Model response not complete: {value.get("status")}')
        return value
    finally:
        connection.close()


def make_model(socket_path):
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class ResponsesLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            result = await asyncio.to_thread(post_socket, socket_path,
                                            responses_payload(llm_request))
            parts = []
            for item in result.get('output', []):
                if item.get('type') == 'message':
                    for block in item.get('content', []):
                        if block.get('type') == 'output_text':
                            parts.append(types.Part(text=block['text']))
                elif item.get('type') == 'function_call':
                    parts.append(types.Part(function_call=types.FunctionCall(
                        id=item['call_id'], name=item['name'], args=json.loads(item['arguments']))))
            if not parts:
                raise RuntimeError('Completed model response contains no text or function call')
            usage = result.get('usage', {})
            yield LlmResponse(content=types.Content(role='model', parts=parts),
                              turn_complete=True,
                              usage_metadata=types.GenerateContentResponseUsageMetadata(
                                  prompt_token_count=usage.get('input_tokens'),
                                  candidates_token_count=usage.get('output_tokens'),
                                  total_token_count=usage.get('total_tokens')))

    return ResponsesLlm(model='deepseek-flash')


def normalize_retrieval_response(response):
    """Strip prose only around one well-formed, schema-valid fenced model list.

    Official extraction takes the first '[' through the last ']'; citation
    links before the list can therefore prevent parsing. No candidate content
    is changed. Ambiguous or malformed responses retain the original behavior.
    """
    if response.content is None:
        return response
    text = '\n'.join(p.text for p in response.content.parts or [] if p.text)
    blocks = []
    for block in re.findall(r'```(?:python|json)?\s*\n(.*?)```', text, flags=re.S):
        try:
            value = ast.literal_eval(block.strip())
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, list) and value and all(
            isinstance(row, dict) and all(isinstance(row.get(k), str) for k in
                                         ['model_name', 'example_code']) for row in value):
            blocks.append(block.strip())
    if len(blocks) != 1:
        return response
    from google.genai import types
    result = response.model_copy(deep=True)
    result.content.parts = [types.Part(text=blocks[0])]
    return result


def make_search(broker_dir, limit=12, wait_seconds=180):
    """The controller must answer with actual web-tool results, never a fixture."""
    broker_dir = Path(broker_dir)
    broker_dir.mkdir(parents=True, exist_ok=True)

    async def search_web(query: str) -> dict:
        """Search public technical documentation and papers for the supplied query."""
        if not query.strip() or len(query) > 1500:
            return {'error': 'Query must contain 1–1500 characters'}
        if len(list(broker_dir.glob('*.request.json'))) >= limit:
            return {'error': 'Frozen public-search budget exhausted'}
        identity = uuid.uuid4().hex
        save(broker_dir / (identity + '.request.json'),
             {'query': query, 'created_unix': time.time(), 'id': identity})
        response = broker_dir / (identity + '.response.json')
        started = time.monotonic()
        while time.monotonic() - started < wait_seconds:
            if response.exists():
                result = json.loads(response.read_text())
                if result.get('provider') != 'web.run' or result.get('query') != query:
                    raise ValueError('Unmatched public-search receipt')
                return result
            await asyncio.sleep(.5)
        return {'error': 'Public-search broker timeout; no search result available'}
    return search_web


def make_executor(workspace_root, evidence_root, training_entry, execution_mode='fit'):
    if execution_mode not in ('fit', 'predict'):
        raise ValueError('Invalid numerical execution mode')
    workspace_root, evidence_root = Path(workspace_root).resolve(), Path(evidence_root)
    training_entry = Path(training_entry).resolve()

    def execute(code_text, run_cwd, py_filepath, exec_timeout):
        cwd = Path(run_cwd).resolve()
        if not cwd.is_relative_to(workspace_root) or not cwd.is_dir():
            raise ValueError('Code workspace is outside the frozen public workspace')
        filename = Path(py_filepath)
        if filename.is_absolute() or filename.name != py_filepath:
            raise ValueError('Expected a local Python filename')
        folder = evidence_root / uuid.uuid4().hex
        folder.mkdir(parents=True)
        code_path = cwd / filename
        if code_path.is_symlink():
            raise ValueError('Generated code path must not be a symlink')
        code_path.write_text(code_text)
        (folder / 'program.py').write_text(code_text)
        (cwd / '.runtime/tmp/mpl').mkdir(parents=True, exist_ok=True)
        command = ['bwrap', '--unshare-user', '--unshare-pid', '--unshare-ipc',
                   '--unshare-uts', '--unshare-net', '--die-with-parent', '--new-session']
        for path in ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc']:
            if Path(path).exists():
                command += ['--ro-bind', path, path]
        command += ['--ro-bind', str(PYTHON), '/opt/python312',
                    '--ro-bind', str(NUMERICAL), '/opt/deps',
                    '--ro-bind', str(training_entry), '/training_entry.py',
                    '--bind', str(cwd), '/work', '--bind', str(folder), '/audit',
                    '--proc', '/proc', '--dev', '/dev', '--chdir', '/work', '--remount-ro', '/',
                    '/opt/python312/bin/python3.12', '-B', '/training_entry.py',
                    '--audit', '/audit/training_audit.json', '--mode', execution_mode,
                    '/work/' + filename.name]
        env = dict(PATH='/opt/python312/bin:/usr/bin:/bin', PYTHONPATH='/opt/deps',
                   PYTHONDONTWRITEBYTECODE='1', TMPDIR='/work/.runtime/tmp',
                   TMP='/work/.runtime/tmp', TEMP='/work/.runtime/tmp',
                   MPLCONFIGDIR='/work/.runtime/tmp/mpl', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', HOME='/work/.runtime')
        started = time.monotonic()
        try:
            process = subprocess.run(command, env=env, capture_output=True, text=True,
                                     timeout=exec_timeout)
            result = dict(returncode=process.returncode, stdout=process.stdout,
                          stderr=process.stderr, execution_time=time.monotonic() - started)
        except subprocess.TimeoutExpired as error:
            result = dict(returncode=1, stdout=(error.stdout or b'').decode(errors='replace'),
                          stderr='Isolated Python execution timeout',
                          execution_time=time.monotonic() - started)
        save(folder / 'PROCESS.json', dict(result, command=command,
                                         code_sha256=hashlib.sha256(code_text.encode()).hexdigest()))
        return result
    return execute


def load_official(package_dir, config_overrides, model, search, executor):
    """Import all original modules after explicitly suppressing cloud bootstrap."""
    import importlib
    import sys
    import types as module_types
    package_dir = Path(package_dir).resolve()
    if 'machine_learning_engineering' in sys.modules:
        raise RuntimeError('Official package already imported; run in a fresh process')
    package = module_types.ModuleType('machine_learning_engineering')
    package.__path__ = [str(package_dir)]
    package.__file__ = str(package_dir / '__init__.py')
    sys.modules[package.__name__] = package
    config = importlib.import_module('machine_learning_engineering.shared_libraries.config')
    for key, value in config_overrides.items():
        if not hasattr(config.CONFIG, key):
            raise ValueError(f'Unknown official configuration key: {key}')
        setattr(config.CONFIG, key, value)
    code_util = importlib.import_module('machine_learning_engineering.shared_libraries.code_util')
    code_util.run_python_code = executor
    module = importlib.import_module('machine_learning_engineering.agent')
    from google.adk.agents import LlmAgent
    inventory = []

    def visit(agent):
        row = {'name': agent.name, 'class': type(agent).__name__}
        if isinstance(agent, LlmAgent):
            agent.model = model
            tools = []
            for tool in agent.tools:
                if getattr(tool, 'name', None) == 'google_search':
                    tools.append(search)
                else:
                    tools.append(tool)
            agent.tools = tools
            if agent.name.startswith('model_retriever_agent_'):
                original_callback = agent.after_model_callback
                def normalize_then_parse(callback_context, llm_response,
                                         _callback=original_callback):
                    return _callback(callback_context=callback_context,
                                     llm_response=normalize_retrieval_response(llm_response))
                agent.after_model_callback = normalize_then_parse
            row.update(model=agent.model.model,
                       tools=[getattr(t, 'name', getattr(t, '__name__', type(t).__name__)) for t in tools])
        inventory.append(row)
        for child in agent.sub_agents:
            visit(child)
    visit(module.root_agent)
    return module, inventory
