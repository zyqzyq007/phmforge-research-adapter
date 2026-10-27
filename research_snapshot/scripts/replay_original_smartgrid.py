"""Run the frozen SmartGridBench AaT with a DeepSeek provider-only adapter.

Agent logic, system prompt, MCP servers and output serializer remain upstream.
Credentials are read at runtime; HTTP bodies retain complete provider turns.
"""
import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shlex
import sys
import types


def literal_credential(path):
    pattern = re.compile(r'^\s*(?:export\s+)?DEEPSEEK_API_KEY\s*=\s*(.*)$')
    values = [m.group(1) for line in path.read_text().splitlines()
              if (m := pattern.match(line))]
    if len(values) != 1:
        raise ValueError('Expected one literal DEEPSEEK_API_KEY assignment')
    parts = shlex.split(values[0], comments=True)
    if len(parts) != 1 or not parts[0] or any(c in parts[0] for c in ['$','`','\n']):
        raise ValueError('Credential must be one literal value')
    return parts[0]


async def execute(args):
    os.environ['OPENAI_AGENTS_DISABLE_TRACING'] = '1'
    os.environ['AAT_MCP_SERVER_LAUNCH_MODE'] = 'python'
    os.environ['AAT_MCP_SERVER_PYTHON'] = str(args.server_python)
    os.environ['AAT_MCP_CLIENT_TIMEOUT_SECONDS'] = '60'
    sys.path.insert(0, str(args.source))
    import httpx
    from openai import AsyncOpenAI
    from agents import OpenAIChatCompletionsModel, set_tracing_disabled
    set_tracing_disabled(True)
    key = literal_credential(args.credential_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = args.output.parent / 'provider_turns.jsonl'
    models = set()
    sequence = 0

    def save(value):
        serialized = json.dumps(value, ensure_ascii=False).replace(key, '[REDACTED_SECRET]')
        with trace.open('a') as out:
            out.write(serialized + '\n')

    async def request_hook(request):
        nonlocal sequence
        sequence += 1
        body = json.loads(request.content)
        if body.get('model') != 'deepseek-flash':
            raise ValueError('Only deepseek-flash is authorized')
        save({'direction':'request','sequence':sequence,'method':request.method,
              'url':str(request.url),'body':body})

    async def response_hook(response):
        body = await response.aread()
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = {'unparsed_response':body.decode(errors='replace')}
        if isinstance(parsed, dict) and parsed.get('model'):
            models.add(parsed['model'])
        save({'direction':'response','sequence':sequence,
              'status_code':response.status_code,'body':parsed})

    class ConfiguredProvider(OpenAIChatCompletionsModel):
        async def get_response(self, *positional, **keyword):
            if 'model_settings' in keyword:
                settings = keyword['model_settings']
                keyword['model_settings'] = replace(settings, reasoning=None,
                    extra_body={**(settings.extra_body or {}), 'reasoning_effort':'max'})
            else:
                positional = list(positional)
                settings = positional[2]
                positional[2] = replace(settings, reasoning=None,
                    extra_body={**(settings.extra_body or {}), 'reasoning_effort':'max'})
            return await super().get_response(*positional, **keyword)

    async with httpx.AsyncClient(timeout=300, trust_env=False,
            event_hooks={'request':[request_hook], 'response':[response_hook]}) as http:
        async with AsyncOpenAI(api_key=key, base_url='https://api.deepseek.com',
                               http_client=http, max_retries=2) as client:
            def provider_factory(*, model, base_url=None, api_key=None):
                if model != 'deepseek-flash':
                    raise ValueError('Only deepseek-flash is authorized')
                return ConfiguredProvider(model=model, openai_client=client)

            adapter = types.ModuleType('agents.extensions.models.litellm_model')
            adapter.LitellmModel = provider_factory
            sys.modules[adapter.__name__] = adapter
            from scripts import aat_runner
            namespace = argparse.Namespace(
                prompt=args.prompt.read_text(), output=str(args.output),
                model_id='deepseek-flash', mcp_mode='baseline', max_turns=args.max_turns,
                parallel_tool_calls=False, scenarios_glob=None, verbose=False)
            code = await aat_runner._main(namespace)
    (args.output.parent / 'provider_receipt.json').write_text(json.dumps({
        'requested_model':'deepseek-flash','returned_models':sorted(models),
        'base_url':'https://api.deepseek.com','wire_api':'chat_completions',
        'reasoning_effort':'max','max_retries':2,'requests_observed':sequence,
        'provider_adapter_only':True,'upstream_agent_and_tools_unchanged':True,
        'exit_code':code}, indent=2)+'\n')
    return code


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--server-python', type=Path, required=True)
    p.add_argument('--credential-file', type=Path, required=True)
    p.add_argument('--max-turns', type=int, default=100)
    args = p.parse_args()
    for name in ['TMPDIR','TMP','TEMP','MPLCONFIGDIR']:
        if not os.environ.get(name) or Path(os.environ[name]) == Path('/tmp'):
            p.error(f'{name} must point to this run directory')
    return asyncio.run(execute(args))


if __name__ == '__main__':
    sys.exit(main())
