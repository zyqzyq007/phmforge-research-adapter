"""Reused immutable-program registration, with a battery regression runtime."""
import argparse
import asyncio
import json
from pathlib import Path
from controlled_feedback import save, serve, remaining_budget
from program_revision_tools import Broker as BaseBroker, client as program_client, public
from hust_program_runtime import Runtime


def client(request):
    return program_client(request, timeout_seconds=930)


class Broker(BaseBroker):
    def __init__(self, run):
        self.root = Path(run)
        self.config = json.loads((self.root / 'BROKER_CONFIG.json').read_text())
        self.engine = Runtime(self.root)
        self.workspace = self.root / 'app'
        self.path = self.root / 'broker/STATE.json'
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            programs = {}
            for name, path in self.config['initial_programs'].items():
                programs[name] = dict(path=path, report=json.loads((Path(path) / 'REPORT.json').read_text()))
            self.state = dict(programs=programs, attempts=0, probe=None, submission=None)
            save(self.path, self.state)

    def dispatch(self, request):
        if request['action'] == 'info':
            return dict(contract=json.loads(Path(self.config['contract']).read_text()),
                training_versions_remaining=self.config['max_programs']-self.state['attempts'],
                programs={i: public(v['report']) for i, v in self.state['programs'].items()},
                submission=self.state['submission'], episode_budget=remaining_budget(self.root))
        assert request['action'] in ['register_program', 'submit'], 'Allowed:info/register_program/submit'
        return super().dispatch(request)


async def mcp():
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    server = Server('measured-battery-code')
    fields = dict(info={}, register_program=dict(files=dict(type='array', items=dict(type='string')),
        claim=dict(type='string'), rival=dict(type='string')), submit=dict(program_id=dict(type='string'), reason=dict(type='string')))
    descriptions = dict(info='Read all seven actually trained alternatives, whole-cell CV, legal development evidence and budgets.',
        register_program='Freeze source, genuinely fit three whole-cell CV models and a full training model; cold-test each single query and ID invariance. Changed source needs a new registration.',
        submit='Seal any complete registered program, including an existing reference. No final test feedback.')
    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=n, description=descriptions[n], inputSchema=dict(type='object', properties=f,
            required=list(f), additionalProperties=False)) for n, f in fields.items()]
    @server.call_tool()
    async def call_tool(name, arguments):
        assert name in fields
        response = await asyncio.to_thread(client, dict(action=name, **arguments))
        return [types.TextContent(type='text', text=json.dumps(response, ensure_ascii=False))]
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['serve', 'mcp', 'client'])
    p.add_argument('--run', type=Path)
    p.add_argument('--request')
    a = p.parse_args()
    if a.action == 'mcp':
        asyncio.run(mcp())
    elif a.action == 'serve':
        serve(a.run, broker_class=Broker)
    else:
        print(json.dumps(client(json.loads(a.request))))
