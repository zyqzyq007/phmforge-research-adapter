"""Identical real training, prefix checking and submission tools for both arms."""
import argparse
import asyncio
import json
from pathlib import Path
from controlled_feedback import save, serve
from program_revision_tools import Broker as BaseBroker, client, public
from cmapss_program_runtime import Runtime


class Broker(BaseBroker):
    def __init__(self, run):
        self.root = Path(run)
        self.config = json.loads((self.root/'BROKER_CONFIG.json').read_text())
        self.engine = Runtime(self.root)
        self.workspace = self.root/'app'
        self.path = self.root/'broker/STATE.json'
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = dict(programs={i:dict(path=p, report=json.loads((Path(p)/'REPORT.json').read_text()))
                                       for i,p in self.config['initial_programs'].items()},
                              attempts=0, submission=None, probe=None, checks={})
            self.persist()

    def dispatch(self, request):
        action = request['action']
        if action=='info':
            return dict(contract=json.loads(Path(self.config['contract']).read_text()),
                        programs={i:public(v['report']) for i,v in self.state['programs'].items()},
                        checks=self.state['checks'], submission=self.state['submission'],
                        training_versions_remaining=self.config['max_programs']-self.state['attempts'],
                        episode_budget=self.budget())
        if action=='check':
            assert self.state['submission'] is None, 'Submission already sealed'
            ident = request['program_id']
            assert ident in self.state['programs']
            if ident not in self.state['checks']:
                result = self.engine.check(Path(self.state['programs'][ident]['path']), self.root/'checks'/ident)
                self.state['checks'][ident] = result
                self.persist()
            return self.state['checks'][ident]
        assert action in ['register_program','submit']
        return super().dispatch(request)


async def mcp():
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    server = Server('prospective-cmapss-programs')
    fields = dict(info={}, check=dict(program_id=dict(type='string')),
                  register_program=dict(files=dict(type='array',items=dict(type='string')),
                                        claim=dict(type='string'),rival=dict(type='string')),
                  submit=dict(program_id=dict(type='string'),reason=dict(type='string')))
    descriptions = dict(info='Read available actual programs, full-trajectory development results and remaining budgets.',
        check='Cold-run the exact registered program on real development engine prefixes. Compare the same query with full-context prediction. No retraining or final information. Adaptive development results cached per immutable version.',
        register_program='Freeze arbitrary complete Python program, genuinely fit on training engines and evaluate on separate development engines. All supervised fitting through this tool; changed source needs a new registration.',
        submit='Seal any complete registered program, including an unchanged reference. No final feedback.')
    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=n,description=descriptions[n],inputSchema=dict(type='object',properties=f,
                          required=list(f),additionalProperties=False)) for n,f in fields.items()]
    @server.call_tool()
    async def call_tool(name, arguments):
        assert name in fields
        result = await asyncio.to_thread(client,dict(action=name,**arguments),690)
        return [types.TextContent(type='text',text=json.dumps(result,ensure_ascii=False))]
    async with stdio_server() as (r,w):
        await server.run(r,w,server.create_initialization_options())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['serve','mcp','client']);p.add_argument('--run',type=Path);p.add_argument('--request')
    a=p.parse_args()
    if a.action=='serve':serve(a.run,broker_class=Broker)
    elif a.action=='mcp':asyncio.run(mcp())
    else:print(json.dumps(client(json.loads(a.request))))
