"""Read-only, real validation evidence and sealed PHM program choices."""
import argparse
import asyncio
import json
import math
from pathlib import Path

from controlled_feedback import client, save, serve, utc, remaining_budget


def validate_selected(case, selected):
    """Allow continuous mixtures only in an explicitly declared update task."""
    if selected in case['candidate_ids']:
        return
    assert case.get('continuous_update') is True and isinstance(selected, str)
    assert selected.startswith('mix:'), 'Use an existing program or mix:ALPHA'
    weight = float(selected[4:])
    assert math.isfinite(weight) and 0 <= weight <= 1, 'ALPHA must be finite and in [0,1]'


class Broker:
    def __init__(self, run):
        self.run = Path(run)
        config = json.loads((self.run/'BROKER_CONFIG.json').read_text())
        self.config = config
        self.delegate = None
        if config.get('query_private'):
            from program_condition_queries import QueryBroker
            self.delegate = QueryBroker(run, config)
            return
        self.public = Path(config['public'])
        self.cases = json.loads((self.public/'cases.json').read_text())
        self.checked = []
        self.choice = None

    def dispatch(self, request):
        if self.delegate is not None:
            return self.delegate.dispatch(request)
        action = request['action']
        if action == 'cases':
            return {'cases': self.cases, 'final_choice': self.choice, 'episode_budget': remaining_budget(self.run)}
        if action == 'evidence':
            case = next(c for c in self.cases if c['id'] == request['case_id'])
            self.checked.append(case['id'])
            return dict(case, detailed_rows='/data/rows/'+case['id']+'.json',
                        evidence_source='Hash-bound genuine fitted-program legal validation predictions. No final-test information.',
                        new_fitting=False, episode_budget=remaining_budget(self.run))
        if action == 'select':
            assert self.choice is None, 'Choice is already sealed'
            rows = request['choices']
            expected = {(c['id'], old) for c in self.cases for old in c.get('incumbent_ids', c['candidate_ids'])}
            found = {(r['case'], r['incumbent']) for r in rows}
            assert len(rows) == len(found) and found == expected, 'Supply every case/incumbent pair exactly once'
            assert set(self.checked) == {c['id'] for c in self.cases}, 'Read evidence for all cases before selection'
            for r in rows:
                case = next(c for c in self.cases if c['id'] == r['case'])
                validate_selected(case, r['selected'])
                assert isinstance(r.get('reason'), str) and r['reason'].strip()
            self.choice = dict(utc=utc(), choices=rows, checked_cases=sorted(set(self.checked)))
            save(self.run/'broker/FINAL_CHOICE.json', self.choice)
            return self.choice
        raise ValueError('Unknown action')


async def mcp_server():
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    server = Server('phm-program-update')
    fields = {
        'cases': {},
        'evidence': {'case_id': {'type': 'string'}},
        'select': {'choices': {'type': 'array', 'items': {'type': 'object',
            'properties': {k: {'type': 'string'} for k in ['case', 'incumbent', 'selected', 'reason']},
            'required': ['case', 'incumbent', 'selected', 'reason'], 'additionalProperties': False}}},
    }
    query_mode = Path('/data/contract.json').exists() and json.loads(Path('/data/contract.json').read_text()).get('allow_validation_queries', False)
    if query_mode:
        fields['query'] = {'slot': {'type': 'string'}, 'expectation': {'type': 'string'}, 'rival': {'type': 'string'}}
    descriptions = {
        'cases': 'List measured PHM candidate pools and all legal grouped validation losses.',
        'evidence': 'Read the hash-bound whole-program CV/development evidence and row-level evidence path for a case.',
        'select': 'Seal every case/incumbent choice. No final data or metric feedback is supplied.',
        'query': 'Reveal one selected whole calibration condition real-label packet;unique-query budget applies. Final records cannot be queried.',
    }
    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=k, description=descriptions[k], inputSchema={
            'type': 'object', 'properties': v, 'required': ['slot'] if k == 'query' else list(v), 'additionalProperties': False}) for k, v in fields.items()]
    @server.call_tool()
    async def call_tool(name, arguments):
        assert name in fields
        result = await asyncio.to_thread(client, dict(action=name, **arguments))
        return [types.TextContent(type='text', text=json.dumps(result, ensure_ascii=False))]
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['mcp', 'serve', 'client'])
    p.add_argument('--run', type=Path); p.add_argument('--request'); a = p.parse_args()
    if a.action == 'mcp':
        asyncio.run(mcp_server())
    elif a.action == 'serve':
        serve(a.run, broker_class=Broker)
    else:
        print(json.dumps(client(json.loads(a.request)), ensure_ascii=False))
