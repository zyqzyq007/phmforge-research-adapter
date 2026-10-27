"""Cold MCP checks for a prospective original-task coverage batch."""
import argparse
import asyncio
import json
from pathlib import Path
import sys


async def execute(args):
    sys.path.insert(0, str(args.source))
    from scripts.aat_tools_mcp import build_mcp_servers
    servers = await build_mcp_servers(args.source)
    result = {'catalogs': {}, 'calls': [], 'assertions': [], 'new_LLM_calls': 0}
    by_name = {s.name: s for s in servers}

    async def call(domain, name, arguments):
        raw = await by_name[domain].call_tool(name, arguments)
        encoded = raw.model_dump(mode='json')
        result['calls'].append({'domain': domain, 'name': name,
                                'arguments': arguments, 'result': encoded})
        if raw.isError:
            raise RuntimeError('MCP execution failed: ' + domain + '.' + name)
        if raw.structuredContent:
            data = raw.structuredContent
            return data.get('result', data)
        return json.loads(''.join(c.text for c in raw.content if getattr(c, 'type', None) == 'text'))

    def check(name, actual, expected):
        ok = actual == expected
        result['assertions'].append({'name': name, 'actual': actual,
                                      'expected': expected, 'passed': ok})
        if not ok:
            raise AssertionError(name)

    try:
        for server in servers:
            result['catalogs'][server.name] = [t.model_dump(mode='json') for t in await server.list_tools()]
        for asset in ('T-002', 'T-003', 'T-005', 'T-006', 'T-008', 'T-011', 'T-014', 'T-015', 'T-016'):
            await call('iot', 'get_asset_metadata', {'transformer_id': asset})
            await call('iot', 'list_sensors', {'transformer_id': asset})
        for asset, sensor in (('T-005', 'winding_temp_top_c'), ('T-008', 'voltage_hv_kv'),
                              ('T-008', 'voltage_lv_kv'), ('T-014', 'voltage_hv_kv'),
                              ('T-014', 'voltage_lv_kv'), ('T-015', 'load_current_a'),
                              ('T-002', 'voltage_hv_kv')):
            await call('iot', 'get_sensor_readings', {'transformer_id': asset, 'sensor_id': sensor, 'limit': 1000})
        for asset, horizon in (('T-016', 180), ('T-011', 180), ('T-015', 30)):
            now = await call('tsfm', 'get_rul', {'transformer_id': asset})
            future = await call('tsfm', 'forecast_rul', {'transformer_id': asset, 'horizon_days': horizon})
            check(asset + '_rul_linear_reference', future['projected_rul_days'], max(0, now['rul_days'] - horizon))
        await call('tsfm', 'trend_analysis', {'transformer_id': 'T-002', 'sensor_id': 'winding_temp_top_c'})
        for asset in ('T-015', 'T-016'):
            gas = await call('fmsr', 'get_dga_record', {'transformer_id': asset})
            await call('fmsr', 'analyze_dga', {'transformer_id': asset,
                       **{k: gas['dissolved_' + k + '_ppm'] for k in ('h2', 'ch4', 'c2h2', 'c2h4', 'c2h6')}})
        await call('fmsr', 'search_failure_modes', {'query': 'voltage'})
        for asset in ('T-006', 'T-003', 'T-016'):
            history = await call('wo', 'list_fault_records', {'transformer_id': asset})
            check(asset + '_history_nonempty', bool(history), True)
            await call('wo', 'get_fault_record', {'fault_id': history[0]['fault_id']})
            downtime = await call('wo', 'estimate_downtime', {'transformer_id': asset,
                                  'severity': 'high', 'fault_type': history[0]['fault_type']})
            check(asset + '_high_downtime_hours', downtime['estimated_typical_hours'], 24)
            wo = await call('wo', 'create_work_order', {'transformer_id': asset,
                            'issue_description': 'Cold qualification only; isolated local process.',
                            'priority': 'high', 'fault_type': history[0]['fault_type'],
                            'estimated_downtime_hours': downtime['estimated_typical_hours']})
            check(asset + '_created_status', wo['status'], 'open')
            updated = await call('wo', 'update_work_order', {'work_order_id': wo['work_order_id'],
                                 'status': 'in_progress', 'note': '24-hour estimate recorded.'})
            check(asset + '_updated_status', updated['status'], 'in_progress')
            listed = await call('wo', 'list_work_orders', {'transformer_id': asset})
            check(asset + '_state_persists_within_MCP_session',
                  any(x['work_order_id'] == wo['work_order_id'] and x['status'] == 'in_progress' for x in listed), True)
    finally:
        for server in reversed(servers):
            await server.cleanup()
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'tools': sum(len(x) for x in result['catalogs'].values()),
                      'calls': len(result['calls']), 'checks': len(result['assertions']),
                      'passed': sum(x['passed'] for x in result['assertions']), 'new_LLM_calls': 0}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    asyncio.run(execute(p.parse_args()))


if __name__ == '__main__':
    main()
