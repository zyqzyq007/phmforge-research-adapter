"""Check raw-call evaluation views against unchanged source tools and WO state."""
import argparse
import datetime
import json
from pathlib import Path
import sys


def canonical(value):
    def encode(item):
        if isinstance(item, (datetime.datetime, datetime.date)):
            return item.isoformat()
        return str(item)
    return json.loads(json.dumps(value, default=encode))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--normalized', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not hasattr(datetime, 'UTC'):
        datetime.UTC = datetime.timezone.utc
    sys.path.insert(0, str(args.source))
    from mcp_servers.direct_adapter import get_tools
    registry = {x.name.rsplit('.', 1)[-1]: x.fn for x in get_tools()}
    mutable = {'create_work_order', 'update_work_order', 'list_work_orders'}
    jobs = []
    for path in sorted(args.normalized.glob('*.json')):
        if path.name == 'index.json':
            continue
        data = json.loads(path.read_text())
        checks, states, actions = [], {}, []
        for call in data['calls']:
            name = call.get('name', '').rsplit('__', 1)[-1]
            arguments = call.get('arguments', {})
            observed = call.get('decoded_output')
            if name not in registry or not isinstance(arguments, dict):
                checks.append({'call_id': call['call_id'], 'name': name, 'status': 'unqualified_name_or_arguments'})
                continue
            if name not in mutable:
                try:
                    expected = canonical(registry[name](**arguments))
                    # MCP text-content encoding may put a one-row list in one
                    # object block. Restore only the declared list result shape.
                    if isinstance(expected, list) and isinstance(observed, dict) and 'error' not in observed:
                        observed = [observed]
                    checks.append({'call_id': call['call_id'], 'name': name, 'status': 'matched' if observed == expected else 'mismatch',
                                   **({'observed': observed, 'expected': expected} if observed != expected else {})})
                except Exception as error:
                    checks.append({'call_id': call['call_id'], 'name': name,
                                   'status': 'original_function_exception', 'exception_type': type(error).__name__,
                                   'error': str(error), 'observed': observed})
                continue
            if name == 'create_work_order' and isinstance(observed, dict) and observed.get('work_order_id'):
                key = observed['work_order_id']
                ok = (observed['transformer_id'] == arguments['transformer_id'] and
                      observed['status'] == 'open' and
                      observed['priority'] == arguments.get('priority', 'medium').strip().lower())
                if arguments.get('estimated_downtime_hours') is not None:
                    ok = ok and observed['estimated_downtime_hours'] == arguments['estimated_downtime_hours']
                states[key] = observed
                actions.append({'call_id': call['call_id'], 'name': name, 'work_order_id': key, 'passed': ok,
                                'record': observed})
            elif name == 'update_work_order' and isinstance(observed, dict) and observed.get('work_order_id'):
                key = arguments['work_order_id']
                ok = key in states and key == observed['work_order_id']
                for field in ('status', 'priority'):
                    if arguments.get(field) is not None:
                        ok = ok and observed[field] == arguments[field].strip().lower()
                if arguments.get('note'):
                    ok = ok and any(n.get('text') == arguments['note'] for n in observed['notes'])
                states[key] = observed
                actions.append({'call_id': call['call_id'], 'name': name, 'work_order_id': key, 'passed': ok,
                                'record': observed})
            elif name == 'list_work_orders' and isinstance(observed, list):
                ok = all(x['work_order_id'] in states for x in observed)
                actions.append({'call_id': call['call_id'], 'name': name, 'passed': ok, 'record': observed})
            else:
                actions.append({'call_id': call['call_id'], 'name': name, 'passed': None,
                                'scope': 'error or unsupported state record', 'record': observed})
        jobs.append({'job_id': data['job_id'], 'scenario_id': data['scenario_id'],
                     'condition': data['condition'], 'readonly_checks': checks,
                     'action_state_checks': actions, 'last_observed_work_orders': list(states.values()),
                     'command_count': len(data['commands']),
                     'command_errors': sum(x.get('exit_code') not in (None, 0) for x in data['commands']),
                     'no_op_commands': sum(x.get('command', '').endswith(' true') for x in data['commands']),
                     'original_saved_trajectory_mismatches': len(data['upstream_serialization_mismatches'])})
    summary = {'jobs': len(jobs),
               'readonly_calls': sum(len(x['readonly_checks']) for x in jobs),
               'readonly_matched': sum(c['status'] == 'matched' for x in jobs for c in x['readonly_checks']),
               'readonly_mismatches': sum(c['status'] == 'mismatch' for x in jobs for c in x['readonly_checks']),
               'state_checks': sum(len(x['action_state_checks']) for x in jobs),
               'state_checks_passed': sum(c['passed'] is True for x in jobs for c in x['action_state_checks']),
               'state_checks_failed': sum(c['passed'] is False for x in jobs for c in x['action_state_checks']),
               'no_new_LLM_calls': True,
               'scope': 'MCP readonly return fidelity and same-session WO state only; not complete task/physical validity. Source direct function commands require separate executable audit.'}
    args.output.write_text(json.dumps({'summary': summary, 'jobs': jobs}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
