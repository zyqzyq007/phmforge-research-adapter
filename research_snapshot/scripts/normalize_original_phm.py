"""Build evaluation views from complete raw provider/CLI traces by call ID."""
import argparse
import ast
import json
from pathlib import Path


def decode_output(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    if isinstance(value, dict) and value.get('type') == 'text':
        return [decode_output(value['text'])]
    if isinstance(value, dict) and 'content' in value:
        value = value['content']
    if isinstance(value, list) and value and all(isinstance(x, dict) and x.get('type') == 'text' for x in value):
        return [decode_output(x['text']) for x in value]
    return value


def unwrap(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    if isinstance(value, dict):
        for name in ('structuredContent', 'structured_content'):
            if value.get(name) is not None:
                return unwrap(value[name])
        if value.get('type') == 'text':
            return unwrap(value['text'])
        if 'content' in value:
            return unwrap(value['content'])
    if (isinstance(value, list) and value and
            all(isinstance(x, dict) and x.get('type') == 'text' for x in value)):
        blocks = [unwrap(x) for x in value]
        return blocks[0] if len(blocks) == 1 else blocks
    if isinstance(value, dict) and set(value) == {'result'}:
        return unwrap(value['result'])
    return value


def native(folder):
    calls, commands = {}, {}
    for line in (folder / 'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        item = event.get('item', {})
        if item.get('type') == 'mcp_tool_call' and item.get('status') in ('completed', 'failed'):
            calls[item['id']] = {'name': item.get('tool'), 'server': item.get('server'),
                                'arguments': item.get('arguments'), 'call_id': item['id'],
                                'output': item.get('result'), 'error': item.get('error'),
                                'status': item.get('status')}
        if item.get('type') == 'command_execution' and event.get('type') == 'item.completed':
            commands[item['id']] = item
    answer = folder / 'response.txt'
    return {'answer': answer.read_text() if answer.exists() else '',
            'calls': list(calls.values()), 'commands': list(commands.values()),
            'upstream_serialization_mismatches': []}


def sdk(folder):
    path = folder / 'trajectory.json'
    payload = json.loads(path.read_text()) if path.exists() else {}
    calls = {}
    trace = folder / 'provider_turns.jsonl'
    for line in trace.read_text().splitlines() if trace.exists() else []:
        event = json.loads(line)
        if event['direction'] == 'request':
            messages = event['body'].get('messages', [])
        else:
            messages = [x['message'] for x in event['body'].get('choices', []) if x.get('message')]
        for message in messages:
            for call in message.get('tool_calls', []) or []:
                fn = call['function']
                arguments = fn.get('arguments', '{}')
                try:
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except ValueError:
                    pass
                calls.setdefault(call['id'], {'name': fn['name'], 'arguments': arguments,
                                              'call_id': call['id']})
            if message.get('role') == 'tool':
                call_id = message['tool_call_id']
                calls.setdefault(call_id, {'call_id': call_id})['output'] = message.get('content')
    mismatches = []
    for turn in payload.get('history', []):
        for call in turn.get('tool_calls', []):
            original = calls.get(call.get('call_id'))
            if (original and original.get('output') is not None and
                    decode_output(call.get('output')) != decode_output(original['output'])):
                mismatches.append({'call_id': call.get('call_id'), 'tool': call.get('name'),
                                   'upstream_output': call.get('output'),
                                   'provider_observed_output': original['output']})
    return {'answer': payload.get('answer', ''), 'calls': list(calls.values()), 'commands': [],
            'upstream_serialization_mismatches': mismatches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    index = []
    for job in json.loads(args.plan.read_text())['jobs']:
        folder = args.runs / job['id']
        state = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {'status': 'not_run'}
        if state.get('status') == 'running':
            raise RuntimeError('Normalize only after the batch is terminal')
        if not folder.exists():
            data = {'answer': '', 'calls': [], 'commands': [], 'upstream_serialization_mismatches': []}
        else:
            data = sdk(folder) if job['condition'] == 'official_aat' else native(folder)
        data.update(job_id=job['id'], scenario_id=job['scenario_id'], condition=job['condition'],
                    repeat=job['repeat'], status=state,
                    question=(args.plan.parent / job['prompt_file']).read_text())
        for call in data['calls']:
            call['decoded_output'] = unwrap(call.get('output'))
        (args.output / (job['id'] + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        index.append({'job_id': job['id'], 'scenario_id': job['scenario_id'], 'condition': job['condition'],
                      'status': state.get('status'), 'answer_chars': len(data['answer']),
                      'tool_calls': len(data['calls']), 'commands': len(data['commands']),
                      'serialization_mismatches': len(data['upstream_serialization_mismatches'])})
    (args.output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'normalized_jobs': len(index),
                      'mismatch_calls': sum(x['serialization_mismatches'] for x in index)}))


if __name__ == '__main__':
    main()
