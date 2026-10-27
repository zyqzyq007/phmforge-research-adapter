"""Recount sealed coding experiments without calling a model or predicting again."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    assert (root / 'TERMINAL392.json').exists()
    result = read(root / 'RESULT392.json')
    seal = read(root / 'ALL_SUBMISSIONS_SEALED392.json')
    assert sha(root / 'PROTOCOL392.json') == seal['protocol_sha256']
    assert sha(root / 'PREDICTIONS392.json') == read(root / 'PREDICTIONS_SEAL392.json')['sha256']
    inputs = {r['index']: r for r in read(root / 'EVALUATION_INPUTS392.json')}
    predictions = {(r['program'], r['case']): r for r in read(root / 'PREDICTIONS392.json')}
    errors = {}
    checks = 0
    for row in result['rows']:
        key = row['program'], row['case']
        pred = predictions[key]['prediction']
        if pred is None:
            assert row['error_mm'] is None
            continue
        assert row['prediction_m'] == pred['location_m']
        assert pred['query_sha256'] == inputs[row['case']]['sha256']
        # Explicit scalar arithmetic, distinct from evaluator math.dist.
        error = sum((p - t) ** 2 for p, t in zip(row['prediction_m'], row['target_m'])) ** .5 * 1000
        assert math.isclose(error, row['error_mm'], abs_tol=1e-10)
        assert row['passed_20mm'] == (error <= 20)
        errors[key] = error
        checks += 2
    for name, aggregate in result['episodes'].items():
        ee = [errors.get((name, i)) for i in range(7)]
        assert aggregate['valid_predictions'] == sum(e is not None for e in ee)
        if any(e is None for e in ee):
            assert aggregate['MAE_mm'] is None
            continue
        recomputed = dict(MAE_mm=math.fsum(ee) / 7,
                          harm_mm=math.fsum(max(e-errors['keep', i], 0) for i, e in enumerate(ee)) / 7,
                          improvement_mm=math.fsum(max(errors['keep', i]-e, 0) for i, e in enumerate(ee)) / 7,
                          passed_20mm=sum(e <= 20 for e in ee))
        for field, value in recomputed.items():
            assert math.isclose(value, aggregate[field], abs_tol=1e-10)
            checks += 1
    sealed = {r['id']: r for r in seal['results']}
    for arm, values in result['arms'].items():
        valid = all(sealed[name]['valid'] and result['episodes'][name]['complete_predictions'] for name in values['episodes'])
        assert valid == values['all_valid']
        for field in ['MAE_mm', 'harm_mm', 'improvement_mm']:
            if valid:
                expected = math.fsum(result['episodes'][name][field] for name in values['episodes']) / len(values['episodes'])
                assert math.isclose(values[field], expected, abs_tol=1e-10)
            else:
                assert values[field] is None
            checks += 1
    ordinary, method = result['arms']['ordinary'], result['arms']['edit_review']
    gate = bool(ordinary['all_valid'] and method['all_valid']
                and method['MAE_mm'] < ordinary['MAE_mm']
                and method['MAE_mm'] < result['episodes']['keep']['MAE_mm']
                and method['harm_mm'] <= ordinary['harm_mm']
                and method['improvement_mm'] >= ordinary['improvement_mm'])
    assert gate == result['joint_gate_passed']
    usage, commands, fit_flags, parse_errors, imports, versions = [], [], [], [], set(), []
    parsed = 0
    for episode in read(root / 'SESSIONS392.json'):
        for stage in episode['stages']:
            run = Path(episode['path']) / stage
            name = episode['id'] + '_' + stage
            state = read(run / 'primary' / name / 'status.json')
            assert state['status'] != 'running' and not Path('/proc/' + str(state['cli_pid'])).exists()
            gateway = run / 'model_gateway' / name
            gs = read(gateway / 'status.json')
            assert not gs['active'] and not gs['inflight']
            receipts = [read(p) for p in sorted(gateway.glob('*_receipt.json'))]
            assert len(receipts) == gs['requests']
            for field in ['input_tokens', 'output_tokens']:
                assert sum(r['usage'].get(field, 0) for r in receipts) == gs[field]
            for receipt in receipts:
                prefix = gateway / f"{receipt['sequence']:03d}"
                assert sha(prefix.with_name(prefix.name + '_request.json')) == receipt['request_sha256']
                assert sha(prefix.with_name(prefix.name + '_response.sse')) == receipt['response_sha256']
                assert receipt['model_requested'] == 'deepseek-flash'
            usage.append(dict(episode=episode['id'], stage=stage, status=state['status'],
                              requests=gs['requests'], receipts=len(receipts), input_tokens=gs['input_tokens'],
                              output_tokens=gs['output_tokens'],
                              http_statuses=dict(Counter(r['upstream_status'] for r in receipts)),
                              returned_model_ids=sorted({m for r in receipts for m in r['returned_model_ids']})))
            for line in (run / 'primary' / name / 'events.jsonl').read_text().splitlines():
                event = json.loads(line)
                if event.get('type') == 'item.completed' and event.get('item', {}).get('type') == 'command_execution':
                    item = event['item']
                    commands.append(dict(episode=episode['id'], stage=stage, command=item['command'], exit_code=item.get('exit_code')))
            for p in (run / 'app').rglob('*.py'):
                if '.runtime' in p.parts:
                    continue
                try:
                    tree = ast.parse(p.read_text())
                except Exception as exc:
                    parse_errors.append(dict(path=str(p.relative_to(root)), error=str(exc)))
                    continue
                parsed += 1
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.update(x.name.split('.')[0] for x in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.add(node.module.split('.')[0])
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ['fit', 'fit_transform', 'backward', 'step']:
                        fit_flags.append(dict(path=str(p.relative_to(root)), line=node.lineno, call=node.func.attr))
        if episode['arm'] == 'edit_review':
            proposal = Path(episode['path']) / 'propose' / 'app' / 'solution.py'
            final = root / 'sealed' / episode['id'] / 'solution.py'
            versions.append(dict(episode=episode['id'], proposal_sha256=sha(proposal) if proposal.exists() else None,
                                 final_sha256=sha(final) if final.exists() else None,
                                 review_changed_program=proposal.exists() and final.exists() and sha(proposal) != sha(final),
                                 review_manifest_present=(root / 'sealed' / episode['id'] / 'review_manifest.json').exists()))
    totals = {field: sum(r[field] for r in usage) for field in ['requests', 'receipts', 'input_tokens', 'output_tokens']}
    output = dict(passed=True, recomputed_metric_fields=checks, joint_gate_passed=gate,
                  usage_rows=usage, usage_totals=totals, new_native_sessions=len(usage),
                  new_native_normal=sum(r['status'] == 'finished' for r in usage),
                  new_native_incomplete=sum(r['status'] != 'finished' for r in usage),
                  commands=len(commands), python_files_parsed=parsed, imports=sorted(imports),
                  potential_fit_calls=fit_flags, parse_errors=parse_errors, proposal_final_versions=versions,
                  source_receipt_hashes_checked=True, no_new_predictions_or_model_calls=True,
                  limitation='Static fit flags and command inventory require interpretation; no automatic causal attribution or training-absence proof.')
    for name, value in [('COMPLETE_AUDIT392.json', output), ('COMMAND_INVENTORY392.json', commands)]:
        path = root / name
        assert not path.exists(), f'Preserve prior audit: {path}'
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ['usage_rows', 'imports', 'proposal_final_versions']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run)
