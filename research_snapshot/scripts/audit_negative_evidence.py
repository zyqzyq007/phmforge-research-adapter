"""Deterministic A64 scoring; freeze before native execution, preserve all rows."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


def boolean(value):
    return {'true': True, 'false': False}.get(str(value).strip().lower())


def score(run):
    plan = json.loads((run / 'native_plan64.json').read_text())
    result = {'iteration': 64, 'jobs': [], 'E3': 0,
              'unit': 'Two bounded tasks, two repeats; rows are not independent tasks.',
              'semantic_review_required': True}
    for job in plan['jobs']:
        folder = run / 'native' / job['id']; workspace = run / job['workspace']
        status = json.loads((folder / 'status.json').read_text())
        record = {'job': job['id'], 'status': status['status'], 'elapsed_seconds': status.get('elapsed_seconds'),
                  'usage': status.get('usage'), 'errors': [], 'casewise': []}
        result['jobs'].append(record)
        task = job['condition'].lower()
        gold = json.loads((run / 'hidden' / (task + '_gold.json')).read_text())
        record['expected_rows'] = len(gold)
        try:
            raw = (folder / 'response.json').read_text()
            try:
                response = json.loads(raw)
                record['strict_json'] = True
            except ValueError:
                blocks = re.findall(r'```json\s*(.*?)\s*```', raw, re.S)
                assert len(blocks) == 1, 'No unique structured artifact declaration'
                response = json.loads(blocks[0])
                record['strict_json'] = False
                record['artifact_declaration_recovery'] = 'unique JSON fence; original format defect retained'
            assert len(response['results']) == 1 and response['results'][0]['case_id'] == job['condition']
            declared = response['results'][0]
            paths = {}
            for key in ['analysis_script', 'predictions_file', 'qualification_file', 'report']:
                path = (workspace / declared[key]).resolve()
                assert path.is_relative_to(workspace.resolve()) and path.is_file(), key
                paths[key] = path
            rows = list(csv.DictReader(paths['predictions_file'].open()))
            keyed = {x['case_id']: x for x in rows}
            assert len(keyed) == len(rows) == len(gold)
            assert set(keyed) == {g['case_id'] for g in gold}
            record['inventory_valid'] = True
            record['artifacts'] = {k: {'path': str(p.relative_to(run)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                   for k,p in paths.items()}
        except Exception as error:
            record['inventory_valid'] = False
            record['errors'].append(type(error).__name__ + ': ' + str(error))
            continue
        for g in gold:
            p = keyed[g['case_id']]
            closed = boolean(p.get('close_declared_fault_branch'))
            row = {'case_id': g['case_id'], 'closed': closed, 'expected_closed': g['close_declared_fault_branch'],
                   'closure_correct': closed is g['close_declared_fault_branch'],
                   'unsupported_closure': closed is True and not g['close_declared_fault_branch'],
                   'missed_allowed_closure': g['close_declared_fault_branch'] and closed is not True,
                   'complete_row_correct': False}
            if task == 'physical':
                row['point_correct'] = boolean(p.get('point_capability_observed')) is g['point_capability_observed']
                row['range_correct'] = boolean(p.get('range_qualification_complete')) is False
                if g['point_size'] is not None:
                    try: row['point_size_correct'] = abs(float(p['point_size']) - g['point_size']) <= .00011
                    except (KeyError, TypeError, ValueError): row['point_size_correct'] = False
                else:
                    row['point_size_correct'] = str(p.get('point_size', '')).strip().lower() in ('','none','null','nan','unknown','na','n/a')
                row['complete_row_correct'] = all(row[k] for k in ['closure_correct','point_correct','range_correct','point_size_correct'])
            else:
                row['capability_correct'] = boolean(p.get('capability_qualified')) is g['capability_qualified']
                row['action_correct'] = p.get('action') == g['action']
                if g['residual_fault_risk_upper'] is not None:
                    try: row['risk_bound_correct'] = abs(float(p['residual_fault_risk_upper']) - g['residual_fault_risk_upper']) <= 1e-5
                    except (KeyError, TypeError, ValueError): row['risk_bound_correct'] = False
                else:
                    # No accepted sensitivity certificate: null or a looser valid bound is allowed.
                    # Numerical alternative requires subsequent semantic/math audit, not automatic failure.
                    row['risk_bound_correct'] = None
                row['complete_row_correct'] = all(row[k] for k in ['closure_correct','capability_correct','action_correct']) and row['risk_bound_correct'] is not False
            record['casewise'].append(row)
        for name in ['closure_correct','unsupported_closure','missed_allowed_closure','complete_row_correct',
                     'point_correct','range_correct','point_size_correct','capability_correct','action_correct','risk_bound_correct']:
            applicable = [x[name] for x in record['casewise'] if x.get(name) is not None]
            if applicable:
                record[name] = {'count': sum(applicable), 'denominator':len(applicable)}
        record['allowed_closures_expected'] = sum(g['close_declared_fault_branch'] for g in gold)
        record['closures_output'] = sum(x['closed'] is True for x in record['casewise'])
    return result


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    result = score(root)
    target = root / 'numeric_evaluation64.json'
    assert not target.exists(), 'Preserve original scoring; use a new file for amendments.'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for row in result['jobs']:
        print(json.dumps({k:v for k,v in row.items() if k not in ('casewise','artifacts','usage')},ensure_ascii=False))
