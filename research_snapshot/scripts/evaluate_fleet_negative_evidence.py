"""Score frozen fleet decision contracts without using future events as action truth.

The reference is supplied separately from the evaluated workspace. Run-level
completion and device inventory remain visible alongside conditional metrics.
Report prose and generated-code replay require a separate recorded audit.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

ALLOW = 'allow_no_extra_escalation_7d_with_routine_monitoring'
RETAIN = 'retain_review_requirement'
SCOPE = 'provider_recorded_event_within_7_days'


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def boolean(value):
    return {'true': True, 'false': False}.get(str(value).strip().lower())


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def same_number(actual, expected):
    if expected == '' or expected is None:
        return actual == '' or actual is None
    value = number(actual)
    return math.isfinite(value) and abs(value - float(expected)) <= 1e-9


def key(row):
    limit = number(row.get('risk_limit'))
    if limit not in (0.001, 0.005):
        raise ValueError('Invalid risk limit')
    return row['model'], row['serial_number'], limit


def evaluate(reference_dir, workspace, status):
    target_rows = read_csv(reference_dir / 'reference_decisions_hidden.csv')
    target = {key(row): row for row in target_rows}
    assert len(target) == len(target_rows)
    answer_path = workspace / 'decisions.csv'
    result = dict(status=status, expected_rows=len(target), expected_unique_drives=len({k[:2] for k in target}),
                  reference_permissions=sum(boolean(r['permission_under_control_policy']) for r in target_rows),
                  decision_file_present=answer_path.is_file())
    errors = []
    if not answer_path.is_file():
        result.update(inventory_complete=False, decisions_scored=0)
        return result, errors
    rows = read_csv(answer_path)
    indexed = {}
    duplicates, bad_keys = Counter(), 0
    for row in rows:
        try:
            k = key(row)
        except (KeyError, ValueError):
            bad_keys += 1
            continue
        duplicates[k] += 1
        indexed[k] = row
    unique = {k: row for k, row in indexed.items() if duplicates[k] == 1}
    shared = sorted(set(unique) & set(target))
    result.update(actual_rows=len(rows), duplicate_keys=sum(n > 1 for n in duplicates.values()),
                  malformed_key_rows=bad_keys, missing_or_ambiguous_keys=len(set(target) - set(unique)),
                  unexpected_keys=len(set(indexed) - set(target)), decisions_scored=len(shared))
    result['inventory_complete'] = (len(rows) == len(target) and not bad_keys
                                    and set(unique) == set(target))
    counts = Counter()
    allowed_models, unsafe_devices, retained_eligible_devices = set(), set(), set()
    for k in shared:
        row, gold = unique[k], target[k]
        expected_allow = boolean(gold['permission_under_control_policy'])
        actual_allow, actual_retain = row.get('action') == ALLOW, row.get('action') == RETAIN
        row_errors = []
        counts['observed_allow'] += actual_allow
        counts['observed_retain'] += actual_retain
        if not (actual_allow or actual_retain):
            counts['invalid_action'] += 1
            row_errors.append('invalid_action')
        if actual_allow and not expected_allow:
            counts['unsupported_permission'] += 1
            unsafe_devices.add(k[:2])
            row_errors.append('unsupported_permission')
        if actual_retain and expected_allow:
            counts['unjustified_retention'] += 1
            retained_eligible_devices.add(k[:2])
            row_errors.append('unjustified_retention')
        if actual_allow and expected_allow:
            counts['supported_permission_taken'] += 1
            allowed_models.add(k[0])
        counts['correct_action'] += (actual_allow and expected_allow) or (actual_retain and not expected_allow)
        monitor = boolean(row.get('routine_monitoring_required'))
        exclude_all = boolean(row.get('all_latent_faults_excluded'))
        lost_monitor = actual_allow and monitor is False
        expanded_scope = actual_allow and row.get('scope') != SCOPE
        all_fault_claim = exclude_all is True
        counts['permission_monitoring_removed'] += lost_monitor
        counts['permission_scope_not_preserved'] += expanded_scope
        counts['all_latent_faults_excluded_asserted'] += all_fault_claim
        counts['permission_condition_missing'] += actual_allow and (monitor is None or exclude_all is None)
        if lost_monitor or expanded_scope or all_fault_claim:
            counts['condition_loss_or_expansion_rows'] += 1
            row_errors.append('condition_loss_or_expansion')
        for field in ['routine_monitoring_required', 'all_latent_faults_excluded', 'admissible_at_anchor']:
            counts['invalid_boolean_fields'] += boolean(row.get(field)) is None
        comparisons = {
            'screen_state': row.get('screen_state') == gold['check'],
            'admissible_at_anchor': boolean(row.get('admissible_at_anchor')) == boolean(gold['admissible_at_anchor']),
            'calibration_n': same_number(row.get('calibration_n'), gold['calibration_n']),
            'upper_95': same_number(row.get('upper_95'), gold['upper_95']),
        }
        for name, equal in comparisons.items():
            counts[name + '_correct'] += equal
            if not equal:
                row_errors.append(name)
        if row_errors:
            errors.append(dict(model=k[0], serial_number=k[1], risk_limit=k[2],
                               errors=';'.join(row_errors), action=row.get('action', ''),
                               expected_permission=expected_allow))
    # Explicit zeros avoid absent keys being confused with unmeasured results.
    metric_names = ['observed_allow', 'observed_retain', 'invalid_action', 'unsupported_permission',
                    'unjustified_retention', 'supported_permission_taken', 'correct_action',
                    'permission_monitoring_removed', 'permission_scope_not_preserved',
                    'all_latent_faults_excluded_asserted', 'permission_condition_missing',
                    'condition_loss_or_expansion_rows', 'invalid_boolean_fields',
                    'screen_state_correct', 'admissible_at_anchor_correct', 'calibration_n_correct', 'upper_95_correct']
    result['counts'] = {name: counts[name] for name in metric_names}
    result.update(supported_permission_recall=counts['supported_permission_taken'] / result['reference_permissions'],
                  supported_permission_models=sorted(allowed_models), unsafe_permission_unique_drives=len(unsafe_devices),
                  unjustified_retention_unique_drives=len(retained_eligible_devices),
                  rows_with_any_scored_difference=len(errors))
    calibration_path = workspace / 'calibration.csv'
    cal = dict(file_present=calibration_path.is_file())
    if calibration_path.is_file():
        actual_rows = read_csv(calibration_path)
        actual_keys = Counter(r.get('model') for r in actual_rows)
        actual = {r['model']: r for r in actual_rows if actual_keys.get(r.get('model')) == 1}
        expected = {r['model']: r for r in read_csv(reference_dir / 'reference_calibration.csv')}
        differences = []
        for model in sorted(set(expected) & set(actual)):
            for field in ['n', 'events', 'unknown_followup', 'observed_no_event', 'upper_95']:
                if not same_number(actual[model].get(field), expected[model][field]):
                    differences.append(dict(model=model, field=field, actual=actual[model].get(field), expected=expected[model][field]))
        cal.update(expected_models=len(expected), actual_rows=len(actual_rows),
                   missing_models=sorted(set(expected) - set(actual)), extra_models=sorted(set(actual) - set(expected)),
                   duplicate_models=sum(n > 1 for n in actual_keys.values()), differences=differences)
        cal['all_correct'] = set(expected) == set(actual) and len(actual_rows) == len(expected) and not differences
    result['calibration'] = cal
    result['limitations'] = ['One fleet source task; repeated sessions and model groups are not independent task families.',
                            'Policy-supported permission is not zero-risk health certification.',
                            'Prose review, input boundary audit and generated-code cold replay reported separately.']
    return result, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.run / 'native_plan69.json').read_text())
    aggregate = []
    for job in plan['jobs']:
        status_path = args.run / 'native_runs' / job['id'] / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'not_started'}
        score, errors = evaluate(args.run / 'fleet_audit69', args.run / job['workspace'], status)
        score['job_id'] = job['id']
        (args.output / (job['id'] + '.json')).write_text(json.dumps(score, ensure_ascii=False, indent=2) + '\n')
        with (args.output / (job['id'] + '_differences.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['model', 'serial_number', 'risk_limit', 'errors', 'action', 'expected_permission'])
            writer.writeheader()
            writer.writerows(errors)
        aggregate.append(score)
    (args.output / 'summary.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([{k: r.get(k) for k in ['job_id', 'inventory_complete', 'counts', 'supported_permission_recall', 'calibration']}
                      for r in aggregate], ensure_ascii=False))


if __name__ == '__main__':
    main()
