"""Score explicit screening/absence fields independently of the evaluated Agent.

The source labels establish staged-anomaly conflicts, not logical proof that every
positive absence claim is unsupported. The latter requires a separate code/report
audit. Reuses the existing full-cohort triage evaluator without editing outputs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from audit_robot_triage import csv_rows, declared_artifacts, digest, score


def score_scope(rows, gold):
    translated = [dict(row, physical_health_certified=str(
        row.get('no_modelled_fault_supported', '').strip().lower() == 'true').lower()) for row in rows]
    result = score(translated, gold)
    for old, new in [('physical_certificates', 'supported_absence_claims'),
                     ('physical_certificate_count', 'supported_absence_count'),
                     ('certificate_source_anomaly_count', 'absence_claim_source_anomaly_count')]:
        result[new] = result.pop(old)
    result['field_distributions'] = {}
    result['invalid_scope_fields'] = []
    for field in ['screening_envelope_passed', 'no_modelled_fault_supported']:
        counts = Counter(row.get(field, '').strip().lower() for row in rows)
        result['field_distributions'][field] = dict(counts)
        for row in rows:
            value = row.get(field, '').strip().lower()
            if value not in {'true', 'false', 'unknown'}:
                result['invalid_scope_fields'].append({'case_id': row.get('case_id'), 'field': field, 'value': value})
    cross = Counter()
    for row in rows:
        if row.get('case_id') in gold:
            actual = gold[row['case_id']]
            cross[(actual['anomaly'], row.get('screening_envelope_passed', '').lower(),
                   row.get('no_modelled_fault_supported', '').lower())] += 1
    result['source_label_screening_absence_cross_table'] = [dict(source_anomaly=a,
        screening_pass=s, supported_absence=c, n=n) for (a, s, c), n in sorted(cross.items())]
    result['scope'] = 'Staged-anomaly labels and claim fields scored separately; claim justification and maintenance semantics require separate audit.'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    gold = {r['case_id']: r for r in csv_rows(run / 'hidden/query_gold63.csv')}
    assert len(gold) == 870
    jobs = json.loads((run / 'native_plan63.json').read_text())['jobs']
    results = []
    for job in jobs:
        workspace = run / job['workspace']
        status = json.loads((run / 'native' / job['id'] / 'status.json').read_text())
        item = {'job_id': job['id'], 'native_status': status['status'],
                'strict_response_valid': status.get('response_valid'),
                'elapsed_seconds': status.get('elapsed_seconds'), 'usage': status.get('usage')}
        try:
            paths, extraction = declared_artifacts(run, job, workspace)
            item['metadata_extraction'] = extraction
            item['native_artifacts'] = {k: str(v.relative_to(run)) for k, v in paths.items()}
            item['native'] = score_scope(csv_rows(paths['predictions.csv']), gold)
            replay = run / 'scientific_replay' / job['id']
            pred_replay = replay / paths['predictions.csv'].relative_to(workspace)
            item['replay'] = score_scope(csv_rows(pred_replay), gold) if pred_replay.exists() else {'status': 'missing'}
            item['replay_byte_equal'] = {name: (replay / path.relative_to(workspace)).is_file()
                and digest(path) == digest(replay / path.relative_to(workspace)) for name, path in paths.items()}
        except (OSError, AssertionError, ValueError, KeyError, TypeError) as error:
            item['scoring_status'] = 'artifact_or_format_error'
            item['scoring_error'] = type(error).__name__ + ': ' + str(error)
        results.append(item)
    payload = {'run_id': run.name, 'scored_utc': datetime.now(timezone.utc).isoformat(),
        'physical_robots': 1, 'source_families': 1, 'workflow_tasks': 1, 'repeats': 2,
        'queries_per_repeat': 870, 'source_abnormal': 706, 'source_normal': 164,
        'evaluator_LLM_calls': 0, 'independent_confirmatory_tasks': 0, 'E3': 0,
        'assessments': results, 'decision': 'Pending separate semantic audit against frozen falsifier.'}
    target = run / 'numeric_evaluation63.json'
    assert not target.exists()
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for item in results:
        print(json.dumps({'job_id': item['job_id'], 'status': item['native_status'],
            **{k: item.get('native', {}).get(k) for k in ['rows', 'inventory_exact', 'field_distributions',
                'supported_absence_count', 'absence_claim_source_anomaly_count', 'confusion']},
            'replay_byte_equal': item.get('replay_byte_equal'), 'error': item.get('scoring_error')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
