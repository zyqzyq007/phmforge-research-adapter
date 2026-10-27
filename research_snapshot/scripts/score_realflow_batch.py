"""Source-grounded wear/diagnosis output scoring; no score feedback to agents."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--milling-gold', type=Path, required=True)
    parser.add_argument('--output-name', default='primary_scores83.json')
    args = parser.parse_args()
    run = args.run.resolve()
    gold = read(run / 'hidden/gold83.json')
    bearing_gold = {row['case_id']: row['class_code'] for row in gold if row['family'] == 'bearing'}
    records = []
    for job in read(run / 'plan83.json')['jobs']:
        work = run / job['workspace']
        status_file = run / 'formal83' / job['id'] / 'status.json'
        status = read(status_file) if status_file.exists() else {'status': 'not_started'}
        record = {'job_id': job['id'], 'status': status.get('status'), 'response_valid': status.get('response_valid'), 'elapsed_seconds': status.get('elapsed_seconds'), 'primary_score_is_official': False}
        pred = work / 'predictions.json'
        rows = []
        if pred.is_file():
            try:
                rows = read(pred)['results']
            except (ValueError, KeyError, TypeError):
                record['output_parse_error'] = True
        if job['id'].startswith('Ottawa_'):
            expected = [row['case_id'] for row in read(work / 'queries.json')]
            matched = {row.get('case_id'): row for row in rows if isinstance(row, dict) and row.get('case_id') in expected}
            details = [{'case_id': cid, 'truth': bearing_gold[cid], 'predicted': matched[cid].get('class_code') if cid in matched else None, 'saved': cid in matched} for cid in expected]
            for row in details:
                row['correct'] = row['saved'] and row['predicted'] == row['truth']
            record.update(family='bearing_diagnosis', planned_queries=len(expected), saved_queries=len(matched), correct=sum(x['correct'] for x in details), wrong_answered=sum(x['saved'] and x['predicted'] is not None and not x['correct'] for x in details), abstentions=sum(x['saved'] and x['predicted'] is None for x in details), missing_queries=sum(not x['saved'] for x in details), query_inventory_exact=[x.get('case_id') for x in rows] == expected, rows=details)
        else:
            material = 'milling_material1' if 'material1_' in job['id'] else 'milling_material2'
            with (args.milling_gold / (material + '.csv')).open() as handle:
                truth = {row['query_id']: float(row['VB']) for row in csv.DictReader(handle)}
            with (work / 'query_inventory.csv').open() as handle:
                expected = [row['query_id'] for row in csv.DictReader(handle)]
            matched = {row.get('case_id'): row for row in rows if isinstance(row, dict) and row.get('case_id') in expected}
            details = []
            errors, widths, covered = [], [], []
            for cid in expected:
                row = matched.get(cid, {})
                point = row.get('wear_estimate_mm')
                lo, hi = row.get('interval_low_mm'), row.get('interval_high_mm')
                valid_interval = finite(lo) and finite(hi) and 0 <= lo <= hi
                if finite(point):
                    errors.append(point - truth[cid])
                if valid_interval:
                    widths.append(hi - lo)
                    covered.append(lo <= truth[cid] <= hi)
                details.append({'case_id': cid, 'truth_mm': truth[cid], 'prediction_mm': point, 'saved': cid in matched, 'finite_point': finite(point), 'finite_interval': valid_interval})
            record.update(family='contemporaneous_wear', planned_queries=len(expected), saved_queries=len(matched), finite_points=len(errors), missing_queries=len(expected)-len(matched), MAE_mm=sum(abs(e) for e in errors)/len(errors) if errors else None, RMSE_mm=math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None, finite_intervals=len(widths), interval_coverage=sum(covered)/len(covered) if covered else None, mean_interval_width_mm=sum(widths)/len(widths) if widths else None, query_inventory_exact=[x.get('case_id') for x in rows] == expected, rows=details, comparative_loss_scope='MAE comparison only; no invented binary success threshold or physical maintenance loss')
        record['scientific_artifacts_present'] = {n: (work / n).is_file() for n in ['analysis.py', 'predictions.json', 'validation.json', 'report.md']}
        records.append(record)
    changed = []
    for row in read(run / 'input_manifest83.json'):
        work = run / 'tasks' / row['job_id']
        p = work / row['file']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            changed.append({'job_id': row['job_id'], 'file': row['file']})
    result = {'tasks': records, 'input_changes': changed, 'new_model_sessions_in_scorer': 0, 'scope': 'Partial running scores are observations; terminal full denominator required. Diagnosis errors, abstentions, non-completion and validation/source defects remain separate. No commonness or physical device loss inferred.'}
    (run / args.output_name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status_counts': {s: sum(r['status'] == s for r in records) for s in sorted({r['status'] for r in records})}, 'tasks': [{k: r.get(k) for k in ['job_id', 'status', 'correct', 'wrong_answered', 'abstentions', 'MAE_mm']} for r in records], 'input_changes': len(changed)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
