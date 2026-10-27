"""Score source-bound diagnosis outputs without exposing gold to agents."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def evaluate(job, gold, rows):
    expected = [g['case_id'] for g in gold]
    counts = Counter(r.get('case_id') for r in rows if isinstance(r, dict))
    matched = {r.get('case_id'): r for r in rows if isinstance(r, dict)}
    details = []
    for g in gold:
        cid = g['case_id']
        r = matched.get(cid, {})
        present = counts[cid] == 1
        label = r.get('class_code') if job['family'] == 'engine' else r.get('predicted_label')
        if job['family'] == 'engine':
            valid = 'class_code' in r and (label is None or (type(label) is int and label in range(6)))
            unknown = label is None or label == 5
        else:
            valid = label in ['f16', 'f17', 'f18', 'f19', 'unknown']
            unknown = label == 'unknown'
        category = ('missing' if counts[cid] == 0 else 'invalid' if not present or not valid
                    else 'unknown' if unknown else 'correct' if label == g['expected_label'] else 'wrong')
        details.append({'case_id': cid, 'source_record': g.get('record_id', g.get('source_event_id')),
                        'source_member': g.get('source_member'), 'source_time': g.get('time_seconds'),
                        'truth': g['expected_label'], 'predicted': label, 'category': category})
    summary = dict(Counter(r['category'] for r in details))
    inventory_exact = [r.get('case_id') if isinstance(r, dict) else None for r in rows] == expected
    unexpected = sum(n for cid, n in counts.items() if cid not in expected)
    return {'planned_queries': len(gold), **{k: summary.get(k, 0) for k in ['correct', 'wrong', 'unknown', 'missing', 'invalid']},
            'inventory_exact': inventory_exact, 'unexpected_entries': unexpected, 'rows': details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--plan', default='plan87.json')
    parser.add_argument('--gold', default='hidden/gold87.json')
    parser.add_argument('--native-directory', default='native_runs')
    parser.add_argument('--output', default='primary_scores87.json')
    args = parser.parse_args()
    root = args.run.resolve()
    gold = read(root / args.gold)
    tasks = []
    for job in read(root / args.plan)['jobs']:
        work = root / job['workspace']
        sp = root / args.native_directory / job['id'] / 'status.json'
        status = read(sp) if sp.exists() else {'status': 'not_started'}
        pred = work / 'predictions.json'
        rows, parse_error = [], None
        if pred.exists():
            try:
                rows = read(pred)['results']
                if not isinstance(rows, list):
                    raise TypeError('results must be a list')
            except (ValueError, TypeError, KeyError) as e:
                parse_error = type(e).__name__
                rows = []
        result = evaluate(job, [g for g in gold if g['job_id'] == job['id']], rows)
        result.update(job_id=job['id'], family=job['family'], native_status=status['status'],
                      response_valid=status.get('response_valid'), elapsed_seconds=status.get('elapsed_seconds'),
                      parse_error=parse_error, terminal=status['status'] in ['finished', 'execution_error', 'timeout'],
                      artifacts_present={n: (work / n).is_file() for n in ['analysis.py', 'predictions.json', 'validation.json', 'report.md']},
                      prediction_sha256=hashlib.sha256(pred.read_bytes()).hexdigest() if pred.exists() else None)
        tasks.append(result)
    changes = []
    for entry in read(root / 'input_manifest87.json'):
        p = root / 'tasks' / entry['job_id'] / entry['file']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != entry['sha256']:
            changes.append({'job_id': entry['job_id'], 'file': entry['file']})
    output = {'tasks': tasks, 'input_changes': changes, 'all_native_terminal': all(t['terminal'] for t in tasks),
              'scope': 'Original-source diagnosis labels; correlated historical data discovery, not official Agentbenchmark scores or independent commonness confirmation.', 'LLM_calls_in_scorer': 0}
    (root / args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'tasks': [{k: t[k] for k in ['job_id', 'native_status', 'correct', 'wrong', 'unknown', 'missing', 'invalid']} for t in tasks], 'input_changes': len(changes)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
