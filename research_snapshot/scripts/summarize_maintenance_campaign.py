"""Summarize a mixed original-maintenance campaign without pooling cost units."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from summarize_imprl_eval import paired_costs


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def compare(episodes, reference):
    result = paired_costs(episodes, reference, seed=9000)
    diff = np.array([x['cost']-y['cost'] for x, y in zip(episodes, reference)])
    se = diff.std(ddof=1)/np.sqrt(len(diff))
    p = float(student_t.sf(diff.mean()/se, len(diff)-1)) if se else float(diff.mean() <= 0)
    result['paired_mean_loss_one_sided_p'] = p
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--plan', default='plan90.json')
    parser.add_argument('--output', default='comparison90.json')
    args = parser.parse_args()
    run = args.run
    jobs = read(run/args.plan)['jobs']
    rows = []
    for job in jobs:
        native = read(run/'native_runs'/job['id']/'status.json') or {'status': 'not_started'}
        result = read(run/'evaluation'/job['id']/'status.json') or {'status': 'pending'}
        row = dict(job=job['id'], setting=job['setting'], family=job['family'],
                   repeat=job['repeat'], native_status=native['status'],
                   evaluation_status=result['status'], primary_eligible=False,
                   completed_episodes=result.get('completed_episodes', 0),
                   comparisons={}, cost_screen=False, holm_input_p=1.0)
        if result['status'] == 'finished':
            episodes = read(run/'evaluation'/job['id']/'episodes.json')
            row.update(mean_cost=result['mean_cost'],
                       mean_inspections=result['mean_inspections'],
                       mean_replacements=result['mean_replacements'])
            names = [job['setting']+'_reference', job['setting']+'_fixed180'] if job['family']=='IMPRL' else ['impact_toy_v2_fixed']
            for name in names:
                ref_status = read(run/'evaluation'/name/'status.json')
                if ref_status and ref_status['status'] == 'finished':
                    reference = read(run/'evaluation'/name/'episodes.json')
                    row['comparisons'][name] = compare(episodes, reference)
            row['primary_eligible'] = native['status']=='finished' and len(row['comparisons'])==len(names)
            if row['primary_eligible']:
                row['cost_screen'] = all(x['screening_flag'] for x in row['comparisons'].values())
                # Intersection-union: loss must exist against both IMPRL refs.
                row['holm_input_p'] = max(x['paired_mean_loss_one_sided_p'] for x in row['comparisons'].values())
        rows.append(row)
    largest = 0.
    for rank, row in enumerate(sorted(rows, key=lambda x: x['holm_input_p'])):
        largest = max(largest, min(1., (len(rows)-rank)*row['holm_input_p']))
        row['holm_adjusted_p'] = largest
        row['cost_screen_holm_supported'] = row['cost_screen'] and largest < .05
    summary = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                   planned_native_sessions=len(jobs), native_counts=dict(Counter(x['native_status'] for x in rows)),
                   evaluation_counts=dict(Counter(x['evaluation_status'] for x in rows)),
                   primary_scored_completed=sum(x['primary_eligible'] for x in rows),
                   exploratory_cost_flags=sum(x['cost_screen'] for x in rows),
                   holm_supported_cost_flags=sum(x['cost_screen_holm_supported'] for x in rows),
                   original_configurations=len({x['setting'] for x in rows}),
                   source_families=sorted({x['family'] for x in rows}),
                   scope='Per-program cost screening only; no mechanism, commonality or hardness inferred from scores. Different original cost units not pooled.',
                   statistical_note='Paired bootstrap95CI (10000 draws, seed9000); paired one-sided t test and Holm over all planned sessions; native generation repeats are not independent source families.',
                   rows=rows)
    (run/args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
