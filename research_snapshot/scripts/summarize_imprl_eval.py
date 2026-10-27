"""Frozen per-program paired cost comparisons; retain all native outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_costs(episodes, reference, seed=8800, resamples=10000):
    assert [x['seed'] for x in episodes] == [x['seed'] for x in reference]
    candidate_cost = np.array([x['cost'] for x in episodes])
    reference_cost = np.array([x['cost'] for x in reference])
    diff = candidate_cost-reference_cost
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), (resamples, len(diff)))
    ci = np.quantile(np.mean(diff[indices], axis=1), [0.025, 0.975])
    return dict(mean_cost=float(candidate_cost.mean()),
                reference_cost=float(reference_cost.mean()), difference=float(diff.mean()),
                excess_percent=float(100*diff.mean()/reference_cost.mean()), paired_CI95=ci.tolist(),
                screening_flag=bool(diff.mean()/reference_cost.mean()>0.05 and ci[0]>0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--plan', default='plan88.json')
    p.add_argument('--native-directory', default='native_runs')
    p.add_argument('--output', default='comparison88.json')
    p.add_argument('--fixed-suffix', help='Additional same-tool baseline, e.g. fixed180')
    args = p.parse_args()
    run = args.run
    rows = []
    jobs = json.loads((run/args.plan).read_text())['jobs']
    for job in jobs:
        native_path = run/args.native_directory/job['id']/'status.json'
        evaluation = run/'evaluation'/job['id']
        if not native_path.exists() or not (evaluation/'status.json').exists():
            rows.append(dict(job=job['id'], status='pending'))
            continue
        native = json.loads(native_path.read_text())
        result = json.loads((evaluation/'status.json').read_text())
        row = dict(job=job['id'], setting=job['setting'], native_status=native['status'],
                   evaluation_status=result['status'], completed_episodes=result.get('completed_episodes', 0),
                   native_usage=native.get('usage'), primary_eligible=False)
        if result['status'] == 'finished':
            episodes = json.loads((evaluation/'episodes.json').read_text())
            reference = json.loads((run/'evaluation'/f"{job['setting']}_reference"/'episodes.json').read_text())
            row.update(paired_costs(episodes, reference))
            row.update(primary_eligible=native['status']=='finished',
                       mean_failure_steps=result['mean_failure_steps'], mean_replacements=result['mean_replacements'],
                       mean_inspections=result['mean_inspections'], episodes=len(episodes))
            if args.fixed_suffix:
                folder = run/'evaluation'/f"{job['setting']}_{args.fixed_suffix}"
                fixed_status = json.loads((folder/'status.json').read_text())
                row['fixed_baseline_status'] = fixed_status['status']
                if fixed_status['status'] == 'finished':
                    fixed = json.loads((folder/'episodes.json').read_text())
                    row['fixed_baseline_comparison'] = paired_costs(episodes, fixed)
                    row['excess_above_both_screen'] = (row['screening_flag'] and
                        row['fixed_baseline_comparison']['screening_flag'])
        rows.append(row)
    summary = dict(rows=rows, planned_native_sessions=len(jobs), independent_environment_families=1,
                   analysis='Exploratory per-program paired bootstrap; no population prevalence or exact optimum claim',
                   confirmed_common_PHM_Agent_problems=0)
    (run/args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
