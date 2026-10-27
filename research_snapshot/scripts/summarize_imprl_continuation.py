"""Secondary extended-rollout diagnostics, explicitly separate from benchmark scores."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    rows, raw = [], {}
    for folder in sorted((args.run/'evaluation').glob('*continuation')):
        status = json.loads((folder/'status.json').read_text())
        row = dict(name=folder.name, status=status['status'], episodes=status.get('completed_episodes', 0))
        if status['status'] != 'finished':
            rows.append(row)
            continue
        groups = {}
        with gzip.open(folder/'trajectories.jsonl.gz', 'rt') as f:
            for line in f:
                x = json.loads(line)
                g = groups.setdefault(x['seed'], dict(early_fail=0, late_fail=0, late_cost=0.,
                                                       late_replacements=0, late_inspections=0))
                if x['t'] < 20:
                    g['early_fail'] += x['system_failure']
                else:
                    g['late_fail'] += x['system_failure']
                    g['late_cost'] -= .8**(x['t']-20)*x['reward']
                    g['late_replacements'] += x['action'].count(1)
                    g['late_inspections'] += x['action'].count(2)
        row.update(setting=status['setting'], mean_total_cost=status['mean_cost'],
                   mean_total_failure_steps=status['mean_failure_steps'])
        row.update({k: float(np.mean([g[k] for g in groups.values()])) for k in next(iter(groups.values()))})
        raw[folder.name] = groups
        rows.append(row)
        (folder/'late_phase_metrics.json').write_text(json.dumps(groups, indent=2)+'\n')
    for row in rows:
        name = row['name']
        if row['status'] != 'finished' or '_reference_' in name:
            continue
        ref_name = row['setting']+'_reference_continuation'
        if ref_name not in raw:
            continue
        assert raw[name].keys() == raw[ref_name].keys()
        differences = {}
        for metric in ['late_fail', 'late_cost']:
            diff = np.array([raw[name][s][metric]-raw[ref_name][s][metric] for s in raw[name]])
            rng = np.random.default_rng(8800)
            ci = np.quantile(np.mean(diff[rng.integers(0, len(diff), (10000, len(diff)))], axis=1), [.025, .975])
            differences[metric] = dict(difference=float(np.mean(diff)), paired_CI95=ci.tolist())
        row['versus_reference'] = differences
    result = dict(scope='Posthoc 80-step continuation; late phase t20..79; not original 20-step benchmark score',
                  discount_origin='late_cost restarts discount at t20; total cost keeps original t0 origin', rows=rows)
    (args.run/'continuation_comparison88.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
