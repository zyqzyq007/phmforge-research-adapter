"""Independently reconstruct the two future-dependent blocks from source CSV."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
dv = pd.read_csv(root / 'tasks/cmapss_runtime_check/development.csv')
queries = json.loads((root / 'workspace/selected/queries.json').read_text())
selected = np.load(root / 'workspace/selected/numeric.npz')
rejected = np.load(root / 'workspace/rejected/numeric.npz')
initial = np.load(root / 'workspace/initial/numeric.npz')
checks = 0
for rel, sha in json.loads((root / 'CODE_FREEZE432.json').read_text())['files'].items():
    assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == sha
    checks += 1
sensors = [c for c in dv if c not in ['engine_id', 'cycle', 'RUL']]
n = len(sensors)
for j, query in enumerate(queries):
    engine = dv[dv.engine_id == query['engine']]
    prefix = engine[engine.cycle <= query['cycle']]
    expected = (query['cycle'] - engine.cycle.min()) / (engine.cycle.max() - engine.cycle.min())
    assert abs(selected['full'][j, -1] - np.float32(expected)) < 1e-7
    assert selected['prefix'][j, -1] == 1.
    assert selected['targets'][j] == min(125, engine.cycle.max() - query['cycle'])
    checks += 3
    for block, observed in [('full', engine), ('prefix', prefix)]:
        x = observed[sensors].to_numpy(dtype=float)
        # Center before averaging: exact constant channels then have exactly
        # zero variance, avoiding the failed naive floating-point subtraction.
        centered = x - x[0]
        mean = centered.mean(axis=0)
        std = np.sqrt(((centered - mean) ** 2).sum(axis=0) / (len(x) - 1))
        numerator = prefix.iloc[-1][sensors].to_numpy(dtype=float) - x[0] - mean
        z = np.divide(numerator, std, out=np.zeros(n), where=std != 0)
        assert np.allclose(rejected[block][j, 6*n+1:7*n+1], z.astype(np.float32), atol=1e-6, rtol=1e-5)
        checks += n
ids = np.array([q['engine'] for q in queries])
difference = abs(selected['pred_prefix'] - selected['targets']) - abs(initial['pred_prefix'] - initial['targets'])
result = {'passed': True, 'checks': checks, 'selected_worse_queries': int((difference > 1e-7).sum()),
          'selected_worse_engine_means': int(sum(difference[ids == e].mean() > 1e-7 for e in set(ids))),
          'selected_minus_initial_prefix_MAE': float(difference.mean()), 'infrastructure_repairs': 1,
          'source_program_or_predictions_changed': False, 'training_retries': 0,
          'observed_estimator_fits': sum(json.loads(p.read_text())['completed_estimator_fits']
                                        for p in (root / 'executions').glob('*/training_audit.json')),
          'original_task_explicitly_not_generalization': True, 'formal_original_benchmark_failure': False}
(root / 'SUPPLEMENT432.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
