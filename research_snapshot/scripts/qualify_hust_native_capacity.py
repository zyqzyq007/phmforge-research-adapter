"""Resolve source measurement/label conventions using a TRAIN cell only."""
import json
from pathlib import Path
import resource

import numpy as np
from pandas.compat.pickle_compat import load

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
config = json.loads(Path('/config.json').read_text())
with Path('/cell.pkl').open('rb') as handle:
    raw = load(handle)
assert set(raw) == {config['training_cell']}
cell = raw[config['training_cell']]
keys = sorted(cell['data'], key=int)
rows = []
for key in keys:
    frame = cell['data'][key]
    mask = frame['Status'].str.contains('discharge').to_numpy()
    q = frame['Capacity (mAh)'].to_numpy(float)
    current = frame['Current (mA)'].to_numpy(float)
    time = frame['Time (s)'].to_numpy(float)
    status = frame['Status'].to_numpy()
    groups = [np.flatnonzero(status == name) for name in sorted(set(status[mask]))]
    choices = dict(last_discharge_capacity=float(q[np.flatnonzero(mask)[-1]]),
        maximum_discharge_capacity=float(q[mask].max()),
        sum_discharge_step_last=sum(float(q[g[-1]]) for g in groups),
        sum_discharge_step_maximum=sum(float(q[g].max()) for g in groups),
        sum_discharge_step_range=sum(float(np.ptp(q[g])) for g in groups),
        rectangular_within_status=sum(float(np.sum(-current[g][1:] * np.diff(time[g])) / 3600) for g in groups),
        trapezoidal_within_status=sum(float(np.trapz(-current[g], time[g]) / 3600) for g in groups),
        rectangular_whole_cycle=float(np.sum(np.where(current[1:] < 0, -current[1:] * np.diff(time), 0)) / 3600))
    rows.append(dict(cycle=int(key), source_capacity_mAh=float(cell['dq'][key]),
        source_rul=float(cell['rul'][key]), capacities=choices))
truth = np.asarray([r['source_capacity_mAh'] for r in rows])
count = len(keys)
checks = {}
for name in config['fixed_capacity_formulas']:
    values = np.asarray([r['capacities'][name] for r in rows])
    checks[name] = dict(max_abs_error_mAh=float(np.max(np.abs(values - truth))),
        median_abs_error_mAh=float(np.median(np.abs(values - truth))),
        exact_at_frozen_tolerance=bool(np.all(np.abs(values - truth) <= config['origin_match_tolerance_mAh'])))
rul = np.asarray([r['source_rul'] for r in rows])
cycle = np.asarray([r['cycle'] for r in rows])
assert np.array_equal(rul, count - cycle + 1)
result = dict(training_cell=config['training_cell'], cycles=count,
    capacity_reconstructions=checks,
    exact_native_capacity_formulas=[k for k, v in checks.items() if v['exact_at_frozen_tolerance']],
    provided_RUL_equals_last_published_cycle_minus_completed_cycles=True,
    completed_cycles_before_current= 'source_cycle_number - 1',
    source_provided_RUL_unchanged=True,
    strict_880mAh_crossings=int(np.sum(truth <= 880)),
    physical_threshold_independently_confirmed=False,
    boundary_examples=rows[:1] + rows[-1:],
    new_fits=0, new_predictions=0, new_LLM_calls=0)
Path('/work/NATIVE_CAPACITY364.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['cycles', 'exact_native_capacity_formulas',
    'provided_RUL_equals_last_published_cycle_minus_completed_cycles', 'strict_880mAh_crossings']}))
