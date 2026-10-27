"""Read one predeclared TRAIN cell inside the source qualification namespace.

Never run on the whole source ZIP; only the selected train member is mounted.
"""
from collections import Counter
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd
from pandas.compat.pickle_compat import load

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
config = json.loads(Path('/config.json').read_text())
with Path('/cell.pkl').open('rb') as handle:
    data = load(handle)
assert isinstance(data, dict) and set(data) == {config['training_cell']}
cell = data[config['training_cell']]
assert isinstance(cell, dict) and all(k in cell for k in ['rul', 'dq', 'data'])
frames, capacities, rul = cell['data'], cell['dq'], cell['rul']
assert all(isinstance(x, dict) for x in [frames, capacities, rul])
Path('/work/HEADER363.json').write_text(json.dumps(dict(cell=config['training_cell'],
    top_keys=list(cell), per_field_counts={k: len(cell[k]) for k in ['data', 'dq', 'rul']}), indent=2) + '\n')
keys = sorted(frames, key=int)
assert keys and set(keys) == set(capacities) == set(rul)
required = ['Time (s)', 'Current (mA)', 'Voltage (V)', 'Capacity (mAh)', 'Status']
statistics, statuses, integrals, finite = [], Counter(), [], True
negative_dt, total_rows, extrema = 0, 0, {}
for key in keys:
    frame = frames[key]
    assert isinstance(frame, pd.DataFrame) and all(c in frame for c in required)
    t = frame['Time (s)'].to_numpy(dtype=float)
    current = frame['Current (mA)'].to_numpy(dtype=float)
    assert len(t) > 1
    dt = np.diff(t)
    negative_dt += int(np.sum(dt < 0))
    integral = float(np.sum(np.where(current[1:] < 0, -current[1:] * dt, 0)) / 3_600_000)
    integrals.append(integral)
    statuses.update(str(v) for v in frame['Status'].unique())
    total_rows += len(frame)
    for name in required[:-1]:
        values = frame[name].to_numpy(dtype=float)
        finite = finite and bool(np.isfinite(values).all())
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        old = extrema.get(name, [low, high])
        extrema[name] = [min(low, old[0]), max(high, old[1])]
    statistics.append(dict(cycle=int(key), rows=len(frame), start_time=float(t[0]),
        end_time=float(t[-1]), negative_deltas=int(np.sum(dt < 0)),
        provided_dq=float(capacities[key]), provided_rul=float(rul[key]),
        integrated_discharge_Ah=integral))
q, e, clock = (np.asarray([float(mapping[k]) for k in keys]) for mapping in [capacities, rul, {k: int(k) for k in keys}])
integrals = np.asarray(integrals)
assert np.isfinite(q).all() and np.isfinite(e).all() and finite
relative = {unit: float(np.median(np.abs(q * scale - integrals) / np.maximum(np.abs(integrals), 1e-12)))
            for unit, scale in [('Ah', 1), ('mAh', 0.001)]}
result = dict(schema_passed=True, one_training_cell_only=True, cell=config['training_cell'],
    top_keys=list(cell), frame_columns=list(frames[keys[0]].columns),
    frame_dtypes={str(k): str(v) for k, v in frames[keys[0]].dtypes.items()},
    cycle_count=len(keys), cycle_min=int(keys[0]), cycle_max=int(keys[-1]),
    cycle_keys_contiguous=bool(np.all(np.diff(clock) == 1)), sensor_rows=total_rows,
    numeric_values_finite=finite, negative_time_deltas=negative_dt,
    status_values=dict(statuses), observed_numeric_ranges=extrema,
    capacity_unit_hypothesis_relative_error=relative,
    provided_dq_range=[float(q.min()), float(q.max())],
    rul_plus_cycle_range=[float((e + clock).min()), float((e + clock).max())],
    provided_rul_min=float(e.min()), provided_rul_max=float(e.max()),
    first_last_cycles=statistics[:2] + statistics[-2:],
    nominal_threshold_Ah=0.88, threshold_event_not_yet_admitted=True,
    full_77_cell_semantic_contract_qualified=False,
    pandas_version=pd.__version__, numpy_version=np.__version__,
    new_fits=0, new_predictions=0, new_LLM_calls=0)
Path('/work/SCHEMA363.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['schema_passed', 'cell', 'cycle_count',
    'negative_time_deltas', 'capacity_unit_hypothesis_relative_error', 'rul_plus_cycle_range']}))
