"""Private source conversion with author-compatible, causal charge curves."""
import json
from pathlib import Path
import resource

import numpy as np
from pandas.compat.pickle_compat import load
from scipy.interpolate import interp1d

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
config = json.loads(Path('/config.json').read_text())
with Path('/cell.pkl').open('rb') as handle:
    raw = load(handle)
assert set(raw) == {config['cell']}
cell = raw[config['cell']]
keys = sorted(cell['data'], key=int)
assert [int(k) for k in keys] == list(range(1, len(keys) + 1)), 'Cycle key continuity failed'
assert set(keys) == set(cell['dq']) == set(cell['rul']), 'Source fields use different cycles'
curve, cycles, capacities, labels = [], [], [], []
sensor_rows, max_native_delta = 0, 0.0
for key in keys:
    frame, cycle = cell['data'][key], int(key)
    assert float(cell['rul'][key]) == len(keys) - cycle + 1, 'Source RUL countdown differs'
    sensor_rows += len(frame)
    # The original loader begins at cycle10. The first9remain in raw source.
    if cycle < 10:
        continue
    values = frame[['Time (s)', 'Current (mA)', 'Voltage (V)', 'Capacity (mAh)']].to_numpy(float)
    assert np.isfinite(values).all(), 'Nonfinite eligible measurements'
    assert np.all(np.diff(values[:, 0]) >= 0), 'Within-cycle clock decreases'
    statuses = frame['Status'].astype(str)
    native = sum(float(np.ptp(frame.loc[statuses == name, 'Capacity (mAh)'].to_numpy(float)))
                 for name in sorted(set(statuses[statuses.str.contains('discharge')])))
    delta = abs(native - float(cell['dq'][key]))
    assert delta <= 1e-8, 'Native capacity reconstruction differs'
    max_native_delta = max(max_native_delta, delta)
    charge = frame.loc[~statuses.str.contains('discharge')]
    current = charge['Current (mA)'].to_numpy(float)
    assert np.any(current < 5000) and np.any(current < 1090), 'Missing author charge segment'
    left = int((current < 5000).argmax()) + 1
    right = int((current < 1090).argmax()) - 2
    segment = charge.iloc[left:right]
    t = segment['Time (s)'].to_numpy(float)
    assert len(t) >= 2 and np.all(np.diff(t) > 0), 'Ambiguous partial-charge time'
    grid = np.linspace(t[0], t[-1], 100)
    points = np.stack([interp1d(t, segment[name].to_numpy(float), kind='linear')(grid)
                       for name in ['Voltage (V)', 'Capacity (mAh)']], axis=-1)
    assert np.isfinite(points).all()
    curve.append(points)
    cycles.append(cycle)
    capacities.append(float(cell['dq'][key]))
    labels.append(float(cell['rul'][key]))
curve = np.asarray(curve, dtype=np.float32)
features = np.concatenate([curve, curve - curve[:1]], axis=2)
assert features.shape == (len(keys) - 9, 100, 4) and np.isfinite(features).all()
np.savez_compressed('/work/PRIVATE_CELL.npz', curves=features, cycle=np.asarray(cycles),
    dq_mAh=np.asarray(capacities), rul=np.asarray(labels),
    discharge_rates=np.asarray(config['rates'], dtype=float))
audit = dict(passed=True, role=config['role'], cell=config['cell'],
    raw_cycles=len(keys), sensor_rows=sensor_rows, curve_cycles=len(cycles),
    source_RUL_preserved=True, native_capacity_reconstructed=True,
    max_native_capacity_error_mAh=max_native_delta,
    current_features_use_only_current_partial_charge=True,
    delta_reference_cycle=10, no_test_derived_normalization=True,
    final_data_private=config['role'] == 'final', new_fits=0, new_predictions=0, new_LLM_calls=0)
Path('/work/PRIVATE_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(dict(passed=True, role=config['role'], new_fits=0, new_predictions=0)))
