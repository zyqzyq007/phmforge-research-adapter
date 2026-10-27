"""Causal input construction, shared by training and observation-only inference."""
import numpy as np


def inputs(window, past_capacity_mAh, completed_cycles, rates):
    window = np.asarray(window, dtype=np.float32)
    assert window.ndim == 4 and window.shape[1:] == (10, 100, 4)
    assert np.asarray(past_capacity_mAh).shape == (len(window), 9)
    normalized = window.copy()
    normalized[..., 0] = (normalized[..., 0] - 3.2) / 0.4
    normalized[..., 2] /= 0.4
    normalized[..., [1, 3]] /= 1200.0
    stats = np.stack([normalized.mean(axis=2), normalized.std(axis=2),
                      normalized[:, :, 0], normalized[:, :, -1]], axis=-1).reshape(len(window), -1)
    age = np.asarray(completed_cycles, dtype=np.float32).reshape(-1, 1) / 1000.0
    protocol = np.asarray(rates, dtype=np.float32).reshape(-1, 3) / 5.0
    classic = np.concatenate([stats, np.asarray(past_capacity_mAh, dtype=np.float32) / 1000.0,
                              age, protocol], axis=1)
    neural = np.ascontiguousarray(normalized.transpose(0, 3, 2, 1))
    assert classic.shape[1] == 173 and neural.shape[1:] == (4, 100, 10)
    assert np.isfinite(classic).all() and np.isfinite(neural).all()
    return classic, neural


def cell_examples(data, query_cycles):
    cycle = data['cycle']
    ends = np.asarray([np.searchsorted(cycle, k) for k in query_cycles if k <= cycle[-1]], dtype=int)
    assert np.all(ends >= 29)
    indices = ends[:, None] + np.arange(-27, 1, 3)[None, :]
    assert np.all(cycle[indices[:, :-1]] < cycle[ends, None])
    window, targets = data['curves'][indices], data['dq_mAh'][indices]
    age = cycle[ends] - 1
    rates = np.tile(data['discharge_rates'], (len(ends), 1))
    classic, neural = inputs(window, targets[:, :-1], age, rates)
    return dict(x=classic, nn=neural, age=age.astype(float), y=data['rul'][ends],
                cap=targets.astype(np.float32), cycle=cycle[ends])
