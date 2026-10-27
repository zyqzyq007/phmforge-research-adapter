"""Query-local risk comparison using historical cross-fit residuals."""
import numpy as np


def choose(names, risk, incumbent):
    best = min(risk)
    eligible = [i for i, value in enumerate(risk) if value <= best + 1e-9]
    return min(eligible, key=lambda i: (names[i] != incumbent, names[i]))


def select(query, units, full, folds, names, mean, std, elapsed_column,
           incumbent, global_cv):
    """No target labels, other queries, mutable state, or target predictions cached elsewhere."""
    time = float(query[elapsed_column])
    signed = lambda values: np.sign(values) * np.log1p(np.abs(values))
    normalized = (signed(query) - mean) / std
    decisions, evidence = {}, {}
    for scope in ['feature_all', 'feature_at_risk']:
        selected = [u for u in units if scope == 'feature_all'
                    or u['elapsed'][0] <= time <= u['end']]
        if not selected:
            name = names[choose(names, global_cv, incumbent)]
            decisions['original_' + scope] = decisions['transport_' + scope] = name
            evidence[scope] = dict(units=[], fallback='global_CV')
            continue
        old, corrected, rows = [], [], []
        for unit in selected:
            distance = (signed(unit['features']) - mean) / std - normalized
            index = int(np.argmin(np.einsum('ij,ij->i', distance, distance)))
            residual = unit['residual'][index]
            difference = full - folds[unit['bank']]
            old.append(abs(residual))
            corrected.append(abs(difference - residual))
            rows.append(dict(unit=unit['id'], elapsed=float(unit['elapsed'][index]),
                             bank=unit['bank'], signed_residual=residual.tolist(),
                             full_minus_fold=difference.tolist()))
        original, transport = np.mean(old, axis=0), np.mean(corrected, axis=0)
        decisions['original_' + scope] = names[choose(names, original, incumbent)]
        decisions['transport_' + scope] = names[choose(names, transport, incumbent)]
        evidence[scope] = dict(units=[u['id'] for u in selected], rows=rows,
                               original_risk=original.tolist(), transport_risk=transport.tolist())
    return dict(current_elapsed=time, choices=decisions, evidence=evidence)
