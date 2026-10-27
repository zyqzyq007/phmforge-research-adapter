"""Ten frozen, outcome-blind PHM program-update policies.

Inputs are group losses from genuinely fitted complete candidate programs.
These are executable selection layers, not ten claimed new Agent architectures.
No function accepts final outcomes. Statistical screens are descriptive under
dependent source groups; none certifies unseen-condition safety.
"""
from math import ceil, comb

import numpy as np


METHODS = {
    'M01_group_mean': '全部验证组等权平均损失',
    'M02_worst_group': '最差验证组损失最小化',
    'M03_tail_risk': '最差三分之一验证组平均损失',
    'M04_dispersion': '平均损失加组间标准差',
    'M05_group_regret': '最小化验证组内最大选择损失',
    'M06_paired_bootstrap': '配对分组重采样上界筛选',
    'M07_paired_sign': '精确分组符号检验筛选',
    'M08_delete_group': '删去任一验证组后仍稳定获胜',
    'M09_pareto_update': '每个验证组均不退步才替换',
    'M10_broad_support': '至少半数组改善且开发集不退步',
}
EPS = 1e-12


def validate(case):
    ids = case['candidate_ids']
    risk = np.asarray(case['group_losses'], dtype=float)
    assert len(ids) == len(set(ids)) >= 2
    assert risk.shape == (len(ids), len(case['groups']))
    assert risk.shape[1] >= 3 and np.isfinite(risk).all() and (risk >= 0).all()
    assert set(case['development_columns']) < set(range(risk.shape[1]))
    return ids, risk


def choose(case, incumbent, method):
    ids, r = validate(case)
    old = ids.index(incumbent)
    n, g = r.shape
    mu = r.mean(axis=1)
    # Every ranking ties in favor of retaining the supplied incumbent, then the
    # original (pre-final) candidate registration order. No test-derived order.
    def best(scores, allowed=None):
        allowed = list(range(n)) if allowed is None else list(allowed)
        low = min(float(scores[j]) for j in allowed)
        tied = [j for j in allowed if float(scores[j]) <= low + EPS]
        return old if old in tied else min(tied)

    evidence = {'method': method, 'incumbent': incumbent, 'group_count': g}
    if method == 'keep':
        selected = old
    elif method in ('development', 'cv'):
        columns = case['development_columns'] if method == 'development' else [j for j in range(g) if j not in case['development_columns']]
        selected = best(r[:, columns].mean(axis=1))
    elif method == 'M01_group_mean':
        selected = best(mu)
    elif method == 'M02_worst_group':
        selected = best(r.max(axis=1))
    elif method == 'M03_tail_risk':
        tail = ceil(g / 3)
        selected = best(np.sort(r, axis=1)[:, -tail:].mean(axis=1))
        evidence['tail_groups'] = tail
    elif method == 'M04_dispersion':
        selected = best(mu + r.std(axis=1, ddof=1))
    elif method == 'M05_group_regret':
        selected = best((r - r.min(axis=0, keepdims=True)).max(axis=1))
    elif method in ('M06_paired_bootstrap', 'M07_paired_sign'):
        eligible = [old]
        alpha = .05 / (n - 1)
        evidence.update(alpha_per_comparison=alpha, candidate_checks={})
        indices = np.random.default_rng(328).integers(0, g, size=(4000, g))
        for j in range(n):
            if j == old:
                continue
            delta = r[j] - r[old]
            if method == 'M06_paired_bootstrap':
                upper = float(np.quantile(delta[indices].mean(axis=1), 1 - alpha, method='higher'))
                accept = upper < -EPS
                evidence['candidate_checks'][ids[j]] = {'upper': upper, 'accepted': accept}
            else:
                win, loss = int((delta < -EPS).sum()), int((delta > EPS).sum())
                effective = win + loss
                probability = sum(comb(effective, k) for k in range(win, effective + 1)) / (2 ** effective) if effective else 1.
                accept = probability <= alpha and mu[j] < mu[old] - EPS
                evidence['candidate_checks'][ids[j]] = {'wins': win, 'losses': loss, 'p_value': probability, 'accepted': accept}
            if accept:
                eligible.append(j)
        selected = best(mu, eligible)
        evidence['eligible'] = [ids[j] for j in eligible]
    elif method == 'M08_delete_group':
        winners = [best(np.delete(r, col, axis=1).mean(axis=1)) for col in range(g)]
        selected = winners[0] if len(set(winners)) == 1 and mu[winners[0]] < mu[old] - EPS else old
        evidence['leave_one_group_winners'] = [ids[j] for j in winners]
    elif method == 'M09_pareto_update':
        eligible = [j for j in range(n) if np.all(r[j] <= r[old] + EPS) and mu[j] < mu[old] - EPS] + [old]
        selected = best(mu, eligible)
        evidence['eligible'] = [ids[j] for j in eligible]
    elif method == 'M10_broad_support':
        minimum = max(2, ceil(g / 2))
        dev = r[:, case['development_columns']].mean(axis=1)
        eligible = [j for j in range(n) if int((r[j] < r[old] - EPS).sum()) >= minimum and dev[j] <= dev[old] + EPS and mu[j] < mu[old] - EPS] + [old]
        selected = best(mu, eligible)
        evidence.update(minimum_improving_groups=minimum, eligible=[ids[j] for j in eligible])
    else:
        raise ValueError(method)
    evidence['selected'] = ids[selected]
    return ids[selected], evidence


def run_all(cases):
    return {method: [{'case': case['id'], 'incumbent': old,
                      'selected': choose(case, old, method)[0],
                      'evidence': choose(case, old, method)[1]}
                     for case in cases for old in case['candidate_ids']]
            for method in ['keep', 'development', 'cv', *METHODS]}
