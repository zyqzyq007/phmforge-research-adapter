"""Basic saturation ablation, not a claimed innovative or validated method.

Only the denominator in M10's coverage requirement changes. Final scores,
deployment data and labels are never arguments. This is intended to test
whether restoring feasible updates also restores harmful replacements.
"""
from math import ceil

from program_selection_policies import EPS, validate


def choose_opportunity(case, incumbent):
    ids, risk = validate(case)
    old = ids.index(incumbent)
    mean = risk.mean(axis=1)
    dev = risk[:, case['development_columns']].mean(axis=1)
    opportunity_groups = int((risk[old] > EPS).sum())
    minimum = max(2, ceil(opportunity_groups / 2))
    eligible = [old] + [j for j in range(len(ids))
        if int((risk[j] < risk[old] - EPS).sum()) >= minimum
        and dev[j] <= dev[old] + EPS and mean[j] < mean[old] - EPS]
    low = min(float(mean[j]) for j in eligible)
    tied = [j for j in eligible if mean[j] <= low + EPS]
    selected = old if old in tied else min(tied)
    return ids[selected], dict(opportunity_groups=opportunity_groups,
        required_strict_improvements=minimum,
        eligible=[ids[j] for j in eligible], selected=ids[selected])
