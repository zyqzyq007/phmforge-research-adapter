"""Exact *past* MAE-difference range when one shared event time is censored.

The result does not identify current/future forecast risk. This elementary
piecewise-linear calculation is not a new survival estimator.
"""
import math


def past_difference_interval(incumbent_lifetimes, candidate_lifetimes, observed_until):
    a, b = list(incumbent_lifetimes), list(candidate_lifetimes)
    assert len(a) == len(b) and len(a) > 0
    assert all(math.isfinite(v) for v in [*a, *b, observed_until])
    points = sorted({observed_until, *(v for v in [*a, *b] if v >= observed_until)})
    differences = [math.fsum(abs(t - y) - abs(t - x) for x, y in zip(a, b)) / len(a)
                   for t in points]
    return min(differences), max(differences)


def cv_after_past_dominance(lifetimes, observed_until, cv):
    names = list(cv)
    intervals, dominated = {}, set()
    for incumbent in names:
        for challenger in names:
            if incumbent == challenger:
                continue
            bounds = past_difference_interval(lifetimes[incumbent], lifetimes[challenger], observed_until)
            intervals[incumbent + "->" + challenger] = bounds
            if bounds[1] < -1e-6:
                dominated.add(incumbent)
    admissible = [name for name in names if name not in dominated]
    assert admissible, "Strict common-outcome dominance cannot eliminate all models"
    selected = min(admissible, key=lambda name: (cv[name], names.index(name)))
    return selected, admissible, intervals
