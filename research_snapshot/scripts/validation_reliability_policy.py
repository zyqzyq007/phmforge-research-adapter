"""Outcome-blind selection and proposal checks using nested legal evidence.

This is a concrete mechanism prototype, not a claim that nested CV is novel.
No function accepts final observations, labels, scores or conditions.
"""
import numpy as np

EPS = 1e-12
VIEWS = ['condition_cv', 'record_cv']


def best(loss, incumbent):
    values = np.asarray(loss, dtype=float)
    assert np.isfinite(values).all()
    tied = np.flatnonzero(values <= values.min() + EPS).tolist()
    return incumbent if incumbent in tied else tied[0]


def inspect(case, incumbent):
    ids = case['candidate_ids']
    old = ids.index(incumbent)
    assert len(ids) == len(set(ids))
    mean_risk = {name:np.mean(case['views'][name]['group_losses'], axis=1)
                 for name in VIEWS}
    assert all(v.shape == (len(ids),) and np.isfinite(v).all() for v in mean_risk.values())
    regrets = {name:[] for name in VIEWS}
    for trial in case['calibration']['trials']:
        actual = np.asarray(trial['observed_losses'], dtype=float)
        assert actual.shape == (len(ids),) and np.isfinite(actual).all()
        for view in VIEWS:
            predicted = np.mean(trial['inner_views'][view]['group_losses'], axis=1)
            choice = best(predicted, old)
            regrets[view].append(float(actual[choice]-actual.min()))
    assert all(regrets[v] for v in VIEWS)
    summary = {name:float(np.mean(value)) for name,value in regrets.items()}
    view = min(VIEWS, key=lambda name:(summary[name], VIEWS.index(name)))
    selected = best(mean_risk[view], old)
    return dict(view=view, calibration_regret=summary,
                risk={name:value.tolist() for name,value in mean_risk.items()},
                reranked=ids[selected], incumbent=incumbent)


def gate(case, incumbent, proposed, view='calibrated'):
    evidence = inspect(case, incumbent)
    selected_view = evidence['view'] if view == 'calibrated' else view
    assert selected_view in VIEWS
    ids = case['candidate_ids']; old = ids.index(incumbent); new = ids.index(proposed)
    risk = evidence['risk'][selected_view]
    accept = new == old or risk[new] < risk[old]-EPS
    return proposed if accept else incumbent, dict(evidence, proposed=proposed,
        used_view=selected_view, accepted=accept, risk_difference=risk[new]-risk[old])
