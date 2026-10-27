"""Bound what nonnegative validation-score reweighting can select.

This analyzes archived real predictions, not new PHM models. Exact rational
primal witnesses and dual upper bounds certify numerical LP conclusions.
Unknown cases remain reachable in the optimistic ceiling.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from datetime import datetime, timezone

# Exact minimization/ties of the stored binary64 scores. This defines the
# audited selector family; it is not a claim about tolerance-based selectors.
EPS = F(0)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_choice(losses, weights, priority):
    scores = [sum(F(float(v)) * w for v, w in zip(row, weights)) for row in losses]
    low = min(scores)
    return next(i for i in priority if scores[i] <= low + EPS)


def dual_bound(losses, candidate, others, coefficients, earlier=None):
    # For any simplex w satisfying all later-competitor inequalities,
    # delta <= (q.b + max_g sum_j q_j*(L_jg-L_ig))/sum_earlier(q).
    q = [F(v) for v in coefficients]
    assert min(q) >= 0
    early = set(others) if earlier is None else set(earlier)
    denominator = sum(qj for j, qj in zip(others, q) if j in early)
    assert denominator > 0
    columns = [sum(qj * (F(float(losses[j][g])) - F(float(losses[candidate][g])))
                   for j, qj in zip(others, q)) for g in range(len(losses[0]))]
    constant = sum(qj * EPS for j, qj in zip(others, q) if earlier is not None and j not in early)
    return (max(columns) + constant) / denominator


def solve_case(case):
    import numpy as np
    from scipy.optimize import linprog
    begin = time.monotonic()
    losses = np.asarray(case['group_losses'], dtype=float)
    n, groups = losses.shape
    priority = [case['candidate_ids'].index(case['original_incumbent'])]
    priority += [i for i in range(n) if i not in priority]
    scale = max(1., float(np.max(np.abs(losses))))
    output, solves = [], 0
    for i, cid in enumerate(case['candidate_ids']):
        others = [j for j in range(n) if j != i]
        earlier = priority[:priority.index(i)]
        row = dict(candidate=cid, candidate_index=i, status='unresolved_include_in_ceiling', attempts=[])
        for mode in ['weak', 'priority']:
            if mode == 'priority' and not earlier:
                continue
            coeff = np.ones(len(others)) if mode == 'weak' else np.array([j in earlier for j in others], dtype=float)
            a = np.column_stack(((losses[i] - losses[others]) / scale, coeff))
            b = np.zeros(len(others)) if mode == 'weak' else np.array([0. if j in earlier else float(EPS)/scale for j in others])
            result = linprog(np.r_[np.zeros(groups), -1.], A_ub=a, b_ub=b,
                             A_eq=np.array([[*np.ones(groups), 0.]]), b_eq=[1.],
                             bounds=[(0., None)]*groups + [(None, None)], method='highs',
                             options={'dual_feasibility_tolerance': 1e-9,
                                      'primal_feasibility_tolerance': 1e-9, 'threads': 1})
            solves += 1
            attempt = dict(mode=mode, status=int(result.status), message=result.message)
            row['attempts'].append(attempt)
            if not result.success:
                continue
            weights = [F(float(max(0., v))) for v in result.x[:groups]]
            total = sum(weights)
            assert total > 0
            weights = [v / total for v in weights]
            selected = exact_choice(losses, weights, priority)
            attempt.update(margin=float(result.x[-1]*scale), witness_selected=case['candidate_ids'][selected])
            if selected == i:
                row.update(status='certified_reachable', weights=[str(v) for v in weights])
                break
            q = [str(F(float(max(0., -v)))) for v in result.ineqlin.marginals]
            den = sum(F(v) for j, v in zip(others, q) if mode == 'weak' or j in earlier)
            if den == 0:
                continue
            upper = dual_bound(losses, i, others, q, None if mode == 'weak' else earlier)
            attempt['exact_upper_bound'] = str(upper)
            excluded = upper < -EPS if mode == 'weak' else upper <= EPS
            if excluded:
                row.update(status='certified_unreachable', certificate=dict(mode=mode,
                           others=others, coefficients=q, earlier=earlier,
                           exact_upper_bound=str(upper)))
                break
        output.append(row)
    return dict(case=case['id'], priority=priority, candidates=output,
                numerical_LP_solves=solves, seconds=time.monotonic()-begin)


def solve(run):
    assert not (run/'REACHABILITY441.json').exists(), 'No implicit rerun'
    config = read(run/'PROTOCOL441.json')
    assert sha(run/'public_cases.json') == config['public_sha256']
    cases = read(run/'public_cases.json')
    with ProcessPoolExecutor(max_workers=config['parallel']) as pool:
        values = list(pool.map(solve_case, cases))
    save(run/'REACHABILITY441.json', values)
    save(run/'EVIDENCE_SEALED441.json', dict(utc=utc(),
         public_sha256=sha(run/'public_cases.json'), evidence_sha256=sha(run/'REACHABILITY441.json'),
         source_code_sha256=sha(__file__), final_scores_read_during_solve=False,
         numerical_LP_solves=sum(v['numerical_LP_solves'] for v in values)))


def audit_and_score(run):
    import numpy as np
    start = time.monotonic()
    cfg = read(run/'PROTOCOL441.json')
    seal = read(run/'EVIDENCE_SEALED441.json')
    assert sha(run/'REACHABILITY441.json') == seal['evidence_sha256']
    assert sha(run/'public_cases.json') == cfg['public_sha256'] == seal['public_sha256']
    assert sha(run/'private_scores.json') == cfg['private_sha256']
    cases = read(run/'public_cases.json')
    evidence = {v['case']: v for v in read(run/'REACHABILITY441.json')}
    gold = read(run/'private_scores.json')
    rows, checks = [], 0
    for case in cases:
        ev = evidence[case['id']]
        loss = case['group_losses']
        allowed = []
        status_counts = {}
        for row in ev['candidates']:
            i = row['candidate_index']
            assert row['candidate'] == case['candidate_ids'][i]
            if row['status'] == 'certified_reachable':
                w = [F(x) for x in row['weights']]
                assert min(w) >= 0 and sum(w) == 1
                # Independent scalar rational comparison, without scipy/NumPy.
                scores = [sum(F(float(x))*weight for x, weight in zip(lr, w)) for lr in loss]
                tied = [j for j in ev['priority'] if scores[j] - min(scores) <= EPS]
                assert tied[0] == i
                checks += len(scores)*len(w) + 3
            elif row['status'] == 'certified_unreachable':
                cert = row['certificate']
                q = [F(v) for v in cert['coefficients']]
                early = set(cert['others']) if cert['mode'] == 'weak' else set(cert['earlier'])
                denominator = sum(v for j,v in zip(cert['others'],q) if j in early)
                assert min(q) >= 0 and denominator > 0
                constant = sum(v*EPS for j,v in zip(cert['others'],q)
                               if cert['mode'] != 'weak' and j not in early)
                upper_columns = []
                for g in range(len(loss[0])):
                    val = constant
                    for j,v in zip(cert['others'],q):
                        val += v*(F(float(loss[j][g]))-F(float(loss[i][g])))
                    upper_columns.append(val/denominator)
                bound = max(upper_columns)
                assert bound == F(cert['exact_upper_bound'])
                assert bound < -EPS if cert['mode'] == 'weak' else bound <= EPS
                checks += len(q)*len(loss[0]) + 4
            else:
                assert row['status'] == 'unresolved_include_in_ceiling'
            if row['status'] != 'certified_unreachable':
                allowed.append(i)
            status_counts[row['status']] = status_counts.get(row['status'],0)+1
        assert allowed
        values = np.asarray(gold[case['id']]['losses'])
        low, high = float(values.min()), float(values.max())
        denominator = high-low
        norm = lambda x: (float(x)-low)/denominator if denominator > 1e-12 else 0.
        ordinary = gold[case['id']]['ordinary']
        assert set(ordinary) == {'ordinary_1','ordinary_2'}
        ordinary_loss = float(np.mean([values[case['candidate_ids'].index(cid)] for cid in ordinary.values()]))
        ceiling = float(values[allowed].min())
        matrix = np.asarray(loss)
        incumbent = case['candidate_ids'].index(case['original_incumbent'])
        # Ordinary mean weighting is a known feasible rule; it must remain included.
        selected = exact_choice(loss, [F(1,len(loss[0]))]*len(loss[0]), ev['priority'])
        assert selected in allowed
        best = [case['candidate_ids'][i] for i in allowed if abs(values[i]-ceiling) <= 1e-12]
        rows.append(dict(case=case['id'], source=cfg['cluster_map'][case['source']],
              candidates=len(values), groups=len(loss[0]), status_counts=status_counts,
              reachable_or_unresolved=[case['candidate_ids'][i] for i in allowed],
              unreachable=[r['candidate'] for r in ev['candidates'] if r['status']=='certified_unreachable'],
              optimistic_best_candidates=best, unrestricted_pool_best_loss=low,
              optimistic_reweighting_loss_floor=ceiling, ordinary_mean_loss=ordinary_loss,
              ordinary_choices=ordinary, ordinary_normalized_regret=norm(ordinary_loss),
              optimistic_normalized_regret_floor=norm(ceiling),
              group_mean_choice=case['candidate_ids'][selected],
              group_mean_normalized_regret=norm(values[selected]),
              original_incumbent_loss=float(values[incumbent]), unit=case['metric']))
        checks += len(values)+3
    sources = {}
    for source in sorted({r['source'] for r in rows}):
        ss = [r for r in rows if r['source']==source]
        sources[source] = dict(pools=len(ss), **{k:float(np.mean([r[k] for r in ss])) for k in
             ['ordinary_normalized_regret','optimistic_normalized_regret_floor','group_mean_normalized_regret']})
    possible = sum(v['optimistic_normalized_regret_floor'] < v['ordinary_normalized_regret']-1e-12 for v in sources.values())
    result = dict(utc=utc(), rows=rows, sources=sources,
         source_count=len(sources), pools=len(rows), candidate_memberships=sum(r['candidates'] for r in rows),
         optimistic_possible_strict_source_wins=possible,
         majority_possible_with_omniscient_weight_choice=possible>len(sources)/2,
         panels_with_positive_reweighting_regret_floor=sum(r['optimistic_normalized_regret_floor']>1e-12 for r in rows),
         independent_rational_checks=checks, audit_seconds=time.monotonic()-start,
         new_PHM_fits=0,new_LLM=0,new_model_forwards=0,new_independent_confirmation=0,
         caveat='Oracle ceiling only, not an implementable selector or Agent win. All outcomes previously opened. Only nonnegative weighted group loss with frozen incumbent/order ties; no exclusion of other methods.')
    save(run/'RESULT441.json', result)
    save(run/'AUDIT441.json', dict(passed=True, checks=checks,
         primary_solver='scipy.optimize.linprog/HiGHS', verifier='independent scalar Fraction certificates',
         evidence_sha256=sha(run/'REACHABILITY441.json'), result_sha256=sha(run/'RESULT441.json')))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['solve','score'])
    parser.add_argument('--run', type=Path, required=True)
    args=parser.parse_args(); start=time.monotonic()
    terminal=args.run/(args.mode.upper()+'_TERMINAL441.json')
    assert not terminal.exists(), 'Preserve terminated stages; do not restart implicitly'
    save(args.run/(args.mode.upper()+'_HANDLE441.json'),dict(pid=os.getpid(),utc=utc()))
    try:
        if args.mode=='solve':
            solve(args.run)
        else:
            result=audit_and_score(args.run)
            print(json.dumps({k:result[k] for k in ['source_count','pools','candidate_memberships',
                  'optimistic_possible_strict_source_wins','majority_possible_with_omniscient_weight_choice',
                  'panels_with_positive_reweighting_regret_floor','independent_rational_checks','sources']},ensure_ascii=False))
        save(terminal,dict(status='completed',utc=utc(),seconds=time.monotonic()-start))
    except BaseException:
        save(terminal,dict(status='failed',utc=utc(),seconds=time.monotonic()-start,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
