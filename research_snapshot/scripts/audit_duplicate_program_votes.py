"""Frozen cached-prediction screen for repeated lawful voting behavior."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from math import lcm
from pathlib import Path
import time

from audit_program_evidence import read, save, sha, utc


def bound_reader(archive):
    receipt = read(archive / 'ARCHIVE_VERIFIED.json')
    manifest = archive / receipt['manifest_filename']
    assert sha(manifest) == receipt['manifest_sha256']
    index = {x['path']: x for x in read(manifest)['files']}
    def bound(relative):
        path = archive / relative
        assert sha(path) == index[relative]['sha256']
        return read(path)
    return bound


def load_inputs(root):
    cfg = read(root / 'PROTOCOL400.json')
    old = Path(cfg['sources'][0]['archive'])
    new = Path(cfg['sources'][1]['archive'])
    old_read, new_read = bound_reader(old), bound_reader(new)
    pools = []
    for p in old_read('RESULT397.json')['pools']:
        pools.append(dict(batch=p['batch'], cohort=p['cohort'], subject=p['subject'],
                          complete=p['original_native_completed'],
                          registry_deviation=p['strict_registry_deviation'],
                          native_submission=p['original_submission'],
                          original_scope=p['original_scope'], members=p['members']))
    current = new_read('RESULT399.json')
    for p in current['episodes']:
        state = new_read('subjects/' + p['id'] + '/broker/STATE.json')
        members, composed = {}, {}
        for ident, entry in state['programs'].items():
            report = entry['report']
            key = 'common_p00' if ident == 'p00' else p['id'] + '__' + ident
            views = dict(cv=[x for fold in report['cv'] for x in fold['records']],
                         dev=report['development']['records'],
                         final=current['all_program_results'][key]['records'])
            value = {view: {x['group']: [x['actual'], x['predicted']] for x in rows}
                     for view, rows in views.items()}
            (composed if report.get('program_kind') == 'equal_record_vote' else members)[ident] = value
        pools.append(dict(batch='A399', cohort='A347', subject=p['id'], complete=p['completed'],
                          registry_deviation=False, native_submission=state.get('submission'),
                          original_scope='native_choice', members=members, compositions=composed))
    return dict(pools=pools, prior_native_context=old_read('NATIVE_CONTEXT397.json')['rows'])


def plan_weights(lawful):
    groups = defaultdict(list)
    for name, views in sorted(lawful.items()):
        signature = tuple((view, key, value[1]) for view in ['cv', 'dev']
                          for key, value in sorted(views[view].items()))
        groups[signature].append(name)
    classes = list(groups.values())
    return dict(groups=classes, weights={name: str(Fraction(1, len(group)))
                                        for group in classes for name in group})


def vote(members, weights, field, anchor):
    result = {}
    for record in sorted(anchor):
        count = Counter()
        for name, weight in weights.items():
            count[members[name][field][record][1]] += Fraction(weight)
        tied = [label for label, value in count.items() if value == max(count.values())]
        result[record] = anchor[record] if anchor[record] in tied else min(tied)
    return result


def metrics(pred, truth, base, best):
    n = len(truth)
    err = Fraction(sum(pred[k] != truth[k] for k in truth), n)
    harm = Fraction(sum(base[k] == truth[k] and pred[k] != truth[k] for k in truth), n)
    gain = Fraction(sum(base[k] != truth[k] and pred[k] == truth[k] for k in truth), n)
    return dict(predictions=pred, error=str(err), harm=str(harm), gain=str(gain),
                signed_gap_to_best_single=str(err - best))


def run(root):
    freeze = read(root / 'CODE_FREEZE400.json')
    assert sha(Path(__file__)) == freeze['script_sha256']
    assert sha(Path(__file__).with_name('audit_program_evidence.py')) == freeze['shared_helpers_sha256']
    assert not (root / 'STARTED400.json').exists()
    save(root / 'STARTED400.json', dict(utc=utc(), status='running'))
    start = time.monotonic()
    inputs = load_inputs(root)
    assert len(inputs['pools']) == 20
    save(root / 'INPUTS400.json', inputs)
    plans = []
    for p in inputs['pools']:
        legal = {n: {v: m[v] for v in ['cv', 'dev']} for n, m in p['members'].items()}
        plans.append(dict(batch=p['batch'], subject=p['subject'], **plan_weights(legal)))
    # Weights depend only on lawful outputs. Persist before any final scoring.
    save(root / 'SEALED_WEIGHTS400.json', dict(utc=utc(), plans=plans,
                                               uses_final_predictions_or_truth=False))
    rows, native = [], []
    a342 = [p for p in inputs['pools'] if p['batch'] == 'A342']
    for p, plan in zip(inputs['pools'], plans):
        members = p['members']
        truth = {g: x[0] for g, x in members['p00']['final'].items()}
        base = {g: x[1] for g, x in members['p00']['final'].items()}
        best = min(Fraction(sum(v[0] != v[1] for v in m['final'].values()), len(truth))
                   for m in members.values())
        weights = dict(uniform_vote={n: '1' for n in members},
                       equal_behavior_group_vote=plan['weights'])
        scores = {name: metrics(vote(members, w, 'final', base), truth, base, best)
                  for name, w in weights.items()}
        differing_groups = []
        for group in plan['groups']:
            if len(group) > 1:
                differing_groups.append(dict(members=group, final_disagreements=[g for g in truth
                    if len({members[m]['final'][g][1] for m in group}) > 1]))
        rows.append(dict(**{k: p[k] for k in ['batch', 'cohort', 'subject', 'complete', 'registry_deviation']},
                         base_programs=list(members), groups=plan['groups'], weights=weights,
                         duplicate_groups=differing_groups, scores=scores,
                         changed_final_decisions=sum(scores['uniform_vote']['predictions'][g] !=
                             scores['equal_behavior_group_vote']['predictions'][g] for g in truth)))
        actual = []
        if p['batch'] == 'A342':
            pool_id = 'pool_' + str(a342.index(p) + 1)
            actual = [x for x in inputs['prior_native_context']
                      if x['pool'] == pool_id and x['arm'] == 'ordinary' and x['completed']]
        elif p['complete'] and not p['registry_deviation'] and p['subject'].startswith('ordinary'):
            actual = [dict(selector=p['subject'], selected=p['native_submission']['program_id'])]
        for item in actual:
            selected = item['selected']
            chosen = members.get(selected, p.get('compositions', {}).get(selected))
            anchor = {g: x[1] for g, x in chosen['final'].items()}
            native.append(dict(batch=p['batch'], cohort=p['cohort'], subject=p['subject'],
                               native_id=item['selector'], selected=selected,
                               scores=dict(native=metrics(anchor, truth, base, best), **{
                                   name: metrics(vote(members, w, 'final', anchor), truth, base, best)
                                   for name, w in weights.items()})))
    fields = ['error', 'harm', 'gain', 'signed_gap_to_best_single']
    cohorts = {}
    for cohort in sorted({r['cohort'] for r in rows}):
        rr = [r for r in rows if r['cohort'] == cohort]
        cohorts[cohort] = {arm: {key: str(sum((Fraction(r['scores'][arm][key]) for r in rr), Fraction()) / len(rr))
                                         for key in fields} for arm in weights}
    means = {arm: {key: str(sum((Fraction(v[arm][key]) for v in cohorts.values()), Fraction()) / len(cohorts))
                   for key in fields} for arm in weights}
    old, new = means['uniform_vote'], means['equal_behavior_group_vote']
    wins = sum(Fraction(v['equal_behavior_group_vote']['error']) < Fraction(v['uniform_vote']['error'])
               for v in cohorts.values())
    gate = wins >= 2 and Fraction(new['error']) < Fraction(old['error']) and \
        Fraction(new['harm']) <= Fraction(old['harm']) and Fraction(new['gain']) >= Fraction(old['gain'])
    result = dict(utc=utc(), status='complete', rows=rows, native_anchor_comparisons=native,
                  cohorts=cohorts, means=means, strict_cohort_improvements=wins,
                  joint_qualification_passed=gate, scientific_seconds=time.monotonic() - start,
                  weights_sha256=sha(root / 'SEALED_WEIGHTS400.json'), inputs_sha256=sha(root / 'INPUTS400.json'),
                  related_pools=20, training_cohorts=3, physical_rigs=1,
                  new_LLM_calls=0, new_PHM_fits=0, new_PHM_forward_calls=0,
                  final_confirmation_read=False, innovative_method_confirmed=False,
                  goal_complete=False)
    save(root / 'RESULT400.json', result)
    save(root / 'TERMINAL400.json', dict(utc=utc(), status='scored', passed=gate))
    print(dict(means=means, strict_cohort_improvements=wins, joint_qualification_passed=gate,
               changed_decisions=sum(r['changed_final_decisions'] for r in rows)))


def audit(root):
    result = read(root / 'RESULT400.json')
    inputs = load_inputs(root)
    assert inputs == read(root / 'INPUTS400.json')
    checks = 0
    for p, row in zip(inputs['pools'], result['rows']):
        names = list(p['members'])
        # Independent pairwise equality partition, not the primary tuple hashing.
        classes = []
        for name in names:
            found = next((c for c in classes if all(
                p['members'][name][v][g][1] == p['members'][c[0]][v][g][1]
                for v in ['cv', 'dev'] for g in p['members'][name][v])), None)
            if found is None: classes.append([name])
            else: found.append(name)
        assert {frozenset(c) for c in classes} == {frozenset(c) for c in row['groups']}
        multiplier = lcm(*(len(c) for c in classes))
        unit = {n: multiplier // len(c) for c in classes for n in c}
        for arm, ws in [('uniform_vote', {n: 1 for n in names}), ('equal_behavior_group_vote', unit)]:
            assert all(Fraction(row['weights'][arm][n]) == Fraction(ws[n], multiplier if arm != 'uniform_vote' else 1)
                       for n in names)
            counts = [0, 0, 0]
            for g, (truth, old) in p['members']['p00']['final'].items():
                tallies = [sum(ws[n] for n in names if p['members'][n]['final'][g][1] == label)
                           for label in range(4)]
                tied = [k for k, v in enumerate(tallies) if v == max(tallies)]
                chosen = old if old in tied else min(tied)
                assert chosen == row['scores'][arm]['predictions'][g]
                counts[0] += chosen != truth
                counts[1] += old == truth and chosen != truth
                counts[2] += old != truth and chosen == truth
                checks += 1
            n = len(p['members']['p00']['final'])
            for key, value in zip(['error', 'harm', 'gain'], counts):
                assert Fraction(row['scores'][arm][key]) == Fraction(value, n)
                checks += 1
    # Independently check the aggregation and declared advancement decision.
    means = {}
    for arm in ['uniform_vote', 'equal_behavior_group_vote']:
        means[arm] = {}
        for key in ['error', 'harm', 'gain', 'signed_gap_to_best_single']:
            values = []
            for cohort in result['cohorts']:
                subset = [r for r in result['rows'] if r['cohort'] == cohort]
                value = sum((Fraction(r['scores'][arm][key]) for r in subset), Fraction()) / len(subset)
                assert value == Fraction(result['cohorts'][cohort][arm][key])
                values.append(value)
                checks += 1
            means[arm][key] = sum(values, Fraction()) / len(values)
            assert means[arm][key] == Fraction(result['means'][arm][key])
            checks += 1
    wins = sum(Fraction(v['equal_behavior_group_vote']['error']) < Fraction(v['uniform_vote']['error'])
               for v in result['cohorts'].values())
    a, b = means['uniform_vote'], means['equal_behavior_group_vote']
    expected = wins >= 2 and b['error'] < a['error'] and b['harm'] <= a['harm'] and b['gain'] >= a['gain']
    assert expected == result['joint_qualification_passed']
    checks += 1
    for row in result['native_anchor_comparisons']:
        p = next(p for p in inputs['pools'] if (p['batch'], p['subject']) == (row['batch'], row['subject']))
        members = p['members']
        selected = members.get(row['selected'], p.get('compositions', {}).get(row['selected']))
        base = members['p00']['final']
        main = next(r for r in result['rows'] if (r['batch'], r['subject']) == (row['batch'], row['subject']))
        common = lcm(*(len(c) for c in main['groups']))
        group_weights = {n: common // len(c) for c in main['groups'] for n in c}
        for arm, ws in [('native', None), ('uniform_vote', {n: 1 for n in members}), ('equal_behavior_group_vote', group_weights)]:
            counts = [0, 0, 0]
            for g, (truth, old) in base.items():
                anchor = selected['final'][g][1]
                if ws is None:
                    chosen = anchor
                else:
                    tally = [sum(w for n, w in ws.items() if members[n]['final'][g][1] == label) for label in range(4)]
                    tied = [label for label in range(4) if tally[label] == max(tally)]
                    chosen = anchor if anchor in tied else min(tied)
                assert chosen == row['scores'][arm]['predictions'][g]
                counts[0] += chosen != truth
                counts[1] += old == truth and chosen != truth
                counts[2] += old != truth and chosen == truth
                checks += 1
            for key, value in zip(['error', 'harm', 'gain'], counts):
                assert Fraction(row['scores'][arm][key]) == Fraction(value, len(base))
                checks += 1
    output = dict(utc=utc(), passed=True, checks=checks, independent_method='Pairwise legal-output partition and integer weighted tallies',
                  result_sha256=sha(root / 'RESULT400.json'), source_inputs_rebound=True,
                  new_LLM_calls=0, new_fits=0, new_predictions=0)
    save(root / 'INDEPENDENT_AUDIT400.json', output)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'audit'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    (run if args.action == 'run' else audit)(args.run)
