"""Retrospective, cached pointwise-evidence audit; no new method or model execution."""
import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
import time

from audit_duplicate_program_votes import bound_reader
from audit_program_evidence import read, save, sha, utc


def error(rows):
    return Fraction(sum(y != p for y, p in rows.values()), len(rows))


def order(names):
    return sorted(names, key=lambda n: (n != 'p00', n))


def load_inputs(root):
    cfg = read(root / 'PROTOCOL405.json')
    old = bound_reader(Path(cfg['sources'][0]))('INPUTS403.json')
    newread = bound_reader(Path(cfg['sources'][1])); new = newread('RESULT404.json')
    pools = old['pools']
    for episode in new['episodes']:
        state = newread('subjects/' + episode['id'] + '/broker/STATE.json')
        members = {}
        for name, entry in state['programs'].items():
            report = entry['report']; key = 'common_p00' if name == 'p00' else episode['id'] + '__' + name
            views = dict(cv=[x for fold in report['cv'] for x in fold['records']],
                         dev=report['development']['records'],
                         legal=report['development']['records'] + state['probe']['reports'][name]['records'],
                         final=new['all_program_results'][key]['records'])
            members[name] = {v: {x['group']: [x['actual'], x['predicted']] for x in rows} for v, rows in views.items()}
        pools.append(dict(batch='A404', cohort='A347', subject=episode['id'], complete=episode['completed'],
            registry_deviation=False, native_submission=state['submission'], original_scope='native_choice',
            members=members, check_coverage=dict(complete=True, missing=[], source='broker_probe')))
    return dict(pools=pools, native_context=old['native_context'])


def plan(lawful):
    """Only legal errors enter pair relations or simple-control selections."""
    names = order(lawful); pairs = []
    for a in names:
        for b in names:
            if a == b:
                continue
            gains, harms = {}, {}
            same = True
            for view in ['cv', 'legal']:
                aa, bb = lawful[a][view], lawful[b][view]
                assert set(aa) == set(bb)
                assert all(aa[g][0] == bb[g][0] for g in aa)
                gains[view] = [g for g in aa if aa[g][0] != aa[g][1] and bb[g][0] == bb[g][1]]
                harms[view] = [g for g in aa if aa[g][0] == aa[g][1] and bb[g][0] != bb[g][1]]
                same = same and all(aa[g][1] == bb[g][1] for g in aa)
            g, h = sum(map(len, gains.values())), sum(map(len, harms.values()))
            relation = ('new_dominates' if g and not h else 'old_dominates' if h and not g
                        else 'equal_correctness' if not g and not h else 'tradeoff')
            pairs.append(dict(old=a, new=b, relation=relation, gains=gains, harms=harms,
                              exact_legal_predictions_equal=same))
    scores = {n: dict(development=error(v['dev']), mean_cv=error(v['cv']),
                     equal_CV_development=(error(v['cv']) + error(v['dev'])) / 2,
                     full_checkpoint_validation=error(v['legal'])) for n, v in lawful.items()}
    controls = {rule: min(names, key=lambda n: scores[n][rule]) for rule in scores[names[0]]}
    frontier = [n for n in names if not any(p['old'] == n and p['relation'] == 'new_dominates' for p in pairs)]
    return dict(pairs=pairs, controls=controls, frontier=frontier,
                scores={n: {k: str(v) for k, v in row.items()} for n, row in scores.items()})


def decision(pool, selected, relations, scores, who, arm):
    final = {n: error(v['final']) for n, v in pool['members'].items()}
    better = [n for n in order(final) if final[n] < final[selected]]
    types = {n: next(x['relation'] for x in relations['pairs'] if x['old'] == selected and x['new'] == n) for n in better}
    control_better = [name for name, n in relations['controls'].items() if final[n] < final[selected]]
    category = ('already_best' if not better else
                'visible_dominating_alternative_exists' if 'new_dominates' in types.values() else
                'all_better_alternatives_legally_dominated' if all(t == 'old_dominates' for t in types.values()) else
                'equal_or_tradeoff_evidence')
    return dict(batch=pool['batch'], cohort=pool['cohort'], pool=pool['subject'], selector=who, arm=arm,
                selected=selected, selected_error=str(final[selected]), oracle_error=str(min(final.values())),
                better_final_candidates=types, category=category, simple_controls_that_improve=control_better,
                check_coverage_complete=pool['check_coverage']['complete'],
                native_causal_attribution_not_established=True)


def run(root):
    for name, digest in read(root / 'CODE_FREEZE405.json').items():
        assert sha(root / name) == digest
    assert not (root / 'STARTED405.json').exists()
    save(root / 'STARTED405.json', dict(utc=utc(), status='running'))
    began = time.monotonic(); inputs = load_inputs(root); assert len(inputs['pools']) == 24
    save(root / 'INPUTS405.json', inputs)
    plans = [plan({n: {k: v[k] for k in ['cv','dev','legal']} for n,v in p['members'].items()}) for p in inputs['pools']]
    save(root / 'LEGAL_RELATIONS405.json', dict(utc=utc(), plans=plans, no_final_information_in_plan=True))
    rows, native = [], []; a342 = [p for p in inputs['pools'] if p['batch'] == 'A342']
    for pool, planned in zip(inputs['pools'], plans):
        final = {n: error(v['final']) for n, v in pool['members'].items()}
        best = min(final.values()); best_names = [n for n in order(final) if final[n] == best]
        pairs = [{**p, 'final_old_error': str(final[p['old']]), 'final_new_error': str(final[p['new']]),
                  'final_worsened': final[p['new']] > final[p['old']]} for p in planned['pairs']]
        eligible = pool['complete'] and not pool['registry_deviation'] and pool['check_coverage']['complete']
        rows.append(dict(batch=pool['batch'],cohort=pool['cohort'],pool=pool['subject'],eligible=eligible,
            native_complete=pool['complete'],registry_deviation=pool['registry_deviation'],
            check_coverage=pool['check_coverage'],programs=len(final),final_best=best_names,
            pointwise_frontier=planned['frontier'], all_final_best_excluded=not set(best_names)&set(planned['frontier']),
            improving_legal_pair_worsens_final=any(p['relation']=='new_dominates' and p['final_worsened'] for p in pairs),
            pairs=pairs,controls={k:dict(selected=n,final_error=str(final[n])) for k,n in planned['controls'].items()}))
        if pool['batch'] == 'A342':
            pid = 'pool_' + str(a342.index(pool)+1)
            actual = [x for x in inputs['native_context'] if x['pool']==pid and x['completed']]
            for x in actual:
                native.append(decision(pool,x['selected'],planned,final,x['selector'],x['arm']))
        elif pool['complete'] and not pool['registry_deviation'] and pool['native_submission']:
            arm='ordinary' if pool['subject'].startswith('ordinary') else 'method'
            native.append(decision(pool,pool['native_submission']['program_id'],planned,final,pool['subject'],arm))
    eligible=[r for r in rows if r['eligible']];ordinary=[r for r in native if r['arm']=='ordinary']
    result=dict(utc=utc(),status='complete',seconds=time.monotonic()-began,rows=rows,native_decisions=native,
        all_library_count=24,eligible_library_count=len(eligible),physical_rigs=1,training_cohorts=3,
        all_libraries_with_monotone_legal_improvement_final_regression=sum(r['improving_legal_pair_worsens_final'] for r in rows),
        eligible_libraries_with_monotone_legal_improvement_final_regression=sum(r['improving_legal_pair_worsens_final'] for r in eligible),
        eligible_libraries_whose_all_final_best_excluded=sum(r['all_final_best_excluded'] for r in eligible),
        normal_ordinary_decisions=len(ordinary),ordinary_categories=dict(Counter(r['category'] for r in ordinary)),
        ordinary_decisions_improved_by_any_frozen_simple_control=sum(bool(r['simple_controls_that_improve']) for r in ordinary),
        formal_incremental_method_evidence=False, retrospective=True, no_new_occurrence_rate_claim=True,
        no_GEPA_reproduction_or_refutation=True, new_LLM_calls=0,new_fits=0,new_forwards=0,
        reserved_confirmation_opened=False,goal_complete=False)
    save(root/'RESULT405.json',result);save(root/'TERMINAL405.json',dict(utc=utc(),status='complete'))
    print({k:v for k,v in result.items() if k not in ['rows','native_decisions']})


def audit(root):
    """Second recount uses correctness sets instead of the planner's transitions."""
    source=read(root/'INPUTS405.json');result=read(root/'RESULT405.json');checks=0
    for pool,row in zip(source['pools'],result['rows']):
        correct={n:{(view,g) for view in ['cv','legal'] for g,(y,q) in m[view].items() if y==q} for n,m in pool['members'].items()}
        err={n:Fraction(sum(y!=q for y,q in m['final'].values()),len(m['final'])) for n,m in pool['members'].items()}
        for pair in row['pairs']:
            a,b=correct[pair['old']],correct[pair['new']]
            relation='new_dominates' if a<b else 'old_dominates' if b<a else 'equal_correctness' if a==b else 'tradeoff'
            assert relation==pair['relation'];checks+=1
            assert pair['final_worsened']==(err[pair['new']]>err[pair['old']]);checks+=1
            for v in ['cv','legal']:
                assert set(pair['gains'][v])=={g for view,g in b-a if view==v};checks+=1
                assert set(pair['harms'][v])=={g for view,g in a-b if view==v};checks+=1
        frontier={n for n,s in correct.items() if not any(s<t for t in correct.values())}
        assert frontier==set(row['pointwise_frontier']);checks+=1
        best={n for n,e in err.items() if e==min(err.values())}
        assert row['all_final_best_excluded']==(not (best&frontier));checks+=1
        for decision_row in [d for d in result['native_decisions'] if d['batch']==pool['batch'] and d['pool']==pool['subject']]:
            selected=decision_row['selected'];bests={n for n,e in err.items() if e<err[selected]}
            assert set(decision_row['better_final_candidates'])==bests;checks+=1
            relations=decision_row['better_final_candidates']
            assert all(relations[n]==('new_dominates' if correct[selected]<correct[n] else 'old_dominates' if correct[n]<correct[selected] else 'equal_correctness' if correct[n]==correct[selected] else 'tradeoff') for n in bests);checks+=1
    eligible=[r for r in result['rows'] if r['eligible']];ordinary=[r for r in result['native_decisions'] if r['arm']=='ordinary']
    assert result['eligible_library_count']==len(eligible)
    assert result['eligible_libraries_with_monotone_legal_improvement_final_regression']==sum(r['improving_legal_pair_worsens_final'] for r in eligible)
    assert result['eligible_libraries_whose_all_final_best_excluded']==sum(r['all_final_best_excluded'] for r in eligible)
    assert result['normal_ordinary_decisions']==len(ordinary)
    assert result['ordinary_categories']==dict(Counter(r['category'] for r in ordinary))
    checks+=5
    save(root/'INDEPENDENT_AUDIT405.json',dict(utc=utc(),passed=True,checks=checks,source_result_sha256=sha(root/'RESULT405.json'),
        separate_correctness_set_recount=True,new_LLM_calls=0,new_fits=0,new_forwards=0,
        scope='Numerical recount;source archives are manifest-bound in load_inputs. Not independent physical data.'))
    print(dict(passed=True,checks=checks))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','audit']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args.run.resolve())
