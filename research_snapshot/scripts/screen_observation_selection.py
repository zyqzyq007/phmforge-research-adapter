"""Frozen replay of lawful per-observation selection from actual program pools."""
import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
import math
import time

import numpy as np

from audit_program_evidence import read, save, sha, utc
from audit_duplicate_program_votes import bound_reader, metrics

RULES = ['global_legal', 'nearest_condition', 'ola7']


def metadata(path, require_unlabelled=False):
    with np.load(path, allow_pickle=False) as z:
        if require_unlabelled:
            assert 'labels' not in z.files
        groups, x, hz, condition = (z[k] for k in ['groups', 'features', 'rotation_hz', 'condition_id'])
        return {str(g): dict(x=np.mean(x[groups == g], axis=0).tolist(),
                            rpm=float(np.mean(hz[groups == g])),
                            condition=float(condition[groups == g][0])) for g in sorted(set(groups.tolist()))}


def inputs(root):
    cfg = read(root / 'PROTOCOL403.json')
    cache = Path(cfg['cached_archive']); original = bound_reader(cache)('INPUTS400.json')
    data, bindings, normalizers = {}, [], {}
    for batch, specification in cfg['archives'].items():
        archive = Path(specification['path']); receipt = read(archive / 'ARCHIVE_VERIFIED.json')
        assert receipt['manifest_sha256'] == specification['manifest_sha256']
        manifest = archive / receipt['manifest_filename']; assert sha(manifest) == receipt['manifest_sha256']
        index = {x['path']: x for x in read(manifest)['files']}
        def checked(name):
            p = archive / name
            assert sha(p) == index[name]['sha256']
            bindings.append(dict(batch=batch, path=str(p), sha256=sha(p)))
            return p
        roles = {role: metadata(checked('private_inputs/' + role + '.npz'))
                 for role in ['train', 'development', 'independent_check']}
        query = {}
        for p in sorted((archive / 'evaluation/common_p00/forward').glob('input*.npz')):
            value = metadata(checked(str(p.relative_to(archive))), require_unlabelled=True)
            assert not set(value) & set(query)
            query.update(value)
        train = np.asarray([x['x'] for x in roles['train'].values()])
        cohort = specification['cohort']
        if cohort not in normalizers:
            mu = train.mean(axis=0); std = train.std(axis=0); std[std == 0] = 1
            normalizers[cohort] = (train, mu, std)
        else:
            old_train, mu, std = normalizers[cohort]
            assert np.array_equal(old_train, train)
        data[batch] = dict(**roles, query=query, mean=mu.tolist(), scale=std.tolist())
        for pool in [p for p in original['pools'] if p['batch'] == batch]:
            subject = 'subjects/' + pool['subject']; state = read(checked(subject + '/broker/STATE.json'))
            probe = state.get('probe') or {}
            checks = probe.get('reports', {}) if probe.get('status') == 'complete' else {}
            origin = 'broker_probe' if checks else 'missing'
            if (archive / subject / 'common_check/REPORT.json').exists():
                checks = read(checked(subject + '/common_check/REPORT.json'))['programs']; origin = 'post_build_common_check'
            pool['members'].update(pool.get('compositions', {}))
            names = pool['members']
            missing = [n for n in names if n not in checks or checks[n].get('status', 'complete') != 'complete']
            pool['check_coverage'] = dict(source=origin, missing=missing, complete=not missing)
            for name, model in names.items():
                legal = dict(model['dev'])
                assert set(legal) == set(roles['development'])
                if not missing:
                    rows = {x['group']: [x['actual'], x['predicted']] for x in checks[name]['records']}
                    assert set(rows) == set(roles['independent_check']) and not set(legal) & set(rows)
                    legal.update(rows)
                model['legal'] = legal
                assert set(model['final']) == set(query)
    return dict(pools=original['pools'], metadata=data, native_context=original['prior_native_context'], bindings=bindings)


def plan(pool, meta):
    names = sorted(pool['members'], key=lambda n: (n != 'p00', n))
    models = {n: pool['members'][n]['legal'] for n in names}
    available = sorted(models[names[0]])
    assert all(set(m) == set(available) for m in models.values())
    for g in available:
        assert len({m[g][0] for m in models.values()}) == 1
    ref = {**meta['development'], **meta['independent_check']}
    # This function receives no final labels or predictions as selection evidence.
    error = lambda n, groups: sum(models[n][g][0] != models[n][g][1] for g in groups)
    global_id = min(names, key=lambda n: (error(n, available), names.index(n)))
    conditions = sorted({ref[g]['condition'] for g in available})
    centers = {c: sum(ref[g]['rpm'] for g in available if ref[g]['condition'] == c) /
               sum(ref[g]['condition'] == c for g in available) for c in conditions}
    decisions = {}
    for group, query in sorted(meta['query'].items()):
        chosen_condition = min(conditions, key=lambda c: (abs(query['rpm'] - centers[c]), c))
        near_condition = [g for g in available if ref[g]['condition'] == chosen_condition]
        distances = {g: sum(((a - b) / s) ** 2 for a, b, s in zip(query['x'], ref[g]['x'], meta['scale']))
                     for g in available}
        near_features = sorted(available, key=lambda g: (distances[g], g))[:min(7, len(available))]
        choices = dict(global_legal=global_id,
                       nearest_condition=min(names, key=lambda n: (error(n, near_condition), names.index(n))),
                       ola7=min(names, key=lambda n: (error(n, near_features), names.index(n))))
        decisions[group] = dict(selected=choices, condition_reference=near_condition,
                                feature_reference=near_features, selected_condition=chosen_condition,
                                query_rotation_hz=query['rpm'], candidate_errors={rule: {n: error(n, groups)
                                 for n in names} for rule, groups in [('global_legal', available),
                                     ('nearest_condition', near_condition), ('ola7', near_features)]})
    return dict(batch=pool['batch'], cohort=pool['cohort'], subject=pool['subject'],
                candidates=names, legal_records=len(available), check_coverage=pool['check_coverage'], decisions=decisions)


def summarize(rows):
    fields = ['error', 'harm', 'gain', 'signed_gap_to_best_single']
    cohorts = {}
    for cohort in sorted({r['cohort'] for r in rows}):
        subset = [r for r in rows if r['cohort'] == cohort]
        cohorts[cohort] = {rule: {key: str(sum((Fraction(r['scores'][rule][key]) for r in subset), Fraction()) / len(subset))
                                for key in fields} for rule in RULES}
    means = {rule: {key: str(sum((Fraction(c[rule][key]) for c in cohorts.values()), Fraction()) / len(cohorts))
                    for key in fields} for rule in RULES}
    return dict(cohorts=cohorts, means=means)


def run(root):
    cfg = read(root / 'PROTOCOL403.json')
    for relative, digest in read(root / 'CODE_FREEZE403.json')['files'].items():
        assert sha(root / relative) == digest
    assert not (root / 'STARTED403.json').exists()
    save(root / 'STARTED403.json', dict(utc=utc(), status='running'))
    start = time.monotonic(); source = inputs(root)
    assert len(source['pools']) == 20
    save(root / 'INPUTS403.json', source)
    # The actual planner only needs lawful predictions and one label-free query.
    plans = []
    for p in source['pools']:
        legal_only = {k: p[k] for k in ['batch', 'cohort', 'subject', 'check_coverage']}
        legal_only['members'] = {n: {'legal': v['legal']} for n, v in p['members'].items()}
        plans.append(plan(legal_only, source['metadata'][p['batch']]))
    save(root / 'SEALED_SELECTIONS403.json', dict(utc=utc(), plans=plans, final_label_used_for_selection=False,
                                                query_batch_statistics_used=False, prediction_time_only_observations=True))
    rows, native = [], []
    a342 = [p for p in source['pools'] if p['batch'] == 'A342']
    for p, selected in zip(source['pools'], plans):
        truth = {g: r[0] for g, r in p['members']['p00']['final'].items()}
        base = {g: r[1] for g, r in p['members']['p00']['final'].items()}
        best = min(Fraction(sum(y != q for y, q in m['final'].values()), len(truth)) for m in p['members'].values())
        scores = {}
        for rule in RULES:
            prediction = {g: p['members'][v['selected'][rule]]['final'][g][1] for g, v in selected['decisions'].items()}
            scores[rule] = metrics(prediction, truth, base, best)
        rows.append(dict(batch=p['batch'], cohort=p['cohort'], subject=p['subject'], complete=p['complete'],
                         registry_deviation=p['registry_deviation'], scores=scores, check_coverage=p['check_coverage']))
        actual = []
        if p['batch'] == 'A342':
            pool_id = 'pool_' + str(a342.index(p) + 1)
            actual = [x for x in source['native_context'] if x['pool'] == pool_id and x['arm'] == 'ordinary' and x['completed']]
        elif p['complete'] and not p['registry_deviation'] and p['subject'].startswith('ordinary'):
            actual = [dict(selector=p['subject'], selected=p['native_submission']['program_id'])]
        for item in actual:
            old = {g: x[1] for g, x in p['members'][item['selected']]['final'].items()}
            native.append(dict(batch=p['batch'], cohort=p['cohort'], subject=p['subject'],
                selector=item['selector'], original_selected=item['selected'],
                native=metrics(old, truth, base, best), alternatives=scores))
    aggregate = summarize(rows); eligible = [r for r in rows if r['complete'] and not r['registry_deviation']]
    gates = {}
    for rule in RULES[1:]:
        before, after = aggregate['means']['global_legal'], aggregate['means'][rule]
        wins = sum(Fraction(c[rule]['error']) < Fraction(c['global_legal']['error']) for c in aggregate['cohorts'].values())
        gates[rule] = dict(cohort_wins=wins, passed=wins >= 2 and Fraction(after['error']) < Fraction(before['error'])
                          and Fraction(after['harm']) <= Fraction(before['harm']) and Fraction(after['gain']) >= Fraction(before['gain']))
    result = dict(utc=utc(), status='completed', seconds=time.monotonic() - start, rows=rows, native_comparisons=native,
                  **aggregate, eligible_pool_count=len(eligible), eligible=summarize(eligible), gates=gates,
                  related_libraries=20, physical_rigs=1, training_cohorts=3,
                  additional_check_complete_pools=sum(r['check_coverage']['complete'] for r in rows),
                  new_LLM_calls=0, new_classifier_fits=0, new_classifier_forwards=0,
                  per_training_cohort_unsupervised_normalization=3,
                  official_DESlib_execution=False, established_OLA_adaptation=True,
                  innovative_method_confirmed=False, reserved_confirmation_opened=False, goal_complete=False)
    save(root / 'RESULT403.json', result); save(root / 'TERMINAL403.json', dict(utc=utc(), status='completed'))
    print(dict(means=result['means'], gates=gates, seconds=result['seconds']))


def audit(root):
    source = inputs(root); assert source == read(root / 'INPUTS403.json')
    result = read(root / 'RESULT403.json'); plans = read(root / 'SEALED_SELECTIONS403.json')['plans']; checks = 0
    for pool, stored, row in zip(source['pools'], plans, result['rows']):
        meta = source['metadata'][pool['batch']]
        reference = {**meta['development'], **meta['independent_check']}
        names = sorted(pool['members'], key=lambda n: (n != 'p00', n))
        legal = sorted(pool['members'][names[0]]['legal'])
        matrix = np.asarray([[pool['members'][n]['legal'][g][0] != pool['members'][n]['legal'][g][1]
                              for n in names] for g in legal], dtype=int)
        x = np.asarray([reference[g]['x'] for g in legal]); scale = np.asarray(meta['scale'])
        errors, harms, gains = {r: 0 for r in RULES}, {r: 0 for r in RULES}, {r: 0 for r in RULES}
        for group, query in meta['query'].items():
            cond = Counter(reference[g]['condition'] for g in legal)
            centers = {c: math.fsum(reference[g]['rpm'] for g in legal if reference[g]['condition'] == c) / count for c, count in cond.items()}
            closest = min(centers, key=lambda c: (abs(query['rpm'] - centers[c]), c))
            cidx = [i for i, g in enumerate(legal) if reference[g]['condition'] == closest]
            distance = np.linalg.norm((x - np.asarray(query['x'])) / scale, axis=1)
            fidx = sorted(range(len(legal)), key=lambda i: (float(distance[i]), legal[i]))[:min(7, len(legal))]
            for rule, indices in [('global_legal', list(range(len(legal)))), ('nearest_condition', cidx), ('ola7', fidx)]:
                values = np.sum(matrix[indices], axis=0); chosen = names[int(np.argmin(values))]
                assert chosen == stored['decisions'][group]['selected'][rule]; checks += 1
                y, before = pool['members']['p00']['final'][group]
                after = pool['members'][chosen]['final'][group][1]
                assert row['scores'][rule]['predictions'][group] == after; checks += 1
                errors[rule] += y != after; harms[rule] += y == before and y != after; gains[rule] += y != before and y == after
        for rule in RULES:
            for key, count in [('error', errors[rule]), ('harm', harms[rule]), ('gain', gains[rule])]:
                assert Fraction(row['scores'][rule][key]) == Fraction(count, len(meta['query'])); checks += 1
            best = min(Fraction(sum(a != b for a, b in m['final'].values()), len(meta['query'])) for m in pool['members'].values())
            assert Fraction(row['scores'][rule]['signed_gap_to_best_single']) == Fraction(errors[rule], len(meta['query'])) - best
            checks += 1
    for rule in RULES:
        for key in ['error', 'harm', 'gain', 'signed_gap_to_best_single']:
            cohort_means = []
            for cohort in result['cohorts']:
                values = [Fraction(r['scores'][rule][key]) for r in result['rows'] if r['cohort'] == cohort]
                mean = sum(values, Fraction()) / len(values); cohort_means.append(mean)
                assert mean == Fraction(result['cohorts'][cohort][rule][key]); checks += 1
            assert sum(cohort_means, Fraction()) / len(cohort_means) == Fraction(result['means'][rule][key]); checks += 1
    for rule in RULES[1:]:
        wins = sum(Fraction(c[rule]['error']) < Fraction(c['global_legal']['error']) for c in result['cohorts'].values())
        old, new = result['means']['global_legal'], result['means'][rule]
        passed = wins >= 2 and Fraction(new['error']) < Fraction(old['error']) and Fraction(new['harm']) <= Fraction(old['harm']) and Fraction(new['gain']) >= Fraction(old['gain'])
        assert result['gates'][rule] == dict(cohort_wins=wins, passed=passed); checks += 1
    for native in result['native_comparisons']:
        pool = next(p for p in source['pools'] if (p['batch'], p['subject']) == (native['batch'], native['subject']))
        row = next(p for p in result['rows'] if (p['batch'], p['subject']) == (native['batch'], native['subject']))
        vals = pool['members'][native['original_selected']]['final']
        err = Fraction(sum(a != b for a, b in vals.values()), len(vals))
        assert err == Fraction(native['native']['error']) and native['alternatives'] == row['scores']; checks += 1
    save(root / 'INDEPENDENT_AUDIT403.json', dict(utc=utc(), passed=True, checks=checks,
         source_arrays_and_checks_rebound=True, numeric_method='NumPy norm/matrix sum/argmin vs scalar distances/counts; Fraction aggregate recount.',
         new_fits=0, new_forwards=0))
    print(dict(passed=True, checks=checks))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['run', 'audit']); p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); (run if a.mode == 'run' else audit)(a.run.resolve())
