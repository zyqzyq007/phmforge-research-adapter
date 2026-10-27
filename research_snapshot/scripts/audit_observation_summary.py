"""Recount the eligibility subset and native-control binding of A403.

Supplemental reporting audit only; does not change any sealed selection.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def audit(root):
    read = lambda name: json.loads((root / name).read_text())
    result, source = read('RESULT403.json'), read('INPUTS403.json')
    checks = 0
    for name, digest in read('CODE_FREEZE403.json')['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        checks += 1
    assert read('SEALED_SELECTIONS403.json')['utc'] < result['utc']
    checks += 1
    eligible = [r for r in result['rows'] if r['complete'] and not r['registry_deviation']]
    assert len(eligible) == result['eligible_pool_count'] == 13
    checks += 1
    fields = ['error', 'harm', 'gain', 'signed_gap_to_best_single']
    rules = ['global_legal', 'nearest_condition', 'ola7']
    for rule in rules:
        for field in fields:
            values = []
            for cohort in sorted({r['cohort'] for r in eligible}):
                observations = [Fraction(r['scores'][rule][field]) for r in eligible if r['cohort'] == cohort]
                mean = sum(observations, Fraction()) / len(observations)
                assert mean == Fraction(result['eligible']['cohorts'][cohort][rule][field])
                checks += 1
                values.append(mean)
            assert sum(values, Fraction()) / len(values) == Fraction(result['eligible']['means'][rule][field])
            checks += 1
    banks342 = [p for p in source['pools'] if p['batch'] == 'A342']
    comparison = {r: Counter() for r in rules}
    for row in result['native_comparisons']:
        pool = next(p for p in source['pools'] if (p['batch'], p['subject']) == (row['batch'], row['subject']))
        if row['batch'] == 'A342':
            key = 'pool_' + str(banks342.index(pool) + 1)
            bound = [x for x in source['native_context'] if x['pool'] == key and x['selector'] == row['selector'] and x['arm'] == 'ordinary' and x['completed']]
            assert len(bound) == 1
            assert bound[0]['selected'] == row['original_selected']
            assert Fraction(str(bound[0]['native_error'])) == Fraction(row['native']['error'])
            checks += 3
        else:
            assert pool['complete'] and not pool['registry_deviation']
            assert pool['native_submission']['program_id'] == row['original_selected']
            checks += 2
        for rule in rules:
            before = Fraction(row['native']['error'])
            after = Fraction(row['alternatives'][rule]['error'])
            comparison[rule]['win' if after < before else 'loss' if after > before else 'tie'] += 1
    report = dict(utc=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
                  eligible_aggregate_recounted=True, native_choice_mapping_verified=True,
                  native_pair_count=len(result['native_comparisons']),
                  native_error_comparison={r: {k: comparison[r][k] for k in ['win', 'tie', 'loss']} for r in rules},
                  independent_scenarios_added=0, new_LLM_calls=0, new_classifier_fits=0,
                  warning='Related repeated selectors and pools; not13independent tasks or a prospective Agent trial.')
    (root / 'SUPPLEMENTAL_AUDIT403.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run)
