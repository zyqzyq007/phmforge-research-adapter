"""Selection-only diagnostic on archived, real leave-path-out predictions.

No model execution, generated observations, or confidence-bound claim. The choose
action cannot read scores; scoring is a separate invocation after choices seal.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify(root):
    cfg = read(root / 'PROTOCOL407.json')
    for binding in cfg['bindings']:
        assert sha(Path(binding['path'])) == binding['sha256'], binding['path']
    assert sha(Path(__file__)) == cfg['script_sha256']
    return cfg


def distance2(a, b):
    return math.fsum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def choose(root):
    cfg = verify(root)
    assert not (root / 'CHOICES407.json').exists(), 'No implicit rerun'
    start = time.monotonic()
    panels = {}
    for panel, spec in cfg['panels'].items():
        evidence = read(Path(spec['evidence']))
        if isinstance(evidence, dict):
            evidence = evidence['cases']
        rows = []
        for case in evidence:
            scores = {}
            for program in spec['programs']:
                points = [fold['methods'][program]['location_m'] for fold in case['folds']]
                n = len(points)
                assert n == 4 and all(len(p) == 2 for p in points)
                center = [math.fsum(p[j] for p in points) / n for j in range(2)]
                variance = (n - 1) / n * math.fsum(distance2(p, center) for p in points) * 1e6
                bias2 = (n - 1) ** 2 * distance2(center, case['full_predictions'][program]) * 1e6
                amplitude = math.fsum(f['methods'][program][spec['amplitude_key']]
                                      for f in case['folds']) / n
                scores[program] = dict(variance_mm2=variance, bias2_mm2=bias2,
                                       risk_mm2=variance + bias2, amplitude=amplitude)
            choices = {}
            for rule, key in [('primary', 'risk_mm2'), ('variance_only', 'variance_mm2'),
                              ('amplitude', 'amplitude')]:
                low = min(s[key] for s in scores.values())
                choices[rule] = next(p for p in spec['programs'] if scores[p][key] <= low + 1e-10)
            rows.append(dict(identity=case['identity'], group=case['group'],
                             scores=scores, choices=choices))
        assert len(rows) == 7
        panels[panel] = rows
    save(root / 'CHOICES407.json', panels)
    save(root / 'CHOICE_SEAL407.json', dict(sha256=sha(root / 'CHOICES407.json'),
         seconds=time.monotonic()-start, final_scores_read=False, new_model_calls=0,
         new_fits=0, new_predictions=0, independent_confirmation=False))


def score(root):
    cfg = verify(root)
    assert not (root / 'RESULT407.json').exists(), 'No rescoring'
    assert sha(root / 'CHOICES407.json') == read(root / 'CHOICE_SEAL407.json')['sha256']
    chosen = read(root / 'CHOICES407.json')
    out = {}
    for panel, spec in cfg['panels'].items():
        historical = read(Path(spec['score']))
        errors = {}
        for row in historical['rows']:
            errors[row['identity']] = (row['errors_mm'] if panel == 'within_family' else
                {p: v['full_error_mm'] for p, v in row['metrics'].items()})
        result_rows = []
        for row in chosen[panel]:
            e = errors[row['identity']]
            selected = dict(row['choices'], keep='transmission', successful_ordinary='review_01')
            result_rows.append(dict(identity=row['identity'], group=row['group'], selected=selected,
                errors_mm={rule: e[p] for rule, p in selected.items()},
                candidate_errors_mm={p: e[p] for p in spec['programs']},
                oracle_error_mm=min(e[p] for p in spec['programs'])))
        aggregate = {}
        for rule in result_rows[0]['selected']:
            e = [r['errors_mm'][rule] for r in result_rows]
            baseline = [r['errors_mm']['keep'] for r in result_rows]
            opportunity = math.fsum(b-r['oracle_error_mm'] for b, r in zip(baseline, result_rows))
            gain = math.fsum(max(b-x, 0.) for b, x in zip(baseline, e))
            aggregate[rule] = dict(mean_error_mm=math.fsum(e)/7,
                selection_loss_mm=math.fsum(x-r['oracle_error_mm'] for x, r in zip(e, result_rows))/7,
                harm_mm=math.fsum(max(x-b, 0.) for b, x in zip(baseline, e))/7,
                gain_mm=gain/7, gain_capture=gain/opportunity if opportunity > 0 else None,
                group_mean_mm={g: math.fsum(r['errors_mm'][rule] for r in result_rows if r['group']==g)
                    /sum(r['group']==g for r in result_rows) for g in sorted({r['group'] for r in result_rows})})
        p, ordinary, amplitude = (aggregate[k] for k in ['primary', 'successful_ordinary', 'amplitude'])
        gates = dict(lower_error_than_successful_ordinary=p['mean_error_mm'] < ordinary['mean_error_mm']-1e-10,
            lower_error_than_amplitude=p['mean_error_mm'] < amplitude['mean_error_mm']-1e-10,
            no_extra_harm=p['harm_mm'] <= ordinary['harm_mm']+1e-10,
            preserve_gain=p['gain_mm'] >= ordinary['gain_mm']-1e-10,
            both_geometries_improve=all(p['group_mean_mm'][g] < ordinary['group_mean_mm'][g]-1e-10
                                       for g in p['group_mean_mm']))
        out[panel] = dict(rows=result_rows, aggregate=aggregate, gates=gates, gate_passed=all(gates.values()))
    save(root / 'RESULT407.json', dict(panels=out, primary_gate_passed=out['within_family']['gate_passed'],
        new_LLM_calls=0, new_fits=0, new_predictions=0, independent_confirmation=False,
        novelty_established=False, majority_goal_achieved=False))


def audit(root):
    cfg = verify(root)
    chosen, result = read(root / 'CHOICES407.json'), read(root / 'RESULT407.json')
    checks = 0
    def equal(a, b):
        nonlocal checks
        assert math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-10), (a, b)
        checks += 1
    for panel, spec in cfg['panels'].items():
        evidence = read(Path(spec['evidence']))
        if isinstance(evidence, dict):
            evidence = evidence['cases']
        for original, picked in zip(evidence, chosen[panel], strict=True):
            assert original['identity'] == picked['identity']
            for name in spec['programs']:
                points = [f['methods'][name]['location_m'] for f in original['folds']]
                pair_sum = sum((points[i][k]-points[j][k])**2
                               for i in range(4) for j in range(i) for k in range(2))
                var = 3/16 * pair_sum * 1e6
                delta = [sum(p[k] for p in points)-4*original['full_predictions'][name][k] for k in range(2)]
                bias2 = 9/16 * sum(x*x for x in delta) * 1e6
                equal(var, picked['scores'][name]['variance_mm2'])
                equal(bias2, picked['scores'][name]['bias2_mm2'])
                equal(var+bias2, picked['scores'][name]['risk_mm2'])
            for rule, key in [('primary','risk_mm2'), ('variance_only','variance_mm2'), ('amplitude','amplitude')]:
                low = min(picked['scores'][p][key] for p in spec['programs'])
                expected = [p for p in spec['programs'] if picked['scores'][p][key] <= low+1e-10][0]
                assert expected == picked['choices'][rule]
                checks += 1
        rows = result['panels'][panel]['rows']
        for rule, a in result['panels'][panel]['aggregate'].items():
            selected = [r['errors_mm'][rule] for r in rows]
            equal(sum(selected)/len(rows), a['mean_error_mm'])
            equal(sum(max(r['errors_mm'][rule]-r['errors_mm']['keep'], 0) for r in rows)/len(rows), a['harm_mm'])
            equal(sum(max(r['errors_mm']['keep']-r['errors_mm'][rule], 0) for r in rows)/len(rows), a['gain_mm'])
            equal(a['mean_error_mm'], result['panels'][panel]['aggregate']['keep']['mean_error_mm']+a['harm_mm']-a['gain_mm'])
    save(root / 'AUDIT407.json', dict(passed=True, checks=checks, independent_pair_distance_formula=True,
         source_bindings_verified=True, confidence_coverage_claim=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['choose', 'score', 'audit'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.run)
