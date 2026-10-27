"""Independent numerical checks of the frozen A420 cache experiment."""
import argparse
import bisect
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    started, checks = time.monotonic(), 0
    cfg, chosen, result = [read(root / f) for f in ['PROTOCOL420.json', 'CHOICES420.json', 'RESULT420.json']]
    assert read(root / 'TERMINAL420.json')['exit_code'] == 0
    assert read(root / 'SCORE_STARTED420.json')['choices_sha256'] == digest(root / 'CHOICES420.json')
    assert chosen['utc'] < read(root / 'SCORE_STARTED420.json')['utc']
    assert chosen['code_sha256'] == digest(root / 'code/screen_trajectory_validation.py')
    for path, expected in cfg['bindings'].items():
        assert digest(path) == expected, path
        checks += 1
    source = Path(cfg['source'])
    assert digest(source / 'SOURCE_MANIFEST.json') == cfg['source_manifest_sha256']
    transitions, numerical_fields, source_gates = {}, 0, []
    for task, spec in cfg['tasks'].items():
        scale = load(spec['scaler'])
        normalize = lambda a: (np.sign(a) * np.log1p(np.abs(a)) - scale['mean']) / scale['std']
        source_units = {entry['id']: load(source / entry['path']) for entry in cfg['units'][task]}
        descriptors = {}
        for identity, unit in source_units.items():
            times = unit['elapsed'].tolist()
            # Independent binary-search implementation, not selector searchsorted.
            indices = np.array([[max(0, bisect.bisect_right(times, times[0]+(t-times[0])*fraction)-1)
                                 for fraction in cfg['prefix_fractions']] for t in times])
            descriptors[identity] = (normalize(unit['features']), indices)
            assert np.all(np.asarray(times)[indices] <= np.asarray(times)[:, None]+1e-6)
            checks += len(times)
        cases = {case['sample_id']: case for case in cfg['cases'][task]}
        for row in chosen['tasks'][task]['rows']:
            query = load(source / cases[row['sample_id']]['input'])
            current, history = query['features'][0], query['history_features']
            times = history[:, spec['elapsed_column']].tolist()
            endpoint = float(current[spec['elapsed_column']])
            indices = [max(0, bisect.bisect_right(times, times[0]+(endpoint-times[0])*f)-1)
                       for f in cfg['prefix_fractions']]
            assert indices == row['query_prefix_indices']
            signature = normalize(history[indices]); signature[-1] = normalize(current)
            for name, evidence in row['evidence'].items():
                representation, scope = name.split('_', 1)
                eligible = [entry['id'] for entry in cfg['units'][task]
                            if scope == 'all' or source_units[entry['id']]['elapsed'][0] <= endpoint <= source_units[entry['id']]['end']]
                assert [r['unit'] for r in evidence['references']] == eligible
                loss_rows = []
                for record in evidence['references']:
                    unit = source_units[record['unit']]
                    features, source_indices = descriptors[record['unit']]
                    if representation == 'snapshot':
                        difference = features-normalize(current)
                        terms = difference.shape[1]
                        distance = np.linalg.norm(difference.astype(np.float64), axis=1)**2
                    else:
                        difference = features[source_indices]-signature
                        terms = difference.shape[1]*difference.shape[2]
                        distance = np.sum(np.linalg.norm(difference.astype(np.float64), axis=2)**2, axis=1)/len(indices)
                    index = int(distance.argmin())
                    assert index == record['index']
                    epsilon = np.finfo(difference.dtype).eps
                    gamma = terms*epsilon/(1-terms*epsilon)
                    rounding_bound = (gamma+8*np.finfo(np.float64).eps)*abs(distance[index])+1e-12
                    assert abs(distance[index]-record['distance']) <= rounding_bound
                    assert source_indices[index].tolist() == record['prefix_indices']
                    np.testing.assert_array_equal(abs(unit['residual'][index]), record['losses'])
                    loss_rows.append(abs(unit['residual'][index])); checks += 4
                if loss_rows:
                    risks = [math.fsum(float(x[j]) for x in loss_rows)/len(loss_rows) for j in range(7)]
                else:
                    risks = read(spec['previous_choices'])['global_CV']
                np.testing.assert_allclose(risks, evidence['risk'], rtol=1e-12, atol=1e-7)
                best = min(risks)
                winner = sorted([n for n,r in zip(spec['names'], risks) if r <= best+1e-9], key=lambda n:(n != spec['incumbent'], n))[0]
                assert winner == row['choices'][name]; checks += 2
        original = read(spec['cached_scores'])
        if task == 'LDM_RTF':
            anchor = next(v for v in original['variants'] if v.get('dataset') == task and v['id'] == 'fixed_'+spec['incumbent'])['metrics']['event_predictions']
        else:
            anchor = next(v for v in original['variants'] if v['variant']['id'] == 'bank_'+spec['incumbent'])['results']['final']['metrics']['per_query']
        truth = {r['sample_id']: r for r in anchor}
        assert len(truth) == len(cases) == 15
        pred = {sid: read(source / c['output'] / 'predictions.json')['banks']['full'] for sid,c in cases.items()}
        groups = sorted({r['unit_id'] for r in truth.values()})
        old = {sid: abs(pred[sid][spec['incumbent']]-r['actual_rul']) for sid,r in truth.items()}
        oracle = {sid: min(abs(value-r['actual_rul']) for value in pred[sid].values()) for sid,r in truth.items()}
        opportunity = math.fsum(old[sid]-oracle[sid] for sid in truth)
        for policy, decisions in result['tasks'][task]['policies'].items():
            loss = {sid: abs(pred[sid][name]-truth[sid]['actual_rul']) for sid,name in decisions.items()}
            averaged = lambda values: math.fsum(math.fsum(values[sid] for sid in truth if truth[sid]['unit_id']==g)/sum(truth[sid]['unit_id']==g for sid in truth) for g in groups)/len(groups)
            expected = dict(mean_bearing_MAE=averaged(loss), mean_selection_loss=averaged({sid:loss[sid]-oracle[sid] for sid in truth}),
                mean_query_harm=math.fsum(max(loss[sid]-old[sid],0) for sid in truth)/15,
                mean_query_gain=math.fsum(max(old[sid]-loss[sid],0) for sid in truth)/15,
                query_gain_capture=math.fsum(max(old[sid]-loss[sid],0) for sid in truth)/opportunity)
            recorded = result['tasks'][task]['methods'][policy]
            for key,value in expected.items():
                assert math.isclose(value, recorded[key], rel_tol=1e-12, abs_tol=1e-7), (task,policy,key)
                numerical_fields += 1
        methods = result['tasks'][task]['methods']
        controls = [n for n in methods if n.startswith(('r9_', 'fixed_', 'closure_'))]
        best = min(controls, key=lambda n:(methods[n]['mean_bearing_MAE'],n))
        assert best == result['tasks'][task]['strongest_fixed_control']
        a,b = methods['prefix_at_risk'],methods[best]
        gate = a['mean_bearing_MAE'] < b['mean_bearing_MAE']-1e-6 and a['mean_query_harm'] <= b['mean_query_harm']+1e-6 and a['query_gain_capture'] >= b['query_gain_capture']-1e-12
        assert gate == result['tasks'][task]['primary_gate']; source_gates.append(gate)
        changes = []
        for row in chosen['tasks'][task]['rows']:
            sid = row['sample_id']; before,after = row['choices']['snapshot_at_risk'],row['choices']['prefix_at_risk']
            if before != after:
                y=truth[sid]['actual_rul'];changes.append(dict(sample_id=sid,unit=truth[sid]['unit_id'],before=before,after=after,before_error=abs(pred[sid][before]-y),after_error=abs(pred[sid][after]-y)))
        assert len(changes) == result['tasks'][task]['changed_from_snapshot']
        transitions[task] = changes
    assert result['primary_gate'] == all(source_gates)
    assert not result['confirmation_panel_opened'] and not result['ordinary_Agent_comparison']
    assert time.monotonic()-started <= cfg['budget']['audit_seconds']
    audit = dict(utc=datetime.now(timezone.utc).isoformat(), passed=True, binding_and_selection_checks=checks,
        scoring_fields=numerical_fields, seconds=time.monotonic()-started, transitions=transitions,
        result_sha256=digest(root/'RESULT420.json'), new_PHM_fits=0,new_model_forwards=0,new_LLM_calls=0,
        limitations='Numerical/source audit,not evidence of novelty or independent generalization.')
    (root/'audit/INDEPENDENT_AUDIT420.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='transitions'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run.resolve())
