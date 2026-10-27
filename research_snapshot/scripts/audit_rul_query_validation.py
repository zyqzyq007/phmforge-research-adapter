"""Independently recount the frozen retrospective RUL selection experiment."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as packet:
        return {key: packet[key] for key in packet.files}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
root = parser.parse_args().run.resolve()
cfg = read(root / 'PROTOCOL354.json')
names = cfg['names']
for name, value in read(root / 'CODE_FREEZE354.json').items():
    assert digest(root / name) == value, name
for name, value in read(root / 'PREPARED354.json')['bindings'].items():
    assert digest(root / name) == value, name
for name, value in cfg['source_bindings'].items():
    assert digest(Path(cfg['archive']) / name) == value, name
assert not Path('/proc', str(read(root / 'STARTED354.json')['pid'])).exists()

def check_process(folder, mode):
    process = read(folder / 'PROCESS.json')
    audit = read(folder / 'training_audit.json')
    assert process['returncode'] == 0 and process['mode'] == mode
    command = process['command']
    assert '--unshare-net' in command and '--remount-ro' in command
    assert '/input.npz' in command and '/work' in command
    assert not any('/private' in word for word in command)
    if mode == 'predict':
        assert audit['completed_estimator_fits'] == audit['optimizer_steps'] == 0
    return audit


def mapped(folder):
    models = read(folder / 'predictions.json')['models']
    assert set(models) == set(names)
    result = {name: {r['sample_id']: float(r['predicted_rul']) for r in rows}
              for name, rows in models.items()}
    assert all(len(result[n]) == len(models[n]) for n in names)
    return result


forward = mapped(root / 'preflight/development')
reverse = mapped(root / 'preflight/development_reverse')
maximum_delta = 0.
for name in names:
    assert set(forward[name]) == set(reverse[name])
    for sample, value in forward[name].items():
        other = reverse[name][sample]
        maximum_delta = max(maximum_delta, abs(value - other))
        assert abs(value - other) <= 1e-6 + 1e-6 * abs(other)
for variant in ['development', 'development_reverse']:
    check_process(root / 'preflight' / variant, 'predict')

units = []
cv = []
fit_count = optimizer_steps = changed_steps = dense_rows = 0
for fold in read(root / 'FOLDS354.json'):
    unit = str(fold['unit'])
    folder = root / 'folds' / unit
    audit = check_process(folder / 'fit', 'fit')
    assert Counter(e['class_name'] for e in audit['estimator_fit_events']) == {
        'Ridge': 1, 'ExtraTreesRegressor': 1, 'HistGradientBoostingRegressor': 4}
    assert all(e['status'] == 'completed' and e['rows'] == 1500
               for e in audit['estimator_fit_events'])
    assert audit['optimizer_steps'] == 960 and audit['nonzero_sampled_optimizer_steps'] > 0
    fit_count += audit['completed_estimator_fits']
    optimizer_steps += audit['optimizer_steps']
    changed_steps += audit['nonzero_sampled_optimizer_steps']
    train = load(root / 'inputs' / unit / 'train.npz')
    obs = load(root / 'inputs' / unit / 'observed.npz')
    truth = load(root / 'private' / (unit + '.npz'))
    assert Counter(train['unit_ids'].tolist()) == {u: 250 for u in fold['fit_units']}
    assert fold['unit'] not in set(train['unit_ids'].tolist())
    assert set(obs['unit_ids'].tolist()) == {fold['unit']} and 'labels' not in obs
    assert np.array_equal(obs['sample_ids'], truth['sample_ids'])
    assert np.allclose(truth['elapsed'] + truth['labels'], truth['end'], atol=1e-6, rtol=0)
    artifacts = folder / 'fit/artifacts'
    for relative, value in read(root / 'TRAINING354.json')['folds'][unit]['model_bindings'].items():
        assert digest(artifacts / relative) == value
    for name in names[:-1]:
        model = joblib.load(artifacts / 'candidates' / (name + '.joblib'))
        assert model['fitting_units'] == fold['fit_units']
        expected = list(range(30))
        if name == 'vibration':
            expected = list(range(20)) + [29]
        elif name == 'temperature':
            expected = [26, 27, 28, 29]
        elif name == 'operating_conditions':
            expected = list(range(20, 26)) + [29]
        assert model['columns'] == expected
    neural = read(artifacts / 'mlp/training_audit.json')
    assert neural['epochs_executed'] == neural['selected_epoch'] == 80
    assert neural['optimizer_steps'] == 960 and neural['nonzero_gradient_steps'] > 0
    assert neural['initial_state_sha256'] != neural['selected_state_sha256']
    initial = torch.load(artifacts / 'mlp/untrained_model.pt', weights_only=True, map_location='cpu')
    final = torch.load(artifacts / 'mlp/model.pt', weights_only=True, map_location='cpu')
    assert any(not torch.equal(initial['state_dict'][k], v) for k, v in final['state_dict'].items())
    check_process(folder / 'prediction', 'predict')
    pred = mapped(folder / 'prediction')
    assert all(set(pred[name]) == set(obs['sample_ids'].tolist()) for name in names)
    matrix = np.asarray([[pred[name][sample] for name in names] for sample in obs['sample_ids']])
    oof = load(folder / 'OOF.npz')
    assert np.array_equal(oof['sample_ids'], obs['sample_ids'])
    assert np.array_equal(oof['prediction'], matrix)
    losses = abs(matrix - truth['labels'][:, None])
    positions = np.linspace(int(.1 * (len(losses) - 1)), len(losses) - 1, 8).astype(int)
    cv.append(losses[positions].mean(0))
    units.append(dict(id=fold['unit'], t=truth['elapsed'], end=float(truth['end']),
                      x=truth['features'], losses=losses))
    dense_rows += len(losses)

choices = read(root / 'CHOICES354.json')
assert choices['cached_target_predictions_or_labels_loaded'] is False
assert choices['code_freeze_sha256'] == digest(root / 'CODE_FREEZE354.json')
assert datetime.fromisoformat(read(root / 'SCORE_STARTED354.json')['utc']) > datetime.fromisoformat(choices['utc'])
global_cv = np.mean(cv, axis=0)
np.testing.assert_allclose(global_cv, choices['global_CV'], atol=1e-7, rtol=0)
scaler = load(root / 'inputs/scaler.npz')

def signed(values):
    return np.sign(values) * np.log1p(np.abs(values))


def choose(errors):
    best = min(errors)
    return min((i for i, e in enumerate(errors) if e <= best + 1e-9),
               key=lambda i: (names[i] != 'hgb100', names[i]))


support_sizes = Counter()
for row, packet in zip(choices['rows'], cfg['queries']):
    observed = load(Path(cfg['archive']) / packet)
    assert row['sample_id'] == observed['sample_ids'][0] and 'labels' not in observed
    query = observed['features'][0]
    assert row['current_elapsed'] == float(query[29])
    q = (signed(query) - scaler['mean']) / scaler['std']
    assert row['choices']['keep'] == 'hgb100'
    assert row['choices']['global_CV'] == names[choose(global_cv)]
    assert row['choices']['development'] == names[choose(choices['development_risk'])]
    for policy in ['time_all', 'time_at_risk', 'feature_all', 'feature_at_risk']:
        eligible = [u for u in units if not policy.endswith('at_risk')
                    or u['t'][0] <= query[29] <= u['end']]
        support_sizes[(policy, len(eligible))] += 1
        assert row['support'][policy]['units'] == [u['id'] for u in eligible]
        if not eligible:
            assert row['choices'][policy] == row['choices']['global_CV']
            continue
        losses = []
        for u, recorded in zip(eligible, row['support'][policy]['reference_positions']):
            if policy.startswith('time'):
                candidates = [i for i, t in enumerate(u['t']) if t <= query[29]]
                index = candidates[-1] if candidates else 0
            else:
                delta = (signed(u['x']) - scaler['mean']) / scaler['std'] - q
                index = int(np.argmin(np.sum(delta * delta, axis=1)))
            assert recorded == dict(unit=u['id'], elapsed=float(u['t'][index]))
            losses.append(u['losses'][index])
        risk = np.mean(losses, axis=0)
        np.testing.assert_allclose(risk, row['support'][policy]['estimated_MAE'], atol=1e-7, rtol=0)
        assert row['choices'][policy] == names[choose(risk)]

original = read(Path(cfg['archive']) / cfg['cached_scores'])
variants = {v['id']: v for v in original['variants'] if v.get('dataset') == 'LDM_RTF'}
cached = {name: {r['sample_id']: r for r in variants['fixed_model' if name == 'mlp'
          else 'fixed_' + name]['metrics']['event_predictions']} for name in names}
score = read(root / 'RESULT354.json')
losses = {}
for row in choices['rows']:
    sample = row['sample_id']
    record = cached['hgb100'][sample]
    actual = record['actual_rul']
    unit = record['unit_id']
    baseline = abs(record['predicted_rul'] - actual)
    oracle = min(abs(cached[n][sample]['predicted_rul'] - actual) for n in names)
    losses[sample] = dict(unit=unit, baseline=baseline, oracle=oracle,
                         methods={m: abs(cached[n][sample]['predicted_rul'] - actual)
                                  for m, n in row['choices'].items()})

independent = {}
for method, reference in score['methods'].items():
    per_unit = {u: math.fsum(r['methods'][method] for r in losses.values() if r['unit'] == u) /
                sum(r['unit'] == u for r in losses.values()) for u in {r['unit'] for r in losses.values()}}
    mae = math.fsum(per_unit.values()) / len(per_unit)
    harm = math.fsum(max(r['methods'][method] - r['baseline'], 0) for r in losses.values()) / len(losses)
    gain = math.fsum(max(r['baseline'] - r['methods'][method], 0) for r in losses.values())
    opportunity = math.fsum(r['baseline'] - r['oracle'] for r in losses.values())
    assert abs(mae - reference['mean_bearing_MAE']) < 1e-7
    assert abs(harm - reference['mean_query_harm']) < 1e-7
    assert abs(gain / opportunity - reference['query_gain_capture']) < 1e-12
    independent[method] = dict(mean_bearing_MAE=mae, query_harm=harm,
                               useful_gain_capture=gain / opportunity)

report = dict(status='passed', utc=datetime.now(timezone.utc).isoformat(),
              source_and_code_hashes_verified=True, whole_bearing_isolation_verified=True,
              label_free_predict_mounts_verified=True, seven_real_fit_processes=7,
              actual_sklearn_fits=fit_count, actual_neural_models=7,
              actual_optimizer_steps=optimizer_steps, sampled_nonzero_steps=changed_steps,
              neural_parameters_changed_in_all_seven=True, prediction_processes=9,
              dense_OOF_rows=dense_rows, training_bearings=7, target_bearings=5,
              choices_recomputed_without_importing_selector=True, sealed_before_score=True,
              scalar_metrics_independently_recounted=True, numerical_row_order_max_delta=maximum_delta,
              eligible_support_counts={str(k): v for k, v in support_sizes.items()},
              methods=independent, independent_confirmation_scenarios=0, majority_goal_complete=False)
(root / 'audit/QUALIFICATION354.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
