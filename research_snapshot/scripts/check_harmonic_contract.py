"""Real-input formula witnesses and a frozen 2x2 program-repair experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
from pathlib import Path
import time
import traceback

import numpy as np

from audit_program_evidence import read, save, sha, utc
from program_revision_runtime import Runtime, load, score, subset


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify_archive(path, names):
    receipt = read(path / 'ARCHIVE_VERIFIED.json')
    manifest = path / receipt['manifest_filename']
    assert sha(manifest) == receipt['manifest_sha256']
    table = {r['path']: r for r in read(manifest)['files']}
    for name in names:
        assert sha(path / name) == table[name]['sha256'], name


def contract(root, cfg):
    original = module(root / 'programs/original/source/program.py', 'original_feature_program')
    repaired = module(root / 'programs/pair/source/program.py', 'paired_feature_program')
    rows = []
    tolerance = cfg['formula_tolerance']
    for role in cfg['contract_roles']:
        data = load(Path(cfg['data_archive']) / 'qualification' / (role + '.npz'))
        for group in sorted(set(data['groups'].tolist())):
            batch = subset(data, data['groups'] == group)
            waves = batch['waveforms']
            actual = original.features(batch)
            fixed = repaired.features(batch)
            wrong, expected, shifted_wrong, shifted_expected = [], [], [], []
            for x, hz, fs in zip(waves, batch['rotation_hz'], batch['sampling_rate_hz']):
                centered = x - x.mean(axis=0, keepdims=True)
                k = np.arange(1, 13)[:, None]
                phase = 2 * np.pi * hz * np.arange(len(x))[None, :] / fs
                # Independent complex projection; no transformed/synthetic input.
                coeff = np.exp(1j * k * phase) @ centered * (2 / len(x))
                c, s = coeff.real, coeff.imag
                joined = np.concatenate([c, s], axis=0).reshape(12, 2, 6)
                wrong.append(np.linalg.norm(joined, axis=1))
                expected.append(np.abs(coeff))
                rotated = coeff * np.exp(1j * np.arange(1, 13) * cfg['origin_phase_radians'])[:, None]
                shifted_wrong.append(np.linalg.norm(np.concatenate([rotated.real, rotated.imag], axis=0).reshape(12, 2, 6), axis=1))
                shifted_expected.append(np.abs(rotated))
            wrong, expected = np.asarray(wrong), np.asarray(expected)
            shifted_wrong, shifted_expected = np.asarray(shifted_wrong), np.asarray(shifted_expected)
            scale = np.sqrt(np.mean((waves - waves.mean(axis=1, keepdims=True)) ** 2, axis=1))
            recovered = np.exp(actual[:, 30:102].reshape(-1, 12, 6)) * scale[:, None, :]
            recovered_fixed = np.exp(fixed[:, 30:102].reshape(-1, 12, 6)) * scale[:, None, :]
            norm = lambda x: float(np.linalg.norm(x))
            rel = lambda a, b: norm(a - b) / max(norm(b), 1e-24)
            matches_wrong = rel(recovered, wrong)
            fixed_matches_complex = rel(recovered_fixed, expected)
            assert matches_wrong <= tolerance and fixed_matches_complex <= tolerance
            oracle = np.zeros((len(waves), 12, 6)); oracle[:, :, 0] = 1
            geometry = actual[:, 186:258].reshape(-1, 12, 6)
            const_error = float(np.max(np.abs(geometry - oracle)))
            assert const_error <= tolerance
            row = dict(role=role, group=group, windows=len(waves),
                       source_output_wrong_pair_match=matches_wrong,
                       repaired_output_complex_amplitude_match=fixed_matches_complex,
                       original_amplitude_formula_relative_error=rel(recovered, expected),
                       wrong_origin_change_relative=rel(shifted_wrong, wrong),
                       correct_origin_change_relative=rel(shifted_expected, expected),
                       geometry_max_constant_residual=const_error)
            rows.append(row)
    result = dict(utc=utc(), records=rows, record_count=len(rows), windows=sum(x['windows'] for x in rows),
                  original_formula_violations=sum(x['original_amplitude_formula_relative_error'] > tolerance for x in rows),
                  original_origin_violations=sum(x['wrong_origin_change_relative'] > tolerance for x in rows),
                  corrected_origin_violations=sum(x['correct_origin_change_relative'] > tolerance for x in rows),
                  constant_geometry_records=sum(x['geometry_max_constant_residual'] <= tolerance for x in rows),
                  new_classifier_fits=0, new_classifier_predictions=0,
                  feature_function_calls=2 * len(rows), waveform_values_modified=False,
                  repair_quality_claim=False)
    save(root / 'CONTRACT402.json', result)
    return result


def train_one(root, name):
    engine = Runtime(root)
    path = root / 'programs' / name
    start = time.monotonic()
    try:
        report = engine.fit_program(path)
        check = load(Path(engine.config['input_root']) / 'independent_check.npz')
        prediction, _ = engine.predict(path / 'independent_check', path / 'source', path / 'full_fit/artifacts', check)
        return name, dict(status='complete', seconds=time.monotonic() - start, report=report, check=score(check, prediction))
    except Exception as error:
        result = dict(status='failed', error=str(error), traceback=traceback.format_exc(), seconds=time.monotonic() - start)
        save(path / 'FAILED.json', result)
        return name, result


def evaluate_one(root, name, data):
    engine = Runtime(root); engine.deadline = time.monotonic() + 900
    path = root / 'programs' / name
    a, _ = engine.predict(path / 'final', path / 'source', path / 'full_fit/artifacts', data)
    b, _ = engine.predict(path / 'final_repeat', path / 'source', path / 'full_fit/artifacts', data)
    assert np.array_equal(a, b)
    return name, score(data, a)


def main(root):
    cfg = read(root / 'PROTOCOL402.json')
    freeze = read(root / 'CODE_FREEZE402.json')
    for relative, digest in freeze['files'].items():
        assert sha(root / relative) == digest, relative
    assert not (root / 'STARTED402.json').exists()
    verify_archive(Path(cfg['data_archive']), ['qualification/' + x + '.npz' for x in cfg['contract_roles']])
    verify_archive(Path(cfg['program_archive']), [cfg['source_relative'], cfg['report_relative']])
    save(root / 'STARTED402.json', dict(utc=utc(), status='running'))
    start = time.monotonic()
    witness = contract(root, cfg)
    qualify = (witness['original_formula_violations'] > 0 and witness['corrected_origin_violations'] == 0
               and witness['constant_geometry_records'] == witness['record_count'])
    if not qualify:
        save(root / 'TERMINAL402.json', dict(status='contract_not_qualified', utc=utc()))
        return
    trained = {}
    with ThreadPoolExecutor(max_workers=cfg['workers']) as executor:
        futures = [executor.submit(train_one, root, name) for name in cfg['variants']]
        for future in as_completed(futures):
            name, value = future.result(); trained[name] = value
            save(root / 'TRAINING_PROGRESS402.json', trained)
    assert all(x['status'] == 'complete' for x in trained.values()), 'Incomplete factorial; preserve failures, do not resample'
    old = read(Path(cfg['program_archive']) / cfg['report_relative'])
    # A same-source same-environment refit must reproduce all original legal record predictions.
    lookup = lambda r: {x['group']: x['predicted'] for x in r['records']}
    matched = [lookup(a) == lookup(b) for a, b in zip(old['cv'], trained['original']['report']['cv'])]
    matched.append(lookup(old['development']) == lookup(trained['original']['report']['development']))
    save(root / 'ORIGINAL_REPRODUCTION402.json', dict(passed=all(matched), legal_views=matched))
    assert all(matched), 'Original legal predictions not reproduced; no repair attribution'
    legal = {name: dict(mean_cv=sum(x['record_error'] for x in value['report']['cv']) / 2,
                        development=value['report']['development']['record_error'],
                        check=value['check']['record_error']) for name, value in trained.items()}
    edits = cfg['edit_count']
    choices = dict(keep='original', repair_pair='pair', drop_geometry='geometry', repair_both='both',
                   fixed_mean_cv=min(legal, key=lambda x: (legal[x]['mean_cv'], legal[x]['development'], edits[x], x)),
                   fixed_development=min(legal, key=lambda x: (legal[x]['development'], legal[x]['mean_cv'], edits[x], x)),
                   fixed_full_validation=min(legal, key=lambda x: ((8 * legal[x]['development'] + 4 * legal[x]['check']) / 12, edits[x], x)))
    save(root / 'SEALED_DECISIONS402.json', dict(utc=utc(), legal=legal, choices=choices,
                                               final_loaded_by_controller=False, generated_candidates_fixed_before_training=True))
    verify_archive(Path(cfg['data_archive']), ['qualification/final.npz'])
    data = load(Path(cfg['data_archive']) / 'qualification/final.npz')
    with ThreadPoolExecutor(max_workers=cfg['workers']) as executor:
        final = dict(executor.map(lambda name: evaluate_one(root, name, data), cfg['variants']))
    base = lookup(final['original'])
    labels = {x['group']: x['actual'] for x in final['original']['records']}
    best = min(x['record_error'] for x in final.values())
    opportunity = final['original']['record_error'] - best
    for name, value in final.items():
        pred = lookup(value)
        value.update(selection_loss=value['record_error'] - best,
                     net_harm=max(0, value['record_error'] - final['original']['record_error']),
                     paired_harm=sum(base[g] == labels[g] and pred[g] != labels[g] for g in base) / len(base),
                     paired_gain=sum(base[g] != labels[g] and pred[g] == labels[g] for g in base) / len(base),
                     gain_capture=(max(0, final['original']['record_error'] - value['record_error']) / opportunity) if opportunity else None)
    audits = [read(p) for p in root.glob('programs/*/**/training_audit.json')]
    result = dict(utc=utc(), status='completed', seconds=time.monotonic() - start, legal=legal, final=final, choices=choices,
                  selected_results={k: final[v] for k, v in choices.items()},
                  joint_local_pair_repair_improvement=(final['pair']['record_error'] < final['original']['record_error']
                                                       and final['pair']['paired_harm'] == 0),
                  cost=dict(new_LLM_calls=0, fit_entries=sum(x['mode'] == 'fit' for x in audits),
                            classifier_fits=sum(x['completed_estimator_fits'] for x in audits),
                            optimizer_steps=sum(x['optimizer_steps'] for x in audits),
                            cold_prediction_processes=sum(x['mode'] == 'predict' for x in audits)),
                  independent_confirmation=False, majority_goal_complete=False,
                  scope='One real historical candidate, one already-opened rig cohort; investigator repairs, not a new Agent method.')
    save(root / 'RESULT402.json', result)
    save(root / 'TERMINAL402.json', dict(utc=utc(), status='completed', cost=result['cost']))
    print(dict(legal=legal, final={k: v['record_error'] for k, v in final.items()}, choices=choices, cost=result['cost']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); main(args.run.resolve())
