"""Independent record-vote, cost, source, and fitted-artifact audit for A402."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import joblib
import numpy as np

from audit_program_evidence import read, save, sha, utc


def array_file(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def main(root):
    terminal = read(root / 'TERMINAL402.json')
    assert terminal['status'] == 'completed'
    result = read(root / 'RESULT402.json')
    cfg = read(root / 'PROTOCOL402.json')
    checks = 0

    def check(value, message):
        nonlocal checks
        assert value, message
        checks += 1

    for name, digest in read(root / 'CODE_FREEZE402.json')['files'].items():
        check(sha(root / name) == digest, 'Frozen source ' + name)
    data = {role: array_file(Path(cfg['data_archive']) / 'qualification' / (role + '.npz'))
            for role in ['train', 'development', 'independent_check', 'final']}

    def verify_predictions(folder, source, reported=None):
        packed = array_file(folder / 'PREDICTIONS.npz')
        pred = dict(zip(packed['sample_ids'].tolist(), packed['prediction'].tolist()))
        check(len(pred) == len(source['sample_ids']), str(folder) + ' sample count')
        check(set(pred) == set(source['sample_ids']), str(folder) + ' identifiers')
        cold = {}
        for p in sorted(folder.glob('prediction*/predictions.json')):
            entries = read(p)['predictions']
            for row in entries:
                check(row['sample_id'] not in cold, 'unique raw cold prediction')
                cold[row['sample_id']] = row['predicted_label']
            input_file = folder / ('input' + p.parent.name.removeprefix('prediction') + '.npz')
            check('labels' not in array_file(input_file), 'No labels in inference input')
        # Opaque-ID check retains original IDs only in the host packed artifact.
        if folder.name != 'identity_check':
            check(cold == pred, 'Raw JSON and packed outputs agree')
        rows = {}
        for group in sorted(set(source['groups'].tolist())):
            ids = [str(s) for s, g in zip(source['sample_ids'], source['groups']) if g == group]
            ys = {int(y) for y, g in zip(source['labels'], source['groups']) if g == group}
            check(len(ys) == 1, 'One author label per acquisition')
            y = next(iter(ys)); counts = Counter(pred[s] for s in ids)
            yhat = min(counts, key=lambda label: (-counts[label], label))
            rows[group] = (y, int(yhat), sum(pred[s] == y for s in ids), len(ids))
        error = Fraction(sum(y != p for y, p, _, _ in rows.values()), len(rows))
        if reported is not None:
            check(float(error) == reported['record_error'], 'Record error')
            actual = {row['group']: (row['actual'], row['predicted'], row['window_correct'], row['windows'])
                      for row in reported['records']}
            check(rows == actual, 'All reported per-record details')
            n = len(source['sample_ids'])
            werr = Fraction(sum(pred[str(s)] != int(y) for s, y in zip(source['sample_ids'], source['labels'])), n)
            check(float(werr) == reported['window_error'], 'Window error')
            confusion = [[sum(y == i and p == j for y, p, _, _ in rows.values()) for j in range(4)] for i in range(4)]
            check(confusion == reported['confusion'], 'Confusion matrix')
        return error, rows, pred

    evidence, final = {}, {}
    for name in cfg['variants']:
        path = root / 'programs' / name
        report = read(path / 'REPORT.json')
        cv_errors = []
        for condition in [15, 35]:
            train = data['train']; mask = train['condition_id'] == condition
            heldout = {k: v[mask] for k, v in train.items()}
            scored = next(x for x in report['cv'] if x['condition_hz'] == condition)
            err, _, _ = verify_predictions(path / ('cv_' + str(condition)) / 'check', heldout, scored)
            cv_errors.append(err)
        dev_error, _, dev = verify_predictions(path / 'development', data['development'], report['development'])
        _, _, renamed = verify_predictions(path / 'identity_check', data['development'])
        check(dev == renamed, 'Opaque-ID invariance')
        check_error, _, _ = verify_predictions(path / 'independent_check', data['independent_check'])
        final_error, rows, original = verify_predictions(path / 'final', data['final'], result['final'][name])
        _, _, repeated = verify_predictions(path / 'final_repeat', data['final'])
        check(original == repeated, 'Independent cold final repeat')
        evidence[name] = dict(cv=sum(cv_errors) / len(cv_errors), dev=dev_error, full=(8 * dev_error + 4 * check_error) / 12)
        final[name] = dict(error=final_error, records=rows)
        for fit, expected_rows in [('cv_15/fit', 36), ('cv_35/fit', 36), ('full_fit', 72)]:
            audit = read(path / fit / 'training_audit.json')
            check(audit['completed_estimator_fits'] == 1, 'One actual LDA fit')
            model = joblib.load(path / fit / 'artifacts/model.joblib')
            classifier = model.steps[-1][1]
            check(np.isfinite(classifier.coef_).all() and bool(np.any(classifier.coef_ != 0)), 'Finite learned LDA coefficients')
            check(model.steps[0][1].n_samples_seen_ == expected_rows, 'Actual standardizer training row count')
            check(model.steps[1][1].n_samples_ == expected_rows, 'Actual PCA training row count')
            check(set(classifier.classes_) == {0, 1, 2, 3}, 'All four original classes trained')
    edits = cfg['edit_count']
    expected_choices = dict(keep='original', repair_pair='pair', drop_geometry='geometry', repair_both='both',
        fixed_mean_cv=min(evidence, key=lambda n: (evidence[n]['cv'], evidence[n]['dev'], edits[n], n)),
        fixed_development=min(evidence, key=lambda n: (evidence[n]['dev'], evidence[n]['cv'], edits[n], n)),
        fixed_full_validation=min(evidence, key=lambda n: (evidence[n]['full'], edits[n], n)))
    check(expected_choices == result['choices'] == read(root / 'SEALED_DECISIONS402.json')['choices'], 'All pre-final selectors')
    best = min(x['error'] for x in final.values())
    base = final['original']; opportunity = base['error'] - best
    for name, item in final.items():
        harm = Fraction(sum(base['records'][g][0] == base['records'][g][1] and y != p for g, (y, p, _, _) in item['records'].items()), len(item['records']))
        gain = Fraction(sum(base['records'][g][0] != base['records'][g][1] and y == p for g, (y, p, _, _) in item['records'].items()), len(item['records']))
        for key, expected in [('paired_harm', harm), ('paired_gain', gain), ('selection_loss', item['error'] - best), ('net_harm', max(Fraction(0), item['error'] - base['error']))]:
            check(float(expected) == result['final'][name][key], key)
        capture = float(max(Fraction(0), base['error'] - item['error']) / opportunity) if opportunity else None
        check(capture == result['final'][name]['gain_capture'], 'Gain capture')
    old = read(Path(cfg['program_archive']) / 'RESULT399.json')['all_program_results']['ordinary_1__p01']
    old_rows = {x['group']: (x['actual'], x['predicted'], x['window_correct'], x['windows']) for x in old['records']}
    check(old_rows == final['original']['records'], 'Original final record/window result reproduces archive')
    audits = [read(p) for p in root.glob('programs/*/**/training_audit.json')]
    check(sum(x['mode'] == 'fit' for x in audits) == result['cost']['fit_entries'] == 12, 'All12 fit entries')
    check(sum(x['completed_estimator_fits'] for x in audits) == result['cost']['classifier_fits'] == 12, 'All12 completed learning calls')
    check(sum(x['mode'] == 'predict' for x in audits) == result['cost']['cold_prediction_processes'] == 176, 'All176 cold classifier calls')
    check(all(x['optimizer_steps'] == 0 for x in audits), 'No NN steps claimed')
    check(all(x['completed_estimator_fits'] == 0 for x in audits if x['mode'] == 'predict'), 'No learning in prediction')
    result = dict(utc=utc(), passed=True, checks=checks, finite_learned_models=12,
                  original_final_reproduced=True, new_LLM_calls=0, new_PHM_fits=0,
                  new_PHM_forward_calls=0, independent_confirmation=False,
                  scope='Independent Counter/Fraction recount from raw per-window JSON, independent cost census and learned artifact inspection.')
    save(root / 'INDEPENDENT_AUDIT402.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); main(args.run.resolve())
