"""Independent scalar audit of frozen HUST capacity-check results; no fitting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def main(root):
    started = time.monotonic()
    read = lambda p: json.loads(p.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cfg = read(root / 'PROTOCOL423.json')
    result = read(root / 'RESULT423.json')
    sealed = read(root / 'CHOICES423.json')
    checks = 0

    def check(condition):
        nonlocal checks
        assert condition, f'check {checks + 1} failed'
        checks += 1

    def close(a, b):
        check(abs(float(a) - float(b)) <= 1e-8)

    check(digest(root / 'code/hust_capacity_checks.py') == cfg['code_sha256'])
    for name, expected in cfg['support_code_sha256'].items():
        check(digest(root / 'code' / name) == expected)
    for path, field in [('legal/observations.npz', 'legal_observation_sha256'),
                        ('private/targets.npz', 'private_targets_sha256')]:
        check(digest(root / path) == cfg[field])
    for context in cfg['contexts'].values():
        check(digest(Path(context['checkpoint'])) == context['checkpoint_sha256'])
    check(sealed['utc'] < result['utc'])
    with np.load(root / 'private/targets.npz', allow_pickle=False) as source:
        truth = {k: source[k] for k in source.files}
    with np.load(root / 'legal/observations.npz', allow_pickle=False) as source:
        observed = {k: source[k] for k in source.files}
    check('y' not in observed and 'cap' not in observed)
    check(np.array_equal(observed['sample_ids'], truth['sample_ids']))
    rows = [read(root / 'reports' / f'q{i:02d}.json') for i in range(cfg['queries'])]
    check(len(rows) == 55 and len(set(observed['groups'])) == 55)
    for i, row in enumerate(rows):
        check(digest(root / 'reports' / f'q{i:02d}.json') == sealed['report_hashes'][str(i)])
        check(row['decisions'] == sealed['decisions'][str(i)])
        context = cfg['contexts'][row['context']]
        check(row['source_group'] not in context['fitting_cells'])

    cases = []
    recomputed = {}
    for role, expected_size in [('training_oof', 47), ('development', 8)]:
        indices = np.flatnonzero(truth['roles'] == role)
        check(len(indices) == expected_size)
        targets = truth['y'][indices].astype(float)
        errors = {p: np.abs(np.array([rows[i]['decisions'][p]['rul'] for i in indices]) - targets)
                  for p in cfg['policies']}
        oracle = np.array([min(abs(s['rul'] - truth['y'][i])
                              for s in rows[i]['paths']['6']['snapshots']) for i in indices])
        opportunity = errors['keep'] - oracle
        eligible = opportunity > 1e-6
        recomputed[role] = {}
        for policy, loss in errors.items():
            metrics = {'MAE': float(loss.mean())}
            if policy in ['keep', 'self6', 'hold3']:
                metrics.update(selection_loss=float((loss - oracle).mean()),
                    harmful_update=float(np.maximum(loss - errors['keep'], 0).mean()),
                    gain_available_cells=int(eligible.sum()),
                    gain_capture=float(np.mean(np.maximum(errors['keep'][eligible] - loss[eligible], 0)
                                               / opportunity[eligible])) if eligible.any() else None)
            for key, value in metrics.items():
                actual = result['cohorts'][role][policy][key]
                check(actual is None) if value is None else close(actual, value)
            recomputed[role][policy] = metrics
        for i in indices:
            row = rows[i]
            selected = next(s for s in row['paths']['6']['snapshots']
                            if s['step'] == row['decisions']['hold3']['step'])
            base_error = abs(row['baseline']['rul'] - truth['y'][i])
            updated_error = abs(selected['rul'] - truth['y'][i])
            cases.append(dict(role=role, sample_id=row['sample_id'],
                source_group=row['source_group'], selected_step=selected['step'],
                before_held_capacity_RMSE_mAh=float(np.sqrt(row['baseline']['hold3_MSE']) * 1000),
                after_held_capacity_RMSE_mAh=float(np.sqrt(selected['hold3_MSE']) * 1000),
                before_RUL_absolute_error=float(base_error), after_RUL_absolute_error=float(updated_error),
                RUL_error_increase=float(updated_error - base_error)))
    selected = min(cfg['policies'], key=lambda p: (recomputed['training_oof'][p]['MAE'], cfg['policies'].index(p)))
    check(selected == result['source_CV_selected_policy'])
    a, b = recomputed['development']['hold3'], recomputed['development']['self6']
    gate = (a['MAE'] < b['MAE'] - 1e-6 and a['harmful_update'] <= b['harmful_update'] + 1e-6
            and a['gain_capture'] is not None and a['gain_capture'] >= b['gain_capture'] - 1e-12)
    check(gate == result['primary_gate'])
    report = dict(utc=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
        seconds=time.monotonic() - started, new_fits=0, new_model_forwards=0,
        primary_gate=gate, source_CV_policy=selected, recomputed=recomputed,
        illustrative_cases=[max((x for x in cases if x['role'] == role), key=lambda x: x['RUL_error_increase'])
                            for role in ['training_oof', 'development']],
        code_sha256=digest(Path(__file__)), result_sha256=digest(root / 'RESULT423.json'))
    (root / 'SUPPLEMENT_AUDIT423.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['passed', 'checks', 'seconds', 'primary_gate', 'illustrative_cases']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
