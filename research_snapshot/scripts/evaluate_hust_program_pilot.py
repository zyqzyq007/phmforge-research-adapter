"""Open the eight developmental RUL outcomes once, after joint native sealing."""
import argparse
import json
from pathlib import Path
import numpy as np
from controlled_feedback import save, sha, utc


def evaluate(root):
    read = lambda p: json.loads(Path(p).read_text())
    assert read(root / 'TERMINAL367.json')['exit_code'] == 0
    assert not (root / 'RESULT367.json').exists(), 'Never repeatedly open/tune this outcome panel'
    sealed = read(root / 'ALL_CHOICES_SEALED367.json')
    assert len(sealed['results']) == 2
    with np.load(root / 'private_inputs/discovery_truth.npz', allow_pickle=False) as z:
        actual, ids = z['y'], z['sample_ids']
    with np.load(root / 'private_inputs/REFERENCE_PREDICTIONS.npz', allow_pickle=False) as z:
        references = {k: z[k] for k in z.files}
    all_rows = []
    for row in read(root / 'SESSIONS337.json'):
        run = Path(row['path'])
        state = read(run / 'broker/STATE.json')
        programs = {}
        for ident, item in state['programs'].items():
            folder = Path(item['path'])
            report = read(folder / 'REPORT.json')
            for relative, digest in report['artifacts'].items():
                assert sha(folder / relative) == digest
            if report.get('initial_models_reused'):
                predicted = references[report['model']]
            else:
                with np.load(folder / 'development/PREDICTIONS.npz', allow_pickle=False) as z:
                    assert np.array_equal(z['sample_ids'], ids)
                    predicted = z['prediction']
            # Independent MAE reconstruction also reads each cold JSON result.
            errors = np.abs(predicted-actual)
            assert len(errors) == 8 and np.isfinite(errors).all()
            if not report.get('initial_models_reused'):
                direct = np.asarray([read(folder / 'development' / ('prediction'+str(i)) / 'predictions.json')['rul'][0]
                                     for i in range(8)])
                np.testing.assert_allclose(direct, predicted, rtol=0, atol=0)
            programs[ident] = dict(mean_cell_MAE=float(np.mean(errors)),
                cv_mean_cell_MAE=float(np.mean([c['MAE'] for f in report['cv'] for c in f['cells']])),
                predictions=predicted.tolist(), absolute_errors=errors.tolist())
        choice = state.get('submission')
        initial = programs['p00']['mean_cell_MAE']
        chosen = programs[choice['program_id']]['mean_cell_MAE'] if choice else None
        cv = min(programs, key=lambda i: (programs[i]['cv_mean_cell_MAE'], i != 'p00', i))
        best = min(programs, key=lambda i: (programs[i]['mean_cell_MAE'], i != 'p00', i))
        all_rows.append(dict(id=row['id'], native_result=read(run / 'BUILD_RESULT.json'),
            submission=choice, all_programs=programs, incumbent_MAE=initial, selected_MAE=chosen,
            realized_harm=None if chosen is None else max(0, chosen-initial),
            realized_gain=None if chosen is None else max(0, initial-chosen),
            own_available_pool_selection_loss_diagnostic=None if chosen is None else chosen-programs[best]['mean_cell_MAE'],
            best_available=best, ordinary_fixed_CV_choice=cv,
            fixed_CV_MAE=programs[cv]['mean_cell_MAE'],
            incomplete_not_assigned_a_success_score=choice is None))
    result = dict(utc=utc(), cells=8, source_clusters=1, repeated_ordinary_sessions=2,
        exploratory_only=True, independent_confirmation_scenarios=0, new_method_comparison=False,
        majority_complete=False, final22_read=False, rows=all_rows,
        independent_metric_recount_passed=True, no_score_based_retries=True)
    save(root / 'RESULT367.json', result)
    print(json.dumps(dict(episodes=[{k: r[k] for k in ['id', 'selected_MAE', 'incumbent_MAE', 'best_available', 'fixed_CV_MAE']} for r in all_rows],
                         independent_confirmation_scenarios=0, majority_complete=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    evaluate(p.parse_args().run.resolve())
