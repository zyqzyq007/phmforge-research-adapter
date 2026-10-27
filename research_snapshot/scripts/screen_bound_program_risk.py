"""A370: contextual risk prototypes, with eight-cell outer holdout and controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from controlled_feedback import save, sha, utc


def features(d):
    cap = np.asarray(d['past_cap'], dtype=float)
    return np.column_stack([d['rates'], cap[:, -1]/1000, (cap[:, -1]-cap[:, 0])/24])


def kernel(a, b, bandwidth):
    return np.exp(-np.sum((a[:, None]-b[None, :])**2, axis=2)/(2*bandwidth**2))


def run(root):
    cfg = json.loads((root/'PROTOCOL370.json').read_text())
    source = Path(cfg['source_archive'])
    assert sha(source/'SOURCE_MANIFEST.json') == cfg['source_manifest_sha256']
    result = json.loads((source/'RESULT367.json').read_text())
    with np.load(source/'private_inputs/training_queries_private.npz', allow_pickle=False) as z:
        train_groups, z_train = z['groups'], features(z)
        raw_train = z['x']
    with np.load(source/'private_inputs/development_observations.npz', allow_pickle=False) as z:
        z_dev = features(z)
        raw_dev = z['x']
    programs, cv_columns, dev_columns, bindings = [], [], [], []
    for row in result['rows']:
        for ident, metrics in row['all_programs'].items():
            shared = ident == 'p00' or ident.startswith('ref_')
            name = ident if shared else row['id']+'__'+ident
            if name in programs:
                np.testing.assert_array_equal(metrics['absolute_errors'], dev_columns[programs.index(name)])
                continue
            report_path = source/'reference'/ident/'REPORT.json' if shared else source/'subjects'/row['id']/'programs'/ident/'REPORT.json'
            report = json.loads(report_path.read_text())
            values = {c['cell']: c['MAE'] for fold in report['cv'] for c in fold['cells']}
            assert set(values) == set(train_groups.tolist())
            programs.append(name)
            cv_columns.append([values[str(g)] for g in train_groups])
            dev_columns.append(metrics['absolute_errors'])
            bindings.append(dict(program=name, report=str(report_path), sha256=sha(report_path), artifacts=report['artifacts']))
    assert len(programs) == 11
    order = [programs.index('p00')]+[j for j, name in enumerate(programs) if name != 'p00']
    programs = [programs[j] for j in order]
    cv_columns, dev_columns = [cv_columns[j] for j in order], [dev_columns[j] for j in order]
    bindings = [bindings[j] for j in order]
    cv, dev = np.asarray(cv_columns).T, np.asarray(dev_columns).T
    space = cfg.get('feature_space', 'physical')
    if space == 'raw_input':
        z_train, z_dev = raw_train.astype(float), raw_dev.astype(float)
    elif space in ['prediction_change', 'physical_and_change']:
        path = Path(cfg['profile_cache'])
        assert sha(path) == cfg['profile_cache_sha256']
        with np.load(path, allow_pickle=False) as data:
            assert data['programs'].tolist() == programs
            assert data['train_groups'].tolist() == train_groups.tolist()
            np.testing.assert_allclose(data['cv_errors'], cv, rtol=1e-6, atol=1e-3)
            left, right = data['cv_prediction'], data['development_prediction']
            change_train, change_dev = left[:, 1:]-left[:, :1], right[:, 1:]-right[:, :1]
        if space == 'prediction_change':
            z_train, z_dev = change_train, change_dev
        else:
            z_train, z_dev = np.column_stack([z_train, change_train]), np.column_stack([z_dev, change_dev])
    else:
        assert space == 'physical'
    center, scale = z_train.mean(axis=0), z_train.std(axis=0)
    scale = np.maximum(scale, 1e-8)
    z_train, z_dev = (z_train-center)/scale, (z_dev-center)/scale
    distances = np.sqrt(np.sum((z_train[:, None]-z_train[None, :])**2, axis=2))
    bandwidth = float(np.median(distances[distances > 0]))
    local = kernel(z_dev, z_train, bandwidth)
    local = local/local.sum(axis=1, keepdims=True)
    prior = local@cv
    kd = kernel(z_dev, z_dev, bandwidth)
    rows, risks = [], []
    for held in range(8):
        available = np.arange(8) != held
        observed, prior_train = dev[available], prior[available]
        gram = kd[available][:, available]+np.eye(7)*cfg['ridge_lambda']
        query = kd[held:held+1, available]
        residual = observed-prior_train
        bias = residual.mean(axis=0)
        risk = dict(global_CV=cv.mean(axis=0), local_CV=prior[held],
            full_development_mean=observed.mean(axis=0),
            full_development_nearest=observed[np.argmin(np.sum((z_dev[available]-z_dev[held])**2, axis=1))],
            full_development_kernel=(query@observed/query.sum()).ravel(),
            full_development_GP=(observed.mean(axis=0)+query@np.linalg.solve(gram, observed-observed.mean(axis=0))).ravel(),
            R12a_global_context_correction=prior[held]+bias,
            R12b_local_context_correction=(prior[held]+query@np.linalg.solve(gram, residual)).ravel(),
            R12c_centered_context_correction=(prior[held]+bias+query@np.linalg.solve(gram, residual-bias)).ravel())
        chosen = {name: int(np.argmin(np.maximum(values, 0))) for name, values in risk.items()}
        oracle = float(dev[held].min())
        anchor = float(dev[held, programs.index('p00')])
        rows.append(dict(held_cell=held, fitting_development_cells=np.flatnonzero(available).tolist(),
            source_CV_cells=47, oracle_error=oracle, incumbent_error=anchor,
            choices={name: dict(program=programs[j], absolute_error=float(dev[held, j]),
                selection_loss=float(dev[held, j]-oracle), cell_harm=max(0, float(dev[held, j]-anchor)),
                cell_gain=max(0, float(anchor-dev[held, j]))) for name, j in chosen.items()}))
        risks.append({name: values.tolist() for name, values in risk.items()})
    aggregate = {name: {metric: float(np.mean([row['choices'][name][metric] for row in rows]))
                       for metric in ['absolute_error', 'selection_loss', 'cell_harm', 'cell_gain']}
                 for name in rows[0]['choices']}
    opportunity = sum(max(0, row['incumbent_error']-row['oracle_error']) for row in rows)
    for name, values in aggregate.items():
        values['gain_capture'] = sum(row['choices'][name]['cell_gain'] for row in rows)/opportunity if opportunity else None
    controls = [k for k in aggregate if not k.startswith('R12')]
    best_control = min(controls, key=lambda k: aggregate[k]['absolute_error'])
    # Numerical discovery gate, never a claim of Agent or unseen superiority.
    advance = {k: (v['absolute_error'] < aggregate[best_control]['absolute_error']-1e-9 and
                   v['cell_harm'] <= aggregate[best_control]['cell_harm']+1e-9 and
                   v['cell_gain'] >= aggregate[best_control]['cell_gain']-1e-9)
               for k, v in aggregate.items() if k.startswith('R12')}
    summary = dict(utc=utc(), programs=programs, rows=rows, risk_predictions=risks, aggregate=aggregate,
        best_fixed_control=best_control, numerical_advance=advance, bindings=bindings,
        feature_space=space, feature_dimensions=z_train.shape[1],
        shared_scaler_fit_cells=47, kernel_bandwidth=bandwidth, learned_multioutput_risk_functions=32,
        kernel_linear_system_fits=24, global_residual_bias_vector_fits=8,
        fit_count_scope='R12a/b/candGPcontrol,8outerholds;simpleempiricalcontrolsareseparate.NotnewPHMtaskmodeltraining.',
        new_PHM_fits=0, new_PHM_predictions=0, new_LLM_calls=0,
        all8_outcomes_seen_before_family_design=True, independent_confirmation_scenarios=0,
        physical_development_cells=8, source_clusters=1, ordinary_Agent_comparison=False, novelty_confirmed=False,
        final22_opened=False, majority_complete=False)
    save(root/'RESULT370.json', summary)
    save(root/'TERMINAL370.json', dict(exit_code=0, utc=utc()))
    print(json.dumps(dict(aggregate=aggregate, best_control=best_control, numerical_advance=advance)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    run(p.parse_args().run.resolve())
