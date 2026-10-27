"""Recover existing, genuinely cross-fitted program forecasts at the same stage."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from author_net import CRNN
from controlled_feedback import save, sha, utc


def qualify(root):
    cfg = json.loads((root/'PROTOCOL371.json').read_text())
    source, reference = Path(cfg['source_archive']), Path(cfg['reference_archive'])
    for folder, digest in [(source, cfg['source_manifest_sha256']), (reference, cfg['reference_manifest_sha256'])]:
        assert json.loads((folder/'ARCHIVE_VERIFIED.json').read_text())['source_cleanup'] == 'complete'
        assert sha(folder/'SOURCE_MANIFEST.json') == digest
    assert not (root/'HANDLE371.json').exists(), 'Never implicitly restart profile recovery'
    save(root/'HANDLE371.json', dict(pid=__import__('os').getpid(), utc=utc(), state='running'))
    started = time.monotonic()
    torch.set_num_threads(2)
    with np.load(source/'private_inputs/training_queries_private.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    assert len(data['y']) == len(set(data['groups'])) == 47 and np.all(data['cycle'] == 500)
    results = json.loads((source/'RESULT367.json').read_text())
    names = ['p00']
    for row in results['rows']:
        for ident in row['all_programs']:
            name = ident if ident == 'p00' or ident.startswith('ref_') else row['id']+'__'+ident
            if name not in names:
                names.append(name)
    assert names == cfg['program_order'] and len(names) == 11
    prediction = np.full((47, 11), np.nan)
    development = np.full((8, 11), np.nan)
    expected = np.full_like(prediction, np.nan)
    bindings, calls = [], 0
    jobs = json.loads((reference/'JOBS366.json').read_text())
    assert jobs == json.loads((source/'JOBS366.json').read_text())
    for row in results['rows']:
        for ident, metrics in row['all_programs'].items():
            shared = ident == 'p00' or ident.startswith('ref_')
            name = ident if shared else row['id']+'__'+ident
            column = names.index(name)
            if np.isfinite(development[:, column]).all():
                np.testing.assert_array_equal(development[:, column], metrics['predictions'])
                continue
            development[:, column] = metrics['predictions']
            folder = source/'reference'/ident if shared else source/'subjects'/row['id']/'programs'/ident
            report = json.loads((folder/'REPORT.json').read_text())
            errors = {c['cell']: c['MAE'] for f in report['cv'] for c in f['cells']}
            expected[:, column] = [errors[str(g)] for g in data['groups']]
            if shared:
                model_name = report['model']
                for context, job in jobs.items():
                    if context == 'full':
                        continue
                    mask = np.isin(data['groups'], job['heldout_cells'])
                    assert not set(data['groups'][mask]) & set(job['fitting_cells'])
                    suffix = '.pt' if model_name == 'author_crnn' else '.joblib'
                    path = reference/'models'/context/(model_name+suffix)
                    checkpoint = torch.load(path, weights_only=True) if suffix == '.pt' else joblib.load(path)
                    assert checkpoint['fitting_cells'] == job['fitting_cells']
                    if suffix == '.pt':
                        net = CRNN(ni=100, nc=4, no=64, nh=64)
                        net.load_state_dict(checkpoint['state_dict'])
                        net.eval()
                        with torch.no_grad():
                            r, _ = net(torch.from_numpy(data['nn'][mask]))
                            p = r.reshape(-1).numpy()*3000
                    else:
                        p = checkpoint['model'].predict(data['x'][mask])
                        if model_name == 'lifetime_ridge10':
                            p -= data['age'][mask]
                    prediction[mask, column] = np.maximum(p, 0)
                    calls += 1
                    bindings.append(dict(program=name, context=context, path=str(path), sha256=sha(path),
                        fitting_cells=job['fitting_cells'], heldout_cells=job['heldout_cells']))
            else:
                by_id = {}
                for context in ['fold_0', 'fold_1', 'fold_2']:
                    path = folder/context/'check/PREDICTIONS.npz'
                    with np.load(path, allow_pickle=False) as z:
                        for ident, p in zip(z['sample_ids'], z['prediction']):
                            assert str(ident) not in by_id
                            by_id[str(ident)] = float(p)
                    bindings.append(dict(program=name, context=context, path=str(path), sha256=sha(path), existing_cold_prediction=True))
                assert set(by_id) == set(data['sample_ids'].tolist())
                prediction[:, column] = [by_id[str(i)] for i in data['sample_ids']]
            assert time.monotonic()-started < cfg['wall_seconds']
    actual = np.abs(prediction-data['y'][:, None])
    assert np.isfinite(prediction).all() and np.isfinite(development).all() and calls == 21
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-3)
    path = root/'PROFILES371.npz'
    np.savez_compressed(path, programs=np.asarray(names), train_groups=data['groups'],
        train_sample_ids=data['sample_ids'], cv_prediction=prediction,
        cv_errors=actual, development_prediction=development)
    summary = dict(utc=utc(), passed=True, frozen_sources_and_stage_verified=True,
        programs=names, training_physical_cells=47, development_cells=8,
        actual_new_fixed_model_predict_calls=calls, reused_authored_cv_prediction_cells=188,
        source_error_values_checked=517, max_absolute_error_replay_difference=float(np.max(np.abs(actual-expected))),
        new_PHM_fits=0, new_LLM_calls=0, final22_files_opened=0,
        profile_cache=str(path), profile_cache_sha256=sha(path), bindings=bindings,
        delta_feature_rank=int(np.linalg.matrix_rank(prediction[:, 1:]-prediction[:, :1])),
        information_scope='Deterministicforecastprofiles,notanindependentmeasurementornewmethod. All8areopeneddevelopmentunits.')
    save(root/'QUALIFICATION371.json', summary)
    save(root/'TERMINAL371.json', dict(exit_code=0, utc=utc()))
    print(json.dumps({k: v for k, v in summary.items() if k not in ['bindings', 'programs']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    qualify(p.parse_args().run.resolve())
