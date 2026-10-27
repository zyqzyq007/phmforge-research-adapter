"""Exercise fold-matched reference mounts with genuine fits; no Agent calls."""
import argparse
import json
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
import torch
from controlled_feedback import save, sha, utc
from hust_program_runtime import Runtime, OBSERVABLE


def qualify(root):
    config = json.loads((root / 'PROTOCOL368.json').read_text())
    reference = Path(config['reference_archive'])
    pilot = Path(config['pilot_legal_inputs'])
    assert sha(reference / 'SOURCE_MANIFEST.json') == config['reference_manifest_sha256']
    for name, digest in config['input_bindings'].items():
        assert sha(pilot / name) == digest
    assert not (root / 'HANDLE368.json').exists(), 'No implicit qualification restart'
    save(root / 'HANDLE368.json', dict(pid=__import__('os').getpid(), utc=utc(), state='running'))
    with np.load(pilot / 'train.npz', allow_pickle=False) as z:
        train = {k: z[k] for k in z.files}
    with np.load(pilot / 'training_queries_private.npz', allow_pickle=False) as z:
        queries = {k: z[k] for k in OBSERVABLE}
    with np.load(pilot / 'development_observations.npz', allow_pickle=False) as z:
        current = {k: z[k] for k in z.files}
    assert set(current) == set(OBSERVABLE)
    jobs = json.loads((reference / 'JOBS366.json').read_text())
    names = json.loads((reference / 'PROTOCOL366.json').read_text())['models']
    bindings, views = [], {}
    for context, job in jobs.items():
        view_archive = config.get('reuse_reference_views_archive')
        view = Path(view_archive) / 'reference_views' / context if view_archive else root / 'reference_views' / context
        views[context] = str(view)
        for name in names:
            ident = 'p00' if name == 'author_crnn' else 'ref_'+name
            suffix = '.pt' if name == 'author_crnn' else '.joblib'
            source = reference / 'models' / context / (name+suffix)
            checkpoint = torch.load(source, weights_only=True) if suffix == '.pt' else joblib.load(source)
            assert checkpoint['fitting_cells'] == job['fitting_cells']
            target = view / ident / 'full_fit/artifacts' / (name+suffix)
            if not view_archive:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            assert sha(target) == sha(source)
            bindings.append(dict(context=context, model=name, source=str(source), sha256=sha(source), fitting_cells=job['fitting_cells']))
        binding = dict(context=context, fitting_cells=job['fitting_cells'])
        if view_archive:
            assert json.loads((view / 'CONTEXT.json').read_text()) == binding
        else:
            save(view / 'CONTEXT.json', binding)
    engine = dict(input_root=str(root / 'inputs'), common_tools=str(root / 'common_tools'), reference_views=views,
        fit_seconds=120, predict_seconds=45, program_seconds=600, prediction_workers=1,
        allow_past_capacity_online_fit=True, online_optimizer_steps_max=100)
    save(root / 'ENGINE_CONFIG.json', engine)
    runtime = Runtime(root)
    runtime.deadline = time.monotonic()+config['wall_seconds']
    results = []
    for context, job in jobs.items():
        folder = root / 'exercises' / context
        folder.mkdir(parents=True)
        mask = np.isin(train['groups'], job['fitting_cells'])
        fit = {k: v[mask] for k, v in train.items()}
        input_file = folder / 'train.npz'
        np.savez_compressed(input_file, **fit)
        reused = config.get('reuse_complete_contexts', {}).get(context)
        if reused:
            reused = Path(reused)
            with np.load(reused / 'train.npz', allow_pickle=False) as z:
                assert set(z.files) == set(fit)
                for key in z.files:
                    np.testing.assert_array_equal(z[key], fit[key])
            assert sha(root / 'program/program.py') == config['reused_program_sha256']
            shutil.copytree(reused / 'fit', folder / 'fit')
            audit = json.loads((folder / 'fit/training_audit.json').read_text())
        else:
            _, audit = runtime.command(root / 'program', None, input_file, folder / 'fit', 'fit')
        assert audit['completed_estimator_fits'] == 1 and audit['optimizer_steps'] == 0
        fitting = set(job['fitting_cells'])
        if context == 'full':
            obs = {k: v[:1] for k, v in current.items()}
        else:
            idx = np.flatnonzero(np.isin(queries['groups'], job['heldout_cells']))[:1]
            assert not fitting & set(job['heldout_cells'])
            obs = {k: v[idx] for k, v in queries.items()}
        path = folder / 'query.npz'
        np.savez_compressed(path, **obs)
        _, prediction_audit = runtime.command(root / 'program', folder / 'fit/artifacts', path, folder / 'check', 'predict')
        assert prediction_audit['completed_estimator_fits'] == prediction_audit['optimizer_steps'] == 0
        actual = json.loads((folder / 'check/predictions.json').read_text())['rul']
        original = joblib.load(reference / 'models' / context / 'ridge10.joblib')['model'].predict(obs['x'])
        own = joblib.load(folder / 'fit/artifacts/ridge.joblib')['model'].predict(obs['x'])
        np.testing.assert_allclose(actual, np.maximum(own, 0), rtol=1e-9, atol=1e-6)
        results.append(dict(context=context, fitting_cells=len(fitting), real_fit_entries=int(not reused),
            real_ridge_fits=int(not reused), reused_fit_entries=int(bool(reused)),
            new_scalers=int(not reused), cold_query_processes=1, own_weight_cold_replay_passed=True,
            original_independent_refit_difference_cycles=float(np.max(np.abs(np.asarray(actual)-np.maximum(original, 0))))))
        save(root / 'PROGRESS368.json', results)
    # A full-data checkpoint cannot masquerade as a fold reference context.
    runtime.config['reference_views'] = dict(views, fold_0=views['full'])
    rejected = False
    try:
        runtime.command(root / 'program', None, root / 'exercises/fold_0/train.npz', root / 'negative/fold_0/fit', 'fit')
    except AssertionError as error:
        rejected = 'Reference checkpoint context mismatch' in str(error)
    assert rejected
    assert not (root / 'negative/fold_0/fit').exists()
    result = dict(utc=utc(), passed=True, exercises=results, checkpoint_bindings=bindings,
        mismatched_reference_rejected_before_worker=True, new_fit_entries=sum(r['real_fit_entries'] for r in results),
        reused_fit_entries=sum(r['reused_fit_entries'] for r in results),
        new_ridge_models=sum(r['real_ridge_fits'] for r in results), new_standard_scalers=sum(r['new_scalers'] for r in results),
        new_optimizer_steps=0, cold_query_processes=4,
        total_worker_processes=4+sum(r['real_fit_entries'] for r in results), new_LLM_calls=0, future_query_targets_read=False,
        final22_files_opened=0, method_comparison=False)
    save(root / 'RESULT368.json', result)
    save(root / 'TERMINAL368.json', dict(exit_code=0, passed=True, utc=utc()))
    print(json.dumps({k: v for k, v in result.items() if k not in ['exercises', 'checkpoint_bindings']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    qualify(p.parse_args().run.resolve())
