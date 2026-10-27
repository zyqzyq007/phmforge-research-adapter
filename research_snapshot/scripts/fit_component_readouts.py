"""One frozen supervised readout intervention on archived real component outputs.

No base retraining, synthetic inputs, or in-sample meta-score called validation.
Fit and cold inference use separate label-isolated namespaces.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import numpy as np

from controlled_feedback import save, sha, utc
from program_revision_runtime import DEPS, PYTHON, load


def read(p):
    return json.loads(Path(p).read_text())


def describe(spec, raw):
    kind = spec['kind']
    if kind == 'success':
        x = np.asarray(raw['base_lifetimes'][0], float)
        age = raw['age'][0]
        weight, other = {'stack': (.75, raw['consensus'][0]),
                         'median': (.4, raw['median'][0])}.get(
                             raw['choice'], (.55, raw['consensus'][0]))
        inner = (1 - weight) * other
        outer = .08 * max(raw['hgb'][0], 0) + .02 * max(raw['extra_trees'][0], 0)
        params = [weight, inner, .9, outer, 2800.]
        old = raw['stack'][0]
        original = raw['variants']['original'][0]
    else:
        values = {r['name']: float(np.asarray(r['prediction']).ravel()[0])
                  for r in raw['calls']}
        x = np.asarray([values[n] for n in spec['components']])
        age = raw['age']
        old = .8 * np.median(x) + .2 * np.mean(x)
        if kind == 'dual':
            t = np.array([values[n] for n in ['trend_ridge', 'trend_et', 'trend_hgb']])
            outer = (.23 * max(.7 * np.median(t) + .3 * t.mean() - age, 0)
                     + .10 * max(values['all_hgb'], 0) + .02 * max(values['all_et'], 0))
            params = [1., 0., .65, outer, 0.]
        elif kind == 'compact':
            direct = max(.58 * values['direct_ridge'] + .42 * values['direct_extra'] - age, 0)
            teacher = max(.72 * values['all_hgb'] + .28 * values['all_extra'], 0)
            params = [1., 0., .76, .06 * direct + .18 * teacher, 0.]
        elif kind == 'anchor':
            old = .5 * np.median(x) + .27 * x.mean() + .23 * np.median(x[-3:])
            params = [1., 0., 1., 0., 0.]
        else:
            raise ValueError(kind)
        original = raw['original'][0]
    reproduced = compose(np.array([old]), np.array([age]), np.array([params]))[0]
    assert np.isclose(reproduced, original, rtol=1e-9, atol=1e-6)
    return x, float(age), np.asarray(params), float(original)


def compose(lifetime, age, p):
    result = p[:, 2] * np.maximum(p[:, 0] * lifetime + p[:, 1] - age, 0) + p[:, 3]
    return np.where(p[:, 4] > 0, np.minimum(result, p[:, 4]), result)


def prepare(root):
    cfg = read(root / 'PROTOCOL458.json')
    archive = Path(cfg['source_archive'])
    assert sha(archive / 'SOURCE_MANIFEST.json') == cfg['source_manifest_sha256']
    members = {r['path']: r for r in cfg['resources']}
    restored = set()
    with tarfile.open(archive / 'RUN_CONTENTS.tar.gz', 'r|gz') as stream:
        for entry in stream:
            name = entry.name.removeprefix('./')
            if name not in members:
                continue
            target = root / 'reused' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists() and entry.isfile()
            target.write_bytes(stream.extractfile(entry).read())
            assert target.stat().st_size == members[name]['bytes']
            assert sha(target) == members[name]['sha256']
            restored.add(name)
    assert restored == set(members)
    assert sha(cfg['train']) == cfg['train_sha256']
    train = load(cfg['train'])
    lifetime = {}
    for group in np.unique(train['groups']):
        v = (train['y'] + train['age'])[train['groups'] == group]
        assert np.allclose(v, v[0], rtol=0, atol=1e-8)
        lifetime[str(group)] = float(v[0])
    assert len(lifetime) == cfg['source_cells']
    metadata = []
    for spec in cfg['programs']:
        success = spec['kind'] == 'success'
        tasks = read(root / 'reused' / ('reused/TASKS446.json' if success else 'TASKS.json'))
        if not success:
            tasks = [t for t in tasks if t['program'] == spec['name']]
        source, query = [], []
        for task in sorted(tasks, key=lambda t: t['sample_id']):
            path = root / 'reused' / ('reused/predictions' if success else 'predictions')
            path /= task['id']
            path /= 'COMPONENTS.json' if success else 'COMPONENT_TRACE.json'
            x, age, params, original = describe(spec, read(path))
            row = dict(task=task, x=x, age=age, params=params, original=original,
                       trace=str(path.relative_to(root)), trace_sha256=sha(path))
            (query if task['context'] == 'full' else source).append(row)
        assert len(source) == 47 and len(query) == 8
        assert {r['task']['group'] for r in source} == set(lifetime)
        legal = root / 'legal' / (spec['name'] + '.npz')
        legal.parent.mkdir(exist_ok=True)
        np.savez_compressed(legal, x=np.stack([r['x'] for r in source]),
            y=np.array([lifetime[r['task']['group']] for r in source]),
            sample_ids=np.array([r['task']['sample_id'] for r in source]))
        inp = root / 'queries' / (spec['name'] + '.npz')
        inp.parent.mkdir(exist_ok=True)
        np.savez_compressed(inp, x=np.stack([r['x'] for r in query]),
            age=np.array([r['age'] for r in query]), params=np.stack([r['params'] for r in query]),
            sample_ids=np.array([r['task']['sample_id'] for r in query]))
        metadata.append(dict(**spec, legal=str(legal.relative_to(root)), query=str(inp.relative_to(root)),
            rows=[dict(sample_id=r['task']['sample_id'], context=r['task']['context'],
                       group=r['task']['group'], trace=r['trace'], trace_sha256=r['trace_sha256'],
                       original=r['original']) for r in source + query]))
    save(root / 'CASES458.json', metadata)
    save(root / 'PREPARED458.json', dict(passed=True, utc=utc(), cached_real_rows=220,
        source_groups=47, meta_training_rows_per_program=47, query_rows_per_program=8,
        external_targets_loaded=False, original_outputs_reproduced=220))
    paths = [root / 'PROTOCOL458.json', root / 'CASES458.json',
             *sorted((root / 'legal').glob('*.npz')), *sorted((root / 'queries').glob('*.npz')),
             *sorted((root / 'code').glob('*.py'))]
    save(root / 'LAUNCH_FREEZE458.json', {str(p.relative_to(root)): sha(p) for p in paths})
    print(json.dumps(read(root / 'PREPARED458.json')), flush=True)


def worker(root, mode, input_path):
    import joblib
    assert not Path('/data').exists() and not Path('/srv').exists()
    assert not Path('/home/jupyter-yyf-zyq/.bashrc').exists()
    data = load(input_path)
    if mode == 'fit':
        from sklearn.linear_model import Ridge
        assert set(data) == {'x', 'y', 'sample_ids'} and len(data['y']) == 47
        cfg = read('/protocol.json')['fit']
        model = Ridge(**{k: cfg[k] for k in ['alpha', 'positive', 'fit_intercept', 'solver', 'tol', 'max_iter']})
        started = time.monotonic()
        model.fit(data['x'], data['y'])
        fitted = model.predict(data['x'])
        assert np.isfinite(fitted).all() and np.any(model.coef_ != 0)
        joblib.dump(model, root / 'readout.joblib')
        save(root / 'MODEL.json', dict(utc=utc(), fit_seconds=time.monotonic() - started,
            coefficient=model.coef_.tolist(), intercept=float(model.intercept_),
            parameter_norm=float(np.linalg.norm(model.coef_)), fit_calls=1,
            input_sha256=sha(input_path), input_groups=47,
            in_sample_meta_MAE=float(np.abs(fitted - data['y']).mean()),
            in_sample_meta_MAE_is_validation=False, model_sha256=sha(root / 'readout.joblib'),
            base_fit_calls=0, external_data_visible=False))
    else:
        assert set(data) == {'x', 'age', 'params', 'sample_ids'} and len(data['x']) == 8
        model = joblib.load('/model.joblib')
        lifetime = model.predict(data['x'])
        prediction = compose(lifetime, data['age'], data['params'])
        assert np.isfinite(prediction).all() and np.all(prediction >= 0)
        save(root / 'PREDICTIONS.json', dict(utc=utc(), sample_ids=data['sample_ids'].tolist(),
            lifetime=lifetime.tolist(), rul=prediction.tolist(), input_sha256=sha(input_path),
            model_sha256=sha('/model.joblib'), predict_calls=1, fit_calls=0,
            base_predict_calls=0, targets_visible=False))


def launch(root, mode):
    cfg = read(root / 'PROTOCOL458.json')
    cases = read(root / 'CASES458.json')
    for p, digest in read(root / 'LAUNCH_FREEZE458.json').items():
        assert sha(root / p) == digest
    if mode == 'predict':
        assert read(root / 'FIT_SEALED458.json')['passed']
    started = time.monotonic()
    def one(case):
        out = root / mode / case['name']
        out.mkdir(parents=True, exist_ok=False)
        (out / 'tmp/mpl').mkdir(parents=True)
        cmd = ['bwrap', '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts',
               '--unshare-net', '--die-with-parent', '--new-session']
        for system in ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc']:
            if Path(system).exists():
                cmd += ['--ro-bind', system, system]
        mounts = [(PYTHON, '/opt/python'), (DEPS, '/opt/deps'), (root / 'code', '/code'),
                  (root / case['legal' if mode == 'fit' else 'query'], '/input.npz'),
                  (root / 'PROTOCOL458.json', '/protocol.json')]
        if mode == 'predict':
            model = root / 'fit' / case['name'] / 'readout.joblib'
            assert sha(model) == read(model.with_name('MODEL.json'))['model_sha256']
            mounts.append((model, '/model.joblib'))
        for source, target in mounts:
            cmd += ['--ro-bind', str(source), target]
        cmd += ['--bind', str(out), '/output', '--proc', '/proc', '--dev', '/dev', '--chdir',
                '/code', '--remount-ro', '/', '/opt/python/bin/python3.12', '-B',
                '/code/fit_component_readouts.py', '--run', '/output', '--stage', 'worker',
                '--mode', mode, '--input', '/input.npz']
        env = dict(PATH='/opt/python/bin:/usr/bin:/bin', PYTHONPATH='/opt/deps:/code',
            PYTHONDONTWRITEBYTECODE='1', TMPDIR='/output/tmp', TMP='/output/tmp',
            TEMP='/output/tmp', MPLCONFIGDIR='/output/tmp/mpl', OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=cfg['fit_seconds' if mode == 'fit' else 'predict_seconds'])
        record = dict(program=case['name'], returncode=p.returncode, command=cmd,
                      stdout=p.stdout, stderr=p.stderr, utc=utc())
        save(out / 'PROCESS.json', record)
        return record
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        rows = list(pool.map(one, cases))
    passed = all(r['returncode'] == 0 for r in rows)
    save(root / (mode.upper() + '_SEALED458.json'), dict(utc=utc(), passed=passed,
        records=rows, seconds=time.monotonic() - started, external_targets_joined=False))
    print(json.dumps(dict(stage=mode, passed=passed, seconds=time.monotonic() - started,
                          processes=len(rows))), flush=True)
    assert passed, 'Retain failures; no automatic retry'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'fit', 'predict', 'worker'], required=True)
    p.add_argument('--mode', choices=['fit', 'predict'])
    p.add_argument('--input', type=Path)
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args.run.resolve())
    elif args.stage == 'worker':
        worker(args.run, args.mode, args.input)
    else:
        launch(args.run.resolve(), args.stage)
