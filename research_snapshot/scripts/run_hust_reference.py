"""Prepare legal arrays, run four real fits, then verify each in a fresh process."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from sklearn.model_selection import GroupKFold
from hust_window_features import cell_examples

read = lambda p: json.loads(Path(p).read_text())
save = lambda p, d: Path(p).write_text(json.dumps(d, indent=2) + '\n')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.now(timezone.utc).isoformat()


def prepare(root):
    config = read(root / 'PROTOCOL366.json')
    source = Path(config['source_archive'])
    assert sha(source / 'SOURCE_MANIFEST.json') == config['source_manifest_sha256']
    assert read(source / 'ARCHIVE_VERIFIED.json')['source_cleanup'] == 'complete'
    qualified = read(source / 'QUALIFICATION365.json')
    assert qualified['passed']
    assert not (root / 'PREPARED366.json').exists()
    rows, bindings = {}, {}
    for role in ['train', 'development']:
        groups = []
        for item in qualified['cells']:
            if item['role'] != role:
                continue
            path = source / 'private' / item['cell'] / 'PRIVATE_CELL.npz'
            assert sha(path) == item['curves_sha256']
            bindings[str(path)] = item['curves_sha256']
            with np.load(path, allow_pickle=False) as data:
                # Only complete historical TRAIN/DEV cells, never final22.
                values = {k: data[k] for k in data.files}
            cycles = range(39, int(values['cycle'][-1]) + 1, 10) if role == 'train' else config['development_query_cycles']
            examples = cell_examples(values, cycles)
            examples['groups'] = np.asarray([item['cell']] * len(examples['y']))
            groups.append(examples)
        rows[role] = {key: np.concatenate([g[key] for g in groups]) for key in groups[0]}
        np.savez_compressed(root / 'clients-cache' / (role + '.npz'), **rows[role])
    assert len(set(rows['train']['groups'])) == 47 and len(set(rows['development']['groups'])) == 8
    assert not set(rows['train']['groups']) & set(rows['development']['groups'])
    jobs = dict(full=dict(fit_indices=list(range(len(rows['train']['y']))),
                         fitting_cells=sorted(set(rows['train']['groups'].tolist()))))
    for number, (fit, val) in enumerate(GroupKFold(n_splits=3).split(rows['train']['x'], groups=rows['train']['groups'])):
        jobs['fold_' + str(number)] = dict(fit_indices=fit.tolist(), validation_indices=val.tolist(),
            fitting_cells=sorted(set(rows['train']['groups'][fit].tolist())),
            heldout_cells=sorted(set(rows['train']['groups'][val].tolist())))
    save(root / 'JOBS366.json', jobs)
    save(root / 'PREPARED366.json', dict(utc=now(), passed=True,
        source_bindings=bindings, final_source_files_opened=0,
        train_cells=47, development_cells=8, train_rows=len(rows['train']['y']),
        development_queries=len(rows['development']['y']),
        classic_features=rows['train']['x'].shape[1], neural_shape=list(rows['train']['nn'].shape[1:]),
        arrays_sha256={role: sha(root / 'clients-cache' / (role + '.npz')) for role in rows},
        new_fits=0, new_LLM_calls=0))
    print(json.dumps(dict(prepared=True, train_rows=len(rows['train']['y']), development_queries=len(rows['development']['y']))))


def run(root):
    config = read(root / 'PROTOCOL366.json')
    assert read(root / 'PREPARED366.json')['passed']
    assert not (root / 'HANDLE366.json').exists(), 'Never implicitly restart model fits'
    save(root / 'HANDLE366.json', dict(utc=now(), pid=os.getpid(), run=str(root), status='running'))
    start, results = time.monotonic(), []
    def one(job):
        budget = min(config['budget']['job_seconds'], config['budget']['total_seconds'] - (time.monotonic() - start))
        assert budget > 0
        command = [sys.executable, '-B', str(root / 'code/hust_battery_reference.py'), '--run', str(root), '--job', job]
        with (root / 'audit' / (job + '.stdout')).open('w') as out, (root / 'audit' / (job + '.stderr')).open('w') as err:
            try:
                proc = subprocess.run(command, stdout=out, stderr=err, timeout=budget)
                return dict(job=job, exit_code=proc.returncode)
            except subprocess.TimeoutExpired:
                return dict(job=job, exit_code=None, error_type='fit_timeout')
    with ThreadPoolExecutor(max_workers=config['budget']['workers']) as pool:
        futures = [pool.submit(one, job) for job in read(root / 'JOBS366.json')]
        for future in as_completed(futures):
            results.append(future.result())
            save(root / 'FIT_PROGRESS366.json', dict(utc=now(), completed=results, total_jobs=4))
    assert len(results) == 4
    fits_passed = all(r['exit_code'] == 0 for r in results)
    save(root / 'FITS_TERMINAL366.json', dict(utc=now(), passed=fits_passed, jobs=results, new_LLM_calls=0))
    if not fits_passed:
        save(root / 'TERMINAL366.json', dict(utc=now(), exit_code=1, reason='reference_fit_failure', jobs=results))
        raise RuntimeError('Actual reference fitting failed; retain all statuses')
    # Fresh processes re-read the immutable serialized weights and raw valid inputs.
    for job in read(root / 'JOBS366.json'):
        command = [sys.executable, '-B', str(root / 'code/check_hust_reference.py'), '--run', str(root), '--job', job]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        save(root / 'audit' / (job + '_cold_process.json'), dict(exit_code=result.returncode,
            stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0, 'Independent cold replay failed'
    reports = {job: read(root / 'models' / job / 'REPORT.json') for job in read(root / 'JOBS366.json')}
    summary = dict(utc=now(), passed=True, jobs=reports,
        actual_fit_entry_executions=4, actual_sklearn_regressors=24, actual_neural_models=4,
        learned_standard_scalers=12, optimizer_steps=sum(v['optimizer_steps'] for v in reports.values()),
        cold_processes=4, final_source_files_opened=0, new_LLM_calls=0,
        independent_confirmation_scenarios=0, majority_goal_complete=False)
    save(root / 'RESULT366.json', summary)
    save(root / 'TERMINAL366.json', dict(utc=now(), exit_code=0, passed=True, real_models=28, new_LLM_calls=0))
    print(json.dumps({k: v for k, v in summary.items() if k != 'jobs'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    for name, digest in read(root / 'CODE_FREEZE366.json')['files'].items():
        assert sha(root / name) == digest
    globals()[args.action](root)
