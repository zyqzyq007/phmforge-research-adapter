"""One frozen target-alignment intervention, using the existing real-fit runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np

from controlled_feedback import save, sha, utc
from hust_program_runtime import Runtime


def run(root):
    protocol = json.loads((root / 'PROTOCOL463.json').read_text())
    assert not (root / 'HANDLE463.json').exists(), 'No automatic rerun'
    original = Path(protocol['source'])
    assert sha(original / 'source/program.py') == protocol['source_sha256']
    assert sha(original / 'REPORT.json') == protocol['report_sha256']
    a = (root / 'programs/original/source/program.py').read_text()
    b = (root / 'programs/aligned_target/source/program.py').read_text()
    needle = '_predict_model(model, fold_x[valid_pos])'
    assert a.count(needle) == 1 and b == a.replace(needle, needle + ' - age[idx][valid_pos]')
    save(root / 'HANDLE463.json', dict(pid=os.getpid(), utc=utc(), state='running'))
    started = time.monotonic()
    outcomes = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(Runtime(root).fit_program, root / 'programs' / name): name
                    for name in protocol['arms']}
            for future in as_completed(jobs):
                name = jobs[future]
                try:
                    report = future.result()
                    row = dict(arm=name, complete=True, report_sha256=sha(root / 'programs' / name / 'REPORT.json'))
                    print(json.dumps(dict(arm=name, complete=True, fit_entries=report['fit_entry_executions'])), flush=True)
                except Exception as error:
                    row = dict(arm=name, complete=False, error_type=type(error).__name__, error=str(error))
                    print(json.dumps(row), flush=True)
                outcomes.append(row)
                save(root / 'PROGRESS463.json', dict(utc=utc(), rows=outcomes))
        save(root / 'ALL_CHOICES_SEALED463.json', dict(utc=utc(), rows=outcomes, labels_read=False))
        if not all(r['complete'] for r in outcomes):
            return
        checks = 0
        for name in protocol['arms']:
            program = root / 'programs' / name
            report = json.loads((program / 'REPORT.json').read_text())
            for relative, digest in report['artifacts'].items():
                assert sha(program / relative) == digest
                checks += 1
            for path in program.rglob('training_audit.json'):
                audit = json.loads(path.read_text())
                if audit['mode'] == 'predict':
                    assert audit['completed_estimator_fits'] == 0 and audit['optimizer_steps'] == 0
                    checks += 1
        replay = root / 'programs/original'
        for relative in ['fold_0/check', 'fold_1/check', 'fold_2/check', 'development', 'identity_check']:
            with np.load(original / relative / 'PREDICTIONS.npz', allow_pickle=False) as old, np.load(replay / relative / 'PREDICTIONS.npz', allow_pickle=False) as new:
                np.testing.assert_array_equal(old['sample_ids'], new['sample_ids'])
                np.testing.assert_allclose(old['prediction'], new['prediction'], rtol=0, atol=1e-5)
                checks += 2
        rows = []
        for name in protocol['arms']:
            program = root / 'programs' / name
            report = json.loads((program / 'REPORT.json').read_text())
            errors = [c['MAE'] for f in report['cv'] for c in f['cells']]
            assert len(errors) == 47
            diagnostics = {str(p.relative_to(program)): json.loads(p.read_text()) for p in program.rglob('fit_diagnostics.json')}
            rows.append(dict(arm=name, CV_MAE=float(np.mean(errors)), diagnostics=diagnostics))
        save(root / 'audit/EXECUTION463.json', dict(utc=utc(), passed=True, checks=checks,
             original_reproduction_passed=True, rows=rows, final_labels_read=False))
    finally:
        save(root / 'HANDLE463.json', dict(pid=os.getpid(), utc=utc(), state='terminal', seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run(parser.parse_args().run)
