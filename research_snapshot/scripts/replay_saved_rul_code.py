"""Unchanged-code replay in a fresh writable workspace, without auth or network."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import numpy as np

RUN = None
NATIVE_DIR = 'formal_outputs'
INPUT_MANIFEST = 'task_input_manifest_catalog.json'
SCIENCE_MODE = 'rul'
PREDICTION_FIELDS = ['point_rul','interval_low','interval_high']
PREDICTION_KIND_FIELD = 'interval_kind'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(job, timeout, entrypoint, attempt):
    source = RUN/'tasks'/job
    native = json.loads((RUN/NATIVE_DIR/job/'status.json').read_text())
    assert native['status'] != 'running'
    if entrypoint is None:
        metadata = RUN/NATIVE_DIR/job/'response.json'
        if metadata.is_file():
            reply = json.loads(metadata.read_text())
            entrypoint = reply.get('analysis_code') or reply['results'][0]['analysis_code']
        else:
            entrypoint = 'analysis.py'
    output = RUN/'code_replays'/(job + (('_attempt'+str(attempt)) if attempt > 1 else ''))
    output.mkdir(parents=True, exist_ok=False)
    work = output/'work'
    work.mkdir()
    for name in ('tmp', 'mpl'):
        (work / '.runtime' / name).mkdir(parents=True, exist_ok=True)
    entries = []
    for row in json.loads((RUN/INPUT_MANIFEST).read_text()):
        if 'job_id' in row:
            if row['job_id'] == job: entries.append(row)
        else:
            relative = Path(row['path'])
            prefix = Path('tasks')/job
            if relative.is_relative_to(prefix):
                entries.append(dict(row, file=str(relative.relative_to(prefix))))
    hashes = {}
    for r in entries:
        f = source/r['file']; assert sha(f) == r['sha256']
        dest = work/r['file']; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest); hashes[r['file']] = sha(f)
    scripts = [f for f in source.rglob('*.py') if '__pycache__' not in f.parts]
    for f in scripts:
        dest = work/f.relative_to(source); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest); assert sha(f) == sha(dest)
    script = Path(entrypoint)
    assert not script.is_absolute() and '..' not in script.parts
    assert (work/script).is_file()
    command = ['bwrap', '--unshare-user', '--unshare-pid', '--unshare-ipc',
               '--unshare-uts', '--unshare-net', '--die-with-parent', '--new-session']
    for path in ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc', str(Path.home()/'.local/lib')]:
        if Path(path).exists(): command += ['--ro-bind', path, path]
    command += ['--bind', str(work), str(source), '--proc', '/proc', '--dev', '/dev',
                '--chdir', str(source), 'python3', str(source/script)]
    record = {'job_id':job, 'status':'running', 'started_utc':datetime.now(timezone.utc).isoformat(),
              'command':command, 'entrypoint':entrypoint, 'source_sha256':sha(source/script),
              'data_sha256':hashes, 'unchanged_code':True, 'fresh_input_only_workspace':True,
              'network':False, 'auth_mounted':False, 'gold_mounted':False, 'timeout_seconds':timeout}
    status = output/'status.json'; status.write_text(json.dumps(record, indent=2)+'\n')
    env = {k:v for k,v in os.environ.items()
           if not any(x in k.upper() for x in ['KEY', 'TOKEN', 'PASSWORD', 'SECRET', 'PROXY'])}
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4', TMPDIR=str(source / '.runtime' / 'tmp'),
               TMP=str(source / '.runtime' / 'tmp'), TEMP=str(source / '.runtime' / 'tmp'),
               MPLCONFIGDIR=str(source / '.runtime' / 'mpl'))
    start = time.monotonic()
    with (output/'stdout.txt').open('w') as out, (output/'stderr.txt').open('w') as err:
        proc = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
        record['pid'] = proc.pid; status.write_text(json.dumps(record, indent=2)+'\n')
        try:
            proc.wait(timeout=timeout)
            record.update(status='finished' if proc.returncode == 0 else 'execution_error', exit_code=proc.returncode)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            record.update(status='timeout', exit_code=proc.returncode)
    record.update(elapsed_seconds=time.monotonic()-start, ended_utc=datetime.now(timezone.utc).isoformat())
    if record['status'] == 'finished' and SCIENCE_MODE == 'battery_labels':
        try:
            import pandas as pd
            def events(path):
                value = json.loads(path.read_text())
                rows = value if isinstance(value, list) else value['results']
                return {row['battery']: row for row in rows}
            expected_events = events(source/'cell_events.json')
            actual_events = events(work/'cell_events.json')
            assert set(expected_events) == set(actual_events)
            fields = ['event_observed','event_cycle','last_observed_discharge',
                      'missing_capacity_count','rebound_above_threshold_after_first_event']
            assert all(expected_events[bid][f] == actual_events[bid][f]
                       for bid in expected_events for f in fields)
            expected = pd.read_csv(source/'labels.csv').sort_values(['battery','discharge_cycle'])
            actual = pd.read_csv(work/'labels.csv').sort_values(['battery','discharge_cycle'])
            assert expected['battery'].tolist() == actual['battery'].tolist()
            columns = ['discharge_cycle','source_cycle','event_cycle','rul_proxy','rul_lower_bound']
            left = expected[columns].to_numpy(float); right = actual[columns].to_numpy(float)
            assert left.shape == right.shape
            assert np.allclose(left, right, equal_nan=True, rtol=0, atol=1e-9)
            assert expected['event_observed'].astype(str).str.lower().tolist() == actual['event_observed'].astype(str).str.lower().tolist()
            record.update(output_kind='retrospective_battery_labels', numeric_match=True,
                          label_rows=len(actual), independent_cells=len(actual_events),
                          max_abs_numeric_difference=0.,
                          input_hashes_unchanged=all(sha(work/f) == h for f,h in hashes.items()),
                          code_hash_unchanged=sha(work/script) == record['source_sha256'],
                          labels_bytes_identical=(source/'labels.csv').read_bytes() == (work/'labels.csv').read_bytes(),
                          event_bytes_identical=(source/'cell_events.json').read_bytes() == (work/'cell_events.json').read_bytes())
        except (AssertionError, OSError, KeyError, ValueError, TypeError) as error:
            record.update(output_kind='retrospective_battery_labels', numeric_match=False,
                          audit_error_type=type(error).__name__)
    elif record['status'] == 'finished':
        try:
            expected = json.loads((source/'predictions.json').read_text())['results']
            actual = json.loads((work/'predictions.json').read_text())['results']
            assert [r['case_id'] for r in expected] == [r['case_id'] for r in actual]
            differences = []
            for a,b in zip(expected, actual):
                assert a[PREDICTION_KIND_FIELD] == b[PREDICTION_KIND_FIELD]
                for field in PREDICTION_FIELDS:
                    if a[field] is None or b[field] is None: assert a[field] is b[field]
                    else: differences.append(abs(float(a[field])-float(b[field])))
            record['max_abs_numeric_difference'] = max(differences, default=0.)
            record['numeric_match'] = record['max_abs_numeric_difference'] <= 1e-9
            record['predictions_count'] = len(actual)
            record['input_hashes_unchanged'] = all(sha(work/f) == h for f,h in hashes.items())
            record['code_hash_unchanged'] = sha(work/script) == record['source_sha256']
            record['prediction_bytes_identical'] = (source/'predictions.json').read_bytes() == (work/'predictions.json').read_bytes()
            record['output_kind'] = SCIENCE_MODE
            record['prediction_fields'] = PREDICTION_FIELDS
        except (AssertionError, OSError, KeyError, ValueError, TypeError) as error:
            record.update(numeric_match=False, audit_error_type=type(error).__name__)
    status.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ['job_id','status','elapsed_seconds','predictions_count','numeric_match','max_abs_numeric_difference','output_kind','label_rows']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('jobs', nargs='+')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--native-dir', default='formal_outputs')
    parser.add_argument('--scientific-output', choices=['rul','regression','classification','battery_labels'], default='rul')
    parser.add_argument('--prediction-fields', nargs='+', default=['point_rul','interval_low','interval_high'])
    parser.add_argument('--prediction-kind-field', default='interval_kind')
    parser.add_argument('--input-manifest', default='task_input_manifest_catalog.json')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--entrypoint', help='Override saved metadata entrypoint (default: read metadata)')
    parser.add_argument('--attempt', type=int, default=1); args = parser.parse_args()
    RUN = args.run.resolve(); NATIVE_DIR = args.native_dir; INPUT_MANIFEST = args.input_manifest; SCIENCE_MODE = args.scientific_output
    PREDICTION_FIELDS = args.prediction_fields; PREDICTION_KIND_FIELD = args.prediction_kind_field
    for name in args.jobs: replay(name, args.timeout, args.entrypoint, args.attempt)
