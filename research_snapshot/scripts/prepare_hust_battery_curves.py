"""Bounded private conversion of the verified measured battery source."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
root = parser.parse_args().run.resolve()
read = lambda p: json.loads(Path(p).read_text())
save = lambda p, d: Path(p).write_text(json.dumps(d, indent=2) + '\n')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.now(timezone.utc).isoformat()
config = read(root / 'PROTOCOL365.json')
source = Path(config['source_archive'])
assert read(source / 'ARCHIVE_VERIFIED.json')['source_cleanup'] == 'complete'
assert sha(source / 'SOURCE_MANIFEST.json') == config['source_manifest_sha256']
integrity = read(source / 'INTEGRITY360.json')
assert integrity['passed'] and integrity['all_member_CRCS_verified']
assert integrity['sha256'] == config['publisher_sha256']
for name, expected in read(root / 'CODE_FREEZE365.json')['files'].items():
    assert sha(root / name) == expected
assert not (root / 'HANDLE365.json').exists(), 'Do not restart this conversion'
save(root / 'HANDLE365.json', dict(utc=now(), pid=os.getpid(), run=str(root), status='running'))
started = time.monotonic()
python = Path('/srv/yyf-zyq/tools/python/cpython-3.12.11-linux-x86_64-gnu')
deps = Path('/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps')
inventory = {x['name']: x for x in integrity['members']}


def one(cell, role):
    folder = root / 'private' / cell
    folder.mkdir(exist_ok=False)
    (folder / 'runtime/tmp').mkdir(parents=True)
    (folder / 'runtime/mpl').mkdir(parents=True)
    cache = root / 'clients-cache' / (cell + '.pkl')
    member = inventory['our_data/' + cell + '.pkl']
    local = dict(cell=cell, role=role, rates=config['discharge_rates'][cell])
    task_config = folder / 'INPUT_CONFIG.json'
    save(task_config, local)
    assert member['bytes'] < 512 * 1024**2
    with zipfile.ZipFile(source / 'raw/our_data.zip') as archive:
        body = archive.read(member['name'])
    assert len(body) == member['bytes'] and hashlib.sha256(body).hexdigest() == member['sha256']
    cache.write_bytes(body)
    del body
    remaining = config['budget']['wall_seconds'] - (time.monotonic() - started)
    if remaining <= 0:
        return dict(cell=cell, role=role, passed=False, error_type='whole_conversion_deadline')
    command = ['bwrap', '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts',
               '--unshare-net', '--die-with-parent', '--new-session']
    for path in ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc']:
        if Path(path).exists():
            command += ['--ro-bind', path, path]
    for path, destination in [(python, '/opt/python312'), (deps, '/opt/deps'),
            (root / 'code', '/program'), (task_config, '/config.json'), (cache, '/cell.pkl')]:
        command += ['--ro-bind', str(path), destination]
    command += ['--bind', str(folder), '/work', '--proc', '/proc', '--dev', '/dev',
        '--chdir', '/program', '--remount-ro', '/', '/opt/python312/bin/python3.12', '-B',
        '/program/hust_battery_curves.py']
    environment = dict(PATH='/opt/python312/bin:/usr/bin:/bin', PYTHONPATH='/opt/deps:/program',
        PYTHONDONTWRITEBYTECODE='1', TMPDIR='/work/runtime/tmp', TMP='/work/runtime/tmp',
        TEMP='/work/runtime/tmp', MPLCONFIGDIR='/work/runtime/mpl',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        process = subprocess.run(command, env=environment, capture_output=True, text=True,
            timeout=min(config['budget']['per_cell_seconds'], remaining))
        save(folder / 'PROCESS.json', dict(exit_code=process.returncode, command=command,
             stdout=process.stdout, stderr=process.stderr))
        assert process.returncode == 0, 'Private worker failed; retain stderr without printing final values'
        audit = read(folder / 'PRIVATE_AUDIT.json')
        assert audit['passed']
        result = dict(cell=cell, role=role, passed=True, member_sha256=member['sha256'],
            curves_sha256=sha(folder / 'PRIVATE_CELL.npz'), audit_sha256=sha(folder / 'PRIVATE_AUDIT.json'))
        cache.unlink()
        return result
    except Exception as error:
        save(folder / 'FAILURE.json', dict(error_type=type(error).__name__, message=str(error)))
        return dict(cell=cell, role=role, passed=False, error_type=type(error).__name__)


rows = []
with ThreadPoolExecutor(max_workers=config['budget']['workers']) as pool:
    futures = [pool.submit(one, cell, role) for cell, role in config['roles'].items()]
    for future in as_completed(futures):
        rows.append(future.result())
        save(root / 'PROGRESS365.json', dict(utc=now(), completed=len(rows), total=77,
            passed=sum(r['passed'] for r in rows), seconds=time.monotonic() - started,
            new_fits=0, new_predictions=0, new_LLM_calls=0))
passed = len(rows) == 77 and all(r['passed'] for r in rows)
result = dict(utc=now(), passed=passed, cells=rows,
    counts={role: sum(r['role'] == role for r in rows) for role in ['train', 'development', 'final']},
    final_feature_or_label_values_public=False, model_experiment_admitted=passed,
    label_scope='Original provided inclusive countdown; independent exact physical failure not claimed',
    new_fits=0, new_predictions=0, new_LLM_calls=0)
save(root / 'QUALIFICATION365.json', result)
save(root / 'TERMINAL365.json', dict(utc=now(), exit_code=0 if passed else 1,
    passed=passed, completed=len(rows), new_fits=0, new_LLM_calls=0))
print(json.dumps({k: v for k, v in result.items() if k != 'cells'}))
raise SystemExit(0 if passed else 1)
