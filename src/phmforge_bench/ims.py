"""Prepare a measured IMS recording-end proxy using whole acquisition runs.

This deliberately does not call the end of recording the physical failure
time. The three runs share test rigs and the third has a known endpoint
discrepancy. Only measured records are read; no synthetic inputs are created.
"""
import hashlib
import hmac
import re
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import zipfile

import numpy as np

from .data import _catalog, download, features
from .io import file_sha256, write_json


URL = 'https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip'
ZIP_SIZE = 1075597174
ZIP_SHA256 = '21001ac266c465f5d345ec42d7b508c6a6328487fd9d4d7774422dd5ea10ad83'
SEVEN_SIZE = 1075320408
SEVEN_SHA256 = 'c6925060e702708afc47261d349339d7cd97a98166161da6b29c377eb3e66f2c'
RUNS = {
    'train': {'archive': '1st_test.rar', 'prefix': '1st_test/', 'records': 2156, 'channels': 8,
              'sha256': 'c25241571583948462a3916d97dc11c984a839f38a0da0907fff18c958c0a484'},
    'val': {'archive': '2nd_test.rar', 'prefix': '2nd_test/', 'records': 984, 'channels': 4,
            'sha256': 'b154d5ba1ae5f7f01cdd4f1bde5b08cfdc2f3134d51ad2f6614b1270a86ab632'},
    'test': {'archive': '3rd_test.rar', 'prefix': '4th_test/txt/', 'records': 6324, 'channels': 4,
             'sha256': '01e9ec83c6c55adc0300a20003a261f9a2ac2b714aae4984050325e589252bc8'},
}
STAMP = re.compile(r'\d{4}(?:\.\d{2}){5}\Z')
TARGET = 'Acquisitions remaining until this run\'s last published record; not a verified physical failure time.'


def _call(argv, log):
    with log.open('wb') as stream:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'IMS archive operation failed ({result.returncode}); inspect {log}')


def _list_records(executable, archive, expected):
    process = subprocess.run([executable, 'l', '-slt', '-ba', str(archive)],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
    names = sorted(line[7:] for line in process.stdout.splitlines()
                   if line.startswith('Path = ') and STAMP.fullmatch(Path(line[7:]).name))
    if (len(names) != expected['records'] or len(set(names)) != len(names)
            or any(not name.startswith(expected['prefix']) or '..' in Path(name).parts for name in names)):
        raise ValueError(f'Unexpected IMS source record identities in {archive.name}')
    return names


def _payload(rows, include_labels):
    value = {'dataset': np.asarray('IMS_ENDPOINT'),
             'features': np.asarray([r['features'] for r in rows], dtype=np.float32),
             'sample_ids': np.asarray([r['sample_id'] for r in rows]),
             'unit_ids': np.asarray([r['unit_id'] for r in rows], dtype=np.int64),
             'scenario_ids': np.asarray([r['scenario_id'] for r in rows], dtype=np.int64),
             'elapsed_acquisitions': np.asarray([r['elapsed_acquisitions'] for r in rows], dtype=np.int64)}
    if include_labels:
        value['labels'] = np.asarray([r['actual_rul'] for r in rows], dtype=np.float32)
    return value


def prepare_ims(destination, source=None, extractor=None):
    dest = Path(destination).resolve()
    if (dest / 'READY.json').exists():
        raise FileExistsError(f'Prepared dataset already exists: {dest}')
    executable = str(Path(extractor).resolve()) if extractor else (shutil.which('7zz') or shutil.which('7z'))
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError('IMS nested 7z/RAR extraction requires compatible 7zz/7z; pass --extractor if needed')
    dest.mkdir(parents=True, exist_ok=True)
    source_path = Path(source).resolve() if source else dest / 'raw/IMS_NASA.zip'
    if not source:
        download(URL, source_path, ZIP_SHA256, ZIP_SIZE)
    if not source_path.is_file() or source_path.stat().st_size != ZIP_SIZE or file_sha256(source_path) != ZIP_SHA256:
        raise ValueError(f'IMS NASA source does not match the pinned size and SHA-256: {source_path}')
    private = dest / 'private'
    private.mkdir()
    scratch = private / 'native_extraction'
    scratch.mkdir()
    native = scratch / 'IMS.7z'
    with zipfile.ZipFile(source_path) as archive:
        members = [v for v in archive.infolist() if v.filename == '4. Bearings/IMS.7z' and not v.is_dir()]
        if len(members) != 1 or members[0].file_size != SEVEN_SIZE:
            raise ValueError('Pinned IMS ZIP does not contain the expected nested source')
        with archive.open(members[0]) as incoming, native.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    if native.stat().st_size != SEVEN_SIZE or file_sha256(native) != SEVEN_SHA256:
        raise ValueError('Nested IMS.7z failed the pinned size and SHA-256')
    print('IMS: extracting verified run archives', file=sys.stderr, flush=True)
    _call([executable, 'x', '-y', '-bd', '-bb0', '-bsp0', '-mmt=2', '-o' + str(scratch), str(native),
           *(v['archive'] for v in RUNS.values())], private / 'extract_runs.log')
    names = {}
    for part, config in RUNS.items():
        archive = scratch / config['archive']
        if not archive.is_file() or file_sha256(archive) != config['sha256']:
            raise ValueError(f'Nested IMS run archive hash mismatch: {archive}')
        names[part] = _list_records(executable, archive, config)

    # The split is by complete acquisition run; test prefixes are random and
    # every test query is later predicted in a fresh isolated process.
    selections = {
        'train': np.unique(np.linspace(0, RUNS['train']['records'] - 2, 240, dtype=int)).tolist(),
        'val': [round((RUNS['val']['records'] - 1) * f) for f in (.3, .6, .9)],
        'test': sorted(secrets.SystemRandom().sample(range(int(.15 * RUNS['test']['records']),
                                                           int(.85 * RUNS['test']['records'])), 5)),
    }
    unit_ids = {'train': list(range(1, 5)), 'val': list(range(5, 9)),
                'test': secrets.SystemRandom().sample(range(1000000, 2000000), 4)}
    identity_key = secrets.token_bytes(32)
    (private / 'holdout_identity_key.hex').write_text(identity_key.hex() + '\n')
    rows, gold, selected_members = {'train': [], 'val': [], 'test': []}, [], {}
    for part, config in RUNS.items():
        members = [names[part][i] for i in selections[part]]
        selected_members[part] = members
        member_list = scratch / f'{part}_members.txt'
        member_list.write_text('\n'.join(members) + '\n', encoding='utf-8')
        extracted = scratch / ('records_' + part)
        _call([executable, 'x', '-y', '-bd', '-bb0', '-bsp0', '-mmt=2', '-o' + str(extracted),
               str(scratch / config['archive']), '@' + str(member_list)], private / f'extract_{part}.log')
        for index, member in zip(selections[part], members):
            path = extracted / member
            values = np.loadtxt(path, dtype=np.float64)
            if values.shape != (20480, config['channels']) or not np.isfinite(values).all():
                raise ValueError(f'Invalid IMS measured recording: {member}')
            channel_features = features(values).reshape(15, config['channels']).T
            if part == 'train':
                channel_features = channel_features.reshape(4, 2, 15).mean(axis=1)
            if channel_features.shape != (4, 15) or not np.isfinite(channel_features).all():
                raise ValueError(f'Invalid IMS channel features: {member}')
            for bearing, vector in enumerate(channel_features):
                sample = hmac.new(identity_key, f'{part}:{index}:{bearing}'.encode(), hashlib.sha256).hexdigest()[:24]
                row = {'features': np.r_[vector, index], 'sample_id': sample, 'unit_id': unit_ids[part][bearing],
                       'scenario_id': {'train': 1, 'val': 2, 'test': 3}[part],
                       'elapsed_acquisitions': index, 'actual_rul': config['records'] - 1 - index}
                rows[part].append(row)
                if part == 'test':
                    gold.append({key: row[key] for key in ('sample_id', 'unit_id', 'scenario_id', 'actual_rul')})
        shutil.rmtree(extracted)
        print(f'IMS: processed {part} measured acquisitions: {len(members)}', file=sys.stderr, flush=True)
    pairs = list(zip(rows['test'], gold))
    secrets.SystemRandom().shuffle(pairs)
    rows['test'], gold = map(list, zip(*pairs))
    public = dest / 'public'
    data = public / 'datasets/IMS_ENDPOINT'
    data.mkdir(parents=True)
    (dest / 'holdout').mkdir()
    for part in ('train', 'val'):
        np.savez_compressed(data / f'{part}.npz', **_payload(rows[part], True))
    np.savez_compressed(dest / 'holdout/test.npz', **_payload(rows['test'], False))
    write_json(private / 'labels.json', gold)
    write_json(private / 'source_audit.json', {'split': {'train': '1st_test', 'val': '2nd_test', 'test': '4th_test'},
        'selected_members': selected_members, 'selected_indices': selections, 'unit_ids': unit_ids,
        'counts': {k: v['records'] for k, v in RUNS.items()}})
    write_json(data / 'training_config.json', {'seed': 321, 'epochs': 30, 'batch_size': 128,
        'learning_rate': .001, 'cpu_threads': 2, 'fit_unit_ids': unit_ids['train'], 'target_definition': TARGET})
    write_json(data / 'dataset.json', {'dataset': 'IMS_ENDPOINT', 'task': 'rul', 'feature_count': 16,
        'target_unit': 'acquisitions', 'label': TARGET, 'source': URL,
        'preprocessing': '15 current-record time/FFT statistics per bearing plus acquisition index; two axes averaged for the first run.',
        'split': 'One complete acquisition run per train/validation/test, four bearing channels per run; same laboratory apparatus.',
        'limitations': ['The third run has 6324 records, while the source README describes 4448; the published last record is a proxy, not verified failure time.',
                        'Four bearing channels share a run endpoint; they are not four independent experiments.',
                        'Only four test bearings exist, so a top-five request cannot be fulfilled.',
                        'Public record counts and inputs may be memorized; no private blind test is claimed.']})
    write_json(public / 'dataset_registry.json', {'IMS_ENDPOINT': {'task': 'rul', 'format': 'measured_endpoint_npz',
        'path': 'datasets/IMS_ENDPOINT', 'architectures': ['mlp'], 'available_splits': ['train', 'val'],
        'test_inputs_visible': False}})
    _catalog(public)
    write_json(dest / 'SOURCE_BINDINGS.json', {'url': URL, 'bytes': ZIP_SIZE, 'sha256': ZIP_SHA256,
        'digest_origin': 'Full-source fingerprint independently computed in the prior source audit, not a publisher-provided SHA-256.',
        'nested_sha256': {v['archive']: v['sha256'] for v in RUNS.values()}})
    report = {'task_id': 'research_ims_recording_endpoint', 'dataset': 'IMS_ENDPOINT', 'task_kind': 'rul',
        'status': 'prepared', 'training_runs': 1, 'development_runs': 1, 'test_runs': 1,
        'test_bearings': 4, 'test_queries': len(gold), 'labels_in_test_input': False,
        'scope': 'Measured IMS recording-end proxy; not physical failure-time RUL or official PHMForge score.'}
    write_json(dest / 'READY.json', report)
    shutil.rmtree(scratch)
    return report
