"""Prepare publisher-bound EngineMT-QA examples for 28 research scenarios.

Range access is tied to a single frozen HDF5 layout and verifies every selected
array against a separately frozen digest. It does not verify the entire HDF5.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request

import numpy as np

from .data import _catalog, download, file_sha256, resource_json, write_json

BASE = 'https://huggingface.co/datasets/pandalin98/EngineMT-QA/resolve/ea2051a15512212976c22c1a075ec549c7d48eef/'
QA = {
    'train_qa.jsonl': (40512929, '195e66022038a2895e4e988050937195760f33a9f92c3fbf2b1661869114c2cf'),
    'test_qa.jsonl': (16961495, 'd9960b9f16554453a509087aab9f26263d14c6513d699ac7d9cf7a2835f13113'),
}
HEADS = ('fan', 'lpc', 'hpc', 'hpt', 'lpt')
CONDITIONS = ('Good Condition', 'Moderate Condition', 'Poor Condition', 'Bad Condition', 'Extremely Poor Condition')
BINS = ('1%-10%', '10%-30%', '30%-50%', '50%-70%', '70%-100%')


def _range(url, begin, length, total):
    end = begin + length - 1
    request = urllib.request.Request(url + f'?phm_range={begin}-{end}', headers={
        'Range': f'bytes={begin}-{end}', 'Accept-Encoding': 'identity',
        'User-Agent': 'PHMForge-research-benchmark/0.6'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {begin}-{end}/{total}':
                    raise ValueError('Source server did not return the exact requested HDF5 range')
                body = response.read(length + 1)
                if len(body) != length:
                    raise ValueError('Incomplete HDF5 range')
                return body
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def _hash(body):
    return hashlib.sha256(body).hexdigest()


def _options(question):
    return {a: b.strip() for a, b in re.findall(r'([a-z]):\s*([^\n]+)', question)}


def _annotation(row):
    labels = {}
    conversations = row['conversations']
    for current, answer in zip(conversations, conversations[1:]):
        if current.get('from') != 'human' or answer.get('from') != 'gpt':
            continue
        question, value = current.get('value', ''), answer.get('value', '').strip()
        stage = current.get('stage')
        if isinstance(row['id'], str) and stage == '2':
            if 'health status' in question.lower():
                found = [h for h in HEADS if re.search(r'\b' + h + r'\b', question, re.I)]
                if len(found) == 1 and value in ('a', 'b'):
                    labels[found[0]] = str(int(value == 'b'))
            elif 'potential health state' in question.lower():
                labels['state'] = _options(question).get(value, '')
            elif 'timing issue' in question.lower():
                options = _options(question)
                selected = re.findall(r'\b[a-f]\b', value.lower())
                labels['timing'] = '|'.join(sorted(options[x] for x in selected if x in options))
        elif isinstance(row['id'], list) and stage == '3':
            key = 'condition' if 'qualitative condition' in question else 'bin' if 'probability range' in question else None
            if key:
                labels[key] = _options(question).get(value, '')
    if isinstance(row['id'], str):
        if not all(head in labels for head in HEADS):
            raise ValueError(f'Incomplete component QA for source record {row["id"]}')
        return {'joint': ''.join(labels[h] for h in HEADS), 'state': labels.get('state', ''),
                'timing': labels.get('timing', '')}
    if labels.get('condition') not in CONDITIONS or labels.get('bin') not in BINS:
        raise ValueError('Incomplete trend QA annotation')
    return {'joint': labels['condition'] + '|' + labels['bin']}


def _load_qa(paths, singles, tuples):
    found = {}
    wanted = set(singles) | set(tuples)
    for path in paths:
        with path.open(encoding='utf-8') as source:
            for line in source:
                row = json.loads(line)
                key = tuple(map(str, row['id'])) if isinstance(row['id'], list) else str(row['id'])
                if key not in wanted:
                    continue
                value = _annotation(row)
                if key in found and found[key] != value:
                    raise ValueError(f'Conflicting publisher QA annotations for {key}')
                found[key] = value
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f'Publisher QA is missing {len(missing)} selected source records/tuples: {list(missing)[:3]}')
    return found


def _native_records(dest, source, manifest, workers):
    cache = dest / 'raw/records'
    cache.mkdir(parents=True, exist_ok=True)
    full, prefix = manifest['full_record_array_sha256'], manifest['prefix_record_array_sha256']
    specs = [(ident, 600, digest) for ident, digest in full.items()]
    specs += [(ident, 60, digest) for ident, digest in prefix.items() if ident not in full]
    h5 = source / 'time_series_data.h5' if source else None
    if h5 and h5.is_file():
        if h5.stat().st_size != manifest['hdf5_size'] or file_sha256(h5) != manifest['hdf5_publisher_lfs_sha256']:
            raise ValueError('Local HDF5 does not match the pinned full-file size and SHA-256')
    else:
        h5 = None

    def checked(path, digest, length):
        if not path.exists():
            return False
        if path.stat().st_size != length or file_sha256(path) != digest:
            raise ValueError(f'Cached source record failed SHA-256 verification: {path}')
        return True

    pending = []
    for ident, rows, digest in specs:
        path = cache / f'{ident}-{rows}.bin'
        if not checked(path, digest, rows * 33 * 8):
            pending.append((ident, rows, digest, path))
    if pending:
        length = manifest['ids_count'] * 8
        ids_path = dest / 'raw/data_ID.bin'
        if not checked(ids_path, manifest['ids_array_sha256'], length):
            if h5:
                with h5.open('rb') as f:
                    f.seek(manifest['ids_offset'])
                    body = f.read(length)
            else:
                body = _range(BASE + 'time_series_data.h5', manifest['ids_offset'], length, manifest['hdf5_size'])
            if len(body) != length or _hash(body) != manifest['ids_array_sha256']:
                raise ValueError('HDF5 data_ID array failed SHA-256 verification')
            ids_path.write_bytes(body)
        ids = np.frombuffer(ids_path.read_bytes(), dtype='<i8')
        lookup = {str(v): i for i, v in enumerate(ids)}
        if len(lookup) != manifest['ids_count'] or any(x[0] not in lookup for x in pending):
            raise ValueError('Invalid HDF5 source record identities')

        def fetch(item):
            ident, rows, digest, path = item
            offset = manifest['sequences_offset'] + lookup[ident] * 600 * 33 * 8
            count = rows * 33 * 8
            if h5:
                with h5.open('rb') as f:
                    f.seek(offset)
                    body = f.read(count)
            else:
                body = _range(BASE + 'time_series_data.h5', offset, count, manifest['hdf5_size'])
            if len(body) != count or _hash(body) != digest:
                raise ValueError(f'Source array failed its frozen record SHA-256: {ident}')
            path.write_bytes(body)

        with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
            jobs = [pool.submit(fetch, item) for item in pending]
            for i, job in enumerate(as_completed(jobs), 1):
                job.result()
                if i % 25 == 0 or i == len(jobs):
                    print(f'Verified EngineMT-QA native records: {i}/{len(jobs)}', file=sys.stderr, flush=True)
    arrays = {}
    for ident, rows, digest in specs:
        body = (cache / f'{ident}-{rows}.bin').read_bytes()
        if _hash(body) != digest:
            raise ValueError(f'Source array changed during preparation: {ident}')
        a = np.frombuffer(body, dtype='<f8').reshape(rows, 33)
        if not np.isfinite(a).all():
            raise ValueError(f'Nonfinite native source values: {ident}')
        arrays[ident] = a
    for ident, digest in prefix.items():
        if _hash(arrays[ident][:60].tobytes()) != digest:
            raise ValueError(f'Source prefix digest disagrees: {ident}')
    return arrays, {'full_file_sha256_verified': bool(h5), 'full_records': len(full),
                    'prefix_records': len(prefix), 'downloaded_or_read_records': len(pending)}


def _features(a):
    return np.concatenate((a.mean(0), a.std(0), a.min(0), a.max(0), a[0], a[-1])).astype(np.float32)


def prepare_engine(destination, source=None, workers=4):
    """Create separate data-root directories for the 28 source-bound cases."""
    dest = Path(destination).resolve()
    if (dest / 'READY.json').exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve() if source else None
    manifest = resource_json('engine_source_manifest.json')
    qa_paths = []
    for name, (size, digest) in QA.items():
        path = source / name if source else dest / 'raw' / name
        if not source:
            download(BASE + name, path, digest, size)
        if not path.is_file() or path.stat().st_size != size or file_sha256(path) != digest:
            raise ValueError(f'Publisher QA source missing or SHA-256 mismatch: {path}')
        qa_paths.append(path)
    health = manifest['health_training_records']
    trend = manifest['trend_training_tuples']
    cases = manifest['cases']
    singles = {r['source_id'] for r in health} | {x[0] for x in cases.values() if len(x) == 1}
    tuples = {tuple(r['source_ids']) for r in trend} | {tuple(x) for x in cases.values() if len(x) == 10}
    annotations = _load_qa(qa_paths, singles, tuples)
    arrays, source_audit = _native_records(dest, source, manifest, workers)
    temp = dest / 'prepared-shared'
    temp.mkdir(exist_ok=True)
    for dataset in ('ENGINE_QA_HEALTH', 'ENGINE_QA_TREND'):
        output = temp / dataset
        output.mkdir(exist_ok=True)
        entries = health if dataset == 'ENGINE_QA_HEALTH' else trend
        classes = ([f'{i:05b}' for i in range(32)] if dataset == 'ENGINE_QA_HEALTH' else
                   [condition + '|' + label for condition in CONDITIONS for label in BINS])
        for part in ('train', 'val'):
            subset = [row for row in entries if row['part'] == part]
            features, labels, groups, sample_ids, states, timing = [], [], [], [], [], []
            for row in subset:
                if dataset == 'ENGINE_QA_HEALTH':
                    ident = row['source_id']
                    annotation = annotations[ident]
                    values = _features(arrays[ident])
                    group = 'record_' + ident
                else:
                    ids = tuple(row['source_ids'])
                    annotation = annotations[ids]
                    values = np.concatenate([_features(arrays[ident][:60]) for ident in ids])
                    group = row['group']
                sample = hashlib.sha256((dataset + ':' + str(row)).encode()).hexdigest()[:24]
                features.append(values)
                labels.append(annotation['joint'])
                groups.append(group)
                sample_ids.append(sample)
                states.append(annotation.get('state', ''))
                timing.append(annotation.get('timing', ''))
            np.savez_compressed(output / (part + '.npz'), features=np.asarray(features, dtype=np.float32),
                labels=np.asarray(labels), groups=np.asarray(groups), sample_ids=np.asarray(sample_ids),
                states=np.asarray(states), timing=np.asarray(timing), dataset=np.asarray(dataset))
        info = {'dataset': dataset, 'seed': 2026102309, 'class_names': classes, 'architectures': ['mlp'],
                'feature_count': 198 if dataset == 'ENGINE_QA_HEALTH' else 1980,
                'group_definition': ('Distinct source cycles; physical engine separation is unverified.' if dataset == 'ENGINE_QA_HEALTH'
                    else 'Previously frozen connected groups of overlapping source tuples; physical engine separation is unverified.'),
                'data_origin': 'Pinned EngineMT-QA publisher simulation records and categorical QA annotations.',
                'annotation_not_calibrated_physical_risk': True,
                'feature_definition': 'Per anonymous channel: mean, std, min, max, first, last. Trend uses the first 60 rows of each of ten ordered records.'}
        write_json(output / 'dataset.json', info)
    for task, ids in cases.items():
        is_health = len(ids) == 1
        dataset = 'ENGINE_QA_HEALTH' if is_health else 'ENGINE_QA_TREND'
        root = dest / 'cases' / task
        if root.exists():
            raise FileExistsError(f'Case directory already exists: {root}; choose a fresh output directory')
        (root / 'public/datasets').mkdir(parents=True)
        shutil.copytree(temp / dataset, root / 'public/datasets' / dataset)
        write_json(root / 'public/dataset_registry.json', {dataset: {'task': 'classification',
            'path': f'datasets/{dataset}', 'architectures': ['mlp'], 'available_splits': ['train', 'val'],
            'test_inputs_visible': False}})
        _catalog(root / 'public')
        (root / 'holdout').mkdir()
        (root / 'private').mkdir()
        identity = 'case_' + hashlib.sha256(task.encode()).hexdigest()[:24]
        if is_health:
            f, annotation = _features(arrays[ids[0]]), annotations[ids[0]]
        else:
            f = np.concatenate([_features(arrays[ident][:60]) for ident in ids])
            annotation = annotations[tuple(ids)]
        np.savez_compressed(root / 'holdout/test.npz', features=f.reshape(1, -1),
                            sample_ids=np.asarray([identity]), dataset=np.asarray(dataset))
        write_json(root / 'private/labels.json', [{'sample_id': identity, 'group': identity,
                   'label': annotation['joint'], 'state': annotation.get('state', ''),
                   'timing': annotation.get('timing', '')}])
        write_json(root / 'READY.json', {'status': 'prepared', 'task_id': task, 'dataset': dataset,
            'task_kind': 'classification', 'scope': 'EngineMT-QA publisher annotation adaptation, not official full-task or physical-risk evaluation.'})
    write_json(dest / 'READY.json', {'status': 'prepared', 'dataset': 'ENGINE_MT_QA_CASES', 'case_count': len(cases),
        'source': BASE, 'source_audit': source_audit, 'whole_file_verified_only_if_local_source': True,
        'limitations': ['Public QA and case identities are public and do not provide a private benchmark.',
                        'Physical-engine independence and calibration of failure-probability annotations are unverified.',
                        'This release evaluates a numeric portion of each question, not the complete original answer.']})
    return {'status': 'prepared', 'case_count': len(cases), 'case_root': str(dest / 'cases'), 'source_audit': source_audit}
