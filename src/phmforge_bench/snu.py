"""Prepare measured SNU planetary-gear component classification cutouts.

The labels cover healthy, planet gear and sun gear. Ring-gear damage and
physical safety targets are absent. Split by whole published recordings;
the same test rig and hardware can occur in different recordings.
"""
import hashlib
import hmac
from pathlib import Path
import secrets
import shutil
import sys
import zipfile

import numpy as np
from scipy.io import loadmat

from .data import _catalog, download, features, resource_json
from .io import file_sha256, write_json


def _decode(path, variable, expected_rows):
    try:
        import h5py
    except ImportError as error:
        raise ImportError('SNU MATLAB v7.3 cutouts require h5py; install the declared requirements.') from error
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as source:
            values = np.asarray(source[variable]).T
    else:
        values = np.asarray(loadmat(path, variable_names=[variable])[variable])
    if values.shape != (expected_rows, 3) or not np.isfinite(values).all():
        raise ValueError(f'Invalid measured SNU cutout: {variable}: {values.shape}')
    return values


def prepare_snu(destination, source=None):
    """Verify a publisher ZIP and prepare disjoint record-level splits."""
    dest = Path(destination).resolve()
    if (dest / 'READY.json').exists():
        raise FileExistsError(dest)
    pin = resource_json('snu_sources.json')
    dest.mkdir(parents=True, exist_ok=True)
    archive_path = Path(source).resolve() if source else dest / 'raw/SNU_Figshare_code.zip'
    if not source:
        download(pin['url'], archive_path, pin['sha256'], pin['bytes'])
    if (not archive_path.is_file() or archive_path.stat().st_size != pin['bytes']
            or file_sha256(archive_path) != pin['sha256']):
        raise ValueError('SNU source ZIP differs from the pinned publisher release')
    private = dest / 'private'
    private.mkdir()
    identity_key = secrets.token_bytes(32)
    (private / 'holdout_identity_key.hex').write_text(identity_key.hex() + '\n')
    rows = {'train': [], 'val': [], 'test': []}
    source_audit = []
    scratch = private / 'current_source_member.mat'
    width = pin['window_samples']
    with zipfile.ZipFile(archive_path) as archive:
        declared = {item['path'] for item in pin['members']}
        actual = {name for name in archive.namelist() if name.startswith('5m/') and name.endswith('.mat')}
        if actual != declared:
            raise ValueError('SNU source cutout members differ from the frozen source')
        for item in pin['members']:
            member = archive.getinfo(item['path'])
            if member.file_size != item['bytes']:
                raise ValueError('SNU member size differs from the source manifest')
            with archive.open(member) as incoming, scratch.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if file_sha256(scratch) != item['sha256']:
                raise ValueError('SNU measured member failed SHA-256 verification')
            values = _decode(scratch, item['variable'], pin['samples_per_member'])
            count = len(values) // width
            if count != 60 or len(values) % width != 1:
                raise ValueError('Unexpected SNU complete-window count')
            for index in range(count):
                vector = features(values[index * width:(index + 1) * width, :1])
                if vector.shape != (15,) or not np.isfinite(vector).all():
                    raise ValueError('Invalid derived SNU signal features')
                original = f"{item['path']}:{index}"
                sid = (hmac.new(identity_key, original.encode(), hashlib.sha256).hexdigest()[:24]
                       if item['split'] == 'test' else hashlib.sha256(original.encode()).hexdigest()[:24])
                rows[item['split']].append({'features': vector, 'sample_id': sid,
                    'label': item['label'], 'group': item['path'], 'source_window': index,
                    'operating_condition': item['condition']})
            source_audit.append({'member': item['path'], 'sha256': item['sha256'], 'split': item['split'],
                                 'label': item['label'], 'windows': count, 'trailing_samples_excluded': 1})
            del values
            scratch.unlink()
            print(f"SNU measured cutout verified: {item['path']}", file=sys.stderr, flush=True)
    groups = {part: {row['group'] for row in values} for part, values in rows.items()}
    if any(groups[a] & groups[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Source recordings overlap across SNU splits')
    secrets.SystemRandom().shuffle(rows['test'])
    public = dest / 'public'
    data = public / 'datasets/SNU_PLANETARY'
    data.mkdir(parents=True)
    (dest / 'holdout').mkdir()
    for part, values in rows.items():
        payload = {'dataset': np.asarray('SNU_PLANETARY'),
                   'features': np.asarray([row['features'] for row in values], dtype=np.float32),
                   'sample_ids': np.asarray([row['sample_id'] for row in values])}
        if part != 'test':
            payload.update(labels=np.asarray([row['label'] for row in values]),
                           groups=np.asarray([row['group'] for row in values]))
        np.savez_compressed(dest / 'holdout/test.npz' if part == 'test' else data / (part + '.npz'), **payload)
    write_json(private / 'labels.json', [{key: value for key, value in row.items() if key != 'features'} for row in rows['test']])
    write_json(private / 'source_audit.json', {'members': source_audit, 'source_groups': {p: sorted(v) for p, v in groups.items()}})
    limits = ['No ring-gear damage, bearing life, physical risk or compliance labels.',
              'Record-disjoint splits do not establish physical hardware independence; the same rig and gears may be reused.',
              'Development contains planet and sun faults but no healthy recording.',
              'Final operating mode changes speed and load together; their effects cannot be separated.',
              'Uses the publisher measured 60-to-65 minute cutouts, not all original 100-minute waveforms.']
    write_json(data / 'dataset.json', {'dataset': 'SNU_PLANETARY', 'seed': 322,
        'class_names': pin['classes'], 'architectures': ['mlp'], 'feature_count': 15,
        'group_definition': 'Whole published five-minute cutout kept in one split; independent physical hardware is unverified.',
        'source': pin['source_url'], 'source_sha256': pin['sha256'], 'license': pin['license'],
        'label_definition': pin['label_basis'], 'preprocessing': 'Fifteen time/FFT statistics from each complete nonoverlapping five-second vibration window; one trailing sample excluded.',
        'train_rows': len(rows['train']), 'validation_rows': len(rows['val']), 'limitations': limits})
    write_json(public / 'dataset_registry.json', {'SNU_PLANETARY': {'task': 'classification',
        'path': 'datasets/SNU_PLANETARY', 'architectures': ['mlp'], 'available_splits': ['train', 'val'],
        'test_inputs_visible': False}})
    _catalog(public)
    write_json(dest / 'SOURCE_BINDINGS.json', pin)
    report = {'status': 'prepared', 'dataset': 'SNU_PLANETARY', 'task_id': 'research_snu_planetary_component_classification',
              'task_kind': 'classification', 'rows': {p: len(v) for p, v in rows.items()},
              'source_recordings': {p: len(v) for p, v in groups.items()}, 'source_record_overlap': False,
              'physical_hardware_independence_verified': False, 'labels_in_test_input': False,
              'scope': 'Measured SNU component-core research alternative; no original full-task or safety completion.', 'limitations': limits}
    write_json(dest / 'READY.json', report)
    return report
