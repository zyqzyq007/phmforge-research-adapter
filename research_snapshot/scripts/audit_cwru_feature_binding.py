"""Read-only lineage audit of native CWRU files and the old feature table.

This does not fit a model, infer fault predictions, or score a submitted model.
Three matching moments are reported as summary alignment, not waveform equality.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import struct
from urllib.parse import urlparse
import zlib

import numpy as np
from scipy.io import loadmat

from qualify_planetary_source import element


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.items = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.current = [dict(attrs).get('href', ''), '']

    def handle_data(self, data):
        if self.current is not None:
            self.current[1] += data

    def handle_endtag(self, tag):
        if tag == 'a' and self.current is not None:
            self.items.append(self.current)
            self.current = None


def numeric_matrix(stream, end):
    fields = []
    for _ in range(3):
        kind, size, at, after = element(stream)
        stream.seek(at)
        fields.append((kind, stream.read(size)))
        stream.seek(after)
    flags = struct.unpack('<II', fields[0][1])
    if flags[0] & 0x0800:
        raise ValueError('Complex native array not admitted')
    dimensions = np.frombuffer(fields[1][1], dtype='<i4')
    name = fields[2][1].decode('ascii')
    kind, size, at, after = element(stream)
    types = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4',
             6: '<u4', 7: '<f4', 9: '<f8', 12: '<i8', 13: '<u8'}
    if kind not in types or after > end:
        raise ValueError('Unsupported or truncated native numeric array')
    stream.seek(at)
    array = np.frombuffer(stream.read(size), dtype=types[kind]).copy()
    if array.size != int(np.prod(dimensions)):
        raise ValueError('Native MAT dimensions differ from data length')
    return name, array.reshape(tuple(dimensions), order='F')


def native_arrays(path):
    """Independent little-endian MAT-v5 numeric decoder, including miCOMPRESSED."""
    result = {}
    with path.open('rb') as stream:
        header = stream.read(128)
        if not header.startswith(b'MATLAB 5.0') or header[126:] != b'IM':
            raise ValueError('Expected little-endian MATLAB v5 source')
        while stream.tell() < path.stat().st_size:
            kind, size, at, after = element(stream)
            if kind == 14:
                name, array = numeric_matrix(stream, after)
                stream.seek(after)
            elif kind == 15:
                stream.seek(at)
                inner = io.BytesIO(zlib.decompress(stream.read(size)))
                nested_kind, _, _, nested_end = element(inner)
                if nested_kind != 14:
                    raise ValueError('Compressed member is not a numeric matrix')
                name, array = numeric_matrix(inner, nested_end)
                # scipy and MATLAB store compressed top-level elements unpadded.
                stream.seek(at + size)
            else:
                raise ValueError('Unexpected native top-level MAT element')
            if name in result:
                raise ValueError('Duplicate native matrix name')
            result[name] = array
    return result


def groups_from_csv(path):
    groups = []
    previous = None
    for row in csv.DictReader(path.open()):
        segment = int(row['segment_idx'])
        key = (row['fault_size'], row['label'])
        if previous is None or key != previous[0] or segment <= previous[1]:
            groups.append(dict(group=len(groups), label=int(row['label']),
                               fault_size=float(row['fault_size']), segments=[], moments=[]))
        groups[-1]['segments'].append(segment)
        groups[-1]['moments'].append(ast.literal_eval(row['all_features'])[:3])
        previous = key, segment
    return groups


def moments(windows):
    return np.column_stack((windows.mean(axis=1), windows.std(axis=1),
                            np.sqrt(np.mean(windows ** 2, axis=1))))


def audit(root, scan_offsets=False):
    protocol = json.loads((root / 'PROTOCOL299.json').read_text())
    source = root / 'source/snapshot'
    manifest = json.loads((source / 'manifest.json').read_text())
    assert all(x['status'] == 'saved' for x in manifest), 'Not all fixed sources acquired'
    for item in manifest:
        path = source / item['file']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    assert sum(x['bytes'] for x in manifest) <= protocol['resources']['source_byte_cap']
    page = Anchors()
    page.feed((source / 'official_12k.html').read_text())
    metadata = {}
    for href, title in page.items:
        match = re.fullmatch(r'(IR|B|OR)(007|014|021)(@6)?_([0-3])', title.strip())
        if not match or (match[1] == 'OR' and match[3] != '@6'):
            continue
        filename = Path(urlparse(href).path).name
        metadata[filename] = dict(official_link_text=title.strip(),
            physical_fault={'IR': 'inner_race', 'B': 'ball', 'OR': 'outer_race'}[match[1]],
            diameter_inches=int(match[2]) / 1000, motor_load_HP=int(match[4]),
            sampling_hz=12000, source_page=protocol['page_targets'][1]['url'])
    old = Path(protocol['old_feature_csv'])
    assert sha(old) == protocol['old_feature_sha256']
    groups = groups_from_csv(old)
    setting = protocol['matching_precommit']
    matches, probes = [], []
    for target in protocol['native_targets']:
        path = source / (target['id'] + '.mat')
        direct = native_arrays(path)
        scipy = loadmat(path)
        channel_rows = []
        for name, array in direct.items():
            assert np.array_equal(array, scipy[name]), (path.name, name)
            assert np.isfinite(array).all(), (path.name, name)
            channel_rows.append(dict(name=name, shape=list(array.shape), values=int(array.size),
                float64_sha256=hashlib.sha256(array.astype('<f8').ravel(order='F').tobytes()).hexdigest(),
                finite=True, manual_and_scipy_equal=True))
            if not name.endswith('_time'):
                continue
            signal = array.ravel(order='F')
            for width in setting['window_lengths']:
                if len(signal) < width:
                    continue
                first = moments(signal[:width][None, :])[0]
                if scan_offsets:
                    prefix = np.r_[0., np.cumsum(signal, dtype=np.float64)]
                    means = (prefix[width:] - prefix[:-width]) / width
                    mean_order = np.argsort(means)
                    ordered_means = means[mean_order]
                for group in groups:
                    if len(group['segments']) < setting['minimum_full_group_rows']:
                        continue
                    reference = np.asarray(group['moments'])
                    tolerance = setting['maximum_absolute_tolerance']
                    if scan_offsets:
                        lo, hi = np.searchsorted(ordered_means,
                            [reference[0, 0] - tolerance, reference[0, 0] + tolerance])
                        offsets = mean_order[lo:hi].tolist()
                    else:
                        offsets = [0] if np.max(np.abs(first - reference[0])) <= tolerance else []
                    for offset in offsets:
                        if np.max(np.abs(moments(signal[offset:offset + width][None, :])[0] - reference[0])) > tolerance:
                            continue
                        for hop in setting['hops']:
                            starts = offset + np.asarray(group['segments']) * hop
                            if starts[-1] + width > len(signal):
                                continue
                            windows = np.lib.stride_tricks.sliding_window_view(signal, width)[starts]
                            observed = moments(windows)
                            delta = float(np.max(np.abs(observed - reference)))
                            if delta <= tolerance:
                                matches.append(dict(native_file=path.name, native_variable=name,
                                    official_native=metadata.get(Path(target['url']).name),
                                    published_group=group['group'], published_label=group['label'],
                                    published_size=group['fault_size'], rows=len(starts), width=width,
                                    starting_offset=int(offset), hop=hop, maximum_absolute_three_moment_delta=delta,
                                    all17_features_compared=False, waveform_equality_claimed=False))
        assert sum(row['name'].endswith('_DE_time') for row in channel_rows) == 1
        probes.append(dict(file=path.name, sha256=sha(path), official_native=metadata.get(Path(target['url']).name),
                           channels=channel_rows, calibration_units_certified=False))
    legacy = {0: 'normal', 1: 'inner_race', 2: 'outer_race', 3: 'ball'}
    disagreements = [dict(row, previous_physical_alias=legacy[row['published_label']]) for row in matches
        if row['official_native'] and legacy[row['published_label']] != row['official_native']['physical_fault']]
    result = dict(utc=datetime.now(timezone.utc).isoformat(), status='native_and_feature_summary_audited',
        exploratory_offset_search=scan_offsets,
        source_sha256=protocol['old_feature_sha256'], fixed_probes=len(probes), published_groups=len(groups),
        acquired_bytes=sum(x['bytes'] for x in manifest), native_probes=probes,
        complete_group_three_moment_alignments=matches, alias_disagreements=disagreements,
        native_raw_values_finite=sum(c['values'] for p in probes for c in p['channels']),
        independent_native_reader_all_equal=True, LLM_calls=0, fitting_calls=0, model_inference_calls=0,
        submitted_model_scores=0, paid_Agent_admitted=False,
        limitations=['Summary alignment is not proof of waveform identity or all17feature equality.',
            'No unmatched file is certified unused;old feature table lacks explicit recording IDs.',
            'No independent-bearing or field-generalization claim.',
            'Physical alias disagreement concerns interpretation,not automatic numeric label corruption.'])
    filename = 'CWRU_NATIVE_BINDING_OFFSET299.json' if scan_offsets else 'CWRU_NATIVE_BINDING299.json'
    save(root / 'audit' / filename, result)
    print(json.dumps({k: result[k] for k in ['status','fixed_probes','published_groups','acquired_bytes',
        'native_raw_values_finite','independent_native_reader_all_equal','paid_Agent_admitted']}, indent=2))
    print(json.dumps(dict(complete_group_matches=len(matches), physical_alias_disagreements=len(disagreements)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--scan-offsets', action='store_true')
    args = parser.parse_args()
    audit(args.run.resolve(), args.scan_offsets)
