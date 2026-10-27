"""Audit real Pretoria MAT measurements without fitting or evaluating a model.

The downloaded raw DEFLATE bytes must already have an exact-range/ETag receipt.
This verifies the complete member CRC, scans native MAT numeric arrays, and checks
the benchmark prefix. A matching prefix is not a claim about all CSV rows/labels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np
from scipy.io import loadmat, whosmat


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def element(stream):
    """Read a little-endian MAT v5 tag, leaving the cursor at its payload."""
    head = stream.read(8)
    if len(head) != 8:
        raise ValueError('Truncated MAT element tag')
    tag, count = struct.unpack('<II', head)
    if tag >> 16:
        dtype, size = tag & 0xffff, tag >> 16
        if not 0 < size <= 4:
            raise ValueError('Invalid packed MAT element')
        return dtype, size, stream.tell() - 4, stream.tell()
    return tag, count, stream.tell(), stream.tell() + ((count + 7) // 8) * 8


def mat_index(path, allow_partial=False):
    arrays = []
    with path.open('rb') as stream:
        header = stream.read(128)
        if not header.startswith(b'MATLAB 5.0') or header[126:] != b'IM':
            raise ValueError('Expected little-endian native MATLAB v5 file')
        while stream.tell() < path.stat().st_size:
            dtype, size, payload, after = element(stream)
            if dtype != 14 or (after > path.stat().st_size and not allow_partial):
                raise ValueError('Unsupported/truncated top-level MAT element')
            fields = []
            for _ in range(3):
                kind, length, at, following = element(stream)
                stream.seek(at)
                fields.append((kind, stream.read(length)))
                stream.seek(following)
            name = fields[2][1].decode('ascii')
            dimensions = list(struct.unpack('<' + 'i' * (len(fields[1][1]) // 4), fields[1][1]))
            if name.startswith('Channel_') and name.endswith('_Data'):
                kind, length, at, following = element(stream)
                if kind != 9 or length != int(np.prod(dimensions)) * 8 or following > after:
                    raise ValueError('Unexpected native measurement type or dimensions')
                arrays.append(dict(name=name, shape=dimensions, dtype='<f8', offset=at, bytes=length))
            stream.seek(after)
    if len(arrays) != 9 and not allow_partial:
        raise ValueError('Expected all nine measured channels')
    return arrays


def decode(raw, target, expected):
    if target.exists():
        raise ValueError('Refusing to overwrite native source')
    decoder = zlib.decompressobj(-15)
    crc = 0
    size = 0
    sha = hashlib.sha256()
    with raw.open('rb') as source, target.open('xb') as out:
        while block := source.read(1024 * 1024):
            body = decoder.decompress(block)
            size += len(body)
            if size > expected['bytes']:
                raise ValueError('Decoded source exceeds native size')
            out.write(body)
            crc = zlib.crc32(body, crc)
            sha.update(body)
        tail = decoder.flush()
        out.write(tail)
        size += len(tail)
        crc = zlib.crc32(tail, crc)
        sha.update(tail)
    if not decoder.eof or decoder.unused_data or size != expected['bytes']:
        raise ValueError('Incomplete or overlong DEFLATE member')
    if f'{crc:08x}' != expected['CRC32']:
        raise ValueError('Publisher archive member CRC mismatch')
    return dict(bytes=size, sha256=sha.hexdigest(), CRC32=f'{crc:08x}', complete_member_CRC_verified=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--acquisition-name', default='g1p0_parallel')
    p.add_argument('--command-receipt', default='audit/PARALLEL_NATIVE_COMMAND279.json')
    p.add_argument('--output-name', default='g1p0')
    args = p.parse_args()
    root = args.run.resolve()
    acquisition = json.loads((root / (args.acquisition_name + '_manifest.json')).read_text())
    record = json.loads((root / args.command_receipt).read_text())
    item = record['zip_member']
    raw = root / acquisition['file']
    if raw.stat().st_size != item['compressed_bytes']:
        raise ValueError('Compressed member length mismatch')
    h = hashlib.file_digest(raw.open('rb'), 'sha256').hexdigest()
    if h != acquisition['sha256'] or acquisition['source_offset'] != record['data_offset']:
        raise ValueError('Source receipt mismatch')
    out = root / 'outputs' / args.output_name
    out.mkdir(parents=True, exist_ok=False)
    native = out / Path(item['path']).name
    audit = dict(started_utc=datetime.now(timezone.utc).isoformat(), source_member=item,
                 acquisition_manifest_sha256=hashlib.sha256((root / (args.acquisition_name + '_manifest.json')).read_bytes()).hexdigest(),
                 LLM_calls=0, training_fits=0, test_scores=0, whole_ZIP_checksum_verified=False)
    try:
        audit['native_integrity'] = decode(raw, native, item)
        index = mat_index(native)
        scipy_index = {name: (shape, dtype) for name, shape, dtype in whosmat(native)}
        report = []
        for field in index:
            shape, dtype = scipy_index[field['name']]
            if list(shape) != field['shape'] or dtype != 'double':
                raise ValueError('Independent SciPy MAT directory disagrees')
            a = np.memmap(native, mode='r', dtype='<f8', offset=field['offset'], shape=(field['bytes'] // 8,))
            h = hashlib.sha256()
            lo, hi = np.inf, -np.inf
            differences = []
            for start in range(0, a.size, 1024 * 1024):
                block = a[start:start + 1024 * 1024]
                if not np.isfinite(block).all():
                    raise ValueError('Non-finite native measurement')
                lo, hi = min(lo, float(block.min())), max(hi, float(block.max()))
                h.update(block.tobytes())
                if field['name'] == 'Channel_1_Data':
                    d = np.diff(a[max(0, start - 1):start + 1024 * 1024])
                    differences.append([float(d.min()), float(d.max())])
            row = dict(**field, count=int(a.size), finite=True, minimum=lo, maximum=hi,
                       raw_float64_sha256=h.hexdigest(), first_three=a[:3].tolist())
            if differences:
                row['time_step_minimum'] = min(x[0] for x in differences)
                row['time_step_maximum'] = max(x[1] for x in differences)
            report.append(row)
        audit['channels'] = report
        # Independent reader and complete channel comparison, not only a few samples.
        carrier = loadmat(native, variable_names=['Channel_2_Data'])['Channel_2_Data'].ravel(order='F')
        c = next(x for x in report if x['name'] == 'Channel_2_Data')
        audit['independent_carrier_decode'] = dict(reader='scipy.io.loadmat', values=int(carrier.size),
            complete_float64_hash_equal=hashlib.sha256(carrier.astype('<f8', copy=False).tobytes()).hexdigest() == c['raw_float64_sha256'])
        if not audit['independent_carrier_decode']['complete_float64_hash_equal']:
            raise ValueError('Independent complete carrier decode differs')
        prefix = (root / 'source/planetary_prefix1MiB.csv.partial').read_text()
        body = prefix[prefix.index('[') + 1:prefix.rfind(',')]
        numbers = np.fromstring(body, sep=',')
        if numbers.size < 128 or not np.isfinite(numbers).all():
            raise ValueError('Unusable benchmark numeric prefix')
        audit['benchmark_prefix'] = dict(values=int(numbers.size), comparison='native Channel_2_Data from sample zero, no transform',
            exact_equal=bool(np.array_equal(numbers, carrier[:numbers.size])),
            maximum_absolute_difference=float(np.max(np.abs(numbers-carrier[:numbers.size]))),
            all_CSV_rows_verified=False, CSV_label_mapping_verified=False)
        intervals = []
        for i, voltage in enumerate(['9.8','9.6','9.4','9.2','9.0','8.8']):
            begin, end = (42 + 150 * i) * 38400, (179 + 150 * i) * 38400
            intervals.append(dict(author_voltage_setting=voltage, begin_sample=begin, end_sample_inclusive=end,
                                  samples=end-begin+1, within_native_record=end < carrier.size))
        audit['author_G1_condition_intervals'] = intervals
        audit['status'] = 'native_measurement_audited'
        audit['not_yet_qualified'] = ['Full 14-row benchmark mapping and labels',
            'Healthy/damage group allocation for a new experiment', 'Availability of an effective task model',
            'Original sun/planet/ring classification is not supported by this crack-only source']
    except BaseException as error:
        audit.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        audit['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save(out / 'QUALIFICATION.json', audit)
    print(json.dumps({k: audit[k] for k in ['status','native_integrity','benchmark_prefix','independent_carrier_decode']}, indent=2))


if __name__ == '__main__':
    main()
