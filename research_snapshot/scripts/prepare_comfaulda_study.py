"""Recover frozen acquired records, checking source bytes before feature use."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib

from fetch_zip_members import RangeReader


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def fetch(root, unit):
    protocol = json.loads((root / 'PROTOCOL333.json').read_text())
    row = next(r for r in protocol['selected_records'] if r['unit_id'] == unit)
    folder = root / 'records' / unit
    folder.mkdir(exist_ok=False)
    result = dict(unit_id=unit, passed=False, new_model_calls=0)
    if row['reuse_qualified_pilot']:
        source = Path(protocol['source_archive']) / 'records' / row['class_name'] / 'fetch'
        receipt = json.loads((source / 'RECEIPT.json').read_text())
        assert receipt['passed'] and receipt['original_member'] == row['path']
        path = source / 'record.csv'
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt['sha256']
        assert f'{zlib.crc32(raw):08x}' == row['CRC32'] and len(raw) == row['bytes']
        result.update(passed=True, raw_path=str(path), source_reference=True,
                      sha256=receipt['sha256'], bytes=len(raw), CSV_CRC_verified=True)
        save(folder / 'RECEIPT.json', result)
        return result
    (folder / 'ranges').mkdir()
    reader = RangeReader(protocol['source_url'], protocol['outer_bytes'], folder / 'ranges',
                         protocol['per_new_record_network_cap_bytes'], network_route='direct',
                         range_chunk_bytes=protocol['range_chunk_bytes'], max_request_attempts=2)
    reader.session.verify = '/etc/ssl/certs/ca-certificates.crt'
    try:
        assert row['compression'] == 8 and row['bytes'] <= protocol['raw_decoded_cap_bytes']
        reader.seek(row['nested_data_offset'] + row['header_offset'])
        header = struct.unpack('<IHHHHHIIIHH', reader.read(30))
        assert header[0] == 0x04034b50 and header[3] == 8
        name = reader.read(header[-2]).decode('utf-8')
        assert name == row['path']
        reader.seek(header[-1], 1)
        compressed = reader.read(row['compressed_bytes'])
        assert len(compressed) == row['compressed_bytes']
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(compressed, protocol['raw_decoded_cap_bytes'] + 1)
        assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
        assert len(raw) == row['bytes'] and f'{zlib.crc32(raw):08x}' == row['CRC32']
        path = folder / 'record.csv'
        path.write_bytes(raw)
        result.update(passed=True, raw_path=str(path), bytes=len(raw),
                      sha256=hashlib.sha256(raw).hexdigest(), CSV_CRC_verified=True,
                      source_reference=False, whole_outer_MD5_verified=False)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        reader.session.close()
        result.update(received_bytes=reader.received, requests=reader.records)
        save(folder / 'RECEIPT.json', result)
    return result


def capture(root):
    protocol = json.loads((root / 'PROTOCOL333.json').read_text())
    source = Path(protocol['source_archive'])
    receipt = json.loads((source / 'ARCHIVE_VERIFIED.json').read_text())
    assert hashlib.sha256((source / receipt['manifest_filename']).read_bytes()).hexdigest() == receipt['manifest_sha256']
    inventory = {r['path']:r for r in json.loads((source/receipt['manifest_filename']).read_text())['files']}
    for config in protocol['configs']:
        prefix = 'records/'+config['class_name']+'/'
        for relative in [prefix+'index/nested_index.json', prefix+'fetch/RECEIPT.json']:
            assert hashlib.sha256((source/relative).read_bytes()).hexdigest() == inventory[relative]['sha256']
        assert hashlib.sha256((root/config['index_file']).read_bytes()).hexdigest() == config['index_sha256']
        assert config['index_sha256'] == inventory[prefix+'index/nested_index.json']['sha256']
        original = json.loads((source/(prefix+'fetch/RECEIPT.json')).read_text())
        assert config['nested_data_offset'] == original['nested_data_offset']
        for row in [r for r in protocol['selected_records'] if r['class_name']==config['class_name']]:
            member = next(m for m in json.loads((root/config['index_file']).read_text()) if m['path']==row['path'])
            assert all(member[key] == row[key] for key in ['header_offset','bytes','compressed_bytes','CRC32','compression'])
    save(root / 'SOURCE_BINDING333.json', receipt)
    rows = []
    def execute(row):
        unit = row['unit_id']
        start = time.monotonic()
        command = [sys.executable, '-B', __file__, '--run', str(root), '--unit', unit]
        try:
            p = subprocess.run(command, capture_output=True, text=True,
                               timeout=protocol['per_new_record_wall_seconds'])
            result = dict(unit_id=unit, exit_code=p.returncode, seconds=time.monotonic()-start,
                          stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired as error:
            result = dict(unit_id=unit, exit_code=None, seconds=time.monotonic()-start,
                          error_type='TimeoutExpired', timeout_seconds=error.timeout,
                          incomplete_network_bytes_unknown=True)
        save(root / 'logs' / (unit + '_capture.json'), result)
        return result
    with ThreadPoolExecutor(max_workers=protocol['source_max_parallel']) as pool:
        futures = [pool.submit(execute, row) for row in protocol['selected_records']]
        for future in as_completed(futures):
            rows.append(future.result())
            save(root / 'SOURCE_PROGRESS333.json', rows)
    passed = all(row['exit_code'] == 0 for row in rows) and len(rows) == 16
    save(root / 'SOURCE_CAPTURE333.json', dict(passed=passed, rows=rows, new_agent_sessions=0))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--unit')
    args = parser.parse_args()
    root = args.run.resolve()
    if args.unit:
        value = fetch(root, args.unit)
        print(json.dumps({k:v for k,v in value.items() if k != 'requests'}))
        return
    handle = dict(controller_pid=os.getpid(), status='running', phase='source_capture')
    save(root / 'HANDLE333.json', handle)
    try:
        if not capture(root):
            handle.update(status='terminal_source_failure', phase='source_capture', new_PHM_fits=0)
            return
        handle['phase'] = 'real_candidate_fitting'
        save(root / 'HANDLE333.json', handle)
        protocol = json.loads((root / 'PROTOCOL333.json').read_text())
        command = [sys.executable, '-B', str(root / 'code/validation_reliability_study.py'), '--run', str(root)]
        p = subprocess.run(command, capture_output=True, text=True, timeout=protocol['fit_wall_seconds'])
        save(root / 'logs/TRAIN333.json', dict(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr))
        handle.update(status='terminal' if p.returncode == 0 else 'terminal_training_error',
                      phase='real_candidate_fitting', exit_code=p.returncode)
    except Exception as error:
        handle.update(status='terminal_error', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(root / 'HANDLE333.json', handle)
        print(json.dumps(handle))


if __name__ == '__main__':
    main()
