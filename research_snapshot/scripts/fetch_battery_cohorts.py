"""Fetch selected original NASA nested ZIPs with bounded, verified HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import time
import zipfile
import zlib

import requests


def fetch_member(run, source, entry, workers):
    folder = run / 'battery_raw' / Path(entry['filename']).stem
    folder.mkdir(parents=True, exist_ok=False)
    transactions = []
    state = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'member': entry, 'official_url': source['url'], 'etag': source['etag']}
    status_path = folder / 'status.json'
    status_path.write_text(json.dumps(state, indent=2) + '\n')

    def piece(span):
        first, last = span
        errors = []
        for attempt in range(1, 3):
            start = time.monotonic()
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    with session.get(source['url'], stream=True, timeout=(10, 30),
                                     headers={'Range': f'bytes={first}-{last}',
                                              'If-Match': source['etag']}) as response:
                        response.raise_for_status()
                        expected = f'bytes {first}-{last}/{source["remote_zip_bytes"]}'
                        if (response.status_code != 206 or
                            response.headers.get('Content-Range') != expected or
                            response.headers.get('ETag') != source['etag']):
                            raise ValueError('Source version or requested range mismatch')
                        body = bytearray()
                        for chunk in response.iter_content(65536):
                            body.extend(chunk)
                            if len(body) > last - first + 1 or time.monotonic() - start > 60:
                                raise ValueError('HTTP range exceeded byte/time budget')
                        if len(body) != last - first + 1:
                            raise ValueError('Truncated HTTP range')
                record = dict(first=first, last=last, bytes=len(body), attempt=attempt,
                              elapsed_seconds=time.monotonic() - start,
                              sha256=hashlib.sha256(body).hexdigest(), errors=errors)
                transactions.append(record)
                return bytes(body)
            except Exception as error:
                errors.append({'type': type(error).__name__, 'message': str(error)})
                if attempt == 2:
                    transactions.append(dict(first=first, last=last, errors=errors))
                    raise

    try:
        offset = entry['local_header_offset']
        header = piece((offset, offset + 29))
        fields = struct.unpack('<4s5H3L2H', header)
        if fields[0] != b'PK\x03\x04' or fields[3] != entry['compression']:
            raise ValueError('Local header differs from frozen central directory')
        begin = offset + 30 + fields[9] + fields[10]
        end = begin + entry['compressed_bytes'] - 1
        spans = [(i, min(end, i + 262143)) for i in range(begin, end + 1, 262144)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            compressed = b''.join(pool.map(piece, spans))
        raw = zlib.decompress(compressed, -15)
        if len(raw) != entry['uncompressed_bytes'] or zlib.crc32(raw) & 0xffffffff != entry['crc32']:
            raise ValueError('Original ZIP size/CRC mismatch')
        output = folder / Path(entry['filename']).name
        output.write_bytes(raw)
        with zipfile.ZipFile(output) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError('Nested member CRC mismatch: ' + bad)
            inventory = [dict(filename=x.filename, bytes=x.file_size, crc32=x.CRC)
                         for x in archive.infolist()]
            for item in archive.infolist():
                if 'readme' in item.filename.lower() and item.file_size < 100000:
                    (folder / Path(item.filename).name).write_bytes(archive.read(item))
        state.update(status='finished', nested_zip=str(output.relative_to(run)),
                     nested_zip_bytes=len(raw), nested_zip_sha256=hashlib.sha256(raw).hexdigest(),
                     nested_inventory=inventory)
    except Exception as error:
        state.update(status='download_error', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state.update(ended_utc=datetime.now(timezone.utc).isoformat(), transactions=transactions)
        status_path.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({k: state[k] for k in ['status', 'nested_zip', 'nested_zip_sha256']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--members', nargs='+', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not all(1 <= n <= 6 for n in args.members):
        parser.error('Expected member indices 1..6, workers 1..8')
    source = json.loads((args.run / 'battery_zip_inventory.json').read_text())
    for n in args.members:
        entry = next(x for x in source['entries']
                     if Path(x['filename']).suffix == '.zip'
                     and Path(x['filename']).name.startswith(f'{n}. '))
        fetch_member(args.run, source, entry, args.workers)


if __name__ == '__main__':
    main()
