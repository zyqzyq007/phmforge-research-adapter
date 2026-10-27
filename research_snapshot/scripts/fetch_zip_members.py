"""Acquire selected public ZIP members with verified HTTP ranges and member CRCs.

This does not verify a publisher's whole-ZIP checksum without downloading it.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import requests


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, folder, budget, network_route='environment_default', range_chunk_bytes=8*1024*1024,
                 max_request_attempts=1):
        super().__init__()
        self.url, self.size, self.folder = url, size, folder
        self.position = self.received = 0
        self.budget = budget
        self.records = []
        self.session = requests.Session()
        if network_route not in {'environment_default', 'direct'}:
            raise ValueError('Unknown public-download network route')
        self.session.trust_env = network_route == 'environment_default'
        if not isinstance(range_chunk_bytes, int) or not 1 <= range_chunk_bytes <= 8*1024*1024:
            raise ValueError('Expected a positive byte-range chunk size at most 8 MiB')
        self.range_chunk_bytes = range_chunk_bytes
        if not isinstance(max_request_attempts, int) or not 1 <= max_request_attempts <= 3:
            raise ValueError('Expected one to three bounded request attempts')
        self.max_request_attempts = max_request_attempts

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if whence not in [0, 1, 2] or not 0 <= target <= self.size:
            raise ValueError('Invalid remote ZIP seek')
        self.position = target
        return target

    def read(self, n=-1):
        wanted = min(self.size - self.position, n if n >= 0 else self.size)
        if wanted <= 0:
            return b''
        if self.received + wanted > self.budget:
            raise ValueError('Remote acquisition byte budget exceeded')
        parts = []
        while wanted:
            length = min(wanted, self.range_chunk_bytes)
            start, end = self.position, self.position + length - 1
            requested = f'bytes={start}-{end}'
            for attempt in range(1, self.max_request_attempts + 1):
                record = dict(url=self.url, range=requested, attempt=attempt,
                              started_utc=datetime.now(timezone.utc).isoformat())
                try:
                    with self.session.get(self.url, headers={'Range': requested, 'Cache-Control': 'no-cache'},
                                          stream=True, timeout=(10, 30)) as response:
                        record.update(status_code=response.status_code, content_range=response.headers.get('Content-Range'))
                        if response.status_code != 206 or record['content_range'] != f'bytes {start}-{end}/{self.size}':
                            raise ValueError('Server did not honor the exact byte range; response closed')
                        body = response.raw.read(length + 1)
                        record['received_body_bytes'] = len(body)
                        if len(body) != length:
                            raise ValueError('Unexpected range response length')
                    filename = f'range_{len(self.records):05d}_{start}_{end}.bin'
                    (self.folder / filename).write_bytes(body)
                    record.update(status='saved', file=filename, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
                except Exception as error:
                    record.update(status='failed', error_type=type(error).__name__)
                    self.records.append(record)
                    if attempt == self.max_request_attempts or isinstance(error, ValueError):
                        raise
                    continue
                self.records.append(record)
                break
            parts.append(body)
            self.position += length
            self.received += length
            wanted -= length
        return b''.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include', help='Regex over original member names; omitted acquires the index only')
    parser.add_argument('--download-budget', type=int, default=134217728)
    parser.add_argument('--decoded-budget', type=int, default=134217728)
    parser.add_argument('--prefix-bytes', type=int,
                        help='Acquire at most this many decoded bytes per member; partial members have no CRC claim')
    args = parser.parse_args()
    if args.prefix_bytes is not None and args.prefix_bytes <= 0:
        parser.error('--prefix-bytes must be positive')
    source = json.loads(args.source_config.read_text())
    if not source['url'].startswith('https://') or source['bytes'] <= 0:
        raise ValueError('Expected explicit public HTTPS source and known whole-file size')
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / 'ranges'
    raw.mkdir()
    reader = RangeReader(source['url'], source['bytes'], raw, args.download_budget,
                         source.get('network_route', 'environment_default'),
                         source.get('range_chunk_bytes', 8*1024*1024), source.get('max_request_attempts', 1))
    manifest = dict(source=source, started_utc=datetime.now(timezone.utc).isoformat(),
        source_config_sha256=hashlib.sha256(args.source_config.read_bytes()).hexdigest(),
        acquisition_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection=args.include, whole_ZIP_checksum_verified=False,
        integrity_scope='Exact HTTP ranges; CRC32 only for complete decoded members. Prefixes are explicitly partial. SHA256 values are locally computed; publisher whole-ZIP checksum is not verified.',
        download_budget_bytes=args.download_budget, decoded_budget_bytes=args.decoded_budget,
        prefix_bytes=args.prefix_bytes, members=[])
    decoded = 0
    try:
        with zipfile.ZipFile(reader) as archive:
            index = [dict(path=x.filename, bytes=x.file_size, compressed_bytes=x.compress_size,
                          CRC32=f'{x.CRC:08x}', header_offset=x.header_offset, compression=x.compress_type)
                     for x in archive.infolist()]
            (args.output / 'zip_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
            manifest['whole_ZIP_member_count'] = len(index)
            if args.include:
                regex = re.compile(args.include)
                selected = [x for x in archive.infolist() if not x.is_dir() and regex.search(x.filename)]
                if not selected:
                    raise ValueError('No archive members matched the explicit selection')
                for item in selected:
                    wanted = min(item.file_size, args.prefix_bytes or item.file_size)
                    partial = wanted < item.file_size
                    if decoded + wanted > args.decoded_budget:
                        raise ValueError('Selected decoded byte budget exceeded')
                    target = args.output / 'members' / (item.filename + ('.partial' if partial else ''))
                    if not target.resolve().is_relative_to((args.output / 'members').resolve()):
                        raise ValueError('Unsafe ZIP member path')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # ZipExtFile verifies CRC only when the complete member is consumed.
                    with archive.open(item) as member:
                        body = member.read(wanted)
                    if len(body) != wanted:
                        raise ValueError('Decoded member size mismatch')
                    target.write_bytes(body)
                    decoded += len(body)
                    manifest['members'].append(dict(path=item.filename, bytes=len(body),
                        saved_path=str(target.relative_to(args.output)), original_member_bytes=item.file_size,
                        partial=partial, sha256=hashlib.sha256(body).hexdigest(),
                        CRC32=f'{item.CRC:08x}', member_CRC_verified=not partial))
        manifest['status'] = 'completed'
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        reader.session.close()
        manifest.update(ended_utc=datetime.now(timezone.utc).isoformat(), received_bytes=reader.received,
                        decoded_bytes=decoded, range_requests=reader.records)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['status','whole_ZIP_member_count','received_bytes','decoded_bytes','whole_ZIP_checksum_verified']}))


if __name__ == '__main__':
    main()
