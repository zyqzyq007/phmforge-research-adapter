"""Download the explicit public source list and record byte-level provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


def fetch(target, output, direct=False, max_mib=10, read_timeout=30):
    record = dict(target)
    record['retrieved_at'] = datetime.now(timezone.utc).isoformat()
    record['network_route'] = 'direct' if direct else 'environment_default'
    record['read_timeout_seconds'] = read_timeout
    try:
        with requests.Session() as client:
            if direct:
                client.trust_env = False
            response = client.get(target['url'], timeout=(10, read_timeout), stream=True,
                                  headers={'User-Agent': 'PHM-Research-SourceSnapshot/1.0',
                                           **target.get('headers', {})})
            record.update(status_code=response.status_code, final_url=response.url,
                          content_type=response.headers.get('Content-Type'))
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_content(65536):
                body.extend(chunk)
                if len(body) > max_mib * 1024 * 1024:
                    raise ValueError(f'Source exceeds the {max_mib} MiB snapshot limit')
        path = output / (target['id'] + target.get('suffix', '.html'))
        path.write_bytes(body)
        record.update(status='saved', file=path.name, bytes=len(body),
                      sha256=hashlib.sha256(body).hexdigest())
        if 'expected_bytes' in target and len(body) != target['expected_bytes']:
            raise ValueError('Downloaded byte count differs from the pinned source')
        if 'git_blob_sha1' in target:
            actual = hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()
            if actual != target['git_blob_sha1']:
                raise ValueError('Downloaded content differs from the pinned Git blob')
    except Exception as error:
        record.update(status='unavailable', error_type=type(error).__name__,
                      error=str(error))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--direct', action='store_true',
                        help='Use direct access for this public source list')
    parser.add_argument('--max-mib', type=int, default=10,
                        help='Per-source byte limit in MiB; default 10')
    parser.add_argument('--read-timeout-seconds', type=int, default=30,
                        help='Maximum idle read time per response, 1–180 seconds; default 30')
    args = parser.parse_args()
    if not 1 <= args.max_mib <= 128:
        parser.error('--max-mib must be between 1 and 128')
    if not 1 <= args.read_timeout_seconds <= 180:
        parser.error('--read-timeout-seconds must be between 1 and 180')
    args.output.mkdir(parents=True, exist_ok=False)
    targets = json.loads(args.sources.read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda target: fetch(target, args.output, args.direct,
                                                   args.max_mib, args.read_timeout_seconds), targets))
    (args.output / 'manifest.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    for record in records:
        print(json.dumps({key: record[key] for key in
                          ['id', 'status', 'status_code', 'bytes', 'error_type']
                          if key in record}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
