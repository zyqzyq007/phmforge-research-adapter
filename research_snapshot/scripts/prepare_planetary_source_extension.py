"""Acquire preregistered, previously unused full recordings; never fit a model.

Reuse the exact-range receiver and native MAT auditors. The process freezes no
Agent experiment, opens no final scores, and never imports an old model file.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def audit_one(root, name):
    from prepare_planetary_probe import native_audit
    from qualify_planetary_source import decode
    protocol = json.loads((root / 'protocol.json').read_text())
    item = next(x for x in protocol['members'] if Path(x['path']).stem.lower() == name)
    receipt = json.loads((root / (name + '_manifest.json')).read_text())
    compressed = root / receipt['file']
    command = json.loads((root / 'audit' / (name + '_COMMAND.json')).read_text())
    assert receipt['source_offset'] == command['data_offset']
    assert compressed.stat().st_size == item['compressed_bytes'] == receipt['bytes']
    assert sha(compressed) == receipt['sha256']
    native = root / 'inputs' / Path(item['path']).name
    decoded = decode(compressed, native, item)
    audited = native_audit(native, minimum_duration_seconds=929)
    assert audited['native_sha256'] == decoded['sha256']
    old = json.loads((root / 'audit/HISTORICAL_NATIVE_FOOTPRINT302.json').read_text())
    previous_hashes = {x['sha256'] for xs in old['native_G1_file_footprint'].values() for x in xs}
    assert decoded['sha256'] not in previous_hashes
    carrier = next(x for x in audited['channels'] if x['name'] == 'Channel_2_Data')
    assert carrier['raw_float64_sha256'] not in protocol['excluded_carrier_sha256']
    assert len({x['count'] for x in audited['channels']}) == 1
    save(root / 'audit' / (name + '_NATIVE.json'), dict(
        status='qualified_native_measurement', utc=utc(), original_member=item,
        integrity=decoded, native_audit=audited, differs_from_old_full_file_hashes=True,
        new_fits=0, new_LLM_calls=0, new_model_predictions=0, new_test_scores=0))


def run(root):
    import requests
    protocol = json.loads((root / 'protocol.json').read_text())
    started = time.monotonic()
    recovery_path = root / 'audit/PREFLIGHT_RECOVERY302.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    source_epoch = datetime.fromisoformat(recovery['original_started_utc']).timestamp() if recovery else time.time()
    state = dict(status='running', phase='source_preflight', pid=os.getpid(),
                 started_utc=utc(), workers=[], new_fits=0, new_LLM_calls=0,
                 new_model_inference=0, final_scores_opened=False,
                 original_source_started_epoch=source_epoch, preflight_recovery=recovery)
    jobs = []
    native_jobs = []
    logs = []
    save(root / 'status.json', state)
    try:
        with requests.Session() as session:
            with session.get(protocol['metadata_url'], stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                body = response.raw.read(1024 * 1024 + 1, decode_content=True)
                save(root / 'audit/PUBLIC_METADATA_RESPONSE302.json', dict(
                    utc=utc(), status=response.status_code, content_encoding=response.headers.get('Content-Encoding'),
                    decoded_body_bytes=len(body), decoded_body_sha256=hashlib.sha256(body).hexdigest()))
                assert len(body) <= 1024 * 1024
            (root / 'source/CURRENT_PUBLIC_METADATA302.json').write_bytes(body)
            metadata = json.loads(body)
            assert metadata['version'] == 3 and metadata['id'] == 13476525
            file = next(x for x in metadata['files'] if x['id'] == 28992879)
            assert file['size'] == protocol['source_bytes']
            assert file['supplied_md5'] == protocol['publisher_whole_zip_md5_not_verified']
            for item in protocol['members']:
                begin = item['header_offset']
                name = Path(item['path']).stem.lower()
                with session.get(protocol['source_url'], headers={
                    'Range': f'bytes={begin}-{begin+511}', 'Accept-Encoding': 'identity',
                    'If-Match': protocol['source_etag']},
                    params={'phm_source302': name}, stream=True, timeout=(15, 60)) as response:
                    body = response.raw.read(513)
                    (root / 'source' / (name + '_header_response.bin')).write_bytes(body)
                    save(root / 'audit' / (name + '_HEADER_RESPONSE302.json'), dict(
                        utc=utc(), status=response.status_code, content_range=response.headers.get('Content-Range'),
                        ETag=response.headers.get('ETag'), body_bytes=len(body),
                        body_sha256=hashlib.sha256(body).hexdigest()))
                    assert response.status_code == 206 and len(body) == 512
                    assert response.headers['Content-Range'] == f'bytes {begin}-{begin+511}/{protocol["source_bytes"]}'
                    assert response.headers['ETag'] == protocol['source_etag']
                fields = struct.unpack('<4s5H3I2H', body[:30])
                assert fields[0] == b'PK\x03\x04' and fields[3] == item['compression'] == 8
                assert fields[6] == int(item['CRC32'], 16)
                assert fields[7:9] == (item['compressed_bytes'], item['bytes'])
                assert body[30:30+fields[9]].decode('utf-8') == item['path']
                offset = begin + 30 + fields[9] + fields[10]
                (root / 'source' / (name + '_local_header.bin')).write_bytes(body)
                code = root / 'code/acquire_range_source.py'
                argv = [sys.executable, '-B', str(code), '--run', str(root),
                        '--url', protocol['source_url'], '--bytes', str(protocol['source_bytes']),
                        '--etag', protocol['source_etag'], '--name', name,
                        '--destination', 'inputs/' + name + '.deflate',
                        '--chunk-bytes', str(4 * 1024**2), '--workers', str(protocol['workers_per_member']),
                        '--max-attempts', '3', '--max-failures', '12', '--route', 'proxy',
                        '--format', 'raw', '--offset', str(offset), '--length', str(item['compressed_bytes']),
                        '--connect-timeout', '15', '--read-timeout', '90',
                        '--refresh-public-request', '--drop-completed-cache']
                save(root / 'audit' / (name + '_COMMAND.json'), dict(
                    argv=argv, zip_member=item, data_offset=offset, code_sha256=sha(code), utc=utc()))
                log = (root / 'logs' / (name + '_acquire.log')).open('w')
                logs.append(log)
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                jobs.append((name, process))
        state['phase'] = 'acquiring_original_records'
        while any(p.poll() is None for _, p in jobs):
            state['workers'] = [dict(name=n, pid=p.pid, exit_code=p.poll()) for n, p in jobs]
            state['elapsed_seconds'] = time.monotonic() - started
            save(root / 'status.json', state)
            if time.time() - source_epoch > protocol['max_source_seconds']:
                raise TimeoutError('Frozen acquisition wall budget reached')
            time.sleep(5)
        state['phase'] = 'native_auditing'
        save(root / 'status.json', state)
        for name, process in jobs:
            if process.returncode != 0:
                continue
            log = (root / 'logs' / (name + '_native.log')).open('w')
            logs.append(log)
            audit = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                      '--run', str(root), '--audit-member', name],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            native_jobs.append(audit)
            state['native_worker'] = dict(name=name, pid=audit.pid)
            save(root / 'status.json', state)
            try:
                remaining = source_epoch + protocol['max_total_seconds'] - time.time()
                audit.wait(timeout=max(0, min(protocol['max_native_seconds_per_member'], remaining)))
            except subprocess.TimeoutExpired:
                stop(audit)
            state.setdefault('native_results', []).append(dict(name=name, exit_code=audit.poll()))
            save(root / 'status.json', state)
        qualified = sorted(p.name for p in (root / 'audit').glob('*_NATIVE.json'))
        if len(qualified) == len(protocol['members']):
            hashes = [next(x for x in json.loads((root / 'audit' / name).read_text())['native_audit']['channels']
                           if x['name'] == 'Channel_2_Data')['raw_float64_sha256'] for name in qualified]
            assert len(set(hashes)) == len(hashes)
        state.update(status='finished' if len(qualified) == len(protocol['members']) else 'incomplete',
                     phase='source_audit_closed', qualified_members=qualified,
                     new_Agent_experiment_admitted=False, new_fits=0, new_LLM_calls=0)
    except BaseException as error:
        state.update(status='failed', phase='source_audit_stopped', error_type=type(error).__name__)
        raise
    finally:
        for process in native_jobs:
            stop(process)
        for _, process in jobs:
            stop(process)
        for log in logs:
            log.close()
        state.update(ended_utc=utc(), elapsed_seconds=time.monotonic()-started,
                     cumulative_source_elapsed_seconds=time.time()-source_epoch,
                     workers=[dict(name=n, pid=p.pid, exit_code=p.poll()) for n, p in jobs])
        save(root / 'status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--audit-member')
    args = parser.parse_args()
    if args.audit_member:
        audit_one(args.run.resolve(), args.audit_member)
    else:
        run(args.run.resolve())
