"""Recover one pinned PHMForge input with its unchanged published recipe.

This is data preparation only: no models, primary sessions, or live-campaign edits.
The run protocol, source copies and expected public-file digest must exist first.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import subprocess
import sys
import time
import threading
import traceback
import urllib.parse
import urllib.request


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_name(path.name + '.new')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def interrupted(signum, _frame):
    raise TimeoutError('Bounded preparation interrupted by signal ' + str(signum))


def download_raw(run, cfg, state):
    """Verify a whole source file; optionally resume an archived prefix in ranges."""
    base = ('https://huggingface.co/datasets/' + cfg['repo'] + '/resolve/' +
            cfg['revision'] + '/PdM.csv?download=true')
    raw, partial = run / 'inputs/PdM.csv', run / 'inputs/PdM.csv.partial'
    if cfg.get('transport') == 'official_huggingface_xet':
        env = os.environ.copy()
        env.update(PYTHONPATH=cfg['sdk_dependencies'], PYTHONNOUSERSITE='1',
                   HF_HOME=str(run / 'clients-cache/hfhome'),
                   HF_HUB_CACHE=str(run / 'clients-cache/hf_cache'),
                   HF_XET_CACHE=str(run / 'clients-cache/xet_cache'),
                   HF_HUB_OFFLINE='0', HF_HUB_DISABLE_XET='0',
                   HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                   HF_HUB_VERBOSITY='error', RUST_LOG='error', HF_XET_LOG_FILE='/dev/null')
        ca_bundle = Path('/etc/ssl/certs/ca-certificates.crt')
        if ca_bundle.is_file():
            env.update(REQUESTS_CA_BUNDLE=str(ca_bundle), SSL_CERT_FILE=str(ca_bundle))
        command = [cfg['sdk_python'], '-B', '-u', str(run / 'source/sdk_download.py'),
                   '--run', str(run)]
        save(run / 'outputs/SDK_COMMAND.json', {'utc': now(), 'command': command,
                                              'explicit_anonymous_token_false': True})
        with (run / 'logs/sdk.stdout').open('wb') as out, (run / 'logs/sdk.stderr').open('wb') as err:
            completed = subprocess.run(command, env=env, stdout=out, stderr=err,
                                       timeout=cfg['max_seconds'] - 30)
        if completed.returncode or not raw.is_file():
            raise ValueError('Official-client acquisition failed; diagnostics retained')
        actual = digest(raw)
        if raw.stat().st_size != cfg['raw_bytes'] or actual != cfg['raw_sha256']:
            raise ValueError('Official-client output differs from pinned original')
        return raw, raw.stat().st_size, actual
    if not cfg.get('resume_prefix'):
        h, size = hashlib.sha256(), 0
        req = urllib.request.Request(base, headers={
            'User-Agent': 'PHMForge-original-input-recovery/1.0', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(req, timeout=45) as resp, partial.open('xb') as f:
            save(run / 'outputs/DOWNLOAD_HEADERS.json', {
                'status': resp.status, 'content_range': resp.headers.get('Content-Range'),
                'final_host': urllib.parse.urlparse(resp.url).netloc, 'source_url': base})
            for block in iter(lambda: resp.read(4 * 1024 * 1024), b''):
                size += len(block)
                if size > cfg['raw_bytes']:
                    raise ValueError('Original file exceeds pinned byte count')
                f.write(block)
                h.update(block)
                if size // (64 * 1024 * 1024) != (size - len(block)) // (64 * 1024 * 1024):
                    state.update(updated_utc=now(), downloaded_bytes=size)
                    save(run / 'status.json', state)
        if size != cfg['raw_bytes'] or h.hexdigest() != cfg['raw_sha256']:
            raise ValueError('Original file size/SHA256 mismatch; partial retained')
        partial.rename(raw)
        return raw, size, h.hexdigest()

    prefix = cfg['resume_prefix']
    source = Path(prefix['path'])
    if source.stat().st_size != prefix['bytes'] or digest(source) != prefix['sha256']:
        raise ValueError('Archived prefix changed; no continuation attempted')
    pieces = run / 'scratch/download_parts'
    pieces.mkdir(exist_ok=False)
    first = pieces / '0000.bin'
    shutil.copyfile(source, first)
    records = [{'index': 0, 'start': 0, 'end': prefix['bytes'] - 1,
                'bytes': prefix['bytes'], 'sha256': prefix['sha256'],
                'origin': 'archived incomplete original stream; full digest still required'}]
    ranges = []
    offset = prefix['bytes']
    while offset < cfg['raw_bytes']:
        end = min(cfg['raw_bytes'] - 1, offset + cfg['range_chunk_bytes'] - 1)
        ranges.append((len(ranges) + 1, offset, end))
        offset = end + 1
    stop = threading.Event()

    def fetch(item):
        index, start, end = item
        record = dict(index=index, start=start, end=end, started_utc=now())
        path = pieces / (str(index).zfill(4) + '.bin')
        h, size = hashlib.sha256(), 0
        req = urllib.request.Request(base + '&phm_range_start=' + str(start), headers={
            'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
            'User-Agent': 'PHMForge-pinned-range-recovery/1.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, path.open('xb') as out:
                record.update(status=resp.status, content_range=resp.headers.get('Content-Range'),
                              final_host=urllib.parse.urlparse(resp.url).netloc)
                expected = f'bytes {start}-{end}/{cfg["raw_bytes"]}'
                if resp.status != 206 or resp.headers.get('Content-Range') != expected:
                    raise ValueError('Range contract mismatch; oversized response not read')
                for block in iter(lambda: resp.read(1024 * 1024), b''):
                    if stop.is_set():
                        raise RuntimeError('Range recovery cancelled')
                    size += len(block)
                    if size > end - start + 1:
                        raise ValueError('Response exceeds declared range')
                    out.write(block)
                    h.update(block)
            if size != end - start + 1:
                raise ValueError('Range stream ended before declared byte count')
            record.update(ok=True, bytes=size, sha256=h.hexdigest())
            return record
        except BaseException as exc:
            record.update(ok=False, bytes=size, error_type=type(exc).__name__, error=str(exc)[:1000])
            raise
        finally:
            record['ended_utc'] = now()
            save(run / 'outputs'/('RANGE_' + str(index).zfill(4) + '.json'), record)

    pool = ThreadPoolExecutor(max_workers=cfg['download_parallelism'])
    try:
        futures = [pool.submit(fetch, item) for item in ranges]
        for future in as_completed(futures):
            records.append(future.result())
            state.update(updated_utc=now(), completed_parts=len(records),
                         retained_part_bytes=sum(x['bytes'] for x in records))
            save(run / 'status.json', state)
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
    h, size = hashlib.sha256(), 0
    with partial.open('xb') as out:
        for record in sorted(records, key=lambda x: x['index']):
            path = pieces / (str(record['index']).zfill(4) + '.bin')
            if digest(path) != record['sha256']:
                raise ValueError('Completed transport part changed')
            with path.open('rb') as inp:
                for block in iter(lambda: inp.read(4 * 1024 * 1024), b''):
                    out.write(block)
                    h.update(block)
                    size += len(block)
    if size != cfg['raw_bytes'] or h.hexdigest() != cfg['raw_sha256']:
        raise ValueError('Assembled original differs from pinned full-file digest')
    partial.rename(raw)
    save(run / 'outputs/RANGE_MAP.json', {'utc': now(), 'records': records,
         'complete_original_sha256': h.hexdigest(), 'complete_original_verified': True,
         'part_cache_removed_only_after_full_verification': True})
    for record in records:
        (pieces / (str(record['index']).zfill(4) + '.bin')).unlink()
    return raw, size, h.hexdigest()


def inspect_rows(paths):
    """Strict serialized-cell diagnostics; no normalization to conceal differences."""
    csv.field_size_limit(64 * 1024 * 1024)
    all_rows, labels = Counter(), Counter()
    count, columns = 0, None
    for path in paths:
        with path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if columns is None:
                columns = reader.fieldnames
            if reader.fieldnames != columns:
                raise ValueError('Column schema changed during authored split')
            label_cols = [x for x in columns if x.lower() in
                          ('label', 'rul', 'rul_percentage', 'estimated_rul')]
            for row in reader:
                raw = json.dumps(row, sort_keys=True, ensure_ascii=False,
                                 separators=(',', ':')).encode()
                all_rows[hashlib.sha256(raw).hexdigest()] += 1
                label = json.dumps({k: row[k] for k in label_cols},
                                   sort_keys=True, separators=(',', ':'))
                labels[label] += 1
                count += 1
    return count, columns, all_rows, labels


def native(run, cfg):
    resource.setrlimit(resource.RLIMIT_AS, (cfg['native_memory_bytes'],) * 2)
    recipe = run / 'source/original_recipe'
    os.environ['PHMFORGE_DATA_DIR'] = str(recipe / 'PDMBench_Data_Directory/submission096')
    tools = module('recovered_original_data_tools', recipe / 'tools/data_tools.py')
    tool = tools.LoadDatasetTool()
    results = {}
    for split in ('train', 'val', 'test'):
        signal.alarm(cfg['original_tool_timeout_seconds'])
        try:
            value = tool.invoke({'dataset_name': cfg['dataset'], 'split': split})
            results[split] = {'returned': value,
                              'loader_passed': value.startswith('Loaded ' + cfg['dataset'] + ' ')}
        except Exception as exc:
            results[split] = {'loader_passed': False, 'error_type': type(exc).__name__,
                              'error': str(exc)[:2000]}
        finally:
            signal.alarm(0)
    out = {'utc': now(), 'native_memory_limit': resource.getrlimit(resource.RLIMIT_AS)[0],
           'results': results, 'all_three_loader_calls_passed': all(
               x['loader_passed'] for x in results.values()), 'LLM_calls': 0}
    save(run / 'outputs/NATIVE_LOADER.json', out)
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return 0 if out['all_three_loader_calls_passed'] else 2


def recover(run, cfg):
    started = time.monotonic()
    state = {'status': 'running', 'pid': os.getpid(), 'started_utc': now(),
             'dataset': cfg['dataset'], 'phase': 'verify_pinned_recipe', 'LLM_calls': 0}
    save(run / 'status.json', state)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (cfg['preparation_memory_bytes'],) * 2)
        signal.alarm(cfg['max_seconds'])
        for item in cfg['original_source_files']:
            if digest(run / item['path']) != item['sha256']:
                raise ValueError('Original recipe source differs: ' + item['path'])
        state['phase'] = 'download_full_pinned_original'
        save(run / 'status.json', state)
        raw, size, raw_digest = download_raw(run, cfg, state)
        save(run / 'outputs/FULL_ORIGINAL.json', dict(
            utc=now(), repo=cfg['repo'], revision=cfg['revision'], bytes=size,
            sha256=raw_digest, pinned_digest_verified=True, complete_file=True))
        recipe = run / 'source/original_recipe'
        data = recipe / 'PDMBench_Data_Directory/submission096'
        data.mkdir(parents=True, exist_ok=False)
        # This exact placement is established by the author's data.py formula.
        os.link(raw, data / (cfg['dataset'] + '_train.csv'))
        state.update(phase='unchanged_authored_60_20_20_split', updated_utc=now())
        save(run / 'status.json', state)
        splitter = module('recovered_original_splitter', recipe / 'utils/data_splitter.py')
        result = splitter.split_dataset(cfg['dataset'], random_state=42)
        if 'error' in result:
            raise ValueError('Authored splitter failed: ' + str(result))
        save(run / 'outputs/AUTHORED_SPLIT.json', result)
        gc.collect()
        state.update(phase='original_native_loader_under_4GiB', updated_utc=now())
        save(run / 'status.json', state)
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                   '--run', str(run), '--native']
        save(run / 'outputs/NATIVE_COMMAND.json', {'utc': now(), 'command': command})
        remaining = max(1, cfg['max_seconds'] - int(time.monotonic() - started))
        with (run / 'logs/native.stdout').open('wb') as out, (run / 'logs/native.stderr').open('wb') as err:
            proc = subprocess.run(command, stdout=out, stderr=err, timeout=min(remaining, 570))
        save(run / 'outputs/NATIVE_EXIT.json', {'utc': now(), 'exit_code': proc.returncode})
        native_path = run / 'outputs/NATIVE_LOADER.json'
        loaded = json.loads(native_path.read_text()) if native_path.exists() else None
        state.update(phase='streaming_serialization_and_label_diagnostics', updated_utc=now())
        save(run / 'status.json', state)
        count, cols, rows, labels = inspect_rows([raw])
        split_count, split_cols, split_rows, split_labels = inspect_rows([
            data / (cfg['dataset'] + '_' + s + '_split.csv') for s in ('train', 'val', 'test')])
        if count != split_count or cols != split_cols:
            raise ValueError('Authored split changed row count or columns')
        diagnostic = {'utc': now(), 'rows': count, 'columns': cols,
                      'split_rows': split_count, 'strict_serialized_row_multiset_equal': rows == split_rows,
                      'strict_serialized_label_multiset_equal': labels == split_labels,
                      'raw_label_counts': labels, 'split_label_counts': split_labels,
                      'diagnostic_only_no_tolerance_or_data_repair': True}
        save(run / 'outputs/ROW_DIAGNOSTICS.json', diagnostic)
        retained_bytes = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
        summary = {'utc': now(), 'dataset': cfg['dataset'], 'original_bytes_verified': True,
                   'original_authored_recipe_unmodified': True, 'rows': count,
                   'original_split': result, 'native_loader': loaded,
                   'serialized_row_exact': rows == split_rows, 'serialized_labels_exact': labels == split_labels,
                   'strict_serialization_is_diagnostic_not_a_PHM_Agent_result': True,
                   'related_original_task_ids': cfg['original_task_ids'],
                   'retained_logical_bytes': retained_bytes, 'elapsed_seconds': time.monotonic() - started,
                   'LLM_calls': 0, 'Agent_sessions': 0, 'scientific_scores': 0,
                   'physical_correctness_qualified': False, 'existing_campaigns_modified': False,
                   'problem_confirmation_increment': 0, 'overall_goal_complete': False}
        save(run / 'SUMMARY.json', summary)
        if retained_bytes > cfg['max_retained_bytes']:
            raise ValueError('Retained run exceeds declared storage budget; no cleanup or false pass')
        state.update(status='finished', phase='source_preparation_ended', summary='SUMMARY.json')
    except BaseException as exc:
        state.update(status='execution_error', phase='source_preparation_stopped',
                     error_type=type(exc).__name__, error=str(exc)[:2000])
        (run / 'logs/preparation_traceback.txt').write_text(traceback.format_exc())
    finally:
        signal.alarm(0)
        state.update(ended_utc=now(), elapsed_seconds=time.monotonic() - started)
        save(run / 'status.json', state)
        owner = json.loads((run / 'OWNERSHIP.json').read_text())
        owner.update(status=state['status'], writers_stopped=True, ended_utc=state['ended_utc'])
        save(run / 'OWNERSHIP.json', owner)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return 0 if state['status'] == 'finished' else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--native', action='store_true')
    args = ap.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / 'PROTOCOL.json').read_text())
    for key in list(os.environ):
        if key.startswith('CODEX_') or any(x in key.upper() for x in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            os.environ.pop(key, None)
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(run / 'scratch')
    os.environ.update(MPLCONFIGDIR=str(run / 'scratch/mpl'), OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                      LANGSMITH_TRACING='false', LANGCHAIN_TRACING_V2='false')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    return native(run, cfg) if args.native else recover(run, cfg)


if __name__ == '__main__':
    raise SystemExit(main())
