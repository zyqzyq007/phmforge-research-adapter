"""Audit rolling subject capacity from original status records, including tar archives.

Read metadata only. Archived receipts older than the window cannot contain
newer immutable sessions. Missing or ambiguous evidence is reported separately;
absence of task-work evidence never implies absence of requests or cost.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile

EXCLUDED = {'workspace_snapshot', 'source_history', 'source', 'clients-cache',
            '.git', '__pycache__', '_shared', 'node_modules', 'codex_home', 'deps'}


def date(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def status_row(raw, location, cutoff):
    item = json.loads(raw)
    if not isinstance(item, dict) or not all(item.get(x) for x in ('job_id', 'cli_pid', 'started_utc')):
        return None
    if date(item['started_utc']) < cutoff:
        return None
    job = item['job_id']
    usage = item.get('usage') or {}
    tools = item.get('observed_non_message_item_types') or []
    work = bool((usage.get('input_tokens') or 0) > 0 or tools or item.get('status') == 'timeout')
    reserved_live = item.get('status') == 'running'
    role = 'judge_or_connectivity_probe' if any(x in job.lower() for x in ('_judge', 'health', 'connectivity_probe')) else 'primary'
    return dict(job_id=job, started_utc=item['started_utc'], ended_utc=item.get('ended_utc'),
                pid=item['cli_pid'], status=item.get('status'), condition=item.get('condition'),
                model_requested=item.get('model_requested'), usage_input=usage.get('input_tokens'),
                observed_tool_types=tools, role=role, entered_task_evidence=work,
                reserved_live=reserved_live, counted_for_capacity=role == 'primary' and (work or reserved_live),
                source=location, source_sha256=hashlib.sha256(raw).hexdigest())


def archive_rows(root, cutoff, errors, coverage, cache_dir=None):
    receipt = root / 'ARCHIVE_VERIFIED.json'
    if not receipt.is_file():
        return
    r = json.loads(receipt.read_text())
    if date(r['verified_utc']) < cutoff:
        return
    manifest = root / r.get('manifest_filename', 'SOURCE_MANIFEST.json')
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != r['manifest_sha256']:
        raise ValueError('Archive manifest differs: ' + str(manifest))
    entries = [x for x in json.loads(raw)['files'] if Path(x['path']).name == 'status.json'
               and not (set(Path(x['path']).parts) & EXCLUDED)]
    coverage.append(dict(archive=str(root), targets=len(entries), portable=r.get('archive_format') == 'tar.gz'))
    expected = {x['path']: x for x in entries}
    if not entries:
        return
    cache = None
    if cache_dir is not None:
        key = hashlib.sha256((str(root.resolve()) + r['manifest_sha256']).encode()).hexdigest()
        cache = cache_dir / (key + '.json')
        if cache.exists():
            cached = json.loads(cache.read_text())
            if cached['manifest_sha256'] != r['manifest_sha256'] or set(cached['statuses']) != set(expected):
                raise ValueError('Cached status inventory differs from verified manifest')
            for relative, encoded in cached['statuses'].items():
                raw = base64.b64decode(encoded, validate=True)
                if hashlib.sha256(raw).hexdigest() != expected[relative]['sha256']:
                    raise ValueError('Cached status digest differs: ' + relative)
                row = status_row(raw, str(root) + '::' + relative, cutoff)
                if row:
                    yield row
            coverage[-1]['verified_metadata_cache_hit'] = True
            return
    captured = {}
    if r.get('archive_format') == 'tar.gz':
        with tarfile.open(root / r['bundle_filename'], 'r|gz') as handle:
            for member in handle:
                if member.name not in expected:
                    continue
                raw = handle.extractfile(member).read()
                entry = expected.pop(member.name)
                if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                    raise ValueError('Status digest differs: ' + member.name)
                captured[member.name] = base64.b64encode(raw).decode('ascii')
                row = status_row(raw, str(root / r['bundle_filename']) + '::' + member.name, cutoff)
                if row:
                    yield row
                if not expected:
                    break
        if expected:
            raise ValueError('Missing portable status members: ' + str(sorted(expected)))
    else:
        for relative, entry in expected.items():
            p = root / relative
            raw = p.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Status digest differs: ' + str(p))
            captured[relative] = base64.b64encode(raw).decode('ascii')
            row = status_row(raw, str(p), cutoff)
            if row:
                yield row
    if cache is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.new')
        temporary.write_text(json.dumps(dict(manifest_sha256=r['manifest_sha256'], statuses=captured)) + '\n')
        os.replace(temporary, cache)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-root', type=Path, default=Path('/data/yyf-zyq/PHM_20261010'))
    parser.add_argument('--run-root', type=Path, default=Path('/srv/yyf-zyq/PHM_20261010'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--as-of')
    parser.add_argument('--cache-dir', type=Path,
                        default=os.environ.get('PHM_CAPACITY_CACHE'),
                        help='Optional cache of immutable archived status bytes, checked against the verified manifest on every read; live statuses are always read directly.')
    args = parser.parse_args()
    now = date(args.as_of) if args.as_of else datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    rows, errors, coverage = [], [], []
    for root in sorted(args.archive_root.iterdir()):
        if root.is_dir():
            try:
                rows.extend(archive_rows(root, cutoff, errors, coverage, args.cache_dir) or [])
            except Exception as exc:
                errors.append(dict(path=str(root), error=str(exc)))
    for base, dirs, files in os.walk(args.run_root):
        dirs[:] = [x for x in dirs if x not in EXCLUDED]
        if 'status.json' in files:
            p = Path(base) / 'status.json'
            try:
                row = status_row(p.read_bytes(), str(p), cutoff)
                if row:
                    rows.append(row)
            except Exception as exc:
                errors.append(dict(path=str(p), error=str(exc)))
    unique = {}
    for row in rows:
        key = (row['job_id'], row['started_utc'], row['pid'])
        if key in unique:
            unique[key].setdefault('duplicate_sources', []).append(row['source'])
        else:
            unique[key] = row
    rows = sorted(unique.values(), key=lambda r: r['started_utc'])
    counted = [x for x in rows if x['counted_for_capacity']]
    report = dict(as_of_utc=now.isoformat(), cutoff_utc=cutoff.isoformat(), rows=rows,
                  counted_primary_sessions=len(counted),
                  by_terminal_state=dict(Counter(x['status'] for x in counted)),
                  no_task_work_evidence_records=sum(not x['counted_for_capacity'] for x in rows),
                  archive_coverage=coverage, errors=errors,
                  caveats=['No-work-evidence records do not imply zero requests or cost.',
                           'Timeouts and live running sessions are reserved conservatively.',
                           'Role-name filtering and unreceipted legacy archives need explicit review before new launch.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'archive_coverage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
