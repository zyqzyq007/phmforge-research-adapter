"""Bounded retrieval of the complete, pinned official MLE-STAR example."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
import urllib.request


def main(run):
    cfg = json.loads((run / 'PLAN430.json').read_text())
    tree = json.loads((run / 'TREE430.json').read_text())
    assert tree['sha'] == cfg['commit'] and not tree.get('truncated')
    entries = [x for x in tree['tree'] if x['type'] == 'blob' and x['path'].startswith(cfg['prefix'])]
    entries += [x for x in tree['tree'] if x['type'] == 'blob' and x['path'] == 'LICENSE']
    assert len(entries) + 2 <= cfg['network_request_budget']
    assert sum(x['size'] for x in entries) + (run/'TREE430.json').stat().st_size + (run/'paper/paper.html').stat().st_size <= cfg['source_bytes_cap']
    assert not (run/'SOURCE_FETCH430.json').exists(), 'No implicit retry'
    def get(item):
        started = time.monotonic()
        relative = item['path'].removeprefix(cfg['prefix'])
        assert '..' not in PurePosixPath(relative).parts and not PurePosixPath(relative).is_absolute()
        target = run/'source'/relative
        url = f"https://raw.githubusercontent.com/{cfg['source_repository']}/{cfg['commit']}/{item['path']}"
        row = dict(path=relative, url=url, expected_git_blob=item['sha'], expected_bytes=item['size'])
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PHM-source-reproduction'})
            with urllib.request.urlopen(req, timeout=cfg['per_request_seconds']) as f:
                data = f.read(item['size']+1)
                row['http_status'] = f.status
            assert len(data) == item['size'], 'Size mismatch'
            blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert blob == item['sha'], 'Pinned Git blob mismatch'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            row.update(status='verified', bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), git_blob=blob)
        except Exception as exc:
            row.update(status='failed', error=repr(exc))
        row['seconds'] = time.monotonic()-started
        return row
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=cfg['parallel']) as executor:
        rows = list(executor.map(get, entries))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), rows=rows, files=len(entries),
                  passed=all(r['status']=='verified' for r in rows), seconds=time.monotonic()-started,
                  source_bytes=sum(r.get('bytes', 0) for r in rows), new_LLM_calls=0, new_PHM_fits=0)
    (run/'SOURCE_FETCH430.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)
    if not result['passed']:
        print(json.dumps([r for r in rows if r['status']!='verified']), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
