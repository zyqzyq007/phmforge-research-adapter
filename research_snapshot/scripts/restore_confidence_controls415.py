"""Restore only the predeclared probability-control evidence, verifying source hashes."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--tag', default='415', choices=['415', '416'])
    args = parser.parse_args()
    run = args.run
    plan = json.loads((run / ('SOURCE_PLAN' + args.tag + '.json')).read_text())
    started = time.monotonic()
    bindings = []

    def restore(key, spec):
        archive = Path(spec['archive'])
        receipt = json.loads((archive / 'ARCHIVE_VERIFIED.json').read_text())
        manifest_bytes = (archive / 'SOURCE_MANIFEST.json').read_bytes()
        assert hashlib.sha256(manifest_bytes).hexdigest() == receipt['manifest_sha256']
        (run / 'source' / (key + '_manifest.json')).write_bytes(manifest_bytes)
        expected = {x['path']: x for x in spec['files']}
        source_entries = {x['path']: x for x in json.loads(manifest_bytes)['files']}
        assert all(source_entries[k] == v for k, v in expected.items())
        pending = set(expected)
        done = []

        def accept(name, data):
            assert time.monotonic() - started < 180, 'Restoration wall budget reached'
            entry = expected[name]
            assert len(data) == entry['bytes']
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
            dest = run / 'inputs' / key / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            pending.remove(name)
            done.append(dict(archive=str(archive), member=name, **{k:entry[k] for k in ['bytes','sha256']},
                             local=str(dest.relative_to(run))))

        for name in sorted(tuple(pending)):
            if (archive / name).is_file():
                accept(name, (archive / name).read_bytes())
        if pending:
            with tarfile.open(archive / receipt['bundle_filename'], 'r|gz') as handle:
                for member in handle:
                    assert time.monotonic() - started < 180, 'Restoration wall budget reached'
                    if member.name in pending:
                        assert member.isfile()
                        accept(member.name, handle.extractfile(member).read())
                        if not pending:
                            break
        assert not pending, sorted(pending)
        return done

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(restore, k, v) for k, v in plan.items()]
        for future in futures:
            bindings.extend(future.result())
    result = dict(seconds=time.monotonic()-started, files=len(bindings),
                  bytes=sum(x['bytes'] for x in bindings), bindings=bindings,
                  new_fits=0, new_model_forwards=0, new_LLM=0)
    assert result['bytes'] < 67108864
    (run / ('RESTORED' + args.tag + '.json')).write_text(json.dumps(result, indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='bindings'}, flush=True)


if __name__ == '__main__':
    main()
