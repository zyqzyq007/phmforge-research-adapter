"""Archive one finished PHM run, verify every retained file, then clear its source."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def symlink_allowed(relative, target, policy, portable):
    rule = policy.get('preserve_portable_symlinks', {})
    # Per-member authorization preserves sandbox input links without widening
    # the legacy inputs/data pattern to arbitrary absolute targets.
    exact = rule.get('exact_entries', {})
    if (portable and rule.get('enabled') is True and
            isinstance(relative, str) and isinstance(target, str) and
            isinstance(exact, dict) and exact.get(relative) == target):
        member_path, target_path = PurePosixPath(relative), PurePosixPath(target)
        allowed_namespace = any(
            target_path != root and target_path.is_relative_to(root)
            for root in (PurePosixPath('/app/inputs/data'),
                         PurePosixPath('/opt/phm_data')))
        # Closed coding subjects can create input aliases inside their own
        # scratch directories. Require both an exact inventory and opt-in.
        app_workspace_input = (
            rule.get('preserve_exact_app_workspace_inputs') is True and
            len(member_path.parts) >= 5 and
            member_path.parts[0] == 'workspaces' and
            member_path.parts[2] == '.runtime' and
            any(target_path != root and target_path.is_relative_to(root)
                for root in (PurePosixPath('/app/data'),
                             PurePosixPath('/app/.runtime'))))
        pytest_alias = (
            rule.get('preserve_exact_pytest_aliases') is True and
            member_path.match('**/scratch/pytest*/*current') and
            member_path.name.startswith('test_') and
            len(target_path.parts) == 4 and target_path.parts[1] == 'workruntime' and
            target_path.parts[2] == member_path.parent.name and
            member_path.parent.name in ('pytest', 'pytest-shared') and
            target_path.name.startswith(member_path.name[:-len('current')]))
        dataset_input_alias = (
            rule.get('preserve_exact_dataset_input_aliases') is True and
            '.runtime' in member_path.parts and
            ('workspaces' in member_path.parts or 'app' in member_path.parts) and
            target_path != PurePosixPath('/data/datasets') and
            target_path.is_relative_to(PurePosixPath('/data/datasets')))
        if (relative not in ('', '.') and not member_path.is_absolute() and
                '..' not in member_path.parts and str(member_path) == relative and
                '\\' not in relative and '\x00' not in relative and
                target_path.is_absolute() and '..' not in target_path.parts and
                str(target_path) == target and '\\' not in target and '\x00' not in target and
                (allowed_namespace or app_workspace_input or pytest_alias or dataset_input_alias)):
            return True
    configured_target = rule.get('target', '')
    if not isinstance(configured_target, str):
        return False
    target_path = PurePosixPath(configured_target)
    namespace = PurePosixPath('/opt/phm_data')
    return (portable and not PurePosixPath(relative).is_absolute() and
            '..' not in PurePosixPath(relative).parts and
            rule.get('enabled') is True and
            target_path.is_absolute() and target_path != namespace and
            target_path.is_relative_to(namespace) and '..' not in target_path.parts and
            target == configured_target and any(
                PurePosixPath(relative).match(pattern) for pattern in rule.get('paths', [])))


def symlink_entry(run, path, policy, portable):
    """Preserve only declared runtime bindings or exact pytest scratch aliases."""
    relative = str(path.relative_to(run))
    target = str(path.readlink())
    if not symlink_allowed(relative, target, policy, portable):
        raise ValueError('Unauthorized retained symlink: ' + relative)
    raw = target.encode('utf-8')
    return dict(path=relative, target=target, bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def verify_saved_symlinks(run, manifest, policy, portable):
    seen = set()
    for item in manifest.get('symlinks', []):
        name = item['path']
        if (PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or
                name in seen or not (run / name).is_symlink()):
            raise RuntimeError('Invalid saved symlink inventory: ' + name)
        if symlink_entry(run, run / name, policy, portable) != item:
            raise RuntimeError('Saved symlink changed: ' + name)
        seen.add(name)


def verify_portable(bundle, expected, expected_dirs, symlinks, policy=None):
    """Read every member without extracting or following symbolic/hard links."""
    expected_links = {item['path']: item for item in symlinks}
    if any(not symlink_allowed(item['path'], item['target'], policy or {}, True)
           for item in symlinks):
        raise RuntimeError('Unauthorized portable symlink inventory')
    if (len(expected_links) != len(symlinks) or
            set(expected) & (set(expected_dirs) | set(expected_links)) or
            set(expected_dirs) & set(expected_links)):
        raise RuntimeError('Overlapping portable inventory')
    seen, seen_dirs, seen_links, all_seen = set(), set(), set(), set()
    with tarfile.open(bundle, 'r:gz') as handle:
        for member in handle:
            name = member.name
            if (name in all_seen or PurePosixPath(name).is_absolute() or
                    '..' in PurePosixPath(name).parts):
                raise RuntimeError('Duplicate or unsafe portable member: ' + name)
            all_seen.add(name)
            if member.isdir():
                if name not in expected_dirs or member.size != 0:
                    raise RuntimeError('Unexpected portable directory: ' + name)
                seen_dirs.add(name)
            elif member.issym():
                entry = expected_links.get(name)
                raw = member.linkname.encode('utf-8')
                if (entry is None or member.size != 0 or member.linkname != entry['target'] or
                        len(raw) != entry['bytes'] or
                        hashlib.sha256(raw).hexdigest() != entry['sha256']):
                    raise RuntimeError('Portable symlink verification failed: ' + name)
                seen_links.add(name)
            elif member.isfile():
                if name not in expected:
                    raise RuntimeError('Unexpected portable file: ' + name)
                entry = expected[name]
                h = hashlib.sha256()
                with handle.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(chunk)
                if member.size != entry['bytes'] or h.hexdigest() != entry['sha256']:
                    raise RuntimeError('Portable member verification failed: ' + name)
                seen.add(name)
            else:
                raise RuntimeError('Unexpected portable member type: ' + name)
    if seen != set(expected) or seen_dirs != set(expected_dirs) or seen_links != set(expected_links):
        raise RuntimeError('Portable archive inventory differs; source retained')
    return dict(logical_files_verified=len(seen), directories_verified=len(seen_dirs),
                symlinks_verified=len(seen_links))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_retained(run, destination):
    # Only the run-root cache is excluded by the retention policy.
    def ignore_root_cache(directory, names):
        return {'clients-cache'} if Path(directory).resolve() == run and 'clients-cache' in names else set()
    shutil.copytree(run, destination, ignore=ignore_root_cache)


def copy_portable(run, destination, entries, manifest_path, compression_level=0):
    """Preserve upstream filenames which the archive filesystem cannot store."""
    retained = [p for p in sorted(run.rglob('*'))
                if p.relative_to(run).parts[0] != 'clients-cache']
    symlinks = json.loads(manifest_path.read_text()).get('symlinks', [])
    actual_links = [p for p in retained if p.is_symlink()]
    policy = json.loads((run / 'retention_policy.json').read_text()) if actual_links else {}
    actual_inventory = [symlink_entry(run, p, policy, True) for p in actual_links]
    if actual_inventory != symlinks:
        raise RuntimeError('Portable symlink inventory differs from source')
    destination.mkdir()
    bundle = destination / 'RUN_CONTENTS.tar.gz'
    with tarfile.open(bundle, 'w:gz', dereference=False, compresslevel=compression_level) as handle:
        for p in retained:
            handle.add(p, arcname=str(p.relative_to(run)), recursive=False)
    expected = {e['path']: e for e in entries}
    expected[manifest_path.name] = dict(bytes=manifest_path.stat().st_size,
                                       sha256=digest(manifest_path))
    expected_dirs = {str(p.relative_to(run)) for p in retained if p.is_dir() and not p.is_symlink()}
    verified = verify_portable(bundle, expected, expected_dirs, symlinks, policy)
    shutil.copy2(manifest_path, destination / manifest_path.name)
    if digest(destination / manifest_path.name) != digest(manifest_path):
        raise RuntimeError('Portable external manifest differs; source retained')
    return dict(archive_format='tar.gz', bundle_filename=bundle.name,
                bundle_compression_level=compression_level,
                bundle_sha256=digest(bundle), bundle_bytes=bundle.stat().st_size,
                **verified)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--reuse-snapshot', action='store_true')
    parser.add_argument('--manifest-name', default='SOURCE_MANIFEST.json')
    parser.add_argument('--portable-bundle', action='store_true')
    parser.add_argument('--compression-level', type=int, choices=range(10), default=0,
                        help='Gzip level for portable bundles; 0 avoids recompressing large NPZ/model artifacts. Content verification is unchanged.')
    args = parser.parse_args()
    run = args.run.resolve()
    base = Path('/srv/yyf-zyq/PHM_20261010').resolve()
    if run.parent != base or run.name.startswith('_') or not run.is_dir():
        raise ValueError('Expected one finished run directly under the managed experiment directory')
    destination = (args.destination or (Path('/data/yyf-zyq/PHM_20261010') / run.name)).resolve()
    if destination.parent != Path('/data/yyf-zyq/PHM_20261010').resolve():
        raise ValueError('Archive destination must be a new folder under the managed archive directory')
    if destination.exists():
        raise FileExistsError(destination)
    if Path(args.manifest_name).name != args.manifest_name or not args.manifest_name.endswith('.json'):
        raise ValueError('Manifest name must be a JSON filename')
    statuses = set(run.rglob('status.json')) | set(run.glob('*_status.json'))
    for p in statuses:
        payload = json.loads(p.read_text())
        records = payload if isinstance(payload, list) else [payload]
        if any(isinstance(item, dict) and item.get('status') == 'running' for item in records):
            raise RuntimeError('Cannot archive a running model process: ' + str(p))
    policy = json.loads((run / 'retention_policy.json').read_text())
    if policy['excluded_rebuildable_directory'] != 'clients-cache':
        raise ValueError('Unexpected retention exclusion')
    if not Path(policy['shared_dependencies']['path']).is_dir():
        raise FileNotFoundError('Shared dependencies missing')
    workspace = args.workspace.resolve()
    snapshot = run / 'workspace_snapshot'
    if args.reuse_snapshot:
        old_manifest = run / 'SOURCE_MANIFEST.json'
        if not snapshot.is_dir() or not old_manifest.is_file():
            raise ValueError('Recovery requires the original frozen snapshot and source manifest')
        old_payload = json.loads(old_manifest.read_text())
        for item in old_payload['files']:
            p = run / item['path']
            if p.is_symlink() or not p.is_file() or p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
                raise RuntimeError('Original frozen source changed; recovery stopped: ' + item['path'])
        verify_saved_symlinks(run, old_payload, policy, args.portable_bundle)
    else:
        snapshot.mkdir(exist_ok=False)
        for name in ['AGENTS.md', 'AGNETS.md', 'docs', 'scripts', 'cases', 'logs', 'results']:
            source = workspace / name
            target = snapshot / name
            if source.is_dir():
                shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            elif source.is_file():
                shutil.copy2(source, target)
    entries, symlinks = [], []
    for path in sorted(run.rglob('*')):
        relative = path.relative_to(run)
        if relative.parts[0] == 'clients-cache':
            continue
        if path.is_symlink():
            symlinks.append(symlink_entry(run, path, policy, args.portable_bundle))
        elif path.is_file():
            entries.append(dict(path=str(relative), bytes=path.stat().st_size, sha256=digest(path)))
    manifest = dict(run_id=run.name, created_utc=datetime.now(timezone.utc).isoformat(),
                    excluded=['clients-cache'], file_count=len(entries),
                    bytes=sum(item['bytes'] for item in entries), files=entries,
                    symlinks=symlinks, symlink_count=len(symlinks))
    manifest_path = run / args.manifest_name
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    manifest_hash = digest(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    portable = {}
    if args.portable_bundle:
        portable = copy_portable(run, destination, entries, manifest_path, args.compression_level)
    else:
        copy_retained(run, destination)
        for item in entries:
            p = destination / item['path']
            if not p.is_file() or p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
                raise RuntimeError('Archive verification failed; retained source: ' + item['path'])
        if digest(destination / args.manifest_name) != manifest_hash:
            raise RuntimeError('Manifest verification failed; retained source')
        expected = {item['path'] for item in entries} | {args.manifest_name}
        actual = {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}
        if actual != expected:
            raise RuntimeError('Archive file inventory differs; retained source')
    report = dict(status='verified', run_id=run.name, archive_path=str(destination),
                  files_verified=len(entries), retained_bytes=manifest['bytes'],
                  manifest_sha256=manifest_hash, verified_utc=datetime.now(timezone.utc).isoformat(),
                  manifest_filename=args.manifest_name,
                  symlinks_verified=0,
                  shared_dependencies=policy['shared_dependencies'], source_cleanup='pending')
    report.update(portable)
    report_path = destination / 'ARCHIVE_VERIFIED.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    shutil.rmtree(run)
    report['source_cleanup'] = 'complete'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
