"""Relocatable, hash-verified access to the actual research environment.

The external object store contains original data, fitted weights, task inputs,
programs and execution records. It is deliberately separate from the agent's
filesystem. Only a recorded command's mounts enter its fresh sandbox.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import threading

from .io import file_sha256, write_json


def _relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ("..", "") for p in path.parts):
        raise ValueError(f"Unsafe snapshot path: {value}")
    return path.as_posix()


def _equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right): return False
        if left and all(isinstance(r, dict) and 'sample_id' in r for r in left + right):
            left, right = sorted(left, key=lambda r: str(r['sample_id'])), sorted(right, key=lambda r: str(r['sample_id']))
        return all(_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)
    return left == right


class Environment:
    def __init__(self, store):
        self.root = Path(store).resolve()
        ready = json.loads((self.root / 'READY.json').read_text())
        path = self.root / 'environment.json.gz'
        if file_sha256(path) != ready['manifest_sha256']:
            raise ValueError('Environment manifest digest differs from READY.json')
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            self.manifest = json.load(stream)
        if self.manifest['schema_version'] != 1:
            raise ValueError('Unsupported environment schema')
        self.files = self.manifest['files']
        self.aliases = sorted(self.manifest['aliases'].items(), key=lambda pair: -len(pair[0]))
        self.executables = set(self.manifest.get('executable_files', []))
        self.verified = set()
        self.ready = ready
        self._cases = None
        self._replay_mounts = {}
        self._mount_locks = defaultdict(threading.Lock)

    def replay_mount(self, key, destination):
        """Share verified read-only trees across a batch's independent sandboxes."""
        with self._mount_locks[key]:
            saved = self._replay_mounts.get(key)
            if saved is not None and saved.exists(): return saved
            saved = self.materialize(key, destination)
            self._replay_mounts[key] = saved
            return saved

    def object(self, key):
        key = _relative(key)
        row = self.files[key]
        digest = row['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid content digest')
        path = self.root / 'objects' / digest[:2] / digest[2:]
        if digest not in self.verified:
            if path.stat().st_size != row['bytes'] or file_sha256(path) != digest:
                raise ValueError(f'Corrupt or incomplete environment object: {key}')
            self.verified.add(digest)
        return path

    def read_json(self, key):
        return json.loads(self.object(key).read_text())

    def translate(self, original):
        original = str(original).rstrip('/')
        for old, new in self.aliases:
            if original == old or original.startswith(old + '/'):
                return _relative(new + original[len(old):])
        if original in self.files or any(k.startswith(original + '/') for k in self.files):
            return _relative(original)
        raise FileNotFoundError(f'No frozen dependency for {original}')

    def materialize(self, prefix, destination, writable=False):
        """Restore a file/tree; writable program copies never share object inodes."""
        prefix = _relative(prefix)
        dest = Path(destination).resolve()
        if dest == self.root or self.root.is_relative_to(dest) or dest.is_relative_to(self.root):
            raise ValueError('Materialized files must be outside the immutable store')
        matches = [k for k in self.files if k == prefix or k.startswith(prefix + '/')]
        links = {k: v for k, v in self.manifest.get('symlinks', {}).items()
                 if k == prefix or k.startswith(prefix + '/')}
        if not matches and not links:
            if prefix in self.manifest.get('empty_directories', []):
                dest.mkdir(parents=True, exist_ok=True)
                return dest
            raise FileNotFoundError(prefix)
        for key in matches:
            _relative(key)
            target = dest if key == prefix else dest / key[len(prefix) + 1:]
            if key != prefix and not target.resolve().is_relative_to(dest):
                raise ValueError(f'Escaping restored file: {key}')
            if target.exists():
                if target.is_file() and file_sha256(target) == self.files[key]['sha256']:
                    continue
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = self.object(key)
            if writable:
                shutil.copyfile(source, target)
                target.chmod(0o755 if key in self.executables else 0o644)
            else:
                # The frozen object is never writable by a replayed process.
                source.chmod(0o555 if key in self.executables else 0o444)
                try: os.link(source, target)
                except OSError: shutil.copy2(source, target)
        for key, link in links.items():
            _relative(key)
            target = dest if key == prefix else dest / key[len(prefix) + 1:]
            if target.is_symlink():
                if os.readlink(target) == link: continue
                raise FileExistsError(target)
            if PurePosixPath(link).is_absolute():
                if link.startswith(('/app/', '/data/datasets/', '/reference/', '/tools/')):
                    # These are original *sandbox* paths. Preserve the link but
                    # never dereference it in the host coordinator.
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(link)
                else:
                    self.materialize(self.translate(link), target, writable=writable)
            else:
                candidate = (target.parent / link).resolve()
                if not candidate.is_relative_to(dest if prefix not in self.files else dest.parent):
                    raise ValueError(f'Escaping relative link: {key}')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(link)
        return dest

    def verify(self, full=False, workers=4):
        unique = {}
        for key, row in self.files.items(): unique.setdefault(row['sha256'], (key, row))
        def one(item):
            digest, (key, row) = item
            path = self.root / 'objects' / digest[:2] / digest[2:]
            if not path.is_file() or path.stat().st_size != row['bytes']:
                raise ValueError(f'Missing/wrong-size object: {key}')
            if full and file_sha256(path) != digest:
                raise ValueError(f'Object digest mismatch: {key}')
        with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
            list(pool.map(one, unique.items()))
        return {'status': 'verified', 'mode': 'full_sha256' if full else 'manifest_and_sizes',
                'files': len(self.files), 'objects': len(unique), 'bytes': sum(row['bytes'] for _, row in unique.values()),
                'archives': len(self.manifest['archives'])}

    def records(self):
        """Enumerate executable evidence, not fabricated benchmark completions."""
        suffixes = ('/final_command.json', '/reference_command.json', '/PROCESS.json')
        return [key for key in self.files if key.startswith('runs/') and (key.endswith(suffixes)
                or (key.endswith('/command.json') and any(x in key for x in ['/evaluation278/', '/evaluation315/'])))]

    def cases(self):
        if self._cases is not None: return self._cases
        cases = []
        command_records = [k for k in self.files if k.endswith(('/final_command.json', '/command.json'))]
        for key in self.files:
            if not key.endswith('/PLAN187.json') or not key.startswith('runs/'):
                continue
            if any(x in key.split('/') for x in ['source', 'audit', 'attempts', 'workspace_snapshot']):
                continue
            plan = self.read_json(key)
            if not isinstance(plan, dict) or not isinstance(plan.get('jobs'), list): continue
            base = str(PurePosixPath(key).parent)
            for job in plan['jobs']:
                if job.get('family') not in ('forge_real_coding', 'forge_verified', 'forge_measured_rul', 'forge_real_rul_coding', 'real_rul_review'):
                    continue
                ident = base + '/' + job['id']
                candidates = [k for k in command_records if k.startswith(base + '/evaluation') and k.endswith('/final_command.json')
                              and (('/' + job['id'] + '/') in k or len(plan['jobs']) == 1)]
                if not candidates:
                    candidates = [k for k in command_records if k.startswith(base + '/evaluation') and k.endswith('/command.json')
                                  and (('/' + job['id'] + '/') in k or len(plan['jobs']) == 1)]
                    final_only = [k for k in candidates if '/final/' in k]
                    if final_only: candidates = final_only
                if not candidates:
                    parent = str(PurePosixPath(base).parent)
                    seal_key = parent+'/ALL_CHOICES_SEALED278.json'
                    if seal_key in self.files:
                        variants = self.read_json(seal_key)['variants']
                        relative = PurePosixPath(base).name+'/'+job['workspace']
                        variants = [v for v in variants if v.get('workspace') == relative and v['id'].startswith(('agent_', 'wrapper_'))]
                        variants.sort(key=lambda v: not v['id'].startswith('agent_'))
                        if variants:
                            prefix = parent+'/evaluation278/'+variants[0]['id']+'/final/'
                            candidates = [k for k in command_records if k.startswith(prefix) and k.endswith('/command.json')]
                audit_key = base + '/CODING_CHOICE_AUDIT.json'
                state = None
                if audit_key in self.files:
                    rows = self.read_json(audit_key).get('rows', [])
                    state = next((r.get('status') for r in rows if r.get('job') == job['id']), None)
                cases.append({'id': ident, 'kind': 'legacy_coding', 'plan': key, 'job': job['id'], 'task_id': job.get('task_id'),
                              'dataset': job.get('dataset', job.get('task_id')), 'historical_status': state,
                              'replay_records': candidates, 'historical_workspace': job.get('workspace'),
                              'test_inputs_visible_during_development': job.get('test_inputs_visible_during_development', 'not_declared')})
        for key in self.files:
            if not key.startswith('runs/') or not key.endswith('/SESSIONS337.json'): continue
            sessions = self.read_json(key)
            if isinstance(sessions, dict): sessions = sessions.get('sessions', [])
            if not isinstance(sessions, list): continue
            base = str(PurePosixPath(key).parent)
            for row in sessions:
                if not isinstance(row, dict): continue
                ident = row.get('id', row.get('session'))
                if not ident: continue
                cases.append({'id': base+'/'+str(ident), 'kind': 'native_program_revision', 'plan': key,
                              'job': ident, 'task_id': None, 'dataset': row.get('dataset'),
                              'historical_status': row.get('status'), 'replay_records': [],
                              'historical_workspace': row.get('path'),
                              'test_inputs_visible_during_development': False})
        self._cases = cases
        return cases


def _parse_command(value):
    command = value if isinstance(value, list) else value.get('command')
    if not command or Path(command[0]).name != 'bwrap':
        raise ValueError('This record does not contain a bubblewrap execution command')
    mounts, options, pos = [], [], 1
    flags = {'--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts', '--unshare-net',
             '--die-with-parent', '--new-session', '--clearenv'}
    unary = {'--proc', '--dev', '--dir', '--chdir', '--remount-ro', '--hostname', '--uid', '--gid'}
    while pos < len(command):
        arg = command[pos]
        if arg in ('--ro-bind', '--bind', '--ro-bind-try', '--bind-try'):
            mounts.append((arg, command[pos+1], command[pos+2])); pos += 3
        elif arg == '--setenv':
            options += command[pos:pos+3]; pos += 3
        elif arg in unary:
            options += command[pos:pos+2]; pos += 2
        elif arg in flags:
            pos += 1
        elif arg == '--':
            pos += 1; break
        elif arg.startswith('--'):
            raise ValueError(f'Unsupported recorded sandbox option: {arg}')
        else: break
    if not command[pos:]: raise ValueError('Missing inner command')
    return mounts, options, command[pos:]


def replay(store, record, destination, timeout=300, compare=True, submission=None, artifacts=None):
    """Execute the saved program again; never substitute stored predictions."""
    env = store if isinstance(store, Environment) else Environment(store)
    if submission is not None and compare:
        raise ValueError('A new submission must be evaluated without claiming historical output equality')
    record = _relative(record)
    recorded = env.read_json(record)
    mounts, options, inner = _parse_command(recorded)
    out = Path(destination).resolve()
    if out.exists(): raise FileExistsError(out)
    if out.is_relative_to(env.root) or env.root.is_relative_to(out):
        raise ValueError('Replay output must be outside the immutable environment')
    if not shutil.which('bwrap'): raise FileNotFoundError('bubblewrap is required')
    out.mkdir(parents=True)
    (out/'output/tmp/mpl').mkdir(parents=True)
    (out/'output/home').mkdir()
    args = ['bwrap', '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts', '--unshare-net',
            '--die-with-parent', '--new-session', '--clearenv']
    old_output = None
    for number, (flag, source, target) in enumerate(mounts):
        if source in ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc'] and source == target:
            if Path(source).exists(): args += ['--ro-bind', source, target]
            continue
        if target in ['/output', '/work']:
            old_output = source
            args += ['--bind', str(out/'output'), target]
            continue
        if target == '/artifacts' and artifacts is not None:
            weights = Path(artifacts).resolve()
            if not weights.is_dir(): raise FileNotFoundError(weights)
            args += ['--ro-bind', str(weights), target]
            continue
        key = env.translate(source)
        writable = flag.startswith('--bind') and target == '/app'
        restored = out/'mounts'/str(number)
        if submission is not None and target in ['/app', '/program']:
            shutil.copytree(submission, restored, symlinks=True)
        elif not writable:
            restored = env.replay_mount(key, restored)
        else:
            env.materialize(key, restored, writable=writable)
        if writable:
            (restored/'.runtime/tmp').mkdir(parents=True, exist_ok=True)
            (restored/'.runtime/mpl').mkdir(parents=True, exist_ok=True)
        args += ['--bind' if writable else '--ro-bind', str(restored), target]
    mount_targets = {v[2] for v in mounts}
    work = '/work' if '/work' in mount_targets else '/output'
    variables = {'HOME': work+'/home', 'PATH': '/opt/python312/bin:/usr/bin:/bin',
                 'PYTHONPATH': '/opt/phm_python/extra:/opt/deps:/tools:/program', 'PYTHONDONTWRITEBYTECODE': '1',
                 'TMPDIR': work+'/tmp', 'TMP': work+'/tmp', 'TEMP': work+'/tmp',
                 'MPLCONFIGDIR': work+'/tmp/mpl', 'XDG_CACHE_HOME': work+'/tmp',
                 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    for k, v in variables.items(): args += ['--setenv', k, v]
    args += options
    # Preserve other recorded options but always keep caches out of /tmp.
    for k in ['TMPDIR', 'TMP', 'TEMP', 'MPLCONFIGDIR', 'XDG_CACHE_HOME']:
        args += ['--setenv', k, variables[k]]
    args += inner
    write_json(out/'COMMAND.json', {'source_record': record, 'command': args, 'credential_environment_passed': False})
    started = time.monotonic()
    result = {'source_record': record, 'status': 'failed', 'stored_predictions_used_as_execution': False,
              'historical_contract_preserved': True, 'new_agent_run': False,
              'submission_override': submission is not None, 'new_fitted_artifacts_used': artifacts is not None}
    try:
        with (out/'stdout.log').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
            proc = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=timeout)
        result['exit_code'] = proc.returncode
        if proc.returncode != 0: raise RuntimeError('Historical program failed; inspect stderr.log')
        prediction = out/'output/predictions.json'
        if '--output' in inner:
            name = PurePosixPath(inner[inner.index('--output')+1]).name
            prediction = out/'output'/name
            if not prediction.is_file():
                raise FileNotFoundError('Program exited without producing the requested output')
        if prediction.is_file():
            actual = json.loads(prediction.read_text())
            result.update(prediction_file=str(prediction), prediction_sha256=file_sha256(prediction))
            if compare and old_output:
                expected_key = env.translate(old_output+'/'+prediction.name)
                if expected_key not in env.files:
                    raise FileNotFoundError('Expected prediction not preserved: '+expected_key)
                expected = env.read_json(expected_key)
                content_a = actual.get('predictions', actual) if isinstance(actual, dict) else actual
                content_b = expected.get('predictions', expected) if isinstance(expected, dict) else expected
                result['matches_historical_predictions'] = _equal(content_a, content_b)
                if not result['matches_historical_predictions']:
                    raise ValueError('Cold predictions differ from historical predictions')
        audit = out/'output/training_audit.json'
        if 'fit' in inner and any('program_training_entry.py' in v for v in inner) and not audit.is_file():
            raise FileNotFoundError('Audited training exited without producing its new training audit')
        if audit.exists():
            values = json.loads(audit.read_text())
            result['training_audit'] = {k: values.get(k) for k in ['completed_estimator_fits', 'optimizer_steps', 'nonzero_sampled_optimizer_steps']}
            if 'fit' in inner:
                if not (values.get('completed_estimator_fits', 0) > 0 or values.get('nonzero_sampled_optimizer_steps', 0) > 0):
                    raise ValueError('Recorded training command did not actually fit/update parameters')
        result['status'] = 'completed'
        return result
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        write_json(out/'RESULT.json', result)


def main(args):
    if args.environment_command == 'replay':
        return replay(args.store, args.record, args.out, args.timeout, not args.no_compare)
    env = Environment(args.store)
    if args.environment_command in ('workspace', 'agent', 'replay-case'):
        from .workbench import prepare, run, replay_case
        if args.environment_command == 'workspace': return prepare(env, args.case, args.out)
        if args.environment_command == 'replay-case': return replay_case(env, args.case, args.out)
        return run(args.store, args.case, args.out, args.cli, args.model_base_url,
                   args.timeout, args.max_requests, args.max_input_tokens, args.max_output_tokens,
                   args.max_output_per_request, args.smoke)
    if args.environment_command == 'verify': return env.verify(args.full, args.workers)
    if args.environment_command == 'list':
        cases = env.cases()
        if args.dataset: cases = [v for v in cases if v['dataset'] == args.dataset]
        return {'cases': cases, 'count': len(cases), 'archive_count': len(env.manifest['archives']),
                'scope': env.manifest['scope'], 'records': env.records() if args.records else None}
    if args.environment_command == 'restore':
        prefix = env.translate(args.path) if args.path.startswith('/') else args.path
        env.materialize(prefix, args.out, writable=True)
        return {'status': 'restored', 'path': prefix, 'destination': str(args.out.resolve())}
    raise ValueError(args.environment_command)
