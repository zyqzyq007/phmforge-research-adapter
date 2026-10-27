"""Evaluate frozen A185 policies with the qualified original IMP-MARL controller."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


ARCH = Path('/data/yyf-zyq/PHM_20261010')
ORIGINAL = ARCH / '20261017T_background129/imprl129'
DEPS = ARCH / '20261016T220500Z_inventory127/infra_runtime127/deps'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_worker(request_path, group_id):
    """Separate processes keep the original simulator's global RNG independent."""
    sys.path.insert(0, '/controller')
    import evaluate129 as original
    request = json.loads(request_path.read_text())
    group = next(g for g in request['groups'] if g['id'] == group_id)
    config = json.loads(Path('/policies', group_id, 'config.json').read_text())
    policy = Path('/policies', group_id, 'policy.py') if not group['native'] else None
    rows = []
    for index, seed in enumerate(request['seeds']):
        out = Path('/eval') / f'episode_{index:03}'
        result = original.run(policy, config, seed, out, native=group['native'], manual=True)
        rows.append(result)
        put(Path('/eval/PROGRESS.json'), {'group': group_id, 'attempted': len(rows),
            'planned': len(request['seeds']), 'complete': sum(r['complete'] for r in rows)})
    put(Path('/eval/SUMMARY.json'), {'group': group_id, 'native': group['native'],
        'policy_sha256': sha(policy) if policy else None, 'episodes': rows,
        'attempted': len(rows), 'complete': sum(r['complete'] for r in rows)})


def run_group(root, controller, policies, group):
    out = root / group['id']
    out.mkdir()
    runtime = out / 'runtime'
    runtime.mkdir()
    command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session']
    for name in ['/usr', '/bin', '/lib', '/lib64']:
        if Path(name).exists():
            command += ['--ro-bind', name, name]
    command += ['--ro-bind', str(DEPS), '/opt/numdeps',
                '--ro-bind', str(ORIGINAL / 'source/imp_marl_original'), '/opt/imp_source',
                '--ro-bind', str(controller), '/controller',
                '--ro-bind', str(policies), '/policies', '--bind', str(out), '/eval',
                # bwrap needs this empty mount point for its nested root setup.
                # It stays empty/read-only; every work/cache path remains /eval/runtime.
                '--dir', '/tmp', '--proc', '/proc', '--dev', '/dev', '--chdir', '/eval', '--remount-ro', '/',
                '/usr/bin/python3', '-I', '-B', '/controller/driver.py',
                '--request', '/controller/REQUEST.json', '--group', group['id']]
    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'TMPDIR': '/eval/runtime',
           'TMP': '/eval/runtime', 'TEMP': '/eval/runtime', 'MPLCONFIGDIR': '/eval/runtime',
           'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    put(out / 'COMMAND.json', {'argv': command, 'environment': env,
         'policy_process_limits': {'startup_seconds': 20, 'per_call_seconds': 5,
                                   'episode_seconds': 120, 'CPU_seconds': 60, 'memory_bytes': 2147483648}})
    with (out / 'stdout.txt').open('wb') as stdout, (out / 'stderr.txt').open('wb') as stderr:
        process = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    result = {'group': group['id'], 'returncode': process.returncode,
              'summary_exists': (out / 'SUMMARY.json').exists()}
    put(out / 'PROCESS_STATUS.json', result)
    return result


def evaluate(run, qualify=False, label=None):
    name = 'imprl_qualification185' if qualify else 'imprl_scoring185'
    if label:
        assert qualify and label.isalnum()
        name += '_' + label
    if not qualify:
        assert (run / 'PRE_SCORE_AUDIT185.json').is_file(), 'All16 freeze and scoring gate required'
    root = run / name
    root.mkdir(exist_ok=False)
    controller = root / 'controller'
    controller.mkdir()
    for filename in ['evaluate129.py', 'worker129.py']:
        shutil.copy2(ORIGINAL / 'controller' / filename, controller / filename)
    shutil.copy2(Path(__file__), controller / 'driver.py')
    policies = root / 'policies'
    policies.mkdir()
    settings = ['c3_k2_campaign', 'c5_k4_campaign']
    groups = []
    if qualify:
        seeds = [1]
        for setting in settings:
            for native in [False, True]:
                ident = setting + ('_native' if native else '_starter')
                groups.append({'id': ident, 'native': native, 'setting': setting})
                dst = policies / ident
                dst.mkdir()
                source = ORIGINAL / 'workspaces' / setting / 'inputs'
                shutil.copy2(source / 'config.json', dst / 'config.json')
                if not native:
                    shutil.copy2(source / 'starter_policy.py', dst / 'policy.py')
    else:
        from prepare_validation_scope_review import live_readers
        assert (run / 'PRE_SCORE_AUDIT185.json').is_file(), 'All16 freeze and scoring gate required'
        assert not live_readers([run / 'workspaces', run / 'review_workspaces', run / 'source_history'])
        seeds = json.loads((run / 'score_data/IMPRL_SEEDS185.json').read_text())['final_seeds']
        assert len(seeds) == 64 and len(set(seeds)) == 64 and not set(seeds).intersection(range(1, 65))
        audit = json.loads((run / 'SCOPE_AUDIT185.json').read_text())
        all_jobs = []
        for plan in ['PLAN185.json', 'REVIEW_PLAN185.json']:
            all_jobs += json.loads((run / plan).read_text())['jobs']
        for job in all_jobs:
            if job['family'] != 'imprl':
                continue
            source = run / job['workspace']
            policy = source / 'policy.py'
            record = next(a for a in audit['jobs'][job['id']]['artifacts']
                          if a['path'] == str(policy.relative_to(run)))
            assert policy.is_file() and not policy.is_symlink() and sha(policy) == record['sha256']
            assert policy.stat().st_size <= 2097152
            setting = job['setting']
            group = {'id': job['id'], 'native': False, 'setting': setting, 'phase': job.get('phase', 'initial')}
            groups.append(group)
            dst = policies / group['id']
            dst.mkdir()
            shutil.copy2(policy, dst / 'policy.py')
            original_config = ORIGINAL / 'workspaces' / setting / 'inputs/config.json'
            assert sha(source / 'inputs/config.json') == sha(original_config)
            shutil.copy2(original_config, dst / 'config.json')
        for setting in settings:
            group = {'id': setting + '_native_reference', 'native': True, 'setting': setting}
            groups.append(group)
            dst = policies / group['id']
            dst.mkdir()
            shutil.copy2(ORIGINAL / 'workspaces' / setting / 'inputs/config.json', dst / 'config.json')
        assert len(groups) == 6
    request = {'created_utc': datetime.now(timezone.utc).isoformat(), 'qualification_only': qualify,
               'seeds': seeds, 'groups': groups, 'episodes_planned': len(seeds) * len(groups),
               'model_calls': 0, 'source_controller_sha256': sha(controller / 'evaluate129.py'),
               'limits': 'Original reward/dynamics, centralized observation interface; not MARL training. '
                         'Paired seeds need not give identical physical events after different actions.'}
    put(controller / 'REQUEST.json', request)
    with ThreadPoolExecutor(max_workers=6) as executor:
        statuses = list(executor.map(lambda g: run_group(root, controller, policies, g), groups))
    put(root / 'PROCESS_SUMMARY.json', {'groups': statuses, 'planned_episode_attempts': request['episodes_planned']})
    print(json.dumps({'root': str(root), 'groups': statuses, 'model_calls': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--label')
    parser.add_argument('--request', type=Path)
    parser.add_argument('--group')
    args = parser.parse_args()
    if args.request:
        group_worker(args.request, args.group)
    else:
        assert args.run
        evaluate(args.run.resolve(), args.qualify, args.label)
