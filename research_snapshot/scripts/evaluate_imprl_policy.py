"""Independent original-environment rollouts; submitted policy cannot see truth."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from imprl_reference import Reference, load_module


class IsolatedPolicy:
    def __init__(self, source, work, output, worker=None, runtime_mounts=(), workspace_alias=None):
        # Preserve all submitted dependencies, including planning caches under
        # .runtime. Removing a required cache could create an evaluator failure.
        shutil.copytree(source, work, ignore=shutil.ignore_patterns('__pycache__'))
        self.manifest = []
        for p in sorted(work.rglob('*')):
            if p.is_file():
                self.manifest.append(dict(file=str(p.relative_to(work)), bytes=p.stat().st_size,
                                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        runtime = work/'.runtime'
        for name in ['tmp', 'mpl']:
            (runtime/name).mkdir(parents=True, exist_ok=True)
        worker = Path(worker or Path(__file__).with_name('imprl_policy_worker.py')).resolve()
        cmd = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session']
        for p in [Path('/usr'), Path('/bin'), Path('/lib'), Path('/lib64'), Path('/etc'), Path.home()/'.local/lib']:
            if p.exists():
                cmd += ['--ro-bind', str(p), str(p)]
        for source_path, destination in runtime_mounts:
            cmd += ['--ro-bind', str(source_path), str(destination)]
        cmd += ['--bind', str(work), str(work)]
        if workspace_alias is not None:
            if workspace_alias != '/app':
                raise ValueError('Only the documented /app workspace alias is supported')
            cmd += ['--bind', str(work), workspace_alias]
        visible_work = Path(workspace_alias) if workspace_alias else work
        cmd += ['--ro-bind', str(worker), '/worker.py',
                '--proc', '/proc', '--dev', '/dev', '--chdir', str(visible_work), '--remount-ro', '/',
                sys.executable, '-u', '/worker.py']
        env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in
               ['KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PROXY'])}
        visible_runtime = visible_work/'.runtime'
        env.update(TMPDIR=str(visible_runtime/'tmp'), TMP=str(visible_runtime/'tmp'), TEMP=str(visible_runtime/'tmp'),
                   MPLCONFIGDIR=str(visible_runtime/'mpl'), PYTHONDONTWRITEBYTECODE='1',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        env.pop('PYTHONPATH', None)
        if runtime_mounts:
            env['PYTHONPATH'] = ':'.join(str(destination) for _, destination in runtime_mounts)
        self.err = (output/'policy_stderr.txt').open('w')
        self.start = time.monotonic()
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.err, text=True, bufsize=1, env=env, start_new_session=True)
        (output/'submitted_manifest.json').write_text(json.dumps(self.manifest, indent=2)+'\n')
        (output/'policy_command.json').write_text(json.dumps(cmd, indent=2)+'\n')
        try:
            assert self.read(300).get('status') == 'ready'
        except BaseException:
            self.close()
            raise

    def read(self, timeout):
        left = min(timeout, 900-(time.monotonic()-self.start))
        if left <= 0 or not select.select([self.p.stdout], [], [], left)[0]:
            raise TimeoutError('Frozen policy inference budget exhausted')
        reply = json.loads(self.p.stdout.readline())
        if 'error' in reply:
            raise RuntimeError(str(reply))
        return reply

    def request(self, value):
        self.p.stdin.write(json.dumps(value)+'\n')
        self.p.stdin.flush()
        return self.read(20)

    def reset(self):
        self.request({'op': 'reset'})

    def act(self, belief):
        return self.request({'op': 'act', 'belief': belief.tolist()})['action']

    def close(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGTERM)
            try:
                self.p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait()
        self.err.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--setting', required=True)
    p.add_argument('--policy', choices=['reference', 'do_nothing', 'submitted'], required=True)
    p.add_argument('--job')
    p.add_argument('--episodes', type=int, default=512)
    p.add_argument('--seed-start', type=int, default=10000)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--label')
    args = p.parse_args()
    name = args.label or args.job or f'{args.setting}_{args.policy}'
    output = args.run/'evaluation'/name
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = dict(name=name, setting=args.setting, policy=args.policy, planned_episodes=args.episodes, steps=args.steps,
                  seed_start=args.seed_start, status='running', started_utc=datetime.now(timezone.utc).isoformat())
    (output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    cfg = json.loads((args.run/'tasks'/f'{args.setting}_r1/inputs/config.json').read_text())
    core = load_module('original_imprl', args.run/'sources/imprl/imprl/envs/structural_envs/k_out_of_n_infinite.py')
    kwargs = dict(percept_type='belief', time_limit=args.steps, reward_shaping=True, is_failure_self_announcing=False)
    policy = None
    episodes = []
    try:
        if args.policy == 'submitted':
            policy = IsolatedPolicy(args.run/'tasks'/args.job, args.run/'evaluation_workspaces'/name, output)
        elif args.policy == 'reference':
            policy = Reference(core.KOutOfN(cfg, **kwargs), args.run/'sources/reference_members'/args.setting)
        with gzip.open(output/'trajectories.jsonl.gz', 'wt') as trajectories:
            for seed in range(args.seed_start, args.seed_start+args.episodes):
                np.random.seed(seed)
                env = core.KOutOfN(cfg, **kwargs)
                belief, _ = env.reset()
                if hasattr(policy, 'reset'):
                    policy.reset()
                cost, failed, replaced, inspected = 0., 0, 0, 0
                for t in range(args.steps):
                    action = np.asarray(policy.act(belief.copy()) if policy else [0]*cfg['n_components'])
                    assert action.shape == (cfg['n_components'],) and np.all(np.isin(action, [0, 1, 2])), 'Illegal action'
                    action = action.astype(int).tolist()
                    before, hidden = belief.copy(), env.damage_state.copy()
                    belief, reward, terminated, truncated, info = env.step(action)
                    cost -= cfg['discount_factor']**t * reward
                    failed += bool(info['system_failure'])
                    replaced += action.count(1)
                    inspected += action.count(2)
                    trajectories.write(json.dumps(dict(seed=seed, t=t, belief=before.tolist(), action=action,
                        hidden_state=hidden.tolist(), observation=info['observation'].tolist(), reward=float(reward),
                        system_failure=bool(info['system_failure']), next_belief=belief.tolist()))+'\n')
                episodes.append(dict(seed=seed, cost=float(cost), failure_steps=failed,
                                     replacements=replaced, inspections=inspected))
                if len(episodes) % 64 == 0:
                    (output/'episodes.json').write_text(json.dumps(episodes)+'\n')
                    print(json.dumps({'name': name, 'completed': len(episodes)}), flush=True)
        status.update(status='finished', mean_cost=float(np.mean([x['cost'] for x in episodes])),
                      mean_failure_steps=float(np.mean([x['failure_steps'] for x in episodes])),
                      mean_replacements=float(np.mean([x['replacements'] for x in episodes])),
                      mean_inspections=float(np.mean([x['inspections'] for x in episodes])))
    except BaseException as exc:
        status.update(status='evaluation_error', error_type=type(exc).__name__, error=str(exc))
    finally:
        if isinstance(policy, IsolatedPolicy):
            policy.close()
        (output/'episodes.json').write_text(json.dumps(episodes)+'\n')
        status.update(completed_episodes=len(episodes), elapsed_seconds=time.monotonic()-start,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        (output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
