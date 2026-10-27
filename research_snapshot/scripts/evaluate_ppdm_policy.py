"""Replay the author's PPDMEnv; submitted code receives only obs and mask.

This is a coding-policy adaptation of an RL environment, not a native LLM bench.
The original source, reward, action mask and terminal rules are not modified.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from evaluate_imprl_policy import IsolatedPolicy


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def legal_action(raw, mask, components):
    action = np.asarray(raw)
    if (action.shape != (components,) or action.dtype.kind not in 'iu'
            or np.any(action < 0) or np.any(action > 2)):
        raise ValueError('Policy returned an illegal action shape/type/value')
    if not mask[np.arange(components), action].all():
        raise ValueError('Policy selected a masked action')
    return action.astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--policy', choices=['submitted', 'greedy', 'hold'], required=True)
    parser.add_argument('--job')
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--seed-start', type=int, required=True)
    args = parser.parse_args()
    args.run = args.run.resolve()
    output = args.run/'evaluation'/args.label
    output.mkdir(parents=True, exist_ok=False)
    runtime = output/'runtime'
    runtime.mkdir()
    for key in ['TMPDIR', 'TMP', 'TEMP', 'MPLCONFIGDIR']:
        os.environ[key] = str(runtime)
    os.chdir(runtime)
    lane = args.run/'qualification/ppdm101'
    sys.path.insert(0, str(lane/'controls'))
    sys.path.insert(0, str(lane/'source/PPDM_framework-master'))
    from compat_imports import install
    install()
    from ppdm_env import PPDMEnv
    from hyperparameters import h

    started = time.monotonic()
    status = dict(label=args.label, policy=args.policy, planned_episodes=args.episodes,
                  seed_start=args.seed_start, status='running',
                  started_utc=datetime.now(timezone.utc).isoformat())
    status_path = output/'status.json'
    status_path.write_text(json.dumps(status, indent=2)+'\n')
    policy = None
    episodes = []
    try:
        if args.policy == 'submitted':
            policy = IsolatedPolicy(args.run/'tasks'/args.job,
                args.run/'evaluation_workspaces'/args.label, output,
                worker=Path(__file__).with_name('impmarl_policy_worker.py'),
                workspace_alias='/app')
        with (output/'environment_stdout.log').open('w') as env_log, gzip.open(
                output/'trajectories.jsonl.gz', 'wt') as trace:
            for seed in range(args.seed_start, args.seed_start+args.episodes):
                record = dict(seed=seed, status='running', steps=0, return_sum=0.)
                try:
                    np.random.seed(seed)
                    random.seed(seed)
                    with contextlib.redirect_stdout(env_log):
                        env = PPDMEnv(train=True, components=h.components)
                    obs = env.state.copy()
                    if policy is not None:
                        policy.reset()
                    for step in range(h.horizon+1):
                        mask = np.asarray(env.action_masks(), dtype=bool).reshape(h.components, 3)
                        observation = dict(obs=obs.tolist(), mask=mask.tolist())
                        if policy is not None:
                            raw = policy.request(dict(op='act', observation=observation))['action']
                        elif args.policy == 'greedy':
                            raw = [1 if row[1] else 2 if row[2] else 0 for row in mask]
                        else:
                            raw = [0]*h.components
                        action = legal_action(raw, mask, h.components)
                        before = dict(mean_rul=env.mean_rul_norm.copy(), sigma=env.sigma.copy(),
                                      time=env.time, slots=env.next_slots.copy(), capacity=env.capacity.copy())
                        # Only the legal observation above is sent through the isolated pipe.
                        obs, reward, done, info = env.step(action)
                        if not np.isfinite(obs).all() or not np.isfinite(reward):
                            raise ArithmeticError('Non-finite original environment output')
                        record['steps'] += 1
                        record['return_sum'] += float(reward)
                        trace.write(json.dumps(plain(dict(seed=seed, step=step,
                            observation=observation, action=action, hidden_before=before,
                            next_obs=obs, reward=reward, done=done, info=info,
                            reason=env.reason_done, reward_types=env.reward_types.copy())), allow_nan=False)+'\n')
                        if done:
                            record.update(status='terminal', horizon_reached='horizon end' in env.reason_done,
                                terminal_reason=env.reason_done, final_time=float(env.time),
                                reward_types=plain(env.reward_types.copy()))
                            break
                    else:
                        raise RuntimeError('Original episode exceeded its frozen full-horizon bound')
                except Exception as exc:
                    record.update(status='evaluation_error', error_type=type(exc).__name__, error=str(exc))
                episodes.append(record)
                (output/'episodes.json').write_text(json.dumps(episodes, indent=2)+'\n')
        terminal = [e for e in episodes if e['status'] == 'terminal']
        status.update(status='finished', terminal_episodes=len(terminal),
            error_episodes=len(episodes)-len(terminal),
            horizon_reached=sum(e.get('horizon_reached', False) for e in episodes),
            mean_original_return_terminal_only=float(np.mean([e['return_sum'] for e in terminal])) if terminal else None)
    except Exception as exc:
        status.update(status='evaluation_error', error_type=type(exc).__name__, error=str(exc))
    finally:
        if policy is not None:
            policy.close()
        status.update(completed_episodes=len(episodes), elapsed_seconds=time.monotonic()-started,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        status_path.write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
