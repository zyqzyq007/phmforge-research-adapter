"""Complete original IMP-MARL episodes with submitted code isolated from RNG."""
import argparse
from datetime import datetime, timezone
import gzip
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from evaluate_imprl_policy import IsolatedPolicy


def legal_observation(obs, t, info=None, previous_reward=None):
    return dict(observation={k: np.asarray(v).tolist() for k, v in obs.items()},
                time_step=int(t),
                inspection=None if info is None else np.asarray(info['inspection']).tolist(),
                previous_reward=previous_reward)


def validate_action(action, agents):
    if not isinstance(action, dict) or set(action) != set(agents):
        raise ValueError('Expected exactly one action per original agent ID')
    if any(type(action[k]) is not int or action[k] not in (0, 1, 2) for k in agents):
        raise ValueError('Original actions must be integers 0,1,2')
    return action


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=Path, required=True)
    p.add_argument('--policy', choices=['submitted', 'reference', 'do_nothing'], required=True)
    p.add_argument('--reference-parameters', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    config = json.loads((args.workspace/'inputs/config.json').read_text())
    seeds = json.loads(args.seeds.read_text())
    sys.path.insert(0, str(args.source.resolve()))
    Struct = importlib.import_module('imp_marl.environments.struct_env').Struct
    status = dict(policy=args.policy, config=config, planned_episodes=len(seeds),
                  status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  interpretation='original simulator/reward/horizon; centralized observation-union coding adaptation',
                  inference_limits='existing isolated runner:300s initial,20s/request,900s/all episodes')
    (args.output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    policy = None
    rows = []
    try:
        if args.policy == 'submitted':
            work = args.output/'isolated_workspace'
            policy = IsolatedPolicy(args.workspace, work, args.output,
                worker=Path(__file__).with_name('impmarl_policy_worker.py'),
                workspace_alias='/app')
        params = json.loads(args.reference_parameters.read_text()) if args.reference_parameters else None
        if args.policy == 'reference' and params is None:
            raise ValueError('Reference must use recorded public author parameters')
        with gzip.open(args.output/'trajectories.jsonl.gz', 'wt') as trace:
            for seed in seeds:
                np.random.seed(seed)
                env = Struct(config)
                obs = env.reset()
                if policy is not None:
                    policy.reset()
                info = None
                previous_reward = None
                cost, inspections, repairs = 0.0, 0, 0
                for t in range(env.ep_length):
                    visible = legal_observation(obs, t, info, previous_reward)
                    if args.policy == 'submitted':
                        action = policy.request({'op': 'act', 'observation': visible})['action']
                    else:
                        action = {agent: 0 for agent in env.agent_list}
                        if args.policy == 'reference':
                            if t > 0 and t % params['insp_interv'] == 0:
                                pf = np.asarray([obs[a][-2] for a in env.agent_list])
                                for i in (-pf).argsort()[:params['insp_comp']]:
                                    action[env.agent_list[int(i)]] = 1
                            if info is not None:
                                for i, inspection in enumerate(info['inspection']):
                                    if inspection == 1:
                                        action[env.agent_list[i]] = 2
                    action = validate_action(action, env.agent_list)
                    before = visible
                    obs, rewards, done, info = env.step(action)
                    previous_reward = float(rewards['agent_0'])
                    if any(v != previous_reward for v in rewards.values()):
                        raise AssertionError('Unexpected unshared reward')
                    # Original reward already contains discount**time_step.
                    cost -= previous_reward
                    inspections += sum(a == 1 for a in action.values())
                    repairs += sum(a == 2 for a in action.values())
                    trace.write(json.dumps(dict(seed=seed, t=t, visible=before, action=action,
                        reward=previous_reward, next_visible=legal_observation(obs,t+1,info,previous_reward)),
                        allow_nan=False)+'\n')
                    if done != (t == env.ep_length-1):
                        raise AssertionError('Unexpected original horizon')
                rows.append(dict(seed=seed, cost=cost, inspections=inspections, repairs=repairs))
                if len(rows) % 32 == 0:
                    (args.output/'episodes.json').write_text(json.dumps(rows)+'\n')
                    print(json.dumps(dict(completed=len(rows), output=str(args.output))), flush=True)
        status.update(status='finished', mean_cost=float(np.mean([r['cost'] for r in rows])),
                      mean_inspections=float(np.mean([r['inspections'] for r in rows])),
                      mean_repairs=float(np.mean([r['repairs'] for r in rows])))
    except BaseException as error:
        status.update(status='evaluation_error', error_type=type(error).__name__, error=str(error))
    finally:
        if policy is not None:
            policy.close()
        (args.output/'episodes.json').write_text(json.dumps(rows)+'\n')
        status.update(completed_episodes=len(rows), elapsed_seconds=time.monotonic()-start,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        (args.output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
