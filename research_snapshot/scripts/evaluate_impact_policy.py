"""Independent original IMP-act rollouts, with policy and latent truth isolated."""
import argparse
from datetime import datetime, timezone
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from evaluate_imprl_policy import IsolatedPolicy


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--seed-start', type=int, default=41000)
    parser.add_argument('--label')
    parser.add_argument('--deps', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    source = run/'tasks'/args.job
    label = args.label or args.job
    output = run/'evaluation'/label
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = dict(name=label, job=args.job, setting='IMP-act_ToyExample-v2',
                  family='IMP-act', planned_episodes=args.episodes, steps=50,
                  seed_start=args.seed_start, status='running',
                  started_utc=datetime.now(timezone.utc).isoformat())
    (output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    sys.path.insert(0, str(args.deps))
    spec = importlib.util.spec_from_file_location('original_impact_adapter', source/'environment_api.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    config = api.load_config()
    assert config['horizon'] == 50
    rows = []
    policy = None
    try:
        policy = IsolatedPolicy(source, run/'evaluation_workspaces'/label, output,
                                worker=Path(__file__).with_name('impact_policy_worker.py'),
                                runtime_mounts=[(args.deps, '/opt/phm_python/extra')])
        with gzip.open(output/'trajectories.jsonl.gz', 'wt') as traces:
            for seed in range(args.seed_start, args.seed_start+args.episodes):
                env = api.make_environment(seed)
                observation = plain(api.legal_observation(env.reset()))
                policy.reset()
                cost = 0.
                requested = np.zeros(5, dtype=int)
                applied = np.zeros(5, dtype=int)
                components = {k: 0. for k in ['maintenance_reward', 'travel_time_reward', 'terminal_reward']}
                for t in range(50):
                    # No environment seed, hidden state, simulator info or score
                    # is sent to the evaluated policy process.
                    action = np.asarray(policy.request({'op': 'act', 'observation': observation})['action'])
                    assert action.shape == (12, 1) and np.issubdtype(action.dtype, np.integer), 'Illegal action shape/type'
                    assert np.all((action >= 0) & (action <= 4)), 'Illegal action'
                    requested += np.bincount(action.ravel(), minlength=5)
                    following, reward, done, info = env.step(action.astype(int).tolist())
                    following = plain(api.legal_observation(following))
                    cost -= float(reward)
                    applied += np.bincount(np.asarray(info['applied_actions']).ravel(), minlength=5)
                    for k in components:
                        components[k] += float(info['reward_elements'][k])
                    traces.write(json.dumps(plain(dict(seed=seed, t=t, observation=observation,
                        requested_action=action, applied_action=info['applied_actions'], reward=float(reward),
                        next_observation=following, evaluator_only_info=info)), allow_nan=False)+'\n')
                    observation = following
                    assert done == (t == 49), 'Unexpected original environment termination'
                assert abs(cost+sum(components.values())) <= max(1e-6, abs(cost)*1e-12)
                rows.append(dict(seed=seed, cost=cost, requested_actions=requested.tolist(),
                    applied_actions=applied.tolist(), reward_elements=components,
                    inspections=int(applied[1]), replacements=int(applied[4])))
                if len(rows) % 64 == 0:
                    (output/'episodes.json').write_text(json.dumps(rows)+'\n')
                    print(json.dumps(dict(name=label, completed=len(rows))), flush=True)
        status.update(status='finished', mean_cost=float(np.mean([r['cost'] for r in rows])),
                      mean_inspections=float(np.mean([r['inspections'] for r in rows])),
                      mean_replacements=float(np.mean([r['replacements'] for r in rows])))
    except BaseException as exc:
        status.update(status='evaluation_error', error_type=type(exc).__name__, error=str(exc))
    finally:
        if policy is not None:
            policy.close()
        (output/'episodes.json').write_text(json.dumps(rows)+'\n')
        status.update(completed_episodes=len(rows), elapsed_seconds=time.monotonic()-start,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        (output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
