"""Qualified adapter for published IMPRL SARSOP vectors; no policy training."""
import argparse
import ast
import importlib.util
import itertools
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Reference:
    def __init__(self, env, folder):
        self.env = env
        self.states = np.array([s[::-1] for s in itertools.product(range(3), repeat=env.n_components)])
        self.actions = self.states.copy()
        self.discount = env.discount_factor
        policy = ET.parse(next(Path(folder).rglob('policy.policy'))).getroot().find('AlphaVector')
        self.alpha = np.array([np.fromstring(v.text, sep=' ') for v in policy.findall('Vector')])
        self.alpha_actions = np.array([int(v.attrib['action']) for v in policy.findall('Vector')])
        assert self.alpha.shape == (int(policy.attrib['numVectors']), len(self.states))
        raw = next(Path(folder).rglob('pomdp.pomdp')).read_text()
        # Author XML has a malformed xmlns declaration; replace root attributes
        # only. Numerical entries are preserved exactly and checked below.
        xml = ET.fromstring(re.sub(r'<pomdpx\s[^>]*>', '<pomdpx>', raw, count=1))
        n = len(self.states)
        self.T = np.zeros((n, n, n))
        self.O = np.zeros_like(self.T)
        self.R = np.zeros((n, n))
        self.b0 = np.zeros(n)
        for name, matrix, value_name in [
            ('StateTransitionFunction/CondProb', self.T, 'ProbTable'),
            ('ObsFunction/CondProb', self.O, 'ProbTable'),
            ('RewardFunction/Func', self.R, 'ValueTable'),
            ('InitialStateBelief/CondProb', self.b0, 'ProbTable'),
        ]:
            for entry in xml.findall(name + '/Parameter/Entry'):
                indices = tuple(int(x[1:]) for x in entry.findtext('Instance').split())
                matrix[indices] = float(entry.findtext(value_name))
        assert float(xml.findtext('Discount')) == self.discount
        self.unique_obs = [np.unique(o, axis=1, return_counts=True) for o in self.O]
        self.cache = {}

    def joint(self, belief):
        return np.prod(np.asarray(belief)[np.arange(self.env.n_components), self.states], axis=1)

    def q(self, belief):
        b = self.joint(belief)
        q = self.R @ b
        for a in range(len(self.actions)):
            obs, multiplicity = self.unique_obs[a]
            unnormalized = obs * (b @ self.T[a])[:, None]
            q[a] += self.discount * (np.max(self.alpha @ unnormalized, axis=0) @ multiplicity)
        return q

    def act(self, belief):
        key = np.asarray(belief).tobytes()
        if key not in self.cache:
            self.cache[key] = self.actions[int(np.argmax(self.q(belief)))].copy()
        return self.cache[key]

    def qualify(self, official_source):
        env = self.env
        T, O, R = np.zeros_like(self.T), np.zeros_like(self.O), np.zeros_like(self.R)
        for a, action in enumerate(self.actions):
            for s, state in enumerate(self.states):
                T[a, s] = np.prod(env.transition_model[np.arange(env.n_components), action, state, self.states], axis=1)
                O[a, s] = np.prod(env.observation_model[np.arange(env.n_components), action, state, self.states], axis=1)
                delta = np.eye(3)[state]
                R[a, s] = env.get_reward(state, delta, action, delta)[0]
        errors = dict(transition=float(np.max(abs(T-self.T))), observation=float(np.max(abs(O-self.O))),
                      reward=float(np.max(abs(R-self.R))), initial_belief=float(np.max(abs(self.joint(env.belief)-self.b0))))
        assert max(errors.values()) < 1e-10, errors
        assert np.allclose(self.T.sum(2), 1) and np.allclose(self.O.sum(2), 1)
        tree = ast.parse(Path(official_source).read_text())
        klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'SARSOPAgent')
        namespace = {'np': np, 'itertools': itertools}
        exec(compile(ast.Module(body=[klass], type_ignores=[]), str(official_source), 'exec'), namespace)
        original = object.__new__(namespace['SARSOPAgent'])
        original.env = env
        original.num_components = env.n_components
        original.num_damage_states = 3
        original.num_joint_actions = len(self.actions)
        original.joint_state_space_array = self.states
        original.alpha_vectors = self.alpha
        original.reward_model = self.R
        original.transition_model = self.T
        original.observation_model = self.O
        original.discount = self.discount
        rng = np.random.default_rng(8801)
        error = 0.0
        for _ in range(16):
            belief = rng.dirichlet(np.ones(3), env.n_components)
            b = original.get_joint_belief(belief)
            assert np.allclose(b, self.joint(belief), atol=1e-12)
            expected = np.array([original._estimate_action_return(b, a) for a in range(len(self.actions))])
            error = max(error, float(np.max(abs(expected-self.q(belief)))))
        assert error < 1e-9, error
        return dict(status='qualified', model_errors=errors, official_Q_max_error=error,
                    random_belief_checks=16, alpha_vectors=len(self.alpha),
                    scope='Published approximate alpha vectors plus official one-step lookahead; not exact optimal action labels',
                    XML_parser_adaptation='Malformed root attributes removed only; all model numbers verified against original core')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    core_path = args.run/'sources/imprl/imprl/envs/structural_envs/k_out_of_n_infinite.py'
    core = load_module('original_imprl', core_path)
    result = {}
    for setting in ['n2_k1_nomob', 'n2_k2_nomob', 'n3_k1_nomob']:
        cfg = json.loads((args.run/'tasks'/f'{setting}_r1/inputs/config.json').read_text())
        env = core.KOutOfN(cfg, percept_type='belief', time_limit=20, reward_shaping=True,
                          is_failure_self_announcing=False)
        ref = Reference(env, args.run/'sources/reference_members'/setting)
        result[setting] = ref.qualify(args.run/'sources/imprl/imprl/agents/SARSOP.py')
    (args.run/'reference_model_qualification88.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
