"""Task-visible adapter for the established SARSOP solver and original model.

No reference policy, evaluation answer, future state or evaluation seed is used.
"""
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parent


def solve_policy(seconds=180, output=".runtime/planned.policy"):
    """Run the provided CPU planner; time is a numerical budget, not asset life."""
    destination = ROOT / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / "inputs/pomdpsol"), "--timeout", str(seconds),
               "--output", str(destination), str(ROOT / "inputs/model.pomdpx")]
    with destination.with_suffix(".solver.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=float(seconds) + 120)
    return destination


class SARSOPPolicy:
    """Stationary joint belief policy, with optional published one-step lookahead.

    Policy values are rewards (higher is better). The original reward is negative
    maintenance/failure cost. Observation branches, shared mobilisation cost and
    k-out-of-n system reliability are taken from the unchanged environment.
    """
    def __init__(self, config, policy_file, lookahead=True):
        spec = importlib.util.spec_from_file_location(
            "original_maintenance_model", ROOT / "inputs/environment.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.env = module.KOutOfN(
            config, percept_type="belief", time_limit=80, reward_shaping=True,
            is_failure_self_announcing=False)
        n = self.env.n_components
        self.states = np.array([s[::-1] for s in itertools.product(range(3), repeat=n)])
        self.actions = self.states.copy()
        vectors = ET.parse(ROOT / policy_file).getroot().find("AlphaVector")
        self.alpha = np.array([np.fromstring(v.text, sep=" ")
                               for v in vectors.findall("Vector")])
        self.alpha_actions = np.array([int(v.attrib["action"])
                                       for v in vectors.findall("Vector")])
        assert self.alpha.shape[1] == len(self.states)
        self.lookahead = lookahead
        size = len(self.states)
        self.T = np.empty((size, size, size))
        self.O = np.empty_like(self.T)
        self.R = np.empty((size, size))
        for a, action in enumerate(self.actions):
            for s, state in enumerate(self.states):
                self.T[a, s] = np.prod(self.env.transition_model[
                    np.arange(n), action, state, self.states], axis=1)
                self.O[a, s] = np.prod(self.env.observation_model[
                    np.arange(n), action, state, self.states], axis=1)
                delta = np.eye(3)[state]
                self.R[a, s] = self.env.get_reward(state, delta, action, delta)[0]
        self.unique_obs = [np.unique(o, axis=1, return_counts=True) for o in self.O]
        self.cache = {}

    def q_values(self, belief):
        b = np.prod(np.asarray(belief)[np.arange(self.env.n_components), self.states], axis=1)
        q = self.R @ b
        for a in range(len(self.actions)):
            obs, counts = self.unique_obs[a]
            unnormalized = obs * (b @ self.T[a])[:, None]
            q[a] += self.env.discount_factor * (
                np.max(self.alpha @ unnormalized, axis=0) @ counts)
        return q

    def act(self, belief):
        key = np.asarray(belief).tobytes()
        if key not in self.cache:
            if self.lookahead:
                a = int(np.argmax(self.q_values(belief)))
            else:
                b = np.prod(np.asarray(belief)[np.arange(self.env.n_components), self.states], axis=1)
                a = self.alpha_actions[int(np.argmax(self.alpha @ b))]
            self.cache[key] = self.actions[a].copy()
        return self.cache[key].copy()

    def reset(self):
        pass
