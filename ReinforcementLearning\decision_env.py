"""Deterministic-seed simulation for comparing RL with a rule baseline.

This environment is synthetic and has no connection to host operating systems.
Observation: normalized CPU, memory, disk pressure, suspicious connection pressure,
and workload level. Actions match the shared decision-policy action vocabulary.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_NAMES = [
    "notify_operator",
    "collect_diagnostics",
    "reduce_noncritical_workload",
    "review_suspicious_activity",
]

class DecisionOptimizationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, horizon: int = 80):
        super().__init__()
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.horizon = int(horizon)
        self.state = np.zeros(5, dtype=np.float32)
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.state = np.array([
            self.np_random.uniform(.35, .98),
            self.np_random.uniform(.25, .95),
            self.np_random.uniform(.15, .95),
            self.np_random.uniform(.05, .95),
            self.np_random.uniform(.25, .95),
        ], dtype=np.float32)
        return self._observation(), {}

    def _observation(self):
        """Return simulated state plus five known-available flags."""
        availability = np.ones(5, dtype=np.float32)

        return np.concatenate(
            (self.state.copy(), availability)
        ).astype(np.float32)

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action index: {action}")
        cpu, memory, disk, suspicious, workload = self.state.astype(float).tolist()
        cost = 0.0
        # Actions have small bounded effects in simulation only.
        if action == 1:  # collect diagnostics: cost, no direct remediation
            cost = .025
        elif action == 2:  # reduce workload, only if not already low
            workload = max(.10, workload - .12)
            cpu = max(.05, cpu - .10)
            memory = max(.05, memory - .035)
            cost = .045
        elif action == 3:  # review suspicious activity; reduces simulated uncertainty
            suspicious = max(.0, suspicious - .10)
            cost = .035
        # Simulated environment drifts; no real metrics are read or written.
        noise = self.np_random.normal(0.0, .018, size=4)
        cpu = float(np.clip(cpu + noise[0] + .012 * workload, 0, 1))
        memory = float(np.clip(memory + noise[1] + .006 * workload, 0, 1))
        disk = float(np.clip(disk + noise[2] * .35, 0, 1))
        suspicious = float(np.clip(suspicious + noise[3], 0, 1))
        self.state = np.array([cpu, memory, disk, suspicious, workload], dtype=np.float32)
        # Reward rewards reduced pressure while charging for unnecessary interventions.
        pressure = .28 * cpu + .23 * memory + .14 * disk + .20 * suspicious + .15 * workload
        reward = float(1.0 - pressure - cost)
        self.steps += 1
        terminated = self.steps >= self.horizon
        info = {"pressure": pressure, "intervention_cost": cost, "action_name": ACTION_NAMES[action]}
        return self._observation(), reward, terminated, False, info


