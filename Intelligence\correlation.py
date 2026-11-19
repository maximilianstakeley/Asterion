
from __future__ import annotations

import gymnasium as gym
import numpy as np

from gymnasium import spaces


class AsterionOptimizationEnv(gym.Env):

    metadata = {"render_modes": []}

    ACTIONS = {
        0: "observe",
        1: "collect_diagnostics",
        2: "notify_operator",
        3: "reduce_noncritical_workload",
    }

    ACTION_COSTS = {
        0: 0.0,
        1: 0.02,
        2: 0.03,
        3: 0.10,
    }

    ACTION_RISKS = {
        0: 0.0,
        1: 0.01,
        2: 0.01,
        3: 0.15,
    }

    def __init__(self, horizon=300, seed=None):
        super().__init__()

        self.horizon = horizon

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        self.rng = np.random.default_rng(seed)

        self.reset(seed=seed)

    def _observation(self):
        values = np.clip(
            self.state,
            0.0,
            1.0
        )

        return np.concatenate([
            values,
            self.available.astype(np.float32)
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = self.rng.uniform(
            0.15, 0.45, size=5
        ).astype(np.float32)

        self.available = np.ones(
            5, dtype=np.float32
        )

        self.steps = 0
        self.previous_anomaly = 0.0

        self.workload = self.rng.uniform(
            0.1, 0.5
        )

        self.fault = self.rng.choice(
            [0.0, 0.0, 0.0, 0.6, 0.9]
        )

        return self._observation(), {
            "simulated": True
        }

    def _simulate_workload(self):
        self.workload += self.rng.normal(
            0.0, 0.04
        )

        self.workload = float(
            np.clip(self.workload, 0.0, 1.0)
        )

        # Simulated system response.
        self.state[0] += (
            0.20 * self.workload
            + 0.10 * self.fault
        )

        self.state[1] += (
            0.12 * self.workload
            + 0.05 * self.fault
        )

        self.state[2] += (
            0.002 * self.workload
        )

        self.state[3] += (
            0.10 * self.workload
        )

        self.state[4] += (
            0.08 * self.fault
        )

        self.state += self.rng.normal(
            0.0, 0.015, size=5
        )

        self.state = np.clip(
            self.state, 0.0, 1.0
        )

        # Random sensor dropout.
        self.available = (
            self.rng.random(5) > 0.05
        ).astype(np.float32)

    def _anomaly_score(self):
        measured = self.state[
            self.available > 0
        ]

        if len(measured) == 0:
            return 0.5

        return float(
            np.mean(
                np.maximum(
                    measured - 0.65,
                    0.0
                )
            )
        )

    def step(self, action):
        action = int(action)

        if action not in self.ACTIONS:
            raise ValueError("Invalid action")

        self.steps += 1

        previous_score = self._anomaly_score()

        self._simulate_workload()

        # Intervention effects are deliberately simulated.
        if action == 1:
            # Diagnostics improve information quality.
            self.available = np.ones(
                5, dtype=np.float32
            )

        elif action == 2:
            # Notification has no direct system-state effect.
            pass

        elif action == 3:
            # Simulated reduction in noncritical workload.
            self.workload *= 0.65
            self.state[0] *= 0.90
            self.state[1] *= 0.95
            self.state[3] *= 0.90

        current_score = self._anomaly_score()

        benefit = previous_score - current_score

        cost = self.ACTION_COSTS[action]
        risk = self.ACTION_RISKS[action]

        unnecessary_intervention = (
            action != 0 and previous_score < 0.03
        )

        reward = (
            5.0 * benefit
            - cost
            - risk
            - (0.10 if unnecessary_intervention else 0.0)
        )

        # Penalize operating in a persistently unhealthy state.
        reward -= current_score * 0.5

        terminated = self.steps >= self.horizon

        truncated = False

        info = {
            "simulated": True,
            "action": self.ACTIONS[action],
            "anomaly_before": previous_score,
            "anomaly_after": current_score,
            "benefit": benefit,
            "cost": cost,
            "risk": risk,
            "workload": self.workload,
            "fault": self.fault,
        }

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info
        )
