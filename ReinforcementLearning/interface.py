
"""Synthetic reinforcement-learning environment using the shared contract."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .telemetry_generator import generate_observation
from ..feature_contract import encode_features


ACTION_NAMES = [
    "notify_operator",
    "collect_diagnostics",
    "reduce_noncritical_workload",
    "review_suspicious_activity",
]


class DecisionOptimizationEnv(gym.Env):
    """Simulation-only environment with ten-element telemetry observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon: int = 80,
        missing_probability: float = 0.08,
        collection_failure_probability: float = 0.015,
    ):
        super().__init__()

        if horizon < 1:
            raise ValueError("horizon must be positive")

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        self.horizon = int(horizon)
        self.missing_probability = missing_probability
        self.collection_failure_probability = (
            collection_failure_probability
        )

        self.steps = 0
        self.rng = np.random.default_rng()
        self.last_observation = None

    def _new_observation(self):
        return generate_observation(
            self.rng,
            missing_probability=self.missing_probability,
            collection_failure_probability=(
                self.collection_failure_probability
            ),
        )

    def _apply_simulated_action(self, metrics, action):
        """Apply bounded effects to synthetic measurements only."""

        adjusted = dict(metrics)

        if action == 2:
            # Simulated reduction of workload.
            for name, reduction in (
                ("cpu_percent", 10.0),
                ("memory_percent", 4.0),
                ("load_1m", 0.7),
            ):
                if adjusted.get(name) is not None:
                    adjusted[name] = max(
                        0.0,
                        float(adjusted[name]) - reduction,
                    )

        elif action == 3:
            # Simulated reduction in connection volume.
            value = adjusted.get("network_connections")
            if value is not None:
                adjusted["network_connections"] = max(
                    0,
                    int(value * 0.85),
                )

        # Diagnostics and notification do not change host measurements.
        return adjusted

    @staticmethod
    def _pressure(encoded):
        """Calculate pressure using available measurements only."""

        values = encoded.vector[:5]
        masks = encoded.vector[5:]

        weights = np.asarray(
            [0.28, 0.23, 0.14, 0.20, 0.15],
            dtype=np.float32,
        )

        effective_weights = weights * masks
        denominator = float(effective_weights.sum())

        if denominator <= 0.0:
            # No measurements are available. Do not interpret missing
            # data as healthy or as measured zero.
            return None

        return float(
            np.dot(values, effective_weights) / denominator
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.last_observation = self._new_observation()

        vector = self.last_observation["encoded"].vector.copy()

        return vector, {
            "collection_status": (
                self.last_observation["collection_status"]
            ),
            "contract_version": (
                self.last_observation["encoded"].contract_version
            ),
        }

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action index: {action}")

        previous = self.last_observation
        base = self._new_observation()

        adjusted_metrics = self._apply_simulated_action(
            base["metrics"],
            int(action),
        )

        encoded = encode_features(adjusted_metrics)

        # Reward only considers measurements that are actually available.
        pressure = self._pressure(encoded)

        intervention_costs = {
            0: 0.0,
            1: 0.025,
            2: 0.045,
            3: 0.035,
        }

        cost = intervention_costs[int(action)]

        if pressure is None:
            # Explicit uncertainty penalty: no telemetry is not a healthy
            # system state and should not produce an artificially high reward.
            reward = -0.25 - cost
        else:
            reward = 1.0 - pressure - cost

        self.steps += 1
        terminated = self.steps >= self.horizon
        truncated = False

        self.last_observation = {
            **base,
            "metrics": adjusted_metrics,
            "encoded": encoded,
        }

        info = {
            "pressure": pressure,
            "intervention_cost": cost,
            "action_name": ACTION_NAMES[int(action)],
            "collection_status": base["collection_status"],
            "feature_contract_version": encoded.contract_version,
            "available_feature_count": int(
                encoded.vector[5:].sum()
            ),
        }

        return (
            encoded.vector.copy(),
            float(reward),
            terminated,
            truncated,
            info,
        )
