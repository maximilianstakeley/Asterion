"""RL state encoding with explicit measurement-availability masks.

Observation layout (10 finite float32 values):
  [cpu, memory, disk, connections, load, cpu_available, memory_available,
   disk_available, connections_available, load_available]

Missing value slots use a neutral tensor placeholder and MUST be interpreted
alongside their zero availability mask. This is a new model contract: retrain
PPO policies; old five-input models are incompatible.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

FEATURES = (
    ("cpu_percent", 100.0),
    ("memory_percent", 100.0),
    ("disk_percent", 100.0),
    ("network_connections", 500.0),
    ("load_1m", 8.0),
)


def _finite_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        return None
    if key.endswith("_percent") and value > 100:
        return None
    if key == "network_connections" and not value.is_integer():
        return None
    return value


def state_from_metrics(metrics: dict[str, Any]) -> np.ndarray:
    """Encode values and explicit masks without inferring missing load."""
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary")

    values: list[float] = []
    masks: list[float] = []

    for key, scale in FEATURES:
        value = _finite_metric(metrics, key)
        available = value is not None
        masks.append(1.0 if available else 0.0)
        if not available:
            # Tensor placeholder only; mask=0 means this is not a measurement.
            values.append(0.0)
        elif key == "network_connections":
            values.append(min(1.0, value / scale))
        else:
            values.append(min(1.0, value / scale))

    state = np.asarray(values + masks, dtype=np.float32)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError("RL state must be a finite 10-element vector")
    return state
