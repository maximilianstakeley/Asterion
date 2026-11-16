"""Normalize metrics while explicitly encoding measurement availability."""
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


def _valid_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)

    if isinstance(value, bool):
        return None

    if not isinstance(value, (int, float)):
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
    """Return 5 normalized values followed by 5 availability flags.

    Layout:
      [cpu, memory, disk, connection_pressure, load,
       cpu_available, memory_available, disk_available,
       connections_available, load_available]

    A zero in a value slot is not evidence of a measured zero unless its
    corresponding availability flag is 1.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary")

    values: list[float] = []
    availability: list[float] = []

    for key, scale in FEATURES:
        value = _valid_metric(metrics, key)
        is_available = value is not None

        availability.append(1.0 if is_available else 0.0)

        if value is None:
            # Finite tensor placeholder; the mask marks it unavailable.
            values.append(0.0)
        else:
            values.append(min(1.0, value / scale))

    state = np.asarray(values + availability, dtype=np.float32)

    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError("Encoded RL state must be a finite 10-element vector")

    return state