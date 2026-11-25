
"""Synthetic telemetry generator for Asterion model training.

All generated samples pass through the production feature contract.
No host telemetry is collected and no system actions are executed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..feature_contract import (
    FEATURE_NAMES,
    UNITS,
    encode_features,
)


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def generate_observation(
    rng: np.random.Generator,
    *,
    missing_probability: float = 0.08,
    collection_failure_probability: float = 0.015,
) -> dict[str, Any]:
    """Generate one correlated synthetic host observation."""

    if not 0.0 <= missing_probability <= 1.0:
        raise ValueError("missing_probability must be in [0, 1]")

    if not 0.0 <= collection_failure_probability <= 1.0:
        raise ValueError(
            "collection_failure_probability must be in [0, 1]"
        )

    # Shared latent factors create realistic correlation.
    workload = rng.beta(2.2, 2.0)
    storage_pressure = rng.beta(2.0, 3.0)
    network_pressure = rng.beta(1.8, 3.0)

    # CPU, memory and system load share workload pressure.
    cpu = _clip(
        8.0 + 82.0 * workload + rng.normal(0.0, 7.0),
        0.0,
        100.0,
    )

    memory = _clip(
        15.0 + 70.0 * workload + rng.normal(0.0, 8.0),
        0.0,
        100.0,
    )

    # Disk changes more slowly and is only weakly correlated with workload.
    disk = _clip(
        12.0 + 78.0 * storage_pressure
        + 8.0 * workload
        + rng.normal(0.0, 3.0),
        0.0,
        100.0,
    )

    # Connection counts are non-negative integers and respond to
    # network pressure, with some influence from host workload.
    connections = int(
        max(
            0,
            round(
                8.0
                + 700.0 * network_pressure
                + 90.0 * workload
                + rng.normal(0.0, 35.0)
            ),
        )
    )

    # System load is correlated with CPU and workload, but is not
    # numerically identical to CPU utilization.
    load_1m = max(
        0.0,
        0.04
        + 7.0 * workload
        + 0.018 * cpu
        + rng.normal(0.0, 0.35),
    )

    generated = {
        "cpu_percent": cpu,
        "memory_percent": memory,
        "disk_percent": disk,
        "network_connections": connections,
        "load_1m": load_1m,
    }

    # A full collection failure makes every measurement unavailable.
    full_failure = (
        rng.random() < collection_failure_probability
    )

    metrics: dict[str, Any] = {}
    metric_status: dict[str, dict[str, Any]] = {}

    for name in FEATURE_NAMES:
        unavailable = full_failure or (
            rng.random() < missing_probability
        )

        if unavailable:
            metrics[name] = None
            metric_status[name] = {
                "status": "unavailable",
                "unit": UNITS[name],
                "error": (
                    "Simulated collection failure"
                    if full_failure
                    else "Simulated metric unavailable"
                ),
            }
        else:
            metrics[name] = generated[name]
            metric_status[name] = {
                "status": "ok",
                "unit": UNITS[name],
                "error": None,
            }

    outcomes = [
        item["status"] for item in metric_status.values()
    ]

    if all(status == "ok" for status in outcomes):
        collection_status = "ok"
    elif all(status == "unavailable" for status in outcomes):
        collection_status = "failed"
    else:
        collection_status = "partial"

    # Encode the raw observation using the production runtime encoder.
    encoded = encode_features(metrics)

    return {
        "metrics": metrics,
        "metric_status": metric_status,
        "collection_status": collection_status,
        "encoded": encoded,
    }


def generate_dataset(
    sample_count: int,
    *,
    seed: int = 42,
    missing_probability: float = 0.08,
    collection_failure_probability: float = 0.015,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return model-ready vectors and their source observations."""

    if isinstance(sample_count, bool) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")

    rng = np.random.default_rng(seed)

    observations = [
        generate_observation(
            rng,
            missing_probability=missing_probability,
            collection_failure_probability=(
                collection_failure_probability
            ),
        )
        for _ in range(sample_count)
    ]

    vectors = np.stack(
        [item["encoded"].vector for item in observations]
    ).astype(np.float32)

    if vectors.shape != (sample_count, 10):
        raise RuntimeError(
            f"Unexpected training dataset shape: {vectors.shape}"
        )

    if not np.isfinite(vectors).all():
        raise RuntimeError("Training dataset contains non-finite values")

    return vectors, observations
