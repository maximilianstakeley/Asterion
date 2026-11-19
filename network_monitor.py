
"""Authoritative telemetry-to-model feature contract for Asterion.

Model vector layout:
    [cpu, memory, disk, network_connections, load_1m,
     cpu_available, memory_available, disk_available,
     network_connections_available, load_1m_available]

The first five values are normalized to [0, 1].
The final five values are binary availability indicators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


CONTRACT_VERSION = "asterion_telemetry_features_v1"
FEATURE_DIMENSION = 10
MEASUREMENT_DIMENSION = 5

# The order here is authoritative. Do not independently reorder features
# in the agent, API, training environment, or policy adapter.
FEATURE_NAMES = (
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "network_connections",
    "load_1m",
)

AVAILABILITY_NAMES = tuple(
    f"{name}_available" for name in FEATURE_NAMES
)

# Scales are normalization reference points, not physical maximums.
NORMALIZATION_SCALES = {
    "cpu_percent": 100.0,
    "memory_percent": 100.0,
    "disk_percent": 100.0,
    "network_connections": 500.0,
    "load_1m": 8.0,
}

UNITS = {
    "cpu_percent": "%",
    "memory_percent": "%",
    "disk_percent": "%",
    "network_connections": "connections",
    "load_1m": "load average",
}

# Validation limits are deliberately distinct from normalization scales.
# For example, a connection count above 500 is valid; its normalized
# value is clipped to 1.0 rather than rejected.
VALID_RANGES = {
    "cpu_percent": (0.0, 100.0),
    "memory_percent": (0.0, 100.0),
    "disk_percent": (0.0, 100.0),
    "network_connections": (0.0, 10_000_000.0),
    "load_1m": (0.0, 1_000_000.0),
}


@dataclass(frozen=True)
class EncodedFeatures:
    """Encoded model input plus inspectable measurement metadata."""

    vector: np.ndarray
    values: tuple[float | None, ...]
    availability: tuple[int, ...]
    feature_names: tuple[str, ...]
    availability_names: tuple[str, ...]
    contract_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "feature_names": list(self.feature_names),
            "availability_names": list(self.availability_names),
            "values": list(self.values),
            "availability": list(self.availability),
            "vector": self.vector.tolist(),
        }


def _extract_value(
    metrics: dict[str, Any],
    name: str,
) -> tuple[Any, str | None]:
    """Read raw values or structured telemetry records."""

    record = metrics.get(name)

    if isinstance(record, dict):
        status = record.get("status")
        unit = record.get("unit")

        if status == "unavailable":
            if record.get("value") is not None:
                raise ValueError(
                    f"{name}: unavailable measurement must have value=None"
                )
            return None, unit

        if status != "ok":
            raise ValueError(
                f"{name}: status must be 'ok' or 'unavailable'"
            )

        if unit != UNITS[name]:
            raise ValueError(
                f"{name}: expected unit {UNITS[name]!r}, got {unit!r}"
            )

        return record.get("value"), unit

    # Raw dictionaries use None to represent unavailable measurements.
    return record, None


def _validate_value(name: str, value: Any) -> float | None:
    """Validate a measurement without confusing zero with missing data."""

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric or None")

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not a valid number") from exc

    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")

    lower, upper = VALID_RANGES[name]

    if not lower <= numeric <= upper:
        raise ValueError(
            f"{name} must be in [{lower}, {upper}], got {numeric}"
        )

    if name == "network_connections" and not numeric.is_integer():
        raise ValueError("network_connections must be an integer")

    return numeric


def encode_features(metrics: dict[str, Any]) -> EncodedFeatures:
    """Encode telemetry into the fixed ten-element model vector.

    Missing measurements:
      - normalized value = 0.0
      - availability flag = 0.0

    Measured zero:
      - normalized value = 0.0
      - availability flag = 1.0

    This preserves the distinction between missing and measured zero.
    """

    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary")

    normalized_values: list[float] = []
    actual_values: list[float | None] = []
    availability: list[int] = []

    for name in FEATURE_NAMES:
        raw_value, _ = _extract_value(metrics, name)
        value = _validate_value(name, raw_value)

        is_available = value is not None
        actual_values.append(value)
        availability.append(1 if is_available else 0)

        if not is_available:
            normalized_values.append(0.0)
            continue

        scale = NORMALIZATION_SCALES[name]
        normalized = min(1.0, max(0.0, value / scale))
        normalized_values.append(normalized)

    vector = np.asarray(
        normalized_values + [float(x) for x in availability],
        dtype=np.float32,
    )

    if vector.shape != (FEATURE_DIMENSION,):
        raise RuntimeError(
            f"Feature contract produced shape {vector.shape}; "
            f"expected ({FEATURE_DIMENSION},)"
        )

    if not np.isfinite(vector).all():
        raise RuntimeError("Feature contract produced non-finite values")

    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise RuntimeError("Feature contract produced values outside [0, 1]")

    return EncodedFeatures(
        vector=vector,
        values=tuple(actual_values),
        availability=tuple(availability),
        feature_names=FEATURE_NAMES,
        availability_names=AVAILABILITY_NAMES,
        contract_version=CONTRACT_VERSION,
    )


def encode_feature_vector(metrics: dict[str, Any]) -> np.ndarray:
    """Convenience API for policy and model consumers."""

    return encode_features(metrics).vector


def feature_contract_metadata() -> dict[str, Any]:
    """Return serializable metadata for model artifacts and audit records."""

    return {
        "contract_version": CONTRACT_VERSION,
        "feature_dimension": FEATURE_DIMENSION,
        "measurement_dimension": MEASUREMENT_DIMENSION,
        "feature_names": list(FEATURE_NAMES),
        "availability_names": list(AVAILABILITY_NAMES),
        "units": dict(UNITS),
        "normalization_scales": dict(NORMALIZATION_SCALES),
        "valid_ranges": {
            key: list(value)
            for key, value in VALID_RANGES.items()
        },
        "missing_value": 0.0,
        "availability_encoding": {
            "available": 1.0,
            "unavailable": 0.0,
        },
    }
