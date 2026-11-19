
"""Compatibility wrapper for Asterion's authoritative feature contract.

The authoritative feature encoder remains strict. This RL boundary converts
invalid individual telemetry measurements into unavailable measurements
before encoding the policy state.

This preserves the distinction between:
    - measured zero: normalized value 0, availability 1
    - unavailable/invalid: normalized value 0, availability 0
"""

from __future__ import annotations

from typing import Any

from ..feature_contract import (
    FEATURE_NAMES,
    AVAILABILITY_NAMES,
    FEATURE_DIMENSION,
    encode_features,
    encode_feature_vector,
)


def state_from_metrics(metrics: dict[str, Any]):
    """Return the ten-element RL state vector.

    Invalid individual measurements are treated as unavailable rather than
    allowing one bad measurement to invalidate the entire policy state.

    Structural errors, such as a non-dictionary input, remain errors.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary")

    sanitized_metrics = dict(metrics)

    for name in FEATURE_NAMES:
        try:
            # Validate this feature independently while leaving all other
            # features unavailable for this validation pass.
            encode_features({name: sanitized_metrics.get(name)})

        except ValueError:
            # Invalid measurement: preserve missingness rather than
            # substituting a measured zero.
            sanitized_metrics[name] = None

    return encode_feature_vector(sanitized_metrics)


__all__ = [
    "FEATURE_NAMES",
    "AVAILABILITY_NAMES",
    "FEATURE_DIMENSION",
    "encode_features",
    "encode_feature_vector",
    "state_from_metrics",
]
