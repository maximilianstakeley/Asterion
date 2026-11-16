
"""Compatibility wrapper for Asterion's authoritative feature contract."""

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
    """Return the authoritative ten-element telemetry state vector."""

    return encode_feature_vector(metrics)


__all__ = [
    "FEATURE_NAMES",
    "AVAILABILITY_NAMES",
    "FEATURE_DIMENSION",
    "encode_features",
    "encode_feature_vector",
    "state_from_metrics",
]
