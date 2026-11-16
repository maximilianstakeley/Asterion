
"""Local network connection measurements with explicit availability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True)
class NetworkMeasurement:
    available: bool
    connection_count: int | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available:
            if self.connection_count is None or self.connection_count < 0:
                raise ValueError(
                    "Available measurements require a non-negative count."
                )
            if self.reason is not None:
                raise ValueError(
                    "Available measurements must not have a failure reason."
                )
        elif self.connection_count is not None:
            raise ValueError(
                "Unavailable measurements must not contain a count."
            )


def measure_network_connections(
    provider: Callable[..., Any] | None = None,
) -> NetworkMeasurement:
    """Return a count or an explicit unavailable result.

    `provider` is injectable for deterministic testing. In production,
    psutil.net_connections is used.
    """
    try:
        import psutil
    except ImportError:
        return NetworkMeasurement(
            available=False,
            connection_count=None,
            reason="psutil_not_installed",
        )

    source = provider if provider is not None else psutil.net_connections

    try:
        connections = source(kind="all")
        if connections is None:
            return NetworkMeasurement(
                available=False,
                connection_count=None,
                reason="provider_returned_no_measurement",
            )

        # A successful empty list means a genuine measured zero.
        return NetworkMeasurement(
            available=True,
            connection_count=len(connections),
        )

    except (psutil.Error, OSError, AttributeError) as exc:
        return NetworkMeasurement(
            available=False,
            connection_count=None,
            reason=type(exc).__name__,
        )
