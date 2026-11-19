
"""
Asterion network monitoring and advisory optimization.

Collects per-interface network counters, calculates rates from successive
samples, and reports connection states when permitted by the operating system.

This module is observation-only. It does not modify network configuration,
firewall rules, routes, sockets, or processes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import threading
import time
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class NetworkMonitor:
    """Collect per-interface network statistics across sampling intervals."""

    COUNTER_ATTRIBUTES = {
        "bytes_sent": "bytes_sent",
        "bytes_recv": "bytes_recv",
        "packets_sent": "packets_sent",
        "packets_recv": "packets_recv",
        "errors_out": "errout",
        "errors_in": "errin",
        "drops_out": "dropout",
        "drops_in": "dropin",
    }

    REQUIRED_COUNTERS = (
        "bytes_sent",
        "bytes_recv",
        "packets_sent",
        "packets_recv",
    )

    def __init__(self, *, include_loopback: bool = False) -> None:
        self.include_loopback = include_loopback

        # Interface name -> (monotonic timestamp, previous counter values).
        self._previous: dict[
            str, tuple[float, dict[str, int | None]]
        ] = {}

        self._lock = threading.Lock()

    def collect(self) -> dict[str, Any]:
        """Collect one observation without substituting missing measurements."""

        timestamp = utc_now()

        if psutil is None:
            return {
                "timestamp": timestamp,
                "collection_status": "failed",
                "interfaces": None,
                "connection_summary": None,
                "metric_status": {
                    "network_interfaces": {
                        "status": "unavailable",
                        "reason": "psutil is not installed",
                    },
                    "network_connections": {
                        "status": "unavailable",
                        "reason": "psutil is not installed",
                    },
                },
            }

        try:
            counters = psutil.net_io_counters(
                pernic=True,
                nowrap=False,
            )

            if counters is None:
                raise RuntimeError(
                    "Operating system returned no interface counters"
                )

        except Exception as exc:
            return {
                "timestamp": timestamp,
                "collection_status": "failed",
                "interfaces": None,
                "connection_summary": None,
                "metric_status": {
                    "network_interfaces": {
                        "status": "unavailable",
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                    "network_connections": {
                        "status": "unavailable",
                        "reason": (
                            "Interface collection failed; "
                            "connection inspection was not attempted"
                        ),
                    },
                },
            }

        sample_time = time.monotonic()

        with self._lock:
            previous_samples = dict(self._previous)

        interfaces: dict[str, Any] = {}
        current_samples: dict[
            str, tuple[float, dict[str, int | None]]
        ] = {}

        for interface_name, counter_object in counters.items():

            if (
                not self.include_loopback
                and interface_name.lower()
                in {"lo", "lo0", "loopback"}
            ):
                continue

            values: dict[str, int | None] = {}

            for field, attribute in self.COUNTER_ATTRIBUTES.items():
                value = getattr(counter_object, attribute, None)

                if isinstance(value, int) and value >= 0:
                    values[field] = value
                else:
                    # Some operating systems do not expose every counter.
                    values[field] = None

            previous = previous_samples.get(interface_name)

            elapsed = (
                sample_time - previous[0]
                if previous is not None
                else None
            )

            def calculate_rate(field: str) -> float | None:
                """Calculate a rate only from valid, monotonic counter data."""

                if previous is None or elapsed is None or elapsed <= 0:
                    return None

                current_value = values.get(field)
                previous_value = previous[1].get(field)

                if current_value is None or previous_value is None:
                    return None

                # A decreasing counter may indicate reset or wraparound.
                # Do not manufacture a rate from that observation.
                if current_value < previous_value:
                    return None

                rate = (current_value - previous_value) / elapsed

                if not math.isfinite(rate) or rate < 0:
                    return None

                return rate

            required_available = all(
                values[field] is not None
                for field in self.REQUIRED_COUNTERS
            )

            rates = {
                "bytes_sent_per_second": calculate_rate("bytes_sent"),
                "bytes_recv_per_second": calculate_rate("bytes_recv"),
                "packets_sent_per_second": calculate_rate("packets_sent"),
                "packets_recv_per_second": calculate_rate("packets_recv"),
            }

            valid_interval = elapsed is not None and elapsed > 0

            interfaces[interface_name] = {
                "status": (
                    "ok" if required_available else "partial"
                ),
                **{
                    f"{field}_total": value
                    for field, value in values.items()
                },
                **rates,
                "sample_interval_seconds": (
                    elapsed if valid_interval else None
                ),
                "rate_status": (
                    "ok"
                    if valid_interval
                    and any(value is not None for value in rates.values())
                    else "unavailable"
                ),
                "rate_unavailable_reason": (
                    None
                    if valid_interval
                    and any(value is not None for value in rates.values())
                    else (
                        "A second valid sample is required, "
                        "or counters reset or became unavailable"
                    )
                ),
            }

            current_samples[interface_name] = (
                sample_time,
                values,
            )

        with self._lock:
            self._previous = current_samples

        connection_summary, connection_status = (
            self._collect_connections()
        )

        interface_status = (
            "ok"
            if interfaces
            and all(
                item["status"] == "ok"
                for item in interfaces.values()
            )
            else "partial"
        )

        return {
            "timestamp": timestamp,
            "collection_status": (
                "ok"
                if interface_status == "ok"
                and connection_status["status"] == "ok"
                else "partial"
            ),
            "interfaces": interfaces,
            "connection_summary": connection_summary,
            "metric_status": {
                "network_interfaces": {
                    "status": interface_status,
                    "reason": (
                        None
                        if interface_status == "ok"
                        else (
                            "Some interface counters are unavailable "
                            "or no eligible interfaces were found"
                        )
                    ),
                },
                "network_connections": connection_status,
            },
        }

    @staticmethod
    def _collect_connections(
    ) -> tuple[dict[str, Any] | None, dict[str, str | None]]:
        """
        Count visible Internet socket states.

        Connection addresses and process names are not retained.
        Permission failures produce unavailable data, never a count of zero.
        """

        if psutil is None:
            return None, {
                "status": "unavailable",
                "reason": "psutil is not installed",
            }

        try:
            connections = psutil.net_connections(kind="inet")

            state_counts: dict[str, int] = {}

            for connection in connections:
                state = str(
                    getattr(connection, "status", None) or "UNKNOWN"
                ).upper()

                state_counts[state] = state_counts.get(state, 0) + 1

            return {
                "total_visible_connections": len(connections),
                "states": state_counts,
                "scope": (
                    "Connections visible to the current process; "
                    "results may be permission-limited"
                ),
            }, {
                "status": "ok",
                "reason": None,
            }

        except Exception as exc:
            access_denied = isinstance(exc, PermissionError)

            if psutil is not None:
                access_denied = access_denied or isinstance(
                    exc,
                    getattr(psutil, "AccessDenied", PermissionError),
                )

            reason = (
                "Operating system denied connection inspection"
                if access_denied
                else "Connection inspection failed"
            )

            return None, {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {reason}",
            }


def propose_network_optimizations(
    observation: dict[str, Any],
    *,
    high_receive_bytes_per_second: float = 50_000_000,
    high_error_delta: int = 10,
    high_drop_delta: int = 10,
) -> list[dict[str, Any]]:
    """
    Generate evidence-based recommendations without executing them.

    The thresholds are configurable starting points, not universal
    definitions of network failure.
    """

    proposals: list[dict[str, Any]] = []

    interfaces = observation.get("interfaces")

    if not isinstance(interfaces, dict):
        return proposals

    for interface_name, data in interfaces.items():

        if not isinstance(data, dict):
            continue

        receive_rate = data.get("bytes_recv_per_second")

        if (
            isinstance(receive_rate, (int, float))
            and math.isfinite(receive_rate)
            and receive_rate >= high_receive_bytes_per_second
        ):
            proposals.append({
                "category": "capacity_review",
                "severity": "info",
                "interface": interface_name,
                "evidence": {
                    "bytes_recv_per_second": receive_rate,
                    "threshold": high_receive_bytes_per_second,
                },
                "recommendation": (
                    "Verify whether this throughput is expected. "
                    "Compare application demand and interface capacity "
                    "before changing configuration."
                ),
                "execution": "disabled",
            })

        checks = (
            (
                "errors_in_delta",
                "receive_errors",
                high_error_delta,
                (
                    "Inspect link health, driver logs, and upstream "
                    "equipment. Confirm repeated error deltas."
                ),
            ),
            (
                "errors_out_delta",
                "transmit_errors",
                high_error_delta,
                (
                    "Inspect link health and driver logs. Confirm "
                    "repeated transmit-error deltas."
                ),
            ),
            (
                "drops_in_delta",
                "receive_drops",
                high_drop_delta,
                (
                    "Check receive queue pressure and host load "
                    "across multiple samples before tuning buffers."
                ),
            ),
            (
                "drops_out_delta",
                "transmit_drops",
                high_drop_delta,
                (
                    "Check transmit queue pressure and congestion "
                    "across multiple samples before changing queue settings."
                ),
            ),
        )

        for field, category, threshold, recommendation in checks:
            value = data.get(field)

            if isinstance(value, int) and value >= threshold:
                proposals.append({
                    "category": category,
                    "severity": "warning",
                    "interface": interface_name,
                    "evidence": {
                        field: value,
                        "threshold": threshold,
                    },
                    "recommendation": recommendation,
                    "execution": "disabled",
                })

    return proposals
