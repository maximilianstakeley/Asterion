
"""Collect host telemetry while preserving missing values and collection errors."""

from __future__ import annotations

import math
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO 8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _metric(
    value: Any,
    unit: str,
    error: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Build a consistent metric record without treating missing data as zero."""
    status = "ok" if error is None and value is not None else "unavailable"
    result = {
        "value": value,
        "unit": unit,
        "status": status,
        "error": error,
    }
    result.update(metadata)
    return result


def _safe_collect(
    collector: Callable[[], Any],
    unit: str,
) -> dict[str, Any]:
    """Collect one metric without allowing its failure to stop other metrics."""
    try:
        value = collector()
        if value is None:
            return _metric(None, unit, error="Collector returned no value")
        return _metric(value, unit)
    except Exception as exc:
        return _metric(None, unit, error=f"{type(exc).__name__}: {exc}")


def _is_finite_number(value: Any) -> bool:
    """Return True for finite numeric values, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class TelemetryCollector:
    """Collect CPU, memory, disk, load, network, and connection telemetry."""

    def __init__(
        self,
        agent_id: str,
        host_id: str | None = None,
        disk_path: str | None = None,
        cpu_sample_seconds: float = 0.1,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")

        if (
            not _is_finite_number(cpu_sample_seconds)
            or cpu_sample_seconds < 0
        ):
            raise ValueError(
                "cpu_sample_seconds must be a finite number >= 0"
            )

        self.agent_id = agent_id.strip()
        self.host_id = host_id or socket.gethostname()
        self.disk_path = disk_path or os.getenv("ASTERION_DISK_PATH", "/")
        self.cpu_sample_seconds = float(cpu_sample_seconds)

        # Separate previous samples for aggregate and per-interface rates.
        self._previous_network: dict[str, Any] | None = None
        self._previous_interfaces: dict[str, dict[str, Any]] | None = None

    @staticmethod
    def _filesystem_identity(path: str) -> dict[str, str | None]:
        """Return verified mount metadata, or explicit unknown values."""
        try:
            resolved_path = Path(path).resolve()
            matches = []

            for partition in psutil.disk_partitions(all=True):
                try:
                    mount_path = Path(partition.mountpoint).resolve()
                    resolved_path.relative_to(mount_path)
                    matches.append(
                        (len(str(mount_path)), partition, mount_path)
                    )
                except (ValueError, OSError, RuntimeError):
                    continue

            if not matches:
                return {
                    "mount_point": None,
                    "device": None,
                    "filesystem_type": None,
                }

            _, partition, mount_path = max(
                matches,
                key=lambda item: item[0],
            )
            return {
                "mount_point": str(mount_path),
                "device": partition.device or None,
                "filesystem_type": partition.fstype or None,
            }
        except (OSError, RuntimeError, ValueError):
            return {
                "mount_point": None,
                "device": None,
                "filesystem_type": None,
            }

    def _collect_cpu(self) -> float:
        value = psutil.cpu_percent(interval=self.cpu_sample_seconds)
        if not _is_finite_number(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid CPU utilization: {value!r}")
        return float(value)

    @staticmethod
    def _collect_memory() -> float:
        value = psutil.virtual_memory().percent
        if not _is_finite_number(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid memory utilization: {value!r}")
        return float(value)

    def _collect_disk(self) -> dict[str, Any]:
        requested_path = str(Path(self.disk_path).resolve())
        usage = psutil.disk_usage(requested_path)

        if usage.total <= 0:
            raise ValueError(
                f"Filesystem reports invalid total capacity: {usage.total}"
            )

        percent = usage.percent
        if not _is_finite_number(percent) or not 0 <= percent <= 100:
            raise ValueError(f"Invalid disk utilization: {percent!r}")

        return {
            "percent": float(percent),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "requested_path": requested_path,
            **self._filesystem_identity(requested_path),
        }

    @staticmethod
    def _collect_load_average() -> dict[str, float]:
        if not hasattr(os, "getloadavg"):
            raise NotImplementedError(
                "Load average is not supported on this platform"
            )

        load_1m, load_5m, load_15m = os.getloadavg()
        values = (load_1m, load_5m, load_15m)
        if not all(_is_finite_number(value) for value in values):
            raise ValueError("Load average returned a non-finite value")

        return {
            "1m": float(load_1m),
            "5m": float(load_5m),
            "15m": float(load_15m),
        }

    @staticmethod
    def _counter_values(counters: Any) -> dict[str, int]:
        """Extract validated cumulative network counters."""
        if counters is None:
            raise RuntimeError("Network I/O counters are unavailable")

        names = (
            "bytes_sent",
            "bytes_recv",
            "packets_sent",
            "packets_recv",
        )
        values: dict[str, int] = {}

        for name in names:
            value = getattr(counters, name, None)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"Invalid network counter {name}: {value!r}")
            values[name] = value

        return values

    @staticmethod
    def _calculate_rates(
        current: dict[str, int],
        previous: dict[str, Any] | None,
        elapsed: float | None,
    ) -> tuple[dict[str, float] | None, str | None]:
        """Calculate rates only when two valid samples exist."""
        if previous is None or elapsed is None:
            return None, None

        if not _is_finite_number(elapsed) or elapsed <= 0:
            return None, "Invalid sample interval"

        deltas: dict[str, int] = {}
        for name, value in current.items():
            old_value = previous.get(name)
            if not isinstance(old_value, int) or value < old_value:
                return None, f"{name} decreased or became invalid"

            deltas[name] = value - old_value

        return {
            "bytes_sent_per_second": deltas["bytes_sent"] / elapsed,
            "bytes_recv_per_second": deltas["bytes_recv"] / elapsed,
            "packets_sent_per_second": deltas["packets_sent"] / elapsed,
            "packets_recv_per_second": deltas["packets_recv"] / elapsed,
        }, None

    def _collect_network_traffic(self) -> dict[str, Any]:
        """Collect aggregate counters and rates without inventing first-sample rates."""
        sampled_at = utc_timestamp()
        sampled_monotonic = time.monotonic()
        counters = self._counter_values(
            psutil.net_io_counters(pernic=False)
        )

        previous = self._previous_network
        elapsed = (
            sampled_monotonic - previous["sample_monotonic"]
            if previous is not None
            else None
        )
        rates, rate_error = self._calculate_rates(
            counters,
            previous["counters"] if previous is not None else None,
            elapsed,
        )

        self._previous_network = {
            **counters,
            "counters": counters.copy(),
            "sampled_at": sampled_at,
            "sample_monotonic": sampled_monotonic,
        }

        result: dict[str, Any] = {
            **counters,
            "sampled_at": sampled_at,
            "rate_interval_seconds": elapsed,
            "rates": rates,
            "bytes_sent_per_second": None,
            "bytes_recv_per_second": None,
            "packets_sent_per_second": None,
            "packets_recv_per_second": None,
            "rate_status": "unavailable",
            "rate_error": rate_error,
        }

        if rates is not None:
            result.update(rates)
            result["rate_status"] = "ok"
        elif previous is None:
            result["rate_error"] = (
                "First sample establishes a baseline; rates unavailable"
            )

        return result

    def _collect_network_interfaces(self) -> dict[str, Any]:
        """Collect per-interface counters and rates where available."""
        sampled_at = utc_timestamp()
        sampled_monotonic = time.monotonic()
        all_counters = psutil.net_io_counters(pernic=True)

        if all_counters is None:
            raise RuntimeError("Per-interface network counters unavailable")

        previous = self._previous_interfaces or {}
        interfaces: dict[str, Any] = {}
        current_samples: dict[str, dict[str, Any]] = {}

        for interface, raw_counters in all_counters.items():
            try:
                counters = self._counter_values(raw_counters)
                old = previous.get(interface)
                elapsed = (
                    sampled_monotonic - old["sample_monotonic"]
                    if old is not None
                    else None
                )
                rates, rate_error = self._calculate_rates(
                    counters,
                    old["counters"] if old is not None else None,
                    elapsed,
                )

                current_samples[interface] = {
                    "counters": counters.copy(),
                    "sample_monotonic": sampled_monotonic,
                }

                interfaces[interface] = {
                    **counters,
                    "sampled_at": sampled_at,
                    "rate_interval_seconds": elapsed,
                    "rates": rates,
                    "rate_status": "ok" if rates is not None else "unavailable",
                    "rate_error": (
                        rate_error
                        if old is not None
                        else "First sample establishes interface baseline"
                    ),
                }
            except Exception as exc:
                interfaces[interface] = {
                    "bytes_sent": None,
                    "bytes_recv": None,
                    "packets_sent": None,
                    "packets_recv": None,
                    "sampled_at": sampled_at,
                    "rates": None,
                    "rate_status": "unavailable",
                    "rate_error": f"{type(exc).__name__}: {exc}",
                }

        # Removed interfaces are not carried forward into the next baseline.
        self._previous_interfaces = current_samples

        return {
            "sampled_at": sampled_at,
            "interfaces": interfaces,
        }

    @staticmethod
    def _collect_connections() -> dict[str, Any]:
        """Count Internet connections only when operating-system access permits."""
        connections = psutil.net_connections(kind="inet")
        state_counts: dict[str, int] = {}

        for connection in connections:
            state = getattr(connection, "status", None) or "UNKNOWN"
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "count": len(connections),
            "state_counts": state_counts,
        }

    @staticmethod
    def _record_status(
        metrics: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Expose each metric's status, error, and units separately."""
        return {
            name: {
                "status": record["status"],
                "error": record["error"],
                "unit": record["unit"],
            }
            for name, record in metrics.items()
        }

    def collect(self) -> dict[str, Any]:
        """Return one timestamped observation, retaining partial results."""
        observation_timestamp = utc_timestamp()

        cpu = _safe_collect(self._collect_cpu, "%")
        memory = _safe_collect(self._collect_memory, "%")
        disk = _safe_collect(self._collect_disk, "bytes; %")
        load = _safe_collect(self._collect_load_average, "load average")
        network = _safe_collect(self._collect_network_traffic, "bytes; packets")
        interfaces = _safe_collect(
            self._collect_network_interfaces,
            "bytes; packets",
        )
        connections = _safe_collect(self._collect_connections, "connections")

        metric_records = {
            "cpu_percent": cpu,
            "memory_percent": memory,
            "disk": disk,
            "load_average": load,
            "network_traffic": network,
            "network_interfaces": interfaces,
            "connection_metadata": connections,
        }

        disk_value = disk["value"]
        load_value = load["value"]
        network_value = network["value"]
        interfaces_value = interfaces["value"]
        connection_value = connections["value"]

        disk_percent = (
            disk_value.get("percent")
            if isinstance(disk_value, dict)
            else None
        )
        load_1m = (
            load_value.get("1m")
            if isinstance(load_value, dict)
            else None
        )
        connection_count = (
            connection_value.get("count")
            if isinstance(connection_value, dict)
            else None
        )

        metrics = {
            "cpu_percent": cpu["value"],
            "memory_percent": memory["value"],
            "disk_percent": disk_percent,
            "network_connections": connection_count,
            "load_1m": load_1m,
            "disk": disk_value,
            "load_average": load_value,
            "network_traffic": network_value,
            "network_interfaces": interfaces_value,
            "connection_metadata": connection_value,
        }

        statuses = self._record_status(metric_records)

        # Core metrics determine whether the observation is usable.
        core_metric_names = (
            "cpu_percent",
            "memory_percent",
            "disk",
            "load_average",
            "network_traffic",
            "connection_metadata",
        )

        core_states = [
            statuses[name]["status"]
            for name in core_metric_names
        ]

        # If all core collectors fail, retain the raw interface data if
        # present, but mark it unavailable in this failed observation.

        if all(status == "unavailable" for status in core_states):
            statuses["network_interfaces"]["status"] = "unavailable"
            if statuses["network_interfaces"]["error"] is None:
                statuses["network_interfaces"]["error"] = (
                    "Core telemetry collection failed; "
                    "interface data is supplementary"
                )
            collection_status = "failed"
        elif all(status == "ok" for status in core_states):
            collection_status = "ok"
        else:
            collection_status = "partial"

        return {
            "agent_id": self.agent_id,
            "host": self.host_id,
            "timestamp": observation_timestamp,
            "collection_status": collection_status,
            "metrics": metrics,
            "metric_status": statuses,
        }



def propose_network_optimizations(
    observation: dict[str, Any],
    *,
    receive_bytes_per_second_threshold: float = 10_000_000,
    error_delta_threshold: int = 100,
    drop_delta_threshold: int = 100,
) -> list[dict[str, Any]]:
    """Create evidence-based network recommendations; never execute changes."""
    if not _is_finite_number(receive_bytes_per_second_threshold) or (
        receive_bytes_per_second_threshold < 0
    ):
        raise ValueError("receive threshold must be finite and >= 0")

    if (
        not isinstance(error_delta_threshold, int)
        or isinstance(error_delta_threshold, bool)
        or error_delta_threshold < 0
    ):
        raise ValueError("error_delta_threshold must be an integer >= 0")

    if (
        not isinstance(drop_delta_threshold, int)
        or isinstance(drop_delta_threshold, bool)
        or drop_delta_threshold < 0
    ):
        raise ValueError("drop_delta_threshold must be an integer >= 0")

    metrics = observation.get("metrics", {})
    if not isinstance(metrics, dict):
        return []

    traffic = metrics.get("network_traffic")
    if not isinstance(traffic, dict):
        return []

    recommendations: list[dict[str, Any]] = []
    receive_rate = traffic.get("bytes_recv_per_second")

    if (
        _is_finite_number(receive_rate)
        and receive_rate >= receive_bytes_per_second_threshold
    ):
        recommendations.append({
            "category": "network_throughput",
            "severity": "review",
            "evidence": {
                "bytes_received_per_second": receive_rate,
                "threshold": receive_bytes_per_second_threshold,
            },
            "recommendation": (
                "Inspect per-interface traffic, expected workload, and "
                "application demand before considering network changes."
            ),
            "execution": "disabled",
        })

    # Counter deltas require a previous-sample comparison. Cumulative totals
    # alone are not evidence of current errors or drops.
    interface_record = metrics.get("network_interfaces")
    interface_data = (
        interface_record.get("interfaces")
        if isinstance(interface_record, dict)
        else None
    )

    if isinstance(interface_data, dict):
        for interface, data in interface_data.items():
            if not isinstance(data, dict):
                continue

            for field, threshold, category in (
                ("errors_delta", error_delta_threshold, "network_errors"),
                ("drops_delta", drop_delta_threshold, "network_drops"),
            ):
                delta = data.get(field)
                if (
                    isinstance(delta, int)
                    and not isinstance(delta, bool)
                    and delta >= threshold
                ):
                    recommendations.append({
                        "category": category,
                        "severity": "review",
                        "evidence": {
                            "interface": interface,
                            field: delta,
                            "threshold": threshold,
                        },
                        "recommendation": (
                            "Inspect interface health, link conditions, "
                            "driver diagnostics, and relevant system logs."
                        ),
                        "execution": "disabled",
                    })

    return recommendations
