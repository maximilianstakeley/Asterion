
"""
Asterion telemetry monitoring agent.

Collects system telemetry independently, validates observations, and submits
them to the central Asterion API.

Run:
    python -m Asterion.init \
        --central-url http://127.0.0.1:8765 \
        --agent-id host-01 \
        --interval 5

Environment:
    ASTERION_API_TOKEN  API authentication token
    ASTERION_DISK_PATH Filesystem path to monitor; defaults to /
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

from datetime import datetime, timezone
from typing import Any

import psutil
import requests


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def finite_number(value: Any) -> bool:
    """Return True for finite numeric values, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def metric(
    value: Any,
    unit: str,
    status: str = "ok",
    error: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Create a consistently structured metric record."""
    result = {
        "value": value,
        "unit": unit,
        "status": status,
        "error": error,
    }
    result.update(metadata)
    return result


def unavailable(unit: str, exc: BaseException) -> dict[str, Any]:
    """Represent a failed or unsupported measurement without fabricating zero."""
    return metric(
        value=None,
        unit=unit,
        status="unavailable",
        error=f"{type(exc).__name__}: {exc}",
    )


def validate_percentage(value: Any, name: str) -> float:
    """Validate a percentage in the inclusive range 0–100."""
    if not finite_number(value) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be a finite percentage from 0 to 100")
    return float(value)


class TelemetryCollector:
    """Collect CPU, memory, disk, load, network, and connection telemetry."""

    def __init__(
        self,
        agent_id: str,
        host_id: str | None = None,
        disk_path: str | None = None,
        cpu_sample_seconds: float = 0.1,
    ) -> None:
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id must not be empty")

        if not finite_number(cpu_sample_seconds) or cpu_sample_seconds <= 0:
            raise ValueError("cpu_sample_seconds must be greater than zero")

        self.agent_id = agent_id.strip()
        self.host_id = (host_id or socket.gethostname()).strip()
        if not self.host_id:
            raise ValueError("host_id must not be empty")

        self.disk_path = os.path.abspath(
            disk_path or os.getenv("ASTERION_DISK_PATH", "/")
        )
        self.cpu_sample_seconds = float(cpu_sample_seconds)
        self._previous_network: dict[str, Any] | None = None

    @staticmethod
    def _run_metric(
        unit: str,
        collector: Any,
    ) -> dict[str, Any]:
        """Isolate collection errors so one metric cannot discard the others."""
        try:
            return collector()
        except Exception as exc:
            return unavailable(unit, exc)

    def _collect_cpu(self) -> dict[str, Any]:
        value = validate_percentage(
            psutil.cpu_percent(interval=self.cpu_sample_seconds),
            "CPU utilization",
        )
        return metric(value, "%")

    def _collect_memory(self) -> dict[str, Any]:
        value = validate_percentage(
            psutil.virtual_memory().percent,
            "Memory utilization",
        )
        return metric(value, "%")

    def _collect_disk(self) -> dict[str, Any]:
        """
        Report utilization for the configured path and identify its filesystem.

        The mount is determined by the longest matching mount point. If the
        filesystem cannot be identified reliably, the measurement is marked
        unavailable rather than inventing filesystem metadata.
        """
        usage = psutil.disk_usage(self.disk_path)
        percent = validate_percentage(usage.percent, "Disk utilization")

        resolved_path = os.path.realpath(self.disk_path)
        matching_mounts = []


        for partition in psutil.disk_partitions(all=True):
            mountpoint = os.path.realpath(partition.mountpoint)

            try:
                if os.path.commonpath(
                    [resolved_path, mountpoint]
                ) == mountpoint:
                    matching_mounts.append((len(mountpoint), partition))
            except (ValueError, OSError):
                continue

        if not matching_mounts:
            raise RuntimeError(
                f"Could not identify the mounted filesystem for "
                f"{self.disk_path!r}"
            )

        _, partition = max(matching_mounts, key=lambda item: item[0])

        if (
            not partition.mountpoint
            or not partition.device
            or not partition.fstype
        ):
            raise RuntimeError(
                "The operating system did not provide complete filesystem "
                f"identity for monitored path {self.disk_path!r}"
            )

        return metric(
            value=percent,
            unit="%",
            path=self.disk_path,
            resolved_path=resolved_path,
            mountpoint=partition.mountpoint,
            device=partition.device,
            filesystem_type=partition.fstype or None,
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,)

    def _collect_load(self) -> dict[str, Any]:
        if not hasattr(os, "getloadavg"):
            raise NotImplementedError(
                "Load average is not supported on this platform"
            )

        load_1m, load_5m, load_15m = os.getloadavg()
        values = (load_1m, load_5m, load_15m)

        if not all(finite_number(value) and value >= 0 for value in values):
            raise ValueError("Load average returned invalid values")

        return metric(
            value=float(load_1m),
            unit="load",
            load_1m=float(load_1m),
            load_5m=float(load_5m),
            load_15m=float(load_15m),
        )

    def _collect_network(self) -> dict[str, Any]:
        """
        Collect system-wide interface counters and rates.

        The first successful sample establishes a baseline; rates remain null
        until a second sample is available. Counters and rates are separate.
        """
        counters = psutil.net_io_counters(pernic=False)
        if counters is None:
            raise RuntimeError("Network I/O counters are unavailable")

        sampled_at = utc_now()
        sampled_monotonic = time.monotonic()

        current = {
            "bytes_sent": int(counters.bytes_sent),
            "bytes_received": int(counters.bytes_recv),
            "packets_sent": int(counters.packets_sent),
            "packets_received": int(counters.packets_recv),
            "sampled_at": sampled_at,
            "sampled_monotonic": sampled_monotonic,
        }

        rates = {
            "bytes_sent_per_second": None,
            "bytes_received_per_second": None,
            "packets_sent_per_second": None,
            "packets_received_per_second": None,
        }
        rate_status = "baseline"
        rate_error = None

        previous = self._previous_network
        if previous is not None:
            elapsed = sampled_monotonic - previous["sampled_monotonic"]

            if elapsed > 0:
                pairs = (
                    ("bytes_sent", "bytes_sent_per_second"),
                    ("bytes_received", "bytes_received_per_second"),
                    ("packets_sent", "packets_sent_per_second"),
                    ("packets_received", "packets_received_per_second"),
                )

                deltas = {
                    name: current[name] - previous[name]
                    for name, _ in pairs
                }

                if any(delta < 0 for delta in deltas.values()):
                    rate_status = "unavailable"
                    rate_error = (
                        "Network counters decreased between samples; "
                        "rates were not calculated"
                    )
                else:
                    for counter_name, rate_name in pairs:
                        rates[rate_name] = (
                            deltas[counter_name] / elapsed
                        )
                    rate_status = "ok"
            else:
                rate_status = "unavailable"
                rate_error = "Non-positive sample interval"

        self._previous_network = current

        return metric(
            value=None,
            unit="bytes, packets, bytes/s, packets/s",
            counters={
                "bytes_sent": current["bytes_sent"],
                "bytes_received": current["bytes_received"],
                "packets_sent": current["packets_sent"],
                "packets_received": current["packets_received"],
            },
            rates=rates,
            rate_status=rate_status,
            rate_error=rate_error,
            sampled_at=sampled_at,
        )

    def _collect_connections(self) -> dict[str, Any]:
        """
        Count internet sockets when the operating system permits access.

        Permission failures produce an unavailable metric, not a zero count.
        """
        connections = psutil.net_connections(kind="inet")
        state_counts: dict[str, int] = {}

        for connection in connections:
            state = getattr(connection, "status", None) or "UNKNOWN"
            state_counts[state] = state_counts.get(state, 0) + 1

        return metric(
            value=len(connections),
            unit="connections",
            state_counts=state_counts,
        )

    def collect(self) -> dict[str, Any]:
        """Collect a complete observation, retaining successful metrics."""
        collection_started = utc_now()

        collectors = {
            "cpu_percent": ("%", self._collect_cpu),
            "memory_percent": ("%", self._collect_memory),
            "disk_percent": ("%", self._collect_disk),
            "load_average": ("load", self._collect_load),
            "network_io": (
                "bytes, packets, bytes/s, packets/s",
                self._collect_network,
            ),
            "network_connections": (
                "connections",
                self._collect_connections,
            ),
        }

        metrics: dict[str, Any] = {}
        metric_status: dict[str, str] = {}

        for name, (unit, collector) in collectors.items():
            result = self._run_metric(unit, collector)

            # Record a timestamp for every metric, including failed samples.
            result.setdefault("sampled_at", utc_now())

            metrics[name] = result
            metric_status[name] = result["status"]

        failed_count = sum(
            status != "ok" for status in metric_status.values()
        )

        if failed_count == 0:
            collection_status = "ok"
        elif failed_count == len(metric_status):
            collection_status = "failed"
        else:
            collection_status = "partial"

        observation = {
            "timestamp": collection_started,
            "agent_id": self.agent_id,
            "host": self.host_id,
            "collection_status": collection_status,
            "metrics": metrics,
            "metric_status": metric_status,
        }

        validate_observation(observation)
        return observation


def validate_observation(observation: dict[str, Any]) -> None:
    """Validate the observation envelope and key telemetry invariants."""
    for field in ("timestamp", "agent_id", "host", "collection_status"):
        value = observation.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Observation field {field!r} must be non-empty")

    timestamp = datetime.fromisoformat(
        observation["timestamp"].replace("Z", "+00:00")
    )
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Observation timestamp must include a timezone")

    if observation["collection_status"] not in {
        "ok",
        "partial",
        "failed",
    }:
        raise ValueError("Invalid collection_status")

    metrics = observation.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Observation metrics must be a non-empty object")

    metric_status = observation.get("metric_status")
    if not isinstance(metric_status, dict):
        raise ValueError("metric_status must be an object")


    # Require one status entry for every metric, and no unrelated entries.
    if set(metric_status.keys()) != set(metrics.keys()):
        raise ValueError(
            "metric_status keys must exactly match metrics keys"
        )

    # Validate metric record structure before deriving collection status.
    for name, record in metrics.items():
        if not isinstance(record, dict):
            raise ValueError(f"Metric {name!r} must be an object")

        if record.get("status") not in {"ok", "unavailable"}:
            raise ValueError(f"Metric {name!r} has an invalid status")


    # Derive the overall collection status from individual metric outcomes.
    metric_outcomes = [
        record.get("status")
        for record in metrics.values()
        if isinstance(record, dict)
    ]

    if metric_outcomes and all(status == "ok" for status in metric_outcomes):
        expected_collection_status = "ok"
    elif metric_outcomes and all(
        status == "unavailable" for status in metric_outcomes
    ):
        expected_collection_status = "failed"
    else:
        expected_collection_status = "partial"

    if observation["collection_status"] != expected_collection_status:
        raise ValueError(
            "collection_status is inconsistent with individual metric statuses"
        )


    for name, record in metrics.items():
        if not isinstance(record, dict):
            raise ValueError(f"Metric {name!r} must be an object")

        if record.get("status") not in {"ok", "unavailable"}:
            raise ValueError(f"Metric {name!r} has an invalid status")

        if not isinstance(record.get("unit"), str) or not record["unit"]:
            raise ValueError(f"Metric {name!r} must declare its unit")

        value = record.get("value")

        if record["status"] == "unavailable":
            if value is not None:
                raise ValueError(
                    f"Unavailable metric {name!r} must have value=None"
                )
            if not record.get("error"):
                raise ValueError(
                    f"Unavailable metric {name!r} must include an error"
                )

        if name in {"cpu_percent", "memory_percent", "disk_percent"}:
            if record["status"] == "ok":
                validate_percentage(value, name)

        if name == "network_connections" and record["status"] == "ok":
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    "network_connections must be a non-negative integer"
                )

        if name == "load_average" and record["status"] == "ok":
            if not finite_number(value) or value < 0:
                raise ValueError(
                    "load_average must be a finite non-negative number"
                )

        if metric_status.get(name) != record["status"]:
            raise ValueError(
                f"metric_status disagrees with metric {name!r}"
            )

    # Ensure the complete payload can be serialized without NaN or Infinity.
    json.dumps(observation, allow_nan=False)


def submit_observation(
    session: requests.Session,
    central_url: str,
    observation: dict[str, Any],
    token: str | None,
    timeout: float,
) -> None:
    """Submit one validated observation using the API's bearer authentication."""
    if not isinstance(token, str) or not token.strip():
        raise ValueError(
            "ASTERION_API_TOKEN is required; refusing unauthenticated submission."
        )

    validate_observation(observation)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token.strip()}",
    }

    response = session.post(
        f"{central_url.rstrip('/')}/v1/telemetry",
        json=observation,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asterion-monitor",
        description="Collect and submit Asterion host telemetry.",
    )

    parser.add_argument(
        "--central-url",
        default="http://127.0.0.1:8765",
        help="Base URL of the central Asterion API.",
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="Stable identifier for this monitoring agent.",
    )
    parser.add_argument(
        "--host-id",
        default=None,
        help="Host identifier; defaults to the system hostname.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between collection cycles (minimum 1 second).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of cycles; 0 means run continuously.",
    )
    parser.add_argument(
        "--disk-path",
        default=os.getenv("ASTERION_DISK_PATH", "/"),
        help="Filesystem path to monitor.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.agent_id.strip():
        parser.error("--agent-id must not be empty")

    if args.host_id is not None and not args.host_id.strip():
        parser.error("--host-id must not be empty")

    if not finite_number(args.interval) or args.interval < 1:
        parser.error("--interval must be at least 1 second")

    if args.cycles < 0:
        parser.error("--cycles must be zero or greater")

    if not finite_number(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    if not args.central_url.startswith(("http://", "https://")):
        parser.error("--central-url must start with http:// or https://")

    token = os.getenv("ASTERION_API_TOKEN")
    collector = TelemetryCollector(
        agent_id=args.agent_id,
        host_id=args.host_id,
        disk_path=args.disk_path,)

    if not token or not token.strip():
        parser.error(
            "ASTERION_API_TOKEN is required. Set it in the environment first."
        )
    successful_submissions = 0
    failed_submissions = 0
    cycle = 0

    print(
        f"Asterion monitor started: agent={collector.agent_id}, "
        f"host={collector.host_id}, interval={args.interval}s, "
        f"endpoint={args.central_url.rstrip('/')}/v1/telemetry",
        flush=True,
    )

    try:
        with requests.Session() as session:
            while args.cycles == 0 or cycle < args.cycles:
                cycle += 1
                started = time.monotonic()

                try:
                    observation = collector.collect()
                    print(
                        json.dumps(observation, allow_nan=False),
                        flush=True,
                    )

                    submit_observation(
                        session=session,
                        central_url=args.central_url,
                        observation=observation,
                        token=token,
                        timeout=args.timeout,
                    )

                    successful_submissions += 1
                    print(
                        f"[cycle {cycle}] Telemetry submitted successfully.",
                        flush=True,
                    )

                except (requests.RequestException, ValueError, OSError) as exc:
                    failed_submissions += 1
                    print(
                        f"[cycle {cycle}] Telemetry submission/validation "
                        f"failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

                except Exception as exc:
                    # Keep the monitor alive after an unexpected cycle error.
                    failed_submissions += 1
                    print(
                        f"[cycle {cycle}] Unexpected error: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

                if args.cycles == 0 or cycle < args.cycles:
                    remaining = args.interval - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nAsterion monitor interrupted.", flush=True)

    print(
        f"Monitor stopped. Cycles={cycle}, "
        f"successful_submissions={successful_submissions}, "
        f"failed_submissions={failed_submissions}",
        flush=True,
    )

    return 0 if failed_submissions == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
