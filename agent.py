
import argparse
import math
import os
import socket
import time
from datetime import datetime, timezone
import requests
from .telemetry_collector import TelemetryCollector
from dotenv import load_dotenv
load_dotenv()


def collect_metrics(collector):
    """Return one observation from the shared telemetry collector."""
    return collector.collect()


def _is_finite_number(value):
    """Return True only for finite numeric values, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_percentage(metrics, key):
    """Preserve unavailable values; reject invalid percentages."""
    value = metrics.get(key)

    if value is None:
        return None

    if not _is_finite_number(value) or not 0 <= value <= 100:
        return None

    return value
def _build_structured_observation(observation, agent_id, fallback_host):
    """Convert collector output into the /v1/telemetry API contract."""
    import math

    raw_metrics = observation.get("metrics", {})
    raw_status = observation.get("metric_status", {})

    if not isinstance(raw_metrics, dict):
        raise ValueError("Collector metrics must be a dictionary")

    if not isinstance(raw_status, dict):
        raise ValueError("Collector metric_status must be a dictionary")

    metrics = {}
    metric_status = {}

    def add_metric(name, value, unit, source_status, error=None):
        if source_status not in {"ok", "unavailable"}:
            source_status = "unavailable"

        valid_number = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

        if source_status == "ok" and not valid_number:
            source_status = "unavailable"
            error = error or "Value is missing or is not a finite number"

        if source_status == "unavailable":
            value = None
            error = error or "Measurement unavailable"

        metrics[name] = {
            "value": value,
            "unit": unit,
            "status": source_status,
            "error": error,
        }
        metric_status[name] = source_status

    def source_state(name):
        record = raw_status.get(name, {})
        if not isinstance(record, dict):
            return "unavailable", "Invalid collector status record"

        return (
            record.get("status", "unavailable"),
            record.get("error"),
        )

    # Scalar host measurements.
    scalar_specs = (
        ("cpu_percent", "cpu_percent", "%", "cpu_percent"),
        ("memory_percent", "memory_percent", "%", "memory_percent"),
        ("disk_percent", "disk_percent", "%", "disk"),
        ("network_connections", "network_connections",
         "connections", "connection_metadata"),
        ("load_1m", "load_1m", "load average", "load_average"),
    )

    for output_name, input_name, unit, source_name in scalar_specs:
        status, error = source_state(source_name)
        add_metric(
            output_name,
            raw_metrics.get(input_name),
            unit,
            status,
            error,
        )

    # Preserve network counters as individual numeric metrics.
    traffic = raw_metrics.get("network_traffic")
    if not isinstance(traffic, dict):
        traffic = {}

    traffic_status, traffic_error = source_state("network_traffic")

    for name, unit in (
        ("bytes_sent", "bytes"),
        ("bytes_recv", "bytes"),
        ("packets_sent", "packets"),
        ("packets_recv", "packets"),
    ):
        add_metric(
            f"network_{name}",
            traffic.get(name),
            unit,
            traffic_status,
            traffic_error,
        )

    # A rate is unavailable until the collector has two valid samples.
    for name, unit in (
        ("bytes_sent_per_second", "bytes/second"),
        ("bytes_recv_per_second", "bytes/second"),
        ("packets_sent_per_second", "packets/second"),
        ("packets_recv_per_second", "packets/second"),
    ):
        value = traffic.get(name)
        rate_error = (
            traffic.get("rate_error")
            or "Rate unavailable until a valid sampling interval exists"
        )

        add_metric(
            f"network_{name}",
            value,
            unit,
            "ok" if value is not None else "unavailable",
            None if value is not None else rate_error,
        )

    # Preserve detailed nested records separately from scalar API metrics.
    timestamp = observation.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("Collector observation requires a timestamp")

    return {
        "timestamp": timestamp,
        "agent_id": observation.get("agent_id") or agent_id,
        "host": observation.get("host") or fallback_host,
        "collection_status": observation.get(
            "collection_status", "partial"
        ),
        "metrics": metrics,
        "metric_status": metric_status,
        "metric_details": {
            "disk": raw_metrics.get("disk"),
            "load_average": raw_metrics.get("load_average"),
            "network_traffic": traffic,
            "connection_metadata": raw_metrics.get(
                "connection_metadata"
            ),
            "collector_metric_status": raw_status,
        },
    }

def _build_payload(observation, agent_id, fallback_host, text):
    """Build an API payload while preserving available observation metadata."""
    if not isinstance(observation, dict):
        raise ValueError("Collector must return a dictionary")

    observed_metrics = observation.get("metrics")
    if not isinstance(observed_metrics, dict):
        raise ValueError("Collector observation must contain a metrics dictionary")

    host = observation.get("host") or fallback_host

    # Keep missing metrics as None rather than inventing zero values.
    metrics = {
        "cpu_percent": _validate_percentage(
            observed_metrics, "cpu_percent"
        ),
        "memory_percent": _validate_percentage(
            observed_metrics, "memory_percent"
        ),
        "disk_percent": _validate_percentage(
            observed_metrics, "disk_percent"
        ),
        "network_connections": observed_metrics.get("network_connections"),
        "load_1m": observed_metrics.get("load_1m"),
    }

    # Retain the original observation timestamp when the collector provides it.
    timestamp = (
        observation.get("timestamp")
        or datetime.now(timezone.utc).isoformat()
    )

    body = {
        "agent_id": observation.get("agent_id") or agent_id,
        "host": host,
        "text": text,
        "metrics": metrics,
    }

    # Build the structured payload using the telemetry API contract.
    structured_observation = _build_structured_observation(
        observation,
        agent_id,
        fallback_host,
    )
    return body, structured_observation


def main():
    parser = argparse.ArgumentParser(
        description="Authorized telemetry-only agent"
    )
    parser.add_argument("--central-url", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--cycles", type=int, default=0)
    parser.add_argument("--text", default="routine system telemetry")
    args = parser.parse_args()

    if args.interval < 1:
        parser.error("--interval must be at least 1 second")

    if args.cycles < 0:
        parser.error("--cycles must be zero or greater")

    token = os.getenv("ASTERION_API_TOKEN")
    if not token:
        parser.error("Set ASTERION_API_TOKEN first")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
    })

    collector = TelemetryCollector(agent_id=args.agent_id)
    central_url = args.central_url.rstrip("/")
    analyze_url = central_url + "/v1/analyze"

    # Default to the verified central telemetry endpoint.
    # Allow an explicit environment variable to override it.

    # Submit structured telemetry to the verified API endpoint by default.
    telemetry_url = os.getenv(
        "ASTERION_TELEMETRY_URL",
        f"{central_url}/v1/telemetry",
    ).rstrip("/")

    done = 0
    fallback_host = socket.gethostname()

    try:
        while args.cycles == 0 or done < args.cycles:
            cycle_start = time.monotonic()

            try:
                observation = collect_metrics(collector)
                body, structured_observation = _build_payload(
                    observation,
                    args.agent_id,
                    fallback_host,
                    args.text,
                )
            except Exception as exc:
                print(f"Collection or payload preparation failed: {exc}")
                done += 1
                if args.cycles == 0 or done < args.cycles:
                    time.sleep(args.interval)
                continue

            # Preserve compatibility with the existing analysis endpoint.
            try:
                response = session.post(
                    analyze_url,
                    json=body,
                    timeout=8,
                )
                response.raise_for_status()
                print("Analysis submission succeeded:", response.status_code)
            except requests.RequestException as exc:
                print(f"Analysis submission failed: {exc}")
                if exc.response is not None:
                    print(f"HTTP status: {exc.response.status_code}")
                    print(f"API response: {exc.response.text}")

            # Send the full observation only to a verified compatible endpoint.
            if telemetry_url:
                try:
                    response = session.post(
                        telemetry_url,
                        json=structured_observation,
                        timeout=8,
                    )
                    response.raise_for_status()
                    print(
                        "Structured telemetry submission succeeded:",
                        response.status_code,
                    )
                except requests.RequestException as exc:
                    print(f"Structured telemetry submission failed: {exc}")
                    if exc.response is not None:
                        print(f"HTTP status: {exc.response.status_code}")
                        print(f"API response: {exc.response.text}")

            done += 1

            if args.cycles == 0 or done < args.cycles:
                elapsed = time.monotonic() - cycle_start
                time.sleep(max(0, args.interval - elapsed))

    finally:
        session.close()


if __name__ == "__main__":
    main()
