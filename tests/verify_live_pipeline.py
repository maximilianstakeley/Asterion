
"""Live integration check for Asterion telemetry ingestion and persistence.

Uses a uniquely named synthetic test host. Does not execute system actions.
Requires a running Asterion API and ASTERION_API_TOKEN.
"""

import os
import sys
import uuid
from datetime import datetime, timezone

import requests


BASE_URL = os.getenv(
    "ASTERION_CENTRAL_URL",
    "http://127.0.0.1:8765",
).rstrip("/")

TOKEN = os.getenv("ASTERION_API_TOKEN")

if not TOKEN:
    sys.exit("FAIL: Set ASTERION_API_TOKEN before running this test.")

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {TOKEN}"})

test_id = uuid.uuid4().hex[:12]
test_host = f"asterion-test-{test_id}"
test_agent = f"integration-test-{test_id}"

observation = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "agent_id": test_agent,
    "host": test_host,
    "collection_status": "partial",
    "metrics": {
        "cpu_percent": {
            "value": 23.5,
            "unit": "%",
            "status": "ok",
            "error": None,
        },
        "memory_percent": {
            "value": 61.2,
            "unit": "%",
            "status": "ok",
            "error": None,
        },
        "disk_percent": {
            "value": 47.0,
            "unit": "%",
            "status": "ok",
            "error": None,
        },
        "load_1m": {
            "value": 1.25,
            "unit": "load average",
            "status": "ok",
            "error": None,
        },
        "network_connections": {
            "value": None,
            "unit": "connections",
            "status": "unavailable",
            "error": "Intentionally unavailable in integration test",
        },
    },
    "metric_status": {
        "cpu_percent": "ok",
        "memory_percent": "ok",
        "disk_percent": "ok",
        "load_1m": "ok",
        "network_connections": "unavailable",
    },
}


def main():
    # 1. Verify that the API is running.
    response = session.get(f"{BASE_URL}/health", timeout=5)
    response.raise_for_status()
    assert response.json().get("status") == "ok"
    print("PASS: API health check")

    # 2. Submit a uniquely identified telemetry observation.
    response = session.post(
        f"{BASE_URL}/v1/telemetry",
        json=observation,
        timeout=10,
    )
    response.raise_for_status()
    result = response.json()
    assert result.get("status") == "stored", result
    print("PASS: Authenticated telemetry submission")

    # 3. Retrieve observations for this test host.
    response = session.get(
        f"{BASE_URL}/v1/telemetry/timeseries",
        params={"host": test_host, "limit": 100},
        timeout=10,
    )
    response.raise_for_status()
    records = response.json().get("records")

    assert isinstance(records, list), (
        "Timeseries response did not contain a records list"
    )

    matching = [
        record for record in records
        if record.get("agent_id") == test_agent
        and record.get("host") == test_host
    ]
    assert matching, (
        "Submitted observation was not found in persisted timeseries"
    )

    print("PASS: Observation retrieved from persisted timeseries")

    # 4. Verify that unavailable data was not converted to zero.
    stored = matching[-1]
    stored_metrics = stored.get("metrics", {})
    connection_metric = stored_metrics.get("network_connections")

    assert isinstance(connection_metric, dict), (
        "Stored network_connections metric is missing or malformed"
    )
    assert connection_metric.get("status") == "unavailable"
    assert connection_metric.get("value") is None
    assert connection_metric.get("error"), (
        "Unavailable metric has no explanatory error"
    )

    print("PASS: Unavailable metric remains explicitly unavailable")
    print("PASS: Live telemetry storage verification complete")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, AssertionError, ValueError) as exc:
        sys.exit(f"FAIL: {exc}")
