from datetime import datetime, timedelta, timezone
import uuid

import pytest
from fastapi.testclient import TestClient
import Asterion.api as api_module
from Asterion.init import TelemetryCollector, validate_observation
from Asterion.api import app
from Asterion.configuration import API_TOKEN
from Asterion.database import (
    get_telemetry_timeseries,
    init_db,)


@pytest.fixture
def client():
    init_db()
    with TestClient(app) as test_client:
        yield test_client


def make_observation(host: str, timestamp: datetime) -> dict:
    """Build a valid telemetry observation for integration testing."""
    return {
        "timestamp": timestamp.isoformat(),
        "agent_id": f"test-agent-{host}",
        "host": host,
        "collection_status": "partial",
        "metrics": {
            "cpu_percent": {
                "value": 27.5,
                "unit": "%",
                "status": "ok",
                "error": None,
            },
            "memory_percent": {
                "value": None,
                "unit": "%",
                "status": "unavailable",
                "error": "PermissionError: simulated collection failure",
            },
        },
        "metric_status": {
            "cpu_percent": "ok",
            "memory_percent": "unavailable",
        },
    }

def auth_headers() -> dict[str, str]:
    """Use the API's configured Bearer-token authentication."""
    if API_TOKEN:
        return {"Authorization": f"Bearer {API_TOKEN}"}
    return {}

def test_collector_includes_required_identity_and_status():
    collector = TelemetryCollector(
        agent_id=f"test-{uuid.uuid4().hex[:8]}",
        host_id="collector-test-host",
    )

    observation = collector.collect()
    validate_observation(observation)

    assert observation["agent_id"].startswith("test-")
    assert observation["host"] == "collector-test-host"
    assert observation["collection_status"] in {
        "ok", "partial", "failed"
    }
    assert observation["metrics"]
    assert observation["metric_status"]

    timestamp = datetime.fromisoformat(
        observation["timestamp"].replace("Z", "+00:00")
    )
    assert timestamp.tzinfo is not None

    for name, record in observation["metrics"].items():
        assert record["unit"]
        assert record["status"] in {"ok", "unavailable"}
        assert observation["metric_status"][name] == record["status"]

        if record["status"] == "unavailable":
            assert record["value"] is None
            assert record["error"]


def test_validator_rejects_invalid_cpu_percentage():
    observation = make_observation(
        "validation-test-host",
        datetime.now(timezone.utc),
    )
    observation["metrics"]["cpu_percent"]["value"] = 101.0

    with pytest.raises(ValueError):
        validate_observation(observation)


def test_validator_preserves_missing_metrics_as_none():
    observation = make_observation(
        "missing-value-host",
        datetime.now(timezone.utc),
    )

    validate_observation(observation)

    assert observation["metrics"]["memory_percent"]["value"] is None
    assert (
        observation["metrics"]["memory_percent"]["status"]
        == "unavailable"
    )


def test_api_persists_time_series_and_separates_hosts(client):
    """
    Exercise HTTP ingestion, persistence, chronological ordering,
    and host filtering through the real application.
    """
    unique = uuid.uuid4().hex[:10]
    host_a = f"e2e-host-a-{unique}"
    host_b = f"e2e-host-b-{unique}"

    base_time = datetime.now(timezone.utc).replace(microsecond=0)
    observations = [
        make_observation(host_a, base_time + timedelta(seconds=10)),
        make_observation(host_a, base_time),
        make_observation(host_b, base_time + timedelta(seconds=5)),
    ]

    for observation in observations:
        response = client.post(
            "/v1/telemetry",
            json=observation,
            headers=auth_headers(),
        )
        assert response.status_code in {200, 201}, response.text

    # Query the persistence layer for host A only.
    records_a = get_telemetry_timeseries(
        host=host_a,
        start_time=(base_time - timedelta(seconds=1)).isoformat(),
        end_time=(base_time + timedelta(seconds=20)).isoformat(),
        limit=100,
    )

    assert len(records_a) == 2
    assert all(record["host"] == host_a for record in records_a)

    returned_times = [
        datetime.fromisoformat(
            record["timestamp"].replace("Z", "+00:00")
        )
        for record in records_a
    ]
    assert returned_times == sorted(returned_times)

    # A second host must not appear in the first host's results.
    assert all(record["host"] != host_b for record in records_a)

    # Confirm the unavailable value survives the HTTP/database round trip.
    earlier_record = records_a[0]
    stored_metrics = earlier_record["metrics"]

    assert stored_metrics["memory_percent"]["value"] is None
    assert stored_metrics["memory_percent"]["status"] == "unavailable"


def test_api_timeseries_filters_by_agent_host_and_both(client):
    """Verify agent, host, and combined filters through the API."""
    unique = uuid.uuid4().hex[:10]
    host_a = f"filter-host-a-{unique}"
    host_b = f"filter-host-b-{unique}"
    agent_a = f"filter-agent-a-{unique}"
    agent_b = f"filter-agent-b-{unique}"

    base_time = datetime.now(timezone.utc)

    # Two agents on host A, and agent A also reporting from host B.
    test_cases = [
        (host_a, agent_a, 0),
        (host_a, agent_b, 1),
        (host_b, agent_a, 2),
    ]

    for host, agent_id, offset in test_cases:
        observation = make_observation(
            host,
            base_time + timedelta(seconds=offset),
        )
        observation["agent_id"] = agent_id

        response = client.post(
            "/v1/telemetry",
            json=observation,
            headers=auth_headers(),
        )

        assert response.status_code in {200, 201}, response.text

    # Agent A should return its observations from both hosts.
    response = client.get(
        "/v1/telemetry/timeseries",
        params={"agent_id": agent_a, "limit": 100},
        headers=auth_headers(),
    )

    assert response.status_code == 200, response.text
    agent_records = response.json()["records"]

    assert len(agent_records) == 2
    assert all(row["agent_id"] == agent_a for row in agent_records)
    assert {row["host"] for row in agent_records} == {host_a, host_b}

    # Host A should return observations from both agents.
    response = client.get(
        "/v1/telemetry/timeseries",
        params={"host": host_a, "limit": 100},
        headers=auth_headers(),
    )

    assert response.status_code == 200, response.text
    host_records = response.json()["records"]

    assert len(host_records) == 2
    assert all(row["host"] == host_a for row in host_records)
    assert {row["agent_id"] for row in host_records} == {
        agent_a,
        agent_b,
    }

    # Combining filters must apply both conditions.
    response = client.get(
        "/v1/telemetry/timeseries",
        params={
            "agent_id": agent_a,
            "host": host_a,
            "limit": 100,
        },
        headers=auth_headers(),
    )

    assert response.status_code == 200, response.text
    combined_records = response.json()["records"]

    assert len(combined_records) == 1
    assert combined_records[0]["agent_id"] == agent_a
    assert combined_records[0]["host"] == host_a

    # An unknown agent must return an empty result, not an error.
    response = client.get(
        "/v1/telemetry/timeseries",
        params={
            "agent_id": f"missing-agent-{unique}",
            "limit": 100,
        },
        headers=auth_headers(),
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"count": 0, "records": []}


@pytest.mark.parametrize("agent_id", ["", "   "])
def test_api_timeseries_rejects_empty_agent_id(client, agent_id):
    """Reject supplied agent identifiers that contain only whitespace."""
    response = client.get(
        "/v1/telemetry/timeseries",
        params={"agent_id": agent_id},
        headers=auth_headers(),
    )

    assert response.status_code == 400, response.text


@pytest.mark.parametrize(
    ("collection_status", "metric_statuses"),
    [
        ("ok", ["ok", "unavailable"]),
        ("failed", ["ok", "ok"]),
    ],
)
def test_validator_rejects_inconsistent_collection_status(
    collection_status,
    metric_statuses,
):
    """Reject overall collection status inconsistent with metric outcomes."""
    observation = make_observation(
        "collection-status-test-host",
        datetime.now(timezone.utc),
    )

    observation["collection_status"] = collection_status

    observation["metrics"]["cpu_percent"]["status"] = metric_statuses[0]
    observation["metrics"]["memory_percent"]["status"] = metric_statuses[1]

    observation["metrics"]["memory_percent"]["value"] = (
        None if metric_statuses[1] == "unavailable" else 18.0
    )
    observation["metrics"]["memory_percent"]["error"] = (
        "Simulated failure"
        if metric_statuses[1] == "unavailable"
        else None
    )

    observation["metric_status"] = {
        name: record["status"]
        for name, record in observation["metrics"].items()
    }

    with pytest.raises(ValueError):
        validate_observation(observation)


def test_validator_rejects_extra_metric_status_key():
    """Require metric_status to describe exactly the collected metrics."""
    observation = make_observation(
        "metric-status-keys-test-host",
        datetime.now(timezone.utc),
    )

    observation["metric_status"]["unexpected_metric"] = "ok"

    with pytest.raises(ValueError):
        validate_observation(observation)

ANALYZE_PAYLOAD = {
    "agent_id": "auth-boundary-test-agent",
    "host": "auth-boundary-test-host",
    "text": "Routine system status review.",
    "metrics": {},
}


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="missing-token"),
        pytest.param(
            {"Authorization": "Bearer definitely-invalid-token"},
            id="invalid-token",
        ),
    ],
)
def test_analyze_rejects_missing_or_invalid_token(client, headers):
    response = client.post(
        "/v1/analyze",
        json=ANALYZE_PAYLOAD,
        headers=headers,
    )

    assert response.status_code == 401, response.text



def test_authenticated_analyze_does_not_grant_action_authorization(
    client, monkeypatch
):
    captured = {}

    def fake_analyze(
        agent_id,
        host,
        text,
        metrics,
        *,
        authorized=False,
        persist_telemetry=True,
    ):
        captured["authorized"] = authorized
        return {
            "safety": {"no_actions_executed": True},
            "executor_result": {
                "executed": False,
                "status": "dry_run_only",
            },
        }

    monkeypatch.setattr(api_module, "analyze", fake_analyze)

    response = client.post(
        "/v1/analyze",
        json=ANALYZE_PAYLOAD,
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )

    assert response.status_code == 200, response.text
    assert captured["authorized"] is False
    assert response.json()["safety"]["no_actions_executed"] is True
    assert response.json()["executor_result"]["executed"] is False
