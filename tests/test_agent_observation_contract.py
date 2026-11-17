"""Regression tests for the telemetry agent's structured observation contract."""

from datetime import datetime, timezone

from Asterion.agent import _build_structured_observation


def _observation():
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": "agent-test",
        "host": "host-test",
        "metrics": {
            "cpu_percent": 0.0,
            "memory_percent": 42.5,
            "disk_percent": 0.0,
            "network_connections": None,
            "load_1m": 1.25,
            "disk": {"percent": 0.0, "mount_point": "/", "device": "/dev/test"},
            "load_average": {"1m": 1.25, "5m": 1.0, "15m": 0.8},
            "network_traffic": {
                "bytes_sent": 1000,
                "bytes_recv": 2000,
                "packets_sent": 10,
                "packets_recv": 20,
                "bytes_sent_per_second": None,
                "bytes_recv_per_second": None,
                "packets_sent_per_second": None,
                "packets_recv_per_second": None,
                "rate_error": "No previous sample",
            },
            "connection_metadata": None,
        },
        "metric_status": {
            "cpu_percent": {"status": "ok", "error": None},
            "memory_percent": {"status": "ok", "error": None},
            "disk": {"status": "ok", "error": None},
            "load_average": {"status": "ok", "error": None},
            "network_traffic": {"status": "ok", "error": None},
            "connection_metadata": {
                "status": "unavailable",
                "error": "Permission denied",
            },
        },
    }


def test_builder_preserves_measured_zero_and_unavailable_connections():
    result = _build_structured_observation(_observation(), "fallback", "fallback-host")

    assert result["metrics"]["cpu_percent"]["value"] == 0.0
    assert result["metrics"]["cpu_percent"]["status"] == "ok"
    assert result["metrics"]["disk_percent"]["value"] == 0.0
    assert result["metrics"]["network_connections"]["value"] is None
    assert result["metrics"]["network_connections"]["status"] == "unavailable"
    assert result["collection_status"] == "partial"


def test_builder_emits_authoritative_ten_element_model_features():
    result = _build_structured_observation(_observation(), "fallback", "fallback-host")
    features = result["model_features"]

    assert features["contract_version"] == "asterion_telemetry_features_v1"
    assert len(features["vector"]) == 10
    # CPU and disk are measured zeros, so their availability flags remain 1.
    assert features["vector"][0] == 0.0
    assert features["vector"][2] == 0.0
    assert features["vector"][5] == 1.0
    assert features["vector"][7] == 1.0
    # Connection metadata was unavailable, not measured zero.
    assert features["vector"][8] == 0.0


def test_builder_does_not_mark_failed_metrics_as_measured_zero():
    observation = _observation()
    observation["metrics"]["cpu_percent"] = None
    observation["metric_status"]["cpu_percent"] = {
        "status": "unavailable",
        "error": "CPU counter unavailable",
    }

    result = _build_structured_observation(observation, "fallback", "fallback-host")

    assert result["metrics"]["cpu_percent"]["value"] is None
    assert result["metrics"]["cpu_percent"]["status"] == "unavailable"
    assert result["model_features"]["vector"][5] == 0.0
