import copy
import math
import pytest


def test_api_rejects_nonfinite_telemetry_values():
    from Asterion.api import _validate_observation

    payload = {
        "timestamp": "2026-10-27T12:00:00+00:00",
        "agent_id": "test-agent",
        "host": "test-host",
        "collection_status": "ok",
        "metrics": {
            "cpu_percent": {
                "value": math.nan,
                "unit": "%",
                "status": "ok",
                "error": None,
            }
        },
        "metric_status": {
            "cpu_percent": "ok",
        },
    }

    with pytest.raises(ValueError):
        _validate_observation(payload)


def test_api_rejects_unavailable_metric_with_fabricated_zero():
    from Asterion.api import _validate_observation

    payload = {
        "timestamp": "2026-10-27T12:00:00+00:00",
        "agent_id": "test-agent",
        "host": "test-host",
        "collection_status": "partial",
        "metrics": {
            "network_connections": {
                "value": 0,
                "unit": "connections",
                "status": "unavailable",
                "error": "PermissionError",
            }
        },
        "metric_status": {
            "network_connections": "unavailable",
        },
    }

    with pytest.raises(ValueError, match="must have value null"):
        _validate_observation(payload)


def test_api_accepts_unavailable_connection_count_as_null():
    from Asterion.api import _validate_observation

    payload = {
        "timestamp": "2026-10-27T12:00:00+00:00",
        "agent_id": "test-agent",
        "host": "test-host",
        "collection_status": "failed",
        "metrics": {
            "network_connections": {
                "value": None,
                "unit": "connections",
                "status": "unavailable",
                "error": "PermissionError",
            }
        },
        "metric_status": {
            "network_connections": "unavailable",
        },
    }

    validated = _validate_observation(payload)

    assert (
        validated["metrics"]["network_connections"]["value"]
        is None
    )
    assert (
        validated["metrics"]["network_connections"]["status"]
        == "unavailable"
    )