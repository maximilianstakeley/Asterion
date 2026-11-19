
from Asterion.agent import _build_structured_observation


def test_first_network_sample_produces_partial_status():
    observation = {
        "timestamp": "2026-11-14T14:00:00+00:00",
        "agent_id": "test-agent",
        "host": "test-host",
        "collection_status": "ok",

        "metrics": {
            "cpu_percent": 20.0,
            "memory_percent": 40.0,
            "disk_percent": 30.0,
            "network_connections": 10,
            "load_1m": 1.0,

            "network_traffic": {
                "bytes_sent": 1000,
                "bytes_recv": 2000,
                "packets_sent": 10,
                "packets_recv": 20,
                "bytes_sent_per_second": None,
                "bytes_recv_per_second": None,
                "packets_sent_per_second": None,
                "packets_recv_per_second": None,
            },
        },

        "metric_status": {
            "cpu_percent": {"status": "ok", "error": None},
            "memory_percent": {"status": "ok", "error": None},
            "disk": {"status": "ok", "error": None},
            "connection_metadata": {
                "status": "ok", "error": None
            },
            "load_average": {"status": "ok", "error": None},
            "network_traffic": {
                "status": "ok", "error": None
            },
        },
    }

    result = _build_structured_observation(
        observation,
        "test-agent",
        "test-host",
    )

    assert result["collection_status"] == "partial"

    assert (
        result["metrics"]["network_bytes_sent_per_second"]["status"]
        == "unavailable"
    )

    assert (
        result["metrics"]["cpu_percent"]["status"]
        == "ok"
    )
