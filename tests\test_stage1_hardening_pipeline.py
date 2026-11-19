
"""Integration tests for structured network telemetry."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import psutil

from Asterion.telemetry_collector import (
    TelemetryCollector,
    propose_network_optimizations,
)


def make_counters(
    *,
    bytes_sent: int,
    bytes_recv: int,
    packets_sent: int,
    packets_recv: int,
):
    """Create a public-interface-compatible network counter fixture."""
    return SimpleNamespace(
        bytes_sent=bytes_sent,
        bytes_recv=bytes_recv,
        packets_sent=packets_sent,
        packets_recv=packets_recv,
        errin=0,
        errout=0,
        dropin=0,
        dropout=0,
    )


def test_first_network_sample_does_not_invent_rates():
    collector = TelemetryCollector("test-agent", cpu_sample_seconds=0)

    with (
        patch(
            "Asterion.telemetry_collector.psutil.net_io_counters",
            return_value=make_counters(
                bytes_sent=100,
                bytes_recv=200,
                packets_sent=10,
                packets_recv=20,
            ),
        ),
        patch(
            "Asterion.telemetry_collector.time.monotonic",
            return_value=10.0,
        ),
    ):
        result = collector._collect_network_traffic()

    assert result["bytes_sent"] == 100
    assert result["bytes_recv"] == 200
    assert result["bytes_sent_per_second"] is None
    assert result["bytes_recv_per_second"] is None
    assert result["rate_status"] == "unavailable"


def test_second_network_sample_calculates_rates():
    collector = TelemetryCollector("test-agent", cpu_sample_seconds=0)
    collector._previous_network = {
        "sample_monotonic": 10.0,
        "counters": {
            "bytes_sent": 100,
            "bytes_recv": 200,
            "packets_sent": 10,
            "packets_recv": 20,
        },
    }

    with (
        patch(
            "Asterion.telemetry_collector.psutil.net_io_counters",
            return_value=make_counters(
                bytes_sent=160,
                bytes_recv=400,
                packets_sent=16,
                packets_recv=40,
            ),
        ),
        patch(
            "Asterion.telemetry_collector.time.monotonic",
            return_value=12.0,
        ),
    ):
        result = collector._collect_network_traffic()

    assert result["bytes_sent_per_second"] == 30
    assert result["bytes_recv_per_second"] == 100
    assert result["packets_sent_per_second"] == 3
    assert result["packets_recv_per_second"] == 10
    assert result["rate_status"] == "ok"


def test_counter_reset_invalidates_rates():
    collector = TelemetryCollector("test-agent", cpu_sample_seconds=0)
    collector._previous_network = {
        "sample_monotonic": 10.0,
        "counters": {
            "bytes_sent": 100,
            "bytes_recv": 200,
            "packets_sent": 10,
            "packets_recv": 20,
        },
    }

    with (
        patch(
            "Asterion.telemetry_collector.psutil.net_io_counters",
            return_value=make_counters(
                bytes_sent=50,
                bytes_recv=250,
                packets_sent=12,
                packets_recv=25,
            ),
        ),
        patch(
            "Asterion.telemetry_collector.time.monotonic",
            return_value=12.0,
        ),
    ):
        result = collector._collect_network_traffic()

    assert result["rates"] is None
    assert result["rate_status"] == "unavailable"
    assert "decreased" in result["rate_error"]


def test_connection_permission_error_stays_unavailable():
    collector = TelemetryCollector("test-agent", cpu_sample_seconds=0)

    with patch(
        "Asterion.telemetry_collector.psutil.net_connections",
        side_effect=psutil.AccessDenied(pid=None),
    ):
        result = collector.collect()

    assert result["metrics"]["network_connections"] is None
    assert result["metric_status"]["connection_metadata"]["status"] == (
        "unavailable"
    )


def test_network_recommendations_require_measured_rate():
    observation = {
        "metrics": {
            "network_traffic": {
                "bytes_recv_per_second": 12_000_000,
            }
        }
    }

    proposals = propose_network_optimizations(
        observation,
        receive_bytes_per_second_threshold=10_000_000,
    )

    assert len(proposals) == 1
    assert proposals[0]["category"] == "network_throughput"
    assert proposals[0]["execution"] == "disabled"


def test_network_recommendations_ignore_unavailable_rates():
    observation = {
        "metrics": {
            "network_traffic": {
                "bytes_recv_per_second": None,
            }
        }
    }

    assert propose_network_optimizations(observation) == []


def test_network_recommendations_do_not_use_cumulative_totals_as_deltas():
    observation = {
        "metrics": {
            "network_traffic": {
                "bytes_recv_per_second": 0,
                "bytes_recv": 10_000_000,
            },
            "network_interfaces": {
                "interfaces": {
                    "en0": {
                        "bytes_recv": 10_000_000,
                    }
                }
            },
        }
    }

    assert propose_network_optimizations(observation) == []
