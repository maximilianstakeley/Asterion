"""Regression tests for safety and missingness contracts.

Adapt imports to the local package layout. These tests intentionally assert
that no executor call occurs after a safety evaluation failure.
"""
from __future__ import annotations
import numpy as np
import pytest
import copy
import math
from Asterion.ReinforcementLearning.state import state_from_metrics


def test_rl_state_has_explicit_missingness_masks():
    state = state_from_metrics({
        "cpu_percent": 40.0,
        "memory_percent": None,
        "disk_percent": 20.0,
        "network_connections": None,
        "load_1m": None,
    })
    assert state.shape == (10,)
    assert np.isfinite(state).all()
    # First five are values; last five are availability flags.
    assert state[5:].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    # Missing load is not inferred from CPU/memory.
    assert state[4] == 0.0
    assert state[9] == 0.0


def test_rl_state_rejects_nonfinite_and_invalid_metrics():
    state = state_from_metrics({
        "cpu_percent": float("nan"),
        "memory_percent": float("inf"),
        "disk_percent": -1,
        "network_connections": 2.5,
        "load_1m": None,
    })
    assert state[5:].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_safety_evaluation_exception_does_not_call_executor(monkeypatch):
    # Replace these imports if your test package uses a different module path.
    from Asterion import service
    from Asterion.contracts import CandidateAction

    observation = service.Observation(
        agent_id="test-agent",
        host="test-host",
        text="test",
        metrics={"cpu_percent": 1.0},
    )
    monkeypatch.setattr(service, "add_telemetry", lambda *a, **k: None)
    monkeypatch.setattr(service._classifier, "predict", lambda text: {
        "available": False, "label": "unknown", "confidence": 0.0
    })
    monkeypatch.setattr(service, "detect_anomalies", lambda metrics: [])
    monkeypatch.setattr(service, "rank_hypotheses", lambda *a: [])
    monkeypatch.setattr(service, "generate_candidates", lambda evidence: [
        CandidateAction(
            action="notify_operator",
            risk=0.05,
            rationale="test-only candidate",
        )
    ])
    monkeypatch.setattr(service, "_check_resource_budget", lambda candidate: True)
    monkeypatch.setattr(service, "_check_preconditions", lambda candidate: True)
    monkeypatch.setattr(
        service._safety_gate,
        "evaluate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("safety failed")),
    )
    monkeypatch.setattr(
        service._executor,
        "execute",
        lambda *a, **k: pytest.fail("executor must not run after safety failure"),
    )
    monkeypatch.setattr(service, "_audit_failure", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="safety failed"):
        service.analyze(
            "test-agent", "test-host", "test", {"cpu_percent": 1.0},
            authorized=True,
        )


def test_executor_execution_claim_is_rejected():
    from Asterion import service

    with pytest.raises(RuntimeError, match="executed=True"):
        service._assert_nonexecuting_result({
            "status": "executed",
            "executed": True,
            "dry_run": False,
            "execution_enabled": True,
        })


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"executed": False},
        {
            "executed": False,
            "dry_run": True,
        },
        {
            "executed": None,
            "dry_run": True,
            "execution_enabled": False,
        },
        {
            "executed": 0,
            "dry_run": True,
            "execution_enabled": False,
        },
        {
            "executed": False,
            "dry_run": False,
            "execution_enabled": False,
        },
        {
            "executed": False,
            "dry_run": 1,
            "execution_enabled": False,
        },
        {
            "executed": False,
            "dry_run": True,
            "execution_enabled": None,
        },
        {
            "executed": False,
            "dry_run": True,
            "execution_enabled": 0,
        },
    ],
)
def test_executor_metadata_must_prove_nonexecution(metadata):
    from Asterion import service

    with pytest.raises(RuntimeError, match="Safety contract violation"):
        service._assert_nonexecuting_result(metadata)


def test_executor_metadata_accepts_explicit_nonexecution():
    from Asterion import service

    metadata = {
        "status": "dry_run_only",
        "executed": False,
        "dry_run": True,
        "execution_enabled": False,
    }

    assert service._assert_nonexecuting_result(metadata) == metadata



def test_collector_marks_all_unavailable_metrics_as_failed(monkeypatch):
    from Asterion.telemetry_collector import TelemetryCollector

    collector = TelemetryCollector(agent_id="test-agent")
    def unavailable(*args, **kwargs):
        raise PermissionError("test unavailable")

    monkeypatch.setattr(collector, "_collect_cpu", unavailable)
    monkeypatch.setattr(collector, "_collect_memory", unavailable)
    monkeypatch.setattr(collector, "_collect_disk", unavailable)
    monkeypatch.setattr(collector, "_collect_load_average", unavailable)
    monkeypatch.setattr(collector, "_collect_network_traffic", unavailable)
    monkeypatch.setattr(collector, "_collect_connections", unavailable)

    observation = collector.collect()
    assert observation["collection_status"] == "failed"
    assert all(
        record["status"] == "unavailable"
        for record in observation["metric_status"].values()
    )

import pytest


def test_safety_exception_fails_closed(monkeypatch):
    from Asterion import service
    from Asterion.contracts import CandidateAction

    audit_events = []
    executor_calls = []

    monkeypatch.setattr(
        service,
        "add_telemetry",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        service._classifier,
        "predict",
        lambda text: {
            "available": False,
            "label": "unknown",
            "confidence": 0.0,
        },
    )

    monkeypatch.setattr(
        service,
        "detect_anomalies",
        lambda metrics: [],
    )

    monkeypatch.setattr(
        service,
        "rank_hypotheses",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        service,
        "generate_candidates",
        lambda evidence: [
            CandidateAction(
                action="notify_operator",
                risk=0.05,
                rationale="Regression test candidate",
            )
        ],
    )

    monkeypatch.setattr(
        service,
        "_check_resource_budget",
        lambda candidate: True,
    )

    monkeypatch.setattr(
        service,
        "_check_preconditions",
        lambda candidate: True,
    )

    def fail_safety(*args, **kwargs):
        raise RuntimeError("deliberate safety-gate failure")

    monkeypatch.setattr(
        service._safety_gate,
        "evaluate",
        fail_safety,
    )

    def forbidden_executor_call(*args, **kwargs):
        executor_calls.append((args, kwargs))
        pytest.fail("Executor must not be called after safety failure")

    monkeypatch.setattr(
        service._executor,
        "execute",
        forbidden_executor_call,
    )

    monkeypatch.setattr(
        service,
        "_audit_failure",
        lambda **kwargs: audit_events.append(kwargs),
    )

    with pytest.raises(
        RuntimeError,
        match="deliberate safety-gate failure",
    ):
        service.analyze(
            agent_id="test-agent",
            host="test-host",
            text="routine test observation",
            metrics={"cpu_percent": 10.0},
            authorized=True,
        )

    assert executor_calls == []
    assert len(audit_events) == 1
    assert audit_events[0]["stage"] == "safety_evaluation"

def test_executor_metadata_cannot_claim_live_execution():
    from Asterion import service

    with pytest.raises(
        RuntimeError,
        match="executed=True",
    ):
        service._assert_nonexecuting_result(
            {
                "status": "executed",
                "executed": True,
                "dry_run": False,
                "execution_enabled": True,
            }
        )

def test_filesystem_identity_does_not_invent_unknown_metadata(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from Asterion.telemetry_collector import TelemetryCollector
    import Asterion.telemetry_collector as telemetry_module

    collector = object.__new__(TelemetryCollector)
    monkeypatch.setattr(telemetry_module.psutil, "disk_partitions", lambda all=True: [])
    identity = collector._filesystem_identity(str(tmp_path))
    assert identity == {
        "mount_point": None,
        "device": None,
        "filesystem_type": None,
    }


def test_filesystem_identity_reports_matched_partition(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from Asterion.telemetry_collector import TelemetryCollector
    import Asterion.telemetry_collector as telemetry_module

    collector = object.__new__(TelemetryCollector)
    partition = SimpleNamespace(
        mountpoint=str(tmp_path),
        device="/dev/test-device",
        fstype="testfs",
    )
    monkeypatch.setattr(
        telemetry_module.psutil,
        "disk_partitions",
        lambda all=True: [partition],
    )
    identity = collector._filesystem_identity(str(tmp_path))
    assert identity == {
        "mount_point": str(tmp_path.resolve()),
        "device": "/dev/test-device",
        "filesystem_type": "testfs",
    }
def test_structured_agent_payload_matches_telemetry_api_contract():
    from Asterion.agent import _build_structured_observation
    from Asterion.api import _validate_observation

    observation = {
        "agent_id": "test-agent",
        "host": "test-host",
        "timestamp": "2026-10-27T12:00:00+00:00",
        "collection_status": "partial",
        "metrics": {
            "cpu_percent": 25.0,
            "memory_percent": 40.0,
            "disk_percent": 55.0,
            "network_connections": None,
            "load_1m": 1.25,
            "disk": {
                "percent": 55.0,
                "total_bytes": 1000,
                "used_bytes": 550,
                "free_bytes": 450,
            },
            "load_average": {
                "1m": 1.25,
                "5m": 1.10,
                "15m": 0.95,
            },
            "network_traffic": {
                "bytes_sent": 100,
                "bytes_recv": 200,
                "packets_sent": 3,
                "packets_recv": 4,
                "bytes_sent_per_second": None,
                "bytes_recv_per_second": None,
                "packets_sent_per_second": None,
                "packets_recv_per_second": None,
                "rate_error": None,
            },
            "connection_metadata": None,
        },
        "metric_status": {
            "cpu_percent": {
                "status": "ok",
                "error": None,
                "unit": "%",
            },
            "memory_percent": {
                "status": "ok",
                "error": None,
                "unit": "%",
            },
            "disk": {
                "status": "ok",
                "error": None,
                "unit": "bytes; %",
            },
            "load_average": {
                "status": "ok",
                "error": None,
                "unit": "load average",
            },
            "network_traffic": {
                "status": "ok",
                "error": None,
                "unit": "bytes; packets",
            },
            "connection_metadata": {
                "status": "unavailable",
                "error": "PermissionError",
                "unit": "connections",
            },
        },
    }

    payload = _build_structured_observation(
        observation,
        "test-agent",
        "test-host",
    )

    validated = _validate_observation(payload)

    assert validated["collection_status"] == "partial"
    assert validated["metrics"]["cpu_percent"]["value"] == 25.0

    connection_metric = validated["metrics"]["network_connections"]
    assert connection_metric["status"] == "unavailable"
    assert connection_metric["value"] is None
    assert connection_metric["error"]

    # The first network sample cannot establish a rate.
    rate_metric = validated["metrics"]["network_bytes_sent_per_second"]
    assert rate_metric["status"] == "unavailable"
    assert rate_metric["value"] is None

    # Detailed collector records survive separately from scalar API metrics.
    assert validated["metric_details"]["network_traffic"]["bytes_sent"] == 100

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