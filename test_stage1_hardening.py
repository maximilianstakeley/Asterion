"""Regression tests for safety and missingness contracts.

Adapt imports to the local package layout. These tests intentionally assert
that no executor call occurs after a safety evaluation failure.
"""
from __future__ import annotations

import numpy as np
import pytest

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
