'python'
"""Failure-audit and recovery tests for the Asterion analysis pipeline."""

import pytest

from Asterion import database as db
from Asterion import service


class UnavailableClassifier:
    """Keep text classification deterministic for these tests."""

    def predict(self, text):
        return {
            "available": False,
            "label": "unknown",
            "confidence": 0.0,
            "reason": "Classifier unavailable in isolated test",
        }


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    """Give each test an isolated database."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "failure_audit_test.db")
    db.init_db()
    monkeypatch.setattr(service, "_classifier", UnavailableClassifier())
    return db


@pytest.fixture
def test_metrics():
    return {
        "cpu_percent": 97.0,
        "memory_percent": 30.0,
        "disk_percent": 40.0,
        "network_connections": None,
        "load_1m": 1.0,
    }


def test_pipeline_failure_is_persisted_with_stage_and_error(
    isolated_database,
    test_metrics,
    monkeypatch,
):
    def fail_policy_selection(*args, **kwargs):
        raise RuntimeError("injected policy-selection failure")

    monkeypatch.setattr(
        service,
        "_select_policy",
        fail_policy_selection,
    )

    with pytest.raises(
        RuntimeError,
        match="injected policy-selection failure",
    ):
        service.analyze(
            "failure-test-agent",
            "failure-test-host",
            "CPU utilization is elevated",
            test_metrics,
        )

    records = isolated_database.get_decision_audit(limit=20)

    failures = [
        item for item in records
        if item["record"].get("status") == "failed"
    ]

    assert len(failures) == 1

    failure = failures[0]
    record = failure["record"]

    assert record["failure_stage"] == "policy_selection"
    assert "injected policy-selection failure" in record["error"]
    assert record["host"] == "failure-test-host"
    assert record["agent_id"] == "failure-test-agent"
    assert record["safety"]["execution_permitted"] is False
    assert record["safety"]["failure_audit_only"] is True


def test_failure_audit_write_error_does_not_mask_original_exception(
    isolated_database,
    test_metrics,
    monkeypatch,
):
    def fail_policy_selection(*args, **kwargs):
        raise RuntimeError("original analysis failure")

    def fail_audit_write(*args, **kwargs):
        raise OSError("injected audit storage failure")

    monkeypatch.setattr(
        service,
        "_select_policy",
        fail_policy_selection,
    )
    monkeypatch.setattr(
        service,
        "add_decision_audit",
        fail_audit_write,
    )

    with pytest.raises(RuntimeError, match="original analysis failure"):
        service.analyze(
            "failure-test-agent",
            "failure-test-host",
            "CPU utilization is elevated",
            test_metrics,
        )


def test_valid_analysis_succeeds_after_recoverable_pipeline_failure(
    isolated_database,
    test_metrics,
    monkeypatch,
):
    def fail_policy_selection(*args, **kwargs):
        raise RuntimeError("temporary policy failure")

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(
            service,
            "_select_policy",
            fail_policy_selection,
        )

        with pytest.raises(RuntimeError, match="temporary policy failure"):
            service.analyze(
                "recovery-test-agent",
                "recovery-test-host",
                "CPU utilization is elevated",
                test_metrics,
            )

    # The temporary policy failure is no longer active.
    result = service.analyze(
        "recovery-test-agent",
        "recovery-test-host",
        "CPU utilization is elevated",
        test_metrics,
    )

    assert result["decision"]["decision_id"]
    assert result["safety"]["no_actions_executed"] is True
    assert result["decision"]["executor_result"]["executed"] is False

    records = isolated_database.get_decision_audit(limit=20)
    host_records = [
        item for item in records
        if item["host"] == "recovery-test-host"
    ]

    assert any(
        item["record"].get("status") == "failed"
        for item in host_records
    )
    assert any(
        item["record"].get("decision_id") == result["decision"]["decision_id"]
        for item in host_records
    )


def test_late_pipeline_failure_leaves_traceable_partial_records(
    isolated_database,
    test_metrics,
    monkeypatch,
):
    def fail_policy_selection(*args, **kwargs):
        raise RuntimeError("injected late-stage failure")

    monkeypatch.setattr(
        service,
        "_select_policy",
        fail_policy_selection,
    )

    with pytest.raises(RuntimeError, match="injected late-stage failure"):
        service.analyze(
            "partial-test-agent",
            "partial-test-host",
            "CPU utilization is elevated",
            test_metrics,
        )

    # Verify telemetry persisted before the injected failure.
    with isolated_database.connect() as connection:
        telemetry_rows = connection.execute(
            "SELECT COUNT(*) FROM telemetry WHERE agent_id = ? AND host = ?",
            ("partial-test-agent", "partial-test-host"),
        ).fetchone()[0]

    assert telemetry_rows >= 1

    # Verify the failure is also represented in the audit trail.
    records = isolated_database.get_decision_audit(limit=50)
    failures = [
        item for item in records
        if item["host"] == "partial-test-host"
        and item["record"].get("status") == "failed"
    ]

    assert len(failures) == 1
    assert failures[0]["record"]["failure_stage"] == "policy_selection"
    assert failures[0]["record"]["safety"]["execution_permitted"] is False
