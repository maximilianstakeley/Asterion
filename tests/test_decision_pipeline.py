
import sqlite3

import pytest

from Asterion.decision_pipeline import (
    ActionCandidate,
    AuditLedger,
    diagnose,
    execute_action,
    generate_candidates,
    safety_gate,
    select_action,
    validate_feedback,
)


def sample_findings():
    return [
        {
            "category": "resource_threshold",
            "severity": "high",
            "host": "test-host",
            "timestamp": "2026-01-01T00:00:00Z",
            "evidence": {"metric": "cpu_percent", "observed_value": 97.0},
        },
        {
            "category": "abrupt_metric_change",
            "severity": "medium",
            "host": "test-host",
            "timestamp": "2026-01-01T00:01:00Z",
            "evidence": {
                "metric": "cpu_percent",
                "previous_value": 25.0,
                "current_value": 97.0,
            },
        },
    ]


def test_diagnosis_preserves_supporting_evidence():
    result = diagnose(sample_findings())

    assert result["input_finding_count"] == 2
    assert result["hypotheses"]
    assert all(h["supporting_evidence"] for h in result["hypotheses"])
    assert all("uncertainty" in h for h in result["hypotheses"])


def test_diagnosis_does_not_invent_unknown_causes():
    result = diagnose([{
        "category": "unknown_detector_category",
        "host": "test-host",
        "evidence": {},
    }])

    assert result["hypotheses"] == []


def test_candidate_selection_requires_host_authorization():
    hypotheses = diagnose(sample_findings())
    candidates = generate_candidates(hypotheses)

    denied = select_action(candidates, authorized_hosts=set())
    assert denied["selected_action"] == "no_op"

    allowed = select_action(
        candidates,
        authorized_hosts={"test-host"},
        max_risk=0.25,
    )
    assert allowed["selected_action"] in {
        "collect_diagnostics",
        "increase_observation_frequency",
        "scale_resource_simulated",
        "no_op",
    }
    assert allowed["estimated_risk"] <= 0.25


def test_unknown_actions_are_rejected():
    candidate = ActionCandidate(
        action_id="run_arbitrary_shell_command",
        target_host="test-host",
        rationale="not allowed",
        expected_benefit=1.0,
        estimated_risk=0.0,
        reversible=True,
    )

    with pytest.raises(ValueError):
        candidate.validate()


def test_live_execution_is_blocked():
    decision = {
        "selected_action": "collect_diagnostics",
        "target_host": "test-host",
        "estimated_risk": 0.05,
    }
    gate = safety_gate(
        decision,
        authorized_hosts={"test-host"},
        live_execution_enabled=False,
    )

    result = execute_action(decision, gate, simulation=False)

    assert result["status"] == "blocked"
    assert result["reason"] == "live_execution_disabled"


def test_simulation_requires_explicit_simulator():
    decision = {
        "selected_action": "scale_resource_simulated",
        "target_host": "test-host",
        "estimated_risk": 0.2,
    }
    gate = safety_gate(
        decision,
        authorized_hosts={"test-host"},
        max_risk=0.25,
        simulation_mode=True,
    )

    assert gate["approved"] is True

    result = execute_action(decision, gate, simulation=True)
    assert result["status"] == "blocked"
    assert result["reason"] == "simulation_adapter_required"


def test_simulator_result_is_marked_as_simulated():
    decision = {
        "selected_action": "scale_resource_simulated",
        "target_host": "test-host",
        "estimated_risk": 0.2,
    }
    gate = safety_gate(
        decision,
        authorized_hosts={"test-host"},
        max_risk=0.25,
        simulation_mode=True,
    )

    result = execute_action(
        decision,
        gate,
        simulation=True,
        simulator=lambda action: {
            "status": "success",
            "simulated_action": action["selected_action"],
            "observed_effect": "synthetic test result",
        },
    )

    assert result["status"] == "simulated"
    assert result["action_execution"] == "simulated_only"


def test_audit_ledger_verifies_records(tmp_path):
    ledger = AuditLedger(tmp_path / "audit.sqlite3")

    first = ledger.append("decision", {"action": "no_op"})
    second = ledger.append("feedback", {"label": "neutral"})

    assert first
    assert second
    assert ledger.verify() == {
        "valid": True,
        "checked_records": 2,
    }


def test_audit_ledger_detects_record_tampering(tmp_path):
    database = tmp_path / "audit.sqlite3"
    ledger = AuditLedger(database)
    ledger.append("decision", {"action": "no_op"})

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE audit_records SET payload_json = ?",
            ('{"action":"different"}',),
        )

    result = ledger.verify()
    assert result["valid"] is False
    assert result["reason"] == "record_hash_mismatch"


def test_unverified_feedback_is_not_eligible_for_training():
    result = validate_feedback({
        "decision_id": "decision-001",
        "outcome_label": "beneficial",
        "evidence_refs": ["telemetry-001"],
        "verified": False,
    })

    assert result["accepted_for_training"] is False
    assert result["reason"] == "outcome_not_verified"


def test_verified_labeled_feedback_can_enter_review_queue():
    result = validate_feedback({
        "decision_id": "decision-002",
        "outcome_label": "beneficial",
        "evidence_refs": ["telemetry-010", "finding-004"],
        "verified": True,
    })

    assert result["accepted_for_training"] is True
    assert result["training_data_status"] == "eligible_pending_dataset_review"


def test_feedback_rejects_missing_evidence():
    result = validate_feedback({
        "decision_id": "decision-003",
        "outcome_label": "harmful",
        "evidence_refs": [],
        "verified": True,
    })

    assert result["accepted_for_training"] is False
    assert result["reason"] == "evidence_refs_required"
