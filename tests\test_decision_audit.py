import uuid
from datetime import datetime, timezone

# Adjust this import to your actual persistence module.
from Asterion import database as db


def test_decision_audit_round_trip():
    db.init_db()

    decision_id = f"audit-test-{uuid.uuid4()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "policy_name": "deterministic_baseline",
        "selected_action": "collect_diagnostics",
        "confidence": 0.95,
        "rationale": "Isolated persistence test",
        "safety_verdict": {
            "allowed": True,
            "executed": False,
            "verdict": "allowed_but_execution_disabled",
        },
        "executor_result": {
            "status": "dry_run_only",
            "executed": False,
        },
    }

    returned_id = db.add_decision_audit(
        decision_id=decision_id,
        timestamp=timestamp,
        agent_id="audit-test-agent",
        host="audit-test-host",
        record=record,
    )

    assert returned_id == decision_id

    results = db.get_decision_audit(decision_id=decision_id)

    assert len(results) == 1
    assert results[0]["decision_id"] == decision_id
    assert results[0]["agent_id"] == "audit-test-agent"
    assert results[0]["record"] == record
    assert results[0]["record"]["safety_verdict"]["executed"] is False
    assert results[0]["record"]["executor_result"]["executed"] is False