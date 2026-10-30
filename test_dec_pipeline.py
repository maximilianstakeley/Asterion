"""Integration tests for the connected observation-to-audit pipeline."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from Asterion.contracts import Observation, CandidateAction
from Asterion.safety import SafetyGate
from Asterion.executor import ActionExecutor
from Asterion import database as db
from Asterion import service

class FakeClassifier:
    def predict(self, text):
        return {"available": False, "label": "unknown", "confidence": 0.0,
                "reason": "test classifier deliberately unavailable"}

class DecisionPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db, "DB_PATH", Path(self.tempdir.name) / "test.db")
        self.db_patch.start()
        db.init_db()
        self.classifier_patch = patch.object(service, "_classifier", FakeClassifier())
        self.classifier_patch.start()
        self.metrics = {"cpu_percent": 97.0, "memory_percent": 30.0,
                        "disk_percent": 40.0, "network_connections": 25, "load_1m": 1.0}

    def tearDown(self):
        self.classifier_patch.stop()
        self.db_patch.stop()
        self.tempdir.cleanup()


    def _enable_test_only_safety_checks(self):
        """Allow integration tests to exercise the dry-run path only."""
        self.budget_patch = patch.object(
            service,
            "_check_resource_budget",
            return_value=True,
        )
        self.preconditions_patch = patch.object(
            service,
            "_check_preconditions",
            return_value=True,
        )
        self.budget_patch.start()
        self.preconditions_patch.start()
        self.addCleanup(self.budget_patch.stop)
        self.addCleanup(self.preconditions_patch.stop)


    def test_contract_rejects_invalid_metrics(self):
        bad = dict(self.metrics, cpu_percent=101)
        with self.assertRaises(ValidationError):
            Observation(agent_id="agent-1", host="host-1", text="normal telemetry", metrics=bad)

    def test_full_pipeline_records_evidence_policy_safety_and_executor(self):
        self._enable_test_only_safety_checks()
        result = service.analyze("agent-1", "host-1", "CPU load elevated", self.metrics, authorized=True)
        decision = result["decision"]
        self.assertTrue(decision["decision_id"])
        self.assertTrue(any(e["category"] == "high_cpu" for e in decision["evidence"]))
        self.assertGreaterEqual(len(decision["candidate_actions"]), 1)
        self.assertTrue(decision["policy_output"]["selected_action"])
        self.assertFalse(decision["safety_verdict"]["executed"])
        self.assertEqual(decision["executor_result"]["status"], "dry_run_only")
        self.assertTrue(result["safety"]["no_actions_executed"])
        audit = db.get_decision_audit(10)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["decision_id"], decision["decision_id"])
        self.assertIn("model_outputs", audit[0])
        self.assertIn("candidate_actions", audit[0])

    def test_safety_blocks_without_explicit_authorization(self):
        action = CandidateAction(action="notify_operator", risk=0.02, rationale="test")
        verdict = SafetyGate().evaluate(action, authorized=False)
        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.verdict, "blocked")
        result = ActionExecutor().execute(action, verdict)
        self.assertEqual(result["status"], "not_executed")

    def test_safety_never_enables_live_execution_in_this_release(self):
        action = CandidateAction(action="notify_operator", risk=0.02, rationale="test")
        # Even a caller asking for execution-enabled mode fails closed.
        gate = SafetyGate(execution_enabled=True)
        verdict = gate.evaluate(
            action,
            authorized=True,
            resource_budget_ok=True,
            preconditions_ok=True,)
        result = ActionExecutor().execute(action, verdict)
        self.assertTrue(verdict.allowed)
        self.assertFalse(verdict.executed)
        self.assertEqual(result["status"], "dry_run_only")

    def test_invalid_resource_budget_blocks_action(self):
        action = CandidateAction(action="collect_diagnostics", risk=0.05, rationale="test")
        verdict = SafetyGate().evaluate(action, authorized=True, resource_budget_ok=False)
        self.assertFalse(verdict.allowed)
        self.assertIn(
            "Resource budget constraint not verified",
            verdict.reasons,)

if __name__ == "__main__":
    unittest.main()
