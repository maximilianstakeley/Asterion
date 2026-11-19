"""Integration tests for RL interface and simulation-only transactional execution."""
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from Asterion.contracts import CandidateAction, PolicyOutput
from Asterion.safety import SafetyGate
from Asterion.simulated_exe import (
    AuthorizationAuthority, SimulatedHostAdapter, SimulatedTransactionalExecutor,
)
from Asterion.ReinforcementLearning.state import state_from_metrics
try:
    from Asterion.ReinforcementLearning.decision_env import DecisionOptimizationEnv, ACTION_NAMES
    GYM_AVAILABLE = True
except ModuleNotFoundError:
    DecisionOptimizationEnv = None
    ACTION_NAMES = ["notify_operator", "collect_diagnostics", "reduce_noncritical_workload", "review_suspicious_activity"]
    GYM_AVAILABLE = False
from Asterion import database as db, service

class FakeClassifier:
    def predict(self, text):
        return {"available": False, "label": "unknown", "confidence": 0.0}

class FakeRLPolicy:
    def select(self, candidates, metrics):
        selected = next((c for c in candidates if c.action == "collect_diagnostics"), candidates[0])
        return PolicyOutput(selected_action=selected.action, confidence=.55,
            rationale="fake RL policy selected an available candidate for integration test",
            policy_name="ppo_simulation_policy")

class RLAndExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db, "DB_PATH", Path(self.tmp.name) / "test.db")
        self.db_patch.start(); db.init_db()
        self.classifier_patch = patch.object(service, "_classifier", FakeClassifier())
        self.classifier_patch.start()
        self.metrics = {"cpu_percent": 97.0, "memory_percent": 90.0, "disk_percent": 30.0,
                        "network_connections": 25, "load_1m": 1.0}

    def tearDown(self):
        self.classifier_patch.stop(); self.db_patch.stop(); self.tmp.cleanup()
        service._rl_policy = None; service._rl_policy_error = None


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


    def test_rl_policy_is_routed_through_shared_decision_pipeline(self):
        self._enable_test_only_safety_checks()
        service._rl_policy = FakeRLPolicy()
        service._rl_policy_error = None
        with patch.dict(os.environ, {"HUNTERSEEKER_POLICY": "ppo"}):
            result = service.analyze("agent-test", "sim-host", "CPU elevated", self.metrics, authorized=True)
        decision = result["decision"]
        self.assertEqual(decision["policy_output"]["policy_name"], "ppo_simulation_policy")
        self.assertEqual(decision["policy_output"]["selected_action"], "collect_diagnostics")
        self.assertTrue(decision["model_outputs"]["policy_runtime"]["rl_loaded"])
        self.assertEqual(decision["executor_result"]["status"], "dry_run_only")
        self.assertFalse(decision["safety_verdict"]["executed"])

    def test_rl_state_mapping_is_bounded(self):
        state = state_from_metrics({"cpu_percent": 100, "memory_percent": 100,
            "disk_percent": 100, "network_connections": 10000000, "load_1m": 999})
        self.assertEqual(state.shape, (10,))
        self.assertTrue((state >= 0).all() and (state <= 1).all())

    @unittest.skipUnless(GYM_AVAILABLE, "gymnasium is installed by requirements.txt; unavailable in this test runner")
    def test_synthetic_environment_matches_shared_action_vocabulary(self):
        env = DecisionOptimizationEnv(horizon=3)
        obs, _ = env.reset(seed=11)
        self.assertEqual(obs.shape, (10,))
        self.assertEqual(env.action_space.n, len(ACTION_NAMES))
        for action in range(env.action_space.n):
            env.reset(seed=11)
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(-1 <= reward <= 1)
            self.assertEqual(info["action_name"], ACTION_NAMES[action])

    def _setup_simulator(self, fault_hook=None):
        self.authority = AuthorizationAuthority(b"test-secret-material-at-least-32-bytes")
        self.adapter = SimulatedHostAdapter("sim-host-1", .70)
        self.executor = SimulatedTransactionalExecutor(self.authority, self.adapter, fault_hook)
        self.candidate = CandidateAction(action="reduce_noncritical_workload", risk=.18,
                                         rationale="simulation test only")
        self.verdict = SafetyGate().evaluate(
            self.candidate,
            authorized=True,
            resource_budget_ok=True,
            preconditions_ok=True,)
        self.grant = self.authority.issue(target="sim-host-1", action=self.candidate.action)

    def test_simulated_executor_requires_scoped_authorization(self):
        self._setup_simulator()
        before = self.adapter.snapshot()
        result = self.executor.execute(self.candidate, self.verdict, None)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.adapter.snapshot(), before)

    def test_simulated_executor_applies_only_bounded_in_memory_change(self):
        self._setup_simulator()
        before = self.adapter.snapshot()
        result = self.executor.execute(self.candidate, self.verdict, self.grant)
        self.assertEqual(result["status"], "simulated_success")
        self.assertFalse(result["live_system_changes"])
        self.assertLess(result["after"]["workload_level"], before["workload_level"])
        self.assertEqual(result["after"]["revision"], before["revision"] + 1)

    def test_fault_after_apply_triggers_verified_rollback(self):
        def fail(stage):
            if stage == "after_apply":
                raise RuntimeError("injected fault")
        self._setup_simulator(fail)
        before = self.adapter.snapshot()
        result = self.executor.execute(self.candidate, self.verdict, self.grant)
        self.assertEqual(result["status"], "rolled_back")
        self.assertTrue(result["rollback_verified"])
        self.assertEqual(self.adapter.snapshot(), before)

    def test_fault_before_apply_leaves_state_unchanged(self):
        def fail(stage):
            if stage == "before_apply":
                raise RuntimeError("injected pre-apply fault")
        self._setup_simulator(fail)
        before = self.adapter.snapshot()
        result = self.executor.execute(self.candidate, self.verdict, self.grant)
        self.assertEqual(result["status"], "rolled_back")
        self.assertTrue(result["rollback_verified"])
        self.assertEqual(self.adapter.snapshot(), before)

    def test_fault_after_verification_still_rolls_back(self):
        def fail(stage):
            if stage == "after_verify":
                raise RuntimeError("injected post-verification fault")
        self._setup_simulator(fail)
        before = self.adapter.snapshot()
        result = self.executor.execute(self.candidate, self.verdict, self.grant)
        self.assertEqual(result["status"], "rolled_back")
        self.assertTrue(result["rollback_verified"])
        self.assertEqual(self.adapter.snapshot(), before)

    def test_grant_scope_tampering_is_rejected(self):
        self._setup_simulator()
        from dataclasses import replace
        forged = replace(self.grant, target="different-host")
        before = self.adapter.snapshot()
        result = self.executor.execute(self.candidate, self.verdict, forged)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.adapter.snapshot(), before)

    def test_authorization_grant_is_single_use(self):
        self._setup_simulator()
        first = self.executor.execute(self.candidate, self.verdict, self.grant)
        self.assertEqual(first["status"], "simulated_success")
        second = self.executor.execute(self.candidate, self.verdict, self.grant)
        self.assertEqual(second["status"], "blocked")
        self.assertEqual(second["reason"], "authorization_grant_replayed")

    def test_simulation_executor_rechecks_risk_limit(self):
        self._setup_simulator()
        risky = CandidateAction(action="reduce_noncritical_workload", risk=.21, rationale="risk-bound test")
        grant = self.authority.issue(target="sim-host-1", action=risky.action)
        before = self.adapter.snapshot()
        risky_verdict = SafetyGate().evaluate(
            risky,
            authorized=True,
            resource_budget_ok=True,
            preconditions_ok=True,)

        result = self.executor.execute(risky, risky_verdict, grant)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "simulator_risk_limit_exceeded")
        self.assertEqual(self.adapter.snapshot(), before)

if __name__ == "__main__":
    unittest.main()
