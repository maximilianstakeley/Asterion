
"""Validated observation-to-decision pipeline with fail-closed safety and audit."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from .contracts import Observation, Evidence, DecisionRecord
from .database import (
    add_telemetry,
    add_finding,
    add_proposal,
    add_decision_audit,
)
from .detection import TextClassifier, detect_anomalies
from .rootcause import rank_hypotheses
from .policy import generate_candidates, select_action
from .configuration import MODEL_DIR
from .safety import SafetyGate
from .executor import ActionExecutor


logger = logging.getLogger(__name__)

_classifier = TextClassifier()
_safety_gate = SafetyGate()
_executor = ActionExecutor()

_rl_policy = None
_rl_policy_error: str | None = None


def _utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _serialize(value: Any) -> Any:
    """Convert supported models and containers into JSON-compatible values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def _audit_failure(
    *,
    observation: Observation,
    stage: str,
    error: Exception | str,
) -> None:
    """Persist a failure audit when possible; never conceal the original error."""
    timestamp = _utc_timestamp()
    decision_id = str(uuid4())

    error_text = (
        f"{type(error).__name__}: {error}"
        if isinstance(error, Exception)
        else str(error)
    )

    failure_record = {
        "decision_id": decision_id,
        "timestamp": timestamp,
        "agent_id": observation.agent_id,
        "host": observation.host,
        "status": "failed",
        "failure_stage": stage,
        "error": error_text,
        "observation": _serialize(observation),
        "safety": {
            "execution_permitted": False,
            "failure_audit_only": True,
        },
    }

    try:
        add_decision_audit(
            decision_id,
            timestamp,
            observation.agent_id,
            observation.host,
            failure_record,
        )
    except Exception:
        logger.exception(
            "Could not persist failure audit for decision %s",
            decision_id,
        )


def reload_classifier() -> None:
    """Reload the text classifier."""
    global _classifier
    _classifier = TextClassifier()


def reload_rl_policy() -> dict[str, Any]:
    """Clear cached PPO state so the next selection attempts to load it again."""
    global _rl_policy, _rl_policy_error

    _rl_policy = None
    _rl_policy_error = None

    return {
        "status": "reset",
        "message": "PPO will be loaded on the next policy-selection request.",
    }


def _get_rl_model_path() -> Path:
    """Resolve the configured PPO model path."""
    configured_path = (
        os.getenv("ASTERION_RL_MODEL")
        or os.getenv("HUNTERSEEKER_RL_MODEL")
    )

    if configured_path:
        return Path(configured_path).expanduser()

    return MODEL_DIR / "rl" / "decision_policy_ppo.zip"


def _policy_mode() -> str:
    """Read the policy mode, preferring the current Asterion variable."""
    return (
        os.getenv("ASTERION_POLICY")
        or os.getenv("HUNTERSEEKER_POLICY")
        or "baseline"
    ).strip().lower()


def _validate_policy_selection(candidates: list[Any], policy_output: Any) -> Any:
    """Return the exact matching candidate or reject the policy output."""
    if not candidates:
        raise ValueError("No candidate actions were generated.")

    selected_action = getattr(policy_output, "selected_action", None)

    if selected_action is None:
        raise ValueError("Policy output has no selected_action.")

    matching_candidates = [
        candidate
        for candidate in candidates
        if getattr(candidate, "action", None) == selected_action
    ]

    if len(matching_candidates) != 1:
        raise ValueError(
            "Policy selection must match exactly one generated candidate; "
            f"selected_action={selected_action!r}, "
            f"matches={len(matching_candidates)}."
        )

    return matching_candidates[0]


def _select_policy(
    candidates: list[Any],
    metrics: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Select a candidate using PPO when configured, otherwise the baseline."""
    global _rl_policy, _rl_policy_error

    if not candidates:
        raise ValueError("Cannot select a policy action from an empty candidate list.")

    mode = _policy_mode()

    if mode != "ppo":
        baseline_output = select_action(candidates)
        _validate_policy_selection(candidates, baseline_output)

        return baseline_output, {
            "mode": "baseline",
            "rl_loaded": False,
        }

    if _rl_policy is None and _rl_policy_error is None:
        model_path = _get_rl_model_path()

        try:
            from .ReinforcementLearning.interface import (
                ReinforcementLearningPolicy,
            )

            if not model_path.is_file():
                raise FileNotFoundError(
                    f"PPO model file does not exist: {model_path}"
                )

            _rl_policy = ReinforcementLearningPolicy(model_path)

        except Exception as exc:
            _rl_policy_error = f"{type(exc).__name__}: {exc}"
            logger.exception("PPO policy could not be loaded")

    if _rl_policy is None:
        baseline_output = select_action(candidates)
        _validate_policy_selection(candidates, baseline_output)

        # Preserve the existing PolicyOutput contract.
        from .contracts import PolicyOutput

        fallback_output = PolicyOutput(
            selected_action=baseline_output.selected_action,
            confidence=baseline_output.confidence,
            rationale=(
                "PPO was requested but unavailable. "
                "The validated deterministic baseline was used. "
                f"Load error: {_rl_policy_error}"
            ),
            policy_name="deterministic_baseline_fallback",
        )

        return fallback_output, {
            "mode": "ppo_requested",
            "rl_loaded": False,
            "fallback": "deterministic_baseline",
            "error": _rl_policy_error,
        }

    try:
        policy_output = _rl_policy.select(candidates, metrics)
        _validate_policy_selection(candidates, policy_output)

    except Exception as exc:
        # Invalid PPO output must not be silently replaced by candidates[0].
        # Fall back only through the explicit, validated baseline path.
        logger.exception("PPO selection failed; using validated baseline")

        baseline_output = select_action(candidates)
        _validate_policy_selection(candidates, baseline_output)

        from .contracts import PolicyOutput

        fallback_output = PolicyOutput(
            selected_action=baseline_output.selected_action,
            confidence=baseline_output.confidence,
            rationale=(
                "PPO selection failed. A validated deterministic baseline "
                f"was used. Selection error: {type(exc).__name__}: {exc}"
            ),
            policy_name="deterministic_baseline_fallback",
        )

        return fallback_output, {
            "mode": "ppo_selection_fallback",
            "rl_loaded": True,
            "fallback": "deterministic_baseline",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return policy_output, {
        "mode": "ppo",
        "rl_loaded": True,
    }


def _check_resource_budget(candidate: Any) -> bool:
    """Fail closed until a real resource-budget validator is connected.

    Do not replace this default with True merely to permit actions.
    Implement and test a real budget check before returning True.
    """
    checker = getattr(_safety_gate, "check_resource_budget", None)

    if not callable(checker):
        logger.warning(
            "No resource-budget checker is configured; failing closed."
        )
        return False

    try:
        result = checker(candidate)
        return result is True
    except Exception:
        logger.exception("Resource-budget validation failed")
        return False


def _check_preconditions(candidate: Any) -> bool:
    """Fail closed until a real action-precondition validator is connected."""
    checker = getattr(_safety_gate, "check_preconditions", None)

    if not callable(checker):
        logger.warning(
            "No action-precondition checker is configured; failing closed."
        )
        return False

    try:
        result = checker(candidate)
        return result is True
    except Exception:
        logger.exception("Action-precondition validation failed")
        return False


def _assert_nonexecuting_result(executor_result):
    """Require affirmative evidence that live execution remained disabled."""
    metadata = _executor_metadata(executor_result)

    if metadata.get("executed") is True:
        raise RuntimeError(
            "Safety contract violation: executor reported executed=True "
            "while live execution is disabled."
        )

    if metadata.get("execution_enabled") is True:
        raise RuntimeError(
            "Safety contract violation: executor reported "
            "execution_enabled=True while live execution is disabled."
        )

    if metadata.get("executed") is not False:
        raise RuntimeError(
            "Safety contract violation: executor must explicitly report "
            "executed=False."
        )

    if metadata.get("dry_run") is not True:
        raise RuntimeError(
            "Safety contract violation: executor must explicitly report "
            "dry_run=True."
        )

    if metadata.get("execution_enabled") is not False:
        raise RuntimeError(
            "Safety contract violation: executor must explicitly report "
            "execution_enabled=False."
        )

    return metadata


def _executor_metadata(executor_result: Any) -> dict[str, Any]:
    """Normalize executor output without assuming that a call executed an action."""
    result = _serialize(executor_result)

    if isinstance(result, dict):
        return result

    if result is None:
        return {"status": "unknown", "result": None}

    return {
        "status": str(getattr(executor_result, "status", "unknown")),
        "executed": getattr(executor_result, "executed", None),
        "dry_run": getattr(executor_result, "dry_run", None),
        "result": str(result),
    }

def _proposal_status(verdict: Any, executor_result: Any) -> str:
    """Derive proposal status from safety and execution evidence."""
    if getattr(verdict, "allowed", False) is not True:
        return "blocked"

    metadata = _executor_metadata(executor_result)

    # A positive execution claim requires an explicit result field.
    if metadata.get("executed") is True:
        return "executed"

    if metadata.get("dry_run") is True:
        return "dry_run_only"

    status = str(metadata.get("status", "")).strip().lower()

    if status in {"dry_run", "dry_run_only", "simulated"}:
        return "dry_run_only"

    if status in {"blocked", "rejected", "failed", "error"}:
        return status

    # Never claim dry-run or successful execution without evidence.
    return "executor_status_unknown"

def analyze(
    agent_id: str,
    host: str,
    text: str,
    metrics: dict[str, Any],
    *,
    authorized: bool = False,
    persist_telemetry: bool = True,
) -> dict[str, Any]:
    """Validate, analyze, select, safety-check, and audit one observation."""
    observation = Observation(
        agent_id=agent_id,
        host=host,
        text=text,
        metrics=metrics,
    )

    stage = "telemetry_persistence"

    try:
        normalized_metrics = observation.metrics

        if persist_telemetry:
            add_telemetry(
                observation.agent_id,
                observation.host,
                normalized_metrics,
            )

        stage = "classification"
        classification = _classifier.predict(observation.text)

        stage = "anomaly_detection"
        raw_findings = detect_anomalies(normalized_metrics)

        if (
            classification.get("available")
            and classification.get("label", "").lower()
            in {"malware", "malicious", "intrusion"}
        ):
            raw_findings.append(
                {
                    "category": "malware_text",
                    "severity": (
                        "high"
                        if classification["confidence"] >= 0.8
                        else "medium"
                    ),
                    "description": (
                        "Classifier predicts a malicious class; "
                        "independent validation is required."
                    ),
                    "evidence": {
                        "label": classification["label"],
                        "confidence": classification["confidence"],
                    },
                }
            )

        stage = "finding_persistence"
        saved_findings = []
        evidence = []

        for item in raw_findings:
            finding_id = add_finding(
                observation.agent_id,
                observation.host,
                item["severity"],
                item["category"],
                item,
            )

            saved_item = {**item, "finding_id": finding_id}
            saved_findings.append(saved_item)

            evidence.append(
                Evidence(
                    source=f"finding:{finding_id}",
                    category=item["category"],
                    severity=item["severity"],
                    description=item["description"],
                    details=item.get("evidence", {}),
                )
            )

        stage = "root_cause_analysis"
        hypotheses = rank_hypotheses(
            normalized_metrics,
            raw_findings,
            classification,
        )

        stage = "candidate_generation"
        candidates = generate_candidates(
            [_serialize(item) for item in evidence]
        )

        if not candidates:
            raise ValueError(
                "No candidate actions were generated; no action was selected."
            )

        stage = "policy_selection"
        policy_output, policy_runtime = _select_policy(
            candidates,
            normalized_metrics,
        )

        # Resolve exactly once. Never silently substitute candidates[0].
        selected = _validate_policy_selection(candidates, policy_output)

        stage = "safety_evaluation"

        # Both checks default to False unless real validators exist.
        resource_budget_ok = _check_resource_budget(selected)
        preconditions_ok = _check_preconditions(selected)

        verdict = _safety_gate.evaluate(
            selected,
            authorized=authorized,
            resource_budget_ok=resource_budget_ok,
            preconditions_ok=preconditions_ok,
        )

        stage = "executor_evaluation"

        # The safety gate must deny actions when any required check fails.
        # ActionExecutor must independently enforce dry-run/no-execution mode.

        executor_result = _executor.execute(selected, verdict)
        executor_metadata = _assert_nonexecuting_result(executor_result)

        proposal_status = _proposal_status(verdict, executor_result)

        stage = "proposal_persistence"

        if saved_findings:
            for finding in saved_findings:
                add_proposal(
                    finding["finding_id"],
                    selected.action,
                    selected.risk,
                    proposal_status,
                    selected.rationale,
                )
        else:
            add_proposal(
                None,
                selected.action,
                selected.risk,
                proposal_status,
                selected.rationale,
            )

        stage = "decision_audit"

        timestamp = _utc_timestamp()
        decision_id = str(uuid4())

        record = DecisionRecord(
            decision_id=decision_id,
            timestamp=timestamp,
            agent_id=observation.agent_id,
            host=observation.host,
            observation=_serialize(observation),
            model_outputs={
                "text_classifier": classification,
                "policy_runtime": policy_runtime,
                "root_cause_hypotheses": _serialize(hypotheses),
                "rl_state_mapping": (
                    "normalized_cpu_memory_disk_connections_load_v1"
                ),
            },
            evidence=evidence,
            candidate_actions=candidates,
            policy_output=policy_output,
            safety_verdict=verdict,
            executor_result=executor_result,
        )

        serialized = record.model_dump(mode="json")

        add_decision_audit(
            decision_id,
            timestamp,
            observation.agent_id,
            observation.host,
            serialized,)

        return {
            "decision": serialized,
            "anomalies": saved_findings,
            "root_cause_hypotheses": _serialize(hypotheses),
            "action_proposals": [
                {
                    "action": candidate.action,
                    "risk": candidate.risk,
                    "rationale": candidate.rationale,
                    "selected": candidate is selected,
                }
                for candidate in candidates
            ],
            "safety": {
                "verdict": verdict.verdict,
                "allowed": verdict.allowed,
                "reasons": verdict.reasons,
                "resource_budget_ok": resource_budget_ok,
                "preconditions_ok": preconditions_ok,
                "executor_result": executor_metadata,
                "proposal_status": proposal_status,
                "no_actions_executed": True,
            },
        }

    except Exception as exc:
        logger.exception(
            "Decision pipeline failed at stage %s for host %s",
            stage,
            observation.host,
        )

        _audit_failure(
            observation=observation,
            stage=stage,
            error=exc,
        )

        raise
