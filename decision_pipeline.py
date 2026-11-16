
"""Asterion decision pipeline: diagnosis, constrained selection, safety, audit.

This module never executes operating-system commands.
Live action execution is deliberately blocked.
Simulation requires an explicitly supplied simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA_VERSION = 1
LIVE_EXECUTION_ENABLED = False

ALLOWED_ACTIONS = {
    "collect_diagnostics",
    "increase_observation_frequency",
    "reduce_optional_workload",
    "scale_resource_simulated",
    "isolate_host_simulated",
    "no_op",
}

# These are illustrative mappings, not proof of causality.
CATEGORY_RULES = {
    "resource_threshold": (
        "resource_pressure",
        "Resource utilization exceeded a configured threshold.",
    ),
    "abrupt_metric_change": (
        "rapid_state_change",
        "A measured system metric changed unusually quickly.",
    ),
    "persistent_threshold_violation": (
        "persistent_resource_pressure",
        "A resource threshold appears to have been breached repeatedly.",
    ),
    "multivariate_baseline_deviation": (
        "multivariate_anomaly",
        "Several measurements differ from the supplied baseline.",
    ),
    "security_indicator": (
        "security_signal",
        "A security event or configured indicator requires investigation.",
    ),
    "network_intrusion": (
        "network_security_signal",
        "A network security detector reported a possible intrusion.",
    ),
}


def _canonical(value: Any) -> str:
    """Stable JSON representation for reproducibility and hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _severity_weight(severity: Any) -> float:
    return {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.55,
        "low": 0.3,
        "info": 0.1,
    }.get(str(severity).lower(), 0.2)


def _stable_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def diagnose(findings: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Group detector findings into ranked, explicitly provisional hypotheses."""
    items = list(findings)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for index, finding in enumerate(items):
        if not isinstance(finding, dict):
            continue

        category = finding.get("category")
        rule = CATEGORY_RULES.get(str(category))
        if not rule:
            continue

        hypothesis_type, explanation = rule
        host = str(finding.get("host") or "unknown-host")
        groups.setdefault((host, hypothesis_type), []).append({
            "finding_index": index,
            "finding_id": finding.get("finding_id"),
            "category": category,
            "severity": finding.get("severity", "unknown"),
            "timestamp": finding.get("timestamp"),
            "evidence": finding.get("evidence", {}),
        })

    hypotheses = []
    for (host, hypothesis_type), evidence_items in groups.items():
        # This is a transparent ranking heuristic, not a calibrated probability.
        weights = [
            _severity_weight(item.get("severity"))
            for item in evidence_items
        ]
        score = min(1.0, sum(weights) / max(1, len(weights)) *
                    min(1.0, len(weights) / 3.0))

        supporting = [item["finding_index"] for item in evidence_items]
        hypothesis = {
            "host": host,
            "hypothesis_type": hypothesis_type,
            "explanation": next(
                explanation
                for category, (kind, explanation) in CATEGORY_RULES.items()
                if kind == hypothesis_type
            ),
            "ranking_score": round(score, 4),
            "score_semantics": (
                "Heuristic priority score; not a probability or proof of cause."
            ),
            "supporting_evidence": evidence_items,
            "contradictory_evidence": [],
            "uncertainty": (
                "Causal relationships have not been established. "
                "Correlated findings may share one underlying cause."
            ),
        }
        hypothesis["hypothesis_id"] = _stable_id("hyp", hypothesis)
        hypotheses.append(hypothesis)

    hypotheses.sort(
        key=lambda item: (item["ranking_score"], item["hypothesis_type"]),
        reverse=True,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "input_finding_count": len(items),
        "hypotheses": hypotheses,
    }


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    target_host: str
    rationale: str
    expected_benefit: float
    estimated_risk: float
    reversible: bool
    requires_authorization: bool = True

    def validate(self) -> None:
        if self.action_id not in ALLOWED_ACTIONS:
            raise ValueError(f"Action is not allowlisted: {self.action_id}")
        if not self.target_host or self.target_host == "unknown-host":
            raise ValueError("Action requires a known target host")
        if not _finite_number(self.expected_benefit):
            raise ValueError("expected_benefit must be finite")
        if not _finite_number(self.estimated_risk):
            raise ValueError("estimated_risk must be finite")
        if not 0 <= self.expected_benefit <= 1:
            raise ValueError("expected_benefit must be between 0 and 1")
        if not 0 <= self.estimated_risk <= 1:
            raise ValueError("estimated_risk must be between 0 and 1")


def generate_candidates(
    hypotheses: dict[str, Any],
) -> list[ActionCandidate]:
    """Generate conservative candidates; does not execute them."""
    candidates: list[ActionCandidate] = []

    for hypothesis in hypotheses.get("hypotheses", []):
        host = hypothesis.get("host")
        kind = hypothesis.get("hypothesis_type")

        # Read-only diagnostics are preferred as an initial response.
        candidates.append(ActionCandidate(
            action_id="collect_diagnostics",
            target_host=host,
            rationale=f"Collect additional evidence for {kind}.",
            expected_benefit=0.45,
            estimated_risk=0.05,
            reversible=True,
        ))

        if kind in {
            "resource_pressure",
            "persistent_resource_pressure",
            "rapid_state_change",
            "multivariate_anomaly",
        }:
            candidates.append(ActionCandidate(
                action_id="increase_observation_frequency",
                target_host=host,
                rationale="Gather more frequent measurements to characterize the event.",
                expected_benefit=0.35,
                estimated_risk=0.08,
                reversible=True,
            ))

        if kind in {"resource_pressure", "persistent_resource_pressure"}:
            candidates.append(ActionCandidate(
                action_id="reduce_optional_workload",
                target_host=host,
                rationale="Candidate for reducing optional workload after authorization.",
                expected_benefit=0.55,
                estimated_risk=0.35,
                reversible=True,
            ))
            candidates.append(ActionCandidate(
                action_id="scale_resource_simulated",
                target_host=host,
                rationale="Evaluate resource scaling in simulation before considering deployment.",
                expected_benefit=0.65,
                estimated_risk=0.2,
                reversible=True,
            ))

        if kind in {"security_signal", "network_security_signal"}:
            candidates.append(ActionCandidate(
                action_id="isolate_host_simulated",
                target_host=host,
                rationale="Evaluate host isolation in simulation; this signal alone does not authorize isolation.",
                expected_benefit=0.7,
                estimated_risk=0.8,
                reversible=False,
            ))

    candidates.append(ActionCandidate(
        action_id="no_op",
        target_host="system",
        rationale="Take no corrective action while collecting or reviewing evidence.",
        expected_benefit=0.0,
        estimated_risk=0.0,
        reversible=True,
        requires_authorization=False,
    ))

    return candidates


def _candidate_score(candidate: ActionCandidate, risk_weight: float) -> float:
    return candidate.expected_benefit - risk_weight * candidate.estimated_risk


def select_action(
    candidates: Iterable[ActionCandidate],
    *,
    authorized_hosts: set[str] | None = None,
    max_risk: float = 0.25,
    risk_weight: float = 1.0,
    learned_proposal: str | None = None,
) -> dict[str, Any]:
    """Select a safe candidate; a learned policy can only suggest an allowlisted ID."""
    if not _finite_number(max_risk) or not 0 <= max_risk <= 1:
        raise ValueError("max_risk must be between 0 and 1")
    if not _finite_number(risk_weight) or risk_weight < 0:
        raise ValueError("risk_weight must be finite and nonnegative")

    candidates = list(candidates)
    eligible: list[ActionCandidate] = []

    for candidate in candidates:
        candidate.validate()
        if candidate.estimated_risk > max_risk:
            continue
        if candidate.requires_authorization:
            if authorized_hosts is None:
                continue
            if candidate.target_host not in authorized_hosts:
                continue
        eligible.append(candidate)

    # A policy proposal cannot override the risk cap or authorization.
    proposal_used = False
    if learned_proposal:
        proposed = [
            candidate for candidate in eligible
            if candidate.action_id == learned_proposal
        ]
        if proposed:
            eligible = proposed
            proposal_used = True

    if not eligible:
        return {
            "selected_action": "no_op",
            "target_host": None,
            "selection_score": 0.0,
            "selection_reason": "No candidate satisfied the configured safety constraints.",
            "learned_proposal_used": False,
        }

    selected = max(
        eligible,
        key=lambda item: (
            _candidate_score(item, risk_weight),
            item.action_id == "no_op",
        ),
    )

    return {
        "selected_action": selected.action_id,
        "target_host": selected.target_host,
        "selection_score": round(_candidate_score(selected, risk_weight), 4),
        "selection_reason": selected.rationale,
        "expected_benefit": selected.expected_benefit,
        "estimated_risk": selected.estimated_risk,
        "reversible": selected.reversible,
        "learned_proposal_used": proposal_used,
        "action_execution": "disabled",}


def safety_gate(
    decision: dict[str, Any],
    *,
    authorized_hosts: set[str],
    max_risk: float = 0.25,
    live_execution_enabled: bool = LIVE_EXECUTION_ENABLED,
    simulation_mode: bool = False,
) -> dict[str, Any]:
    """Validate authorization and risk without performing any action."""
    action_id = decision.get("selected_action")
    target = decision.get("target_host")
    risk = decision.get("estimated_risk")
    reasons = []

    if action_id not in ALLOWED_ACTIONS:
        reasons.append("action_not_allowlisted")

    if action_id != "no_op" and target not in authorized_hosts:
        reasons.append("target_not_authorized")

    if not _finite_number(risk) or not 0 <= risk <= max_risk:
        reasons.append("risk_limit_exceeded")

    # Simulation is not permission to execute against a live host.
    if not simulation_mode and not live_execution_enabled:
        reasons.append("live_execution_disabled")

    if simulation_mode and action_id in {
        "reduce_optional_workload",
        "scale_resource_simulated",
        "isolate_host_simulated",
    }:
        if action_id not in {"scale_resource_simulated", "isolate_host_simulated"}:
            reasons.append("action_not_approved_for_simulation")

    return {
        "approved": len(reasons) == 0,
        "action_id": action_id,
        "target_host": target,
        "reasons": reasons,
        "simulation_mode": simulation_mode,
        "live_execution_enabled": live_execution_enabled, }

def execute_action(
    decision: dict[str, Any],
    gate: dict[str, Any],
    *,
    simulation: bool = False,
    simulator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Live execution is impossible here. Simulation requires an explicit adapter."""
    if not simulation:
        return {
            "status": "blocked",
            "reason": "live_execution_disabled",
            "action_execution": "disabled",}

    if decision.get("selected_action") not in ALLOWED_ACTIONS:
        return {"status": "blocked", "reason": "action_not_allowlisted"}

    if simulator is None:
        return {
            "status": "blocked",
            "reason": "simulation_adapter_required",}

    # A simulated action still requires explicit authorization and a risk gate.
    if not gate.get("approved"):
        return {
            "status": "blocked",
            "reason": "safety_gate_rejected",
            "gate_reasons": gate.get("reasons", []),}

    result = simulator(dict(decision))
    if not isinstance(result, dict) or "status" not in result:
        return {
            "status": "invalid_simulator_result",
            "reason": "simulator_must_return_a_status",}

    return {
        "status": "simulated",
        "simulator_result": result,
        "action_execution": "simulated_only",}


class AuditLedger:
    """SQLite hash-linked audit log for decisions and feedback.

    The hash chain detects modifications to stored records. It is not a
    substitute for filesystem access control, backups, or signed checkpoints.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS audit_records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    record_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL UNIQUE
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def append(self, record_type: str, payload: dict[str, Any]) -> str:
        if not record_type or not isinstance(payload, dict):
            raise ValueError("record_type and dictionary payload are required")

        payload_json = _canonical(payload)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT sequence, record_hash FROM audit_records "
                "ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = row[1] if row else "GENESIS"
            created_at = time.time()
            body = {
                "created_at": created_at,
                "record_type": record_type,
                "payload_json": payload_json,
                "previous_hash": previous_hash,
            }
            record_hash = hashlib.sha256(
                _canonical(body).encode("utf-8")
            ).hexdigest()
            connection.execute(
                """INSERT INTO audit_records
                (created_at, record_type, payload_json, previous_hash, record_hash)
                VALUES (?, ?, ?, ?, ?)""",
                (created_at, record_type, payload_json, previous_hash, record_hash),
            )
        return record_hash

    def verify(self) -> dict[str, Any]:
        expected_previous = "GENESIS"
        checked = 0
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT created_at, record_type, payload_json,
                          previous_hash, record_hash
                   FROM audit_records ORDER BY sequence"""
            ).fetchall()

        for created_at, record_type, payload_json, previous_hash, record_hash in rows:
            if previous_hash != expected_previous:
                return {"valid": False, "checked_records": checked,
                        "reason": "previous_hash_mismatch"}

            body = {
                "created_at": created_at,
                "record_type": record_type,
                "payload_json": payload_json,
                "previous_hash": previous_hash,
            }
            calculated = hashlib.sha256(
                _canonical(body).encode("utf-8")
            ).hexdigest()
            if calculated != record_hash:
                return {"valid": False, "checked_records": checked,
                        "reason": "record_hash_mismatch"}

            expected_previous = record_hash
            checked += 1

        return {"valid": True, "checked_records": checked}


def validate_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    """Admit only explicitly verified, labeled feedback to a training queue."""
    required = ("decision_id", "outcome_label", "evidence_refs", "verified")
    missing = [key for key in required if key not in feedback]
    if missing:
        return {"accepted_for_training": False, "reason": "missing_fields",
                "missing_fields": missing}

    allowed_labels = {"beneficial", "neutral", "harmful"}
    if feedback["outcome_label"] not in allowed_labels:
        return {"accepted_for_training": False, "reason": "invalid_outcome_label"}
    if feedback["verified"] is not True:
        return {"accepted_for_training": False, "reason": "outcome_not_verified"}
    if not isinstance(feedback["evidence_refs"], list) or not feedback["evidence_refs"]:
        return {"accepted_for_training": False, "reason": "evidence_refs_required"}
    if not all(isinstance(ref, str) and ref.strip()
               for ref in feedback["evidence_refs"]):
        return {"accepted_for_training": False, "reason": "invalid_evidence_refs"}
    if not isinstance(feedback["decision_id"], str) or not feedback["decision_id"]:
        return {"accepted_for_training": False, "reason": "invalid_decision_id"}

    return {
        "accepted_for_training": True,
        "reason": "verified_labeled_feedback",
        "training_data_status": "eligible_pending_dataset_review",
    }
