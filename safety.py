
"""Deterministic safety gate for Asterion.

This release never performs live remediation.

The gate validates candidate actions, risk limits, authorization, and explicit
preconditions. Passing the gate is not permission to execute an action:
live execution remains disabled by design.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .contracts import CandidateAction, SafetyVerdict


# Allowlisted recommendations and their maximum heuristic risk scores.
# These limits are policy thresholds, not calibrated probabilities.
ACTION_RISK_LIMITS: dict[str, float] = {
    "collect_diagnostics": 0.20,
    "notify_operator": 0.10,
    "reduce_noncritical_workload": 0.25,
    "review_suspicious_activity": 0.20,
}


class SafetyGate:
    """Fail-closed validation gate; never enables live execution."""

    def __init__(
        self,
        max_action_risk: float = 0.25,
        execution_enabled: bool = False,
    ) -> None:
        if isinstance(max_action_risk, bool):
            raise ValueError("max_action_risk must be a number between 0 and 1")

        try:
            risk_limit = float(max_action_risk)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "max_action_risk must be a number between 0 and 1"
            ) from exc

        if not math.isfinite(risk_limit) or not 0.0 <= risk_limit <= 1.0:
            raise ValueError("max_action_risk must be finite and in [0, 1]")

        self.max_action_risk = risk_limit

        # Deliberately ignore attempts to enable execution. This release does
        # not have a verified production executor or rollback implementation.
        self.execution_enabled = False

    @staticmethod
    def _read(candidate: Any, field: str, default: Any = None) -> Any:
        """Read a field from a mapping or an object."""
        if isinstance(candidate, Mapping):
            return candidate.get(field, default)

        return getattr(candidate, field, default)

    @staticmethod
    def _valid_risk(value: Any) -> float | None:
        """Return a finite risk score in [0, 1], or None if invalid."""
        if isinstance(value, bool):
            return None

        try:
            risk = float(value)
        except (TypeError, ValueError, OverflowError):
            return None

        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            return None

        return risk

    def check_resource_budget(self, candidate: CandidateAction) -> bool:
        """Validate the candidate's declared risk against configured limits.

        This is a risk-budget check, not a measurement of actual host resource
        availability. External resource-budget telemetry is not available to
        this method in the current interface.
        """
        if candidate is None:
            return False

        action = self._read(candidate, "action")
        risk = self._valid_risk(self._read(candidate, "risk"))

        if not isinstance(action, str) or not action.strip():
            return False

        if risk is None:
            return False

        action_limit = ACTION_RISK_LIMITS.get(action)
        if action_limit is None:
            return False

        return risk <= min(action_limit, self.max_action_risk)

    def check_preconditions(self, candidate: CandidateAction) -> bool:
        """Return whether execution preconditions have been verified.

        The current release has no action-specific precondition evaluator
        connected to live system state. Therefore, it cannot verify that an
        operational action's preconditions hold and must return False.

        Implement a real, independently tested precondition evaluator before
        changing this behavior.
        """
        return False

    def evaluate(
        self,
        candidate: CandidateAction,
        *,
        authorized: bool = False,
        resource_budget_ok: bool = False,
        preconditions_ok: bool = False,
    ) -> SafetyVerdict:
        """Evaluate a proposed action and return a fail-closed verdict.

        `authorized`, `resource_budget_ok`, and `preconditions_ok` must each
        be explicitly supplied as True. Missing or invalid conditions fail
        closed. Even if all checks pass, execution remains disabled.
        """
        reasons: list[str] = []

        if candidate is None:
            reasons.append("No candidate action supplied")
            action = None
            risk = None
        else:
            action = self._read(candidate, "action")
            risk = self._valid_risk(self._read(candidate, "risk"))

            if not isinstance(action, str) or not action.strip():
                reasons.append("Candidate action is missing or invalid")
                action = None

            if risk is None:
                reasons.append("Candidate risk is missing or invalid")

        action_limit = ACTION_RISK_LIMITS.get(action) if action else None

        if action is not None and action_limit is None:
            reasons.append("Action is not allowlisted")

        if action_limit is not None and risk is not None:
            effective_limit = min(action_limit, self.max_action_risk)

            if risk > effective_limit:
                reasons.append(
                    "Action risk exceeds deterministic limit "
                    f"({risk:.3f} > {effective_limit:.3f})"
                )

        if authorized is not True:
            reasons.append("No explicit authorization supplied")

        if resource_budget_ok is not True:
            reasons.append("Resource budget constraint not verified")

        if preconditions_ok is not True:
            reasons.append("Action preconditions not verified")

        if reasons:
            return SafetyVerdict(
                allowed=False,
                executed=False,
                verdict="blocked",
                reasons=reasons,
            )

        # A passing verdict means only that the supplied checks passed.
        # It never enables or performs live execution.
        return SafetyVerdict(
            allowed=True,
            executed=False,
            verdict="allowed_but_execution_disabled",
            reasons=[
                "Validation checks passed.",
                "Live execution remains disabled in this release.",
            ],
        )
