
"""Fail-closed executor interface.

This build never performs live system changes. Actions can only be evaluated
and reported as blocked or dry-run-only. No process control, network changes,
file deletion, or workload mutation is implemented.
"""

from __future__ import annotations

import logging
from typing import Any

from .contracts import CandidateAction, SafetyVerdict


logger = logging.getLogger(__name__)


class ActionExecutor:
    """Safely report action outcomes without executing live system changes."""

    EXECUTION_ENABLED = False
    EXECUTION_MODE = "disabled_by_design"

    @staticmethod
    def _result(
        *,
        status: str,
        reason: str,
        action: str | None,
        safety_allowed: bool,
        risk: str | None = None,
    ) -> dict[str, Any]:
        """Create a consistent, audit-friendly, non-executing result."""
        return {
            "status": status,
            "action": action,
            "risk": risk,
            "executed": False,
            "dry_run": True,
            "execution_enabled": False,
            "execution_mode": ActionExecutor.EXECUTION_MODE,
            "safety_allowed": safety_allowed,
            "reason": reason,
        }

    def execute(
        self,
        candidate: CandidateAction,
        verdict: SafetyVerdict,
    ) -> dict[str, Any]:
        """Evaluate an action request without applying any system changes.

        Safety policy:
        - A missing or malformed candidate is rejected.
        - A missing or malformed safety verdict is rejected.
        - An explicit denial from the safety gate blocks the request.
        - An allowed verdict permits dry-run reporting only.
        - No code path performs live execution.
        """
        action: str | None = None
        risk: str | None = None

        # Validate the candidate without trusting arbitrary caller input.
        if candidate is None:
            return self._result(
                status="rejected",
                reason="missing_candidate",
                action=None,
                risk=None,
                safety_allowed=False,
            )

        raw_action = getattr(candidate, "action", None)
        raw_risk = getattr(candidate, "risk", None)

        if isinstance(raw_action, str) and raw_action.strip():
            action = raw_action.strip()
        else:
            logger.warning("Rejected action request with an invalid candidate.")
            return self._result(
                status="rejected",
                reason="invalid_candidate_action",
                action=None,
                risk=None,
                safety_allowed=False,
            )

        if isinstance(raw_risk, str):
            risk = raw_risk

        # Fail closed if the verdict is missing or malformed.
        if verdict is None:
            logger.warning(
                "Action '%s' rejected because no safety verdict was supplied.",
                action,
            )
            return self._result(
                status="rejected",
                reason="missing_safety_verdict",
                action=action,
                risk=risk,
                safety_allowed=False,
            )

        allowed = getattr(verdict, "allowed", None)

        if not isinstance(allowed, bool):
            logger.warning(
                "Action '%s' rejected because the safety verdict is invalid.",
                action,
            )
            return self._result(
                status="rejected",
                reason="invalid_safety_verdict",
                action=action,
                risk=risk,
                safety_allowed=False,
            )

        if not allowed:
            reasons = getattr(verdict, "reasons", None)
            logger.info("Safety gate blocked action '%s'.", action)

            return {
                **self._result(
                    status="not_executed",
                    reason="safety_gate_blocked",
                    action=action,
                    risk=risk,
                    safety_allowed=False,
                ),
                "safety_reasons": (
                    list(reasons)
                    if isinstance(reasons, (list, tuple))
                    else []
                ),
            }

        # This build deliberately has no live execution implementation.
        # A permitted safety verdict is NOT permission to execute an action.
        logger.info(
            "Action '%s' passed the supplied safety verdict but remains "
            "dry-run-only because live execution is disabled.",
            action,
        )

        return self._result(
            status="dry_run_only",
            reason="live_execution_disabled_by_design",
            action=action,
            risk=risk,
            safety_allowed=True,
        )
