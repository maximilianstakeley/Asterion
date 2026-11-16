
"""Deterministic baseline policy for Asterion.

Responsibilities:
- Convert evidence into a deduplicated set of candidate actions.
- Select a candidate deterministically using explicit priorities and risk.
- Return the same PolicyOutput contract used by optional reinforcement-learning
  policies.

This module proposes actions only. It does not execute system changes or
authorize execution; those responsibilities belong to the safety and executor
layers.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from .contracts import CandidateAction, PolicyOutput

logger = logging.getLogger(__name__)


# Candidate-generation rules.
# Risk values are heuristic scores, not calibrated probabilities or measured
# probabilities of harm.
ACTION_BY_CATEGORY: dict[str, tuple[str, float]] = {
    "high_cpu": ("reduce_noncritical_workload", 0.18),
    "high_memory": ("collect_diagnostics", 0.05),
    "high_disk": ("collect_diagnostics", 0.05),
    "connection_volume": ("review_suspicious_activity", 0.12),
    "malware_text": ("review_suspicious_activity", 0.12),
}

DEFAULT_ACTION = "notify_operator"
DEFAULT_RISK = 0.02

# Lower values have higher priority. The priority determines the primary
# selection order; risk breaks ties between candidates at the same priority.
ACTION_PRIORITY: dict[str, int] = {
    "review_suspicious_activity": 0,
    "collect_diagnostics": 1,
    "reduce_noncritical_workload": 2,
    "notify_operator": 3,
}

BASELINE_CONFIDENCE = 0.60


def _read_field(item: Any, field: str, default: Any = None) -> Any:
    """Read a field from a mapping or an object with model attributes."""
    if isinstance(item, Mapping):
        return item.get(field, default)

    return getattr(item, field, default)


def _safe_text(value: Any, default: str) -> str:
    """Return non-empty text without trusting arbitrary input types."""
    if isinstance(value, str) and value.strip():
        return value.strip()

    return default


def _valid_risk(value: Any, default: float = DEFAULT_RISK) -> float:
    """Return a finite risk score in the supported interval [0, 1]."""
    if isinstance(value, bool):
        return default

    try:
        risk = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
        return default

    return risk


def _candidate_key(candidate: CandidateAction) -> tuple[int, float]:
    """Produce a stable selection key for a validated candidate."""
    action = candidate.action
    risk = _valid_risk(candidate.risk)

    return ACTION_PRIORITY.get(action, 99), risk


def generate_candidates(
    evidence: list[dict[str, Any]],
) -> list[CandidateAction]:
    """Generate deduplicated action candidates from collected evidence.

    Evidence may be supplied as dictionaries or compatible model objects.
    Only the first occurrence of each action is retained, preserving evidence
    order. If no usable evidence produces a candidate, the result is an
    observation-only notification candidate.
    """
    if not isinstance(evidence, (list, tuple)):
        raise TypeError("evidence must be a list or tuple")

    candidates: list[CandidateAction] = []
    seen_actions: set[str] = set()

    for index, item in enumerate(evidence):
        if item is None:
            logger.warning("Ignoring null evidence item at index %d", index)
            continue

        category = _safe_text(
            _read_field(item, "category"),
            "unknown",
        )

        description = _safe_text(
            _read_field(item, "description"),
            _safe_text(
                _read_field(item, "detail"),
                "No description provided",
            ),
        )

        action, risk = ACTION_BY_CATEGORY.get(
            category,
            (DEFAULT_ACTION, DEFAULT_RISK),
        )

        if action in seen_actions:
            continue

        candidates.append(
            CandidateAction(
                action=action,
                risk=risk,
                rationale=(
                    f"Candidate generated from {category} evidence: "
                    f"{description}"
                ),
            )
        )
        seen_actions.add(action)

    if not candidates:
        candidates.append(
            CandidateAction(
                action=DEFAULT_ACTION,
                risk=DEFAULT_RISK,
                rationale=(
                    "No usable evidence produced an actionable candidate; "
                    "retain observation-only posture."
                ),
            )
        )

    return candidates


def select_action(
    candidates: list[CandidateAction],
) -> PolicyOutput:
    """Select one candidate using the deterministic baseline policy.

    Selection is deterministic for a given ordered candidate list. The
    function does not invoke an executor, bypass the safety gate, or claim
    that the selected action is safe to execute.
    """
    if not isinstance(candidates, (list, tuple)):
        raise TypeError("candidates must be a list or tuple")

    if not candidates:
        raise ValueError("Cannot select an action from an empty candidate list")

    for index, candidate in enumerate(candidates):
        if candidate is None:
            raise ValueError(
                f"Candidate at index {index} is null"
            )

        action = getattr(candidate, "action", None)
        if not isinstance(action, str) or not action.strip():
            raise ValueError(
                f"Candidate at index {index} has an invalid action"
            )

        risk = getattr(candidate, "risk", None)
        if isinstance(risk, bool):
            raise ValueError(
                f"Candidate at index {index} has an invalid risk score"
            )

        try:
            numeric_risk = float(risk)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Candidate at index {index} has an invalid risk score"
            ) from exc

        if not math.isfinite(numeric_risk) or not 0.0 <= numeric_risk <= 1.0:
            raise ValueError(
                f"Candidate at index {index} risk must be finite and in [0, 1]"
            )

    chosen = min(candidates, key=_candidate_key)

    return PolicyOutput(
        selected_action=chosen.action,
        confidence=BASELINE_CONFIDENCE,
        rationale=(
            "Selected by deterministic baseline priority; "
            f"candidate risk={float(chosen.risk):.2f}. "
            "This selection is a proposal, not execution authorization."
        ),
    )
