
"""Evidence-based root-cause hypothesis ranking for Asterion.

This module ranks possible explanations for observed findings. It does not
establish causality, execute remediation, or treat heuristic confidence scores
as calibrated probabilities.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


# These values are heuristic ranking scores, not calibrated probabilities.
HYPOTHESIS_RULES = (
    (
        "high_memory",
        "Memory pressure or a memory-intensive workload",
        0.72,
        "memory_percent",
    ),
    (
        "high_cpu",
        "CPU saturation from workload growth or an inefficient task",
        0.68,
        "cpu_percent",
    ),
    (
        "high_disk",
        "Storage capacity pressure",
        0.78,
        "disk_percent",
    ),
    (
        "connection_volume",
        (
            "High connection volume from workload, misconfiguration, "
            "or potentially suspicious activity"
        ),
        0.55,
        "network_connections",
    ),
)

MALICIOUS_LABELS = frozenset(
    {"malware", "malicious", "intrusion"}
)

DEFAULT_NEXT_CHECKS = [
    "Compare the metric trend over time.",
    "Correlate with recent authorized change records.",
]


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Convert supported mapping-like models to a mapping."""
    if isinstance(value, Mapping):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            result = model_dump()
        except Exception:
            logger.debug("Could not serialize model input", exc_info=True)
            return None

        if isinstance(result, Mapping):
            return result

    return None


def _read(value: Any, field: str, default: Any = None) -> Any:
    """Read a field from a mapping or an object."""
    mapping = _as_mapping(value)
    if mapping is not None:
        return mapping.get(field, default)

    return getattr(value, field, default)


def _finite_number(value: Any) -> float | None:
    """Return a finite numeric value, excluding booleans and invalid values."""
    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _normalize_findings(findings: Any) -> list[Mapping[str, Any]]:
    """Keep valid finding records and ignore malformed entries."""
    if findings is None:
        return []

    if not isinstance(findings, (list, tuple)):
        logger.warning("Expected findings to be a list or tuple")
        return []

    normalized: list[Mapping[str, Any]] = []

    for index, finding in enumerate(findings):
        mapping = _as_mapping(finding)

        if mapping is None:
            logger.warning(
                "Ignoring malformed finding at index %d",
                index,
            )
            continue

        normalized.append(mapping)

    return normalized


def _metric_value(metrics: Any, metric_name: str) -> float | None:
    """Read a metric without confusing unavailable data with a true zero.

    Supports either a scalar value or a structured value such as:
        {"value": 92.0, "status": "ok"}

    Structured metrics whose status is not 'ok' are treated as unavailable.
    """
    mapping = _as_mapping(metrics)

    if mapping is None:
        return None

    if metric_name not in mapping:
        return None

    value = mapping[metric_name]

    if isinstance(value, Mapping):
        status = value.get("status")

        if status is not None and status != "ok":
            return None

        value = value.get("value")

    return _finite_number(value)


def _finding_detail(finding: Mapping[str, Any]) -> str | None:
    """Extract a concise detail string when one is available."""
    for field in ("description", "detail", "message"):
        value = finding.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _classification_is_malicious(classification: Any) -> bool:
    """Determine whether an available classifier output has a target label."""
    available = _read(classification, "available", False)
    label = _read(classification, "label", "")

    if available is not True or not isinstance(label, str):
        return False

    return label.strip().lower() in MALICIOUS_LABELS


def rank_hypotheses(
    metrics: Any,
    findings: Any,
    classification: Any,
) -> list[dict[str, Any]]:
    """Rank plausible explanations using findings and classifier output.

    Args:
        metrics: Mapping of metric names to values or structured metric records.
        findings: List of finding dictionaries or compatible model objects.
        classification: Classifier output with available, label, and confidence.

    Returns:
        Hypotheses sorted by descending heuristic confidence. Each result
        contains hypothesis, confidence, evidence, and next_checks fields.
    """
    normalized_findings = _normalize_findings(findings)

    categories = {
        category.strip()
        for finding in normalized_findings
        if isinstance(
            (category := finding.get("category")),
            str,
        )
        and category.strip()
    }

    findings_by_category: dict[str, list[Mapping[str, Any]]] = {}

    for finding in normalized_findings:
        category = finding.get("category")

        if isinstance(category, str) and category.strip():
            findings_by_category.setdefault(
                category.strip(), []
            ).append(finding)

    hypotheses: list[dict[str, Any]] = []

    for category, explanation, confidence, metric_name in HYPOTHESIS_RULES:
        if category not in categories:
            continue

        evidence: list[str] = [
            f"A finding was recorded in category '{category}'."
        ]

        observed_value = _metric_value(metrics, metric_name)

        if observed_value is not None:
            evidence.append(
                f"Observed {metric_name} value: {observed_value:g}."
            )
        else:
            evidence.append(
                f"The corresponding metric '{metric_name}' "
                "was unavailable or could not be validated."
            )

        for finding in findings_by_category.get(category, []):
            detail = _finding_detail(finding)

            if detail and detail not in evidence:
                evidence.append(detail)

        hypotheses.append(
            {
                "hypothesis": explanation,
                "confidence": confidence,
                "evidence": evidence,
                "next_checks": list(DEFAULT_NEXT_CHECKS),
            }
        )

    if _classification_is_malicious(classification):
        classifier_confidence = _finite_number(
            _read(classification, "confidence", 0.0)
        )

        if classifier_confidence is None:
            classifier_confidence = 0.0

        # Keep the score bounded, but do not interpret it as a calibrated
        # probability that the host is compromised.
        classifier_confidence = min(
            0.95,
            max(0.0, classifier_confidence),
        )

        hypotheses.append(
            {
                "hypothesis": (
                    "Text resembles a class labeled as malicious "
                    "by the configured classifier."
                ),
                "confidence": classifier_confidence,
                "evidence": [
                    "The text classifier returned a malicious-category label."
                ],
                "next_checks": [
                    "Validate the classification using independent "
                    "endpoint or network evidence.",
                    "Request analyst confirmation before taking action.",
                ],
            }
        )

    # Python's sort is stable, so equal scores preserve rule/insertion order.
    return sorted(
        hypotheses,
        key=lambda item: item["confidence"],
        reverse=True,
    )
