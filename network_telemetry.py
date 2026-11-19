
"""Evidence-based behavioral anomaly and security-indicator detection.

This module detects suspicious conditions; it does not prove malicious intent.
It does not execute remediation or modify the monitored system.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from statistics import median
from typing import Any, Iterable


DEFAULT_THRESHOLDS = {
    "cpu_percent": 90.0,
    "memory_percent": 90.0,
    "disk_percent": 95.0,
}

DEFAULT_CHANGE_THRESHOLDS = {
    "cpu_percent": 25.0,
    "memory_percent": 20.0,
    "disk_percent": 15.0,
    "load_1m": 3.0,
}

DEFAULT_SECURITY_EVENT_TYPES = {
    "malware_detected",
    "intrusion_detected",
    "indicator_match",
    "suspicious_dns",
    "unauthorized_access",
    "brute_force_detected",
}


def _timestamp(value: Any) -> datetime | None:
    """Parse a timestamp; return None when it is missing or invalid."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _finite_number(value: Any) -> float | None:
    """Accept real finite numeric values, excluding booleans and nulls."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _host_key(sample: dict[str, Any]) -> tuple[str, str]:
    return (
        str(sample.get("agent_id", "unknown-agent")),
        str(sample.get("host", "unknown-host")),
    )


def _finding(
    *,
    detector: str,
    category: str,
    severity: str,
    title: str,
    evidence: dict[str, Any],
    timestamp: str | None = None,
    agent_id: str | None = None,
    host: str | None = None,
) -> dict[str, Any]:
    return {
        "detector": detector,
        "category": category,
        "severity": severity,
        "title": title,
        "timestamp": timestamp,
        "agent_id": agent_id,
        "host": host,
        "evidence": evidence,
        "interpretation": (
            "This finding identifies an observed condition for investigation; "
            "it does not independently establish malicious activity."
        ),
        "action_execution": "disabled",
    }


def detect_threshold_violations(
    samples: Iterable[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Detect valid metric values strictly above configured upper limits."""
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        limits.update(thresholds)

    findings = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        metrics = sample.get("metrics")
        if not isinstance(metrics, dict):
            continue

        for metric, limit in limits.items():
            value = _finite_number(metrics.get(metric))
            valid_limit = _finite_number(limit)
            if value is None or valid_limit is None:
                continue

            # Percentages must be physically valid before threshold comparison.
            if metric.endswith("_percent") and not 0.0 <= value <= 100.0:
                continue

            if value > valid_limit:
                findings.append(
                    _finding(
                        detector="threshold_v1",
                        category="resource_threshold",
                        severity="high" if value >= valid_limit + 5 else "medium",
                        title=f"{metric} exceeded its configured threshold",
                        timestamp=sample.get("timestamp"),
                        agent_id=sample.get("agent_id"),
                        host=sample.get("host"),
                        evidence={
                            "metric": metric,
                            "observed_value": value,
                            "threshold": valid_limit,
                            "excess": value - valid_limit,
                        },
                    )
                )
    return findings


def detect_temporal_changes(
    samples: Iterable[dict[str, Any]],
    change_thresholds: dict[str, float] | None = None,
    *,
    max_gap_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    """Detect large successive changes per agent and host.

    Samples with invalid timestamps are excluded. A change is not calculated
    across an excessive time gap or between different hosts.
    """
    limits = dict(DEFAULT_CHANGE_THRESHOLDS)
    if change_thresholds is not None:
        limits.update(change_thresholds)

    grouped: dict[tuple[str, str], list[tuple[datetime, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        timestamp = _timestamp(sample.get("timestamp"))
        if timestamp is not None:
            grouped[_host_key(sample)].append((timestamp, sample))

    findings = []
    for (agent_id, host), records in grouped.items():
        records.sort(key=lambda item: item[0])

        for (previous_time, previous), (current_time, current) in zip(
            records, records[1:]
        ):
            gap = (current_time - previous_time).total_seconds()
            if gap <= 0 or gap > max_gap_seconds:
                continue

            previous_metrics = previous.get("metrics", {})
            current_metrics = current.get("metrics", {})
            if not isinstance(previous_metrics, dict) or not isinstance(
                current_metrics, dict
            ):
                continue

            for metric, limit in limits.items():
                before = _finite_number(previous_metrics.get(metric))
                after = _finite_number(current_metrics.get(metric))
                valid_limit = _finite_number(limit)
                if before is None or after is None or valid_limit is None:
                    continue
                if metric.endswith("_percent") and not (
                    0 <= before <= 100 and 0 <= after <= 100
                ):
                    continue

                change = after - before
                if abs(change) < valid_limit:
                    continue

                findings.append(
                    _finding(
                        detector="temporal_change_v1",
                        category="abrupt_metric_change",
                        severity="high" if abs(change) >= 2 * valid_limit else "medium",
                        title=f"Unusual change in {metric}",
                        timestamp=current.get("timestamp"),
                        agent_id=agent_id,
                        host=host,
                        evidence={
                            "metric": metric,
                            "previous_value": before,
                            "current_value": after,
                            "change": change,
                            "absolute_change": abs(change),
                            "elapsed_seconds": gap,
                            "change_per_second": change / gap,
                            "configured_change_threshold": valid_limit,
                        },
                    )
                )
    return findings


def detect_persistent_violations(
    samples: Iterable[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
    *,
    min_samples: int = 3,
    persistence_fraction: float = 0.6,
) -> list[dict[str, Any]]:
    """Report sustained threshold breaches for each host and metric."""
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    if not 0.0 < persistence_fraction <= 1.0:
        raise ValueError("persistence_fraction must be in (0, 1]")

    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        limits.update(thresholds)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if isinstance(sample, dict):
            grouped[_host_key(sample)].append(sample)

    findings = []
    for (agent_id, host), records in grouped.items():
        for metric, limit in limits.items():
            valid_limit = _finite_number(limit)
            if valid_limit is None:
                continue

            observed = []
            for record in records:
                metrics = record.get("metrics", {})
                value = (
                    _finite_number(metrics.get(metric))
                    if isinstance(metrics, dict)
                    else None
                )
                if value is not None and (
                    not metric.endswith("_percent") or 0 <= value <= 100
                ):
                    observed.append((record, value))

            if len(observed) < min_samples:
                continue

            breached = [(record, value) for record, value in observed if value > valid_limit]
            fraction = len(breached) / len(observed)

            if len(breached) >= min_samples and fraction >= persistence_fraction:
                last_record, _ = breached[-1]
                findings.append(
                    _finding(
                        detector="persistence_v1",
                        category="persistent_resource_pressure",
                        severity="high",
                        title=f"Persistent threshold violations for {metric}",
                        timestamp=last_record.get("timestamp"),
                        agent_id=agent_id,
                        host=host,
                        evidence={
                            "metric": metric,
                            "threshold": valid_limit,
                            "valid_sample_count": len(observed),
                            "violation_count": len(breached),
                            "persistence_fraction": fraction,
                            "required_fraction": persistence_fraction,
                        },
                    )
                )
    return findings


def detect_multivariate_deviations(
    samples: Iterable[dict[str, Any]],
    baseline_samples: Iterable[dict[str, Any]],
    *,
    features: tuple[str, ...] = (
        "cpu_percent",
        "memory_percent",
        "disk_percent",
        "load_1m",
    ),
    minimum_features: int = 3,
    robust_z_threshold: float = 3.5,
) -> list[dict[str, Any]]:
    """Detect robust deviations from a supplied baseline using median/MAD.

    This is a transparent heuristic, not a trained intrusion classifier.
    Features with no usable baseline or zero median absolute deviation are
    omitted rather than assigned an invented scale.
    """
    if minimum_features < 1 or minimum_features > len(features):
        raise ValueError("minimum_features must be between 1 and len(features)")
    if robust_z_threshold <= 0 or not math.isfinite(robust_z_threshold):
        raise ValueError("robust_z_threshold must be finite and positive")

    baseline_rows = [
        row for row in baseline_samples
        if isinstance(row, dict) and isinstance(row.get("metrics"), dict)
    ]
    current_rows = [
        row for row in samples
        if isinstance(row, dict) and isinstance(row.get("metrics"), dict)
    ]

    statistics: dict[str, tuple[float, float]] = {}
    for feature in features:
        values = [
            number
            for row in baseline_rows
            if (number := _finite_number(row["metrics"].get(feature))) is not None
        ]
        if len(values) < 3:
            continue
        center = median(values)
        mad = median([abs(value - center) for value in values])
        # Zero MAD means the baseline cannot provide a robust scale estimate.
        if mad > 0 and math.isfinite(mad):
            statistics[feature] = (center, mad)

    findings = []
    for sample in current_rows:
        deviations = {}
        for feature, (center, mad) in statistics.items():
            value = _finite_number(sample["metrics"].get(feature))
            if value is None:
                continue
            # 0.67449 scales MAD-based deviation to a normal-distribution z score.
            robust_z = 0.67449 * (value - center) / mad
            if abs(robust_z) >= robust_z_threshold:
                deviations[feature] = {
                    "value": value,
                    "baseline_median": center,
                    "baseline_mad": mad,
                    "robust_z_score": robust_z,
                }

        if len(deviations) >= minimum_features:
            findings.append(
                _finding(
                    detector="multivariate_robust_v1",
                    category="multivariate_baseline_deviation",
                    severity="high",
                    title="Multiple metrics deviate from the supplied baseline",
                    timestamp=sample.get("timestamp"),
                    agent_id=sample.get("agent_id"),
                    host=sample.get("host"),
                    evidence={
                        "deviating_features": deviations,
                        "deviating_feature_count": len(deviations),
                        "minimum_features": minimum_features,
                        "robust_z_threshold": robust_z_threshold,
                        "baseline_sample_count": len(baseline_rows),
                    },
                )
            )
    return findings


def detect_security_indicators(
    events: Iterable[dict[str, Any]],
    *,
    known_indicators: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Flag explicit security alerts and exact matches to configured indicators.

    Event schemas vary by source. This function only inspects fields that are
    explicitly present; it does not infer missing network or process evidence.
    """
    known_indicators = known_indicators or {}
    supported_fields = ("src_ip", "dst_ip", "domain", "sha256", "process_name")
    findings = []

    for event in events:
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("event_type", "")).strip().lower()
        matched = []

        if event_type in DEFAULT_SECURITY_EVENT_TYPES:
            matched.append({
                "match_type": "explicit_security_event",
                "event_type": event_type,
            })

        for field in supported_fields:
            value = event.get(field)
            configured = known_indicators.get(field, set())
            if isinstance(value, str) and value and value in configured:
                matched.append({
                    "match_type": "configured_indicator_match",
                    "field": field,
                    "matched_value": value,
                })

        if not matched:
            continue

        severity = str(event.get("severity", "")).lower()
        if severity not in {"low", "medium", "high", "critical"}:
            severity = (
                "critical"
                if event_type in {"malware_detected", "intrusion_detected"}
                else "high"
            )

        # Keep the event's relevant evidence; avoid claiming that an indicator
        # match independently proves a compromise.
        evidence = {
            "event_type": event_type or None,
            "source": event.get("source"),
            "src_ip": event.get("src_ip"),
            "dst_ip": event.get("dst_ip"),
            "domain": event.get("domain"),
            "sha256": event.get("sha256"),
            "process_name": event.get("process_name"),
            "matched_indicators": matched,
        }

        findings.append(
            _finding(
                detector="security_indicator_v1",
                category="security_indicator",
                severity=severity,
                title="Security event or configured indicator requires review",
                timestamp=event.get("timestamp"),
                agent_id=event.get("agent_id"),
                host=event.get("host"),
                evidence=evidence,
            )
        )

    return findings


def detect_behavior(
    samples: Iterable[dict[str, Any]],
    *,
    baseline_samples: Iterable[dict[str, Any]] | None = None,
    security_events: Iterable[dict[str, Any]] | None = None,
    thresholds: dict[str, float] | None = None,
    change_thresholds: dict[str, float] | None = None,
    known_indicators: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Run all available detectors and return an auditable result."""
    sample_list = list(samples)
    baseline_list = list(baseline_samples or [])
    event_list = list(security_events or [])

    findings = []
    findings.extend(detect_threshold_violations(sample_list, thresholds))
    findings.extend(detect_temporal_changes(sample_list, change_thresholds))
    findings.extend(detect_persistent_violations(sample_list, thresholds))

    if baseline_list:
        findings.extend(
            detect_multivariate_deviations(sample_list, baseline_list)
        )

    if event_list:
        findings.extend(
            detect_security_indicators(
                event_list,
                known_indicators=known_indicators,
            )
        )

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(
        key=lambda item: (
            severity_order.get(item["severity"], 4),
            item.get("timestamp") or "",
            item["detector"],
        )
    )

    return {
        "detector": "behavior_detection_v1",
        "sample_count": len(sample_list),
        "baseline_sample_count": len(baseline_list),
        "security_event_count": len(event_list),
        "finding_count": len(findings),
        "findings": findings,
        "action_execution": "disabled",
        "interpretation": (
            "Findings are evidence for investigation, not proof of malicious "
            "activity. Detection quality must be measured against labeled data."
        ),
    }
