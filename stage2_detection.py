"""Stage 2 time-window and multivariate telemetry detection.

A finding is an investigation signal, not proof of malicious activity.

Feature policy:
- Only finite numeric values explicitly marked available are measurements.
- Missing values are never imputed as zero.
- Time-window calculations use actual timestamps, not assumed sample spacing.
- Rates are not computed across excessive gaps or non-positive time deltas.
- Multivariate scores use robust median/MAD normalization.
- A score is not a calibrated probability of intrusion.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from statistics import median
from typing import Any, Iterable


DEFAULT_METRICS = (
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "load_1m",
    "network_bytes_per_sec",
    "network_packets_per_sec",)

DEFAULT_THRESHOLDS = {
    "cpu_percent": 90.0,
    "memory_percent": 90.0,
    "disk_percent": 95.0,}

# Conservative defaults. Configure these for the collection interval and host.
DEFAULT_WINDOW_SECONDS = 300.0
DEFAULT_MIN_SAMPLES = 3
DEFAULT_MAX_GAP_SECONDS = 120.0
DEFAULT_PERSISTENCE_FRACTION = 0.60
DEFAULT_REPEATED_ANOMALY_MINIMUM = 3
DEFAULT_RATE_THRESHOLDS_PER_SECOND = {
    "cpu_percent": 0.50,
    "memory_percent": 0.50,
    "disk_percent": 0.25,}

def _timestamp(value: Any) -> datetime | None:
    """Parse a timezone-aware datetime or ISO-8601 timestamp."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None

    return parsed.astimezone(timezone.utc)


def _metric_value(metrics: Any, key: str) -> float | None:
    """Read finite available metrics, including defined network-rate aliases."""
    if not isinstance(metrics, dict):
        return None

    raw = metrics.get(key)

    # Derive total network throughput only when both directional rates exist.
    # Do not interpret an unavailable direction as zero.
    if key in {"network_bytes_per_sec", "network_packets_per_sec"}:
        network_io = metrics.get("network_io")
        if isinstance(network_io, dict) and network_io.get("status", "ok") == "ok":
            rates = network_io.get("rates")
            if isinstance(rates, dict):
                if key == "network_bytes_per_sec":
                    directional = (
                        rates.get("bytes_sent_per_sec"),
                        rates.get("bytes_recv_per_sec"),
                    )
                else:
                    directional = (
                        rates.get("packets_sent_per_sec"),
                        rates.get("packets_recv_per_sec"),
                    )

                if all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and value >= 0
                    for value in directional
                ):
                    return float(directional[0]) + float(directional[1])

    if key not in metrics:
        return None

    raw = metrics[key]

    if isinstance(raw, dict):
        if raw.get("status", "ok") != "ok":
            return None
        raw = raw.get("value")

    if raw is None or isinstance(raw, bool):
        return None

    if not isinstance(raw, (int, float)):
        return None

    value = float(raw)
    return value if math.isfinite(value) else None


def _normalise_sample(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the identity and timestamp of a telemetry sample."""
    if not isinstance(sample, dict):
        return None

    host = sample.get("host")
    timestamp = _timestamp(sample.get("timestamp"))
    metrics = sample.get("metrics")

    if not isinstance(host, str) or not host.strip():
        return None
    if timestamp is None or not isinstance(metrics, dict):
        return None

    return {
        **sample,
        "host": host.strip(),
        "_timestamp": timestamp,
        "metrics": metrics,}


def detect_time_window(
    samples: Iterable[dict[str, Any]],
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    persistence_fraction: float = DEFAULT_PERSISTENCE_FRACTION,
    repeated_anomaly_minimum: int = DEFAULT_REPEATED_ANOMALY_MINIMUM,
    thresholds: dict[str, float] | None = None,
    rate_thresholds_per_second: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate persistence, rate of change, and repeated threshold anomalies.

    The analysis is performed separately for each host and uses a trailing
    window ending at that host's latest valid sample.

    Missing samples are counted as missing, not normal. Large sampling gaps
    break rate-of-change sequences. A finding includes coverage metadata so
    sparse data cannot silently look like a clean window.
    """
    for name, value in (
        ("window_seconds", window_seconds),
        ("max_gap_seconds", max_gap_seconds),
        ("persistence_fraction", persistence_fraction),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    if window_seconds <= 0 or max_gap_seconds <= 0:
        raise ValueError("window_seconds and max_gap_seconds must be positive")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    if not 0 < persistence_fraction <= 1:
        raise ValueError("persistence_fraction must be in (0, 1]")
    if repeated_anomaly_minimum < 2:
        raise ValueError("repeated_anomaly_minimum must be at least 2")

    active_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        active_thresholds.update(thresholds)

    active_rate_thresholds = dict(DEFAULT_RATE_THRESHOLDS_PER_SECOND)
    if rate_thresholds_per_second:
        active_rate_thresholds.update(rate_thresholds_per_second)

    grouped: dict[str, list[dict[str, Any]]] = {}

    for raw_sample in samples:
        sample = _normalise_sample(raw_sample)
        if sample is None:
            continue
        grouped.setdefault(sample["host"], []).append(sample)

    findings: list[dict[str, Any]] = []

    for host, host_samples in grouped.items():
        host_samples.sort(key=lambda row: row["_timestamp"])
        end = host_samples[-1]["_timestamp"]
        start = end.timestamp() - window_seconds

        window = [
            row for row in host_samples
            if start <= row["_timestamp"].timestamp() <= end.timestamp()
        ]

        if not window:
            continue

        for metric, threshold in active_thresholds.items():
            if not math.isfinite(float(threshold)):
                raise ValueError(f"Threshold for {metric} must be finite")

            measured = [
                (row["_timestamp"], _metric_value(row["metrics"], metric))
                for row in window
            ]
            available = [(ts, value) for ts, value in measured if value is not None]
            exceedances = [
                (ts, value) for ts, value in available if value >= threshold
            ]

            coverage = len(available) / len(window)
            persistence = (
                len(exceedances) / len(available) if available else None
            )

            if len(available) >= min_samples and persistence is not None:
                if (
                    len(exceedances) >= min_samples
                    and persistence >= persistence_fraction
                ):
                    findings.append({
                        "category": "persistent_threshold",
                        "severity": "high" if persistence >= 0.90 else "medium",
                        "description": (
                            f"{metric} remained above its configured threshold "
                            "for a substantial fraction of the observed window."
                        ),
                        "host": host,
                        "evidence": {
                            "metric": metric,
                            "threshold": threshold,
                            "window_seconds": window_seconds,
                            "window_start": window[0]["_timestamp"].isoformat(),
                            "window_end": end.isoformat(),
                            "available_samples": len(available),
                            "window_samples": len(window),
                            "coverage_fraction": coverage,
                            "exceedance_count": len(exceedances),
                            "persistence_fraction": persistence,
                            "rule": "time_window_persistence_v1",
                        },
                        "interpretation": (
                            "Persistent resource anomaly; investigate workload, "
                            "configuration, and security context."
                        ),
                    })

            # Count separate high-threshold runs. A below-threshold observation
            # resets the run. A gap larger than max_gap_seconds also resets it.
            runs: list[list[tuple[datetime, float]]] = []
            current: list[tuple[datetime, float]] = []
            previous_ts: datetime | None = None

            for ts, value in measured:
                if value is None:
                    if current:
                        runs.append(current)
                        current = []
                    previous_ts = ts
                    continue

                gap = (
                    (ts - previous_ts).total_seconds()
                    if previous_ts is not None else None
                )

                if gap is not None and gap > max_gap_seconds:
                    if current:
                        runs.append(current)
                    current = []

                if value >= threshold:
                    current.append((ts, value))
                else:
                    if current:
                        runs.append(current)
                    current = []

                previous_ts = ts

            if current:
                runs.append(current)

            repeated = [run for run in runs if run]
            if len(repeated) >= repeated_anomaly_minimum:
                findings.append({
                    "category": "repeated_threshold_anomaly",
                    "severity": "medium",
                    "description": (
                        f"{metric} crossed its threshold repeatedly within "
                        "the configured observation window."
                    ),
                    "host": host,
                    "evidence": {
                        "metric": metric,
                        "threshold": threshold,
                        "window_seconds": window_seconds,
                        "repeated_runs": len(repeated),
                        "run_lengths": [len(run) for run in repeated],
                        "max_gap_seconds": max_gap_seconds,
                        "rule": "repeated_threshold_runs_v1",
                    },
                    "interpretation": (
                        "Repeated resource anomaly; recurrence alone does not "
                        "establish an attack."
                    ),
                })

            # Rates use consecutive available values only when their timestamp
            # gap is positive and no greater than max_gap_seconds.
            rate_limit = active_rate_thresholds.get(metric)
            if rate_limit is not None:
                if not math.isfinite(float(rate_limit)) or rate_limit < 0:
                    raise ValueError(
                        f"Rate threshold for {metric} must be finite and non-negative"
                    )

                rate_events = []
                previous: tuple[datetime, float] | None = None

                for ts, value in measured:
                    if value is None:
                        previous = None
                        continue

                    if previous is not None:
                        prev_ts, prev_value = previous
                        dt = (ts - prev_ts).total_seconds()

                        if 0 < dt <= max_gap_seconds:
                            rate = (value - prev_value) / dt
                            if abs(rate) >= rate_limit:
                                rate_events.append({
                                    "start": prev_ts.isoformat(),
                                    "end": ts.isoformat(),
                                    "rate_per_second": rate,
                                    "elapsed_seconds": dt,
                                })
                    previous = (ts, value)

                if rate_events:
                    findings.append({
                        "category": "rapid_metric_change",
                        "severity": "medium",
                        "description": (
                            f"{metric} changed rapidly across one or more "
                            "valid adjacent samples."
                        ),
                        "host": host,
                        "evidence": {
                            "metric": metric,
                            "absolute_rate_threshold_per_second": rate_limit,
                            "max_gap_seconds": max_gap_seconds,
                            "events": rate_events,
                            "rule": "time_window_rate_v1",
                        },
                        "interpretation": (
                            "Rapid change may reflect legitimate workload changes, "
                            "measurement noise, or an event needing investigation."
                        ),
                    })

        # Explicitly surface poor data coverage. Do not label missing telemetry
        # as normal or silently substitute values.
        for metric in set(active_thresholds) | set(active_rate_thresholds):
            available_count = sum(
                _metric_value(row["metrics"], metric) is not None
                for row in window
            )
            if available_count < min_samples:
                findings.append({
                    "category": "insufficient_window_data",
                    "severity": "low",
                    "description": (
                        f"Insufficient available samples to evaluate {metric} "
                        "reliably over the configured window."
                    ),
                    "host": host,
                    "evidence": {
                        "metric": metric,
                        "window_seconds": window_seconds,
                        "available_samples": available_count,
                        "window_samples": len(window),
                        "required_samples": min_samples,
                        "rule": "time_window_coverage_v1",
                    },
                    "interpretation": "Detection coverage is limited.",
                })

    return findings


def _robust_z(value: float, center: float, scale: float) -> float:
    """Robust standardized distance using median absolute deviation."""
    if scale <= 0 or not math.isfinite(scale):
        return 0.0 if value == center else math.copysign(0.0, value - center)
    return (value - center) / scale


def detect_multivariate(
    samples: Iterable[dict[str, Any]],
    *,
    baseline_samples: Iterable[dict[str, Any]] | None = None,
    features: tuple[str, ...] = DEFAULT_METRICS,
    minimum_features: int = 3,
    score_threshold: float = 3.5,
) -> list[dict[str, Any]]:
    """Detect unusual combinations of available host metrics.

    Baseline values determine per-feature median and MAD normalization. If no
    baseline is supplied, the supplied samples are used to estimate one; this
    is suitable for exploratory checks but not an unbiased evaluation.

    Missing values are excluded per feature. A sample is scored only if at
    least minimum_features are available. Features are not treated as
    independent attack indicators, and the score is not a probability.
    """
    if minimum_features < 2 or minimum_features > len(features):
        raise ValueError("minimum_features must be between 2 and feature count")
    if not math.isfinite(score_threshold) or score_threshold <= 0:
        raise ValueError("score_threshold must be finite and positive")

    valid_samples = [
        sample for sample in (_normalise_sample(x) for x in samples)
        if sample is not None
    ]
    baseline_source = baseline_samples if baseline_samples is not None else valid_samples
    valid_baseline = [
        sample for sample in (_normalise_sample(x) for x in baseline_source)
        if sample is not None
    ]

    if len(valid_baseline) < 3:
        return [{
            "category": "multivariate_baseline_unavailable",
            "severity": "low",
            "description": "Not enough baseline samples for robust normalization.",
            "evidence": {
                "baseline_samples": len(valid_baseline),
                "required_baseline_samples": 3,
                "rule": "multivariate_robust_score_v1",
            },
            "interpretation": "No multivariate anomaly conclusion was produced.",
        }]

    centers: dict[str, float] = {}
    scales: dict[str, float] = {}

    for feature in features:
        values = [
            value for row in valid_baseline
            if (value := _metric_value(row["metrics"], feature)) is not None
        ]
        if len(values) < 3:
            continue

        center = median(values)
        mad = median([abs(value - center) for value in values])
        # 1.4826 scales MAD toward the standard deviation for Gaussian data.
        robust_scale = 1.4826 * mad

        # If the baseline is constant, use a tiny numerical floor only to avoid
        # division by zero; it does not imply the feature has normal variance.
        centers[feature] = center
        scales[feature] = robust_scale

    findings: list[dict[str, Any]] = []

    for row in valid_samples:
        available_scores: dict[str, float] = {}

        for feature in features:
            if feature not in centers:
                continue
            value = _metric_value(row["metrics"], feature)
            if value is None:
                continue

            scale = scales[feature]
            if scale == 0:
                score = 0.0 if value == centers[feature] else score_threshold + 1.0
            else:
                score = _robust_z(value, centers[feature], scale)

            available_scores[feature] = score

        if len(available_scores) < minimum_features:
            findings.append({
                "category": "insufficient_multivariate_features",
                "severity": "low",
                "description": "Not enough available metrics for multivariate scoring.",
                "host": row["host"],
                "evidence": {
                    "available_features": sorted(available_scores),
                    "required_features": minimum_features,
                    "rule": "multivariate_robust_score_v1",
                },
                "interpretation": "No multivariate anomaly conclusion was produced.",
            })
            continue

        # Root-mean-square standardized deviation is an interpretable anomaly
        # score. Correlated features can contribute redundantly, so this score
        # must not be read as independent evidence or as a probability.
        score = math.sqrt(
            sum(value * value for value in available_scores.values())
            / len(available_scores)
        )

        if score >= score_threshold:
            findings.append({
                "category": "multivariate_resource_anomaly",
                "severity": "high" if score >= score_threshold * 1.5 else "medium",
                "description": (
                    "The available resource measurements jointly deviate from "
                    "the configured baseline."
                ),
                "host": row["host"],
                "evidence": {
                    "score": score,
                    "score_threshold": score_threshold,
                    "feature_scores": available_scores,
                    "feature_count": len(available_scores),
                    "features_used": sorted(available_scores),
                    "baseline_centers": {
                        feature: centers[feature]
                        for feature in available_scores
                    },
                    "normalization": "median_and_scaled_MAD",
                    "rule": "multivariate_robust_score_v1",
                },
                "interpretation": (
                    "Joint resource anomaly; correlated symptoms are not "
                    "independent proof of an intrusion."
                ),
            })

    return findings
