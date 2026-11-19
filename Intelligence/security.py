
from __future__ import annotations

import json
import math
import sqlite3
import statistics

from datetime import datetime, timezone

from ..configuration import DB_PATH


FEATURES = (
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "load_1m",
    "network_connections",
)


def metric_value(metrics, name):
    record = metrics.get(name)

    if not isinstance(record, dict):
        return None

    if record.get("status") != "ok":
        return None

    value = record.get("value")

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return None

    return float(value)


def load_history(host, agent_id, limit=300):
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row

        rows = db.execute("""
            SELECT id, timestamp, metrics_json
            FROM telemetry
            WHERE host = ? AND agent_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        """, (host, agent_id, limit)).fetchall()

    return [
        {
            "telemetry_id": row["id"],
            "timestamp": row["timestamp"],
            "metrics": json.loads(row["metrics_json"])
        }
        for row in reversed(rows)
    ]


def robust_z_score(value, baseline):
    if len(baseline) < 5:
        return None

    median = statistics.median(baseline)

    mad = statistics.median(
        abs(x - median) for x in baseline
    )

    if mad == 0:
        # A constant baseline cannot be divided by zero.
        scale = statistics.pstdev(baseline)

        if scale == 0:
            return 0.0 if value == median else float("inf")

        return abs(value - median) / scale

    return abs(value - median) / (1.4826 * mad)


def detect_temporal(history, min_baseline=10, threshold=4.0):
    findings = []

    if len(history) < min_baseline + 1:
        return findings

    current = history[-1]

    for feature in FEATURES:
        current_value = metric_value(
            current["metrics"], feature
        )

        if current_value is None:
            continue

        baseline = []

        for sample in history[:-1]:
            value = metric_value(
                sample["metrics"], feature
            )

            if value is not None:
                baseline.append(value)

        if len(baseline) < min_baseline:
            continue

        score = robust_z_score(current_value, baseline)

        if score is None or score < threshold:
            continue

        findings.append({
            "category": "temporal_anomaly",
            "severity": "high" if score >= 6 else "medium",
            "feature": feature,
            "observed_value": current_value,
            "baseline_median": statistics.median(baseline),
            "robust_deviation": (
                score if math.isfinite(score) else 1_000_000
            ),
            "threshold": threshold,
            "telemetry_id": current["telemetry_id"],
            "timestamp": current["timestamp"],
            "evidence": {
                "baseline_samples": len(baseline),
                "method": "median_absolute_deviation"
            }
        })

    return findings


def detect_multivariate(
    history,
    min_samples=20,
    threshold=4.0
):
    findings = []

    if len(history) < min_samples + 1:
        return findings

    current = history[-1]

    baseline_vectors = []

    for sample in history[:-1]:
        vector = [
            metric_value(sample["metrics"], feature)
            for feature in FEATURES
        ]

        if all(value is not None for value in vector):
            baseline_vectors.append(vector)

    current_vector = [
        metric_value(current["metrics"], feature)
        for feature in FEATURES
    ]

    if (
        len(baseline_vectors) < min_samples
        or any(value is None for value in current_vector)
    ):
        return findings

    deviations = []

    for i, feature in enumerate(FEATURES):
        column = [row[i] for row in baseline_vectors]

        median = statistics.median(column)

        mad = statistics.median(
            abs(x - median) for x in column
        )

        scale = max(1.4826 * mad, 1e-6)

        deviations.append(
            abs(current_vector[i] - median) / scale
        )

    score = math.sqrt(
        sum(x * x for x in deviations)
    )

    if score >= threshold:
        findings.append({
            "category": "multivariate_anomaly",
            "severity": "high" if score >= 8 else "medium",
            "score": score,
            "threshold": threshold,
            "features": list(FEATURES),
            "feature_deviations": dict(
                zip(FEATURES, deviations)
            ),
            "telemetry_id": current["telemetry_id"],
            "timestamp": current["timestamp"],
            "evidence": {
                "baseline_samples": len(baseline_vectors),
                "method": "joint_robust_deviation"
            }
        })

    return findings


def detect_sudden_change(history, threshold=3.0):
    findings = []

    if len(history) < 6:
        return findings

    previous = history[-2]
    current = history[-1]

    for feature in FEATURES:
        old = metric_value(previous["metrics"], feature)
        new = metric_value(current["metrics"], feature)

        if old is None or new is None:
            continue

        delta = new - old

        if old != 0:
            relative_change = abs(delta) / abs(old)
        else:
            relative_change = abs(delta)

        if relative_change >= threshold:
            findings.append({
                "category": "sudden_change",
                "severity": "medium",
                "feature": feature,
                "previous_value": old,
                "current_value": new,
                "change": delta,
                "telemetry_id": current["telemetry_id"],
                "timestamp": current["timestamp"],
            })

    return findings


def persist_findings(findings, agent_id, host):
    created = []

    with sqlite3.connect(DB_PATH) as db:
        for finding in findings:
            evidence_id = (
                f"{finding['telemetry_id']}:"
                f"{finding['category']}:"
                f"{finding.get('feature', 'joint')}"
            )

            db.execute("""
                CREATE TABLE IF NOT EXISTS intelligence_findings (
                    evidence_id TEXT PRIMARY KEY,
                    telemetry_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    host TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    detail_json TEXT NOT NULL
                )
            """)

            cursor = db.execute("""
                INSERT OR IGNORE INTO intelligence_findings (
                    evidence_id,
                    telemetry_id,
                    timestamp,
                    agent_id,
                    host,
                    category,
                    severity,
                    detail_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evidence_id,
                finding["telemetry_id"],
                finding["timestamp"],
                agent_id,
                host,
                finding["category"],
                finding["severity"],
                json.dumps(finding, allow_nan=False)
            ))

            if cursor.rowcount:
                created.append(evidence_id)

    return created


def analyze_host(host, agent_id):
    history = load_history(host, agent_id)

    findings = (
        detect_temporal(history)
        + detect_multivariate(history)
        + detect_sudden_change(history)
    )

    persisted = persist_findings(
        findings, agent_id, host
    )

    return {
        "host": host,
        "agent_id": agent_id,
        "samples_analyzed": len(history),
        "findings": findings,
        "persisted_finding_ids": persisted
    }
