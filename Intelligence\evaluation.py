
from __future__ import annotations

import statistics

from .detection import FEATURES, metric_value


def trend(history, feature, window=10):
    values = []

    for sample in history[-window:]:
        value = metric_value(sample["metrics"], feature)

        if value is not None:
            values.append(value)

    if len(values) < 3:
        return None

    return values[-1] - values[0]


def rank_root_causes(history, findings):
    if not history:
        return []

    current = history[-1]["metrics"]

    trends = {
        feature: trend(history, feature)
        for feature in FEATURES
    }

    categories = {
        finding.get("feature")
        for finding in findings
    }

    hypotheses = []

    # Memory pressure.
    memory = metric_value(current, "memory_percent")
    cpu = metric_value(current, "cpu_percent")
    load = metric_value(current, "load_1m")

    if memory is not None:
        score = 0.0
        supporting = []
        contradicting = []

        if memory >= 85:
            score += 0.35
            supporting.append("High measured memory utilization")

        if trends.get("memory_percent") is not None:
            if trends["memory_percent"] > 10:
                score += 0.25
                supporting.append("Memory utilization is increasing")

        if cpu is not None and cpu >= 80:
            score += 0.15
            supporting.append("CPU pressure is concurrent")

        if memory < 60:
            contradicting.append(
                "Current memory utilization is not elevated"
            )

        hypotheses.append({
            "cause": "memory_pressure",
            "score": score,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting
        })

    # CPU saturation.
    if cpu is not None:
        score = 0.0
        supporting = []
        contradicting = []

        if cpu >= 90:
            score += 0.4
            supporting.append("CPU utilization exceeds 90%")

        if trends.get("cpu_percent") is not None:
            if trends["cpu_percent"] > 15:
                score += 0.25
                supporting.append("CPU utilization is rapidly increasing")

        if load is not None and load >= 4:
            score += 0.2
            supporting.append("Elevated system load accompanies CPU usage")

        if cpu < 50:
            contradicting.append("CPU utilization is not elevated")

        hypotheses.append({
            "cause": "cpu_saturation",
            "score": score,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting
        })

    # Disk pressure.
    disk = metric_value(current, "disk_percent")

    if disk is not None:
        score = 0.0
        supporting = []
        contradicting = []

        if disk >= 90:
            score += 0.6
            supporting.append("Filesystem utilization is critically high")

        if trends.get("disk_percent") is not None:
            if trends["disk_percent"] > 5:
                score += 0.2
                supporting.append("Disk utilization is increasing")

        if disk < 70:
            contradicting.append("Disk capacity is not currently constrained")

        hypotheses.append({
            "cause": "disk_capacity_pressure",
            "score": score,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting
        })

    # Network connection pressure.
    connections = metric_value(
        current, "network_connections"
    )

    if connections is not None:
        score = 0.0
        supporting = []
        contradicting = []

        if connections > 1000:
            score += 0.35
            supporting.append("High connection count")

        if trends.get("network_connections") is not None:
            if trends["network_connections"] > 100:
                score += 0.3
                supporting.append(
                    "Connection count is increasing"
                )

        if connections < 100:
            contradicting.append(
                "Connection count is relatively low"
            )

        hypotheses.append({
            "cause": "network_connection_pressure",
            "score": score,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting
        })

    # Explicitly preserve unknown causes.
    if not hypotheses or all(
        hypothesis["score"] == 0
        for hypothesis in hypotheses
    ):
        hypotheses.append({
            "cause": "unknown_or_unobserved_cause",
            "score": 0.0,
            "supporting_evidence": [],
            "contradicting_evidence": [],
        })

    hypotheses.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    total = sum(x["score"] for x in hypotheses)

    for item in hypotheses:
        item["relative_weight"] = (
            item["score"] / total if total else 0.0
        )

    return hypotheses
