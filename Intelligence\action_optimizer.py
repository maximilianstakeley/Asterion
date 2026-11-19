"""Asterion model-quality assurance, policy benchmarking and safe lifecycle.

This module is deliberately independent of Asterion's decision executor.
It evaluates outputs and manages model artifacts; it never executes actions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import statistics
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert numpy-like scalars and tuples to JSON-safe values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite values cannot be persisted")
    return value


# ---------------------------------------------------------------------------
# 1. Training-data integrity and supervised model evaluation
# ---------------------------------------------------------------------------

def audit_labeled_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    text_key: str = "text",
    label_key: str = "label",
    timestamp_key: str | None = None,
    minimum_per_class: int = 2,
) -> dict[str, Any]:
    """Audit labels, duplicate records, conflicting labels and class balance.

    Returns a report; conflicting duplicates and missing labels are fatal.
    Exact duplicates are counted and removed from the effective row count.
    """
    materialized = list(rows)
    if not materialized:
        raise ValueError("Training dataset is empty")

    normalized: list[tuple[str, str, str | None]] = []
    missing_text = missing_label = 0
    timestamps: list[str] = []

    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise ValueError(f"Row {index} is not a mapping")
        text = row.get(text_key)
        label = row.get(label_key)
        if text is None or not str(text).strip():
            missing_text += 1
            continue
        if label is None or not str(label).strip():
            missing_label += 1
            continue
        stamp = row.get(timestamp_key) if timestamp_key else None
        stamp_text = str(stamp) if stamp is not None else None
        normalized.append((str(text).strip(), str(label).strip(), stamp_text))
        if stamp_text:
            timestamps.append(stamp_text)

    if missing_text or missing_label:
        raise ValueError(
            f"Invalid rows: missing_text={missing_text}, "
            f"missing_label={missing_label}"
        )

    labels_by_text: dict[str, set[str]] = {}
    for text, label, _ in normalized:
        labels_by_text.setdefault(text, set()).add(label)
    conflicts = sorted(text for text, labels in labels_by_text.items() if len(labels) > 1)
    if conflicts:
        raise ValueError(
            f"Conflicting labels for {len(conflicts)} duplicate text samples"
        )

    unique: dict[tuple[str, str], str | None] = {}
    for text, label, stamp in normalized:
        unique.setdefault((text, label), stamp)

    counts: dict[str, int] = {}
    for _, label in unique:
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2:
        raise ValueError("At least two distinct classes are required")
    too_small = {label: count for label, count in counts.items()
                 if count < minimum_per_class}
    if too_small:
        raise ValueError(f"Insufficient examples per class: {too_small}")

    total = len(unique)
    return {
        "input_rows": len(materialized),
        "unique_rows": total,
        "exact_duplicate_rows": len(materialized) - total,
        "class_counts": dict(sorted(counts.items())),
        "class_proportions": {
            label: count / total for label, count in sorted(counts.items())
        },
        "minority_to_majority_ratio": min(counts.values()) / max(counts.values()),
        "timestamp_coverage": (
            len(timestamps) / len(materialized) if timestamp_key else None
        ),
        "timestamp_ordered": (
            timestamps == sorted(timestamps) if timestamp_key and timestamps else None
        ),
        "label_conflicts": 0,
        "ready_for_training": True,
    }



def chronological_split_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    timestamp_key: str = "timestamp",
    text_key: str = "text",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, Any]:
    """Split labeled records by time, keeping duplicate text in one partition.

    This is a leakage-resistant baseline for time-dependent telemetry data.
    Timestamps must be ISO-8601 strings or datetime objects with a consistent
    timezone convention. Duplicate text groups are assigned by their earliest
    timestamp so identical observations cannot appear in multiple partitions.
    """
    from datetime import datetime

    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Invalid chronological split fractions")
    materialized = list(rows)
    if len(materialized) < 3:
        raise ValueError("At least three rows are required")

    groups: dict[str, list[tuple[datetime, Mapping[str, Any]]]] = {}
    for index, row in enumerate(materialized):
        text = str(row.get(text_key, "")).strip()
        label = row.get("label")
        stamp = row.get(timestamp_key)
        if not text or label is None or not str(label).strip() or stamp is None:
            raise ValueError(f"Row {index} has missing text, label, or timestamp")
        try:
            parsed = stamp if isinstance(stamp, datetime) else datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {index} has an invalid ISO-8601 timestamp") from exc
        groups.setdefault(text, []).append((parsed, row))

    ordered_groups = sorted(
        groups.values(), key=lambda group: min(item[0] for item in group)
    )
    if len(ordered_groups) < 3:
        raise ValueError("At least three unique text groups are required")
    train_end = max(1, int(len(ordered_groups) * train_fraction))
    validation_end = max(train_end + 1, int(len(ordered_groups) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(ordered_groups) - 1)

    partitions = {
        "train": ordered_groups[:train_end],
        "validation": ordered_groups[train_end:validation_end],
        "test": ordered_groups[validation_end:],
    }
    output = {}
    for name, group_list in partitions.items():
        partition_rows = [row for group in group_list for _, row in group]
        output[name] = partition_rows
    text_sets = {name: {str(row[text_key]).strip() for row in part}
                 for name, part in output.items()}
    if (text_sets["train"] & text_sets["validation"]
            or text_sets["train"] & text_sets["test"]
            or text_sets["validation"] & text_sets["test"]):
        raise RuntimeError("Duplicate-text leakage detected across partitions")
    output["summary"] = {
        name: {
            "rows": len(part),
            "unique_text_groups": len(text_sets[name]),
            "start": min((str(row[timestamp_key]) for row in part), default=None),
            "end": max((str(row[timestamp_key]) for row in part), default=None),
        }
        for name, part in output.items()
    }
    output["leakage_check"] = "passed"
    return output


def compare_generalization(
    validation_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    *,
    maximum_f1_drop: float = 0.10,
) -> dict[str, Any]:
    """Quantify validation-to-test degradation without hiding negative gaps."""
    if not _finite(maximum_f1_drop) or maximum_f1_drop < 0:
        raise ValueError("maximum_f1_drop must be finite and >= 0")
    required = ("f1", "precision", "recall", "false_positive_rate")
    for name, metrics in (("validation", validation_metrics), ("test", test_metrics)):
        missing = [key for key in required if key not in metrics or not _finite(metrics[key])]
        if missing:
            raise ValueError(f"{name} metrics missing or invalid: {missing}")
    f1_drop = float(validation_metrics["f1"]) - float(test_metrics["f1"])
    return {
        "validation_f1": float(validation_metrics["f1"]),
        "test_f1": float(test_metrics["f1"]),
        "f1_generalization_gap": f1_drop,
        "precision_generalization_gap": float(validation_metrics["precision"]) - float(test_metrics["precision"]),
        "recall_generalization_gap": float(validation_metrics["recall"]) - float(test_metrics["recall"]),
        "false_positive_rate_change": float(test_metrics["false_positive_rate"]) - float(validation_metrics["false_positive_rate"]),
        "within_f1_tolerance": f1_drop <= maximum_f1_drop,
        "maximum_f1_drop": float(maximum_f1_drop),
    }


def evaluate_classifier(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    positive_label: Any,
    timestamps: Sequence[Any] | None = None,
    temporal_windows: int = 5,
) -> dict[str, Any]:
    """Compute binary classification metrics and temporal-window stability."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty and equal length")
    if timestamps is not None and len(timestamps) != len(y_true):
        raise ValueError("timestamps length must match labels")

    labels = set(y_true) | set(y_pred)
    if len(labels) != 2:
        raise ValueError("Binary evaluation requires exactly two observed labels")
    negative = next(label for label in labels if label != positive_label)
    if positive_label not in labels:
        raise ValueError("positive_label is absent from evaluation labels")

    tp = sum(a == positive_label and p == positive_label for a, p in zip(y_true, y_pred))
    tn = sum(a == negative and p == negative for a, p in zip(y_true, y_pred))
    fp = sum(a == negative and p == positive_label for a, p in zip(y_true, y_pred))
    fn = sum(a == positive_label and p == negative for a, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true)

    report: dict[str, Any] = {
        "sample_count": len(y_true),
        "positive_label": str(positive_label),
        "negative_label": str(negative),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "balanced_accuracy": (recall + specificity) / 2,
    }

    if timestamps is not None:
        ordered = sorted(zip(timestamps, y_true, y_pred), key=lambda item: str(item[0]))
        chunks = [chunk for chunk in _split_even(ordered, min(temporal_windows, len(ordered))) if chunk]
        window_metrics = []
        for index, chunk in enumerate(chunks):
            m = evaluate_classifier(
                [item[1] for item in chunk],
                [item[2] for item in chunk],
                positive_label=positive_label,
            )
            window_metrics.append({
                "window": index,
                "start": str(chunk[0][0]),
                "end": str(chunk[-1][0]),
                "sample_count": len(chunk),
                "precision": m["precision"],
                "recall": m["recall"],
                "false_positive_rate": m["false_positive_rate"],
                "f1": m["f1"],
            })
        f1s = [item["f1"] for item in window_metrics]
        report["temporal_stability"] = {
            "window_count": len(window_metrics),
            "windows": window_metrics,
            "f1_standard_deviation": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
            "f1_range": max(f1s) - min(f1s) if f1s else None,
        }
    return report


def _split_even(items: Sequence[Any], count: int) -> list[list[Any]]:
    return [list(items[i * len(items) // count:(i + 1) * len(items) // count])
            for i in range(count)]


# ---------------------------------------------------------------------------
# 2. Anomaly detection evaluation
# ---------------------------------------------------------------------------

def evaluate_anomaly_detector(
    y_true: Sequence[int | bool],
    anomaly_scores: Sequence[float],
    *,
    threshold: float,
    timestamps: Sequence[Any] | None = None,
    temporal_windows: int = 5,
) -> dict[str, Any]:
    """Evaluate score thresholding with explicit positive=anomaly semantics."""
    if len(y_true) != len(anomaly_scores) or len(y_true) == 0:
        raise ValueError("Labels and anomaly scores must be non-empty and equal length")
    if not _finite(threshold):
        raise ValueError("threshold must be finite")
    if timestamps is not None and len(timestamps) != len(y_true):
        raise ValueError("timestamps length must match labels")
    if any(value not in (0, 1, False, True) for value in y_true):
        raise ValueError("Anomaly labels must be binary 0/1")
    if not all(_finite(score) for score in anomaly_scores):
        raise ValueError("Anomaly scores must all be finite")

    predictions = [int(float(score) >= float(threshold)) for score in anomaly_scores]
    result = evaluate_classifier(
        [int(value) for value in y_true],
        predictions,
        positive_label=1,
        timestamps=timestamps,
        temporal_windows=temporal_windows,
    )
    result["threshold"] = float(threshold)
    result["score_mean"] = statistics.fmean(float(x) for x in anomaly_scores)
    result["score_standard_deviation"] = (
        statistics.pstdev(float(x) for x in anomaly_scores)
    )
    return result


# ---------------------------------------------------------------------------
# 3. Reward design and paired RL policy benchmarking
# ---------------------------------------------------------------------------

def compose_reward(
    components: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    clip: float = 10.0,
) -> float:
    """Validate and compose a transparent, bounded scalar reward."""
    if set(components) != set(weights):
        raise ValueError("Reward component and weight keys must match exactly")
    if not _finite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    total = 0.0
    for name, value in components.items():
        weight = weights[name]
        if not _finite(value) or not _finite(weight):
            raise ValueError(f"Reward term {name!r} must be finite")
        total += float(value) * float(weight)
    return max(-float(clip), min(float(clip), total))


def benchmark_policies(
    scenarios: Sequence[Any],
    baseline_policy: Callable[[Any, int], float],
    candidate_policy: Callable[[Any, int], float],
    *,
    seeds: Sequence[int] = (11, 23, 37),
    minimum_pairs: int = 30,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Run both policies on identical scenario/seed pairs and compare rewards.

    Policy callbacks must be pure evaluation functions: they receive a scenario
    and seed and return a finite reward. No executor is passed to this API.
    """
    if not scenarios or not seeds:
        raise ValueError("At least one scenario and one seed are required")
    pairs: list[tuple[float, float]] = []
    for scenario in scenarios:
        for seed in seeds:
            baseline = baseline_policy(scenario, int(seed))
            candidate = candidate_policy(scenario, int(seed))
            if not _finite(baseline) or not _finite(candidate):
                raise ValueError("Policy reward must be finite")
            pairs.append((float(baseline), float(candidate)))
    if len(pairs) < minimum_pairs:
        raise ValueError(
            f"Need at least {minimum_pairs} paired runs; got {len(pairs)}"
        )

    differences = [candidate - baseline for baseline, candidate in pairs]
    rng = random.Random(20261117)
    boot_means = []
    for _ in range(bootstrap_samples):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        boot_means.append(statistics.fmean(sample))
    boot_means.sort()
    lower = boot_means[int(0.025 * (len(boot_means) - 1))]
    upper = boot_means[int(0.975 * (len(boot_means) - 1))]
    mean_delta = statistics.fmean(differences)
    return {
        "paired_runs": len(pairs),
        "scenario_count": len(scenarios),
        "seed_count": len(seeds),
        "baseline_mean_reward": statistics.fmean(x[0] for x in pairs),
        "candidate_mean_reward": statistics.fmean(x[1] for x in pairs),
        "mean_paired_improvement": mean_delta,
        "paired_improvement_std": statistics.pstdev(differences),
        "candidate_win_rate": sum(x > 0 for x in differences) / len(differences),
        "bootstrap_95_percent_ci": [lower, upper],
        "candidate_noninferior": lower >= 0.0,
        "reproducibility": {
            "same_scenarios_for_both_policies": True,
            "seeds": [int(x) for x in seeds],
            "bootstrap_seed": 20261117,
        },
        "promotion_recommendation": "eligible_for_review" if lower >= 0.0 else "reject",
    }


# ---------------------------------------------------------------------------
# 4. Immutable artifacts, registry, evaluation gates and controlled promotion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromotionGates:
    minimum_f1: float = 0.70
    maximum_false_positive_rate: float = 0.05
    minimum_recall: float = 0.80
    minimum_samples: int = 100
    require_temporal_stability: bool = True
    maximum_temporal_f1_std: float = 0.15


class ModelRegistry:
    """SQLite-backed model registry with checksum verification and approvals."""

    def __init__(self, database: str | Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL UNIQUE,
                    artifact_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    promoted_at TEXT,
                    approved_by TEXT
                );
                CREATE TABLE IF NOT EXISTS model_promotion_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    previous_model_id TEXT,
                    event TEXT NOT NULL,
                    approved_by TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    @staticmethod
    def checksum(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def register_candidate(
        self,
        artifact_path: str | Path,
        *,
        model_name: str,
        version: str,
        metrics: Mapping[str, Any],
    ) -> str:
        path = Path(artifact_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not model_name.strip() or not version.strip():
            raise ValueError("model_name and version are required")
        safe_metrics = _json_safe(dict(metrics))
        model_id = str(uuid.uuid4())
        with self._connect() as db:
            db.execute(
                """INSERT INTO model_registry
                (model_id, model_name, version, artifact_path, sha256,
                 metrics_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'candidate', ?)""",
                (model_id, model_name, version, str(path), self.checksum(path),
                 json.dumps(safe_metrics, allow_nan=False), _utc_now()),
            )
        return model_id

    def _get(self, db, model_id: str):
        row = db.execute(
            "SELECT * FROM model_registry WHERE model_id = ?", (model_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown model_id: {model_id}")
        return row

    def promote(
        self,
        model_id: str,
        *,
        approved_by: str,
        reason: str,
        gates: PromotionGates = PromotionGates(),
    ) -> dict[str, Any]:
        if not approved_by.strip() or not reason.strip():
            raise ValueError("Explicit approver and promotion reason are required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            candidate = self._get(db, model_id)
            if candidate["status"] != "candidate":
                raise ValueError("Only candidate artifacts can be promoted")
            path = Path(candidate["artifact_path"])
            if not path.is_file() or self.checksum(path) != candidate["sha256"]:
                raise ValueError("Artifact missing or checksum mismatch")
            metrics = json.loads(candidate["metrics_json"])
            required = ("f1", "recall", "false_positive_rate", "sample_count")
            missing = [key for key in required if key not in metrics]
            if missing:
                raise ValueError(f"Promotion metrics missing: {missing}")
            if metrics["sample_count"] < gates.minimum_samples:
                raise ValueError("Minimum evaluation sample count not met")
            if metrics["f1"] < gates.minimum_f1:
                raise ValueError("Minimum F1 gate failed")
            if metrics["recall"] < gates.minimum_recall:
                raise ValueError("Minimum recall gate failed")
            if metrics["false_positive_rate"] > gates.maximum_false_positive_rate:
                raise ValueError("Maximum false-positive-rate gate failed")
            stability = metrics.get("temporal_stability")
            if gates.require_temporal_stability:
                if not isinstance(stability, Mapping):
                    raise ValueError("Temporal stability evidence is required")
                f1_std = stability.get("f1_standard_deviation")
                if not _finite(f1_std) or f1_std > gates.maximum_temporal_f1_std:
                    raise ValueError("Temporal stability gate failed")

            current = db.execute(
                "SELECT model_id FROM model_registry WHERE model_name=? AND status='production'",
                (candidate["model_name"],),
            ).fetchone()
            previous_id = current["model_id"] if current else None
            db.execute(
                "UPDATE model_registry SET status='archived' WHERE model_name=? AND status='production'",
                (candidate["model_name"],),
            )
            now = _utc_now()
            db.execute(
                """UPDATE model_registry SET status='production',
                   promoted_at=?, approved_by=? WHERE model_id=?""",
                (now, approved_by, model_id),
            )
            db.execute(
                """INSERT INTO model_promotion_events
                   (model_id, previous_model_id, event, approved_by, reason, created_at)
                   VALUES (?, ?, 'promote', ?, ?, ?)""",
                (model_id, previous_id, approved_by, reason, now),
            )
            return {"model_id": model_id, "previous_model_id": previous_id,
                    "status": "production", "approved_by": approved_by}

    def rollback(
        self, model_id: str, *, approved_by: str, reason: str
    ) -> dict[str, Any]:
        if not approved_by.strip() or not reason.strip():
            raise ValueError("Explicit approver and rollback reason are required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            target = self._get(db, model_id)
            if target["status"] != "archived":
                raise ValueError("Rollback target must be an archived model")
            path = Path(target["artifact_path"])
            if not path.is_file() or self.checksum(path) != target["sha256"]:
                raise ValueError("Rollback artifact missing or checksum mismatch")
            current = db.execute(
                "SELECT model_id FROM model_registry WHERE model_name=? AND status='production'",
                (target["model_name"],),
            ).fetchone()
            previous_id = current["model_id"] if current else None
            db.execute(
                "UPDATE model_registry SET status='archived' WHERE model_name=? AND status='production'",
                (target["model_name"],),
            )
            now = _utc_now()
            db.execute(
                "UPDATE model_registry SET status='production', promoted_at=?, approved_by=? WHERE model_id=?",
                (now, approved_by, model_id),
            )
            db.execute(
                """INSERT INTO model_promotion_events
                   (model_id, previous_model_id, event, approved_by, reason, created_at)
                   VALUES (?, ?, 'rollback', ?, ?, ?)""",
                (model_id, previous_id, approved_by, reason, now),
            )
            return {"restored_model_id": model_id, "replaced_model_id": previous_id,
                    "status": "production"}

    def production_model(self, model_name: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                """SELECT * FROM model_registry
                   WHERE model_name=? AND status='production'
                   ORDER BY promoted_at DESC LIMIT 1""",
                (model_name,),
            ).fetchone()
        if row is None:
            return None
        path = Path(row["artifact_path"])
        if not path.is_file() or self.checksum(path) != row["sha256"]:
            raise RuntimeError("Production artifact failed checksum verification")
        return dict(row)


# ---------------------------------------------------------------------------
# 5. Actual telemetry -> decision pipeline smoke test (no executor access)
# ---------------------------------------------------------------------------

def _extract_model_metrics(observation: Mapping[str, Any]) -> dict[str, Any]:
    metrics = observation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Telemetry observation has no metrics mapping")
    result: dict[str, Any] = {}
    for key in ("cpu_percent", "memory_percent", "disk_percent",
                "network_connections", "load_1m"):
        raw = metrics.get(key)
        if isinstance(raw, Mapping):
            status = raw.get("status")
            value = raw.get("value")
            if status == "unavailable":
                if value is not None:
                    raise ValueError(f"{key}: unavailable value must be None")
                result[key] = None
            elif status == "ok":
                result[key] = value
            else:
                raise ValueError(f"{key}: invalid metric status {status!r}")
        else:
            result[key] = raw
    return result


def run_live_telemetry_decision_probe(
    collector: Any,
    decision_function: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Collect one real host sample and pass it to the decision function.

    Inject the project's real TelemetryCollector and decision pipeline entry
    point. This function intentionally has no executor argument and rejects
    any output claiming execution is enabled.
    """
    observation = collector.collect()
    if not isinstance(observation, Mapping):
        raise TypeError("collector.collect() must return an observation mapping")
    if observation.get("collection_status") not in {"ok", "partial", "failed"}:
        raise ValueError("Invalid collection_status")
    model_metrics = _extract_model_metrics(observation)

    # Validate feature encoding using Asterion's authoritative contract when
    # available. Import lazily so this module remains independently testable.
    try:
        from ..feature_contract import encode_features
        encoded = encode_features(model_metrics)
        feature_metadata = encoded.as_dict()
    except ImportError:
        feature_metadata = None

    decision = decision_function(observation)
    if not isinstance(decision, Mapping):
        raise TypeError("decision_function must return a mapping")

    forbidden_flags = ("action_execution_enabled", "live_execution_enabled")
    for flag in forbidden_flags:
        if decision.get(flag) is True:
            raise RuntimeError(f"Safety violation: {flag}=True")
    execution = decision.get("execution")
    if isinstance(execution, Mapping) and execution.get("enabled") is True:
        raise RuntimeError("Safety violation: decision requests live execution")
    if isinstance(execution, str) and execution.strip().lower() in {"enabled", "execute", "live"}:
        raise RuntimeError("Safety violation: decision requests live execution")

    return {
        "status": "passed",
        "timestamp": observation.get("timestamp"),
        "agent_id": observation.get("agent_id"),
        "host": observation.get("host"),
        "collection_status": observation.get("collection_status"),
        "metric_availability": {
            key: value is not None for key, value in model_metrics.items()
        },
        "feature_contract": feature_metadata,
        "decision": _json_safe(dict(decision)),
        "execution_enabled": False,
        "note": "Decision path evaluated; no action executor was invoked.",
    }
