"""Reproducible labeled-data evaluation for Asterion Stage 2.

Examples:
  python -m Asterion.evaluate_labeled_data
    --csv datasets/labeled_telemetry.csv
    --label-column label
    --positive-label intrusion
    --split group
    --group-column host
    --features cpu_percent,memory_percent,disk_percent,load_1m
    --output reports/stage2-evaluation.json

  python -m Asterion.evaluate_labeled_data
    --csv datasets/labeled_flows.csv
    --label-column label
    --positive-label malicious
    --split time
    --time-column timestamp
    --features bytes_transferred,packets,duration_seconds
    --output reports/flow-evaluation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split


SCRIPT_VERSION = "stage2-evaluation-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a rate only when its denominator is mathematically defined."""
    return numerator / denominator if denominator else None


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_label: Any,
) -> dict[str, Any]:
    """Calculate binary confusion counts and metrics for an explicit positive class."""

    # Convert lists, pandas objects, and other array-like inputs into arrays
    # before performing element-wise comparisons.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")

    # Missing labels must not be silently omitted from evaluation.
    if pd.isna(y_true).any() or pd.isna(y_pred).any():
        raise ValueError("y_true and y_pred must not contain missing labels")

    labels = set(y_true.tolist())
    labels.update(y_pred.tolist())
    labels.add(positive_label)

    if len(labels) != 2:
        raise ValueError(
            "Binary metrics require exactly two classes, including the "
            f"configured positive label; found {sorted(map(str, labels))}"
        )

    negative_labels = labels - {positive_label}
    if len(negative_labels) != 1:
        raise ValueError("Could not identify exactly one negative label")

    negative_label = next(iter(negative_labels))

    truth_positive = y_true == positive_label
    pred_positive = y_pred == positive_label
    truth_negative = y_true == negative_label
    pred_negative = y_pred == negative_label

    tp = int(np.sum(truth_positive & pred_positive))
    fp = int(np.sum(truth_negative & pred_positive))
    tn = int(np.sum(truth_negative & pred_negative))
    fn = int(np.sum(truth_positive & pred_negative))

    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    false_positive_rate = _safe_rate(fp, fp + tn)
    false_negative_rate = _safe_rate(fn, fn + tp)
    specificity = _safe_rate(tn, tn + fp)

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "positive_label": str(positive_label),
        "negative_label": str(negative_label),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "specificity": specificity,
        "f1": f1,
        "undefined_metrics_are_null": True,
    }



def detection_latency_metrics(
    test_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_label: str,
    event_time_column: str | None,
    alert_time_column: str | None,
) -> dict[str, Any]:
    """Measure event-to-alert delay only from actual recorded timestamps.

    Rows counted are true-positive detections. No latency is inferred from
    the time taken by an offline model to predict a CSV row.
    """
    if not event_time_column or not alert_time_column:
        return {
            "available": False,
            "reason": (
                "Provide event-time and alert-time columns from a replay "
                "or live detection run."
            ),
            "true_positive_count_with_latency": 0,
        }

    for column in (event_time_column, alert_time_column):
        if column not in test_frame.columns:
            raise ValueError(f"Missing latency column: {column}")

    event_times = pd.to_datetime(
        test_frame[event_time_column], errors="coerce", utc=True
    )
    alert_times = pd.to_datetime(
        test_frame[alert_time_column], errors="coerce", utc=True
    )

    if event_times.isna().any() or alert_times.isna().any():
        raise ValueError("Latency timestamp columns contain invalid values")

    true_positive_mask = (
        (y_true == positive_label) & (y_pred == positive_label)
    )
    latencies = (
        alert_times[true_positive_mask] - event_times[true_positive_mask]
    ).dt.total_seconds()

    if (latencies < 0).any():
        raise ValueError(
            "Alert timestamp precedes event timestamp; check clock alignment"
        )

    values = latencies.to_numpy(dtype=float)

    if len(values) == 0:
        return {
            "available": False,
            "reason": "No true-positive rows have measurable detection latency.",
            "true_positive_count_with_latency": 0,
        }

    return {
        "available": True,
        "event_time_column": event_time_column,
        "alert_time_column": alert_time_column,
        "true_positive_count_with_latency": int(len(values)),
        "mean_seconds": float(np.mean(values)),
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "max_seconds": float(np.max(values)),
    }



def _parse_time_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("time column contains missing or invalid timestamps")
    return parsed


def _split_indices(
    df: pd.DataFrame,
    *,
    split: str,
    test_fraction: float,
    seed: int,
    label_column: str,
    group_column: str | None,
    time_column: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = np.arange(len(df))

    if split == "group":
        if not group_column or group_column not in df:
            raise ValueError("--split group requires a valid --group-column")
        if df[group_column].isna().any():
            raise ValueError("group column contains missing values")

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=seed,
        )
        train_idx, test_idx = next(
            splitter.split(indices, df[label_column], groups=df[group_column])
        )

        train_groups = set(df.iloc[train_idx][group_column].astype(str))
        test_groups = set(df.iloc[test_idx][group_column].astype(str))
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError("Group leakage detected between train and test")

        return train_idx, test_idx, {
            "method": "GroupShuffleSplit",
            "group_column": group_column,
            "train_group_count": len(train_groups),
            "test_group_count": len(test_groups),
            "group_overlap_count": 0,
        }

    if split == "time":
        if not time_column or time_column not in df:
            raise ValueError("--split time requires a valid --time-column")

        times = _parse_time_column(df[time_column])
        order = np.argsort(times.astype("int64").to_numpy(), kind="stable")
        test_count = max(1, int(np.ceil(len(df) * test_fraction)))
        split_at = len(df) - test_count

        if split_at < 1:
            raise ValueError("Not enough records for a time-ordered split")

        train_idx, test_idx = order[:split_at], order[split_at:]
        train_end = times.iloc[train_idx].max()
        test_start = times.iloc[test_idx].min()

        if train_end > test_start:
            raise RuntimeError("Time leakage detected between partitions")

        return train_idx, test_idx, {
            "method": "chronological_holdout",
            "time_column": time_column,
            "train_end_utc": train_end.isoformat(),
            "test_start_utc": test_start.isoformat(),
            "test_records_are_later_than_training_records": True,
        }

    if split == "random":
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=seed,
            stratify=df[label_column],
        )
        return train_idx, test_idx, {
            "method": "stratified_random_holdout",
            "warning": (
                "Random splitting can overestimate performance when records "
                "from the same host or time period are correlated."
            ),
        }

    raise ValueError(f"Unsupported split mode: {split}")


def evaluate_csv(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    csv_path = Path(args.csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    dataset_hash = sha256_file(csv_path)

    required = [args.label_column, *args.features]
    if args.group_column:
        required.append(args.group_column)
    if args.time_column:
        required.append(args.time_column)

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    if not args.features:
        raise ValueError("At least one feature is required")
    if args.label_column in args.features:
        raise ValueError("The label column cannot be a feature")
    if args.group_column in args.features:
        raise ValueError("The group column cannot be a feature")
    if args.time_column in args.features:
        raise ValueError("The time column cannot be a feature")

    # Drop rows with missing labels only. Feature missingness is handled inside
    # the train-fitted preprocessing pipeline. Group/time fields must be valid.
    df = df.loc[df[args.label_column].notna()].copy()
    if df.empty:
        raise ValueError("No labeled records remain")

    labels = df[args.label_column].astype(str)
    label_mapping = {
        label: index
        for index, label in enumerate(sorted(labels.unique().tolist()))
    }
    df[args.label_column] = labels

    train_idx, test_idx, split_metadata = _split_indices(
        df,
        split=args.split,
        test_fraction=args.test_fraction,
        seed=args.seed,
        label_column=args.label_column,
        group_column=args.group_column,
        time_column=args.time_column,
    )

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    y_train = train[args.label_column].to_numpy()
    y_test = test[args.label_column].to_numpy()
    X_train = train[args.features].copy()
    X_test = test[args.features].copy()

    # This implementation intentionally accepts numeric features only.
    # It fails rather than silently encoding arbitrary text as numbers.
    for feature in args.features:
        X_train[feature] = pd.to_numeric(
            X_train[feature],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        X_test[feature] = pd.to_numeric(
            X_test[feature],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

    all_missing = [
        feature for feature in args.features
        if X_train[feature].notna().sum() == 0
    ]
    if all_missing:
        raise ValueError(
            f"Features entirely missing in training data: {all_missing}"
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                args.features,
            )
        ],
        remainder="drop",
    )

    if args.model == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            class_weight="balanced",
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", estimator),
    ])

    fit_started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_seconds = time.perf_counter() - predict_started

    positive_label = str(args.positive_label)
    binary = binary_metrics(
        y_test,
        y_pred,
        positive_label=positive_label,)


    latency = detection_latency_metrics(
        test,
        y_test,
        y_pred,
        positive_label=positive_label,
        event_time_column=args.event_time_column,
        alert_time_column=args.alert_time_column,)


    elapsed_seconds = time.perf_counter() - started
    total_test = len(y_test)

    report = {
        "evaluation_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(csv_path),
            "sha256": dataset_hash,
            "file_bytes": csv_path.stat().st_size,
            "rows_after_missing_label_filter": int(len(df)),
            "label_column": args.label_column,
            "label_mapping": label_mapping,
            "label_counts": {
                str(k): int(v)
                for k, v in df[args.label_column].value_counts().items()
            },
        },
        "features": {
            "ordered_feature_names": list(args.features),
            "types": "numeric_only",
            "missing_policy": (
                "Feature values are coerced to numeric; missing/invalid "
                "values are median-imputed using training data only."
            ),
            "normalization": "StandardScaler fitted on training data only",
        },
        "configuration": {
            "model": args.model,
            "n_estimators": args.n_estimators,
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "split": args.split,
            "split_metadata": split_metadata,
            "positive_label": positive_label,
        },
        "partition": {
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_label_counts": {
                str(k): int(v)
                for k, v in train[args.label_column].value_counts().items()
            },
            "test_label_counts": {
                str(k): int(v)
                for k, v in test[args.label_column].value_counts().items()
            },
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "binary": binary,
            "classification_report": classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix_labels_sorted": sorted(
                set(y_test.tolist()) | set(y_pred.tolist())
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                y_pred,
                labels=sorted(set(y_test.tolist()) | set(y_pred.tolist())),
            ).tolist(),},

        "performance": {
            "fit_seconds": fit_seconds,
            "prediction_seconds": predict_seconds,
            "detection_latency": latency,
            "prediction_seconds_per_record": (
                predict_seconds / total_test if total_test else None),
            "test_records_per_second": (
                total_test / predict_seconds if predict_seconds > 0 else None),
            "evaluation_wall_seconds": elapsed_seconds,},

        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,},

        "limitations": [
            "Evaluation measures agreement with dataset labels, not certainty of intrusion.",
            "Dataset labels may be noisy, incomplete, or domain-specific.",
            "Detection latency from event occurrence to alert is not measured by offline CSV evaluation.",
            "The dataset fingerprint identifies file content but does not prove label correctness.",
        ],}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",)

    return report

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-time-column")
    parser.add_argument("--alert-time-column")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", choices=["group", "time", "random"], required=True)
    parser.add_argument("--group-column")
    parser.add_argument("--time-column")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=["random_forest"],
        default="random_forest",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--output",
        default="reports/stage2-evaluation.json",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.features = [
        item.strip() for item in args.features.split(",") if item.strip()
    ]

    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if args.n_estimators < 1:
        parser.error("--n-estimators must be positive")

    try:
        report = evaluate_csv(args)
    except Exception as exc:
        parser.exit(2, f"Evaluation failed: {type(exc).__name__}: {exc}\n")

    print(json.dumps({
        "status": "completed",
        "report_path": str(Path(args.output).expanduser().resolve()),
        "dataset_sha256": report["dataset"]["sha256"],
        "train_rows": report["partition"]["train_rows"],
        "test_rows": report["partition"]["test_rows"],
        "metrics": report["metrics"]["binary"],
        "performance": report["performance"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
