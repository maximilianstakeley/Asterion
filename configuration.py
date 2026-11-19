
"""Reproducible supervised-learning evaluation for Asterion.

Trains a calibrated text classifier, evaluates it with stratified
cross-validation and a separate holdout set, and saves an immutable
candidate artifact with a provenance manifest.

This module never promotes or activates a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
TEST_FRACTION = 0.20
MAX_FEATURES = 150_000
MINIMUM_ROWS = 12
MAX_CV_FOLDS = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_json_write(data: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(descriptor)

    temporary = Path(temporary_name)

    try:
        temporary.write_text(
            json.dumps(data, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_joblib_write(artifact: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(descriptor)

    temporary = Path(temporary_name)

    try:
        joblib.dump(artifact, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)

    required = {"text", "label"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    frame = frame[["text", "label"]].dropna().copy()

    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()

    frame = frame[
        (frame["text"] != "")
        & (frame["label"] != "")
    ].copy()

    # The same text with conflicting labels is ambiguous training data.
    conflicts = (
        frame.groupby("text")["label"]
        .nunique()
    )
    conflicting_texts = conflicts[conflicts > 1]

    if not conflicting_texts.empty:
        raise ValueError(
            f"Found {len(conflicting_texts)} text samples with conflicting "
            "labels. Resolve these labels before training."
        )

    frame = frame.drop_duplicates(
        subset=["text", "label"]
    ).reset_index(drop=True)

    if len(frame) < MINIMUM_ROWS:
        raise ValueError(
            f"At least {MINIMUM_ROWS} unique rows are required; "
            f"found {len(frame)}."
        )

    counts = frame["label"].value_counts()

    if len(counts) < 2:
        raise ValueError("At least two distinct labels are required.")

    if counts.min() < 3:
        raise ValueError(
            "Every label must have at least three unique samples "
            "for holdout and cross-validation."
        )

    return frame


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Create an order-independent fingerprint of normalized training data."""
    records = sorted(
        (str(row.text), str(row.label))
        for row in frame.itertuples(index=False)
    )

    canonical = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=MAX_FEATURES,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Calculate top-label expected calibration error.

    This is a diagnostic, not proof that probabilities are calibrated.
    """
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)

    classes = np.arange(probabilities.shape[1])
    actual_indices = np.array(
        [np.where(classes == value)[0][0] for value in y_true]
    )

    correct = predictions == actual_indices
    boundaries = np.linspace(0.0, 1.0, bins + 1)

    error = 0.0

    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]

        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        if not np.any(mask):
            continue

        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidence[mask].mean())

        error += (
            float(mask.mean())
            * abs(bin_accuracy - bin_confidence)
        )

    return float(error)


def train_and_evaluate(
    csv_path: str | Path,
    output_directory: str | Path = "artifacts/models/candidates",
) -> dict[str, Any]:
    frame = load_dataset(csv_path)

    X = frame["text"]
    y = frame["label"]

    classes = sorted(y.unique().tolist())
    number_of_classes = len(classes)

    test_rows = math.ceil(len(frame) * TEST_FRACTION)

    if test_rows < number_of_classes:
        raise ValueError(
            "Holdout partition is too small to contain every class. "
            "Add more labeled samples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_rows,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    training_counts = y_train.value_counts()

    if training_counts.min() < 2:
        raise ValueError(
            "Training partition has too few samples for at least one class."
        )

    folds = min(
        MAX_CV_FOLDS,
        int(training_counts.min()),
    )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Cross-validation is performed only on the training partition.
    cv_results = cross_validate(
        build_model(),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "accuracy": "accuracy",
        },
        return_train_score=False,
        error_score="raise",
    )

    cv_metrics = {}

    for metric in (
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
    ):
        values = cv_results[f"test_{metric}"]

        cv_metrics[metric] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "fold_values": [float(value) for value in values],
        }

    # Calibrate probabilities using training data only.
    calibrated_model = CalibratedClassifierCV(
        estimator=build_model(),
        method="sigmoid",
        cv=cv,
    )

    calibrated_model.fit(X_train, y_train)

    predictions = calibrated_model.predict(X_test)
    probabilities = calibrated_model.predict_proba(X_test)

    holdout_metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_precision": float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_test,
                probabilities,
                labels=classes,
            )
        ),
        "expected_calibration_error": expected_calibration_error(
            np.asarray(
                [classes.index(label) for label in y_test]
            ),
            probabilities,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=classes,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=classes,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
    }

    for value in holdout_metrics.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(
                "Model evaluation produced a non-finite metric."
            )

    version = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )

    output_directory = Path(output_directory).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    model_path = output_directory / f"text-classifier-{version}.joblib"
    manifest_path = output_directory / f"text-classifier-{version}.json"

    dataset_hash = dataset_fingerprint(frame)

    artifact = {
        "model": calibrated_model,
        "model_type": "tfidf_logistic_regression_calibrated",
        "version": version,
        "classes": classes,
        "dataset_sha256": dataset_hash,
        "created_at": utc_now(),
    }

    atomic_joblib_write(artifact, model_path)
    model_hash = sha256_file(model_path)

    manifest = {
        "version": version,
        "created_at": utc_now(),
        "model_type": artifact["model_type"],
        "model_sha256": model_hash,
        "dataset_sha256": dataset_hash,
        "dataset_rows": int(len(frame)),
        "training_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "classes": classes,
        "random_state": RANDOM_STATE,
        "cross_validation_folds": folds,
        "cross_validation": cv_metrics,
        "holdout": holdout_metrics,
        "promotion_status": "candidate_only",
        "promotion_approval_required": True,
        "live_execution_enabled": False,
    }

    atomic_json_write(manifest, manifest_path)

    return {
        "version": version,
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "model_sha256": model_hash,
        "dataset_sha256": dataset_hash,
        "cross_validation": cv_metrics,
        "holdout": holdout_metrics,
        "promotion_status": "candidate_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an Asterion ML candidate."
    )

    parser.add_argument(
        "dataset",
        help="CSV file containing text and label columns.",
    )

    parser.add_argument(
        "--output",
        default="artifacts/models/candidates",
        help="Directory for versioned candidate artifacts.",
    )

    args = parser.parse_args()

    result = train_and_evaluate(
        args.dataset,
        args.output,
    )

    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
