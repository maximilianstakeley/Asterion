
"""Train, evaluate, and conditionally promote Asterion's text classifier.

The classifier is trained and evaluated on a stratified holdout split.
Promotion is controlled by a validated macro-F1 threshold. This metric is
necessary for the configured promotion rule, but does not establish production
suitability or operational safety.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .configuration import MODEL_DIR, MIN_PROMOTION_F1
from .database import connect, now

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_FRACTION = 0.20
MINIMUM_ROWS = 10
MAX_FEATURES = 150_000


def _validate_threshold(value: Any) -> float:
    """Validate the macro-F1 threshold used for promotion."""
    if isinstance(value, bool):
        raise ValueError("min_f1 must be a number between 0 and 1")

    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "min_f1 must be a number between 0 and 1"
        ) from exc

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_f1 must be finite and in [0, 1]")

    return threshold


def _atomic_joblib_dump(
    artifact: Any,
    destination: Path,
) -> None:
    """Write a joblib artifact to a temporary file, then atomically replace it.

    The temporary file is created in the destination directory so replacement
    occurs on the same filesystem.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)

    temporary_path = Path(temporary_name)

    try:
        joblib.dump(artifact, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not remove temporary model file %s",
                temporary_path,
            )


def _load_and_validate_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the required text/label dataset."""
    path = Path(csv_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"CSV must contain text and label columns; missing: {missing}"
        )

    # Retain only the columns used by this training pipeline.
    df = df[["text", "label"]].dropna().copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[
        (df["text"] != "")
        & (df["label"] != "")
    ].drop_duplicates()

    if len(df) < MINIMUM_ROWS:
        raise ValueError(
            f"Need at least {MINIMUM_ROWS} unique usable rows; got {len(df)}."
        )

    label_counts = df["label"].value_counts()

    if len(label_counts) < 2:
        raise ValueError("At least two distinct labels are required.")

    if label_counts.min() < 2:
        raise ValueError(
            "Every label needs at least two rows for stratified evaluation."
        )

    return df.reset_index(drop=True)


def _build_model() -> Pipeline:
    """Create a fresh text-classification pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=MAX_FEATURES,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_csv(
    csv_path: str | Path,
    min_f1: float | None = None,
) -> dict[str, Any]:
    """Train and evaluate a text classifier, then conditionally promote it.

    Returns promotion status, artifact paths, and holdout evaluation metrics.

    Note:
        This function uses a single stratified holdout split. It does not
        perform cross-validation, calibration, drift evaluation, or an
        independent external test.
    """
    configured_threshold = (
        MIN_PROMOTION_F1 if min_f1 is None else min_f1
    )
    threshold = _validate_threshold(configured_threshold)

    df = _load_and_validate_csv(csv_path)

    X = df["text"]
    y = df["label"]

    number_of_labels = y.nunique()
    test_rows = math.ceil(len(df) * TEST_FRACTION)
    train_rows = len(df) - test_rows

    # Stratification needs enough samples in both partitions to represent
    # every class. Detect impossible splits before calling scikit-learn.
    if test_rows < number_of_labels or train_rows < number_of_labels:
        raise ValueError(
            "Dataset is too small for a stratified holdout split containing "
            "every label in both train and test partitions. Add more rows "
            "or reduce the number of labels."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_rows,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = _build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "labels": sorted(y.unique().tolist()),
        "label_counts": {
            str(label): int(count)
            for label, count in y.value_counts().sort_index().items()
        },
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "promotion_threshold_macro_f1": threshold,
        "random_state": RANDOM_STATE,
        "test_fraction": TEST_FRACTION,
        "evaluation_method": "stratified_holdout",
        "report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    # Reject non-finite evaluation output rather than writing misleading JSON.
    if not all(
        math.isfinite(metrics[key])
        for key in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
    ):
        raise RuntimeError("Evaluation produced a non-finite metric.")

    promoted = bool(metrics["macro_f1"] >= threshold)

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = model_dir / "text_classifier_candidate.joblib"
    active_path = model_dir / "text_classifier.joblib"

    artifact = {
        "pipeline": model,
        "metrics": metrics,
    }

    # Always preserve the candidate separately from the active artifact.
    _atomic_joblib_dump(artifact, candidate_path)

    if promoted:
        # Replace the active artifact atomically. A failed write should not
        # leave a partially serialized model at the active path.
        _atomic_joblib_dump(artifact, active_path)
        logger.info(
            "Text classifier met the macro-F1 promotion threshold: "
            "%.4f >= %.4f",
            metrics["macro_f1"],
            threshold,
        )
    else:
        logger.info(
            "Text classifier was not promoted: macro-F1 %.4f < %.4f",
            metrics["macro_f1"],
            threshold,
        )

    # Register the result and its evaluated artifact path.
    # File replacement and database registration cannot be one atomic
    # transaction. If registration fails after promotion, the active model
    # may already have changed; the exception is therefore allowed to surface.
    registry_path = active_path if promoted else candidate_path

    with connect() as connection:
        connection.execute(
            """
            INSERT INTO model_registry
                (timestamp, model_path, model_type, metrics_json, promoted)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                now(),
                str(registry_path),
                "tfidf_logistic_regression",
                json.dumps(metrics, allow_nan=False),
                int(promoted),
            ),
        )

    return {
        "promoted": promoted,
        "promotion_threshold_macro_f1": threshold,
        "candidate_path": str(candidate_path),
        "active_model_path": str(active_path) if promoted else None,
        "metrics": metrics,
    }
