
"""Validated text classification and deterministic telemetry anomaly detection."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import joblib

from .configuration import MODEL_DIR


logger = logging.getLogger(__name__)

MODEL_PATH = MODEL_DIR / "text_classifier.joblib"

# Keep thresholds explicit and reproducible.
ANOMALY_RULES = (
    (
        "cpu_percent",
        90.0,
        "high_cpu",
        "CPU usage is unusually high",
    ),
    (
        "memory_percent",
        90.0,
        "high_memory",
        "Memory usage is unusually high",
    ),
    (
        "disk_percent",
        95.0,
        "high_disk",
        "Disk utilization is near capacity",
    ),
)

HIGH_SEVERITY_MULTIPLIER = 1.15


class TextClassifier:
    """Load and safely query a promoted text-classification pipeline.

    The model artifact must be a trusted joblib file containing a dictionary
    with a fitted classification pipeline under the key 'pipeline'.

    An unavailable or invalid model does not prevent rule-based anomaly
    detection from operating.
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else MODEL_PATH
        self.pipeline: Any | None = None
        self.load_error: str | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Load and validate the classifier artifact."""
        if not self.model_path.is_file():
            self.load_error = f"Model artifact not found: {self.model_path}"
            logger.info("%s", self.load_error)
            return

        try:
            # Only load joblib artifacts from trusted sources.
            artifact = joblib.load(self.model_path)

            if not isinstance(artifact, dict):
                raise ValueError(
                    "Model artifact must be a dictionary containing 'pipeline'."
                )

            pipeline = artifact.get("pipeline")

            if pipeline is None:
                raise ValueError(
                    "Model artifact does not contain a 'pipeline' entry."
                )

            if not callable(getattr(pipeline, "predict", None)):
                raise ValueError(
                    "Classifier pipeline does not expose a callable predict method."
                )

            classes = getattr(pipeline, "classes_", None)

            if classes is None or len(classes) == 0:
                raise ValueError(
                    "Classifier pipeline has no fitted classes."
                )

            self.pipeline = pipeline
            self.load_error = None

        except Exception as exc:
            # An invalid model must not be mistaken for a working classifier.
            self.pipeline = None
            self.load_error = f"{type(exc).__name__}: {exc}"

            logger.exception(
                "Could not load text classifier from %s",
                self.model_path,
            )

    def reload(self) -> None:
        """Retry loading the configured model artifact."""
        self.pipeline = None
        self.load_error = None
        self._load_model()

    def predict(self, text: str) -> dict[str, Any]:
        """Classify non-empty text, or report that classification is unavailable."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if self.pipeline is None:
            return {
                "available": False,
                "label": "unknown",
                "confidence": 0.0,
                "reason": (
                    "No valid promoted classifier is available. "
                    "Train and promote a classifier using the project CLI."
                ),
                "load_error": self.load_error,
            }

        predict_proba = getattr(self.pipeline, "predict_proba", None)

        if not callable(predict_proba):
            return {
                "available": False,
                "label": "unknown",
                "confidence": 0.0,
                "reason": (
                    "The loaded classifier does not support predict_proba; "
                    "calibrated class confidence is unavailable."
                ),
                "load_error": None,
            }

        try:
            probabilities = predict_proba([text])
            classes = list(self.pipeline.classes_)

            # Convert supported numeric output into a predictable structure.
            if getattr(probabilities, "ndim", None) != 2:
                raise ValueError(
                    "Classifier probabilities must be a two-dimensional array."
                )

            if probabilities.shape[0] != 1:
                raise ValueError(
                    "Classifier must return exactly one probability row per input."
                )

            if probabilities.shape[1] != len(classes):
                raise ValueError(
                    "Probability count does not match the classifier's classes."
                )

            row = [float(value) for value in probabilities[0]]

            if not row or not all(math.isfinite(value) for value in row):
                raise ValueError(
                    "Classifier returned missing or non-finite probabilities."
                )

            if any(value < 0.0 or value > 1.0 for value in row):
                raise ValueError(
                    "Classifier probabilities must be between 0 and 1."
                )

            if not math.isclose(sum(row), 1.0, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(
                    "Classifier probabilities do not sum to approximately 1."
                )

            best_index = max(range(len(row)), key=row.__getitem__)
            label = str(classes[best_index])
            confidence = row[best_index]

            return {
                "available": True,
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    str(class_label): probability
                    for class_label, probability in zip(classes, row)
                },
            }

        except Exception as exc:
            # Classification failure is surfaced explicitly. It is not
            # represented as a genuine prediction of "unknown".
            logger.exception("Text classifier prediction failed")

            return {
                "available": False,
                "label": "unknown",
                "confidence": 0.0,
                "reason": (
                    "The classifier could not produce a valid prediction."
                ),
                "prediction_error": f"{type(exc).__name__}: {exc}",
            }


def _extract_metric_value(metrics: dict[str, Any], key: str) -> float | None:
    """Extract a valid finite numeric metric; unavailable values stay unavailable.

    Accepts either a scalar:
        {"cpu_percent": 93.0}

    or a structured metric:
        {"cpu_percent": {"value": 93.0, "status": "ok"}}

    Structured metrics explicitly marked unavailable or failed are ignored.
    Booleans, strings, NaN, and infinities are not valid numeric measurements.
    """
    if key not in metrics:
        return None

    raw_value = metrics[key]

    if raw_value is None:
        return None

    if isinstance(raw_value, dict):
        status = raw_value.get("status")

        if status is not None and status != "ok":
            return None

        raw_value = raw_value.get("value")

    if raw_value is None or isinstance(raw_value, bool):
        return None

    if not isinstance(raw_value, (int, float)):
        return None

    value = float(raw_value)

    if not math.isfinite(value):
        return None

    return value


def detect_anomalies(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect deterministic threshold anomalies from available telemetry.

    This function reports threshold crossings only. It does not establish
    malicious intent, diagnose root cause, or authorize an action.

    Unavailable and malformed measurements do not generate anomaly findings.
    Network connection counts are intentionally not required: local connection
    enumeration may be unavailable, and future network measurements may come
    from separately supplied datasets or other telemetry sources.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary")

    findings: list[dict[str, Any]] = []

    for key, threshold, category, description in ANOMALY_RULES:
        value = _extract_metric_value(metrics, key)

        if value is None or value < threshold:
            continue

        severity = (
            "high"
            if value >= threshold * HIGH_SEVERITY_MULTIPLIER
            else "medium"
        )

        findings.append(
            {
                "category": category,
                "severity": severity,
                "description": description,
                "evidence": {
                    "metric": key,
                    "observed": value,
                    "threshold": threshold,
                    "comparison": ">=",
                    "rule": "deterministic_threshold_v1",
                },
            }
        )

    return findings
