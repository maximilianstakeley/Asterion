
import csv
import json

import pytest

from Asterion.ml_evaluation import (
    dataset_fingerprint,
    load_dataset,
    train_and_evaluate,
)


def create_dataset(path):
    rows = []

    for index in range(12):
        rows.append({
            "text": (
                f"Normal system activity: "
                f"routine process completed number {index}"
            ),
            "label": "normal",
        })

    for index in range(12):
        rows.append({
            "text": (
                f"Suspicious system activity: "
                f"unauthorized process detected number {index}"
            ),
            "label": "suspicious",
        })

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["text", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_fingerprint_is_order_independent(tmp_path):
    dataset = tmp_path / "training.csv"
    create_dataset(dataset)

    frame = load_dataset(dataset)

    reversed_frame = frame.iloc[::-1].reset_index(drop=True)

    assert dataset_fingerprint(frame) == dataset_fingerprint(
        reversed_frame
    )


def test_conflicting_labels_are_rejected(tmp_path):
    dataset = tmp_path / "conflicting.csv"

    with dataset.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["text", "label"],
        )
        writer.writeheader()

        writer.writerows([
            {"text": "same observation", "label": "normal"},
            {"text": "same observation", "label": "suspicious"},
            {"text": "normal observation one", "label": "normal"},
            {"text": "normal observation two", "label": "normal"},
            {"text": "suspicious observation one", "label": "suspicious"},
            {"text": "suspicious observation two", "label": "suspicious"},
        ])

    with pytest.raises(ValueError, match="conflicting labels"):
        load_dataset(dataset)


def test_training_creates_candidate_without_promoting(tmp_path):
    dataset = tmp_path / "training.csv"
    output = tmp_path / "candidate_models"

    create_dataset(dataset)

    result = train_and_evaluate(
        dataset,
        output,
    )

    assert result["promotion_status"] == "candidate_only"

    assert result["model_sha256"]
    assert result["dataset_sha256"]

    model_path = tmp_path / result["model_path"] if not (
        __import__("pathlib").Path(result["model_path"]).is_absolute()
    ) else __import__("pathlib").Path(result["model_path"])

    manifest_path = __import__("pathlib").Path(
        result["manifest_path"]
    )

    assert model_path.is_file()
    assert manifest_path.is_file()

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["promotion_approval_required"] is True
    assert manifest["live_execution_enabled"] is False
    assert manifest["cross_validation_folds"] >= 2

    assert "macro_f1" in manifest["cross_validation"]
    assert "balanced_accuracy" in manifest["holdout"]
    assert "confusion_matrix" in manifest["holdout"]
    assert "expected_calibration_error" in manifest["holdout"]

    # The module must not create or replace an active model.
    assert not (output.parent / "text_classifier.joblib").exists()
