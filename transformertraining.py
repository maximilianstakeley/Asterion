
"""Train and evaluate a transformer-based text classifier.

This script saves a standalone Hugging Face model and its evaluation metrics.
It does not automatically promote the model into Asterion's active classifier
or enable operational actions.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_FRACTION = 0.20
MINIMUM_ROWS = 10
MAX_TOKEN_LENGTH = 256


def _parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a transformer text classifier."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV dataset containing text and label columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/transformer",
        help="Directory in which to save the trained model and metrics.",
    )
    parser.add_argument(
        "--model",
        default="distilbert/distilbert-base-uncased",
        help="Hugging Face model identifier or local model directory.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
        help="Number of training epochs; must be finite and positive.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device training and evaluation batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_TOKEN_LENGTH,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for the split and training.",
    )

    args = parser.parse_args()

    if not math.isfinite(args.epochs) or args.epochs <= 0:
        parser.error("--epochs must be finite and greater than zero")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and greater than zero")

    if args.max_length < 8:
        parser.error("--max-length must be at least 8")

    if not args.model.strip():
        parser.error("--model must not be empty")

    if args.seed < 0:
        parser.error("--seed must be non-negative")

    if not args.output_dir.strip():
        parser.error("--output-dir must not be empty")

    return args


def _load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load, normalize, and validate the training dataset."""
    path = Path(csv_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    required = {"text", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "CSV must contain text and label columns; missing: "
            + ", ".join(sorted(missing))
        )

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

    # Conflicting labels for identical normalized text can make evaluation
    # misleading. Require the dataset owner to resolve those conflicts.
    labels_per_text = df.groupby("text")["label"].nunique()
    conflicting_texts = labels_per_text[labels_per_text > 1]

    if not conflicting_texts.empty:
        examples = conflicting_texts.index[:3].tolist()
        raise ValueError(
            f"Found {len(conflicting_texts)} text values with conflicting "
            f"labels. Resolve these before training. Examples: {examples!r}"
        )

    return df.reset_index(drop=True)


def _split_dataset(
    df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic stratified train/evaluation split."""
    number_of_labels = df["label"].nunique()
    evaluation_rows = math.ceil(len(df) * TEST_FRACTION)
    training_rows = len(df) - evaluation_rows

    if (
        evaluation_rows < number_of_labels
        or training_rows < number_of_labels
    ):
        raise ValueError(
            "Dataset is too small for a stratified split that includes every "
            "label in both partitions. Add more examples or reduce the number "
            "of labels."
        )

    train_df, eval_df = train_test_split(
        df,
        test_size=evaluation_rows,
        random_state=seed,
        stratify=df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        eval_df.reset_index(drop=True),
    )


def _tokenize_frame(
    frame: pd.DataFrame,
    tokenizer: Any,
    label_to_id: dict[str, int],
    max_length: int,
) -> Dataset:
    """Convert a dataframe into a tokenized Hugging Face dataset."""
    dataset = Dataset.from_dict(
        {
            "text": frame["text"].tolist(),
            "labels": [
                label_to_id[label]
                for label in frame["label"].tolist()
            ],
        }
    )

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing text",
    )


def _compute_metrics(eval_prediction: Any) -> dict[str, float]:
    """Compute macro-averaged evaluation metrics."""
    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids

    # Some models return tuples in their prediction output.
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def _write_json_atomically(
    destination: Path,
    payload: dict[str, Any],
) -> None:
    """Write JSON through a temporary file to avoid partial output."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)

    temporary_path = Path(temporary_name)

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")

        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    df = _load_dataset(args.csv)
    train_df, eval_df = _split_dataset(df, args.seed)

    labels = sorted(df["label"].unique().tolist())
    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }
    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    logger.info(
        "Dataset: %d usable rows, %d labels, %d training rows, %d evaluation rows",
        len(df),
        len(labels),
        len(train_df),
        len(eval_df),
    )

    # The model and tokenizer are loaded from the supplied identifier or
    # local path. Remote model code is not explicitly trusted by this script.
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    train_dataset = _tokenize_frame(
        train_df,
        tokenizer,
        label_to_id,
        args.max_length,
    )
    eval_dataset = _tokenize_frame(
        eval_df,
        tokenizer,
        label_to_id,
        args.max_length,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    # Evaluate the best checkpoint selected by macro-F1.
    evaluation_metrics = trainer.evaluate()

    predictions_output = trainer.predict(eval_dataset)
    logits = predictions_output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_ids = np.argmax(logits, axis=-1)
    true_ids = predictions_output.label_ids

    per_class_report = classification_report(
        true_ids,
        predicted_ids,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    # Convert metric values to ordinary Python floats for strict JSON output.
    serializable_metrics = {
        key: float(value)
        for key, value in evaluation_metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    }

    result = {
        "model_source": args.model,
        "labels": labels,
        "label_to_id": label_to_id,
        "dataset": {
            "usable_rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "evaluation_rows": int(len(eval_df)),
            "label_counts": {
                str(label): int(count)
                for label, count in df["label"].value_counts().sort_index().items()
            },
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "evaluation_method": "stratified_holdout",
        },
        "metrics": serializable_metrics,
        "per_class_report": per_class_report,
    }

    if "eval_macro_f1" not in serializable_metrics:
        raise RuntimeError(
            "Evaluation did not produce eval_macro_f1; model artifact not finalized."
        )

    if not 0.0 <= serializable_metrics["eval_macro_f1"] <= 1.0:
        raise RuntimeError("Evaluation produced an invalid macro-F1 score.")

    # Save the best evaluated model and tokenizer as a standalone artifact.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    _write_json_atomically(
        output_dir / "evaluation_metrics.json",
        result,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "labels": labels,
                "metrics": serializable_metrics,
                "per_class_report": per_class_report,
                "active_classifier_promoted": False,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
