
from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def chronological_split(
    X,
    y,
    train_fraction=0.70,
    validation_fraction=0.15
):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if len(X) != len(y) or len(X) < 20:
        raise ValueError("Insufficient or mismatched samples")

    if not 0 < train_fraction < 1:
        raise ValueError("Invalid train fraction")

    if not 0 < validation_fraction < 1:
        raise ValueError("Invalid validation fraction")

    if train_fraction + validation_fraction >= 1:
        raise ValueError("Test partition must remain non-empty")

    n = len(X)

    train_end = int(n * train_fraction)

    validation_end = int(
        n * (train_fraction + validation_fraction)
    )

    return {
        "train": (X[:train_end], y[:train_end]),
        "validation": (
            X[train_end:validation_end],
            y[train_end:validation_end]
        ),
        "test": (X[validation_end:], y[validation_end:])
    }


def evaluate_classifier(y_true, y_pred, probabilities=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction dimensions do not match")

    if len(y_true) == 0:
        raise ValueError("Empty evaluation dataset")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = (
        fp / (fp + tn) if fp + tn else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp) if fn + tp else 0.0
    )

    result = {
        "sample_count": len(y_true),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if probabilities is not None:
        probabilities = np.asarray(probabilities)

        if len(np.unique(y_true)) == 2:
            result["roc_auc"] = float(
                roc_auc_score(y_true, probabilities)
            )
        else:
            result["roc_auc"] = None

    return result


def compare_policies(baseline_rewards, candidate_rewards):
    baseline = np.asarray(
        baseline_rewards, dtype=float
    )

    candidate = np.asarray(
        candidate_rewards, dtype=float
    )

    if len(baseline) != len(candidate):
        raise ValueError("Policy evaluation scenarios must match")

    if len(baseline) < 30:
        raise ValueError("At least 30 paired scenarios required")

    if not (
        np.isfinite(baseline).all()
        and np.isfinite(candidate).all()
    ):
        raise ValueError("Rewards must be finite")

    differences = candidate - baseline

    return {
        "scenarios": len(differences),
        "baseline_mean": float(np.mean(baseline)),
        "candidate_mean": float(np.mean(candidate)),
        "mean_improvement": float(np.mean(differences)),
        "improvement_std": float(np.std(differences)),
        "candidate_win_rate": float(
            np.mean(differences > 0)
        ),
        "candidate_noninferiority": bool(
            np.mean(differences) >= 0
        ),
    }
