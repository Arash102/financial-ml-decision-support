"""Binary classification metrics for Stage 06 model selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    *,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Return threshold-free and threshold-based binary metrics."""
    y_true_array = np.asarray(y_true, dtype=int)
    probability = np.asarray(y_probability, dtype=float)

    if y_true_array.ndim != 1 or probability.ndim != 1:
        raise ValueError("y_true and y_probability must be one-dimensional.")

    if len(y_true_array) != len(probability):
        raise ValueError("y_true and y_probability lengths differ.")

    if not np.isfinite(probability).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("Predicted probabilities must be inside [0, 1].")

    classes = np.unique(y_true_array)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(
            "Both binary target classes must be present. "
            f"Observed classes: {classes.tolist()}"
        )

    prediction = (probability >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        prediction,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan

    return {
        "events": int(len(y_true_array)),
        "positive_fraction": float(y_true_array.mean()),
        "roc_auc": float(roc_auc_score(y_true_array, probability)),
        "average_precision": float(
            average_precision_score(y_true_array, probability)
        ),
        "log_loss": float(
            log_loss(
                y_true_array,
                probability,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(y_true_array, probability)
        ),
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true_array, prediction)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, prediction)
        ),
        "precision": float(
            precision_score(
                y_true_array,
                prediction,
                zero_division=0,
            )
        ),
        "sensitivity": float(sensitivity),
        "recall": float(
            recall_score(
                y_true_array,
                prediction,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                y_true_array,
                prediction,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(y_true_array, prediction)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
