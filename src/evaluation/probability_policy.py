"""Train-only probability-calibration evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


METHOD_COMPLEXITY_ORDER = {
    "raw_identity": 0,
    "sigmoid_platt": 1,
    "isotonic": 2,
}


@dataclass
class FittedCalibrator:
    """Small typed wrapper around a fitted calibration mapping."""

    method: str
    estimator: Any | None
    metadata: dict[str, Any]


def _validate_probability_inputs(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)

    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional.")

    if len(score_array) == 0 or len(score_array) != len(label_array):
        raise ValueError("scores and labels must have equal nonzero length.")

    if not np.isfinite(score_array).all():
        raise ValueError("scores contain nonfinite values.")

    if ((score_array < 0.0) | (score_array > 1.0)).any():
        raise ValueError("scores must lie in [0, 1].")

    if not np.isin(label_array, [0, 1]).all():
        raise ValueError("labels must be binary 0/1.")

    if np.unique(label_array).size != 2:
        raise ValueError("both classes are required.")

    return score_array, label_array


def _logit(values: np.ndarray, clip: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), clip, 1.0 - clip)
    return np.log(clipped / (1.0 - clipped))


def fit_probability_calibrator(
    method: str,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 1729,
    clip: float = 1.0e-6,
) -> FittedCalibrator:
    """Fit one candidate calibration mapping on prior-fold OOF scores."""
    score_array, label_array = _validate_probability_inputs(scores, labels)

    if method == "raw_identity":
        return FittedCalibrator(
            method=method,
            estimator=None,
            metadata={
                "method": method,
                "monotonic_direction": "increasing",
                "slope": 1.0,
                "intercept": 0.0,
            },
        )

    if method == "sigmoid_platt":
        estimator = LogisticRegression(
            C=1.0e6,
            solver="lbfgs",
            max_iter=2000,
            random_state=int(seed),
        )
        estimator.fit(_logit(score_array, clip).reshape(-1, 1), label_array)
        slope = float(estimator.coef_[0, 0])
        intercept = float(estimator.intercept_[0])

        return FittedCalibrator(
            method=method,
            estimator=estimator,
            metadata={
                "method": method,
                "slope": slope,
                "intercept": intercept,
                "monotonic_direction": (
                    "increasing" if slope > 0.0 else
                    "decreasing" if slope < 0.0 else
                    "constant"
                ),
            },
        )

    if method == "isotonic":
        estimator = IsotonicRegression(
            y_min=float(clip),
            y_max=float(1.0 - clip),
            out_of_bounds="clip",
            increasing=True,
        )
        estimator.fit(score_array, label_array)

        return FittedCalibrator(
            method=method,
            estimator=estimator,
            metadata={
                "method": method,
                "monotonic_direction": "non_decreasing",
                "threshold_count": int(len(estimator.X_thresholds_)),
            },
        )

    raise ValueError(f"Unknown calibration method: {method}")


def predict_probability_calibrator(
    calibrator: FittedCalibrator,
    scores: np.ndarray,
    *,
    clip: float = 1.0e-6,
) -> np.ndarray:
    """Apply a fitted calibration mapping."""
    score_array = np.asarray(scores, dtype=float)

    if score_array.ndim != 1 or len(score_array) == 0:
        raise ValueError("scores must be a nonempty one-dimensional array.")

    if not np.isfinite(score_array).all():
        raise ValueError("scores contain nonfinite values.")

    if calibrator.method == "raw_identity":
        output = score_array.copy()

    elif calibrator.method == "sigmoid_platt":
        output = calibrator.estimator.predict_proba(
            _logit(score_array, clip).reshape(-1, 1)
        )[:, 1]

    elif calibrator.method == "isotonic":
        output = calibrator.estimator.predict(score_array)

    else:
        raise ValueError(
            f"Unknown fitted calibration method: {calibrator.method}"
        )

    return np.clip(np.asarray(output, dtype=float), clip, 1.0 - clip)


def is_non_decreasing_mapping(
    raw_scores: np.ndarray,
    mapped_scores: np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> bool:
    """Check whether mapped scores are nondecreasing in raw-score order."""
    raw_array = np.asarray(raw_scores, dtype=float)
    mapped_array = np.asarray(mapped_scores, dtype=float)

    if raw_array.shape != mapped_array.shape:
        raise ValueError("raw_scores and mapped_scores must have equal shape.")

    order = np.argsort(raw_array, kind="mergesort")
    sorted_raw = raw_array[order]
    sorted_mapped = mapped_array[order]

    # For tied raw scores, deterministic fitted mappings must be equal.
    tied = np.isclose(np.diff(sorted_raw), 0.0, atol=tolerance, rtol=0.0)
    tied_diff = np.abs(np.diff(sorted_mapped)[tied])
    if tied_diff.size and (tied_diff > tolerance).any():
        return False

    return bool((np.diff(sorted_mapped) >= -tolerance).all())


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Equal-frequency expected calibration error."""
    probability_array, label_array = _validate_probability_inputs(
        probabilities,
        labels,
    )

    order = np.argsort(probability_array, kind="mergesort")
    chunks = np.array_split(order, int(bins))

    ece = 0.0
    total = len(label_array)

    for indices in chunks:
        if len(indices) == 0:
            continue
        observed = float(label_array[indices].mean())
        predicted = float(probability_array[indices].mean())
        ece += (len(indices) / total) * abs(observed - predicted)

    return float(ece)


def probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    ece_bins: int = 10,
) -> dict[str, float]:
    """Return discrimination and calibration diagnostics."""
    probability_array, label_array = _validate_probability_inputs(
        probabilities,
        labels,
    )

    return {
        "roc_auc": float(
            roc_auc_score(label_array, probability_array)
        ),
        "average_precision": float(
            average_precision_score(label_array, probability_array)
        ),
        "brier_score": float(
            brier_score_loss(label_array, probability_array)
        ),
        "log_loss": float(
            log_loss(label_array, probability_array, labels=[0, 1])
        ),
        "expected_calibration_error": float(
            expected_calibration_error(
                label_array,
                probability_array,
                bins=ece_bins,
            )
        ),
        "mean_predicted_score": float(probability_array.mean()),
        "observed_positive_fraction": float(label_array.mean()),
    }
