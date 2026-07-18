"""Deterministic train-only signal-policy utilities."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def select_daily_top_fraction(
    frame: pd.DataFrame,
    *,
    score_column: str,
    date_column: str,
    fraction: float,
    minimum_per_date: int = 1,
    symbol_column: str = "symbol",
    event_id_column: str = "event_id",
) -> pd.DataFrame:
    """Select a deterministic top fraction independently on every signal date."""
    required = {
        score_column,
        date_column,
        symbol_column,
        event_id_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Signal-policy columns are missing: {missing}")

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")

    if int(minimum_per_date) < 1:
        raise ValueError("minimum_per_date must be at least one.")

    result = frame.copy()
    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="raise",
    ).dt.normalize()
    result[score_column] = pd.to_numeric(
        result[score_column],
        errors="raise",
    )

    if not np.isfinite(result[score_column].to_numpy(dtype=float)).all():
        raise ValueError("score column contains nonfinite values.")

    result["_original_order"] = np.arange(len(result), dtype=int)

    result = result.sort_values(
        [
            date_column,
            score_column,
            symbol_column,
            event_id_column,
        ],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    result["daily_candidate_count"] = result.groupby(
        date_column,
        sort=False,
    )[event_id_column].transform("size").astype(int)

    result["daily_rank"] = (
        result.groupby(date_column, sort=False).cumcount() + 1
    ).astype(int)

    result["daily_signal_quota"] = result["daily_candidate_count"].map(
        lambda count: min(
            int(count),
            max(
                int(minimum_per_date),
                int(math.ceil(float(fraction) * int(count))),
            ),
        )
    ).astype(int)

    result["selected_signal"] = (
        result["daily_rank"] <= result["daily_signal_quota"]
    )

    cutoffs = (
        result.loc[result["selected_signal"]]
        .groupby(date_column, sort=False)[score_column]
        .min()
        .rename("daily_score_cutoff")
    )
    result = result.merge(
        cutoffs,
        left_on=date_column,
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    result = result.sort_values(
        "_original_order",
        kind="mergesort",
    ).drop(columns=["_original_order"]).reset_index(drop=True)

    return result


def binary_signal_metrics(
    labels: Iterable[int],
    selected: Iterable[bool],
) -> dict[str, float | int]:
    """Classification diagnostics for a selected-signal mask."""
    y = np.asarray(list(labels), dtype=int)
    prediction = np.asarray(list(selected), dtype=bool).astype(int)

    if y.ndim != 1 or prediction.ndim != 1:
        raise ValueError("labels and selected must be one-dimensional.")

    if len(y) == 0 or len(y) != len(prediction):
        raise ValueError("labels and selected must have equal nonzero length.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1.")

    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction,
        labels=[0, 1],
    ).ravel()

    signals = int(prediction.sum())
    positives = int(y.sum())
    negatives = int((y == 0).sum())

    precision = float(tp / signals) if signals else float("nan")
    specificity = float(tn / negatives) if negatives else float("nan")
    sensitivity = float(tp / positives) if positives else float("nan")
    prevalence = float(y.mean())

    return {
        "events": int(len(y)),
        "signals": signals,
        "signal_rate": float(signals / len(y)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "precision": precision,
        "prevalence": prevalence,
        "precision_lift": float(precision - prevalence),
        "precision_ratio": float(precision / prevalence),
        "specificity": specificity,
        "sensitivity": sensitivity,
    }


def evaluate_signal_policy_by_fold(
    selected_frame: pd.DataFrame,
    *,
    fold_column: str = "fold_id",
    label_column: str = "meta_label",
    selected_column: str = "selected_signal",
    date_column: str = "dEven",
) -> pd.DataFrame:
    """Evaluate one already-applied policy independently in each fold."""
    rows: list[dict[str, float | int]] = []

    for fold_id, fold_frame in selected_frame.groupby(
        fold_column,
        sort=True,
    ):
        metrics = binary_signal_metrics(
            fold_frame[label_column].to_numpy(dtype=int),
            fold_frame[selected_column].to_numpy(dtype=bool),
        )
        metrics.update(
            {
                "fold_id": int(fold_id),
                "dates": int(fold_frame[date_column].nunique()),
                "signals_per_date": float(
                    metrics["signals"]
                    / fold_frame[date_column].nunique()
                ),
            }
        )
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values(
        "fold_id",
        kind="stable",
    ).reset_index(drop=True)


def summarize_signal_policy(
    fold_metrics: pd.DataFrame,
    *,
    fraction: float,
    development_folds: Iterable[int],
) -> dict[str, float | int]:
    """Summarize one policy using development folds only."""
    development_set = {int(value) for value in development_folds}
    dev = fold_metrics[
        fold_metrics["fold_id"].isin(development_set)
    ].copy()

    if dev.empty or set(dev["fold_id"].astype(int)) != development_set:
        raise ValueError("Development folds are incomplete.")

    return {
        "signal_fraction": float(fraction),
        "development_fold_count": int(len(dev)),
        "minimum_signals_per_fold": int(dev["signals"].min()),
        "mean_signal_rate": float(dev["signal_rate"].mean()),
        "worst_fold_precision": float(dev["precision"].min()),
        "mean_precision": float(dev["precision"].mean()),
        "worst_fold_precision_lift": float(
            dev["precision_lift"].min()
        ),
        "mean_precision_lift": float(
            dev["precision_lift"].mean()
        ),
        "worst_fold_specificity": float(dev["specificity"].min()),
        "mean_specificity": float(dev["specificity"].mean()),
        "mean_sensitivity": float(dev["sensitivity"].mean()),
    }


def select_signal_policy_hierarchically(
    summary: pd.DataFrame,
    *,
    minimum_signals_per_fold: int,
    minimum_worst_precision_lift: float,
    near_best_tolerance: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply the frozen conservative hierarchical signal-policy rule."""
    required = {
        "signal_fraction",
        "minimum_signals_per_fold",
        "worst_fold_precision_lift",
        "mean_precision_lift",
        "mean_specificity",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise KeyError(f"Policy-summary columns are missing: {missing}")

    ranked = summary.copy()
    ranked["eligible"] = (
        (
            ranked["minimum_signals_per_fold"]
            >= int(minimum_signals_per_fold)
        )
        & (
            ranked["worst_fold_precision_lift"]
            >= float(minimum_worst_precision_lift)
        )
    )

    eligible = ranked[ranked["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("No signal policy satisfies frozen constraints.")

    best_worst_lift = float(
        eligible["worst_fold_precision_lift"].max()
    )
    eligible["within_near_best_worst_lift_band"] = (
        eligible["worst_fold_precision_lift"]
        >= best_worst_lift - float(near_best_tolerance)
    )

    near_best = eligible[
        eligible["within_near_best_worst_lift_band"]
    ].copy()

    near_best = near_best.sort_values(
        [
            "mean_precision_lift",
            "mean_specificity",
            "signal_fraction",
        ],
        ascending=[False, False, True],
        kind="mergesort",
    )

    selected_index = near_best.index[0]
    ranked["within_near_best_worst_lift_band"] = False
    ranked.loc[
        near_best.index,
        "within_near_best_worst_lift_band",
    ] = True
    ranked["selected"] = False
    ranked.loc[selected_index, "selected"] = True

    ranked = ranked.sort_values(
        [
            "selected",
            "eligible",
            "within_near_best_worst_lift_band",
            "worst_fold_precision_lift",
            "mean_precision_lift",
            "mean_specificity",
            "signal_fraction",
        ],
        ascending=[False, False, False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    selected = ranked.loc[ranked["selected"]].iloc[0]
    return selected, ranked
