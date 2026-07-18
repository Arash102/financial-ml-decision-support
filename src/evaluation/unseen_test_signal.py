"""Confirmatory unseen-test signal-level evaluation helpers."""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


UNSEEN_TEST_SIGNAL_SCHEMA_VERSION = (
    "stage09_v1_confirmatory_unseen_test_signal_level"
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def gross_event_outcome_metrics(
    frame: pd.DataFrame,
    *,
    label_column: str = "meta_label",
    return_column: str = "event_return",
    holding_column: str = "holding_period_observations",
    barrier_column: str = "barrier_touched",
) -> dict[str, float | int]:
    """Summarize label-horizon gross event outcomes without portfolio claims."""
    required = {
        label_column,
        return_column,
        holding_column,
        barrier_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Outcome columns are missing: {missing}")
    if frame.empty:
        raise ValueError("Outcome frame is empty.")

    labels = pd.to_numeric(frame[label_column], errors="raise").astype(int)
    returns = pd.to_numeric(frame[return_column], errors="coerce")
    holding = pd.to_numeric(frame[holding_column], errors="coerce")

    if not labels.isin([0, 1]).all():
        raise ValueError("Outcome labels must be binary 0/1.")

    finite_returns = returns[np.isfinite(returns.to_numpy(dtype=float))]
    positive_returns = finite_returns[finite_returns > 0.0]
    negative_returns = finite_returns[finite_returns < 0.0]
    zero_returns = finite_returns[finite_returns == 0.0]

    average_gain = (
        float(positive_returns.mean())
        if len(positive_returns)
        else float("nan")
    )
    average_loss = (
        float(negative_returns.mean())
        if len(negative_returns)
        else float("nan")
    )

    gross_gain = float(positive_returns.sum())
    gross_loss_magnitude = float(-negative_returns.sum())
    barrier_counts = (
        frame[barrier_column]
        .astype("string")
        .value_counts(dropna=False)
        .to_dict()
    )
    finite_holding = holding[np.isfinite(holding.to_numpy(dtype=float))]

    return {
        "events": int(len(frame)),
        "positive_labels": int(labels.sum()),
        "negative_labels": int((labels == 0).sum()),
        "win_rate": float(labels.mean()),
        "finite_event_returns": int(len(finite_returns)),
        "positive_event_returns": int(len(positive_returns)),
        "negative_event_returns": int(len(negative_returns)),
        "zero_event_returns": int(len(zero_returns)),
        "mean_event_return": float(finite_returns.mean()),
        "median_event_return": float(finite_returns.median()),
        "event_return_standard_deviation": float(
            finite_returns.std(ddof=0)
        ),
        "minimum_event_return": float(finite_returns.min()),
        "maximum_event_return": float(finite_returns.max()),
        "average_positive_event_return": average_gain,
        "average_negative_event_return": average_loss,
        "payoff_ratio": _safe_ratio(average_gain, abs(average_loss)),
        "gross_profit_factor": _safe_ratio(
            gross_gain,
            gross_loss_magnitude,
        ),
        "mean_holding_period_observations": float(
            finite_holding.mean()
        ),
        "median_holding_period_observations": float(
            finite_holding.median()
        ),
        "minimum_holding_period_observations": float(
            finite_holding.min()
        ),
        "maximum_holding_period_observations": float(
            finite_holding.max()
        ),
        "upper_barrier_events": int(barrier_counts.get("upper", 0)),
        "lower_barrier_events": int(barrier_counts.get("lower", 0)),
        "vertical_barrier_events": int(
            barrier_counts.get("vertical", 0)
        ),
    }


def grouped_gross_event_outcome_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    selected_column: str | None = None,
) -> pd.DataFrame:
    """Return gross event-outcome metrics by a deterministic grouping."""
    if group_column not in frame.columns:
        raise KeyError(f"Grouping column is missing: {group_column}")

    source = frame
    if selected_column is not None:
        if selected_column not in frame.columns:
            raise KeyError(f"Selection column is missing: {selected_column}")
        source = frame.loc[frame[selected_column].astype(bool)].copy()

    rows: list[dict[str, object]] = []
    for group_value, group_frame in source.groupby(group_column, sort=True):
        row = gross_event_outcome_metrics(group_frame)
        row[group_column] = group_value
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    columns = [group_column] + [
        column for column in rows[0] if column != group_column
    ]
    return pd.DataFrame(rows)[columns]


def selected_vs_candidate_outcome_summary(
    frame: pd.DataFrame,
    *,
    selected_column: str = "selected_signal",
) -> pd.DataFrame:
    """Compare frozen selected signals with the full candidate population."""
    if selected_column not in frame.columns:
        raise KeyError(f"Selection column is missing: {selected_column}")

    rows: list[dict[str, object]] = []
    for population_name, population_frame in [
        ("all_unseen_test_candidates", frame),
        (
            "frozen_policy_selected_signals",
            frame.loc[frame[selected_column].astype(bool)],
        ),
    ]:
        row = gross_event_outcome_metrics(population_frame)
        row["population"] = population_name
        rows.append(row)

    result = pd.DataFrame(rows)
    columns = ["population"] + [
        column for column in result.columns if column != "population"
    ]
    return result[columns]
