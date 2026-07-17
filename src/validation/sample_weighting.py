"""Fold-local average uniqueness weights for overlapping financial events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import numpy as np
import pandas as pd


AVERAGE_UNIQUENESS_SCHEMA_VERSION = (
    "stage06_v3_fold_local_within_symbol_average_uniqueness"
)


def normalize_symbol_calendar(values: Iterable[object]) -> pd.DatetimeIndex:
    """Return one sorted, unique, valid symbol trading-observation calendar."""
    parsed = pd.to_datetime(
        pd.Series(list(values)),
        errors="coerce",
    )
    parsed = parsed.dropna().dt.normalize()

    calendar = pd.DatetimeIndex(
        sorted(parsed.unique())
    )

    if calendar.empty:
        raise ValueError("Symbol calendar is empty.")

    if calendar.has_duplicates:
        raise AssertionError("Symbol calendar contains duplicate dates.")

    if not calendar.is_monotonic_increasing:
        raise AssertionError("Symbol calendar is not chronological.")

    return calendar


def compute_fold_train_average_uniqueness(
    events: pd.DataFrame,
    symbol_calendars: Mapping[str, pd.DatetimeIndex],
    *,
    event_id_column: str = "event_id",
    symbol_column: str = "symbol",
    event_start_column: str = "dEven",
    event_end_column: str = "event_end_date",
    weight_source_scope: str = "current_fold_training_events_only",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute average uniqueness from the supplied event population only.

    The caller must pass exactly one fold's retained training events. Concurrency
    is calculated independently inside every symbol and over inclusive
    event-start/event-end intervals on that symbol's own trading-observation
    calendar.

    For event i active over T_i observations:

        average_uniqueness_i
        = (1 / T_i) * sum_t (1 / concurrency_t)

    Normalized sample weights divide raw average uniqueness by the fold-training
    mean so that the final weight mean is one.
    """
    required = {
        event_id_column,
        symbol_column,
        event_start_column,
        event_end_column,
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise KeyError(
            f"Average uniqueness input is missing columns: {missing}"
        )

    frame = events[
        [
            event_id_column,
            symbol_column,
            event_start_column,
            event_end_column,
        ]
    ].copy()

    if frame.empty:
        raise ValueError("Fold-training event population is empty.")

    if frame[event_id_column].duplicated().any():
        raise ValueError(
            "Fold-training event identifiers are not unique."
        )

    frame[event_id_column] = frame[event_id_column].astype(str)
    frame[symbol_column] = frame[symbol_column].astype(str)
    frame[event_start_column] = pd.to_datetime(
        frame[event_start_column],
        errors="coerce",
    ).dt.normalize()
    frame[event_end_column] = pd.to_datetime(
        frame[event_end_column],
        errors="coerce",
    ).dt.normalize()

    invalid_date = (
        frame[event_start_column].isna()
        | frame[event_end_column].isna()
    )
    if invalid_date.any():
        raise ValueError(
            "Fold-training events contain invalid start/end dates."
        )

    invalid_interval = (
        frame[event_end_column] < frame[event_start_column]
    )
    if invalid_interval.any():
        raise ValueError(
            "Fold-training events contain end dates before start dates."
        )

    frame["_input_order"] = np.arange(len(frame), dtype=np.int64)

    event_weight_parts: list[pd.DataFrame] = []
    symbol_audit_rows: list[dict[str, object]] = []

    for symbol, group in frame.groupby(
        symbol_column,
        sort=False,
        observed=True,
    ):
        symbol_key = str(symbol)

        if symbol_key not in symbol_calendars:
            raise KeyError(
                f"Missing labeled-train calendar for symbol {symbol_key}."
            )

        calendar = normalize_symbol_calendar(
            symbol_calendars[symbol_key]
        )

        starts = pd.DatetimeIndex(
            group[event_start_column]
        )
        ends = pd.DatetimeIndex(
            group[event_end_column]
        )

        start_positions = calendar.get_indexer(starts)
        end_positions = calendar.get_indexer(ends)

        missing_start = start_positions < 0
        missing_end = end_positions < 0

        if missing_start.any() or missing_end.any():
            examples = group.loc[
                missing_start | missing_end,
                [
                    event_id_column,
                    event_start_column,
                    event_end_column,
                ],
            ].head(10)
            raise ValueError(
                f"Event boundary is absent from the symbol calendar "
                f"for {symbol_key}: "
                f"{examples.to_dict(orient='records')}"
            )

        if (end_positions < start_positions).any():
            raise AssertionError(
                f"Calendar positions reverse an event interval for "
                f"{symbol_key}."
            )

        difference = np.zeros(
            len(calendar) + 1,
            dtype=np.int64,
        )
        np.add.at(difference, start_positions, 1)
        np.add.at(difference, end_positions + 1, -1)

        concurrency = np.cumsum(difference[:-1])

        if (concurrency < 0).any():
            raise AssertionError(
                f"Negative concurrency detected for {symbol_key}."
            )

        inverse_concurrency = np.divide(
            1.0,
            concurrency,
            out=np.zeros_like(concurrency, dtype=float),
            where=concurrency > 0,
        )

        inverse_prefix = np.concatenate(
            [
                np.array([0.0]),
                np.cumsum(inverse_concurrency),
            ]
        )

        duration = (
            end_positions - start_positions + 1
        ).astype(np.int64)

        raw_uniqueness = (
            inverse_prefix[end_positions + 1]
            - inverse_prefix[start_positions]
        ) / duration

        if not np.isfinite(raw_uniqueness).all():
            raise AssertionError(
                f"Non-finite average uniqueness for {symbol_key}."
            )

        if (
            (raw_uniqueness <= 0.0)
            | (raw_uniqueness > 1.0 + 1.0e-12)
        ).any():
            raise AssertionError(
                f"Average uniqueness outside (0, 1] for {symbol_key}."
            )

        symbol_result = group[
            [
                "_input_order",
                event_id_column,
                symbol_column,
                event_start_column,
                event_end_column,
            ]
        ].copy()
        symbol_result["event_duration_observations"] = duration
        symbol_result["average_uniqueness_raw"] = raw_uniqueness
        symbol_result["symbol_max_concurrency"] = int(
            concurrency.max()
        )
        symbol_result["weight_source_scope"] = str(
            weight_source_scope
        )
        symbol_result["concurrency_scope"] = "within_symbol"
        symbol_result["interval_endpoints_inclusive"] = True

        event_weight_parts.append(symbol_result)

        active_concurrency = concurrency[concurrency > 0]

        symbol_audit_rows.append(
            {
                "symbol": symbol_key,
                "train_events": int(len(group)),
                "calendar_observations": int(len(calendar)),
                "active_calendar_observations": int(
                    len(active_concurrency)
                ),
                "maximum_concurrency": int(
                    active_concurrency.max()
                ),
                "mean_concurrency_on_active_observations": float(
                    active_concurrency.mean()
                ),
                "minimum_raw_average_uniqueness": float(
                    raw_uniqueness.min()
                ),
                "mean_raw_average_uniqueness": float(
                    raw_uniqueness.mean()
                ),
                "maximum_raw_average_uniqueness": float(
                    raw_uniqueness.max()
                ),
                "validation_events_used": 0,
            }
        )

    weights = pd.concat(
        event_weight_parts,
        ignore_index=True,
    ).sort_values(
        "_input_order",
        kind="stable",
    )

    fold_mean_raw_uniqueness = float(
        weights["average_uniqueness_raw"].mean()
    )

    if (
        not np.isfinite(fold_mean_raw_uniqueness)
        or fold_mean_raw_uniqueness <= 0.0
    ):
        raise AssertionError(
            "Invalid fold mean raw average uniqueness."
        )

    weights["sample_weight"] = (
        weights["average_uniqueness_raw"]
        / fold_mean_raw_uniqueness
    )
    weights["fold_mean_raw_average_uniqueness"] = (
        fold_mean_raw_uniqueness
    )

    if not np.isfinite(
        weights["sample_weight"].to_numpy(dtype=float)
    ).all():
        raise AssertionError(
            "Normalized average-uniqueness weights are non-finite."
        )

    if (weights["sample_weight"] <= 0.0).any():
        raise AssertionError(
            "Normalized average-uniqueness weights must be positive."
        )

    if not np.isclose(
        float(weights["sample_weight"].mean()),
        1.0,
        atol=1.0e-12,
        rtol=1.0e-12,
    ):
        raise AssertionError(
            "Normalized fold-training sample weights do not average to one."
        )

    weights = weights.drop(columns=["_input_order"]).reset_index(
        drop=True
    )
    symbol_audit = pd.DataFrame(symbol_audit_rows)

    return weights, symbol_audit


def effective_sample_size(sample_weight: Iterable[float]) -> float:
    """Return Kish effective sample size for positive sample weights."""
    values = np.asarray(list(sample_weight), dtype=float)

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "sample_weight must be a non-empty one-dimensional sequence."
        )

    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError(
            "sample_weight must contain finite positive values."
        )

    denominator = float(np.square(values).sum())
    return float(np.square(values.sum()) / denominator)


def summarize_fold_average_uniqueness(
    weights: pd.DataFrame,
    *,
    fold_id: int,
    validation_events_used: int = 0,
) -> dict[str, object]:
    """Summarize one fold's event-level average-uniqueness weights."""
    required = {
        "event_id",
        "symbol",
        "average_uniqueness_raw",
        "sample_weight",
        "event_duration_observations",
        "symbol_max_concurrency",
    }
    missing = sorted(required - set(weights.columns))
    if missing:
        raise KeyError(
            f"Weight table is missing summary columns: {missing}"
        )

    raw = weights["average_uniqueness_raw"].to_numpy(dtype=float)
    normalized = weights["sample_weight"].to_numpy(dtype=float)

    return {
        "fold_id": int(fold_id),
        "train_events": int(len(weights)),
        "symbols": int(weights["symbol"].nunique()),
        "minimum_raw_average_uniqueness": float(raw.min()),
        "mean_raw_average_uniqueness": float(raw.mean()),
        "median_raw_average_uniqueness": float(
            np.median(raw)
        ),
        "maximum_raw_average_uniqueness": float(raw.max()),
        "minimum_sample_weight": float(normalized.min()),
        "mean_sample_weight": float(normalized.mean()),
        "maximum_sample_weight": float(normalized.max()),
        "sum_sample_weight": float(normalized.sum()),
        "effective_sample_size": effective_sample_size(normalized),
        "effective_sample_fraction": float(
            effective_sample_size(normalized) / len(normalized)
        ),
        "minimum_event_duration_observations": int(
            weights["event_duration_observations"].min()
        ),
        "median_event_duration_observations": float(
            weights["event_duration_observations"].median()
        ),
        "maximum_event_duration_observations": int(
            weights["event_duration_observations"].max()
        ),
        "maximum_symbol_concurrency": int(
            weights["symbol_max_concurrency"].max()
        ),
        "nonfinite_raw_uniqueness": int(
            (~np.isfinite(raw)).sum()
        ),
        "nonfinite_sample_weight": int(
            (~np.isfinite(normalized)).sum()
        ),
        "nonpositive_sample_weight": int(
            (normalized <= 0.0).sum()
        ),
        "validation_events_used": int(validation_events_used),
        "weight_source_scope": "current_fold_training_events_only",
        "concurrency_scope": "within_symbol",
        "normalization": "fold_train_mean_one",
    }


def compute_full_train_average_uniqueness(
    events: pd.DataFrame,
    symbol_calendars: Mapping[str, pd.DatetimeIndex],
    *,
    event_id_column: str = "event_id",
    symbol_column: str = "symbol",
    event_start_column: str = "dEven",
    event_end_column: str = "event_end_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute within-symbol Average Uniqueness on full eligible train events."""
    return compute_fold_train_average_uniqueness(
        events,
        symbol_calendars,
        event_id_column=event_id_column,
        symbol_column=symbol_column,
        event_start_column=event_start_column,
        event_end_column=event_end_column,
        weight_source_scope=(
            "complete_eligible_train_candidate_population"
        ),
    )


def summarize_average_uniqueness_population(
    weights: pd.DataFrame,
    *,
    population_id: str,
    validation_events_used: int = 0,
    unseen_test_events_used: int = 0,
) -> dict[str, object]:
    """Summarize one complete Average Uniqueness population."""
    required = {
        "event_id",
        "symbol",
        "average_uniqueness_raw",
        "sample_weight",
        "event_duration_observations",
        "symbol_max_concurrency",
        "weight_source_scope",
    }
    missing = sorted(required - set(weights.columns))
    if missing:
        raise KeyError(f"Weight table is missing columns: {missing}")
    if weights.empty:
        raise ValueError("Average Uniqueness weight table is empty.")

    raw = weights["average_uniqueness_raw"].to_numpy(dtype=float)
    normalized = weights["sample_weight"].to_numpy(dtype=float)
    scopes = weights["weight_source_scope"].astype(str).unique().tolist()
    if len(scopes) != 1:
        raise AssertionError("Weight table has more than one source scope.")

    ess = effective_sample_size(normalized)
    return {
        "population_id": str(population_id),
        "events": int(len(weights)),
        "symbols": int(weights["symbol"].nunique()),
        "minimum_raw_average_uniqueness": float(raw.min()),
        "mean_raw_average_uniqueness": float(raw.mean()),
        "median_raw_average_uniqueness": float(np.median(raw)),
        "maximum_raw_average_uniqueness": float(raw.max()),
        "minimum_sample_weight": float(normalized.min()),
        "mean_sample_weight": float(normalized.mean()),
        "maximum_sample_weight": float(normalized.max()),
        "sum_sample_weight": float(normalized.sum()),
        "effective_sample_size": float(ess),
        "effective_sample_fraction": float(ess / len(normalized)),
        "minimum_event_duration_observations": int(
            weights["event_duration_observations"].min()
        ),
        "median_event_duration_observations": float(
            weights["event_duration_observations"].median()
        ),
        "maximum_event_duration_observations": int(
            weights["event_duration_observations"].max()
        ),
        "maximum_symbol_concurrency": int(
            weights["symbol_max_concurrency"].max()
        ),
        "nonfinite_raw_uniqueness": int((~np.isfinite(raw)).sum()),
        "nonfinite_sample_weight": int((~np.isfinite(normalized)).sum()),
        "nonpositive_sample_weight": int((normalized <= 0.0).sum()),
        "validation_events_used": int(validation_events_used),
        "unseen_test_events_used": int(unseen_test_events_used),
        "weight_source_scope": scopes[0],
        "concurrency_scope": "within_symbol",
        "normalization": "full_train_mean_one",
    }
