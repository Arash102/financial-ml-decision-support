"""Anchored walk-forward fold construction on a frozen trading calendar."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


FOLD_DESIGN_SCHEMA_VERSION = (
    "stage05_v2_candidate_event_count_balanced_contiguous"
)


@dataclass(frozen=True)
class WalkForwardFold:
    """One purged anchored walk-forward fold boundary specification."""

    fold_id: int
    calendar_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    embargo_start_date: pd.Timestamp
    embargo_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp
    validation_end_date: pd.Timestamp
    train_end_calendar_index: int
    validation_start_calendar_index: int
    validation_end_calendar_index: int
    embargo_trading_days: int
    validation_target_candidate_events: float
    validation_candidate_events: int
    validation_event_count_deviation: float
    validation_event_count_relative_deviation: float
    validation_horizon_candidate_events: int
    validation_partition_method: str


def normalize_trading_calendar(values: Iterable[object]) -> pd.DatetimeIndex:
    """Return sorted unique valid trading dates."""
    parsed = pd.to_datetime(pd.Series(list(values)), errors="coerce")
    parsed = parsed.dropna().dt.normalize()
    unique_dates = pd.DatetimeIndex(sorted(parsed.unique()))

    if unique_dates.empty:
        raise ValueError("Trading calendar is empty.")

    if not unique_dates.is_monotonic_increasing:
        raise AssertionError("Trading calendar is not monotonic increasing.")

    if unique_dates.has_duplicates:
        raise AssertionError("Trading calendar contains duplicate dates.")

    return unique_dates


def candidate_event_counts_by_calendar_date(
    trading_calendar: pd.DatetimeIndex,
    candidate_start_dates: Iterable[object],
) -> pd.Series:
    """
    Count candidate events on each frozen trading date.

    Only start-date counts are used. No target or event outcome is accepted by
    this function.
    """
    calendar = pd.DatetimeIndex(
        trading_calendar
    ).normalize().unique().sort_values()

    starts = pd.to_datetime(
        pd.Series(list(candidate_start_dates)),
        errors="coerce",
    ).dropna().dt.normalize()

    if starts.empty:
        raise ValueError("Candidate start-date series is empty.")

    outside_calendar = ~starts.isin(calendar)
    if outside_calendar.any():
        examples = (
            starts.loc[outside_calendar]
            .drop_duplicates()
            .sort_values()
            .head(10)
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        raise ValueError(
            "Candidate start dates exist outside the frozen trading calendar: "
            f"{examples}"
        )

    counts = starts.value_counts(sort=False)
    aligned = pd.Series(
        0,
        index=calendar,
        dtype=np.int64,
        name="candidate_event_count",
    )
    aligned.loc[counts.index] = counts.astype(np.int64).to_numpy()

    if int(aligned.sum()) != len(starts):
        raise AssertionError(
            "Candidate event-count alignment lost at least one event."
        )

    return aligned


def _choose_event_balanced_validation_end_indices(
    validation_counts: pd.Series,
    *,
    number_of_folds: int,
) -> list[int]:
    """
    Select contiguous boundary ends nearest cumulative equal-event targets.

    Boundaries are chosen only on whole calendar dates. Consequently all events
    that share one start date remain together and exact equality of fold event
    counts is not guaranteed.
    """
    counts = validation_counts.to_numpy(dtype=np.int64)
    n_dates = len(counts)

    if n_dates < number_of_folds:
        raise ValueError(
            "Validation horizon has fewer calendar dates than requested folds."
        )

    total_events = int(counts.sum())
    if total_events < number_of_folds:
        raise ValueError(
            "Validation horizon has fewer candidate events than requested folds."
        )

    cumulative = np.cumsum(counts)
    target_per_fold = total_events / float(number_of_folds)

    end_indices: list[int] = []
    previous_end = -1

    for fold_number in range(1, number_of_folds):
        start_index = previous_end + 1
        remaining_folds = number_of_folds - fold_number

        # Leave at least one calendar date for every future validation fold.
        latest_end_index = n_dates - remaining_folds - 1

        target_cumulative_events = target_per_fold * fold_number

        candidate_indices = np.arange(
            start_index,
            latest_end_index + 1,
            dtype=int,
        )

        if candidate_indices.size == 0:
            raise ValueError(
                "No admissible calendar boundary remains for event balancing."
            )

        previous_cumulative = (
            int(cumulative[previous_end])
            if previous_end >= 0
            else 0
        )

        block_event_counts = (
            cumulative[candidate_indices] - previous_cumulative
        )
        remaining_event_counts = (
            total_events - cumulative[candidate_indices]
        )

        admissible = (
            (block_event_counts > 0)
            & (remaining_event_counts >= remaining_folds)
        )

        admissible_indices = candidate_indices[admissible]

        if admissible_indices.size == 0:
            raise ValueError(
                "Could not construct non-empty contiguous validation folds "
                "while preserving candidate-event dates."
            )

        distances = np.abs(
            cumulative[admissible_indices]
            - target_cumulative_events
        )

        # np.argmin is stable: ties select the earlier calendar boundary.
        selected_end_index = int(
            admissible_indices[int(np.argmin(distances))]
        )

        end_indices.append(selected_end_index)
        previous_end = selected_end_index

    end_indices.append(n_dates - 1)

    if len(end_indices) != number_of_folds:
        raise AssertionError("Unexpected number of validation boundaries.")

    if end_indices != sorted(end_indices):
        raise AssertionError("Validation boundary indices are not ordered.")

    if len(set(end_indices)) != len(end_indices):
        raise AssertionError("Validation boundary indices are duplicated.")

    return end_indices


def build_anchored_event_balanced_folds(
    trading_calendar: pd.DatetimeIndex,
    candidate_start_dates: Iterable[object],
    *,
    number_of_folds: int,
    first_validation_start_fraction: float,
    embargo_trading_days: int,
) -> list[WalkForwardFold]:
    """
    Build chronological contiguous validation windows with similar event counts.

    The first validation date remains determined by the pre-registered fraction
    of the frozen train-only trading calendar. Only the remaining validation
    horizon is partitioned by candidate-event start-date counts.

    `meta_label` and all event outcomes are intentionally absent from the
    boundary-construction interface.
    """
    calendar = pd.DatetimeIndex(
        trading_calendar
    ).normalize().unique().sort_values()
    n_dates = len(calendar)

    if number_of_folds < 2:
        raise ValueError("number_of_folds must be at least 2.")

    if not 0.0 < first_validation_start_fraction < 1.0:
        raise ValueError(
            "first_validation_start_fraction must be strictly between 0 and 1."
        )

    if embargo_trading_days < 0:
        raise ValueError("embargo_trading_days cannot be negative.")

    first_validation_index = int(
        math.floor(n_dates * first_validation_start_fraction)
    )
    first_validation_index = max(
        first_validation_index,
        embargo_trading_days + 1,
    )

    if first_validation_index >= n_dates:
        raise ValueError(
            "The first validation boundary is outside the trading calendar."
        )

    candidate_counts = candidate_event_counts_by_calendar_date(
        calendar,
        candidate_start_dates,
    )

    validation_counts = candidate_counts.iloc[
        first_validation_index:
    ].copy()

    validation_horizon_candidate_events = int(
        validation_counts.sum()
    )

    if validation_horizon_candidate_events < number_of_folds:
        raise ValueError(
            "Not enough candidate events in the validation horizon."
        )

    local_end_indices = _choose_event_balanced_validation_end_indices(
        validation_counts,
        number_of_folds=number_of_folds,
    )

    target_candidate_events = (
        validation_horizon_candidate_events / float(number_of_folds)
    )

    folds: list[WalkForwardFold] = []
    local_start_index = 0

    for fold_number, local_end_index in enumerate(
        local_end_indices,
        start=1,
    ):
        validation_start_index = (
            first_validation_index + local_start_index
        )
        validation_end_index = (
            first_validation_index + int(local_end_index)
        )

        train_end_index = (
            validation_start_index - embargo_trading_days - 1
        )

        if train_end_index < 0:
            raise ValueError(
                f"Fold {fold_number} has no historical training calendar "
                "after applying the embargo."
            )

        embargo_start_index = train_end_index + 1
        embargo_end_index = validation_start_index - 1

        validation_candidate_events = int(
            candidate_counts.iloc[
                validation_start_index : validation_end_index + 1
            ].sum()
        )

        if validation_candidate_events <= 0:
            raise AssertionError(
                f"Fold {fold_number} contains no validation candidate events."
            )

        deviation = (
            validation_candidate_events - target_candidate_events
        )
        relative_deviation = (
            deviation / target_candidate_events
            if target_candidate_events > 0
            else np.nan
        )

        folds.append(
            WalkForwardFold(
                fold_id=fold_number,
                calendar_start_date=pd.Timestamp(calendar[0]),
                train_end_date=pd.Timestamp(calendar[train_end_index]),
                embargo_start_date=pd.Timestamp(
                    calendar[embargo_start_index]
                ),
                embargo_end_date=pd.Timestamp(
                    calendar[embargo_end_index]
                ),
                validation_start_date=pd.Timestamp(
                    calendar[validation_start_index]
                ),
                validation_end_date=pd.Timestamp(
                    calendar[validation_end_index]
                ),
                train_end_calendar_index=train_end_index,
                validation_start_calendar_index=validation_start_index,
                validation_end_calendar_index=validation_end_index,
                embargo_trading_days=embargo_trading_days,
                validation_target_candidate_events=float(
                    target_candidate_events
                ),
                validation_candidate_events=validation_candidate_events,
                validation_event_count_deviation=float(deviation),
                validation_event_count_relative_deviation=float(
                    relative_deviation
                ),
                validation_horizon_candidate_events=(
                    validation_horizon_candidate_events
                ),
                validation_partition_method=(
                    "candidate_event_count_balanced_contiguous_calendar"
                ),
            )
        )

        local_start_index = int(local_end_index) + 1

    assigned_events = sum(
        fold.validation_candidate_events
        for fold in folds
    )
    if assigned_events != validation_horizon_candidate_events:
        raise AssertionError(
            "Validation folds do not account for the full validation-horizon "
            "candidate population."
        )

    return folds


def folds_to_frame(folds: list[WalkForwardFold]) -> pd.DataFrame:
    """Convert fold dataclasses to a stable table."""
    return pd.DataFrame([asdict(fold) for fold in folds])
