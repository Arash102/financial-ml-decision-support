"""Calendar-based anchored walk-forward fold construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


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


def build_anchored_calendar_folds(
    trading_calendar: pd.DatetimeIndex,
    *,
    number_of_folds: int,
    first_validation_start_fraction: float,
    embargo_trading_days: int,
) -> list[WalkForwardFold]:
    """
    Split the latter calendar segment into contiguous validation windows.

    The first validation block begins at a pre-registered fraction of the
    train-only trading calendar. Every fold uses all eligible historical events
    ending before the validation start, subject to a conservative pre-validation
    embargo gap.
    """
    calendar = pd.DatetimeIndex(trading_calendar).normalize().unique().sort_values()
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
    first_validation_index = max(first_validation_index, embargo_trading_days + 1)

    remaining_indices = np.arange(first_validation_index, n_dates, dtype=int)

    if len(remaining_indices) < number_of_folds:
        raise ValueError(
            "Not enough trading dates after the first validation boundary "
            "to create the requested folds."
        )

    validation_blocks = np.array_split(remaining_indices, number_of_folds)
    folds: list[WalkForwardFold] = []

    for fold_number, block in enumerate(validation_blocks, start=1):
        if len(block) == 0:
            raise AssertionError("Validation block cannot be empty.")

        validation_start_index = int(block[0])
        validation_end_index = int(block[-1])
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

        embargo_start_date = calendar[embargo_start_index]
        embargo_end_date = calendar[embargo_end_index]

        folds.append(
            WalkForwardFold(
                fold_id=fold_number,
                calendar_start_date=pd.Timestamp(calendar[0]),
                train_end_date=pd.Timestamp(calendar[train_end_index]),
                embargo_start_date=pd.Timestamp(embargo_start_date),
                embargo_end_date=pd.Timestamp(embargo_end_date),
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
            )
        )

    return folds


def folds_to_frame(folds: list[WalkForwardFold]) -> pd.DataFrame:
    """Convert fold dataclasses to a stable table."""
    return pd.DataFrame([asdict(fold) for fold in folds])
