"""Event-end integrity controls used by purged validation and labeling audits."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_event_end_integrity_audit(
    labeled: pd.DataFrame,
    symbol: str,
    partition: str,
    max_holding_period: int,
    date_column: str = "dEven",
) -> dict[str, Any]:
    """Return symbol-level event-span integrity checks."""
    required = {
        date_column,
        "event_start_date",
        "event_end_date",
        "holding_period_observations",
        "barrier_touched",
        "label_status",
        "eligible_for_modeling",
        "label",
        "same_bar_double_touch",
    }
    missing = sorted(required.difference(labeled.columns))
    if missing:
        raise KeyError(f"Missing event-end columns: {missing}")

    frame = labeled.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame["event_start_date"] = pd.to_datetime(
        frame["event_start_date"], errors="coerce"
    )
    frame["event_end_date"] = pd.to_datetime(
        frame["event_end_date"], errors="coerce"
    )

    eligible = frame["eligible_for_modeling"].fillna(False).astype(bool)
    labeled_status = frame["label_status"].eq("labeled")
    right_censored = frame["label_status"].eq("right_censored")
    barrier_events = frame["barrier_touched"].isin(["upper", "lower"])
    vertical_events = frame["barrier_touched"].eq("vertical")
    double_touch = frame["same_bar_double_touch"].fillna(False).astype(bool)

    end_before_start = (
        frame["event_end_date"].notna()
        & frame["event_start_date"].notna()
        & (frame["event_end_date"] < frame["event_start_date"])
    )
    end_after_partition = (
        frame["event_end_date"].notna()
        & (frame["event_end_date"] > frame[date_column].max())
    )
    eligible_missing_end = eligible & frame["event_end_date"].isna()
    eligible_invalid_label = eligible & ~frame["label"].isin([0, 1])
    status_eligibility_mismatch = eligible.ne(labeled_status)
    right_censored_eligible = right_censored & eligible
    barrier_holding_invalid = barrier_events & (
        frame["holding_period_observations"].isna()
        | (frame["holding_period_observations"] < 1)
        | (frame["holding_period_observations"] > max_holding_period)
    )
    vertical_holding_invalid = vertical_events & (
        frame["holding_period_observations"] != max_holding_period
    )
    double_touch_rule_invalid = double_touch & (
        ~frame["barrier_touched"].eq("lower") | ~frame["label"].eq(0)
    )

    return {
        "symbol": symbol,
        "partition": partition,
        "rows": int(len(frame)),
        "partition_first_date": frame[date_column].min(),
        "partition_last_date": frame[date_column].max(),
        "eligible_events": int(eligible.sum()),
        "right_censored_events": int(right_censored.sum()),
        "end_before_start_count": int(end_before_start.sum()),
        "event_end_after_partition_count": int(end_after_partition.sum()),
        "eligible_missing_event_end_count": int(eligible_missing_end.sum()),
        "eligible_invalid_label_count": int(eligible_invalid_label.sum()),
        "status_eligibility_mismatch_count": int(
            status_eligibility_mismatch.sum()
        ),
        "right_censored_eligible_count": int(right_censored_eligible.sum()),
        "barrier_holding_invalid_count": int(barrier_holding_invalid.sum()),
        "vertical_holding_invalid_count": int(vertical_holding_invalid.sum()),
        "double_touch_rule_invalid_count": int(double_touch_rule_invalid.sum()),
        "integrity_passed": bool(
            not end_before_start.any()
            and not end_after_partition.any()
            and not eligible_missing_end.any()
            and not eligible_invalid_label.any()
            and not status_eligibility_mismatch.any()
            and not right_censored_eligible.any()
            and not barrier_holding_invalid.any()
            and not vertical_holding_invalid.any()
            and not double_touch_rule_invalid.any()
        ),
    }
