"""Censoring summaries and model-eligibility controls."""

from __future__ import annotations

from typing import Any

import pandas as pd


LABEL_AUDIT_COLUMNS = [
    "symbol",
    "partition",
    "rows",
    "eligible_events",
    "excluded_events",
    "positive_labels",
    "negative_labels",
    "positive_label_fraction",
    "upper_barrier_events",
    "lower_barrier_events",
    "vertical_barrier_events",
    "same_bar_double_touch_events",
    "right_censored_events",
    "invalid_entry_price_events",
    "invalid_monitoring_price_events",
    "invalid_vertical_price_events",
    "first_event_start_date",
    "last_event_start_date",
    "last_event_end_date",
]


def build_labeling_audit(
    labeled: pd.DataFrame,
    symbol: str,
    partition: str,
) -> dict[str, Any]:
    """Summarize labels, censoring, and invalid-event exclusions."""
    required = {
        "event_start_date",
        "event_end_date",
        "label",
        "label_status",
        "barrier_touched",
        "same_bar_double_touch",
        "eligible_for_modeling",
    }
    missing = sorted(required.difference(labeled.columns))
    if missing:
        raise KeyError(f"Missing labeling-audit columns: {missing}")

    eligible = labeled["eligible_for_modeling"].fillna(False).astype(bool)
    eligible_labels = labeled.loc[eligible, "label"]
    positive = int(eligible_labels.eq(1).sum())
    negative = int(eligible_labels.eq(0).sum())
    eligible_count = int(eligible.sum())

    return {
        "symbol": symbol,
        "partition": partition,
        "rows": int(len(labeled)),
        "eligible_events": eligible_count,
        "excluded_events": int((~eligible).sum()),
        "positive_labels": positive,
        "negative_labels": negative,
        "positive_label_fraction": (
            float(positive / eligible_count) if eligible_count else float("nan")
        ),
        "upper_barrier_events": int(
            labeled["barrier_touched"].eq("upper").sum()
        ),
        "lower_barrier_events": int(
            labeled["barrier_touched"].eq("lower").sum()
        ),
        "vertical_barrier_events": int(
            labeled["barrier_touched"].eq("vertical").sum()
        ),
        "same_bar_double_touch_events": int(
            labeled["same_bar_double_touch"].fillna(False).astype(bool).sum()
        ),
        "right_censored_events": int(
            labeled["label_status"].eq("right_censored").sum()
        ),
        "invalid_entry_price_events": int(
            labeled["label_status"].eq("invalid_entry_price").sum()
        ),
        "invalid_monitoring_price_events": int(
            labeled["label_status"].eq("invalid_monitoring_price").sum()
        ),
        "invalid_vertical_price_events": int(
            labeled["label_status"].eq("invalid_vertical_price").sum()
        ),
        "first_event_start_date": pd.to_datetime(
            labeled["event_start_date"], errors="coerce"
        ).min(),
        "last_event_start_date": pd.to_datetime(
            labeled["event_start_date"], errors="coerce"
        ).max(),
        "last_event_end_date": pd.to_datetime(
            labeled["event_end_date"], errors="coerce"
        ).max(),
    }


def modeling_sample(labeled: pd.DataFrame) -> pd.DataFrame:
    """Return only uncensored, valid events with a binary label."""
    eligible = labeled["eligible_for_modeling"].fillna(False).astype(bool)
    sample = labeled.loc[eligible].copy()

    if sample["label"].isna().any():
        raise ValueError("Eligible modeling events contain missing labels.")
    if not sample["label"].isin([0, 1]).all():
        raise ValueError("Eligible modeling events contain non-binary labels.")
    if not sample["label_status"].eq("labeled").all():
        raise ValueError("Eligible modeling events contain non-labeled statuses.")

    return sample
