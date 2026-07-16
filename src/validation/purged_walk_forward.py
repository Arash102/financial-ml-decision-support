"""Purged anchored walk-forward membership and integrity audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.validation.folds import WalkForwardFold


REQUIRED_EVENT_COLUMNS = {
    "event_id",
    "symbol",
    "dEven",
    "event_end_date",
    "meta_label",
}


@dataclass(frozen=True)
class FoldMasks:
    """Boolean membership masks for one fold."""

    historical_before_embargo: pd.Series
    train: pd.Series
    purged_for_event_overlap: pd.Series
    embargo: pd.Series
    validation: pd.Series


def normalize_candidate_event_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the candidate-event panel."""
    missing = sorted(REQUIRED_EVENT_COLUMNS - set(panel.columns))
    if missing:
        raise KeyError(f"Candidate panel is missing columns: {missing}")

    result = panel.copy()
    result["event_id"] = result["event_id"].astype(str)
    result["symbol"] = result["symbol"].astype(str)
    result["dEven"] = pd.to_datetime(result["dEven"], errors="coerce")
    result["event_end_date"] = pd.to_datetime(
        result["event_end_date"],
        errors="coerce",
    )
    result["meta_label"] = pd.to_numeric(
        result["meta_label"],
        errors="coerce",
    )

    if result["event_id"].duplicated().any():
        duplicates = result.loc[
            result["event_id"].duplicated(keep=False),
            "event_id",
        ].head(10).tolist()
        raise ValueError(f"Duplicate event IDs found: {duplicates}")

    if result["dEven"].isna().any():
        raise ValueError("Candidate event-start dates contain invalid values.")

    if result["event_end_date"].isna().any():
        raise ValueError("Candidate event-end dates contain invalid values.")

    if (~result["meta_label"].isin([0, 1])).any():
        invalid = sorted(
            result.loc[
                ~result["meta_label"].isin([0, 1]),
                "meta_label",
            ].dropna().unique().tolist()
        )
        raise ValueError(f"meta_label must be binary. Invalid values: {invalid}")

    if (result["event_end_date"] < result["dEven"]).any():
        raise ValueError("At least one event ends before its event start.")

    return result.sort_values(
        ["dEven", "symbol", "event_id"],
        kind="stable",
    ).reset_index(drop=True)


def fold_membership_masks(
    panel: pd.DataFrame,
    fold: WalkForwardFold,
) -> FoldMasks:
    """Build train, purge, embargo, and validation masks for one fold."""
    start = panel["dEven"]
    event_end = panel["event_end_date"]

    historical_before_embargo = start.le(fold.train_end_date)

    purged_for_event_overlap = (
        historical_before_embargo
        & event_end.ge(fold.validation_start_date)
    )

    train = (
        historical_before_embargo
        & event_end.lt(fold.validation_start_date)
    )

    embargo = (
        start.ge(fold.embargo_start_date)
        & start.le(fold.embargo_end_date)
    )

    validation = (
        start.ge(fold.validation_start_date)
        & start.le(fold.validation_end_date)
    )

    return FoldMasks(
        historical_before_embargo=historical_before_embargo,
        train=train,
        purged_for_event_overlap=purged_for_event_overlap,
        embargo=embargo,
        validation=validation,
    )


def summarize_fold(
    panel: pd.DataFrame,
    fold: WalkForwardFold,
    masks: FoldMasks,
) -> dict[str, object]:
    """Summarize one fold without fitting a model."""
    train = panel.loc[masks.train]
    validation = panel.loc[masks.validation]
    embargo = panel.loc[masks.embargo]

    train_positive = int(train["meta_label"].eq(1).sum())
    validation_positive = int(validation["meta_label"].eq(1).sum())

    return {
        "fold_id": fold.fold_id,
        "train_start_date": (
            train["dEven"].min() if len(train) else pd.NaT
        ),
        "train_end_date": fold.train_end_date,
        "embargo_start_date": fold.embargo_start_date,
        "embargo_end_date": fold.embargo_end_date,
        "validation_start_date": fold.validation_start_date,
        "validation_end_date": fold.validation_end_date,
        "historical_events_before_embargo": int(
            masks.historical_before_embargo.sum()
        ),
        "purged_event_overlap_count": int(
            masks.purged_for_event_overlap.sum()
        ),
        "embargo_candidate_event_count": int(masks.embargo.sum()),
        "train_events": int(len(train)),
        "train_positive_labels": train_positive,
        "train_negative_labels": int(
            train["meta_label"].eq(0).sum()
        ),
        "train_positive_fraction": (
            train_positive / len(train) if len(train) else np.nan
        ),
        "validation_events": int(len(validation)),
        "validation_positive_labels": validation_positive,
        "validation_negative_labels": int(
            validation["meta_label"].eq(0).sum()
        ),
        "validation_positive_fraction": (
            validation_positive / len(validation)
            if len(validation)
            else np.nan
        ),
        "train_symbols": int(train["symbol"].nunique()),
        "validation_symbols": int(validation["symbol"].nunique()),
        "train_unique_event_start_dates": int(
            train["dEven"].nunique()
        ),
        "validation_unique_event_start_dates": int(
            validation["dEven"].nunique()
        ),
    }


def audit_fold_integrity(
    panel: pd.DataFrame,
    fold: WalkForwardFold,
    masks: FoldMasks,
) -> dict[str, object]:
    """Compute explicit leakage and boundary checks for one fold."""
    train = panel.loc[masks.train]
    validation = panel.loc[masks.validation]

    event_id_overlap = set(train["event_id"]).intersection(
        set(validation["event_id"])
    )
    event_start_date_overlap = set(
        train["dEven"].dt.normalize()
    ).intersection(
        set(validation["dEven"].dt.normalize())
    )

    return {
        "fold_id": fold.fold_id,
        "train_event_ids_in_validation": len(event_id_overlap),
        "train_validation_event_start_date_overlap": len(
            event_start_date_overlap
        ),
        "train_start_after_train_end": int(
            train["dEven"].gt(fold.train_end_date).sum()
        ),
        "train_event_end_on_or_after_validation_start": int(
            train["event_end_date"]
            .ge(fold.validation_start_date)
            .sum()
        ),
        "validation_before_validation_start": int(
            validation["dEven"]
            .lt(fold.validation_start_date)
            .sum()
        ),
        "validation_after_validation_end": int(
            validation["dEven"]
            .gt(fold.validation_end_date)
            .sum()
        ),
        "embargo_events_in_train": int(
            (masks.embargo & masks.train).sum()
        ),
        "purged_events_in_train": int(
            (masks.purged_for_event_overlap & masks.train).sum()
        ),
        "train_has_both_classes": bool(
            train["meta_label"].nunique() == 2
        ),
        "validation_has_both_classes": bool(
            validation["meta_label"].nunique() == 2
        ),
    }


def audit_anchored_training_sets(
    panel: pd.DataFrame,
    folds: list[WalkForwardFold],
) -> pd.DataFrame:
    """Require each earlier training event set to be a subset of the next."""
    rows: list[dict[str, object]] = []
    previous_ids: set[str] | None = None
    previous_fold_id: int | None = None

    for fold in folds:
        masks = fold_membership_masks(panel, fold)
        current_ids = set(panel.loc[masks.train, "event_id"].astype(str))

        if previous_ids is None:
            subset_ok = True
            missing_from_current = 0
        else:
            missing = previous_ids - current_ids
            subset_ok = not missing
            missing_from_current = len(missing)

        rows.append(
            {
                "previous_fold_id": previous_fold_id,
                "current_fold_id": fold.fold_id,
                "anchored_subset_ok": subset_ok,
                "previous_train_events_missing_from_current": (
                    missing_from_current
                ),
                "current_train_events": len(current_ids),
            }
        )

        previous_ids = current_ids
        previous_fold_id = fold.fold_id

    return pd.DataFrame(rows)


def audit_validation_window_overlap(
    folds: list[WalkForwardFold],
) -> pd.DataFrame:
    """Audit overlap between all pairs of validation date windows."""
    rows: list[dict[str, object]] = []

    for left_index, left in enumerate(folds):
        for right in folds[left_index + 1 :]:
            overlap = not (
                left.validation_end_date < right.validation_start_date
                or right.validation_end_date < left.validation_start_date
            )
            rows.append(
                {
                    "left_fold_id": left.fold_id,
                    "right_fold_id": right.fold_id,
                    "validation_windows_overlap": overlap,
                }
            )

    return pd.DataFrame(rows)



def audit_validation_event_count_balance(
    panel: pd.DataFrame,
    folds: list[WalkForwardFold],
) -> pd.DataFrame:
    """Audit approximate event-count balance without using target labels."""
    rows: list[dict[str, object]] = []

    total_validation_events = 0
    target_values: list[float] = []

    for fold in folds:
        masks = fold_membership_masks(panel, fold)
        actual_events = int(masks.validation.sum())
        total_validation_events += actual_events
        target_values.append(
            float(fold.validation_target_candidate_events)
        )

        rows.append(
            {
                "fold_id": fold.fold_id,
                "validation_start_date": fold.validation_start_date,
                "validation_end_date": fold.validation_end_date,
                "target_candidate_events": float(
                    fold.validation_target_candidate_events
                ),
                "actual_candidate_events": actual_events,
                "absolute_deviation_from_target": abs(
                    actual_events
                    - float(fold.validation_target_candidate_events)
                ),
                "signed_deviation_from_target": (
                    actual_events
                    - float(fold.validation_target_candidate_events)
                ),
                "relative_deviation_from_target": (
                    (
                        actual_events
                        - float(fold.validation_target_candidate_events)
                    )
                    / float(fold.validation_target_candidate_events)
                    if fold.validation_target_candidate_events > 0
                    else np.nan
                ),
                "validation_unique_start_dates": int(
                    panel.loc[masks.validation, "dEven"].nunique()
                ),
                "boundary_used_meta_label": False,
            }
        )

    result = pd.DataFrame(rows)

    if len(result):
        event_counts = result["actual_candidate_events"].astype(float)
        mean_events = float(event_counts.mean())
        std_events = float(event_counts.std(ddof=0))

        result["all_fold_event_count_mean"] = mean_events
        result["all_fold_event_count_std"] = std_events
        result["all_fold_event_count_cv"] = (
            std_events / mean_events
            if mean_events > 0
            else np.nan
        )
        result["all_fold_event_count_min"] = int(event_counts.min())
        result["all_fold_event_count_max"] = int(event_counts.max())
        result["all_fold_max_to_min_ratio"] = (
            float(event_counts.max() / event_counts.min())
            if event_counts.min() > 0
            else np.nan
        )
        result["total_validation_candidate_events"] = (
            total_validation_events
        )
        result["common_target_candidate_events"] = (
            float(np.mean(target_values))
        )

    return result
