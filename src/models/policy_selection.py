"""Fixed daily top-fraction policy evaluation and policy-first selection.

The helpers in this module are train-only evaluation utilities for Stage 06.
They do not select a probability threshold. They apply a frozen daily
cross-sectional quota:

- score descending;
- symbol ascending;
- event_id ascending;
- quota = max(minimum_signals_per_date, ceil(candidate_count * fraction)).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


POLICY_SELECTION_SCHEMA_VERSION = (
    "stage06_v4_daily_top_fraction_false_positive_priority"
)


@dataclass(frozen=True)
class DailyTopFractionPolicy:
    """Frozen train-only policy used for operational diagnostics."""

    fraction: float = 0.05
    minimum_signals_per_date: int = 1
    date_column: str = "dEven"
    score_column: str = "probability_positive"
    label_column: str = "meta_label"
    symbol_column: str = "symbol"
    event_id_column: str = "event_id"

    def validate(self) -> None:
        if not 0.0 < float(self.fraction) <= 1.0:
            raise ValueError("fraction must be in (0, 1].")
        if int(self.minimum_signals_per_date) < 1:
            raise ValueError(
                "minimum_signals_per_date must be at least 1."
            )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def apply_daily_top_fraction_policy(
    predictions: pd.DataFrame,
    *,
    policy: DailyTopFractionPolicy | None = None,
) -> pd.DataFrame:
    """Rank predictions within each date and apply the fixed quota."""
    policy = policy or DailyTopFractionPolicy()
    policy.validate()

    required = {
        policy.date_column,
        policy.score_column,
        policy.label_column,
        policy.symbol_column,
        policy.event_id_column,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise KeyError(
            f"Policy predictions are missing columns: {missing}"
        )

    frame = predictions.copy()
    frame[policy.date_column] = pd.to_datetime(
        frame[policy.date_column],
        errors="coerce",
    ).dt.normalize()
    frame[policy.score_column] = pd.to_numeric(
        frame[policy.score_column],
        errors="coerce",
    )
    frame[policy.label_column] = pd.to_numeric(
        frame[policy.label_column],
        errors="coerce",
    )

    if frame[policy.date_column].isna().any():
        raise ValueError("Policy predictions contain invalid dates.")
    if not np.isfinite(
        frame[policy.score_column].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Policy predictions contain non-finite scores."
        )
    if not frame[policy.score_column].between(0.0, 1.0).all():
        raise ValueError(
            "Policy prediction scores must lie in [0, 1]."
        )
    if not frame[policy.label_column].isin([0, 1]).all():
        raise ValueError("Policy labels must be binary 0/1.")
    if frame[policy.event_id_column].duplicated().any():
        raise ValueError(
            "Policy predictions contain duplicate event IDs."
        )

    frame["_original_policy_row_order"] = np.arange(
        len(frame),
        dtype=int,
    )
    frame = frame.sort_values(
        [
            policy.date_column,
            policy.score_column,
            policy.symbol_column,
            policy.event_id_column,
        ],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    grouped = frame.groupby(
        policy.date_column,
        sort=False,
        observed=False,
    )
    frame["daily_candidate_count"] = grouped[
        policy.event_id_column
    ].transform("size").astype(int)
    frame["daily_signal_quota"] = np.maximum(
        int(policy.minimum_signals_per_date),
        np.ceil(
            frame["daily_candidate_count"].to_numpy(dtype=float)
            * float(policy.fraction)
        ).astype(int),
    )
    frame["daily_rank"] = grouped.cumcount() + 1
    frame["selected_signal"] = (
        frame["daily_rank"] <= frame["daily_signal_quota"]
    )

    selected_cutoff = (
        frame.loc[
            frame["selected_signal"],
            [policy.date_column, policy.score_column],
        ]
        .groupby(
            policy.date_column,
            sort=False,
            observed=False,
        )[policy.score_column]
        .min()
        .rename("daily_score_cutoff")
    )
    frame = frame.merge(
        selected_cutoff,
        left_on=policy.date_column,
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    if frame["daily_score_cutoff"].isna().any():
        raise AssertionError(
            "Every date must receive at least one selected signal."
        )
    if not frame.groupby(
        policy.date_column,
        observed=False,
    )["selected_signal"].sum().eq(
        frame.groupby(
            policy.date_column,
            observed=False,
        )["daily_signal_quota"].first()
    ).all():
        raise AssertionError(
            "Selected signal counts do not equal frozen daily quotas."
        )

    return frame.drop(
        columns=["_original_policy_row_order"]
    )


def _confusion_counts(
    labels: np.ndarray,
    selected: np.ndarray,
) -> dict[str, int]:
    y = np.asarray(labels, dtype=int)
    signal = np.asarray(selected, dtype=bool)

    return {
        "true_positive": int(np.sum(signal & (y == 1))),
        "false_positive": int(np.sum(signal & (y == 0))),
        "true_negative": int(np.sum((~signal) & (y == 0))),
        "false_negative": int(np.sum((~signal) & (y == 1))),
    }


def summarize_policy_predictions(
    policy_predictions: pd.DataFrame,
    *,
    policy: DailyTopFractionPolicy | None = None,
    fold_column: str = "fold_id",
) -> tuple[dict[str, object], pd.DataFrame]:
    """Return pooled and fold-level fixed-policy metrics."""
    policy = policy or DailyTopFractionPolicy()
    required = {
        policy.date_column,
        policy.label_column,
        "selected_signal",
        "daily_signal_quota",
    }
    missing = sorted(required - set(policy_predictions.columns))
    if missing:
        raise KeyError(
            f"Policy evaluation is missing columns: {missing}"
        )

    frame = policy_predictions.copy()
    counts = _confusion_counts(
        frame[policy.label_column].to_numpy(dtype=int),
        frame["selected_signal"].to_numpy(dtype=bool),
    )
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]

    signal_count = int(frame["selected_signal"].sum())
    dates = int(frame[policy.date_column].nunique())
    dates_with_signal = int(
        frame.loc[
            frame["selected_signal"],
            policy.date_column,
        ].nunique()
    )
    expected_signal_count = int(
        frame.groupby(
            policy.date_column,
            observed=False,
        )["daily_signal_quota"].first().sum()
    )

    fold_rows: list[dict[str, object]] = []
    if fold_column in frame.columns:
        for fold_id, group in frame.groupby(
            fold_column,
            sort=True,
            observed=False,
        ):
            fold_counts = _confusion_counts(
                group[policy.label_column].to_numpy(dtype=int),
                group["selected_signal"].to_numpy(dtype=bool),
            )
            fold_tp = fold_counts["true_positive"]
            fold_fp = fold_counts["false_positive"]
            fold_tn = fold_counts["true_negative"]
            fold_fn = fold_counts["false_negative"]
            fold_rows.append(
                {
                    "fold_id": int(fold_id),
                    **fold_counts,
                    "signals": int(
                        group["selected_signal"].sum()
                    ),
                    "events": int(len(group)),
                    "dates": int(
                        group[policy.date_column].nunique()
                    ),
                    "precision": _safe_divide(
                        fold_tp,
                        fold_tp + fold_fp,
                    ),
                    "specificity": _safe_divide(
                        fold_tn,
                        fold_tn + fold_fp,
                    ),
                    "sensitivity": _safe_divide(
                        fold_tp,
                        fold_tp + fold_fn,
                    ),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    if fold_metrics.empty:
        min_fold_specificity = float("nan")
        mean_fold_specificity = float("nan")
        std_fold_specificity = float("nan")
        min_fold_precision = float("nan")
    else:
        specificity_values = fold_metrics[
            "specificity"
        ].to_numpy(dtype=float)
        precision_values = fold_metrics[
            "precision"
        ].to_numpy(dtype=float)
        min_fold_specificity = float(
            np.nanmin(specificity_values)
        )
        mean_fold_specificity = float(
            np.nanmean(specificity_values)
        )
        std_fold_specificity = float(
            np.nanstd(specificity_values, ddof=0)
        )
        min_fold_precision = float(
            np.nanmin(precision_values)
        )

    summary: dict[str, object] = {
        **counts,
        "events": int(len(frame)),
        "signals": signal_count,
        "expected_signals": expected_signal_count,
        "signal_rate": _safe_divide(signal_count, len(frame)),
        "dates": dates,
        "dates_with_signal": dates_with_signal,
        "date_coverage": _safe_divide(
            dates_with_signal,
            dates,
        ),
        "precision": _safe_divide(tp, tp + fp),
        "specificity": _safe_divide(tn, tn + fp),
        "sensitivity": _safe_divide(tp, tp + fn),
        "prevalence": _safe_divide(tp + fn, len(frame)),
        "min_fold_specificity": min_fold_specificity,
        "mean_fold_specificity": mean_fold_specificity,
        "std_fold_specificity": std_fold_specificity,
        "min_fold_precision": min_fold_precision,
        "policy_complete": bool(
            signal_count == expected_signal_count
            and dates_with_signal == dates
        ),
        "policy": asdict(policy),
    }
    return summary, fold_metrics


def rank_policy_candidates(
    candidates: pd.DataFrame,
    *,
    model_column: str = "model_name",
    feature_set_column: str = "feature_set_name",
) -> pd.DataFrame:
    """Rank model/feature-set candidates with false positives first."""
    required = {
        model_column,
        feature_set_column,
        "policy_complete",
        "false_positive",
        "precision",
        "min_fold_specificity",
        "std_fold_specificity",
        "mean_average_precision",
        "mean_roc_auc",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise KeyError(
            f"Policy candidate table is missing columns: {missing}"
        )

    frame = candidates.copy()
    if frame.empty:
        raise ValueError("No policy candidates are available.")

    eligible = frame.loc[
        frame["policy_complete"].astype(bool)
    ].copy()
    if eligible.empty:
        raise ValueError(
            "No policy candidate satisfies complete daily coverage."
        )

    eligible = eligible.sort_values(
        [
            "false_positive",
            "precision",
            "min_fold_specificity",
            "std_fold_specificity",
            "mean_average_precision",
            "mean_roc_auc",
            feature_set_column,
            model_column,
        ],
        ascending=[
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)
    eligible["policy_selection_rank"] = (
        np.arange(len(eligible), dtype=int) + 1
    )

    frame = frame.merge(
        eligible[
            [
                model_column,
                feature_set_column,
                "policy_selection_rank",
            ]
        ],
        on=[model_column, feature_set_column],
        how="left",
        validate="one_to_one",
    )
    frame["selected_by_policy_hierarchy"] = (
        frame["policy_selection_rank"].eq(1)
    )
    frame = frame.sort_values(
        [
            "policy_complete",
            "policy_selection_rank",
            "false_positive",
            feature_set_column,
            model_column,
        ],
        ascending=[False, True, True, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    return frame


def select_optuna_trial_by_policy(
    study: Study,
) -> tuple[FrozenTrial, pd.DataFrame]:
    """Select a COMPLETE trial by frozen false-positive-first hierarchy."""
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ]
    if not complete_trials:
        raise ValueError(
            f"Study {study.study_name} has no COMPLETE trials."
        )

    required_attrs = {
        "policy_complete",
        "policy_false_positive",
        "policy_precision",
        "policy_min_fold_specificity",
        "policy_specificity_std",
        "mean_roc_auc",
        "mean_average_precision",
    }

    rows = []
    for trial in complete_trials:
        missing = sorted(
            required_attrs - set(trial.user_attrs)
        )
        if missing:
            raise KeyError(
                f"Trial {trial.number} is missing policy attrs: "
                f"{missing}"
            )
        rows.append(
            {
                "trial_number": int(trial.number),
                "objective_value": float(trial.value),
                "policy_complete": bool(
                    trial.user_attrs["policy_complete"]
                ),
                "false_positive": int(
                    trial.user_attrs[
                        "policy_false_positive"
                    ]
                ),
                "precision": float(
                    trial.user_attrs["policy_precision"]
                ),
                "min_fold_specificity": float(
                    trial.user_attrs[
                        "policy_min_fold_specificity"
                    ]
                ),
                "std_fold_specificity": float(
                    trial.user_attrs[
                        "policy_specificity_std"
                    ]
                ),
                "mean_average_precision": float(
                    trial.user_attrs[
                        "mean_average_precision"
                    ]
                ),
                "mean_roc_auc": float(
                    trial.user_attrs["mean_roc_auc"]
                ),
            }
        )

    ranking = pd.DataFrame(rows)
    eligible = ranking.loc[
        ranking["policy_complete"]
    ].copy()
    if eligible.empty:
        raise ValueError(
            f"Study {study.study_name} has no policy-complete trial."
        )

    eligible = eligible.sort_values(
        [
            "false_positive",
            "precision",
            "min_fold_specificity",
            "std_fold_specificity",
            "mean_average_precision",
            "mean_roc_auc",
            "trial_number",
        ],
        ascending=[
            True,
            False,
            False,
            True,
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)
    eligible["policy_selection_rank"] = (
        np.arange(len(eligible), dtype=int) + 1
    )

    ranking = ranking.merge(
        eligible[
            [
                "trial_number",
                "policy_selection_rank",
            ]
        ],
        on="trial_number",
        how="left",
        validate="one_to_one",
    )
    ranking["selected_by_policy_hierarchy"] = (
        ranking["policy_selection_rank"].eq(1)
    )
    ranking = ranking.sort_values(
        [
            "policy_complete",
            "policy_selection_rank",
            "false_positive",
            "trial_number",
        ],
        ascending=[False, True, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    selected_number = int(
        ranking.loc[
            ranking["selected_by_policy_hierarchy"],
            "trial_number",
        ].iloc[0]
    )
    selected_trial = next(
        trial
        for trial in complete_trials
        if int(trial.number) == selected_number
    )
    return selected_trial, ranking
