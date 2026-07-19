"""Train-only abstention-policy selection for frozen meta-label scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


ABSTENTION_POLICY_SCHEMA_VERSION = (
    "stage08_v3_train_only_breadth_gate_threshold_daily_cap"
)


@dataclass(frozen=True)
class AbstentionPolicy:
    """Frozen post-model decision policy with an explicit no-trade option."""

    gate_name: str
    allowed_regimes: tuple[str, ...]
    minimum_score: float
    maximum_daily_fraction: float = 0.05
    minimum_signals_per_date: int = 0
    date_column: str = "dEven"
    score_column: str = "probability_positive"
    label_column: str = "meta_label"
    regime_column: str = "market_breadth_regime"
    symbol_column: str = "symbol"
    event_id_column: str = "event_id"

    def validate(self) -> None:
        if not self.gate_name:
            raise ValueError("gate_name cannot be empty.")
        if not self.allowed_regimes:
            raise ValueError("allowed_regimes cannot be empty.")
        if not np.isfinite(float(self.minimum_score)):
            raise ValueError("minimum_score must be finite.")
        if not 0.0 <= float(self.minimum_score) <= 1.0:
            raise ValueError("minimum_score must lie in [0, 1].")
        if not 0.0 < float(self.maximum_daily_fraction) <= 1.0:
            raise ValueError(
                "maximum_daily_fraction must lie in (0, 1]."
            )
        if int(self.minimum_signals_per_date) != 0:
            raise ValueError(
                "The abstention policy requires "
                "minimum_signals_per_date = 0."
            )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


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


def apply_abstention_policy(
    predictions: pd.DataFrame,
    *,
    policy: AbstentionPolicy,
) -> pd.DataFrame:
    """
    Apply Breadth gate, fixed score threshold, and daily maximum quota.

    The maximum quota is computed from the complete same-day candidate count,
    before gate or threshold filtering. Zero selected signals on a date is
    explicitly allowed.
    """
    policy.validate()

    required = {
        policy.date_column,
        policy.score_column,
        policy.label_column,
        policy.regime_column,
        policy.symbol_column,
        policy.event_id_column,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise KeyError(
            f"Abstention-policy predictions are missing columns: {missing}"
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
    frame[policy.regime_column] = (
        frame[policy.regime_column].astype("string")
    )

    if frame[policy.date_column].isna().any():
        raise ValueError("Policy predictions contain invalid dates.")
    if not np.isfinite(
        frame[policy.score_column].to_numpy(dtype=float)
    ).all():
        raise ValueError("Policy predictions contain non-finite scores.")
    if not frame[policy.score_column].between(0.0, 1.0).all():
        raise ValueError("Policy scores must lie in [0, 1].")
    if not frame[policy.label_column].isin([0, 1]).all():
        raise ValueError("Policy labels must be binary 0/1.")
    if frame[policy.event_id_column].duplicated().any():
        raise ValueError("Policy event IDs must be unique.")
    if frame[policy.regime_column].isna().any():
        raise ValueError("Policy regimes contain missing values.")

    frame["_original_row_order"] = np.arange(
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
    frame["daily_maximum_quota"] = np.ceil(
        frame["daily_candidate_count"].to_numpy(dtype=float)
        * float(policy.maximum_daily_fraction)
    ).astype(int)

    frame["market_gate_pass"] = frame[
        policy.regime_column
    ].isin(list(policy.allowed_regimes))
    frame["score_threshold_pass"] = frame[
        policy.score_column
    ].ge(float(policy.minimum_score))
    frame["policy_eligible"] = (
        frame["market_gate_pass"]
        & frame["score_threshold_pass"]
    )

    frame["daily_eligible_count"] = (
        frame["policy_eligible"]
        .groupby(frame[policy.date_column], observed=False)
        .transform("sum")
        .astype(int)
    )
    frame["daily_signal_quota"] = np.minimum(
        frame["daily_maximum_quota"].to_numpy(dtype=int),
        frame["daily_eligible_count"].to_numpy(dtype=int),
    ).astype(int)

    eligible_rank = (
        frame["policy_eligible"]
        .groupby(frame[policy.date_column], observed=False)
        .cumsum()
    )
    frame["daily_eligible_rank"] = np.where(
        frame["policy_eligible"],
        eligible_rank,
        0,
    ).astype(int)
    frame["selected_signal"] = (
        frame["policy_eligible"]
        & frame["daily_eligible_rank"].gt(0)
        & frame["daily_eligible_rank"].le(
            frame["daily_signal_quota"]
        )
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
        .rename("daily_selected_score_cutoff")
    )
    frame = frame.merge(
        selected_cutoff,
        left_on=policy.date_column,
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    selected_by_date = frame.groupby(
        policy.date_column,
        observed=False,
    )["selected_signal"].sum().astype(int)
    quota_by_date = frame.groupby(
        policy.date_column,
        observed=False,
    )["daily_signal_quota"].first().astype(int)

    if not selected_by_date.eq(quota_by_date).all():
        raise AssertionError(
            "Selected signal counts do not equal abstention quotas."
        )
    if frame.loc[
        ~frame["policy_eligible"],
        "selected_signal",
    ].any():
        raise AssertionError(
            "A gate- or threshold-rejected row was selected."
        )

    return frame.sort_values(
        "_original_row_order",
        kind="stable",
    ).drop(columns=["_original_row_order"]).reset_index(drop=True)


def summarize_abstention_policy(
    policy_predictions: pd.DataFrame,
    *,
    policy: AbstentionPolicy,
    baseline_signal_count: int,
    baseline_fold_signal_counts: dict[int, int],
    minimum_pooled_coverage: float,
    minimum_fold_coverage: float,
    fold_column: str = "fold_id",
) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarize pooled and fold-level abstention-policy performance."""
    required = {
        policy.date_column,
        policy.label_column,
        "selected_signal",
        "daily_signal_quota",
        fold_column,
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

    fold_rows: list[dict[str, object]] = []
    for fold_id, group in frame.groupby(
        fold_column,
        sort=True,
        observed=False,
    ):
        fold_id_int = int(fold_id)
        if fold_id_int not in baseline_fold_signal_counts:
            raise KeyError(
                f"Missing baseline signal count for fold {fold_id_int}."
            )

        fold_counts = _confusion_counts(
            group[policy.label_column].to_numpy(dtype=int),
            group["selected_signal"].to_numpy(dtype=bool),
        )
        fold_tp = fold_counts["true_positive"]
        fold_fp = fold_counts["false_positive"]
        fold_tn = fold_counts["true_negative"]
        fold_fn = fold_counts["false_negative"]
        fold_signals = int(group["selected_signal"].sum())
        baseline_fold_signals = int(
            baseline_fold_signal_counts[fold_id_int]
        )

        fold_rows.append({
            "fold_id": fold_id_int,
            **fold_counts,
            "signals": fold_signals,
            "baseline_signals": baseline_fold_signals,
            "signal_coverage": _safe_divide(
                fold_signals,
                baseline_fold_signals,
            ),
            "events": int(len(group)),
            "dates": int(group[policy.date_column].nunique()),
            "dates_with_signal": int(
                group.loc[
                    group["selected_signal"],
                    policy.date_column,
                ].nunique()
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
        })

    fold_metrics = pd.DataFrame(fold_rows)
    fold_coverages = fold_metrics[
        "signal_coverage"
    ].to_numpy(dtype=float)
    fold_precisions = fold_metrics[
        "precision"
    ].to_numpy(dtype=float)
    fold_specificities = fold_metrics[
        "specificity"
    ].to_numpy(dtype=float)

    pooled_coverage = _safe_divide(
        signal_count,
        int(baseline_signal_count),
    )
    coverage_constraints_pass = bool(
        np.isfinite(pooled_coverage)
        and pooled_coverage >= float(minimum_pooled_coverage)
        and np.isfinite(fold_coverages).all()
        and np.min(fold_coverages)
        >= float(minimum_fold_coverage)
    )

    summary: dict[str, object] = {
        **counts,
        "events": int(len(frame)),
        "signals": signal_count,
        "baseline_signals": int(baseline_signal_count),
        "signal_coverage": pooled_coverage,
        "dates": dates,
        "dates_with_signal": dates_with_signal,
        "zero_signal_dates": int(dates - dates_with_signal),
        "date_coverage": _safe_divide(
            dates_with_signal,
            dates,
        ),
        "precision": _safe_divide(tp, tp + fp),
        "specificity": _safe_divide(tn, tn + fp),
        "sensitivity": _safe_divide(tp, tp + fn),
        "prevalence": _safe_divide(tp + fn, len(frame)),
        "minimum_fold_signal_coverage": float(
            np.nanmin(fold_coverages)
        ),
        "minimum_fold_precision": float(
            np.nanmin(fold_precisions)
        ),
        "mean_fold_precision": float(
            np.nanmean(fold_precisions)
        ),
        "std_fold_precision": float(
            np.nanstd(fold_precisions, ddof=0)
        ),
        "minimum_fold_specificity": float(
            np.nanmin(fold_specificities)
        ),
        "coverage_constraints_pass": coverage_constraints_pass,
        "minimum_pooled_coverage_required": float(
            minimum_pooled_coverage
        ),
        "minimum_fold_coverage_required": float(
            minimum_fold_coverage
        ),
        "policy": asdict(policy),
    }
    return summary, fold_metrics


def select_abstention_policy(
    candidate_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the frozen false-positive-first hierarchy.

    Feasible candidates must satisfy pooled and every-fold coverage constraints.
    """
    required = {
        "candidate_id",
        "coverage_constraints_pass",
        "false_positive",
        "precision",
        "true_positive",
        "minimum_fold_precision",
        "std_fold_precision",
        "minimum_fold_specificity",
        "signals",
        "gate_complexity",
        "threshold_quantile",
    }
    missing = sorted(required - set(candidate_summaries.columns))
    if missing:
        raise KeyError(
            f"Policy candidate table is missing columns: {missing}"
        )

    frame = candidate_summaries.copy()
    feasible = frame.loc[
        frame["coverage_constraints_pass"].astype(bool)
    ].copy()
    if feasible.empty:
        raise ValueError(
            "No abstention-policy candidate satisfies coverage constraints."
        )

    feasible = feasible.sort_values(
        [
            "false_positive",
            "precision",
            "true_positive",
            "minimum_fold_precision",
            "std_fold_precision",
            "minimum_fold_specificity",
            "signals",
            "gate_complexity",
            "threshold_quantile",
            "candidate_id",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    feasible["policy_selection_rank"] = (
        np.arange(len(feasible), dtype=int) + 1
    )
    frame = frame.merge(
        feasible[
            ["candidate_id", "policy_selection_rank"]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    frame["selected_by_policy_hierarchy"] = frame[
        "policy_selection_rank"
    ].eq(1)

    return frame.sort_values(
        [
            "coverage_constraints_pass",
            "policy_selection_rank",
            "false_positive",
            "candidate_id",
        ],
        ascending=[False, True, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def threshold_grid_from_baseline_scores(
    baseline_selected_scores: Iterable[float],
    *,
    quantiles: Iterable[float],
) -> pd.DataFrame:
    """Create a small deterministic threshold grid from train-only scores."""
    scores = np.asarray(
        list(baseline_selected_scores),
        dtype=float,
    )
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError(
            "baseline_selected_scores must be non-empty."
        )
    if not np.isfinite(scores).all():
        raise ValueError(
            "baseline_selected_scores contain non-finite values."
        )

    rows = []
    for quantile in quantiles:
        q = float(quantile)
        if not 0.0 <= q <= 1.0:
            raise ValueError("Threshold quantiles must lie in [0, 1].")
        rows.append({
            "threshold_quantile": q,
            "minimum_score": float(
                np.quantile(scores, q, method="linear")
            ),
        })

    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["minimum_score"],
        keep="first",
    )
    return frame.sort_values(
        ["threshold_quantile", "minimum_score"],
        kind="stable",
    ).reset_index(drop=True)
