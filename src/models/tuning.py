"""Fold-local preprocessing and hierarchical AUC selection."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def feature_columns_from_manifest(
    feature_manifest: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Read ordered final features and split numeric/categorical roles."""
    required = {"feature_order", "feature", "data_type"}
    missing = sorted(required - set(feature_manifest.columns))
    if missing:
        raise KeyError(
            f"Feature manifest is missing columns: {missing}"
        )

    ordered = feature_manifest.sort_values(
        "feature_order",
        kind="stable",
    ).copy()

    if ordered["feature"].duplicated().any():
        raise ValueError(
            "Feature manifest contains duplicate feature names."
        )

    model_features = ordered["feature"].astype(str).tolist()
    numeric_features = ordered.loc[
        ordered["data_type"].eq("numeric"),
        "feature",
    ].astype(str).tolist()
    categorical_features = ordered.loc[
        ordered["data_type"].eq("categorical"),
        "feature",
    ].astype(str).tolist()

    if set(model_features) != set(
        numeric_features + categorical_features
    ):
        raise AssertionError(
            "Feature-role partition is incomplete."
        )

    return (
        model_features,
        numeric_features,
        categorical_features,
    )


def _categorical_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    missing_values=pd.NA,
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def build_tree_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Fold-local median imputation plus categorical one-hot encoding."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                _categorical_pipeline(),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_linear_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Fold-local median imputation, scaling, and categorical encoding."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                _categorical_pipeline(),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def fold_auc_summary(
    metric_values: Iterable[float],
) -> dict[str, float]:
    """Return equal-fold mean, dispersion, and extrema."""
    values = np.asarray(
        list(metric_values),
        dtype=float,
    )

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "metric_values must be a non-empty one-dimensional sequence."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "metric_values contain non-finite values."
        )

    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def aggregate_model_fold_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Produce equal-fold model summaries without row-count weighting."""
    required = {
        "model_name",
        "fold_id",
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier_score",
        "balanced_accuracy",
        "specificity",
        "sensitivity",
        "precision",
        "f1",
        "mcc",
        "fit_seconds",
        "inference_seconds",
    }
    missing = sorted(required - set(fold_metrics.columns))
    if missing:
        raise KeyError(
            f"Fold metrics are missing columns: {missing}"
        )

    rows: list[dict[str, object]] = []

    for model_name, group in fold_metrics.groupby(
        "model_name",
        sort=False,
    ):
        roc = fold_auc_summary(group["roc_auc"])

        rows.append(
            {
                "model_name": model_name,
                "folds": int(group["fold_id"].nunique()),
                "mean_roc_auc": roc["mean"],
                "std_roc_auc": roc["std"],
                "min_roc_auc": roc["minimum"],
                "max_roc_auc": roc["maximum"],
                "mean_average_precision": float(
                    group["average_precision"].mean()
                ),
                "mean_log_loss": float(
                    group["log_loss"].mean()
                ),
                "mean_brier_score": float(
                    group["brier_score"].mean()
                ),
                "mean_balanced_accuracy_at_0_50": float(
                    group["balanced_accuracy"].mean()
                ),
                "mean_specificity_at_0_50": float(
                    group["specificity"].mean()
                ),
                "mean_sensitivity_at_0_50": float(
                    group["sensitivity"].mean()
                ),
                "mean_precision_at_0_50": float(
                    group["precision"].mean()
                ),
                "mean_f1_at_0_50": float(
                    group["f1"].mean()
                ),
                "mean_mcc_at_0_50": float(
                    group["mcc"].mean()
                ),
                "total_fit_seconds": float(
                    group["fit_seconds"].sum()
                ),
                "total_inference_seconds": float(
                    group["inference_seconds"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def hierarchical_near_best_ranking(
    candidates: pd.DataFrame,
    *,
    candidate_column: str,
    mean_column: str,
    minimum_column: str,
    std_column: str,
    mean_tolerance: float,
    numeric_final_tie_break: bool,
) -> pd.DataFrame:
    """
    Apply the frozen mean/worst/std hierarchy.

    Step 1 defines a near-best mean-AUC band. Inside that band, candidates are
    ranked by highest worst-fold AUC, then lowest fold-AUC standard deviation,
    then a deterministic final tie-break.
    """
    required = {
        candidate_column,
        mean_column,
        minimum_column,
        std_column,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise KeyError(
            f"Candidate ranking table is missing columns: {missing}"
        )

    if mean_tolerance < 0.0:
        raise ValueError(
            "mean_tolerance cannot be negative."
        )

    frame = candidates.copy()

    for column in [
        mean_column,
        minimum_column,
        std_column,
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    if frame.empty:
        raise ValueError(
            "No candidates are available for hierarchical ranking."
        )

    if not np.isfinite(
        frame[
            [
                mean_column,
                minimum_column,
                std_column,
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Hierarchical ranking metrics contain non-finite values."
        )

    best_mean = float(frame[mean_column].max())
    lower_bound = best_mean - float(mean_tolerance)

    frame["best_mean_roc_auc"] = best_mean
    frame["near_best_mean_auc_lower_bound"] = lower_bound
    frame["inside_near_best_mean_auc_band"] = (
        frame[mean_column] >= lower_bound
    )

    eligible = frame.loc[
        frame["inside_near_best_mean_auc_band"]
    ].copy()

    if numeric_final_tie_break:
        eligible[candidate_column] = pd.to_numeric(
            eligible[candidate_column],
            errors="raise",
        )

    eligible = eligible.sort_values(
        [
            minimum_column,
            std_column,
            candidate_column,
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    eligible["hierarchical_rank"] = (
        np.arange(len(eligible), dtype=int) + 1
    )

    frame = frame.merge(
        eligible[
            [
                candidate_column,
                "hierarchical_rank",
            ]
        ],
        on=candidate_column,
        how="left",
        validate="one_to_one",
    )

    frame["selected_by_hierarchy"] = (
        frame["hierarchical_rank"].eq(1)
    )

    frame = frame.sort_values(
        [
            "inside_near_best_mean_auc_band",
            "hierarchical_rank",
            mean_column,
        ],
        ascending=[
            False,
            True,
            False,
        ],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    return frame


def select_optuna_trial_by_hierarchy(
    study: Study,
    *,
    mean_tolerance: float,
) -> tuple[FrozenTrial, pd.DataFrame]:
    """Select one COMPLETE Optuna trial using the frozen hierarchy."""
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ]

    if not complete_trials:
        raise ValueError(
            f"Study {study.study_name} has no COMPLETE trials."
        )

    rows = []

    for trial in complete_trials:
        required_attrs = {
            "mean_roc_auc",
            "min_roc_auc",
            "std_roc_auc",
        }
        missing_attrs = sorted(
            required_attrs - set(trial.user_attrs)
        )
        if missing_attrs:
            raise KeyError(
                f"Trial {trial.number} is missing user attrs: "
                f"{missing_attrs}"
            )

        rows.append(
            {
                "trial_number": int(trial.number),
                "objective_value": float(trial.value),
                "mean_roc_auc": float(
                    trial.user_attrs["mean_roc_auc"]
                ),
                "min_roc_auc": float(
                    trial.user_attrs["min_roc_auc"]
                ),
                "std_roc_auc": float(
                    trial.user_attrs["std_roc_auc"]
                ),
            }
        )

    ranking = hierarchical_near_best_ranking(
        pd.DataFrame(rows),
        candidate_column="trial_number",
        mean_column="mean_roc_auc",
        minimum_column="min_roc_auc",
        std_column="std_roc_auc",
        mean_tolerance=mean_tolerance,
        numeric_final_tie_break=True,
    )

    selected_number = int(
        ranking.loc[
            ranking["selected_by_hierarchy"],
            "trial_number",
        ].iloc[0]
    )

    selected_trial = next(
        trial
        for trial in complete_trials
        if int(trial.number) == selected_number
    )

    return selected_trial, ranking
