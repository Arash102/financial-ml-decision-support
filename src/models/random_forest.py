"""Random Forest search space and pipeline factory."""

from __future__ import annotations

from typing import Any

from optuna.trial import Trial
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.tuning import build_tree_preprocessor


def sample_random_forest_params(trial: Trial) -> dict[str, Any]:
    """
    Sample the frozen Stage 06 v3 Random Forest search space.

    max_samples is sampled for every trial so multivariate TPE sees a fixed
    search space. It is set to None when bootstrap is False.
    """
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 25, 150, step=25
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 15
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 30
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 1, 20
        ),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", 0.5, 0.75],
        ),
        "criterion": trial.suggest_categorical(
            "criterion",
            ["gini", "entropy", "log_loss"],
        ),
        "bootstrap": trial.suggest_categorical(
            "bootstrap",
            [True, False],
        ),
        "max_samples": trial.suggest_float(
            "max_samples", 0.60, 1.00
        ),
        "class_weight_mode": trial.suggest_categorical(
            "class_weight_mode",
            ["none", "balanced_subsample"],
        ),
    }


def build_random_forest_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    params: dict[str, Any],
    seed: int,
) -> Pipeline:
    """Build fold-local preprocessing and weighted Random Forest."""
    model_params = dict(params)

    class_weight_mode = model_params.pop(
        "class_weight_mode",
        "none",
    )
    model_params["class_weight"] = (
        None
        if class_weight_mode == "none"
        else "balanced_subsample"
    )

    if not bool(model_params.get("bootstrap", True)):
        model_params["max_samples"] = None

    classifier = RandomForestClassifier(
        **model_params,
        random_state=int(seed),
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                build_tree_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            ("model", classifier),
        ]
    )
