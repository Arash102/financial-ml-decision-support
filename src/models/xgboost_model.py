"""XGBoost search space and pipeline factory."""

from __future__ import annotations

from typing import Any

from optuna.trial import Trial
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.tuning import build_tree_preprocessor


def sample_xgboost_params(trial: Trial) -> dict[str, Any]:
    """Sample the frozen Stage 06 v3 XGBoost search space."""
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 25, 150, step=25
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 2, 15
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.20, log=True
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.10, 20.0, log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.60, 1.00
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.50, 1.00
        ),
        "gamma": trial.suggest_float(
            "gamma", 0.0, 5.0
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1.0e-8, 10.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1.0e-3, 100.0, log=True
        ),
        "max_delta_step": trial.suggest_int(
            "max_delta_step", 0, 5
        ),
        "class_weight_mode": trial.suggest_categorical(
            "class_weight_mode",
            ["none", "fold_weighted_ratio"],
        ),
    }


def build_xgboost_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    params: dict[str, Any],
    seed: int,
    scale_pos_weight: float,
) -> Pipeline:
    """Build fold-local preprocessing and weighted XGBoost."""
    model_params = dict(params)
    model_params.pop("class_weight_mode", None)

    classifier = XGBClassifier(
        **model_params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=float(scale_pos_weight),
        random_state=int(seed),
        n_jobs=-1,
        verbosity=0,
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
