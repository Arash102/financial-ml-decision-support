"""Frozen baseline model factories for Stage 06 v3."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.tuning import (
    build_linear_preprocessor,
    build_tree_preprocessor,
)


def build_dummy_prior_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    seed: int,
) -> Pipeline:
    """
    Prior-probability Dummy baseline.

    Notebook 06 intentionally fits this baseline without average-uniqueness
    sample weights.
    """
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_tree_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                DummyClassifier(
                    strategy="prior",
                    random_state=int(seed),
                ),
            ),
        ]
    )


def build_logistic_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    seed: int,
) -> Pipeline:
    """
    Fold-local scaled Logistic Regression baseline.

    Notebook 06 passes fold-local average-uniqueness weights to the final model
    step while retaining the frozen class_weight='balanced' baseline setting.
    """
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_linear_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=int(seed),
                ),
            ),
        ]
    )
