from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from src.models.policy_selection import (
    DailyTopFractionPolicy,
    apply_daily_top_fraction_policy,
    rank_policy_candidates,
    select_optuna_trial_by_policy,
    summarize_policy_predictions,
)


def _predictions(
    *,
    dates: int = 2,
    candidates_per_date: int = 20,
) -> pd.DataFrame:
    rows = []
    event_index = 0
    for date_index in range(dates):
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(
            days=date_index
        )
        for rank in range(candidates_per_date):
            rows.append(
                {
                    "fold_id": date_index + 1,
                    "event_id": f"E{event_index:04d}",
                    "symbol": f"S{rank:03d}",
                    "dEven": date,
                    "meta_label": int(rank % 3 == 0),
                    "probability_positive": (
                        1.0 - rank / 100.0
                    ),
                }
            )
            event_index += 1
    return pd.DataFrame(rows)


def test_top_five_percent_uses_ceil_and_minimum_one() -> None:
    frame = pd.concat(
        [
            _predictions(
                dates=1,
                candidates_per_date=1,
            ),
            _predictions(
                dates=1,
                candidates_per_date=21,
            ).assign(
                dEven=pd.Timestamp("2020-01-02"),
                event_id=lambda x: "B" + x["event_id"],
            ),
        ],
        ignore_index=True,
    )
    result = apply_daily_top_fraction_policy(
        frame,
        policy=DailyTopFractionPolicy(
            fraction=0.05,
            minimum_signals_per_date=1,
        ),
    )
    quotas = (
        result.groupby("dEven")[
            "daily_signal_quota"
        ].first().tolist()
    )
    assert quotas == [1, 2]
    assert int(result["selected_signal"].sum()) == 3


def test_ties_use_symbol_then_event_id() -> None:
    frame = pd.DataFrame(
        {
            "fold_id": [1, 1, 1],
            "event_id": ["Z", "A", "B"],
            "symbol": ["BBB", "AAA", "AAA"],
            "dEven": ["2020-01-01"] * 3,
            "meta_label": [0, 1, 0],
            "probability_positive": [0.8, 0.8, 0.8],
        }
    )
    result = apply_daily_top_fraction_policy(
        frame,
        policy=DailyTopFractionPolicy(
            fraction=0.01,
            minimum_signals_per_date=1,
        ),
    )
    selected = result.loc[
        result["selected_signal"],
        "event_id",
    ].tolist()
    assert selected == ["A"]


def test_policy_confusion_and_coverage() -> None:
    frame = _predictions(
        dates=3,
        candidates_per_date=20,
    )
    assigned = apply_daily_top_fraction_policy(
        frame
    )
    summary, fold_metrics = (
        summarize_policy_predictions(
            assigned
        )
    )
    assert summary["policy_complete"] is True
    assert summary["signals"] == 3
    assert summary["expected_signals"] == 3
    assert summary["dates_with_signal"] == 3
    assert len(fold_metrics) == 3
    assert (
        summary["true_positive"]
        + summary["false_positive"]
        == summary["signals"]
    )


def test_model_feature_ranking_prioritizes_false_positive() -> None:
    candidates = pd.DataFrame(
        [
            {
                "model_name": "xgboost",
                "feature_set_name": "A",
                "policy_complete": True,
                "false_positive": 10,
                "precision": 0.60,
                "min_fold_specificity": 0.90,
                "std_fold_specificity": 0.02,
                "mean_average_precision": 0.70,
                "mean_roc_auc": 0.72,
            },
            {
                "model_name": "random_forest",
                "feature_set_name": "B",
                "policy_complete": True,
                "false_positive": 9,
                "precision": 0.55,
                "min_fold_specificity": 0.88,
                "std_fold_specificity": 0.03,
                "mean_average_precision": 0.65,
                "mean_roc_auc": 0.68,
            },
        ]
    )
    ranking = rank_policy_candidates(candidates)
    selected = ranking.loc[
        ranking["selected_by_policy_hierarchy"]
    ].iloc[0]
    assert selected["model_name"] == "random_forest"
    assert selected["feature_set_name"] == "B"


def test_optuna_trial_selection_prioritizes_false_positive() -> None:
    study = optuna.create_study(direction="maximize")

    attrs = [
        {
            "policy_complete": True,
            "policy_false_positive": 12,
            "policy_precision": 0.70,
            "policy_min_fold_specificity": 0.90,
            "policy_specificity_std": 0.01,
            "mean_average_precision": 0.72,
            "mean_roc_auc": 0.75,
        },
        {
            "policy_complete": True,
            "policy_false_positive": 11,
            "policy_precision": 0.60,
            "policy_min_fold_specificity": 0.89,
            "policy_specificity_std": 0.02,
            "mean_average_precision": 0.68,
            "mean_roc_auc": 0.70,
        },
    ]

    for index, trial_attrs in enumerate(attrs):
        trial = optuna.trial.create_trial(
            value=0.8 - index * 0.1,
            params={},
            distributions={},
            user_attrs=trial_attrs,
        )
        study.add_trial(trial)

    selected, ranking = (
        select_optuna_trial_by_policy(study)
    )
    assert selected.number == 1
    assert int(
        ranking.loc[
            ranking["selected_by_policy_hierarchy"],
            "false_positive",
        ].iloc[0]
    ) == 11


def test_notebook_and_config_have_frozen_v4_design() -> None:
    package_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            package_root
            / "configs"
            / "stage06_breadth_retrain.yaml"
        ).read_text(encoding="utf-8")
    )
    notebook = json.loads(
        (
            package_root
            / "notebooks"
            / "06_optuna_model_selection.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert config["status"] == (
        "stage_06_breadth_retrain_v4"
    )
    assert len(config["feature_sets"]) == 9
    assert config["design"]["tuning_feature_set"] == (
        "I_full_40"
    )
    assert "assert len(MODEL_FEATURES) == 40" in source
    assert "assert len(NUMERIC_FEATURES) == 38" in source
    assert "market_breadth_regime" in source
    assert "select_optuna_trial_by_policy" in source
    assert "rank_policy_candidates" in source
    assert "unseen_test_used" in source
