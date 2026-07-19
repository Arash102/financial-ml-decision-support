from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.abstention_policy import (
    ABSTENTION_POLICY_SCHEMA_VERSION,
    AbstentionPolicy,
    apply_abstention_policy,
    select_abstention_policy,
    summarize_abstention_policy,
    threshold_grid_from_baseline_scores,
)


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "fold_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "event_id": [
            "A1", "B1", "C1", "D1",
            "A2", "B2", "C2", "D2",
        ],
        "symbol": [
            "A", "B", "C", "D",
            "A", "B", "C", "D",
        ],
        "dEven": [
            "2020-01-01",
            "2020-01-01",
            "2020-01-01",
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
            "2020-01-02",
            "2020-01-02",
        ],
        "meta_label": [1, 0, 1, 0, 0, 1, 0, 1],
        "probability_positive": [
            0.90, 0.80, 0.70, 0.60,
            0.95, 0.85, 0.75, 0.65,
        ],
        "market_breadth_regime": [
            "recovery_positive",
            "recovery_positive",
            "recovery_positive",
            "recovery_positive",
            "deterioration",
            "deterioration",
            "deterioration",
            "deterioration",
        ],
    })


def test_abstention_policy_allows_zero_signal_dates() -> None:
    frame = _sample_predictions()
    policy = AbstentionPolicy(
        gate_name="recovery_only",
        allowed_regimes=(
            "recovery_negative",
            "recovery_positive",
        ),
        minimum_score=0.0,
        maximum_daily_fraction=0.25,
        minimum_signals_per_date=0,
    )
    result = apply_abstention_policy(
        frame,
        policy=policy,
    )

    selected_by_date = (
        result.groupby("dEven")[
            "selected_signal"
        ].sum().astype(int)
    )
    assert selected_by_date.loc[
        pd.Timestamp("2020-01-01")
    ] == 1
    assert selected_by_date.loc[
        pd.Timestamp("2020-01-02")
    ] == 0


def test_threshold_and_daily_cap_are_both_enforced() -> None:
    frame = _sample_predictions()
    policy = AbstentionPolicy(
        gate_name="all",
        allowed_regimes=(
            "recovery_positive",
            "deterioration",
        ),
        minimum_score=0.88,
        maximum_daily_fraction=0.50,
        minimum_signals_per_date=0,
    )
    result = apply_abstention_policy(
        frame,
        policy=policy,
    )

    selected = result.loc[
        result["selected_signal"],
        "event_id",
    ].tolist()
    assert selected == ["A1", "A2"]
    assert result.groupby("dEven")[
        "selected_signal"
    ].sum().le(
        result.groupby("dEven")[
            "daily_maximum_quota"
        ].first()
    ).all()


def test_summary_applies_pooled_and_each_fold_coverage() -> None:
    frame = _sample_predictions()
    policy = AbstentionPolicy(
        gate_name="all",
        allowed_regimes=(
            "recovery_positive",
            "deterioration",
        ),
        minimum_score=0.0,
        maximum_daily_fraction=0.25,
        minimum_signals_per_date=0,
    )
    result = apply_abstention_policy(
        frame,
        policy=policy,
    )
    summary, folds = summarize_abstention_policy(
        result,
        policy=policy,
        baseline_signal_count=4,
        baseline_fold_signal_counts={1: 2, 2: 2},
        minimum_pooled_coverage=0.50,
        minimum_fold_coverage=0.50,
    )

    assert summary["signals"] == 2
    assert np.isclose(
        summary["signal_coverage"],
        0.50,
    )
    assert summary[
        "coverage_constraints_pass"
    ] is True
    assert folds["signal_coverage"].eq(0.50).all()


def test_false_positive_first_selection_hierarchy() -> None:
    candidates = pd.DataFrame([
        {
            "candidate_id": "A",
            "coverage_constraints_pass": True,
            "false_positive": 10,
            "precision": 0.70,
            "true_positive": 20,
            "minimum_fold_precision": 0.60,
            "std_fold_precision": 0.05,
            "minimum_fold_specificity": 0.90,
            "signals": 30,
            "gate_complexity": 0,
            "threshold_quantile": 0.50,
        },
        {
            "candidate_id": "B",
            "coverage_constraints_pass": True,
            "false_positive": 9,
            "precision": 0.60,
            "true_positive": 14,
            "minimum_fold_precision": 0.50,
            "std_fold_precision": 0.10,
            "minimum_fold_specificity": 0.92,
            "signals": 23,
            "gate_complexity": 2,
            "threshold_quantile": 0.65,
        },
        {
            "candidate_id": "C",
            "coverage_constraints_pass": False,
            "false_positive": 1,
            "precision": 0.90,
            "true_positive": 9,
            "minimum_fold_precision": 0.80,
            "std_fold_precision": 0.01,
            "minimum_fold_specificity": 0.99,
            "signals": 10,
            "gate_complexity": 3,
            "threshold_quantile": 0.75,
        },
    ])
    ranked = select_abstention_policy(candidates)
    selected = ranked.loc[
        ranked["selected_by_policy_hierarchy"],
        "candidate_id",
    ].iloc[0]
    assert selected == "B"


def test_threshold_grid_is_deterministic() -> None:
    grid = threshold_grid_from_baseline_scores(
        [0.4, 0.5, 0.6, 0.7, 0.8],
        quantiles=[0.0, 0.5, 0.75],
    )
    assert grid["threshold_quantile"].tolist() == [
        0.0,
        0.5,
        0.75,
    ]
    assert np.allclose(
        grid["minimum_score"],
        [0.4, 0.6, 0.7],
    )


def test_config_is_train_only_and_zero_minimum() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs"
            / "signal_policy.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["status"] == (
        "stage_08_configured_v3_train_only_abstention_policy_selection"
    )
    assert config["abstention_policy"][
        "minimum_signals_per_date"
    ] == 0
    assert config["coverage_constraints"][
        "minimum_pooled_signal_coverage_vs_baseline"
    ] == 0.25
    assert config["coverage_constraints"][
        "minimum_each_fold_signal_coverage_vs_baseline"
    ] == 0.25
    assert config["safeguards"][
        "unseen_test_loaded"
    ] is False


def test_notebook_does_not_load_evaluation_period() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "08_unseen_test_evaluation.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "08_abstention_signal_policy.json" in source
    assert "minimum_signals_per_date=0" in source
    assert "market_breadth_regime" in source
    assert "06_walk_forward_validation_predictions.csv" not in source
    assert "data_ready/unseen_test" not in source
    assert "09_unseen_test" not in source


def test_all_notebook_code_cells_compile() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "08_unseen_test_evaluation.ipynb"
        ).read_text(encoding="utf-8")
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile(
            "".join(cell.get("source", [])),
            f"cell_{index}",
            "exec",
        )


def test_schema_version() -> None:
    assert ABSTENTION_POLICY_SCHEMA_VERSION == (
        "stage08_v3_train_only_breadth_gate_threshold_daily_cap"
    )
