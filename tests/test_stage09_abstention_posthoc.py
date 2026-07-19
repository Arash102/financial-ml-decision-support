from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.abstention_policy import (
    AbstentionPolicy,
    apply_abstention_policy,
    apply_abstention_policy_inference,
)


def _sample_frame() -> pd.DataFrame:
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
        "xgboost_ranking_score": [
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


def _policy() -> AbstentionPolicy:
    return AbstentionPolicy(
        gate_name="G3_recovery_only",
        allowed_regimes=(
            "recovery_negative",
            "recovery_positive",
        ),
        minimum_score=0.75,
        maximum_daily_fraction=0.25,
        minimum_signals_per_date=0,
        score_column="xgboost_ranking_score",
    )


def test_inference_policy_requires_no_label_column() -> None:
    frame = _sample_frame().drop(columns=["meta_label"])
    result = apply_abstention_policy_inference(
        frame,
        policy=_policy(),
    )

    assert "meta_label" not in result.columns
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


def test_inference_and_labeled_policy_select_identical_events() -> None:
    frame = _sample_frame()
    inference_result = apply_abstention_policy_inference(
        frame.drop(columns=["meta_label"]),
        policy=_policy(),
    )
    labeled_result = apply_abstention_policy(
        frame,
        policy=_policy(),
    )

    assert inference_result[
        "selected_signal"
    ].tolist() == labeled_result[
        "selected_signal"
    ].tolist()
    assert inference_result[
        "daily_signal_quota"
    ].tolist() == labeled_result[
        "daily_signal_quota"
    ].tolist()


def test_config_freezes_exact_stage08_policy_without_test_targets() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs"
            / "unseen_test_evaluation.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config["status"] == (
        "stage_09_configured_v5_abstention_policy_post_hoc_retest"
    )
    assert config["frozen_inputs"][
        "stage08_policy"
    ].endswith("08_abstention_signal_policy.json")
    assert config["frozen_inputs"][
        "expected_stage08_policy_id"
    ] == "G3_recovery_only__q0000"
    assert config["frozen_inputs"][
        "expected_selected_signals"
    ] is None
    assert config["frozen_inputs"][
        "expected_performance_metrics"
    ] is None
    assert config["frozen_signal_policy"][
        "expected_minimum_signals_per_date"
    ] == 0
    assert config["safeguards"][
        "policy_reselected_in_stage09"
    ] is False


def test_notebook_uses_outcome_free_abstention_inference() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "09_signal_level_evaluation.ipynb"
        ).read_text(encoding="utf-8")
    )
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    ]
    complete_source = "\n".join(sources)

    assert "apply_abstention_policy_inference" in complete_source
    assert "08_abstention_signal_policy.json" in complete_source
    assert "08_probability_signal_policy.json" not in complete_source
    assert "select_daily_top_fraction" not in complete_source
    assert "expected_selected_signals" not in complete_source
    assert "policy_reselected_in_stage09" in complete_source

    lock_cell_index = next(
        index
        for index, source in enumerate(sources)
        if "outcome_free_abstention_inference_locked" in source
    )
    outcome_cell_index = next(
        index
        for index, source in enumerate(sources)
        if source.startswith("outcome_columns = [")
    )
    assert lock_cell_index < outcome_cell_index

    lock_cell = sources[lock_cell_index]
    lock_columns_text = lock_cell.split(
        "lock_columns = [",
        1,
    )[1].split("]\n", 1)[0]
    assert "meta_label" not in lock_columns_text
    assert "event_return" not in lock_columns_text
    assert "barrier_touched" not in lock_columns_text


def test_notebook_allows_zero_signal_dates() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "09_signal_level_evaluation.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "zero_signal_dates_locked" in source
    assert "POLICY_MINIMUM_SIGNALS_PER_DATE == 0" in source
    assert '["selected_signal"].sum().ge(1)' not in source
    assert "daily_selected_score_cutoff" in source
    assert "signal_coverage_vs_old_top5" in source


def test_selected_events_must_pass_gate_and_threshold() -> None:
    result = apply_abstention_policy_inference(
        _sample_frame().drop(columns=["meta_label"]),
        policy=_policy(),
    )
    selected = result.loc[result["selected_signal"]]

    assert selected["market_gate_pass"].all()
    assert selected["score_threshold_pass"].all()
    assert selected[
        "market_breadth_regime"
    ].isin([
        "recovery_negative",
        "recovery_positive",
    ]).all()
    assert selected[
        "xgboost_ranking_score"
    ].ge(0.75).all()


def test_all_notebook_code_cells_compile() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "09_signal_level_evaluation.ipynb"
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


def test_policy_module_compiles() -> None:
    root = Path(__file__).resolve().parents[1]
    module_path = (
        root
        / "src"
        / "models"
        / "abstention_policy.py"
    )
    compile(
        module_path.read_text(encoding="utf-8"),
        str(module_path),
        "exec",
    )
