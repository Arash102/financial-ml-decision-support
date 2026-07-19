from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.models.policy_selection import (
    DailyTopFractionPolicy,
    apply_daily_top_fraction_policy,
)


def test_top_fraction_policy_uses_ceil_and_minimum_one() -> None:
    rows = []
    for date, count in [
        ("2020-01-01", 1),
        ("2020-01-02", 21),
    ]:
        for index in range(count):
            rows.append(
                {
                    "event_id": f"{date}-{index}",
                    "symbol": f"S{index:03d}",
                    "dEven": date,
                    "meta_label": int(index == 0),
                    "probability_positive": 1.0 - index / 100.0,
                }
            )
    frame = pd.DataFrame(rows)
    assigned = apply_daily_top_fraction_policy(
        frame,
        policy=DailyTopFractionPolicy(
            fraction=0.05,
            minimum_signals_per_date=1,
        ),
    )
    quotas = (
        assigned.groupby("dEven")[
            "daily_signal_quota"
        ].first().tolist()
    )
    assert quotas == [1, 2]


def test_config_freezes_exact_selected_variant() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs"
            / "signal_policy.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["status"] == (
        "stage_08_configured_v2_stage06_policy_lock"
    )
    assert config["input"]["selected_model"] == "xgboost"
    assert config["input"]["selected_feature_set"] == "I_full_40"
    assert config["input"]["expected_oof_rows"] == 51840
    assert config["policy"]["selected_fraction"] == 0.05
    assert config["policy"]["minimum_signals_per_date"] == 1
    assert config["policy"]["expected_total_oof_signals"] == 3016


def test_notebook_filters_model_and_feature_set() -> None:
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
    compact = "".join(source.split())

    assert '"feature_set_name"' in source
    assert (
        'all_oof["model_name"].eq(SELECTED_MODEL)'
        in compact
    )
    assert (
        'all_oof["feature_set_name"].eq(SELECTED_FEATURE_SET)'
        in compact
    )
    assert "SELECTED_FEATURE_SET == \"I_full_40\"" in source


def test_notebook_does_not_reselect_policy_or_calibration() -> None:
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

    assert "fit_probability_calibrator" not in source
    assert "candidate_fractions" not in source
    assert "select_signal_policy_hierarchically" not in source
    assert "policy_reselected_in_stage08" in source
    assert "raw_identity" in source


def test_notebook_preserves_safeguards() -> None:
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

    for token in [
        "unseen_test_loaded",
        "unseen_test_labels_used",
        "probability_threshold_selected",
        "hard_started_filter_applied",
        "hard_breadth_filter_applied",
        "model_retrained",
    ]:
        assert token in source


def test_notebook_has_exact_stage06_policy_identity_checks() -> None:
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

    assert "selected_true_positive" in source
    assert "selected_false_positive" in source
    assert "selected_precision" in source
    assert "selected_specificity" in source
    assert "1_987" in source
    assert "1_029" in source
    assert "3_016" in source


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
