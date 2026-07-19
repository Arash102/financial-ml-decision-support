from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.frozen_training import (
    FROZEN_TRAINING_SCHEMA_VERSION,
    dataframe_fingerprint,
    weighted_scale_pos_weight,
)


def test_schema_version_is_stage07_v2() -> None:
    assert FROZEN_TRAINING_SCHEMA_VERSION == (
        "stage07_v2_full_train_selected_feature_set_"
        "average_uniqueness_xgboost"
    )


def test_weighted_scale_pos_weight_none() -> None:
    value = weighted_scale_pos_weight(
        [0, 1, 0, 1],
        [1.0, 2.0, 3.0, 4.0],
        mode="none",
    )
    assert value == 1.0


def test_weighted_scale_pos_weight_ratio() -> None:
    value = weighted_scale_pos_weight(
        [0, 1, 0, 1],
        [1.0, 2.0, 3.0, 4.0],
        mode="fold_weighted_ratio",
    )
    assert np.isclose(value, 4.0 / 6.0)


def test_dataframe_fingerprint_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["A", "B"],
            "feature": [1.0, 2.0],
        }
    )
    first = dataframe_fingerprint(
        frame,
        ["event_id", "feature"],
    )
    second = dataframe_fingerprint(
        frame.copy(),
        ["event_id", "feature"],
    )
    assert first == second
    assert len(first) == 64


def test_config_freezes_full_40_feature_set() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs"
            / "frozen_training.yaml"
        ).read_text(encoding="utf-8")
    )
    population = config["training_population"]

    assert config["status"] == (
        "stage_07_configured_v2_breadth_retrain"
    )
    assert config["selected_model"] == "xgboost"
    assert population["expected_events"] == 118464
    assert population["expected_symbols"] == 499
    assert population["expected_feature_set"] == (
        "I_full_40"
    )
    assert population["expected_features"] == 40
    assert population["expected_numeric_features"] == 38
    assert population[
        "expected_categorical_features"
    ] == [
        "gmma_state",
        "market_breadth_regime",
    ]


def test_notebook_consumes_dynamic_stage06_decision() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "07_frozen_model_training.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    compact_source = "".join(source.split())
    assert (
        'stage06_decision["primary_selected_feature_set"]'
        in compact_source
    )
    assert (
        'stage06_decision["primary_selected_features"]'
        in compact_source
    )
    assert "SELECTED_TRIAL_NUMBER == 25" not in source
    assert "len(MODEL_FEATURES) == 35" not in source
    assert "len(NUMERIC_FEATURES) == 34" not in source
    assert "assert len(MODEL_FEATURES) == 40" in source
    assert "assert len(NUMERIC_FEATURES) == 38" in source
    assert "market_breadth_regime" in source
    assert "for feature in CATEGORICAL_FEATURES" in source
    assert "hard_started_filter_applied" in source
    assert "hard_breadth_filter_applied" in source
    assert "unseen_test_used" in source


def test_all_notebook_code_cells_compile() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "07_frozen_model_training.ipynb"
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
