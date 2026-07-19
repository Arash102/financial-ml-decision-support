from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.stage04_breadth_extension import (
    Stage04BreadthConfig,
    build_daily_market_breadth,
)
from src.features.unseen_breadth import (
    UNSEEN_BREADTH_SCHEMA_VERSION,
    load_started_run_length,
    merge_stage04_breadth_features,
    prepare_symbol_breadth_observations,
)


def test_config_discloses_post_hoc_status() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs"
            / "unseen_test_evaluation.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config["status"] == (
        "stage_09_configured_v4_breadth_post_hoc_retest"
    )
    status = config["scientific_status"]
    assert status[
        "prior_test_period_previously_inspected"
    ] is True
    assert status["confirmatory_claim_allowed"] is False
    assert status["future_confirmation_required"] is True

    frozen = config["frozen_inputs"]
    assert frozen["selected_model"] == "xgboost"
    assert frozen["selected_feature_set"] == "I_full_40"
    assert frozen["selected_trial"] == 10
    assert frozen["expected_raw_features"] == 40
    assert frozen["expected_transformed_features"] == 47
    assert frozen["expected_inference_lock_sha256"] is None
    assert frozen["expected_performance_metrics"] is None


def test_breadth_formula_includes_zero_returns_in_denominator() -> None:
    observations = pd.DataFrame(
        {
            "dEven": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
            ],
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "symbol_return": [0.1, -0.2, 0.0, 0.1, 0.2, 0.0],
        }
    )
    config = Stage04BreadthConfig(
        ema_span=2,
        ema_min_periods=1,
        slope_lag_market_dates=1,
    )
    daily, _ = build_daily_market_breadth(
        observations,
        config=config,
    )

    assert np.isclose(
        daily.loc[0, "market_breadth_raw"],
        0.0,
    )
    assert np.isclose(
        daily.loc[1, "market_breadth_raw"],
        2.0 / 3.0,
    )
    assert int(
        daily.loc[0, "valid_symbol_count"]
    ) == 3
    assert int(
        daily.loc[0, "unchanged_symbol_count"]
    ) == 1


def test_prepare_symbol_breadth_uses_previous_valid_observation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "A.csv"
    pd.DataFrame(
        {
            "dEven": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
            ],
            "adj_last_price": [100.0, np.nan, 110.0, 121.0],
        }
    ).to_csv(path, index=False)

    result = prepare_symbol_breadth_observations(
        path,
        symbol="A",
        horizon_end=pd.Timestamp("2020-01-04"),
    )

    assert result["dEven"].tolist() == [
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-04"),
    ]
    assert np.allclose(
        result["symbol_return"].to_numpy(dtype=float),
        [0.10, 0.10],
    )


def test_started_loader_prefers_raw_and_preserves_original_values(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.csv"
    fallback = tmp_path / "fallback.csv"

    pd.DataFrame(
        {
            "dEven": ["2020-01-01", "2020-01-02"],
            "started": [0, 7],
        }
    ).to_csv(raw, index=False)
    pd.DataFrame(
        {
            "dEven": ["2020-01-01", "2020-01-02"],
            "started": [99, 99],
        }
    ).to_csv(fallback, index=False)

    result, source = load_started_run_length(
        symbol="A",
        raw_path=raw,
        fallback_path=fallback,
        horizon_end=pd.Timestamp("2020-01-02"),
    )

    assert source == "raw_data"
    assert result["started_run_length"].tolist() == [0, 7]


def test_merge_preserves_candidate_identity() -> None:
    candidate = pd.DataFrame(
        {
            "event_id": ["A::2020-01-01", "A::2020-01-02"],
            "symbol": ["A", "A"],
            "dEven": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
            ],
            "base_feature": [1.0, 2.0],
        }
    )
    started = pd.DataFrame(
        {
            "dEven": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
            ],
            "started_run_length": [1.0, 2.0],
        }
    )
    breadth = pd.DataFrame(
        {
            "dEven": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
            ],
            "market_breadth_raw": [0.1, -0.1],
            "market_breadth_ema30": [0.2, 0.1],
            "market_breadth_slope5": [0.01, -0.01],
            "market_breadth_regime": [
                "recovery_positive",
                "deterioration",
            ],
        }
    )

    enriched, audit = merge_stage04_breadth_features(
        candidate,
        started_frame=started,
        breadth_frame=breadth,
    )

    assert enriched["event_id"].tolist() == candidate["event_id"].tolist()
    assert len(enriched) == len(candidate)
    assert audit["candidate_identity_preserved"] is True
    assert enriched[
        [
            "started_run_length",
            "market_breadth_raw",
            "market_breadth_ema30",
            "market_breadth_slope5",
            "market_breadth_regime",
        ]
    ].notna().all().all()


def test_notebook_uses_40_features_and_no_old_expected_outputs() -> None:
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

    assert "I_full_40" in source
    assert "len(SELECTED_FEATURES) == 40" in source
    assert "len(transformed_feature_names) == 47" in source
    assert "market_breadth_regime" in source
    assert "started_run_length" in source
    assert "post_hoc_retest_previously_inspected_period" in source
    assert "confirmatory_claim_allowed" in source

    assert (
        "62a7f3e58d53213c92f1803b19135895497ebf24b56fa472c54047d762740688"
        not in source
    )
    assert (
        "c29f1ec3b6d59fc5a2aa163f65b880562271f938fc7208a784ee820f5245c946"
        not in source
    )
    assert "expected_selected_corrected_metrics" not in source
    assert "inference_lock_identical_to_stage09_v1" not in source


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


def test_schema_version() -> None:
    assert UNSEEN_BREADTH_SCHEMA_VERSION == (
        "stage09_v4_causal_stage04_breadth_reconstruction"
    )
