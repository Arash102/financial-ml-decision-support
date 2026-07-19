from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.features.stage04_breadth_extension import (
    STAGE04_BREADTH_FEATURES,
    Stage04BreadthConfig,
    build_daily_market_breadth,
    run_stage04_breadth_extension,
)


def _synthetic_observations() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=45, freq="D")
    rows = []
    patterns = {
        "UP": np.ones(len(dates)),
        "DOWN": -np.ones(len(dates)),
        "FLAT": np.zeros(len(dates)),
    }
    for symbol, signs in patterns.items():
        for date, sign in zip(dates, signs, strict=True):
            rows.append(
                {
                    "dEven": date,
                    "symbol": symbol,
                    "symbol_return": sign * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_daily_breadth_is_bounded_and_prefix_invariant() -> None:
    config = Stage04BreadthConfig(
        ema_span=5,
        ema_min_periods=5,
        slope_lag_market_dates=2,
    )
    observations = _synthetic_observations()
    full, _ = build_daily_market_breadth(
        observations,
        config=config,
    )
    assert full["market_breadth_raw"].between(-1, 1).all()

    cutoff = full["dEven"].iloc[20]
    prefix, _ = build_daily_market_breadth(
        observations.loc[
            observations["dEven"].le(cutoff)
        ],
        config=config,
    )
    expected = full.loc[
        full["dEven"].le(cutoff)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        prefix.reset_index(drop=True),
        expected,
    )


def test_warmup_is_encoded_without_future_data() -> None:
    observations = _synthetic_observations()
    daily, audit = build_daily_market_breadth(
        observations,
        config=Stage04BreadthConfig(
            ema_span=30,
            ema_min_periods=30,
            slope_lag_market_dates=5,
            warmup_numeric_fill_value=0.0,
        ),
    )
    warmup = daily["market_breadth_regime"].eq(
        "warmup_unavailable"
    )
    assert warmup.any()
    assert daily["market_breadth_ema30"].notna().all()
    assert daily["market_breadth_slope5"].notna().all()
    assert daily.loc[
        warmup,
        "market_breadth_slope5",
    ].eq(0.0).all()
    assert (
        audit["ema30_missing_rows_before_warmup_encoding"]
        > 0
    )
    assert (
        audit["slope5_missing_rows_before_warmup_encoding"]
        > 0
    )


def test_unchanged_symbols_remain_in_denominator() -> None:
    observations = pd.DataFrame(
        {
            "dEven": [
                "2020-01-02",
                "2020-01-02",
            ],
            "symbol": ["UP", "FLAT"],
            "symbol_return": [0.10, 0.0],
        }
    )
    daily, _ = build_daily_market_breadth(
        observations,
        config=Stage04BreadthConfig(
            ema_span=1,
            ema_min_periods=1,
            slope_lag_market_dates=1,
        ),
    )
    assert daily.loc[0, "valid_symbol_count"] == 2
    assert daily.loc[0, "unchanged_symbol_count"] == 1
    assert np.isclose(
        daily.loc[0, "market_breadth_raw"],
        0.5,
    )


def test_regime_labels_are_valid() -> None:
    daily, _ = build_daily_market_breadth(
        _synthetic_observations(),
        config=Stage04BreadthConfig(
            ema_span=5,
            ema_min_periods=5,
            slope_lag_market_dates=2,
            transition_lower=-1.0,
            transition_upper=1.0,
        ),
    )
    observed = set(
        daily["market_breadth_regime"].astype(str)
    )
    assert observed.issubset(
        {
            "warmup_unavailable",
            "recovery_negative",
            "recovery_positive",
            "deterioration",
            "neutral_transition",
        }
    )


def test_full_extension_preserves_candidate_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path
    (root / "raw_data").mkdir()
    (root / "notebooks").mkdir()
    (root / "results/manifests").mkdir(parents=True)
    (root / "results/audits").mkdir(parents=True)
    (root / "data_ready/candidates/train").mkdir(parents=True)
    (root / "data_ready/labeled/train").mkdir(parents=True)

    symbols = [f"S{index:03d}" for index in range(499)]
    pd.DataFrame({"symbol": symbols}).to_csv(
        root / "results/manifests/02_frozen_universe.csv",
        index=False,
    )

    base_features = [f"f{index}" for index in range(35)]
    pd.DataFrame(
        {
            "feature_order": range(1, 36),
            "feature": base_features,
            "data_type": ["numeric"] * 35,
        }
    ).to_csv(
        root / "results/manifests/04_approved_model_features.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "feature_order": range(1, 36),
            "feature": base_features,
            "semantic_group": ["base"] * 35,
            "source_feature": ["base"] * 35,
            "transformation": ["base"] * 35,
            "unit_before": ["x"] * 35,
            "unit_after": ["x"] * 35,
            "data_type": ["numeric"] * 35,
            "price_basis": ["adjusted"] * 35,
            "approved_for_pooled_model": [True] * 35,
        }
    ).to_csv(
        root / "results/manifests/04_final_model_feature_schema.csv",
        index=False,
    )
    (
        root
        / "results/manifests/04_feature_and_leakage_audit_manifest.json"
    ).write_text(
        json.dumps(
            {
                "approved_model_feature_count": 35,
                "approved_model_features": base_features,
            }
        ),
        encoding="utf-8",
    )

    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    for index, symbol in enumerate(symbols):
        growth = 1.001 + (index % 3 - 1) * 0.0005
        prices = 100.0 * np.cumprod(
            np.repeat(growth, len(dates))
        )
        pd.DataFrame(
            {
                "dEven": dates,
                "adj_last_price": prices,
                "started": np.arange(len(dates)) % 7,
            }
        ).to_csv(
            root / "raw_data" / f"{symbol}.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "event_id": [
                    f"{symbol}-a",
                    f"{symbol}-b",
                ],
                "dEven": [
                    dates[-2],
                    dates[-1],
                ],
                "base_value": [index, index + 1],
            }
        ).to_csv(
            root
            / "data_ready/candidates/train"
            / f"{symbol}_train_candidates.csv",
            index=False,
        )

    result = run_stage04_breadth_extension(root)
    assert result["summary"]["approved_features"] == 40
    assert result["summary"]["candidate_identity_preserved"] is True
    assert result["summary"]["hard_started_filter_applied"] is False
    assert result["summary"]["hard_breadth_filter_applied"] is False

    sample = pd.read_csv(
        root
        / "data_ready/candidates/train"
        / "S000_train_candidates.csv"
    )
    assert sample["event_id"].tolist() == [
        "S000-a",
        "S000-b",
    ]
    assert set(STAGE04_BREADTH_FEATURES).issubset(
        sample.columns
    )
