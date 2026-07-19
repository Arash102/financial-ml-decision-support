from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.market_breadth_stage10c import (
    STAGE10C_SCHEMA_VERSION,
    _select_daily_top_fraction,
    build_existing_market_features,
    compute_market_breadth,
)


def test_started_configuration_is_exactly_1_to_3() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/market_breadth_stage10c.yaml").read_text(encoding="utf-8")
    )
    assert config["schema_version"] == STAGE10C_SCHEMA_VERSION
    assert config["started_filter"]["minimum_inclusive"] == 1
    assert config["started_filter"]["maximum_inclusive"] == 3


def test_market_breadth_counts_and_range() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B", "C", "C"],
            "dEven": [
                "2021-01-01", "2021-01-02",
                "2021-01-01", "2021-01-02",
                "2021-01-01", "2021-01-02",
            ],
            "adj_last_price": [100, 110, 100, 90, 100, 100],
        }
    )
    daily = compute_market_breadth(
        frame,
        universe_size=3,
        ema_span=1,
        ema_min_periods=1,
        slope_lag=1,
    )
    row = daily.loc[daily["dEven"].eq(pd.Timestamp("2021-01-02"))].iloc[0]
    assert row["positive_symbols"] == 1
    assert row["negative_symbols"] == 1
    assert row["unchanged_symbols"] == 1
    assert np.isclose(row["market_breadth_raw"], 0.0)
    assert daily["market_breadth_raw"].dropna().between(-1, 1).all()


def test_recovery_and_deterioration_use_level_and_slope() -> None:
    dates = pd.date_range("2021-01-01", periods=8)
    parts = []
    # Build a cross-section that moves from negative breadth toward positive.
    prices = {
        "A": [100, 99, 98, 99, 100, 101, 102, 103],
        "B": [100, 99, 98, 97, 98, 99, 100, 101],
        "C": [100, 99, 100, 101, 102, 103, 104, 105],
    }
    for symbol, values in prices.items():
        parts.append(pd.DataFrame({"symbol": symbol, "dEven": dates, "adj_last_price": values}))
    daily = compute_market_breadth(
        pd.concat(parts, ignore_index=True),
        universe_size=3,
        ema_span=2,
        ema_min_periods=2,
        slope_lag=1,
        transition_lower_bound=-1.0,
        transition_upper_bound=1.0,
    )
    assert daily["market_regime"].isin(
        ["warmup_or_missing", "recovery", "deterioration", "transition_flat"]
    ).all()
    assert daily["market_recovery"].any()


def test_daily_top_fraction_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["B::1", "A::1", "C::1", "A::2"],
            "symbol": ["B", "A", "C", "A"],
            "dEven": ["2021-01-01", "2021-01-01", "2021-01-01", "2021-01-02"],
            "score": [0.9, 0.9, 0.8, 0.7],
        }
    )
    ranked = _select_daily_top_fraction(
        frame,
        score_column="score",
        fraction=0.05,
        minimum_per_date=1,
    )
    selected = ranked.loc[ranked["variant_selected_signal"]]
    assert selected.loc[selected["dEven"].eq(pd.Timestamp("2021-01-01")), "symbol"].tolist() == ["A"]
    assert len(selected) == 2


def test_market_feature_prefix_invariance() -> None:
    dates = pd.date_range("2020-01-01", periods=100)
    base = pd.DataFrame(
        {
            "dEven": dates,
            "xNivInuClMresIbs": np.linspace(1000, 1200, 100),
            "xNivInuPbMresIbs": np.linspace(990, 1190, 100),
            "xNivInuPhMresIbs": np.linspace(1010, 1210, 100),
        }
    )
    prefix = build_existing_market_features(base.iloc[:80])
    full = build_existing_market_features(base)
    columns = [column for column in prefix.columns if column != "dEven"]
    pd.testing.assert_frame_equal(
        prefix[columns].reset_index(drop=True),
        full.iloc[:80][columns].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_breadth_prefix_invariance() -> None:
    dates = pd.date_range("2020-01-01", periods=50)
    parts = []
    for offset, symbol in enumerate(["A", "B", "C", "D"]):
        values = 100 + np.arange(50) * (1 if offset % 2 == 0 else -0.2) + offset
        parts.append(pd.DataFrame({"symbol": symbol, "dEven": dates, "adj_last_price": values}))
    panel = pd.concat(parts, ignore_index=True)
    prefix = compute_market_breadth(
        panel.loc[panel["dEven"].le(dates[39])],
        universe_size=4,
        ema_span=5,
        ema_min_periods=5,
        slope_lag=2,
    )
    full = compute_market_breadth(
        panel,
        universe_size=4,
        ema_span=5,
        ema_min_periods=5,
        slope_lag=2,
    )
    columns = [
        "market_breadth_raw",
        "market_breadth_ema30",
        "market_breadth_slope5",
        "market_breadth_scaled_100",
    ]
    pd.testing.assert_frame_equal(
        prefix[columns].reset_index(drop=True),
        full.iloc[: len(prefix)][columns].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
