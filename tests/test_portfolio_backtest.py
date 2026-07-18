from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.portfolio_backtest import (
    BacktestScenario,
    MarketHistory,
    cost_adjusted_stop_loss_fraction,
    load_market_history,
    simulate_scenario,
)


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "portfolio_backtest.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _history(symbol: str, frame: pd.DataFrame) -> MarketHistory:
    return MarketHistory(
        symbol=symbol,
        frame=frame,
        date_to_position={
            pd.Timestamp(value): int(index)
            for index, value in enumerate(frame["dEven"])
        },
        liquidity_source="direct:test",
        raw_path=Path(f"/synthetic/{symbol}.csv"),
    )


def _base_frame(periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "dEven": dates,
            "adj_open": np.full(periods, 100.0),
            "adj_high": np.full(periods, 101.0),
            "adj_low": np.full(periods, 99.0),
            "adj_last_price": np.full(periods, 100.0),
            "traded_value_irr": np.full(periods, 1.0e12),
            "adv20_irr": np.full(periods, 1.0e12),
        }
    )


def _plan(
    symbol: str,
    frame: pd.DataFrame,
    signal_position: int,
    score: float = 0.9,
) -> dict:
    return {
        "event_id": f"{symbol}::{frame.iloc[signal_position]['dEven'].date()}",
        "symbol": symbol,
        "signal_date": frame.iloc[signal_position]["dEven"],
        "entry_date": frame.iloc[signal_position + 1]["dEven"],
        "signal_position": signal_position,
        "entry_position": signal_position + 1,
        "horizon_end_position": signal_position + 30,
        "horizon_end_date": frame.iloc[signal_position + 30]["dEven"],
        "xgboost_ranking_score": score,
        "daily_rank": 1,
        "daily_signal_quota": 1,
    }


def test_cost_adjusted_stop_fraction_is_frozen_value() -> None:
    value = cost_adjusted_stop_loss_fraction(
        buy_fee_rate=0.00464,
        sell_fee_rate=0.00964,
        buy_slippage_fraction=0.002,
        sell_slippage_fraction=0.002,
        stop_loss_fraction=0.15,
    )
    assert np.isclose(value, 0.16375777591973245)


def test_same_bar_adverse_stop_occurs_before_target() -> None:
    config = _config()
    frame = _base_frame()
    frame.loc[1, "adj_high"] = 130.0
    frame.loc[1, "adj_low"] = 70.0

    scenario = BacktestScenario(
        scenario_id="same_bar",
        initial_capital_irr=20_316_282_773.64711,
        slippage_each_side=0.002,
        position_structure="multi_lot",
        exit_style="fixed_take_profit",
        is_primary=True,
    )
    trades, decisions, _, _ = simulate_scenario(
        scenario=scenario,
        signal_plans=pd.DataFrame([_plan("AAA", frame, 0)]),
        market_histories={"AAA": _history("AAA", frame)},
        config=config,
    )
    assert decisions.iloc[0]["decision"] == "accepted"
    assert trades.iloc[0]["exit_reason"] == "initial_intraday_stop"
    assert trades.iloc[0]["net_return"] < 0.0


def test_trailing_protection_releases_planned_risk_only_next_open() -> None:
    config = _config()
    frame = _base_frame()
    frame.loc[2, "adj_high"] = 120.0
    frame.loc[3:, "adj_open"] = 118.0
    frame.loc[3:, "adj_high"] = 120.0
    frame.loc[3:, "adj_low"] = 117.0
    frame.loc[3:, "adj_last_price"] = 118.0

    scenario = BacktestScenario(
        scenario_id="trailing_release",
        initial_capital_irr=20_316_282_773.64711,
        slippage_each_side=0.002,
        position_structure="multi_lot",
        exit_style="trailing",
        is_primary=True,
    )
    trades, _, daily, integrity = simulate_scenario(
        scenario=scenario,
        signal_plans=pd.DataFrame([_plan("AAA", frame, 0)]),
        market_histories={"AAA": _history("AAA", frame)},
        config=config,
    )

    # Activation is observed on 2021-01-03. Protection becomes executable
    # on the next symbol trading open, 2021-01-04.
    pending_day = daily.loc[daily["date"].eq(pd.Timestamp("2021-01-03"))].iloc[0]
    active_day = daily.loc[daily["date"].eq(pd.Timestamp("2021-01-04"))].iloc[0]

    assert pending_day["protected_lots"] == 0
    assert pending_day["planned_open_risk_fraction"] > 0.0
    assert active_day["protected_lots"] == 1
    assert np.isclose(active_day["planned_open_risk_fraction"], 0.0)
    assert trades.iloc[0]["risk_release_date"] == pd.Timestamp("2021-01-04")
    assert integrity["protected_lot_nonzero_planned_risk_events"] == 0


def test_multilot_preserves_repeated_signals_and_single_lot_rejects_them() -> None:
    config = _config()
    frame = _base_frame()
    plans = pd.DataFrame(
        [
            _plan("AAA", frame, 0, 0.90),
            _plan("AAA", frame, 1, 0.89),
            _plan("AAA", frame, 2, 0.88),
            _plan("AAA", frame, 3, 0.87),
        ]
    )
    history = {"AAA": _history("AAA", frame)}

    multi = BacktestScenario(
        scenario_id="multi",
        initial_capital_irr=20_316_282_773.64711,
        slippage_each_side=0.002,
        position_structure="multi_lot",
        exit_style="trailing",
    )
    single = BacktestScenario(
        scenario_id="single",
        initial_capital_irr=20_316_282_773.64711,
        slippage_each_side=0.002,
        position_structure="single_lot",
        exit_style="trailing",
    )

    _, multi_decisions, _, _ = simulate_scenario(
        scenario=multi,
        signal_plans=plans,
        market_histories=history,
        config=config,
    )
    _, single_decisions, _, _ = simulate_scenario(
        scenario=single,
        signal_plans=plans,
        market_histories=history,
        config=config,
    )

    assert multi_decisions["decision"].tolist() == [
        "accepted",
        "accepted",
        "accepted",
        "rejected",
    ]
    assert (
        multi_decisions.iloc[3]["rejection_reason"]
        == "symbol_lot_cap_reached"
    )
    assert single_decisions["decision"].tolist() == [
        "accepted",
        "rejected",
        "rejected",
        "rejected",
    ]
    assert set(
        single_decisions.loc[
            single_decisions["decision"].eq("rejected"),
            "rejection_reason",
        ]
    ) == {"single_lot_symbol_already_open"}


def test_market_loader_drops_nonpositive_ohlc_rows_without_imputation(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "AAA.csv"
    frame = pd.DataFrame(
        {
            "dEven": [
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
                "2021-01-04",
            ],
            "adj_open": [100.0, 0.0, 102.0, 103.0],
            "adj_high": [101.0, 0.0, 103.0, 104.0],
            "adj_low": [99.0, 0.0, 101.0, 102.0],
            "adj_last_price": [100.0, 0.0, 102.0, 103.0],
            "qTotCap": [1.0e9, 0.0, 1.2e9, 1.3e9],
        }
    )
    frame.to_csv(raw_path, index=False)

    history = load_market_history(
        symbol="AAA",
        raw_path=raw_path,
        tail_end=pd.Timestamp("2021-01-31"),
        liquidity_cfg={
            "required": True,
            "adv_window_observations": 2,
            "minimum_adv_history_observations": 1,
            "direct_traded_value_column_candidates": ["qTotCap"],
            "volume_column_candidates": ["volume"],
        },
    )

    assert history.source_rows_after_date_filter == 4
    assert history.dropped_nonfinite_ohlc_rows == 0
    assert history.dropped_nonpositive_ohlc_rows == 1
    assert history.frame["dEven"].tolist() == [
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-01-03"),
        pd.Timestamp("2021-01-04"),
    ]
    assert not history.frame[
        ["adj_open", "adj_high", "adj_low", "adj_last_price"]
    ].le(0.0).any().any()
