from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.stage09_abstention_audit import (
    FrozenPolicy,
    as_bool,
    classification_metrics,
    corrected_outcome_metrics,
    independently_reconstruct_selected_outcomes,
    parse_market_date,
    reconstruct_policy_decisions,
)


def test_parse_market_date_handles_numeric_and_iso() -> None:
    parsed = parse_market_date(
        pd.Series(["20200101", "2020-01-02"])
    )
    assert parsed.dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-01",
        "2020-01-02",
    ]


def test_boolean_parser_rejects_truthy_string_bug() -> None:
    parsed = as_bool(
        pd.Series(["True", "False", "1", "0"])
    )
    assert parsed.tolist() == [True, False, True, False]


def test_independent_policy_reconstruction_allows_zero_dates() -> None:
    frame = pd.DataFrame({
        "event_id": [
            "A::2020-01-01",
            "B::2020-01-01",
            "A::2020-01-02",
            "B::2020-01-02",
        ],
        "symbol": ["A", "B", "A", "B"],
        "dEven": [
            "2020-01-01",
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
        ],
        "market_breadth_regime": [
            "recovery_positive",
            "recovery_positive",
            "deterioration",
            "deterioration",
        ],
        "xgboost_ranking_score": [
            0.8,
            0.7,
            0.9,
            0.85,
        ],
    })
    policy = FrozenPolicy(
        gate_name="G3_recovery_only",
        allowed_regimes=(
            "recovery_negative",
            "recovery_positive",
        ),
        minimum_raw_score=0.75,
        maximum_daily_fraction=0.50,
        minimum_signals_per_date=0,
    )
    result = reconstruct_policy_decisions(
        frame,
        policy=policy,
    )
    selected_by_date = result.groupby(
        "dEven",
        observed=False,
    )["audit_selected_signal"].sum()

    assert selected_by_date.loc[
        pd.Timestamp("2020-01-01")
    ] == 1
    assert selected_by_date.loc[
        pd.Timestamp("2020-01-02")
    ] == 0


def test_classification_metrics_are_independent() -> None:
    result = classification_metrics(
        pd.Series([1, 0, 1, 0, 1]),
        pd.Series([True, True, False, False, True]),
    )
    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_negative"] == 1
    assert np.isclose(result["precision"], 2 / 3)
    assert np.isclose(result["specificity"], 1 / 2)
    assert np.isclose(result["sensitivity"], 2 / 3)


def test_corrected_outcome_metrics() -> None:
    frame = pd.DataFrame({
        "meta_label": [1, 0, 1, 0],
        "corrected_event_return": [
            0.20,
            -0.15,
            0.10,
            0.0,
        ],
        "barrier_touched": [
            "upper",
            "lower",
            "vertical",
            "vertical",
        ],
        "holding_period_observations": [
            2,
            1,
            3,
            3,
        ],
    })
    result = corrected_outcome_metrics(frame)

    assert result["events"] == 4
    assert result["winning_events"] == 2
    assert result["losing_events_negative_return"] == 1
    assert result["breakeven_events"] == 1
    assert np.isclose(result["win_rate"], 0.5)
    assert np.isclose(
        result["average_winning_return"],
        0.15,
    )
    assert np.isclose(
        result["average_losing_return"],
        -0.15,
    )
    assert np.isclose(result["payoff_ratio"], 1.0)
    assert np.isclose(result["profit_factor"], 2.0)


def _make_raw_file(path: Path) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    raw = pd.DataFrame({
        "dEven": dates.strftime("%Y%m%d"),
        "adj_last_price": [
            100, 101, 102, 103, 104,
            105, 106, 107, 108, 109,
        ],
        "adj_high": [
            100, 110, 120, 115, 108,
            109, 110, 111, 112, 113,
        ],
    })
    raw.to_csv(path, index=False)
    return raw


def test_raw_outcome_reconstruction_matches_all_rules(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw_data"
    raw_root.mkdir()
    _make_raw_file(raw_root / "AAA.csv")

    # Upper event: entry 100, next-three max high 120 => +20%.
    # Lower event: fixed -15%.
    # Vertical event: entry 102, day-three last 105.
    vertical_return = 105 / 102 - 1.0
    selected = pd.DataFrame([
        {
            "event_id": "AAA::2020-01-01",
            "symbol": "AAA",
            "dEven": "2020-01-01",
            "meta_label": 1,
            "barrier_touched": "upper",
            "event_end_date": "2020-01-03",
            "original_event_return": 0.15,
            "entry_adjusted_last_price": 100.0,
            "outcome_window_end_date": "2020-01-04",
            "outcome_window_uses_tail": False,
            "day30_adjusted_last_price": 103.0,
            "day30_return_reconstructed": 0.03,
            "maximum_adjusted_high_30": 120.0,
            "maximum_adjusted_high_date_30": "2020-01-03",
            "maximum_adjusted_high_observation_30": 2,
            "maximum_positive_return_30": 0.20,
            "corrected_event_return": 0.20,
            "corrected_return_rule": (
                "upper_maximum_adjusted_high_return_over_"
                "next_30_trading_observations"
            ),
            "corrected_winner": True,
            "corrected_outcome_date": "2020-01-03",
            "corrected_outcome_observation": 2,
        },
        {
            "event_id": "AAA::2020-01-02",
            "symbol": "AAA",
            "dEven": "2020-01-02",
            "meta_label": 0,
            "barrier_touched": "lower",
            "event_end_date": "2020-01-03",
            "original_event_return": -0.15,
            "entry_adjusted_last_price": 101.0,
            "outcome_window_end_date": "2020-01-05",
            "outcome_window_uses_tail": False,
            "day30_adjusted_last_price": 104.0,
            "day30_return_reconstructed": 104 / 101 - 1.0,
            "maximum_adjusted_high_30": 120.0,
            "maximum_adjusted_high_date_30": "2020-01-03",
            "maximum_adjusted_high_observation_30": 1,
            "maximum_positive_return_30": 120 / 101 - 1.0,
            "corrected_event_return": -0.15,
            "corrected_return_rule": "lower_fixed_minus_15_percent",
            "corrected_winner": False,
            "corrected_outcome_date": "2020-01-03",
            "corrected_outcome_observation": 1,
        },
        {
            "event_id": "AAA::2020-01-03",
            "symbol": "AAA",
            "dEven": "2020-01-03",
            "meta_label": 1,
            "barrier_touched": "vertical",
            "event_end_date": "2020-01-06",
            "original_event_return": vertical_return,
            "entry_adjusted_last_price": 102.0,
            "outcome_window_end_date": "2020-01-06",
            "outcome_window_uses_tail": False,
            "day30_adjusted_last_price": 105.0,
            "day30_return_reconstructed": vertical_return,
            "maximum_adjusted_high_30": 115.0,
            "maximum_adjusted_high_date_30": "2020-01-04",
            "maximum_adjusted_high_observation_30": 1,
            "maximum_positive_return_30": 115 / 102 - 1.0,
            "corrected_event_return": vertical_return,
            "corrected_return_rule": (
                "vertical_adjusted_last_return_on_"
                "trading_observation_30"
            ),
            "corrected_winner": True,
            "corrected_outcome_date": "2020-01-06",
            "corrected_outcome_observation": 3,
        },
    ])

    audit, errors = independently_reconstruct_selected_outcomes(
        selected,
        raw_root=raw_root,
        signal_generation_end=pd.Timestamp("2020-01-10"),
        tail_end=pd.Timestamp("2020-01-10"),
        horizon=3,
        upper_barrier=0.15,
        lower_barrier=-0.15,
    )

    assert errors.empty
    assert len(audit) == 3
    assert audit["passed"].all()
    assert audit["failed_field_count"].eq(0).all()


def test_policy_rejects_nonzero_minimum() -> None:
    policy = FrozenPolicy(
        gate_name="bad",
        allowed_regimes=("recovery_positive",),
        minimum_raw_score=0.5,
        maximum_daily_fraction=0.05,
        minimum_signals_per_date=1,
    )
    try:
        policy.validate()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Nonzero minimum signals must be rejected."
        )
