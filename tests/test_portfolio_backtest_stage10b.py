from __future__ import annotations

from pathlib import Path
import copy

import pandas as pd
import pytest
import yaml

from src.evaluation.portfolio_backtest_stage10b import (
    build_stage10b_scenarios,
    file_sha256,
    prepare_stage10b_signals,
)


def _base_filter_config(tmp_path: Path, lock_path: Path) -> dict:
    return {
        "frozen_stage09": {
            "inference_lock_file": str(lock_path.relative_to(tmp_path)),
            "expected_inference_lock_sha256": file_sha256(lock_path),
            "expected_candidate_events": 6,
        },
        "final_signal_filter": {
            "started_filter_rule": "nonzero",
            "started_source": {
                "primary_directory": "data_ready/unseen_test",
                "fallback_directory": "raw_data",
                "date_column": "dEven",
                "column": "started",
            },
            "zigzag": {"threshold_fraction": 0.15},
            "daily_policy": {
                "selected_fraction": 0.50,
                "minimum_per_date": 1,
            },
        },
    }


def test_started_nonzero_is_applied_before_reranking(tmp_path: Path) -> None:
    lock_dir = tmp_path / "results/predictions"
    source_dir = tmp_path / "data_ready/unseen_test"
    lock_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)

    lock = pd.DataFrame(
        {
            "event_id": [
                "AAA::2021-01-01",
                "BBB::2021-01-01",
                "CCC::2021-01-01",
                "AAA::2021-01-02",
                "BBB::2021-01-02",
                "CCC::2021-01-02",
            ],
            "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "dEven": [
                "2021-01-01",
                "2021-01-01",
                "2021-01-01",
                "2021-01-02",
                "2021-01-02",
                "2021-01-02",
            ],
            "xgboost_ranking_score": [0.99, 0.90, 0.80, 0.95, 0.85, 0.75],
            "daily_candidate_count": [3, 3, 3, 3, 3, 3],
            "daily_rank": [1, 2, 3, 1, 2, 3],
            "daily_signal_quota": [1, 1, 1, 1, 1, 1],
            "daily_score_cutoff": [0.99, 0.99, 0.99, 0.95, 0.95, 0.95],
            "selected_signal": [True, False, False, True, False, False],
        }
    )
    lock_path = lock_dir / "lock.csv"
    lock.to_csv(lock_path, index=False)

    pd.DataFrame(
        {
            "dEven": ["2021-01-01", "2021-01-02"],
            "started": [0, 1],
        }
    ).to_csv(source_dir / "AAA.csv", index=False)
    pd.DataFrame(
        {
            "dEven": ["2021-01-01", "2021-01-02"],
            "started": [1, 2],
        }
    ).to_csv(source_dir / "BBB.csv", index=False)
    pd.DataFrame(
        {
            "dEven": ["2021-01-01", "2021-01-02"],
            "started": [1, 1],
        }
    ).to_csv(source_dir / "CCC.csv", index=False)

    config = _base_filter_config(tmp_path, lock_path)
    reranked, selected, _, date_audit, summary = prepare_stage10b_signals(
        repository_root=tmp_path,
        config=config,
    )

    # Day 1: AAA had the highest score but started=0, so BBB becomes rank 1.
    day1 = reranked.loc[reranked["dEven"].eq(pd.Timestamp("2021-01-01"))]
    assert day1["event_id"].tolist() == [
        "BBB::2021-01-01",
        "CCC::2021-01-01",
    ]
    assert day1["daily_rank"].tolist() == [1, 2]
    assert selected.loc[
        selected["dEven"].eq(pd.Timestamp("2021-01-01")), "event_id"
    ].tolist() == ["BBB::2021-01-01"]

    # started=2 is accepted because every nonzero value passes.
    assert "BBB::2021-01-02" in set(reranked["event_id"])
    assert summary["started_nonzero_candidates"] == 5
    assert summary["started_equal_zero_candidates"] == 1
    assert summary["started_greater_than_one_candidates"] == 1
    assert int(date_audit["stage10b_selected_signals"].sum()) == 3


def test_inference_lock_hash_is_mandatory(tmp_path: Path) -> None:
    lock_dir = tmp_path / "results/predictions"
    lock_dir.mkdir(parents=True)
    lock_path = lock_dir / "lock.csv"
    pd.DataFrame(
        {
            "event_id": ["AAA::2021-01-01"],
            "symbol": ["AAA"],
            "dEven": ["2021-01-01"],
            "xgboost_ranking_score": [0.9],
            "daily_candidate_count": [1],
            "daily_rank": [1],
            "daily_signal_quota": [1],
            "daily_score_cutoff": [0.9],
            "selected_signal": [True],
        }
    ).to_csv(lock_path, index=False)
    config = _base_filter_config(tmp_path, lock_path)
    config["frozen_stage09"]["expected_candidate_events"] = 1
    config["frozen_stage09"]["expected_inference_lock_sha256"] = "0" * 64

    with pytest.raises(AssertionError, match="inference lock changed"):
        prepare_stage10b_signals(repository_root=tmp_path, config=config)


def test_scenario_grid_has_36_scenarios_and_one_primary() -> None:
    config_path = Path(__file__).parents[1] / "configs/portfolio_backtest_stage10b.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    capital_summary = {"primary_initial_capital_irr": 20_000_000_000.0}
    scenarios = build_stage10b_scenarios(config, capital_summary)

    assert len(scenarios) == 36
    assert sum(scenario.is_primary for scenario, _, _ in scenarios) == 1
    primary = next(item for item in scenarios if item[0].is_primary)
    scenario, scenario_config, metadata = primary
    assert scenario.scenario_id == "liq_adv50__cap_broad__multi_lot__trailing"
    assert scenario_config["liquidity"]["maximum_fraction_of_adv20"] == 0.50
    assert scenario_config["capacity"]["maximum_distinct_symbols"] == 30
    assert scenario_config["capacity"]["maximum_open_lots"] == 60
    assert scenario_config["capacity"]["maximum_new_lots_per_day"] == 10
    assert scenario_config["exposure"]["maximum_symbol_exposure_fraction"] == 0.20
    assert metadata["maximum_open_lots_per_symbol"] == 3


def test_stage10b_outputs_do_not_use_stage10_filenames() -> None:
    module_path = (
        Path(__file__).parents[1]
        / "src/evaluation/portfolio_backtest_stage10b.py"
    )
    text = module_path.read_text(encoding="utf-8")
    assert '"10b_scenario_summary.csv"' in text
    assert '"10b_trade_ledger.csv"' in text
    assert '"10b_exploratory_portfolio_backtest_manifest.json"' in text
    assert '"10b_started_nonzero_zigzag15_filtered_inference.csv"' in text
    assert '"10b_started_nonzero_zigzag15_selected_signals.csv"' in text
    assert 'backtests_dir / "10_scenario_summary.csv"' not in text
