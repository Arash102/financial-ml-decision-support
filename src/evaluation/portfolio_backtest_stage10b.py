"""Stage 10B exploratory/post-hoc portfolio evaluation.

This stage preserves the frozen Stage 09 model scores and the causal 15% ZigZag
candidate population. It adds exactly one final signal-quality filter requested
by the research owner: the already-computed ``started`` column must be nonzero.
The filter is applied to the full frozen Stage 09 inference-lock population
*before* daily score ranking and the frozen top-5% policy are recomputed.

Stage 10B is explicitly exploratory/post-hoc because the Stage 10 confirmatory
portfolio result was already observed before these rules were specified.
Outputs are written with a ``10b_`` prefix and never overwrite Stage 10 files.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import math
import subprocess

import numpy as np
import pandas as pd

from src.evaluation.portfolio_backtest import (
    BacktestScenario,
    MarketHistory,
    _project_date_parser,
    attach_execution_plan,
    calibrate_initial_capital,
    canonical_json_sha256,
    current_git_commit,
    discover_customer_file,
    discover_raw_file_map,
    file_sha256,
    load_market_history,
    simulate_scenario,
    summarize_scenario,
)


STAGE10B_SCHEMA_VERSION = (
    "stage10b_v1_2_exploratory_started_nonzero_zigzag15_capacity_liquidity"
)


def _git_is_ancestor(repository_root: Path, ancestor_sha: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor_sha, "HEAD"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _discover_csv_map(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    mapping: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.csv")):
        symbol = path.stem.strip()
        if symbol in mapping:
            raise ValueError(
                f"Duplicate CSV stem '{symbol}' below {directory}: "
                f"{mapping[symbol]} and {path}"
            )
        mapping[symbol] = path
    return mapping


def _select_daily_top_fraction(
    frame: pd.DataFrame,
    *,
    score_column: str,
    date_column: str,
    fraction: float,
    minimum_per_date: int,
    symbol_column: str = "symbol",
    event_id_column: str = "event_id",
) -> pd.DataFrame:
    """Reproduce the frozen deterministic Stage 08 daily ranking rule."""
    required = {
        score_column,
        date_column,
        symbol_column,
        event_id_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Signal-policy columns are missing: {missing}")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    if int(minimum_per_date) < 1:
        raise ValueError("minimum_per_date must be at least one.")

    result = frame.copy()
    result[date_column] = pd.to_datetime(
        result[date_column], errors="raise"
    ).dt.normalize()
    result[score_column] = pd.to_numeric(
        result[score_column], errors="raise"
    )
    if not np.isfinite(result[score_column].to_numpy(dtype=float)).all():
        raise ValueError("score column contains nonfinite values.")

    result = result.sort_values(
        [date_column, score_column, symbol_column, event_id_column],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["daily_candidate_count"] = result.groupby(
        date_column, sort=False
    )[event_id_column].transform("size").astype(int)
    result["daily_rank"] = (
        result.groupby(date_column, sort=False).cumcount() + 1
    ).astype(int)
    result["daily_signal_quota"] = result["daily_candidate_count"].map(
        lambda count: min(
            int(count),
            max(
                int(minimum_per_date),
                int(math.ceil(float(fraction) * int(count))),
            ),
        )
    ).astype(int)
    result["selected_signal"] = (
        result["daily_rank"] <= result["daily_signal_quota"]
    )
    cutoffs = (
        result.loc[result["selected_signal"]]
        .groupby(date_column, sort=False)[score_column]
        .min()
        .rename("daily_score_cutoff")
    )
    result = result.merge(
        cutoffs,
        left_on=date_column,
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    return result


def _load_started_for_lock(
    lock: pd.DataFrame,
    *,
    primary_directory: Path,
    fallback_directory: Path,
    date_column: str,
    started_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join the already-computed started value to every frozen lock event."""
    primary_map = _discover_csv_map(primary_directory)
    fallback_map = _discover_csv_map(fallback_directory)

    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for symbol, events in lock.groupby("symbol", sort=True):
        symbol = str(symbol)
        source_candidates: list[tuple[str, Path]] = []
        if symbol in primary_map:
            source_candidates.append(("unseen_test", primary_map[symbol]))
        if symbol in fallback_map:
            source_candidates.append(("raw_fallback", fallback_map[symbol]))

        if not source_candidates:
            error_rows.append(
                {
                    "symbol": symbol,
                    "error_type": "FileNotFoundError",
                    "error_message": (
                        "No symbol file was found in either started-source directory."
                    ),
                }
            )
            continue

        source_failures: list[str] = []
        loaded = False
        for source_role, source_path in source_candidates:
            try:
                source = pd.read_csv(
                    source_path,
                    usecols=[date_column, started_column],
                    low_memory=False,
                )
                source[date_column] = _project_date_parser(source[date_column])
                source[started_column] = pd.to_numeric(
                    source[started_column], errors="coerce"
                )
                source_rows = int(len(source))
                invalid_date_rows = int(source[date_column].isna().sum())
                duplicate_date_rows = int(
                    source[date_column].duplicated(keep=False).sum()
                )
                source = source.loc[source[date_column].notna()].copy()
                source = (
                    source.sort_values(date_column, kind="stable")
                    .drop_duplicates(date_column, keep="last")
                    .reset_index(drop=True)
                )

                joined = events[["event_id", "symbol", "dEven"]].merge(
                    source.rename(columns={date_column: "dEven"})[
                        ["dEven", started_column]
                    ],
                    on="dEven",
                    how="left",
                    validate="many_to_one",
                )
                missing_started = int(joined[started_column].isna().sum())
                noninteger_started = int(
                    (
                        joined[started_column].notna()
                        & ~np.isclose(
                            joined[started_column],
                            np.round(joined[started_column]),
                            atol=1.0e-12,
                        )
                    ).sum()
                )
                if missing_started:
                    raise ValueError(
                        f"{missing_started} frozen events have no started value."
                    )
                if noninteger_started:
                    raise ValueError(
                        f"{noninteger_started} frozen events have non-integer "
                        "started values."
                    )

                joined[started_column] = np.round(
                    joined[started_column]
                ).astype(int)
                joined["started_source_path"] = str(source_path)
                joined["started_source_role"] = source_role
                parts.append(joined)
                audit_rows.append(
                    {
                        "symbol": symbol,
                        "source_path": str(source_path),
                        "source_role": source_role,
                        "source_rows": source_rows,
                        "invalid_date_rows": invalid_date_rows,
                        "duplicate_date_rows_before_deduplication": (
                            duplicate_date_rows
                        ),
                        "lock_events": int(len(events)),
                        "joined_events": int(len(joined)),
                        "started_equal_zero": int(
                            joined[started_column].eq(0).sum()
                        ),
                        "started_nonzero": int(
                            joined[started_column].ne(0).sum()
                        ),
                        "started_equal_one": int(
                            joined[started_column].eq(1).sum()
                        ),
                        "started_greater_than_one": int(
                            joined[started_column].gt(1).sum()
                        ),
                        "started_less_than_zero": int(
                            joined[started_column].lt(0).sum()
                        ),
                        "minimum_started": int(joined[started_column].min()),
                        "maximum_started": int(joined[started_column].max()),
                    }
                )
                loaded = True
                break
            except Exception as exc:
                source_failures.append(
                    f"{source_role}:{source_path} -> "
                    f"{type(exc).__name__}: {exc}"
                )

        if not loaded:
            error_rows.append(
                {
                    "symbol": symbol,
                    "error_type": "StartedSourceResolutionError",
                    "error_message": " | ".join(source_failures),
                }
            )

    joined_panel = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame()
    )
    audit = pd.DataFrame(audit_rows)
    errors = pd.DataFrame(error_rows)
    return joined_panel, audit, errors


def prepare_stage10b_signals(
    *,
    repository_root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply started != 0 before re-ranking the locked ZigZag-eligible candidates."""
    policy = config["final_signal_filter"]
    stage09 = config["frozen_stage09"]
    lock_path = repository_root / stage09["inference_lock_file"]
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)

    actual_lock_sha = file_sha256(lock_path)
    expected_lock_sha = str(stage09["expected_inference_lock_sha256"])
    if actual_lock_sha != expected_lock_sha:
        raise AssertionError(
            f"Stage 09 inference lock changed: {actual_lock_sha}"
        )

    lock = pd.read_csv(lock_path, low_memory=False)
    required = {
        "event_id",
        "symbol",
        "dEven",
        "xgboost_ranking_score",
        "daily_candidate_count",
        "daily_rank",
        "daily_signal_quota",
        "daily_score_cutoff",
        "selected_signal",
    }
    missing = sorted(required - set(lock.columns))
    if missing:
        raise KeyError(f"Inference-lock columns are missing: {missing}")

    expected_rows = int(stage09["expected_candidate_events"])
    if len(lock) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} locked candidates, found {len(lock)}."
        )
    if lock["event_id"].duplicated().any():
        raise AssertionError("Duplicate event IDs exist in the inference lock.")

    lock["dEven"] = pd.to_datetime(lock["dEven"], errors="raise").dt.normalize()
    lock["xgboost_ranking_score"] = pd.to_numeric(
        lock["xgboost_ranking_score"], errors="raise"
    )
    if not np.isfinite(
        lock["xgboost_ranking_score"].to_numpy(dtype=float)
    ).all():
        raise ValueError("The frozen score contains nonfinite values.")

    lock = lock.rename(
        columns={
            "daily_candidate_count": "stage09_daily_candidate_count",
            "daily_rank": "stage09_daily_rank",
            "daily_signal_quota": "stage09_daily_signal_quota",
            "daily_score_cutoff": "stage09_daily_score_cutoff",
            "selected_signal": "stage09_selected_signal",
        }
    )
    selected_text = (
        lock["stage09_selected_signal"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    valid_selected_text = selected_text.isin({"true", "false", "1", "0"})
    if not valid_selected_text.all():
        raise ValueError("The frozen selected_signal column is not boolean.")
    lock["stage09_selected_signal"] = selected_text.isin({"true", "1"})

    started_source = policy["started_source"]
    started_panel, started_audit, started_errors = _load_started_for_lock(
        lock,
        primary_directory=repository_root / started_source["primary_directory"],
        fallback_directory=repository_root / started_source["fallback_directory"],
        date_column=str(started_source["date_column"]),
        started_column=str(started_source["column"]),
    )
    if not started_errors.empty:
        raise RuntimeError(
            f"{len(started_errors)} started-source errors exist."
        )
    if len(started_panel) != len(lock):
        raise AssertionError("The started join changed the lock population.")
    if started_panel["event_id"].duplicated().any():
        raise AssertionError("The started join produced duplicate event IDs.")

    enriched = lock.merge(
        started_panel[
            [
                "event_id",
                str(started_source["column"]),
                "started_source_path",
                "started_source_role",
            ]
        ],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    started_column = str(started_source["column"])
    enriched["started_filter_pass"] = enriched[started_column].ne(0)
    enriched["zigzag_filter_pass"] = True
    enriched["zigzag_filter_provenance"] = (
        "inherited_from_frozen_stage09_candidate_population"
    )
    enriched["zigzag_threshold_fraction"] = float(
        policy["zigzag"]["threshold_fraction"]
    )

    filtered = enriched.loc[enriched["started_filter_pass"]].copy()
    if filtered.empty:
        raise RuntimeError("No frozen candidate remains after started != 0.")

    reranked = _select_daily_top_fraction(
        filtered,
        score_column="xgboost_ranking_score",
        date_column="dEven",
        fraction=float(policy["daily_policy"]["selected_fraction"]),
        minimum_per_date=int(policy["daily_policy"]["minimum_per_date"]),
        symbol_column="symbol",
        event_id_column="event_id",
    )
    selected = reranked.loc[reranked["selected_signal"]].copy()
    selected = selected.sort_values(
        ["dEven", "daily_rank", "symbol", "event_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Base Stage 10 requires all selected rows to be true and their within-date
    # ranks to be consecutive after the final filter.
    if not selected["selected_signal"].all():
        raise AssertionError("The selected output contains an unselected row.")
    expected_rank = selected.groupby("dEven", sort=False).cumcount() + 1
    if not expected_rank.astype(int).eq(selected["daily_rank"]).all():
        raise AssertionError("Post-filter daily ranks are not consecutive.")

    date_audit = (
        enriched.groupby("dEven", sort=True)
        .agg(
            stage09_locked_candidates=("event_id", "size"),
            started_nonzero_candidates=("started_filter_pass", "sum"),
            original_stage09_selected=("stage09_selected_signal", "sum"),
        )
        .reset_index()
    )
    selected_by_date = (
        selected.groupby("dEven", sort=True)
        .agg(
            stage10b_selected_signals=("event_id", "size"),
            stage10b_score_cutoff=("xgboost_ranking_score", "min"),
        )
        .reset_index()
    )
    date_audit = date_audit.merge(
        selected_by_date,
        on="dEven",
        how="left",
        validate="one_to_one",
    )
    date_audit["stage10b_selected_signals"] = (
        date_audit["stage10b_selected_signals"].fillna(0).astype(int)
    )
    date_audit["started_nonzero_candidates"] = (
        date_audit["started_nonzero_candidates"].astype(int)
    )
    date_audit["dates_without_started_nonzero"] = (
        date_audit["started_nonzero_candidates"].eq(0)
    )

    summary = {
        "stage09_inference_lock_sha256": actual_lock_sha,
        "stage09_locked_candidates": int(len(enriched)),
        "stage09_signal_dates": int(enriched["dEven"].nunique()),
        "started_filter_rule": "started != 0",
        "started_nonzero_candidates": int(enriched["started_filter_pass"].sum()),
        "started_equal_one_candidates": int(enriched[started_column].eq(1).sum()),
        "started_equal_zero_candidates": int(enriched[started_column].eq(0).sum()),
        "started_greater_than_one_candidates": int(enriched[started_column].gt(1).sum()),
        "started_less_than_zero_candidates": int(enriched[started_column].lt(0).sum()),
        "dates_with_started_nonzero": int(filtered["dEven"].nunique()),
        "dates_without_started_nonzero": int(
            date_audit["dates_without_started_nonzero"].sum()
        ),
        "post_filter_selected_signals": int(len(selected)),
        "post_filter_signal_dates": int(selected["dEven"].nunique()),
        "daily_selected_fraction": float(
            policy["daily_policy"]["selected_fraction"]
        ),
        "daily_minimum_per_date": int(
            policy["daily_policy"]["minimum_per_date"]
        ),
        "zigzag_threshold_fraction": float(
            policy["zigzag"]["threshold_fraction"]
        ),
        "zigzag_recomputed": False,
        "zigzag_inherited_from_locked_candidate_population": True,
        "filter_order": [
            "load_frozen_stage09_inference_lock",
            "join_existing_started_by_symbol_and_date",
            "retain_started_nonzero",
            "inherit_exact_stage09_causal_zigzag15_candidate_eligibility",
            "rerank_remaining_candidates_score_desc_symbol_asc_event_id_asc",
            "apply_daily_top_fraction_policy",
        ],
    }
    return reranked, selected, started_audit, date_audit, summary


def build_stage10b_scenarios(
    config: dict[str, Any],
    capital_summary: dict[str, Any],
) -> list[tuple[BacktestScenario, dict[str, Any], dict[str, Any]]]:
    """Build targeted capacity/liquidity sensitivity scenarios."""
    grid = config["exploratory_scenario_grid"]
    liquidity_profiles = config["liquidity_profiles"]
    capacity_profiles = config["capacity_profiles"]
    primary_spec = config["primary_scenario"]

    initial_capital = float(capital_summary["primary_initial_capital_irr"])
    slippage = float(config["execution"]["primary_slippage_each_side"])

    scenarios: list[tuple[BacktestScenario, dict[str, Any], dict[str, Any]]] = []
    for liquidity_name in grid["liquidity_profiles"]:
        if liquidity_name not in liquidity_profiles:
            raise KeyError(f"Unknown liquidity profile: {liquidity_name}")
        liquidity_fraction = float(
            liquidity_profiles[liquidity_name]["maximum_fraction_of_adv20"]
        )
        for capacity_name in grid["capacity_profiles"]:
            if capacity_name not in capacity_profiles:
                raise KeyError(f"Unknown capacity profile: {capacity_name}")
            capacity = capacity_profiles[capacity_name]
            for structure in grid["position_structures"]:
                if structure not in {"multi_lot", "single_lot"}:
                    raise ValueError(f"Unknown position structure: {structure}")
                for exit_style in grid["exit_styles"]:
                    if exit_style not in {"trailing", "fixed_take_profit"}:
                        raise ValueError(f"Unknown exit style: {exit_style}")

                    is_primary = (
                        liquidity_name == primary_spec["liquidity_profile"]
                        and capacity_name == primary_spec["capacity_profile"]
                        and structure == primary_spec["position_structure"]
                        and exit_style == primary_spec["exit_style"]
                    )
                    scenario_id = (
                        f"liq_{liquidity_name}__cap_{capacity_name}__"
                        f"{structure}__{exit_style}"
                    )
                    scenario = BacktestScenario(
                        scenario_id=scenario_id,
                        initial_capital_irr=initial_capital,
                        slippage_each_side=slippage,
                        position_structure=structure,
                        exit_style=exit_style,
                        is_primary=is_primary,
                    )
                    scenario_config = deepcopy(config)
                    scenario_config["liquidity"][
                        "maximum_fraction_of_adv20"
                    ] = liquidity_fraction
                    scenario_config["capacity"].update(
                        {
                            "maximum_distinct_symbols": int(
                                capacity["maximum_distinct_symbols"]
                            ),
                            "maximum_open_lots": int(
                                capacity["maximum_open_lots"]
                            ),
                            "maximum_open_lots_per_symbol": int(
                                capacity["maximum_open_lots_per_symbol"]
                            ),
                            "maximum_new_lots_per_day": int(
                                capacity["maximum_new_lots_per_day"]
                            ),
                        }
                    )
                    scenario_config["exposure"][
                        "maximum_symbol_exposure_fraction"
                    ] = float(capacity["maximum_symbol_exposure_fraction"])
                    metadata = {
                        "liquidity_profile": liquidity_name,
                        "maximum_fraction_of_adv20": liquidity_fraction,
                        "capacity_profile": capacity_name,
                        "maximum_distinct_symbols": int(
                            capacity["maximum_distinct_symbols"]
                        ),
                        "maximum_open_lots": int(
                            capacity["maximum_open_lots"]
                        ),
                        "maximum_open_lots_per_symbol": int(
                            capacity["maximum_open_lots_per_symbol"]
                        ),
                        "maximum_new_lots_per_day": int(
                            capacity["maximum_new_lots_per_day"]
                        ),
                        "maximum_symbol_exposure_fraction": float(
                            capacity["maximum_symbol_exposure_fraction"]
                        ),
                    }
                    scenarios.append((scenario, scenario_config, metadata))

    primary_count = sum(item[0].is_primary for item in scenarios)
    if primary_count != 1:
        raise AssertionError(
            f"Expected one Stage 10B primary scenario, found {primary_count}."
        )
    return scenarios


def _add_scenario_metadata(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    for key, value in metadata.items():
        result[key] = value
    return result


def _validate_temporal_scope(config: dict[str, object]) -> None:
    required = {
        "unseen_test_start",
        "signal_generation_end",
        "outcome_observation_tail_end",
    }
    scope = config.get("temporal_scope")
    if not isinstance(scope, dict):
        raise KeyError(
            "Stage 10B configuration is missing the required temporal_scope block."
        )
    missing = sorted(required - set(scope))
    if missing:
        raise KeyError(
            f"Stage 10B temporal_scope fields are missing: {missing}"
        )

    unseen_start = pd.Timestamp(scope["unseen_test_start"])
    signal_end = pd.Timestamp(scope["signal_generation_end"])
    tail_end = pd.Timestamp(scope["outcome_observation_tail_end"])
    if unseen_start > signal_end:
        raise ValueError(
            "unseen_test_start cannot be after signal_generation_end."
        )
    if signal_end > tail_end:
        raise ValueError(
            "signal_generation_end cannot be after outcome_observation_tail_end."
        )


def run_stage10b(
    *,
    repository_root: Path,
    config: dict[str, Any],
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run the complete Stage 10B exploratory signal and portfolio evaluation."""
    repository_root = Path(repository_root).resolve()
    _validate_temporal_scope(config)
    baseline = config["baseline_stage10"]
    baseline_manifest_path = repository_root / baseline["manifest_file"]
    if not baseline_manifest_path.exists():
        raise FileNotFoundError(baseline_manifest_path)
    baseline_manifest = json.loads(
        baseline_manifest_path.read_text(encoding="utf-8")
    )
    if str(baseline_manifest.get("git_commit_sha")) != str(
        baseline["expected_git_commit_sha"]
    ):
        raise AssertionError("The baseline Stage 10 manifest commit changed.")
    if str(baseline_manifest.get("configuration_hash")) != str(
        baseline["expected_configuration_hash"]
    ):
        raise AssertionError("The baseline Stage 10 configuration hash changed.")
    if not _git_is_ancestor(
        repository_root,
        str(baseline["expected_git_commit_sha"]),
    ):
        raise AssertionError(
            "The current repository is not descended from the audited Stage 10 commit."
        )

    reranked, signals, started_audit, date_audit, filter_summary = (
        prepare_stage10b_signals(
            repository_root=repository_root,
            config=config,
        )
    )

    signal_end = pd.Timestamp(config["temporal_scope"]["signal_generation_end"])
    if signals["dEven"].gt(signal_end).any():
        raise AssertionError("A Stage 10B signal occurs after the frozen signal end.")

    customer_path = discover_customer_file(
        repository_root,
        config["initial_capital"]["source_file"],
    )
    customer_frame = pd.read_csv(customer_path, low_memory=False)
    capital_summary, capital_audit = calibrate_initial_capital(
        customer_frame,
        config,
    )

    tail_end = pd.Timestamp(config["temporal_scope"]["outcome_observation_tail_end"])
    raw_dir = repository_root / config["paths"]["raw_data"]
    raw_map = discover_raw_file_map(raw_dir)

    histories: dict[str, MarketHistory] = {}
    inventory_rows: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []
    for symbol in sorted(signals["symbol"].astype(str).unique()):
        raw_path = raw_map.get(symbol)
        if raw_path is None:
            load_errors.append(
                {
                    "symbol": symbol,
                    "error_type": "FileNotFoundError",
                    "error_message": "Raw file stem not found.",
                }
            )
            continue
        try:
            history = load_market_history(
                symbol=symbol,
                raw_path=raw_path,
                tail_end=tail_end,
                liquidity_cfg=config["liquidity"],
            )
            histories[symbol] = history
            inventory_rows.append(
                {
                    "symbol": symbol,
                    "raw_path": str(raw_path),
                    "source_rows_after_date_filter": (
                        history.source_rows_after_date_filter
                    ),
                    "rows": len(history.frame),
                    "dropped_nonfinite_ohlc_rows": (
                        history.dropped_nonfinite_ohlc_rows
                    ),
                    "dropped_nonpositive_ohlc_rows": (
                        history.dropped_nonpositive_ohlc_rows
                    ),
                    "dropped_invalid_ohlc_rows": (
                        history.dropped_nonfinite_ohlc_rows
                        + history.dropped_nonpositive_ohlc_rows
                    ),
                    "execution_valid_row_fraction": (
                        len(history.frame)
                        / history.source_rows_after_date_filter
                        if history.source_rows_after_date_filter > 0
                        else float("nan")
                    ),
                    "first_date": history.frame["dEven"].min(),
                    "last_date": history.frame["dEven"].max(),
                    "liquidity_source": history.liquidity_source,
                    "adv20_available_rows": int(
                        history.frame["adv20_irr"].notna().sum()
                    ),
                }
            )
        except Exception as exc:
            load_errors.append(
                {
                    "symbol": symbol,
                    "raw_path": str(raw_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    inventory = pd.DataFrame(inventory_rows)
    market_errors = pd.DataFrame(load_errors)
    if not market_errors.empty:
        raise RuntimeError(
            f"{len(market_errors)} market-history load errors exist."
        )

    signal_plans, signal_plan_errors = attach_execution_plan(
        signals,
        histories,
        horizon_observations=int(
            config["exit_policy"]["maximum_horizon_observations"]
        ),
    )
    if not signal_plan_errors.empty:
        raise RuntimeError(
            f"{len(signal_plan_errors)} signal execution-plan errors exist."
        )
    if len(signal_plans) != len(signals):
        raise AssertionError("The Stage 10B signal execution population changed.")

    scenario_specs = build_stage10b_scenarios(config, capital_summary)
    all_trades: list[pd.DataFrame] = []
    all_decisions: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []

    for index, (scenario, scenario_config, metadata) in enumerate(
        scenario_specs, start=1
    ):
        print(
            f"[{index:>2}/{len(scenario_specs)}] {scenario.scenario_id}",
            flush=True,
        )
        trades, decisions, daily, integrity = simulate_scenario(
            scenario=scenario,
            signal_plans=signal_plans,
            market_histories=histories,
            config=scenario_config,
        )
        trades = _add_scenario_metadata(trades, metadata)
        decisions = _add_scenario_metadata(decisions, metadata)
        daily = _add_scenario_metadata(daily, metadata)
        summary = summarize_scenario(
            scenario, trades, decisions, daily
        )
        summary.update(metadata)
        integrity.update(metadata)
        integrity["is_primary_scenario"] = scenario.is_primary

        all_trades.append(trades)
        all_decisions.append(decisions)
        all_daily.append(daily)
        summary_rows.append(summary)
        integrity_rows.append(integrity)

    trade_ledger = pd.concat(all_trades, ignore_index=True)
    signal_decisions = pd.concat(all_decisions, ignore_index=True)
    daily_equity = pd.concat(all_daily, ignore_index=True)
    scenario_summary = pd.DataFrame(summary_rows)
    integrity_audit = pd.DataFrame(integrity_rows)

    primary = scenario_summary.loc[scenario_summary["is_primary_scenario"]]
    if len(primary) != 1:
        raise AssertionError("Stage 10B primary scenario is not unique.")

    config_hash = canonical_json_sha256(config)
    manifest = {
        "stage": "10B",
        "status": "completed_exploratory_posthoc_internal_integrity_checks",
        "schema_version": STAGE10B_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": current_git_commit(repository_root),
        "configuration_hash": config_hash,
        "exploratory_posthoc": True,
        "confirmatory_claim_allowed": False,
        "baseline_stage10": {
            "manifest_file": str(baseline_manifest_path),
            "git_commit_sha": baseline_manifest["git_commit_sha"],
            "configuration_hash": baseline_manifest["configuration_hash"],
            "primary_total_return": baseline_manifest.get(
                "primary_scenario", {}
            ).get("total_return"),
        },
        "frozen_stage09": {
            "producer_commit_sha": config["frozen_stage09"][
                "producer_commit_sha"
            ],
            "inference_lock_sha256": filter_summary[
                "stage09_inference_lock_sha256"
            ],
            "model_scores_changed": False,
            "model_retrained": False,
            "zigzag_candidate_rule_changed": False,
        },
        "final_signal_filter": filter_summary,
        "capital_calibration": capital_summary,
        "customer_calibration_file_sha256": file_sha256(customer_path),
        "temporal_scope": config["temporal_scope"],
        "signals": {
            "selected": int(len(signals)),
            "planned": int(len(signal_plans)),
            "symbols": int(signals["symbol"].nunique()),
        },
        "scenario_count": int(len(scenario_summary)),
        "scenario_grid": {
            "liquidity_profiles": config["exploratory_scenario_grid"][
                "liquidity_profiles"
            ],
            "capacity_profiles": config["exploratory_scenario_grid"][
                "capacity_profiles"
            ],
            "position_structures": config["exploratory_scenario_grid"][
                "position_structures"
            ],
            "exit_styles": config["exploratory_scenario_grid"][
                "exit_styles"
            ],
            "slippage_each_side": float(
                config["execution"]["primary_slippage_each_side"]
            ),
            "capital_scenario": "primary_only",
        },
        "primary_scenario": primary.iloc[0].to_dict(),
        "market_data_quality": {
            "symbols_loaded": int(len(inventory)),
            "source_rows_after_date_filter": int(
                inventory["source_rows_after_date_filter"].sum()
            ),
            "valid_execution_rows": int(inventory["rows"].sum()),
            "dropped_nonfinite_ohlc_rows": int(
                inventory["dropped_nonfinite_ohlc_rows"].sum()
            ),
            "dropped_nonpositive_ohlc_rows": int(
                inventory["dropped_nonpositive_ohlc_rows"].sum()
            ),
            "invalid_rows_imputed": False,
        },
        "safeguards": {
            "long_only": True,
            "entry_on_next_trading_open": True,
            "same_bar_rule": "adverse_stop_first",
            "transaction_costs_applied": True,
            "slippage_applied": True,
            "market_regime_filter_added": False,
            "started_recomputed": False,
            "zigzag_recomputed": False,
            "outcome_columns_used_for_selection": False,
        },
    }

    outputs = {
        "filtered_inference": reranked,
        "selected_signals": signals,
        "started_source_audit": started_audit,
        "signal_filter_date_audit": date_audit,
        "capital_summary": capital_summary,
        "capital_audit": capital_audit,
        "market_inventory_audit": inventory,
        "market_errors": market_errors,
        "signal_plans": signal_plans,
        "signal_plan_errors": signal_plan_errors,
        "scenario_summary": scenario_summary,
        "trade_ledger": trade_ledger,
        "signal_decisions": signal_decisions,
        "daily_equity": daily_equity,
        "integrity_audit": integrity_audit,
        "manifest": manifest,
    }

    if write_outputs:
        paths = config["paths"]
        predictions_dir = repository_root / paths["predictions"]
        audits_dir = repository_root / paths["audits"]
        backtests_dir = repository_root / paths["backtests"]
        manifests_dir = repository_root / paths["manifests"]
        for directory in [
            predictions_dir,
            audits_dir,
            backtests_dir,
            manifests_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        reranked.to_csv(
            predictions_dir / "10b_started_nonzero_zigzag15_filtered_inference.csv",
            index=False,
            encoding="utf-8-sig",
        )
        signals.to_csv(
            predictions_dir / "10b_started_nonzero_zigzag15_selected_signals.csv",
            index=False,
            encoding="utf-8-sig",
        )
        started_audit.to_csv(
            audits_dir / "10b_started_source_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        date_audit.to_csv(
            audits_dir / "10b_signal_filter_date_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        capital_audit.to_csv(
            audits_dir / "10b_initial_capital_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        inventory.to_csv(
            audits_dir / "10b_raw_market_inventory_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        signal_plan_errors.to_csv(
            audits_dir / "10b_signal_execution_plan_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )
        market_errors.to_csv(
            audits_dir / "10b_market_history_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )
        integrity_audit.to_csv(
            audits_dir / "10b_portfolio_integrity_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        scenario_summary.to_csv(
            backtests_dir / "10b_scenario_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        trade_ledger.to_csv(
            backtests_dir / "10b_trade_ledger.csv",
            index=False,
            encoding="utf-8-sig",
        )
        signal_decisions.to_csv(
            backtests_dir / "10b_signal_decisions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        daily_equity.to_csv(
            backtests_dir / "10b_daily_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )
        signal_plans.to_csv(
            backtests_dir / "10b_signal_execution_plan.csv",
            index=False,
            encoding="utf-8-sig",
        )
        with (
            manifests_dir / "10b_exploratory_portfolio_backtest_manifest.json"
        ).open("w", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

    return outputs
