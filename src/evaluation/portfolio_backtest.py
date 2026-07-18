"""Stage 10 executable long-only portfolio backtest.

The module consumes the frozen Stage 09 selected-signal file, reconstructs
causal next-session executions from adjusted OHLC histories, and evaluates
pre-registered portfolio rules.

Key safeguards
--------------
- Long-only. A negative model outcome never creates a short position.
- Signals are already frozen before this stage. Outcome columns are ignored.
- Entry occurs at the next trading observation's adjusted open.
- Repeated signals may create independent lots in the same symbol.
- Each lot has its own stop, trailing state, and 30-observation horizon.
- A lot's planned open risk becomes zero only after a break-even-or-better
  trailing stop is executable. Protected lots still consume cash, exposure,
  symbol capacity, and lot capacity.
- Daily OHLC ambiguity is handled conservatively: adverse stop first.
- Transaction fees, sell tax, slippage, liquidity, cash, exposure, and
  concurrent-position constraints are applied before a new lot is accepted.

Adjusted prices imply synthetic share quantities. Portfolio returns are
internally consistent on the adjusted price scale, but liquidity fallback based
on adjusted price times raw volume is explicitly labelled as a proxy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd


STAGE10_SCHEMA_VERSION = "stage10_v1_preregistered_long_only_multilot_portfolio"


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def current_git_commit(repository_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return value or None
    except Exception:
        return None


def _project_date_parser(values: pd.Series) -> pd.Series:
    """Use the project parser when available, otherwise parse common formats."""
    try:
        from src.features.preprocessing import parse_market_date  # type: ignore

        parsed = parse_market_date(values)
        return pd.to_datetime(parsed, errors="coerce").dt.normalize()
    except Exception:
        pass

    as_text = values.astype(str).str.strip()
    numeric_yyyymmdd = as_text.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    if numeric_yyyymmdd.any():
        parsed.loc[numeric_yyyymmdd] = pd.to_datetime(
            as_text.loc[numeric_yyyymmdd],
            format="%Y%m%d",
            errors="coerce",
        )
    if (~numeric_yyyymmdd).any():
        parsed.loc[~numeric_yyyymmdd] = pd.to_datetime(
            as_text.loc[~numeric_yyyymmdd],
            errors="coerce",
        )
    return parsed.dt.normalize()


def _find_first_existing_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    exact = set(columns)
    for candidate in candidates:
        if candidate in exact:
            return candidate

    lower_map = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


# ---------------------------------------------------------------------------
# Capital calibration
# ---------------------------------------------------------------------------

def ten_percent_trimmed_mean(values: pd.Series, trim_fraction: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
    numeric = numeric[numeric > 0.0].sort_values().reset_index(drop=True)

    if numeric.empty:
        raise ValueError("No positive finite values are available for capital calibration.")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5).")

    trim_count = int(math.floor(len(numeric) * trim_fraction))
    if trim_count * 2 >= len(numeric):
        raise ValueError("The trim fraction removes the complete sample.")

    if trim_count == 0:
        return float(numeric.mean())
    return float(numeric.iloc[trim_count:-trim_count].mean())


def cost_adjusted_stop_loss_fraction(
    *,
    buy_fee_rate: float,
    sell_fee_rate: float,
    buy_slippage_fraction: float,
    sell_slippage_fraction: float,
    stop_loss_fraction: float,
) -> float:
    """Worst-case planned loss as a fraction of total entry cash outlay.

    The stop is anchored 15% below the buy execution price. The sale then
    incurs sell-side slippage and the complete sell deduction rate.
    """
    for name, value in {
        "buy_fee_rate": buy_fee_rate,
        "sell_fee_rate": sell_fee_rate,
        "buy_slippage_fraction": buy_slippage_fraction,
        "sell_slippage_fraction": sell_slippage_fraction,
        "stop_loss_fraction": stop_loss_fraction,
    }.items():
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")

    entry_cash_per_reference_share = (
        (1.0 + buy_slippage_fraction) * (1.0 + buy_fee_rate)
    )
    net_stop_proceeds_per_reference_share = (
        (1.0 + buy_slippage_fraction)
        * (1.0 - stop_loss_fraction)
        * (1.0 - sell_slippage_fraction)
        * (1.0 - sell_fee_rate)
    )
    loss_fraction = (
        entry_cash_per_reference_share - net_stop_proceeds_per_reference_share
    ) / entry_cash_per_reference_share

    if not 0.0 < loss_fraction < 1.0:
        raise AssertionError("The cost-adjusted stop-loss fraction is invalid.")
    return float(loss_fraction)


def calibrate_initial_capital(
    customer_frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    capital_cfg = config["initial_capital"]
    source_column = str(capital_cfg["source_column"])
    if source_column not in customer_frame.columns:
        raise KeyError(
            f"Capital source column '{source_column}' is missing. "
            f"Available columns: {customer_frame.columns.tolist()}"
        )

    expected_rows = int(capital_cfg["expected_customer_rows"])
    if len(customer_frame) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} customer rows, found {len(customer_frame)}."
        )

    values = pd.to_numeric(customer_frame[source_column], errors="coerce")
    valid = values[np.isfinite(values.to_numpy(dtype=float)) & values.gt(0.0)]
    if len(valid) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} positive finite '{source_column}' values, "
            f"found {len(valid)}."
        )

    fee_cfg = config["costs"]
    execution_cfg = config["execution"]
    exit_cfg = config["exit_policy"]
    risk_cfg = config["risk"]

    loss_fraction = cost_adjusted_stop_loss_fraction(
        buy_fee_rate=float(fee_cfg["buy_total_rate"]),
        sell_fee_rate=float(fee_cfg["sell_total_rate"]),
        buy_slippage_fraction=float(execution_cfg["primary_slippage_each_side"]),
        sell_slippage_fraction=float(execution_cfg["primary_slippage_each_side"]),
        stop_loss_fraction=float(exit_cfg["initial_stop_loss_fraction"]),
    )
    lot_weight = float(risk_cfg["risk_per_lot_fraction"]) / loss_fraction

    trim_fraction = float(capital_cfg["trim_fraction_each_tail"])
    trimmed_mean = ten_percent_trimmed_mean(valid, trim_fraction)
    median = float(valid.median())
    arithmetic_mean = float(valid.mean())

    primary = trimmed_mean / lot_weight
    median_proxy = median / lot_weight
    mean_proxy = arithmetic_mean / lot_weight

    expected_primary = float(capital_cfg["expected_primary_initial_capital_irr"])
    tolerance = float(capital_cfg.get("expected_value_relative_tolerance", 1.0e-10))
    if not math.isclose(primary, expected_primary, rel_tol=tolerance, abs_tol=1.0):
        raise AssertionError(
            "Recomputed primary capital differs from the pre-registered value: "
            f"computed={primary:.6f}, expected={expected_primary:.6f}"
        )

    summary: dict[str, float | int | str] = {
        "currency": str(capital_cfg["currency"]),
        "customer_rows": int(len(valid)),
        "source_column": source_column,
        "trim_fraction_each_tail": trim_fraction,
        "arithmetic_mean_avg_buy_irr": arithmetic_mean,
        "median_avg_buy_irr": median,
        "trimmed_mean_avg_buy_irr": trimmed_mean,
        "cost_adjusted_stop_loss_fraction": loss_fraction,
        "risk_per_lot_fraction": float(risk_cfg["risk_per_lot_fraction"]),
        "implied_initial_lot_weight": lot_weight,
        "primary_initial_capital_irr": primary,
        "median_sensitivity_initial_capital_irr": median_proxy,
        "arithmetic_mean_sensitivity_initial_capital_irr": mean_proxy,
        "interpretation": "customer_transaction_size_anchored_initial_capital_proxy",
    }

    audit = pd.DataFrame(
        [
            {
                "scenario": "primary_trimmed_mean_proxy",
                "avg_buy_anchor_irr": trimmed_mean,
                "initial_capital_irr": primary,
                "is_primary": True,
            },
            {
                "scenario": "median_proxy",
                "avg_buy_anchor_irr": median,
                "initial_capital_irr": median_proxy,
                "is_primary": False,
            },
            {
                "scenario": "arithmetic_mean_proxy",
                "avg_buy_anchor_irr": arithmetic_mean,
                "initial_capital_irr": mean_proxy,
                "is_primary": False,
            },
        ]
    )
    for key, value in summary.items():
        if key not in audit.columns:
            audit[key] = value
    return summary, audit


# ---------------------------------------------------------------------------
# Scenario and market-data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestScenario:
    scenario_id: str
    initial_capital_irr: float
    slippage_each_side: float
    position_structure: str
    exit_style: str
    is_primary: bool = False


@dataclass
class Lot:
    lot_id: str
    event_id: str
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    signal_position: int
    entry_position: int
    horizon_end_position: int
    score: float
    daily_rank: int
    quantity: int
    entry_reference_open: float
    entry_execution_price: float
    entry_gross_notional: float
    buy_fee_amount: float
    buy_slippage_amount: float
    entry_cash_cost: float
    initial_stop_quote: float
    activation_quote: float
    net_break_even_quote: float
    current_stop_quote: float
    highest_observed_high: float
    trailing_pending: bool
    trailing_active: bool
    trailing_activation_date: pd.Timestamp | None
    risk_release_date: pd.Timestamp | None
    last_mark_price: float
    entry_open_risk_amount: float
    current_open_risk_amount: float
    locked_profit_amount: float
    lot_number_for_symbol: int
    scenario_id: str


@dataclass
class MarketHistory:
    symbol: str
    frame: pd.DataFrame
    date_to_position: dict[pd.Timestamp, int]
    liquidity_source: str
    raw_path: Path


DIRECT_TRADED_VALUE_CANDIDATES = (
    "qTotCap",
    "traded_value",
    "trade_value",
    "transaction_value",
    "value",
    "Value",
    "total_value",
)
VOLUME_CANDIDATES = (
    "qTotTran5J",
    "volume",
    "Volume",
    "trade_volume",
    "total_volume",
    "vol",
)


def discover_customer_file(repository_root: Path, configured_path: str) -> Path:
    candidates = [
        repository_root / configured_path,
        repository_root / "final_feature_dim.csv",
        repository_root / "data_ready" / "final_feature_dim.csv",
        repository_root / "data_ready" / "portfolio" / "final_feature_dim.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Customer calibration file was not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def discover_raw_file_map(raw_data_dir: Path) -> dict[str, Path]:
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw-data directory does not exist: {raw_data_dir}")

    mapping: dict[str, Path] = {}
    for path in sorted(raw_data_dir.rglob("*.csv")):
        symbol = path.stem.strip()
        if symbol in mapping:
            raise ValueError(
                f"Duplicate raw-file stem '{symbol}': {mapping[symbol]} and {path}"
            )
        mapping[symbol] = path
    if not mapping:
        raise FileNotFoundError(f"No CSV files were found below {raw_data_dir}.")
    return mapping


def load_market_history(
    *,
    symbol: str,
    raw_path: Path,
    tail_end: pd.Timestamp,
    liquidity_cfg: dict[str, Any],
) -> MarketHistory:
    raw = pd.read_csv(raw_path, low_memory=False)

    required = {"dEven", "adj_open", "adj_high", "adj_low", "adj_last_price"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise KeyError(f"{symbol}: required raw columns are missing: {missing}")

    keep_columns = list(required)
    direct_value_column = _find_first_existing_column(
        raw.columns,
        liquidity_cfg.get(
            "direct_traded_value_column_candidates",
            DIRECT_TRADED_VALUE_CANDIDATES,
        ),
    )
    volume_column = _find_first_existing_column(
        raw.columns,
        liquidity_cfg.get("volume_column_candidates", VOLUME_CANDIDATES),
    )

    if direct_value_column is not None:
        keep_columns.append(direct_value_column)
    if volume_column is not None and volume_column not in keep_columns:
        keep_columns.append(volume_column)

    market = raw.loc[:, keep_columns].copy()
    market["dEven"] = _project_date_parser(market["dEven"])
    for column in ["adj_open", "adj_high", "adj_low", "adj_last_price"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")

    market = market.loc[
        market["dEven"].notna() & market["dEven"].le(tail_end)
    ].copy()
    market = (
        market.sort_values("dEven", kind="stable")
        .drop_duplicates("dEven", keep="last")
        .reset_index(drop=True)
    )

    if market.empty:
        raise ValueError(f"{symbol}: raw history is empty after date filtering.")

    core = market[["adj_open", "adj_high", "adj_low", "adj_last_price"]]
    if not np.isfinite(core.to_numpy(dtype=float)).all():
        bad = int((~np.isfinite(core.to_numpy(dtype=float))).sum())
        raise ValueError(f"{symbol}: adjusted OHLC contains {bad} nonfinite values.")
    if core.le(0.0).any().any():
        raise ValueError(f"{symbol}: adjusted OHLC contains nonpositive values.")

    market["raw_volume"] = np.nan
    market["traded_value_irr"] = np.nan
    liquidity_source = "unavailable"

    if direct_value_column is not None:
        direct_values = pd.to_numeric(market[direct_value_column], errors="coerce")
        if direct_values.notna().any() and direct_values.gt(0.0).any():
            market["traded_value_irr"] = direct_values
            liquidity_source = f"direct:{direct_value_column}"

    if (
        not market["traded_value_irr"].gt(0.0).any()
        and volume_column is not None
    ):
        volumes = pd.to_numeric(market[volume_column], errors="coerce")
        market["raw_volume"] = volumes
        proxy = volumes * market["adj_last_price"]
        if proxy.notna().any() and proxy.gt(0.0).any():
            market["traded_value_irr"] = proxy
            liquidity_source = f"proxy:adj_last_price_x_{volume_column}"

    if liquidity_source == "unavailable" and bool(liquidity_cfg["required"]):
        raise KeyError(
            f"{symbol}: no usable traded-value or volume source was found. "
            f"Columns: {raw.columns.tolist()}"
        )

    adv_window = int(liquidity_cfg["adv_window_observations"])
    minimum_history = int(liquidity_cfg["minimum_adv_history_observations"])
    market["adv20_irr"] = (
        market["traded_value_irr"]
        .shift(1)
        .rolling(adv_window, min_periods=minimum_history)
        .mean()
    )

    date_to_position = {
        pd.Timestamp(date): int(position)
        for position, date in enumerate(market["dEven"])
    }
    return MarketHistory(
        symbol=symbol,
        frame=market,
        date_to_position=date_to_position,
        liquidity_source=liquidity_source,
        raw_path=raw_path,
    )


def prepare_selected_signals(
    selected_signal_path: Path,
    *,
    expected_rows: int,
    signal_generation_end: pd.Timestamp,
) -> pd.DataFrame:
    source = pd.read_csv(selected_signal_path, low_memory=False)

    allowed = [
        "event_id",
        "symbol",
        "dEven",
        "xgboost_ranking_score",
        "daily_rank",
        "daily_signal_quota",
        "selected_signal",
    ]
    missing = sorted(set(allowed) - set(source.columns))
    if missing:
        raise KeyError(f"Selected-signal columns are missing: {missing}")

    signals = source.loc[:, allowed].copy()
    signals["dEven"] = pd.to_datetime(signals["dEven"], errors="raise").dt.normalize()
    signals["xgboost_ranking_score"] = pd.to_numeric(
        signals["xgboost_ranking_score"],
        errors="raise",
    )
    signals["daily_rank"] = pd.to_numeric(
        signals["daily_rank"],
        errors="raise",
    ).astype(int)
    signals["selected_signal"] = signals["selected_signal"].astype(bool)

    if not signals["selected_signal"].all():
        raise AssertionError("The selected-signal file contains unselected rows.")
    if len(signals) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} frozen signals, found {len(signals)}."
        )
    if signals["event_id"].duplicated().any():
        raise AssertionError("Duplicate selected-signal event IDs exist.")
    if signals["dEven"].gt(signal_generation_end).any():
        raise AssertionError("A selected signal occurs after the frozen signal end.")

    signals = signals.sort_values(
        ["dEven", "xgboost_ranking_score", "symbol", "event_id"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    expected_rank = (
        signals.groupby("dEven", sort=False).cumcount() + 1
    ).astype(int)
    if not expected_rank.eq(signals["daily_rank"]).all():
        raise AssertionError(
            "Frozen daily rank does not match score-desc/symbol/event ordering."
        )

    return signals


def attach_execution_plan(
    signals: pd.DataFrame,
    market_histories: dict[str, MarketHistory],
    *,
    horizon_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plans: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for signal in signals.itertuples(index=False):
        symbol = str(signal.symbol)
        event_date = pd.Timestamp(signal.dEven).normalize()
        history = market_histories.get(symbol)

        try:
            if history is None:
                raise FileNotFoundError(f"No market history is loaded for {symbol}.")
            if event_date not in history.date_to_position:
                raise KeyError(f"Signal date {event_date.date()} is absent for {symbol}.")

            signal_position = history.date_to_position[event_date]
            entry_position = signal_position + 1
            horizon_end_position = signal_position + horizon_observations
            if horizon_end_position >= len(history.frame):
                available = len(history.frame) - signal_position - 1
                raise ValueError(
                    f"Only {available} future observations remain; "
                    f"{horizon_observations} are required."
                )

            entry_row = history.frame.iloc[entry_position]
            plans.append(
                {
                    "event_id": str(signal.event_id),
                    "symbol": symbol,
                    "signal_date": event_date,
                    "entry_date": pd.Timestamp(entry_row["dEven"]),
                    "signal_position": signal_position,
                    "entry_position": entry_position,
                    "horizon_end_position": horizon_end_position,
                    "horizon_end_date": pd.Timestamp(
                        history.frame.iloc[horizon_end_position]["dEven"]
                    ),
                    "xgboost_ranking_score": float(signal.xgboost_ranking_score),
                    "daily_rank": int(signal.daily_rank),
                    "daily_signal_quota": int(signal.daily_signal_quota),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "event_id": str(signal.event_id),
                    "symbol": symbol,
                    "signal_date": event_date,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    plan_frame = pd.DataFrame(plans)
    error_frame = pd.DataFrame(errors)
    if not plan_frame.empty:
        plan_frame = plan_frame.sort_values(
            [
                "entry_date",
                "xgboost_ranking_score",
                "symbol",
                "event_id",
            ],
            ascending=[True, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
    return plan_frame, error_frame


def build_scenarios(
    config: dict[str, Any],
    capital_summary: dict[str, Any],
) -> list[BacktestScenario]:
    scenario_cfg = config["scenario_grid"]

    capital_values = {
        "primary": float(capital_summary["primary_initial_capital_irr"]),
        "median": float(capital_summary["median_sensitivity_initial_capital_irr"]),
        "mean": float(capital_summary["arithmetic_mean_sensitivity_initial_capital_irr"]),
    }
    slippages = {
        "0bp": 0.0,
        "10bp": 0.001,
        "20bp": 0.002,
        "50bp": 0.005,
    }

    scenarios: list[BacktestScenario] = []
    for capital_key in scenario_cfg["capital_scenarios"]:
        if capital_key not in capital_values:
            raise KeyError(f"Unknown capital scenario: {capital_key}")
        for slippage_key in scenario_cfg["slippage_scenarios"]:
            if slippage_key not in slippages:
                raise KeyError(f"Unknown slippage scenario: {slippage_key}")
            for structure in scenario_cfg["position_structures"]:
                if structure not in {"multi_lot", "single_lot"}:
                    raise ValueError(f"Unknown position structure: {structure}")
                for exit_style in scenario_cfg["exit_styles"]:
                    if exit_style not in {"trailing", "fixed_take_profit"}:
                        raise ValueError(f"Unknown exit style: {exit_style}")

                    scenario_id = (
                        f"capital_{capital_key}__slip_{slippage_key}__"
                        f"{structure}__{exit_style}"
                    )
                    primary = (
                        capital_key == "primary"
                        and slippage_key == "20bp"
                        and structure == "multi_lot"
                        and exit_style == "trailing"
                    )
                    scenarios.append(
                        BacktestScenario(
                            scenario_id=scenario_id,
                            initial_capital_irr=capital_values[capital_key],
                            slippage_each_side=slippages[slippage_key],
                            position_structure=structure,
                            exit_style=exit_style,
                            is_primary=primary,
                        )
                    )

    primary_count = sum(scenario.is_primary for scenario in scenarios)
    if primary_count != 1:
        raise AssertionError(f"Expected one primary scenario, found {primary_count}.")
    return scenarios


# ---------------------------------------------------------------------------
# Portfolio simulator
# ---------------------------------------------------------------------------

def _entry_cash_per_share(
    reference_open: float,
    *,
    slippage: float,
    buy_fee_rate: float,
) -> tuple[float, float, float]:
    execution = reference_open * (1.0 + slippage)
    gross = execution
    cash = execution * (1.0 + buy_fee_rate)
    return execution, gross, cash


def _net_sale_per_share(
    reference_quote: float,
    *,
    slippage: float,
    sell_fee_rate: float,
) -> tuple[float, float]:
    execution = reference_quote * (1.0 - slippage)
    net = execution * (1.0 - sell_fee_rate)
    return execution, net


def _lot_open_risk(
    lot: Lot,
    *,
    sell_slippage: float,
    sell_fee_rate: float,
) -> tuple[float, float]:
    _, net_per_share = _net_sale_per_share(
        lot.current_stop_quote,
        slippage=sell_slippage,
        sell_fee_rate=sell_fee_rate,
    )
    net_stop = lot.quantity * net_per_share
    risk = max(0.0, lot.entry_cash_cost - net_stop)
    locked_profit = max(0.0, net_stop - lot.entry_cash_cost)
    return float(risk), float(locked_profit)


def _portfolio_state(
    *,
    cash: float,
    lots: list[Lot],
    sell_slippage: float,
    sell_fee_rate: float,
) -> dict[str, float | int]:
    market_value = float(sum(lot.quantity * lot.last_mark_price for lot in lots))
    equity = float(cash + market_value)
    planned_risk = 0.0
    locked_profit = 0.0
    for lot in lots:
        risk, locked = _lot_open_risk(
            lot,
            sell_slippage=sell_slippage,
            sell_fee_rate=sell_fee_rate,
        )
        lot.current_open_risk_amount = risk
        lot.locked_profit_amount = locked
        planned_risk += risk
        locked_profit += locked

    return {
        "cash_irr": float(cash),
        "market_value_irr": market_value,
        "equity_irr": equity,
        "gross_exposure_fraction": safe_ratio(market_value, equity),
        "planned_open_risk_irr": planned_risk,
        "planned_open_risk_fraction": safe_ratio(planned_risk, equity),
        "locked_profit_at_stops_irr": locked_profit,
        "open_lots": int(len(lots)),
        "open_symbols": int(len({lot.symbol for lot in lots})),
        "protected_lots": int(sum(lot.trailing_active for lot in lots)),
        "unprotected_lots": int(sum(not lot.trailing_active for lot in lots)),
    }


def _exit_lot(
    *,
    lot: Lot,
    date: pd.Timestamp,
    reference_quote: float,
    reason: str,
    observation: int,
    cash: float,
    scenario: BacktestScenario,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    sell_rate = float(config["costs"]["sell_total_rate"])
    tax_rate = float(config["costs"]["sell_tax_rate_included"])
    execution, net_per_share = _net_sale_per_share(
        reference_quote,
        slippage=scenario.slippage_each_side,
        sell_fee_rate=sell_rate,
    )
    gross_proceeds = lot.quantity * execution
    sell_deduction = gross_proceeds * sell_rate
    sell_tax = gross_proceeds * tax_rate
    sell_non_tax_fee = sell_deduction - sell_tax
    net_proceeds = lot.quantity * net_per_share
    sell_slippage_amount = lot.quantity * reference_quote * scenario.slippage_each_side

    pnl = net_proceeds - lot.entry_cash_cost
    net_return = safe_ratio(pnl, lot.entry_cash_cost)

    trade = {
        "scenario_id": scenario.scenario_id,
        "is_primary_scenario": scenario.is_primary,
        "lot_id": lot.lot_id,
        "event_id": lot.event_id,
        "symbol": lot.symbol,
        "lot_number_for_symbol": lot.lot_number_for_symbol,
        "signal_date": lot.signal_date,
        "entry_date": lot.entry_date,
        "exit_date": date,
        "entry_observation": 1,
        "exit_observation": int(observation),
        "calendar_holding_days": int((date - lot.entry_date).days),
        "quantity": int(lot.quantity),
        "score": lot.score,
        "daily_rank": lot.daily_rank,
        "entry_reference_open": lot.entry_reference_open,
        "entry_execution_price": lot.entry_execution_price,
        "entry_cash_cost_irr": lot.entry_cash_cost,
        "entry_gross_notional_irr": lot.entry_gross_notional,
        "buy_fee_irr": lot.buy_fee_amount,
        "buy_slippage_irr": lot.buy_slippage_amount,
        "exit_reference_quote": reference_quote,
        "exit_execution_price": execution,
        "gross_sale_proceeds_irr": gross_proceeds,
        "sell_tax_irr": sell_tax,
        "sell_non_tax_fee_irr": sell_non_tax_fee,
        "sell_total_deduction_irr": sell_deduction,
        "sell_slippage_irr": sell_slippage_amount,
        "net_sale_proceeds_irr": net_proceeds,
        "net_pnl_irr": pnl,
        "net_return": net_return,
        "exit_reason": reason,
        "trailing_ever_activated": bool(
            lot.trailing_active or lot.trailing_activation_date is not None
        ),
        "trailing_activation_date": lot.trailing_activation_date,
        "risk_release_date": lot.risk_release_date,
        "entry_open_risk_irr": lot.entry_open_risk_amount,
        "locked_profit_before_exit_irr": lot.locked_profit_amount,
        "initial_stop_quote": lot.initial_stop_quote,
        "final_protective_stop_quote": lot.current_stop_quote,
        "highest_observed_high": lot.highest_observed_high,
        "net_break_even_quote": lot.net_break_even_quote,
        "position_structure": scenario.position_structure,
        "exit_style": scenario.exit_style,
        "slippage_each_side": scenario.slippage_each_side,
        "initial_capital_irr": scenario.initial_capital_irr,
    }
    return float(cash + net_proceeds), trade


def simulate_scenario(
    *,
    scenario: BacktestScenario,
    signal_plans: pd.DataFrame,
    market_histories: dict[str, MarketHistory],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    costs = config["costs"]
    risk_cfg = config["risk"]
    exposure_cfg = config["exposure"]
    capacity_cfg = config["capacity"]
    liquidity_cfg = config["liquidity"]
    exit_cfg = config["exit_policy"]

    buy_fee = float(costs["buy_total_rate"])
    sell_fee = float(costs["sell_total_rate"])

    risk_per_lot = float(risk_cfg["risk_per_lot_fraction"])
    max_symbol_risk = float(risk_cfg["maximum_symbol_open_risk_fraction"])
    max_portfolio_risk = float(risk_cfg["maximum_portfolio_open_risk_fraction"])

    max_symbol_exposure = float(exposure_cfg["maximum_symbol_exposure_fraction"])
    max_gross_exposure = float(exposure_cfg["maximum_gross_exposure_fraction"])

    max_distinct_symbols = int(capacity_cfg["maximum_distinct_symbols"])
    max_open_lots = int(capacity_cfg["maximum_open_lots"])
    configured_lots_per_symbol = int(capacity_cfg["maximum_open_lots_per_symbol"])
    max_lots_per_symbol = (
        1 if scenario.position_structure == "single_lot"
        else configured_lots_per_symbol
    )
    max_new_lots_per_day = int(capacity_cfg["maximum_new_lots_per_day"])

    stop_fraction = float(exit_cfg["initial_stop_loss_fraction"])
    activation_fraction = float(exit_cfg["trailing_activation_fraction"])
    trailing_distance = float(exit_cfg["trailing_distance_fraction"])
    target_fraction = float(exit_cfg["fixed_take_profit_fraction"])
    horizon = int(exit_cfg["maximum_horizon_observations"])

    cash = float(scenario.initial_capital_irr)
    lots: list[Lot] = []
    trade_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    entry_groups = {
        pd.Timestamp(date): group.copy()
        for date, group in signal_plans.groupby("entry_date", sort=True)
    }

    all_dates = sorted(
        {
            pd.Timestamp(date)
            for history in market_histories.values()
            for date in history.frame["dEven"].tolist()
            if signal_plans["entry_date"].min()
            <= pd.Timestamp(date)
            <= signal_plans["horizon_end_date"].max()
        }
    )
    if not all_dates:
        raise ValueError("The global execution calendar is empty.")

    lot_counter = 0
    entry_constraint_violations = 0
    negative_cash_events = 0
    protected_risk_nonzero_events = 0

    def mark_to_date(date: pd.Timestamp, use_open: bool) -> None:
        for lot in lots:
            history = market_histories[lot.symbol]
            position = history.date_to_position.get(date)
            if position is None:
                continue
            row = history.frame.iloc[position]
            lot.last_mark_price = float(
                row["adj_open"] if use_open else row["adj_last_price"]
            )

    for date in all_dates:
        date = pd.Timestamp(date)
        new_lots_today = 0

        # 1) Opening marks and executable gap exits.
        mark_to_date(date, use_open=True)
        opening_exit_ids: set[str] = set()

        for lot in list(lots):
            history = market_histories[lot.symbol]
            position = history.date_to_position.get(date)
            if position is None:
                continue
            if position < lot.entry_position:
                continue

            row = history.frame.iloc[position]
            open_quote = float(row["adj_open"])
            observation = position - lot.signal_position

            # A trailing stop becomes executable at this symbol's next trading open.
            if lot.trailing_pending:
                proposed = max(
                    lot.net_break_even_quote,
                    (1.0 - trailing_distance) * lot.highest_observed_high,
                )
                lot.current_stop_quote = proposed
                lot.trailing_pending = False

                if open_quote <= proposed:
                    cash, trade = _exit_lot(
                        lot=lot,
                        date=date,
                        reference_quote=open_quote,
                        reason="trailing_gap_exit_on_activation_open",
                        observation=observation,
                        cash=cash,
                        scenario=scenario,
                        config=config,
                    )
                    trade_rows.append(trade)
                    opening_exit_ids.add(lot.lot_id)
                    lots.remove(lot)
                    continue

                lot.trailing_active = True
                lot.trailing_activation_date = date
                lot.risk_release_date = date

            # Existing hard/trailing stop gap.
            if open_quote <= lot.current_stop_quote:
                reason = (
                    "trailing_gap_stop"
                    if lot.trailing_active
                    else "initial_gap_stop"
                )
                cash, trade = _exit_lot(
                    lot=lot,
                    date=date,
                    reference_quote=open_quote,
                    reason=reason,
                    observation=observation,
                    cash=cash,
                    scenario=scenario,
                    config=config,
                )
                trade_rows.append(trade)
                opening_exit_ids.add(lot.lot_id)
                lots.remove(lot)
                continue

            # Fixed take-profit can gap above the target.
            if (
                scenario.exit_style == "fixed_take_profit"
                and open_quote >= lot.activation_quote
            ):
                cash, trade = _exit_lot(
                    lot=lot,
                    date=date,
                    reference_quote=open_quote,
                    reason="fixed_target_gap_exit",
                    observation=observation,
                    cash=cash,
                    scenario=scenario,
                    config=config,
                )
                trade_rows.append(trade)
                opening_exit_ids.add(lot.lot_id)
                lots.remove(lot)

        # Opening survivors with break-even-or-better trailing stops must carry zero
        # planned open risk. This capacity can be reallocated at the same open.
        opening_state = _portfolio_state(
            cash=cash,
            lots=lots,
            sell_slippage=scenario.slippage_each_side,
            sell_fee_rate=sell_fee,
        )
        for lot in lots:
            if lot.trailing_active and lot.current_open_risk_amount > 1.0e-6:
                protected_risk_nonzero_events += 1

        # 2) Process today's frozen signals in deterministic score order.
        todays_signals = entry_groups.get(date)
        if todays_signals is not None:
            todays_signals = todays_signals.sort_values(
                ["xgboost_ranking_score", "symbol", "event_id"],
                ascending=[False, True, True],
                kind="stable",
            )

            for signal in todays_signals.itertuples(index=False):
                base_decision = {
                    "scenario_id": scenario.scenario_id,
                    "is_primary_scenario": scenario.is_primary,
                    "event_id": signal.event_id,
                    "symbol": signal.symbol,
                    "signal_date": signal.signal_date,
                    "scheduled_entry_date": signal.entry_date,
                    "score": signal.xgboost_ranking_score,
                    "daily_rank": signal.daily_rank,
                    "position_structure": scenario.position_structure,
                    "exit_style": scenario.exit_style,
                    "decision": "rejected",
                    "rejection_reason": None,
                    "accepted_quantity": 0,
                    "actual_entry_risk_irr": 0.0,
                }

                symbol_lots = [lot for lot in lots if lot.symbol == signal.symbol]
                open_symbols = {lot.symbol for lot in lots}

                rejection: str | None = None
                if new_lots_today >= max_new_lots_per_day:
                    rejection = "daily_entry_cap_reached"
                elif len(lots) >= max_open_lots:
                    rejection = "portfolio_lot_cap_reached"
                elif len(symbol_lots) >= max_lots_per_symbol:
                    rejection = (
                        "single_lot_symbol_already_open"
                        if scenario.position_structure == "single_lot"
                        else "symbol_lot_cap_reached"
                    )
                elif (
                    signal.symbol not in open_symbols
                    and len(open_symbols) >= max_distinct_symbols
                ):
                    rejection = "distinct_symbol_cap_reached"

                history = market_histories[str(signal.symbol)]
                position = history.date_to_position.get(date)
                if rejection is None and position is None:
                    rejection = "missing_entry_bar"

                if rejection is None:
                    row = history.frame.iloc[position]
                    reference_open = float(row["adj_open"])
                    if not np.isfinite(reference_open) or reference_open <= 0.0:
                        rejection = "invalid_next_open"

                if rejection is None:
                    state = _portfolio_state(
                        cash=cash,
                        lots=lots,
                        sell_slippage=scenario.slippage_each_side,
                        sell_fee_rate=sell_fee,
                    )
                    equity = float(state["equity_irr"])
                    if equity <= 0.0:
                        rejection = "nonpositive_equity"

                if rejection is None:
                    (
                        entry_execution,
                        _gross_per_share,
                        cash_per_share,
                    ) = _entry_cash_per_share(
                        reference_open,
                        slippage=scenario.slippage_each_side,
                        buy_fee_rate=buy_fee,
                    )
                    initial_stop = entry_execution * (1.0 - stop_fraction)
                    activation_quote = entry_execution * (
                        1.0
                        + (
                            target_fraction
                            if scenario.exit_style == "fixed_take_profit"
                            else activation_fraction
                        )
                    )
                    _, net_stop_per_share = _net_sale_per_share(
                        initial_stop,
                        slippage=scenario.slippage_each_side,
                        sell_fee_rate=sell_fee,
                    )
                    risk_per_share = cash_per_share - net_stop_per_share
                    if risk_per_share <= 0.0:
                        rejection = "invalid_risk_per_share"

                if rejection is None:
                    current_portfolio_risk = float(state["planned_open_risk_irr"])
                    current_symbol_risk = float(
                        sum(lot.current_open_risk_amount for lot in symbol_lots)
                    )
                    target_risk = risk_per_lot * equity
                    portfolio_risk_headroom = (
                        max_portfolio_risk * equity - current_portfolio_risk
                    )
                    symbol_risk_headroom = (
                        max_symbol_risk * equity - current_symbol_risk
                    )
                    allowed_risk = min(
                        target_risk,
                        portfolio_risk_headroom,
                        symbol_risk_headroom,
                    )
                    if allowed_risk <= 0.0:
                        rejection = (
                            "symbol_risk_cap_reached"
                            if symbol_risk_headroom <= portfolio_risk_headroom
                            else "portfolio_risk_cap_reached"
                        )

                if rejection is None:
                    market_value = float(state["market_value_irr"])
                    symbol_market_value = float(
                        sum(lot.quantity * lot.last_mark_price for lot in symbol_lots)
                    )
                    gross_headroom = max_gross_exposure * equity - market_value
                    symbol_exposure_headroom = (
                        max_symbol_exposure * equity - symbol_market_value
                    )
                    if gross_headroom <= 0.0:
                        rejection = "gross_exposure_cap_reached"
                    elif symbol_exposure_headroom <= 0.0:
                        rejection = "symbol_exposure_cap_reached"

                if rejection is None:
                    adv20 = float(row["adv20_irr"]) if pd.notna(row["adv20_irr"]) else float("nan")
                    if not np.isfinite(adv20) or adv20 <= 0.0:
                        rejection = "adv20_unavailable"
                    else:
                        liquidity_headroom = (
                            float(liquidity_cfg["maximum_fraction_of_adv20"])
                            * adv20
                        )
                        if liquidity_headroom <= 0.0:
                            rejection = "liquidity_cap_reached"

                if rejection is None:
                    q_risk = int(math.floor(allowed_risk / risk_per_share))
                    q_gross = int(math.floor(gross_headroom / entry_execution))
                    q_symbol = int(
                        math.floor(symbol_exposure_headroom / entry_execution)
                    )
                    q_cash = int(math.floor(cash / cash_per_share))
                    q_liquidity = int(
                        math.floor(liquidity_headroom / entry_execution)
                    )
                    quantity = min(
                        q_risk,
                        q_gross,
                        q_symbol,
                        q_cash,
                        q_liquidity,
                    )

                    if quantity < 1:
                        limiting = {
                            "risk_size_below_one_share": q_risk,
                            "gross_exposure_cap_reached": q_gross,
                            "symbol_exposure_cap_reached": q_symbol,
                            "insufficient_cash": q_cash,
                            "liquidity_cap_reached": q_liquidity,
                        }
                        rejection = min(limiting, key=limiting.get)

                if rejection is not None:
                    base_decision["rejection_reason"] = rejection
                    decision_rows.append(base_decision)
                    continue

                # Execute accepted long lot.
                lot_counter += 1
                entry_gross = quantity * entry_execution
                buy_fee_amount = entry_gross * buy_fee
                entry_cash_cost = entry_gross + buy_fee_amount
                buy_slippage_amount = (
                    quantity * reference_open * scenario.slippage_each_side
                )
                actual_risk = quantity * risk_per_share

                net_break_even_quote = (
                    entry_cash_cost
                    / quantity
                    / (
                        (1.0 - scenario.slippage_each_side)
                        * (1.0 - sell_fee)
                    )
                )

                lot_number_for_symbol = (
                    max(
                        [
                            lot.lot_number_for_symbol
                            for lot in lots
                            if lot.symbol == signal.symbol
                        ],
                        default=0,
                    )
                    + 1
                )

                lot = Lot(
                    lot_id=f"{scenario.scenario_id}::lot_{lot_counter:06d}",
                    event_id=str(signal.event_id),
                    symbol=str(signal.symbol),
                    signal_date=pd.Timestamp(signal.signal_date),
                    entry_date=date,
                    signal_position=int(signal.signal_position),
                    entry_position=int(signal.entry_position),
                    horizon_end_position=int(signal.horizon_end_position),
                    score=float(signal.xgboost_ranking_score),
                    daily_rank=int(signal.daily_rank),
                    quantity=quantity,
                    entry_reference_open=reference_open,
                    entry_execution_price=entry_execution,
                    entry_gross_notional=entry_gross,
                    buy_fee_amount=buy_fee_amount,
                    buy_slippage_amount=buy_slippage_amount,
                    entry_cash_cost=entry_cash_cost,
                    initial_stop_quote=initial_stop,
                    activation_quote=activation_quote,
                    net_break_even_quote=net_break_even_quote,
                    current_stop_quote=initial_stop,
                    highest_observed_high=entry_execution,
                    trailing_pending=False,
                    trailing_active=False,
                    trailing_activation_date=None,
                    risk_release_date=None,
                    last_mark_price=reference_open,
                    entry_open_risk_amount=actual_risk,
                    current_open_risk_amount=actual_risk,
                    locked_profit_amount=0.0,
                    lot_number_for_symbol=lot_number_for_symbol,
                    scenario_id=scenario.scenario_id,
                )
                cash -= entry_cash_cost
                if cash < -1.0e-6:
                    negative_cash_events += 1
                lots.append(lot)
                new_lots_today += 1

                # Verify all entry-time constraints after the accepted fill.
                post_state = _portfolio_state(
                    cash=cash,
                    lots=lots,
                    sell_slippage=scenario.slippage_each_side,
                    sell_fee_rate=sell_fee,
                )
                symbol_risk_after = sum(
                    candidate.current_open_risk_amount
                    for candidate in lots
                    if candidate.symbol == signal.symbol
                )
                symbol_value_after = sum(
                    candidate.quantity * candidate.last_mark_price
                    for candidate in lots
                    if candidate.symbol == signal.symbol
                )
                entry_checks = {
                    "cash_nonnegative": cash >= -1.0e-6,
                    "lot_count": len(lots) <= max_open_lots,
                    "symbol_lot_count": len(
                        [candidate for candidate in lots if candidate.symbol == signal.symbol]
                    )
                    <= max_lots_per_symbol,
                    "distinct_symbols": len({candidate.symbol for candidate in lots})
                    <= max_distinct_symbols,
                    "portfolio_risk": float(post_state["planned_open_risk_irr"])
                    <= max_portfolio_risk * equity + 1.0,
                    "symbol_risk": symbol_risk_after
                    <= max_symbol_risk * equity + 1.0,
                    "gross_exposure": float(post_state["market_value_irr"])
                    <= max_gross_exposure * equity + entry_execution,
                    "symbol_exposure": symbol_value_after
                    <= max_symbol_exposure * equity + entry_execution,
                    "liquidity": entry_gross <= liquidity_headroom + entry_execution,
                }
                if not all(entry_checks.values()):
                    entry_constraint_violations += 1

                base_decision.update(
                    {
                        "decision": "accepted",
                        "rejection_reason": None,
                        "accepted_quantity": quantity,
                        "actual_entry_risk_irr": actual_risk,
                        "entry_cash_cost_irr": entry_cash_cost,
                        "entry_execution_price": entry_execution,
                        "adv20_irr": adv20,
                        "liquidity_cap_irr": liquidity_headroom,
                        "portfolio_equity_before_entry_irr": equity,
                        "portfolio_open_risk_before_entry_irr": current_portfolio_risk,
                        "symbol_open_risk_before_entry_irr": current_symbol_risk,
                        "gross_exposure_before_entry_irr": market_value,
                        "symbol_exposure_before_entry_irr": symbol_market_value,
                        "lot_id": lot.lot_id,
                    }
                )
                decision_rows.append(base_decision)

        # 3) Intraday stop/target/activation logic and time exits.
        for lot in list(lots):
            history = market_histories[lot.symbol]
            position = history.date_to_position.get(date)
            if position is None or position < lot.entry_position:
                continue

            row = history.frame.iloc[position]
            high = float(row["adj_high"])
            low = float(row["adj_low"])
            last = float(row["adj_last_price"])
            observation = position - lot.signal_position

            # Conservative same-bar ordering: the adverse stop is evaluated first.
            if low <= lot.current_stop_quote:
                reason = (
                    "trailing_intraday_stop"
                    if lot.trailing_active
                    else "initial_intraday_stop"
                )
                cash, trade = _exit_lot(
                    lot=lot,
                    date=date,
                    reference_quote=lot.current_stop_quote,
                    reason=reason,
                    observation=observation,
                    cash=cash,
                    scenario=scenario,
                    config=config,
                )
                trade_rows.append(trade)
                lots.remove(lot)
                continue

            if scenario.exit_style == "fixed_take_profit":
                if high >= lot.activation_quote:
                    cash, trade = _exit_lot(
                        lot=lot,
                        date=date,
                        reference_quote=lot.activation_quote,
                        reason="fixed_take_profit",
                        observation=observation,
                        cash=cash,
                        scenario=scenario,
                        config=config,
                    )
                    trade_rows.append(trade)
                    lots.remove(lot)
                    continue
            else:
                if lot.trailing_active:
                    lot.highest_observed_high = max(lot.highest_observed_high, high)
                    lot.current_stop_quote = max(
                        lot.net_break_even_quote,
                        (1.0 - trailing_distance) * lot.highest_observed_high,
                    )
                elif not lot.trailing_pending and high >= lot.activation_quote:
                    lot.highest_observed_high = max(lot.highest_observed_high, high)
                    lot.trailing_pending = True
                else:
                    lot.highest_observed_high = max(lot.highest_observed_high, high)

            # The 30-observation time exit is evaluated at the adjusted last price
            # after intraday stops and targets.
            if observation >= horizon:
                cash, trade = _exit_lot(
                    lot=lot,
                    date=date,
                    reference_quote=last,
                    reason="time_exit_observation_30",
                    observation=observation,
                    cash=cash,
                    scenario=scenario,
                    config=config,
                )
                trade_rows.append(trade)
                lots.remove(lot)
                continue

            lot.last_mark_price = last

        # 4) End-of-day mark and audit state.
        mark_to_date(date, use_open=False)
        state = _portfolio_state(
            cash=cash,
            lots=lots,
            sell_slippage=scenario.slippage_each_side,
            sell_fee_rate=sell_fee,
        )

        symbol_values = {}
        symbol_risks = {}
        for lot in lots:
            symbol_values[lot.symbol] = (
                symbol_values.get(lot.symbol, 0.0)
                + lot.quantity * lot.last_mark_price
            )
            symbol_risks[lot.symbol] = (
                symbol_risks.get(lot.symbol, 0.0)
                + lot.current_open_risk_amount
            )

        daily_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "is_primary_scenario": scenario.is_primary,
                "date": date,
                **state,
                "maximum_symbol_exposure_fraction": (
                    max(symbol_values.values(), default=0.0)
                    / float(state["equity_irr"])
                    if float(state["equity_irr"]) > 0.0
                    else float("nan")
                ),
                "maximum_symbol_open_risk_fraction": (
                    max(symbol_risks.values(), default=0.0)
                    / float(state["equity_irr"])
                    if float(state["equity_irr"]) > 0.0
                    else float("nan")
                ),
                "new_lots_opened": new_lots_today,
                "position_structure": scenario.position_structure,
                "exit_style": scenario.exit_style,
                "slippage_each_side": scenario.slippage_each_side,
                "initial_capital_irr": scenario.initial_capital_irr,
            }
        )

    if lots:
        raise AssertionError(
            f"{scenario.scenario_id}: {len(lots)} lots remain open after the "
            "complete pre-registered outcome tail."
        )

    trades = pd.DataFrame(trade_rows)
    decisions = pd.DataFrame(decision_rows)
    daily = pd.DataFrame(daily_rows)

    integrity = {
        "scenario_id": scenario.scenario_id,
        "entry_constraint_violations": int(entry_constraint_violations),
        "negative_cash_events": int(negative_cash_events),
        "protected_lot_nonzero_planned_risk_events": int(
            protected_risk_nonzero_events
        ),
        "open_lots_after_tail": 0,
        "signals_decided": int(len(decisions)),
        "accepted_signals": int(
            decisions["decision"].eq("accepted").sum()
            if not decisions.empty
            else 0
        ),
        "rejected_signals": int(
            decisions["decision"].eq("rejected").sum()
            if not decisions.empty
            else 0
        ),
        "closed_trades": int(len(trades)),
        "all_positions_long_only": True,
    }

    if integrity["entry_constraint_violations"] != 0:
        raise AssertionError(
            f"{scenario.scenario_id}: entry constraint violations exist."
        )
    if integrity["negative_cash_events"] != 0:
        raise AssertionError(f"{scenario.scenario_id}: negative cash occurred.")
    if integrity["protected_lot_nonzero_planned_risk_events"] != 0:
        raise AssertionError(
            f"{scenario.scenario_id}: a protected lot retained planned risk."
        )
    if integrity["accepted_signals"] != integrity["closed_trades"]:
        raise AssertionError(
            f"{scenario.scenario_id}: accepted signal and closed-trade counts differ."
        )

    return trades, decisions, daily, integrity


# ---------------------------------------------------------------------------
# Metrics and orchestration
# ---------------------------------------------------------------------------

def _maximum_drawdown(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce")
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    return float(drawdown.min())


def summarize_scenario(
    scenario: BacktestScenario,
    trades: pd.DataFrame,
    decisions: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict[str, Any]:
    if daily.empty:
        raise ValueError(f"{scenario.scenario_id}: daily equity is empty.")

    final_equity = float(daily.iloc[-1]["equity_irr"])
    initial = float(scenario.initial_capital_irr)
    total_return = final_equity / initial - 1.0

    first_date = pd.Timestamp(daily.iloc[0]["date"])
    last_date = pd.Timestamp(daily.iloc[-1]["date"])
    calendar_days = max(1, int((last_date - first_date).days))
    years = calendar_days / 365.2425
    cagr = (final_equity / initial) ** (1.0 / years) - 1.0

    daily_returns = pd.to_numeric(daily["equity_irr"], errors="coerce").pct_change()
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()

    annualized_volatility = (
        float(daily_returns.std(ddof=1) * math.sqrt(252))
        if len(daily_returns) >= 2
        else float("nan")
    )
    sharpe = (
        float(daily_returns.mean() / daily_returns.std(ddof=1) * math.sqrt(252))
        if len(daily_returns) >= 2 and daily_returns.std(ddof=1) > 0.0
        else float("nan")
    )
    downside = daily_returns[daily_returns < 0.0]
    sortino = (
        float(daily_returns.mean() / downside.std(ddof=1) * math.sqrt(252))
        if len(downside) >= 2 and downside.std(ddof=1) > 0.0
        else float("nan")
    )
    max_drawdown = _maximum_drawdown(daily["equity_irr"])
    calmar = safe_ratio(cagr, abs(max_drawdown))

    returns = (
        pd.to_numeric(trades["net_return"], errors="coerce")
        if not trades.empty
        else pd.Series(dtype=float)
    )
    positive = returns[returns > 0.0]
    negative = returns[returns < 0.0]
    zeros = returns[np.isclose(returns, 0.0, atol=1.0e-12)]

    average_win = float(positive.mean()) if len(positive) else float("nan")
    average_loss = float(negative.mean()) if len(negative) else float("nan")
    payoff = safe_ratio(average_win, abs(average_loss))
    gross_profit = float(
        trades.loc[returns > 0.0, "net_pnl_irr"].sum()
        if not trades.empty
        else 0.0
    )
    gross_loss = float(
        -trades.loc[returns < 0.0, "net_pnl_irr"].sum()
        if not trades.empty
        else 0.0
    )
    profit_factor = safe_ratio(gross_profit, gross_loss)

    accepted = int(
        decisions["decision"].eq("accepted").sum()
        if not decisions.empty
        else 0
    )
    rejected = int(
        decisions["decision"].eq("rejected").sum()
        if not decisions.empty
        else 0
    )

    return {
        "scenario_id": scenario.scenario_id,
        "is_primary_scenario": scenario.is_primary,
        "position_structure": scenario.position_structure,
        "exit_style": scenario.exit_style,
        "initial_capital_irr": initial,
        "slippage_each_side": scenario.slippage_each_side,
        "first_execution_date": first_date,
        "last_execution_date": last_date,
        "calendar_days": calendar_days,
        "final_equity_irr": final_equity,
        "net_profit_irr": final_equity - initial,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_zero_risk_free": sharpe,
        "sortino_zero_target": sortino,
        "maximum_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "selected_signals": int(len(decisions)),
        "accepted_signals": accepted,
        "rejected_signals": rejected,
        "acceptance_rate": safe_ratio(accepted, len(decisions)),
        "closed_trades": int(len(trades)),
        "winning_trades": int((returns > 0.0).sum()),
        "losing_trades": int((returns < 0.0).sum()),
        "breakeven_trades": int(len(zeros)),
        "net_win_rate": float((returns > 0.0).mean()) if len(returns) else float("nan"),
        "mean_net_trade_return": float(returns.mean()) if len(returns) else float("nan"),
        "median_net_trade_return": float(returns.median()) if len(returns) else float("nan"),
        "average_winning_trade_return": average_win,
        "average_losing_trade_return": average_loss,
        "net_payoff_ratio": payoff,
        "gross_profit_irr": gross_profit,
        "gross_loss_absolute_irr": gross_loss,
        "net_profit_factor": profit_factor,
        "mean_holding_observations": (
            float(pd.to_numeric(trades["exit_observation"]).mean())
            if not trades.empty
            else float("nan")
        ),
        "mean_calendar_holding_days": (
            float(pd.to_numeric(trades["calendar_holding_days"]).mean())
            if not trades.empty
            else float("nan")
        ),
        "trailing_activated_trades": (
            int(trades["trailing_ever_activated"].sum())
            if not trades.empty
            else 0
        ),
        "total_buy_fees_irr": (
            float(trades["buy_fee_irr"].sum()) if not trades.empty else 0.0
        ),
        "total_sell_tax_irr": (
            float(trades["sell_tax_irr"].sum()) if not trades.empty else 0.0
        ),
        "total_sell_non_tax_fees_irr": (
            float(trades["sell_non_tax_fee_irr"].sum())
            if not trades.empty
            else 0.0
        ),
        "total_buy_slippage_irr": (
            float(trades["buy_slippage_irr"].sum()) if not trades.empty else 0.0
        ),
        "total_sell_slippage_irr": (
            float(trades["sell_slippage_irr"].sum()) if not trades.empty else 0.0
        ),
        "average_gross_exposure_fraction": float(
            pd.to_numeric(daily["gross_exposure_fraction"], errors="coerce").mean()
        ),
        "maximum_gross_exposure_fraction": float(
            pd.to_numeric(daily["gross_exposure_fraction"], errors="coerce").max()
        ),
        "average_open_lots": float(pd.to_numeric(daily["open_lots"]).mean()),
        "maximum_open_lots": int(pd.to_numeric(daily["open_lots"]).max()),
        "maximum_open_symbols": int(pd.to_numeric(daily["open_symbols"]).max()),
        "maximum_planned_open_risk_fraction": float(
            pd.to_numeric(
                daily["planned_open_risk_fraction"],
                errors="coerce",
            ).max()
        ),
        "maximum_protected_lots": int(
            pd.to_numeric(daily["protected_lots"]).max()
        ),
    }


def run_stage10(
    *,
    repository_root: Path,
    config: dict[str, Any],
    write_outputs: bool = True,
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()

    stage09_cfg = config["frozen_stage09"]
    expected_commit = str(stage09_cfg["producer_commit_sha"])
    expected_lock_sha = str(stage09_cfg["expected_inference_lock_sha256"])

    selected_path = repository_root / stage09_cfg["selected_signal_file"]
    lock_path = repository_root / stage09_cfg["inference_lock_file"]
    customer_path = discover_customer_file(
        repository_root,
        config["initial_capital"]["source_file"],
    )

    for path in [selected_path, lock_path, customer_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    actual_lock_sha = file_sha256(lock_path)
    if actual_lock_sha != expected_lock_sha:
        raise AssertionError(
            f"Stage 09 inference lock changed: {actual_lock_sha}"
        )

    customer_frame = pd.read_csv(customer_path, low_memory=False)
    capital_summary, capital_audit = calibrate_initial_capital(
        customer_frame,
        config,
    )

    signal_end = pd.Timestamp(config["temporal_scope"]["signal_generation_end"])
    tail_end = pd.Timestamp(config["temporal_scope"]["outcome_observation_tail_end"])

    signals = prepare_selected_signals(
        selected_path,
        expected_rows=int(stage09_cfg["expected_selected_signals"]),
        signal_generation_end=signal_end,
    )

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
                    "rows": len(history.frame),
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
        raise AssertionError("Signal execution-plan population changed.")

    scenarios = build_scenarios(config, capital_summary)

    all_trades: list[pd.DataFrame] = []
    all_decisions: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []

    for index, scenario in enumerate(scenarios, start=1):
        print(
            f"[{index:>2}/{len(scenarios)}] {scenario.scenario_id}",
            flush=True,
        )
        trades, decisions, daily, integrity = simulate_scenario(
            scenario=scenario,
            signal_plans=signal_plans,
            market_histories=histories,
            config=config,
        )
        all_trades.append(trades)
        all_decisions.append(decisions)
        all_daily.append(daily)
        summary_rows.append(
            summarize_scenario(scenario, trades, decisions, daily)
        )
        integrity_rows.append(integrity)

    trade_ledger = pd.concat(all_trades, ignore_index=True)
    signal_decisions = pd.concat(all_decisions, ignore_index=True)
    daily_equity = pd.concat(all_daily, ignore_index=True)
    scenario_summary = pd.DataFrame(summary_rows)
    integrity_audit = pd.DataFrame(integrity_rows)

    primary = scenario_summary.loc[
        scenario_summary["is_primary_scenario"]
    ]
    if len(primary) != 1:
        raise AssertionError("Primary scenario summary is not unique.")

    config_hash = canonical_json_sha256(config)
    manifest = {
        "stage": 10,
        "status": "completed_internal_integrity_checks",
        "schema_version": STAGE10_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": current_git_commit(repository_root),
        "stage09_producer_commit_sha": expected_commit,
        "stage09_inference_lock_sha256": actual_lock_sha,
        "stage09_selected_signal_file_sha256": file_sha256(selected_path),
        "customer_calibration_file_sha256": file_sha256(customer_path),
        "configuration_hash": config_hash,
        "currency": "IRR",
        "capital_calibration": capital_summary,
        "temporal_scope": config["temporal_scope"],
        "signals": {
            "selected": int(len(signals)),
            "planned": int(len(signal_plans)),
            "symbols": int(signals["symbol"].nunique()),
        },
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "primary_scenario": primary.iloc[0].to_dict(),
        "safeguards": {
            "long_only": True,
            "short_positions_created": False,
            "stage09_outcomes_used_for_signal_selection": False,
            "entry_on_next_trading_open": True,
            "same_bar_rule": "adverse_stop_first",
            "trailing_risk_release": (
                "next_symbol_trading_open_when_break_even_or_better_stop_is_executable"
            ),
            "protected_lot_planned_risk_floor": 0.0,
            "protected_lots_still_count_toward_exposure_and_capacity": True,
            "locked_profit_used_as_negative_risk": False,
            "portfolio_backtest_performed": True,
            "transaction_costs_applied": True,
            "slippage_applied": True,
        },
        "software": {
            "python_version": sys.version,
            "platform": os.name,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
    }

    outputs = {
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
        audits_dir = repository_root / paths["audits"]
        backtests_dir = repository_root / paths["backtests"]
        manifests_dir = repository_root / paths["manifests"]
        audits_dir.mkdir(parents=True, exist_ok=True)
        backtests_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        capital_audit.to_csv(
            audits_dir / "10_initial_capital_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        inventory.to_csv(
            audits_dir / "10_raw_market_inventory_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        signal_plan_errors.to_csv(
            audits_dir / "10_signal_execution_plan_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )
        market_errors.to_csv(
            audits_dir / "10_market_history_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )
        integrity_audit.to_csv(
            audits_dir / "10_portfolio_integrity_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )

        scenario_summary.to_csv(
            backtests_dir / "10_scenario_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        trade_ledger.to_csv(
            backtests_dir / "10_trade_ledger.csv",
            index=False,
            encoding="utf-8-sig",
        )
        signal_decisions.to_csv(
            backtests_dir / "10_signal_decisions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        daily_equity.to_csv(
            backtests_dir / "10_daily_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )
        signal_plans.to_csv(
            backtests_dir / "10_signal_execution_plan.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with (manifests_dir / "10_portfolio_backtest_manifest.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

    return outputs
