"""Stage 10C exploratory causal market-breadth evaluation.

This module does not retrain or rescore the frozen Stage 09 model. It:

1. builds a causal daily cross-sectional market-breadth series from the frozen
   Stage 02 universe;
2. joins the already-computed ``started`` value to all 78,189 frozen Stage 09
   candidates and defines the early-move rule ``1 <= started <= 3``;
3. compares candidate and frozen-score-selected outcomes across breadth bins,
   recovery/deterioration regimes, calendar years, and filter variants;
4. measures Pearson and Spearman relationships between breadth and the nine
   existing market-regime features.

The analysis is exploratory/post-hoc because unseen-test results were observed
before these rules were specified. Signal-level returns remain diagnostic gross
Triple-Barrier outcomes, not executable portfolio returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math
import subprocess

import numpy as np
import pandas as pd


STAGE10C_SCHEMA_VERSION = (
    "stage10c_v1_causal_breadth_started_1_to_3_no_retraining"
)

MARKET_FEATURE_COLUMNS = (
    "market_return_1",
    "market_return_5",
    "market_return_20",
    "market_volatility_20",
    "market_ema_20_distance",
    "market_ema_60_distance",
    "market_range_fraction",
    "market_close_location",
    "market_drawdown_60",
)

BREADTH_FEATURE_COLUMNS = (
    "market_breadth_raw",
    "market_breadth_ema30",
    "market_breadth_slope5",
    "market_breadth_scaled_100",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_git_commit(repository_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def parse_market_date(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    numeric_like = raw.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if numeric_like.any():
        parsed.loc[numeric_like] = pd.to_datetime(
            raw.loc[numeric_like], format="%Y%m%d", errors="coerce"
        )
    if (~numeric_like).any():
        parsed.loc[~numeric_like] = pd.to_datetime(
            raw.loc[~numeric_like], errors="coerce"
        )
    return parsed.dt.normalize()


def _discover_csv_map(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    mapping: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.csv")):
        symbol = path.stem.strip()
        for suffix in (
            "_unseen_test_labeled",
            "_unseen_test",
            "_train_labeled",
            "_train",
        ):
            if symbol.endswith(suffix):
                symbol = symbol[: -len(suffix)]
                break
        if symbol in mapping:
            raise ValueError(
                f"Duplicate CSV mapping for symbol '{symbol}' below {directory}."
            )
        mapping[symbol] = path
    return mapping


def _read_available_columns(path: Path, requested: Iterable[str]) -> pd.DataFrame:
    requested = list(dict.fromkeys(requested))
    header = pd.read_csv(path, nrows=0)
    available = [column for column in requested if column in header.columns]
    if not available:
        raise KeyError(f"None of the requested columns exist in {path}.")
    return pd.read_csv(path, usecols=available, low_memory=False)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    valid = np.isfinite(num) & np.isfinite(den) & den.gt(0)
    result = pd.Series(np.nan, index=num.index, dtype=float)
    result.loc[valid] = num.loc[valid] / den.loc[valid]
    return result


def compute_market_breadth(
    symbol_price_panel: pd.DataFrame,
    *,
    universe_size: int,
    ema_span: int = 30,
    ema_min_periods: int = 30,
    slope_lag: int = 5,
    zero_return_tolerance: float = 0.0,
    transition_lower_bound: float = -0.30,
    transition_upper_bound: float = 0.30,
) -> pd.DataFrame:
    """Build causal daily breadth from valid symbol trading observations."""
    required = {"symbol", "dEven", "adj_last_price"}
    missing = sorted(required - set(symbol_price_panel.columns))
    if missing:
        raise KeyError(f"Breadth input columns are missing: {missing}")
    if universe_size <= 0:
        raise ValueError("universe_size must be positive.")
    if ema_span <= 0 or ema_min_periods <= 0 or slope_lag <= 0:
        raise ValueError("Breadth smoothing parameters must be positive.")

    panel = symbol_price_panel[list(required)].copy()
    panel["symbol"] = panel["symbol"].astype(str)
    panel["dEven"] = parse_market_date(panel["dEven"])
    panel["adj_last_price"] = pd.to_numeric(
        panel["adj_last_price"], errors="coerce"
    )
    panel = panel.loc[
        panel["dEven"].notna()
        & np.isfinite(panel["adj_last_price"])
        & panel["adj_last_price"].gt(0)
    ].copy()
    panel = (
        panel.sort_values(["symbol", "dEven"], kind="mergesort")
        .drop_duplicates(["symbol", "dEven"], keep="last")
        .reset_index(drop=True)
    )
    panel["previous_valid_adjusted_close"] = panel.groupby(
        "symbol", sort=False
    )["adj_last_price"].shift(1)
    panel["daily_adjusted_return"] = (
        panel["adj_last_price"] / panel["previous_valid_adjusted_close"] - 1.0
    )
    valid_return = (
        np.isfinite(panel["daily_adjusted_return"])
        & panel["previous_valid_adjusted_close"].gt(0)
    )
    panel["direction"] = np.nan
    tolerance = float(zero_return_tolerance)
    panel.loc[valid_return & panel["daily_adjusted_return"].gt(tolerance), "direction"] = 1
    panel.loc[valid_return & panel["daily_adjusted_return"].lt(-tolerance), "direction"] = -1
    panel.loc[
        valid_return & panel["daily_adjusted_return"].abs().le(tolerance),
        "direction",
    ] = 0

    active_prices = (
        panel.groupby("dEven", sort=True)["symbol"]
        .nunique()
        .rename("symbols_with_valid_price")
    )
    directional = panel.loc[panel["direction"].notna()].copy()
    grouped = directional.groupby("dEven", sort=True)["direction"]
    daily = pd.DataFrame(
        {
            "positive_symbols": grouped.apply(lambda s: int((s > 0).sum())),
            "negative_symbols": grouped.apply(lambda s: int((s < 0).sum())),
            "unchanged_symbols": grouped.apply(lambda s: int((s == 0).sum())),
            "valid_return_symbols": grouped.size().astype(int),
        }
    ).reset_index()
    daily = daily.merge(
        active_prices.reset_index(),
        on="dEven",
        how="outer",
        validate="one_to_one",
    ).sort_values("dEven", kind="stable").reset_index(drop=True)
    count_columns = [
        "positive_symbols",
        "negative_symbols",
        "unchanged_symbols",
        "valid_return_symbols",
        "symbols_with_valid_price",
    ]
    daily[count_columns] = daily[count_columns].fillna(0).astype(int)
    daily["frozen_universe_size"] = int(universe_size)
    daily["breadth_coverage_fraction"] = (
        daily["valid_return_symbols"] / float(universe_size)
    )
    daily["market_breadth_raw"] = np.where(
        daily["valid_return_symbols"].gt(0),
        (
            daily["positive_symbols"] - daily["negative_symbols"]
        ) / daily["valid_return_symbols"],
        np.nan,
    )
    if daily["market_breadth_raw"].dropna().abs().gt(1.0 + 1e-12).any():
        raise AssertionError("Raw market breadth left the [-1, 1] interval.")

    daily["market_breadth_ema30"] = daily["market_breadth_raw"].ewm(
        span=int(ema_span),
        adjust=False,
        min_periods=int(ema_min_periods),
    ).mean()
    daily["market_breadth_slope5"] = (
        daily["market_breadth_ema30"]
        - daily["market_breadth_ema30"].shift(int(slope_lag))
    )
    daily["market_breadth_scaled_100"] = 50.0 * (
        daily["market_breadth_ema30"] + 1.0
    )

    lower = float(transition_lower_bound)
    upper = float(transition_upper_bound)
    if not -1.0 <= lower < upper <= 1.0:
        raise ValueError("Transition bounds must satisfy -1 <= lower < upper <= 1.")

    breadth = daily["market_breadth_ema30"]
    slope = daily["market_breadth_slope5"]
    in_transition = breadth.between(lower, upper, inclusive="both")
    daily["market_regime"] = "warmup_or_missing"
    daily.loc[breadth.lt(lower), "market_regime"] = "broad_decline"
    daily.loc[breadth.gt(upper), "market_regime"] = "broad_advance"
    daily.loc[in_transition & slope.gt(0), "market_regime"] = "recovery"
    daily.loc[in_transition & slope.lt(0), "market_regime"] = "deterioration"
    daily.loc[in_transition & slope.eq(0), "market_regime"] = "transition_flat"
    daily["market_recovery"] = daily["market_regime"].eq("recovery")
    daily["market_deterioration"] = daily["market_regime"].eq("deterioration")
    return daily


def build_existing_market_features(
    canonical_market_index: pd.DataFrame,
    *,
    volatility_window: int = 20,
    ema_fast_window: int = 20,
    ema_slow_window: int = 60,
    drawdown_window: int = 60,
) -> pd.DataFrame:
    """Reconstruct the nine causal market-regime features used by Stage 09."""
    required = {
        "dEven",
        "xNivInuClMresIbs",
        "xNivInuPbMresIbs",
        "xNivInuPhMresIbs",
    }
    missing = sorted(required - set(canonical_market_index.columns))
    if missing:
        raise KeyError(f"Canonical market-index columns are missing: {missing}")
    market = canonical_market_index[list(required)].copy()
    market["dEven"] = parse_market_date(market["dEven"])
    for column in required - {"dEven"}:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market = (
        market.dropna(subset=["dEven"])
        .sort_values("dEven", kind="stable")
        .drop_duplicates("dEven", keep="last")
        .reset_index(drop=True)
    )

    close = market["xNivInuClMresIbs"]
    low = market["xNivInuPbMresIbs"]
    high = market["xNivInuPhMresIbs"]
    result = pd.DataFrame({"dEven": market["dEven"]})
    result["market_return_1"] = close.pct_change(1, fill_method=None)
    result["market_return_5"] = close.pct_change(5, fill_method=None)
    result["market_return_20"] = close.pct_change(20, fill_method=None)

    previous_close = close.shift(1)
    log_return = pd.Series(np.nan, index=market.index, dtype=float)
    valid_log = (
        close.gt(0)
        & previous_close.gt(0)
        & np.isfinite(close)
        & np.isfinite(previous_close)
    )
    log_return.loc[valid_log] = np.log(
        close.loc[valid_log] / previous_close.loc[valid_log]
    )
    result["market_volatility_20"] = log_return.rolling(
        int(volatility_window), min_periods=int(volatility_window)
    ).std(ddof=0)

    ema_fast = close.ewm(
        span=int(ema_fast_window),
        adjust=False,
        min_periods=int(ema_fast_window),
    ).mean()
    ema_slow = close.ewm(
        span=int(ema_slow_window),
        adjust=False,
        min_periods=int(ema_slow_window),
    ).mean()
    result["market_ema_20_distance"] = _safe_ratio(close - ema_fast, close)
    result["market_ema_60_distance"] = _safe_ratio(close - ema_slow, close)

    valid_ohlc = (
        np.isfinite(close)
        & np.isfinite(low)
        & np.isfinite(high)
        & close.gt(0)
        & low.gt(0)
        & high.gt(0)
        & high.ge(low)
        & close.ge(low)
        & close.le(high)
    )
    result["market_range_fraction"] = np.nan
    result.loc[valid_ohlc, "market_range_fraction"] = (
        high.loc[valid_ohlc] - low.loc[valid_ohlc]
    ) / close.loc[valid_ohlc]

    close_location = pd.Series(np.nan, index=market.index, dtype=float)
    ordinary = valid_ohlc & high.gt(low)
    close_location.loc[ordinary] = (
        close.loc[ordinary] - low.loc[ordinary]
    ) / (high.loc[ordinary] - low.loc[ordinary])
    locked = valid_ohlc & high.eq(low) & high.eq(close)
    locked_previous = locked & previous_close.notna()
    close_location.loc[locked_previous & close.gt(previous_close)] = 1.0
    close_location.loc[locked_previous & close.lt(previous_close)] = 0.0
    result["market_close_location"] = close_location

    rolling_max = close.rolling(
        int(drawdown_window), min_periods=int(drawdown_window)
    ).max()
    result["market_drawdown_60"] = _safe_ratio(close, rolling_max) - 1.0
    return result


def _canonicalize_market_index(
    observations: pd.DataFrame,
    *,
    relative_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = [
        "xNivInuClMresIbs",
        "xNivInuPbMresIbs",
        "xNivInuPhMresIbs",
    ]
    required = {"dEven", "source_symbol", *fields}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise KeyError(f"Market-index observation columns are missing: {missing}")
    panel = observations.copy()
    panel["dEven"] = parse_market_date(panel["dEven"])
    for field in fields:
        panel[field] = pd.to_numeric(panel[field], errors="coerce")
    panel = panel.dropna(subset=["dEven"]).reset_index(drop=True)

    audit_parts: list[pd.DataFrame] = []
    canonical_parts: list[pd.Series] = []
    grouped = panel.groupby("dEven", sort=True)
    for field in fields:
        stats = grouped[field].agg(
            nonmissing_source_rows="count",
            minimum="min",
            maximum="max",
            canonical_value="median",
        )
        scale = stats["canonical_value"].abs().clip(lower=1.0)
        stats["relative_spread"] = (
            stats["maximum"] - stats["minimum"]
        ).abs() / scale
        stats["inconsistent_across_raw_files"] = (
            stats["relative_spread"] > float(relative_tolerance)
        )
        stats["market_index_field"] = field
        audit_parts.append(stats.reset_index())
        canonical_parts.append(
            stats["canonical_value"].rename(field)
        )
    canonical = pd.concat(canonical_parts, axis=1).reset_index()
    canonical = canonical.sort_values("dEven", kind="stable").reset_index(drop=True)
    audit = pd.concat(audit_parts, ignore_index=True).sort_values(
        ["dEven", "market_index_field"], kind="stable"
    ).reset_index(drop=True)
    return canonical, audit


def _select_daily_top_fraction(
    frame: pd.DataFrame,
    *,
    score_column: str,
    fraction: float,
    minimum_per_date: int,
) -> pd.DataFrame:
    required = {"event_id", "symbol", "dEven", score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Signal-policy columns are missing: {missing}")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    result = frame.copy()
    result["dEven"] = pd.to_datetime(result["dEven"], errors="raise").dt.normalize()
    result[score_column] = pd.to_numeric(result[score_column], errors="raise")
    result = result.sort_values(
        ["dEven", score_column, "symbol", "event_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["variant_daily_candidate_count"] = result.groupby(
        "dEven", sort=False
    )["event_id"].transform("size").astype(int)
    result["variant_daily_rank"] = (
        result.groupby("dEven", sort=False).cumcount() + 1
    ).astype(int)
    result["variant_daily_signal_quota"] = result[
        "variant_daily_candidate_count"
    ].map(
        lambda count: min(
            int(count),
            max(
                int(minimum_per_date),
                int(math.ceil(float(fraction) * int(count))),
            ),
        )
    ).astype(int)
    result["variant_selected_signal"] = (
        result["variant_daily_rank"] <= result["variant_daily_signal_quota"]
    )
    cutoffs = (
        result.loc[result["variant_selected_signal"]]
        .groupby("dEven", sort=False)[score_column]
        .min()
        .rename("variant_daily_score_cutoff")
    )
    result = result.merge(
        cutoffs,
        left_on="dEven",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    return result


def _outcome_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "events": 0,
            "positive_events": 0,
            "negative_events": 0,
            "positive_rate": np.nan,
            "mean_score": np.nan,
            "mean_event_return": np.nan,
            "median_event_return": np.nan,
            "average_positive_return": np.nan,
            "average_negative_return": np.nan,
            "payoff_ratio": np.nan,
            "profit_factor": np.nan,
        }
    labels = pd.to_numeric(frame["meta_label"], errors="coerce")
    returns = pd.to_numeric(frame["original_event_return"], errors="coerce")
    scores = pd.to_numeric(frame["xgboost_ranking_score"], errors="coerce")
    positives = returns.gt(0)
    negatives = returns.lt(0)
    average_positive = float(returns.loc[positives].mean()) if positives.any() else np.nan
    average_negative = float(returns.loc[negatives].mean()) if negatives.any() else np.nan
    gross_profit = float(returns.loc[positives].sum())
    gross_loss = float(-returns.loc[negatives].sum())
    return {
        "events": int(len(frame)),
        "positive_events": int(labels.eq(1).sum()),
        "negative_events": int(labels.eq(0).sum()),
        "positive_rate": float(labels.mean()),
        "mean_score": float(scores.mean()),
        "mean_event_return": float(returns.mean()),
        "median_event_return": float(returns.median()),
        "average_positive_return": average_positive,
        "average_negative_return": average_negative,
        "payoff_ratio": (
            average_positive / abs(average_negative)
            if np.isfinite(average_positive)
            and np.isfinite(average_negative)
            and average_negative != 0.0
            else np.nan
        ),
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0.0 else np.nan
        ),
    }


def _variant_mask(frame: pd.DataFrame, variant: str) -> pd.Series:
    started = frame["started_1_to_3_pass"].astype(bool)
    recovery = frame["market_recovery"].astype(bool)
    deterioration = frame["market_deterioration"].astype(bool)
    if variant == "baseline":
        return pd.Series(True, index=frame.index)
    if variant == "started_1_to_3":
        return started
    if variant == "recovery":
        return recovery
    if variant == "deterioration":
        return deterioration
    if variant == "started_1_to_3_and_recovery":
        return started & recovery
    if variant == "started_1_to_3_and_deterioration":
        return started & deterioration
    raise KeyError(f"Unknown analysis variant: {variant}")


def _build_variant_outputs(
    candidate_panel: pd.DataFrame,
    *,
    variants: list[str],
    fraction: float,
    minimum_per_date: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    for variant in variants:
        eligible = candidate_panel.loc[_variant_mask(candidate_panel, variant)].copy()
        reranked = _select_daily_top_fraction(
            eligible,
            score_column="xgboost_ranking_score",
            fraction=fraction,
            minimum_per_date=minimum_per_date,
        ) if not eligible.empty else eligible
        selected = (
            reranked.loc[reranked["variant_selected_signal"]].copy()
            if not reranked.empty
            else reranked
        )
        selected["analysis_variant"] = variant
        selected_parts.append(selected)

        candidate_metrics = _outcome_metrics(eligible)
        selected_metrics = _outcome_metrics(selected)
        summary_rows.append(
            {
                "analysis_variant": variant,
                "candidate_dates": int(eligible["dEven"].nunique()),
                **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
                "selected_dates": int(selected["dEven"].nunique()) if not selected.empty else 0,
                **{f"selected_{key}": value for key, value in selected_metrics.items()},
            }
        )

        for year in range(2021, 2025):
            candidate_year = eligible.loc[eligible["calendar_year"].eq(year)]
            selected_year = selected.loc[selected["calendar_year"].eq(year)]
            for scope, subset in (
                ("candidate_population", candidate_year),
                ("daily_top5_selected", selected_year),
            ):
                yearly_rows.append(
                    {
                        "calendar_year": year,
                        "analysis_variant": variant,
                        "population_scope": scope,
                        "dates": int(subset["dEven"].nunique()) if not subset.empty else 0,
                        **_outcome_metrics(subset),
                    }
                )

    selected_all = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else pd.DataFrame()
    )
    return selected_all, pd.DataFrame(summary_rows), pd.DataFrame(yearly_rows)


def _build_group_metrics(
    candidate_panel: pd.DataFrame,
    *,
    group_column: str,
    scopes: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope_name, scope_mask in scopes.items():
        scoped = candidate_panel.loc[scope_mask].copy()
        for group_value, group in scoped.groupby(group_column, observed=True, dropna=False):
            rows.append(
                {
                    "analysis_scope": scope_name,
                    group_column: group_value,
                    "dates": int(group["dEven"].nunique()),
                    **_outcome_metrics(group),
                }
            )
    return pd.DataFrame(rows)


def _correlation_long(
    daily: pd.DataFrame,
    *,
    method: str,
    breadth_columns: list[str],
    market_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    partitions = {
        "train": daily["partition"].eq("train"),
        "unseen_test": daily["partition"].eq("unseen_test"),
        "full": pd.Series(True, index=daily.index),
    }
    for partition, mask in partitions.items():
        subset = daily.loc[mask]
        for breadth_column in breadth_columns:
            for market_column in market_columns:
                pair = subset[[breadth_column, market_column]].dropna()
                has_variation = (
                    len(pair) >= 3
                    and pair[breadth_column].nunique(dropna=True) > 1
                    and pair[market_column].nunique(dropna=True) > 1
                )
                correlation = (
                    float(pair[breadth_column].corr(pair[market_column], method=method))
                    if has_variation
                    else np.nan
                )
                rows.append(
                    {
                        "partition": partition,
                        "method": method,
                        "breadth_feature": breadth_column,
                        "market_feature": market_column,
                        "paired_dates": int(len(pair)),
                        "correlation": correlation,
                    }
                )
    return pd.DataFrame(rows)


def _load_started_for_candidates(
    lock: pd.DataFrame,
    *,
    raw_map: dict[str, Path],
    unseen_map: dict[str, Path],
    date_column: str,
    started_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for symbol, events in lock.groupby("symbol", sort=True):
        symbol = str(symbol)
        sources: list[tuple[str, Path]] = []
        if symbol in raw_map:
            sources.append(("raw_data", raw_map[symbol]))
        if symbol in unseen_map:
            sources.append(("unseen_test_data", unseen_map[symbol]))
        loaded = False
        failures: list[str] = []
        for role, path in sources:
            try:
                source = _read_available_columns(path, [date_column, started_column])
                if started_column not in source.columns:
                    raise KeyError(f"{started_column} is missing from {path}.")
                source[date_column] = parse_market_date(source[date_column])
                source[started_column] = pd.to_numeric(
                    source[started_column], errors="coerce"
                )
                source = (
                    source.dropna(subset=[date_column])
                    .sort_values(date_column, kind="stable")
                    .drop_duplicates(date_column, keep="last")
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
                if missing_started:
                    raise ValueError(
                        f"{missing_started} candidate events have no started value."
                    )
                noninteger = (
                    joined[started_column].notna()
                    & ~np.isclose(
                        joined[started_column],
                        np.round(joined[started_column]),
                        atol=1e-12,
                    )
                )
                if noninteger.any():
                    raise ValueError(
                        f"{int(noninteger.sum())} candidate events have non-integer started."
                    )
                joined[started_column] = np.round(joined[started_column]).astype(int)
                joined["started_source_role"] = role
                joined["started_source_path"] = str(path)
                parts.append(joined)
                audits.append(
                    {
                        "symbol": symbol,
                        "source_role": role,
                        "source_path": str(path),
                        "candidate_events": int(len(joined)),
                        "started_0": int(joined[started_column].eq(0).sum()),
                        "started_1": int(joined[started_column].eq(1).sum()),
                        "started_2": int(joined[started_column].eq(2).sum()),
                        "started_3": int(joined[started_column].eq(3).sum()),
                        "started_1_to_3": int(joined[started_column].between(1, 3).sum()),
                        "started_above_3": int(joined[started_column].gt(3).sum()),
                        "started_below_0": int(joined[started_column].lt(0).sum()),
                        "minimum_started": int(joined[started_column].min()),
                        "maximum_started": int(joined[started_column].max()),
                    }
                )
                loaded = True
                break
            except Exception as exc:
                failures.append(f"{role}:{path} -> {type(exc).__name__}: {exc}")
        if not loaded:
            errors.append(
                {
                    "symbol": symbol,
                    "error_type": "StartedSourceResolutionError",
                    "error_message": " | ".join(failures) or "No candidate source file.",
                }
            )
    panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return panel, pd.DataFrame(audits), pd.DataFrame(errors)


def run_stage10c(
    *,
    repository_root: Path,
    config: dict[str, Any],
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run the complete Stage 10C exploratory diagnostic."""
    repository_root = Path(repository_root).resolve()
    if str(config.get("schema_version")) != STAGE10C_SCHEMA_VERSION:
        raise AssertionError("Unexpected Stage 10C schema version.")

    frozen = config["frozen_inputs"]
    paths = config["paths"]
    lock_path = repository_root / frozen["inference_lock_file"]
    evaluation_path = repository_root / frozen["stage09_evaluation_file"]
    universe_path = repository_root / frozen["frozen_universe_manifest"]
    for path in (lock_path, evaluation_path, universe_path):
        if not path.exists():
            raise FileNotFoundError(path)

    lock_hash = file_sha256(lock_path)
    if lock_hash != str(frozen["expected_inference_lock_sha256"]):
        raise AssertionError(f"Stage 09 inference lock changed: {lock_hash}")

    universe = pd.read_csv(universe_path, low_memory=False)
    if "symbol" not in universe.columns:
        raise KeyError("Frozen universe manifest has no symbol column.")
    symbols = sorted(universe["symbol"].astype(str).unique())
    if len(symbols) != int(frozen["expected_universe_size"]):
        raise AssertionError(
            f"Expected {frozen['expected_universe_size']} symbols, found {len(symbols)}."
        )

    lock = pd.read_csv(lock_path, low_memory=False)
    expected_lock_columns = {
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
    missing_lock = sorted(expected_lock_columns - set(lock.columns))
    if missing_lock:
        raise KeyError(f"Inference-lock columns are missing: {missing_lock}")
    lock["dEven"] = pd.to_datetime(lock["dEven"], errors="raise").dt.normalize()
    lock["symbol"] = lock["symbol"].astype(str)
    lock["xgboost_ranking_score"] = pd.to_numeric(
        lock["xgboost_ranking_score"], errors="raise"
    )
    selected_text = lock["selected_signal"].astype(str).str.strip().str.lower()
    if not selected_text.isin({"true", "false", "1", "0"}).all():
        raise ValueError("Inference-lock selected_signal is not boolean.")
    lock["stage09_selected_signal"] = selected_text.isin({"true", "1"})
    if len(lock) != int(frozen["expected_candidate_events"]):
        raise AssertionError("Unexpected frozen candidate count.")
    if lock["dEven"].nunique() != int(frozen["expected_signal_dates"]):
        raise AssertionError("Unexpected frozen signal-date count.")
    if int(lock["stage09_selected_signal"].sum()) != int(
        frozen["expected_stage09_selected_signals"]
    ):
        raise AssertionError("Unexpected frozen selected-signal count.")
    if lock["event_id"].duplicated().any():
        raise AssertionError("Duplicate event IDs exist in the inference lock.")

    evaluation = pd.read_csv(evaluation_path, low_memory=False)
    required_evaluation = {
        "event_id",
        "meta_label",
        "original_event_return",
    }
    missing_evaluation = sorted(required_evaluation - set(evaluation.columns))
    if missing_evaluation:
        raise KeyError(f"Stage 09 evaluation columns are missing: {missing_evaluation}")
    evaluation = evaluation[[
        "event_id", "meta_label", "original_event_return"
    ]].copy()
    evaluation["meta_label"] = pd.to_numeric(
        evaluation["meta_label"], errors="raise"
    ).astype(int)
    evaluation["original_event_return"] = pd.to_numeric(
        evaluation["original_event_return"], errors="raise"
    )
    if not evaluation["meta_label"].isin([0, 1]).all():
        raise ValueError("Stage 09 meta labels are not binary.")
    if evaluation["event_id"].duplicated().any():
        raise AssertionError("Duplicate event IDs exist in Stage 09 evaluation.")
    if set(evaluation["event_id"].astype(str)) != set(lock["event_id"].astype(str)):
        raise AssertionError("Stage 09 evaluation population differs from the lock.")

    raw_map_all = _discover_csv_map(repository_root / paths["raw_data"])
    unseen_map = _discover_csv_map(repository_root / paths["unseen_test_data"])
    missing_raw_symbols = sorted(set(symbols) - set(raw_map_all))
    if missing_raw_symbols:
        raise FileNotFoundError(
            f"Raw files are missing for {len(missing_raw_symbols)} frozen symbols."
        )
    raw_map = {symbol: raw_map_all[symbol] for symbol in symbols}

    temporal = config["temporal_scope"]
    signal_end = pd.Timestamp(temporal["signal_generation_end"])
    train_end = pd.Timestamp(temporal["train_end"])
    test_start = pd.Timestamp(temporal["unseen_test_start"])

    breadth_cfg = config["breadth"]
    index_cfg = config["market_index"]
    price_parts: list[pd.DataFrame] = []
    index_parts: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    source_errors: list[dict[str, Any]] = []
    required_source_columns = [
        breadth_cfg["date_column"],
        breadth_cfg["adjusted_close_column"],
        index_cfg["close_column"],
        index_cfg["low_column"],
        index_cfg["high_column"],
    ]
    for symbol in symbols:
        path = raw_map[symbol]
        try:
            source = _read_available_columns(path, required_source_columns)
            missing_required = sorted(set(required_source_columns) - set(source.columns))
            if missing_required:
                raise KeyError(f"Missing columns {missing_required}")
            source["dEven"] = parse_market_date(source[breadth_cfg["date_column"]])
            source = source.loc[
                source["dEven"].notna() & source["dEven"].le(signal_end)
            ].copy()
            price = source[["dEven", breadth_cfg["adjusted_close_column"]]].rename(
                columns={breadth_cfg["adjusted_close_column"]: "adj_last_price"}
            )
            price["symbol"] = symbol
            price_parts.append(price)

            index = source[[
                "dEven",
                index_cfg["close_column"],
                index_cfg["low_column"],
                index_cfg["high_column"],
            ]].rename(
                columns={
                    index_cfg["close_column"]: "xNivInuClMresIbs",
                    index_cfg["low_column"]: "xNivInuPbMresIbs",
                    index_cfg["high_column"]: "xNivInuPhMresIbs",
                }
            )
            index["source_symbol"] = symbol
            index_parts.append(index)
            inventory_rows.append(
                {
                    "symbol": symbol,
                    "raw_path": str(path),
                    "rows_through_signal_end": int(len(source)),
                    "first_date": source["dEven"].min(),
                    "last_date": source["dEven"].max(),
                }
            )
        except Exception as exc:
            source_errors.append(
                {
                    "symbol": symbol,
                    "path": str(path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    source_errors_df = pd.DataFrame(source_errors)
    if not source_errors_df.empty:
        raise RuntimeError(f"{len(source_errors_df)} market source-loading errors exist.")

    price_panel = pd.concat(price_parts, ignore_index=True)
    index_observations = pd.concat(index_parts, ignore_index=True)
    daily_breadth = compute_market_breadth(
        price_panel,
        universe_size=len(symbols),
        ema_span=int(breadth_cfg["ema_span"]),
        ema_min_periods=int(breadth_cfg["ema_min_periods"]),
        slope_lag=int(breadth_cfg["slope_lag_market_observations"]),
        zero_return_tolerance=float(breadth_cfg["zero_return_tolerance"]),
        transition_lower_bound=float(breadth_cfg["transition_lower_bound"]),
        transition_upper_bound=float(breadth_cfg["transition_upper_bound"]),
    )
    canonical_index, index_consistency = _canonicalize_market_index(
        index_observations,
        relative_tolerance=float(index_cfg["consistency_relative_tolerance"]),
    )
    market_features = build_existing_market_features(
        canonical_index,
        volatility_window=int(index_cfg["market_volatility_window"]),
        ema_fast_window=int(index_cfg["market_ema_fast_window"]),
        ema_slow_window=int(index_cfg["market_ema_slow_window"]),
        drawdown_window=int(index_cfg["market_drawdown_window"]),
    )
    daily = daily_breadth.merge(
        market_features,
        on="dEven",
        how="outer",
        validate="one_to_one",
    ).sort_values("dEven", kind="stable").reset_index(drop=True)
    daily["partition"] = np.select(
        [daily["dEven"].le(train_end), daily["dEven"].ge(test_start)],
        ["train", "unseen_test"],
        default="gap",
    )

    started_cfg = config["started_filter"]
    started_panel, started_audit, started_errors = _load_started_for_candidates(
        lock,
        raw_map=raw_map,
        unseen_map=unseen_map,
        date_column=str(started_cfg["date_column"]),
        started_column=str(started_cfg["column"]),
    )
    if not started_errors.empty:
        raise RuntimeError(f"{len(started_errors)} started-source errors exist.")
    if len(started_panel) != len(lock):
        raise AssertionError("The started join changed the candidate population.")

    candidate = (
        lock.merge(evaluation, on="event_id", how="left", validate="one_to_one")
        .merge(
            started_panel[[
                "event_id", "started", "started_source_role", "started_source_path"
            ]],
            on="event_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            daily[[
                "dEven",
                *BREADTH_FEATURE_COLUMNS,
                "market_regime",
                "market_recovery",
                "market_deterioration",
                "breadth_coverage_fraction",
                *MARKET_FEATURE_COLUMNS,
            ]],
            on="dEven",
            how="left",
            validate="many_to_one",
        )
    )
    if candidate[list(BREADTH_FEATURE_COLUMNS)].isna().any().any():
        missing_count = int(candidate[list(BREADTH_FEATURE_COLUMNS)].isna().any(axis=1).sum())
        raise AssertionError(
            f"{missing_count} frozen candidates have missing breadth features."
        )
    candidate["started_1_to_3_pass"] = candidate["started"].between(
        int(started_cfg["minimum_inclusive"]),
        int(started_cfg["maximum_inclusive"]),
        inclusive="both",
    )
    candidate["calendar_year"] = candidate["dEven"].dt.year.astype(int)

    signal_cfg = config["signal_policy"]
    baseline_reconstructed = _select_daily_top_fraction(
        candidate,
        score_column="xgboost_ranking_score",
        fraction=float(signal_cfg["selected_fraction"]),
        minimum_per_date=int(signal_cfg["minimum_per_date"]),
    )
    baseline_match = baseline_reconstructed["variant_selected_signal"].eq(
        baseline_reconstructed["stage09_selected_signal"]
    )
    if not baseline_match.all():
        raise AssertionError(
            f"Reconstructed baseline selection differs on {int((~baseline_match).sum())} rows."
        )

    variants = [str(value) for value in config["analysis_variants"]]
    selected_variants, variant_summary, yearly_summary = _build_variant_outputs(
        candidate,
        variants=variants,
        fraction=float(signal_cfg["selected_fraction"]),
        minimum_per_date=int(signal_cfg["minimum_per_date"]),
    )

    edges = [float(value) for value in breadth_cfg["bins"]]
    labels = [str(value) for value in breadth_cfg["bin_labels"]]
    if len(labels) != len(edges) - 1:
        raise ValueError("Breadth bin labels do not match edges.")
    candidate["breadth_ema30_bin"] = pd.cut(
        candidate["market_breadth_ema30"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    # Include the exact upper endpoint +1 in the final bin.
    candidate.loc[
        np.isclose(candidate["market_breadth_ema30"], edges[-1]),
        "breadth_ema30_bin",
    ] = labels[-1]

    scopes = {
        "all_candidates": pd.Series(True, index=candidate.index),
        "started_1_to_3_candidates": candidate["started_1_to_3_pass"],
    }
    breadth_bin_metrics = _build_group_metrics(
        candidate,
        group_column="breadth_ema30_bin",
        scopes=scopes,
    )
    regime_metrics = _build_group_metrics(
        candidate,
        group_column="market_regime",
        scopes=scopes,
    )

    pearson = _correlation_long(
        daily,
        method="pearson",
        breadth_columns=list(BREADTH_FEATURE_COLUMNS[:3]),
        market_columns=[str(value) for value in index_cfg["correlation_features"]],
    )
    spearman = _correlation_long(
        daily,
        method="spearman",
        breadth_columns=list(BREADTH_FEATURE_COLUMNS[:3]),
        market_columns=[str(value) for value in index_cfg["correlation_features"]],
    )

    result_paths = {
        "daily_breadth": repository_root / paths["metrics"] / "10c_daily_market_breadth.csv",
        "candidate_enriched": repository_root / paths["predictions"] / "10c_candidate_breadth_enriched.csv",
        "selected_variants": repository_root / paths["predictions"] / "10c_selected_signals_by_variant.csv",
        "variant_summary": repository_root / paths["metrics"] / "10c_filter_variant_summary.csv",
        "breadth_bin_metrics": repository_root / paths["metrics"] / "10c_breadth_bin_positive_rate.csv",
        "regime_metrics": repository_root / paths["metrics"] / "10c_recovery_deterioration_comparison.csv",
        "yearly_summary": repository_root / paths["metrics"] / "10c_yearly_filter_comparison.csv",
        "pearson": repository_root / paths["metrics"] / "10c_market_feature_correlation_pearson.csv",
        "spearman": repository_root / paths["metrics"] / "10c_market_feature_correlation_spearman.csv",
        "inventory": repository_root / paths["audits"] / "10c_market_breadth_source_inventory.csv",
        "started_audit": repository_root / paths["audits"] / "10c_started_1_to_3_source_audit.csv",
        "index_consistency": repository_root / paths["audits"] / "10c_market_index_consistency_audit.csv",
        "manifest": repository_root / paths["manifests"] / "10c_market_breadth_exploratory_manifest.json",
    }

    manifest = {
        "stage": "10C",
        "status": "completed_internal_integrity_checks",
        "schema_version": STAGE10C_SCHEMA_VERSION,
        "git_commit_sha": current_git_commit(repository_root),
        "configuration_hash": canonical_json_sha256(config),
        "lineage": {
            "stage09_inference_lock_file": str(lock_path),
            "stage09_inference_lock_sha256": lock_hash,
            "frozen_universe_file": str(universe_path),
            "frozen_universe_size": len(symbols),
            "stage09_evaluation_file": str(evaluation_path),
        },
        "candidate_population": {
            "events": int(len(candidate)),
            "symbols": int(candidate["symbol"].nunique()),
            "signal_dates": int(candidate["dEven"].nunique()),
            "stage09_selected_signals": int(candidate["stage09_selected_signal"].sum()),
            "started_1_to_3_candidates": int(candidate["started_1_to_3_pass"].sum()),
            "recovery_candidates": int(candidate["market_recovery"].sum()),
            "deterioration_candidates": int(candidate["market_deterioration"].sum()),
            "combined_started_recovery_candidates": int(
                (candidate["started_1_to_3_pass"] & candidate["market_recovery"]).sum()
            ),
        },
        "breadth": {
            "daily_rows": int(len(daily)),
            "first_date": daily["dEven"].min(),
            "last_date": daily["dEven"].max(),
            "minimum_raw": float(daily["market_breadth_raw"].min()),
            "maximum_raw": float(daily["market_breadth_raw"].max()),
            "minimum_test_coverage_fraction": float(
                daily.loc[daily["partition"].eq("unseen_test"), "breadth_coverage_fraction"].min()
            ),
            "median_test_coverage_fraction": float(
                daily.loc[daily["partition"].eq("unseen_test"), "breadth_coverage_fraction"].median()
            ),
        },
        "analysis_variants": variants,
        "safeguards": {
            "baseline_stage09_selection_reconstructed_exactly": True,
            "labels_used_for_filtering_or_ranking": False,
            "model_retrained": False,
            "scores_recomputed": False,
            "frozen_model_feature_matrix_changed": False,
            "breadth_is_causal": True,
            "portfolio_backtest_performed": False,
            "signal_level_return_interpretation": (
                "gross original Triple-Barrier diagnostic outcome, not executable portfolio return"
            ),
            "confirmatory_claim_allowed": False,
        },
        "outputs": {key: str(value) for key, value in result_paths.items()},
    }

    if write_outputs:
        for path in result_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(result_paths["daily_breadth"], index=False, encoding="utf-8-sig")
        candidate.to_csv(result_paths["candidate_enriched"], index=False, encoding="utf-8-sig")
        selected_variants.to_csv(result_paths["selected_variants"], index=False, encoding="utf-8-sig")
        variant_summary.to_csv(result_paths["variant_summary"], index=False, encoding="utf-8-sig")
        breadth_bin_metrics.to_csv(result_paths["breadth_bin_metrics"], index=False, encoding="utf-8-sig")
        regime_metrics.to_csv(result_paths["regime_metrics"], index=False, encoding="utf-8-sig")
        yearly_summary.to_csv(result_paths["yearly_summary"], index=False, encoding="utf-8-sig")
        pearson.to_csv(result_paths["pearson"], index=False, encoding="utf-8-sig")
        spearman.to_csv(result_paths["spearman"], index=False, encoding="utf-8-sig")
        pd.DataFrame(inventory_rows).to_csv(result_paths["inventory"], index=False, encoding="utf-8-sig")
        started_audit.to_csv(result_paths["started_audit"], index=False, encoding="utf-8-sig")
        index_consistency.to_csv(result_paths["index_consistency"], index=False, encoding="utf-8-sig")
        result_paths["manifest"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    return {
        "manifest": manifest,
        "paths": result_paths,
        "daily_breadth": daily,
        "candidate_enriched": candidate,
        "selected_signals_by_variant": selected_variants,
        "filter_variant_summary": variant_summary,
        "breadth_bin_positive_rate": breadth_bin_metrics,
        "recovery_deterioration_comparison": regime_metrics,
        "yearly_filter_comparison": yearly_summary,
        "correlation_pearson": pearson,
        "correlation_spearman": spearman,
        "started_source_audit": started_audit,
        "market_index_consistency_audit": index_consistency,
    }
