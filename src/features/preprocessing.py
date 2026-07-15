"""Causal, cross-sectionally comparable feature engineering for pooled models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


EMA_WINDOWS = (3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60, 200)
SHORT_EMA_WINDOWS = (3, 5, 8, 10, 12, 15)
LONG_EMA_WINDOWS = (30, 35, 40, 45, 50, 60)

FEATURE_ENGINEERING_SCHEMA_VERSION = (
    "stage04_pooled_v7_market_regime_features"
)

RAW_REQUIRED_COLUMNS = (
    "dEven",
    "buy_I_Volume",
    "buy_I_Count",
    "sell_I_Volume",
    "sell_I_Count",
    "buy_N_Volume",
    "sell_N_Volume",
    "pDrCotVal",
    "priceFirst",
    "priceMax",
    "priceMin",
    "priceChange",
    "xNivInuClMresIbs",
)

MARKET_INDEX_REQUIRED_COLUMNS = (
    "dEven",
    "xNivInuClMresIbs",
    "xNivInuPbMresIbs",
    "xNivInuPhMresIbs",
)

MARKET_INDEX_VALUE_COLUMNS = (
    "xNivInuClMresIbs",
    "xNivInuPbMresIbs",
    "xNivInuPhMresIbs",
)

MARKET_REGIME_NUMERIC_FEATURES = (
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

ENGINEERED_NUMERIC_FEATURES = tuple(
    [f"ema_{window}_distance" for window in EMA_WINDOWS]
    + [
        "rsi_14_centered",
        "macd_relative",
        "log_power_of_buy",
        "log_volume_ratio_30",
        "ho_buy_fraction",
        "ho_sell_fraction",
        "x_relative_strength_zscore",
        "y_return_zscore",
        "log_positive_rs_run_length",
        "body_ratio",
    ]
    + list(MARKET_REGIME_NUMERIC_FEATURES)
)

CARRIED_STAGE04_NUMERIC_FEATURES = (
    "distance_above_confirmed_low_fraction",
    "distance_below_confirmed_high_fraction",
)

FINAL_NUMERIC_FEATURES = (
    ENGINEERED_NUMERIC_FEATURES
    + CARRIED_STAGE04_NUMERIC_FEATURES
)

FINAL_CATEGORICAL_FEATURES = ("gmma_state",)

ENGINEERED_MODEL_FEATURES = (
    ENGINEERED_NUMERIC_FEATURES
    + FINAL_CATEGORICAL_FEATURES
)

FINAL_MODEL_FEATURES = (
    FINAL_NUMERIC_FEATURES
    + FINAL_CATEGORICAL_FEATURES
)


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Frozen deterministic feature-engineering parameters."""

    relative_strength_window: int = 12
    return_zscore_window: int = 12
    volume_window: int = 30
    market_volatility_window: int = 20
    market_ema_fast_window: int = 20
    market_ema_slow_window: int = 60
    market_drawdown_window: int = 60
    market_index_consistency_relative_tolerance: float = 1e-10


def parse_market_date(series: pd.Series) -> pd.Series:
    """Parse YYYYMMDD market dates and conventional datetime strings."""
    raw = series.astype("string").str.strip()
    numeric_like = raw.str.fullmatch(r"\d{8}", na=False)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if numeric_like.any():
        parsed.loc[numeric_like] = pd.to_datetime(
            raw.loc[numeric_like],
            format="%Y%m%d",
            errors="coerce",
        )

    if (~numeric_like).any():
        parsed.loc[~numeric_like] = pd.to_datetime(
            raw.loc[~numeric_like],
            errors="coerce",
        )

    return parsed


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def prepare_raw_feature_source(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize immutable stock-level raw columns required by final features."""
    missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(raw_frame.columns))
    if missing:
        raise KeyError(f"Raw feature source is missing columns: {missing}")

    optional_columns = [
        column
        for column in ("structural_break", "adjustment_factor")
        if column in raw_frame.columns
    ]

    columns = list(RAW_REQUIRED_COLUMNS) + optional_columns
    result = raw_frame[columns].copy()
    result["dEven"] = parse_market_date(result["dEven"])

    numeric_columns = [column for column in columns if column != "dEven"]
    result = _numeric(result, numeric_columns)

    return (
        result.dropna(subset=["dEven"])
        .sort_values("dEven", kind="stable")
        .drop_duplicates(subset=["dEven"], keep="last")
        .reset_index(drop=True)
    )


def prepare_market_index_source(
    raw_frame: pd.DataFrame,
    *,
    source_symbol: str,
) -> pd.DataFrame:
    """Extract equal-weight market-index observations from one raw symbol file."""
    missing = sorted(
        set(MARKET_INDEX_REQUIRED_COLUMNS) - set(raw_frame.columns)
    )
    if missing:
        raise KeyError(
            f"Market-index source is missing columns: {missing}"
        )

    result = raw_frame[list(MARKET_INDEX_REQUIRED_COLUMNS)].copy()
    result["dEven"] = parse_market_date(result["dEven"])
    result = _numeric(result, MARKET_INDEX_VALUE_COLUMNS)
    result["source_symbol"] = str(source_symbol)

    return (
        result.dropna(subset=["dEven"])
        .sort_values("dEven", kind="stable")
        .drop_duplicates(subset=["dEven"], keep="last")
        .reset_index(drop=True)
    )


def build_canonical_market_index(
    market_observations: pd.DataFrame,
    *,
    relative_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pool repeated index observations before calculating rolling market features.

    A stock file may omit dates when that individual security did not trade.
    Therefore market returns/volatility cannot be calculated independently on
    each stock's active-date sequence.
    """
    required = {
        "dEven",
        "source_symbol",
        *MARKET_INDEX_VALUE_COLUMNS,
    }
    missing = sorted(required - set(market_observations.columns))
    if missing:
        raise KeyError(
            f"Market observation panel is missing columns: {missing}"
        )

    panel = market_observations.copy()
    panel["dEven"] = parse_market_date(panel["dEven"])
    panel = _numeric(panel, MARKET_INDEX_VALUE_COLUMNS)
    panel = panel.dropna(subset=["dEven"]).reset_index(drop=True)

    audit_parts: list[pd.DataFrame] = []
    canonical_parts: list[pd.Series] = []
    grouped = panel.groupby("dEven", sort=True)

    for column in MARKET_INDEX_VALUE_COLUMNS:
        stats = grouped[column].agg(
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
        stats["market_index_field"] = column
        stats = stats.reset_index()

        audit_parts.append(stats)
        canonical_parts.append(
            stats.set_index("dEven")["canonical_value"].rename(column)
        )

    consistency_audit = pd.concat(
        audit_parts,
        ignore_index=True,
    ).sort_values(
        ["dEven", "market_index_field"],
        kind="stable",
    ).reset_index(drop=True)

    canonical = pd.concat(
        canonical_parts,
        axis=1,
    ).reset_index()
    canonical = canonical.sort_values(
        "dEven",
        kind="stable",
    ).reset_index(drop=True)

    return canonical, consistency_audit


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    require_positive_denominator: bool = True,
) -> pd.Series:
    numerator_values = pd.to_numeric(numerator, errors="coerce")
    denominator_values = pd.to_numeric(denominator, errors="coerce")

    valid = np.isfinite(numerator_values) & np.isfinite(denominator_values)
    if require_positive_denominator:
        valid &= denominator_values > 0
    else:
        valid &= denominator_values != 0

    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    result.loc[valid] = (
        numerator_values.loc[valid] / denominator_values.loc[valid]
    )
    return result


def _rolling_population_zscore(
    values: pd.Series,
    *,
    window: int,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    rolling_mean = numeric.rolling(
        window=window,
        min_periods=window,
    ).mean()
    rolling_std = numeric.rolling(
        window=window,
        min_periods=window,
    ).std(ddof=0)

    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.notna() & rolling_mean.notna() & rolling_std.notna()
    nonzero_std = valid & rolling_std.gt(0)
    zero_std = valid & rolling_std.eq(0)

    result.loc[nonzero_std] = (
        numeric.loc[nonzero_std] - rolling_mean.loc[nonzero_std]
    ) / rolling_std.loc[nonzero_std]
    result.loc[zero_std] = 0.0
    return result


def _positive_run_length(values: pd.Series) -> pd.Series:
    """Consecutive observations with positive relative-strength z-score."""
    numeric = pd.to_numeric(values, errors="coerce")
    output = np.zeros(len(numeric), dtype=np.int64)
    run_length = 0

    for position, value in enumerate(numeric.to_numpy(dtype=float)):
        if np.isfinite(value) and value > 0:
            run_length += 1
        else:
            run_length = 0
        output[position] = run_length

    return pd.Series(output, index=values.index, dtype=np.int64)


def _build_body_ratio(
    frame: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int]]:
    """Build the intentionally unadjusted-price stock candle feature."""
    raw_last = pd.to_numeric(frame["pDrCotVal"], errors="coerce")
    raw_open = pd.to_numeric(frame["priceFirst"], errors="coerce")
    raw_high = pd.to_numeric(frame["priceMax"], errors="coerce")
    raw_low = pd.to_numeric(frame["priceMin"], errors="coerce")
    price_change = pd.to_numeric(frame["priceChange"], errors="coerce")

    finite = (
        np.isfinite(raw_last)
        & np.isfinite(raw_open)
        & np.isfinite(raw_high)
        & np.isfinite(raw_low)
    )
    positive_prices = (
        raw_last.gt(0)
        & raw_open.gt(0)
        & raw_high.gt(0)
        & raw_low.gt(0)
    )
    ordered_range = raw_high.ge(raw_low)
    inside_range = (
        raw_open.ge(raw_low)
        & raw_open.le(raw_high)
        & raw_last.ge(raw_low)
        & raw_last.le(raw_high)
    )
    valid_ohlc = finite & positive_prices & ordered_range & inside_range

    locked = (
        valid_ohlc
        & raw_high.eq(raw_low)
        & raw_open.eq(raw_last)
        & raw_high.eq(raw_last)
    )
    ordinary = valid_ohlc & raw_high.gt(raw_low)

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[ordinary] = (
        raw_last.loc[ordinary] - raw_open.loc[ordinary]
    ) / (
        raw_high.loc[ordinary] - raw_low.loc[ordinary]
    )

    locked_with_change = locked & price_change.notna()
    locked_up = locked_with_change & price_change.gt(0)
    locked_down = locked_with_change & price_change.lt(0)
    locked_unchanged = locked_with_change & price_change.eq(0)

    result.loc[locked_up] = 1.0
    result.loc[locked_down] = -1.0
    result.loc[locked_unchanged] = 0.0

    outside_bounds = result.notna() & (
        result.lt(-1.0 - 1e-12)
        | result.gt(1.0 + 1e-12)
    )
    if outside_bounds.any():
        raise AssertionError(
            "body_ratio left the theoretical [-1, 1] interval."
        )

    audit = {
        "body_valid_ordinary_rows": int(ordinary.sum()),
        "body_locked_rows": int(locked.sum()),
        "body_locked_up_rows": int(locked_up.sum()),
        "body_locked_down_rows": int(locked_down.sum()),
        "body_locked_unchanged_rows": int(locked_unchanged.sum()),
        "body_locked_price_change_missing_rows": int(
            (locked & price_change.isna()).sum()
        ),
        "body_invalid_ohlc_rows": int((~valid_ohlc).sum()),
        "body_missing_rows": int(result.isna().sum()),
    }
    return result, audit


def build_market_regime_feature_frame(
    canonical_market_index: pd.DataFrame,
    *,
    config: FeatureEngineeringConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build nine causal equal-weight market-regime features."""
    config = config or FeatureEngineeringConfig()

    required = {"dEven", *MARKET_INDEX_VALUE_COLUMNS}
    missing = sorted(required - set(canonical_market_index.columns))
    if missing:
        raise KeyError(
            f"Canonical market index is missing columns: {missing}"
        )

    market = canonical_market_index[
        ["dEven", *MARKET_INDEX_VALUE_COLUMNS]
    ].copy()
    market["dEven"] = parse_market_date(market["dEven"])
    market = _numeric(market, MARKET_INDEX_VALUE_COLUMNS)
    market = (
        market.dropna(subset=["dEven"])
        .sort_values("dEven", kind="stable")
        .drop_duplicates(subset=["dEven"], keep="last")
        .reset_index(drop=True)
    )

    close = market["xNivInuClMresIbs"]
    low = market["xNivInuPbMresIbs"]
    high = market["xNivInuPhMresIbs"]

    feature_frame = pd.DataFrame({"dEven": market["dEven"]})

    feature_frame["market_return_1"] = close.pct_change(
        periods=1,
        fill_method=None,
    )
    feature_frame["market_return_5"] = close.pct_change(
        periods=5,
        fill_method=None,
    )
    feature_frame["market_return_20"] = close.pct_change(
        periods=20,
        fill_method=None,
    )

    previous_close = close.shift(1)
    log_return = pd.Series(np.nan, index=market.index, dtype=float)
    valid_log_return = (
        close.gt(0)
        & previous_close.gt(0)
        & np.isfinite(close)
        & np.isfinite(previous_close)
    )
    log_return.loc[valid_log_return] = np.log(
        close.loc[valid_log_return]
        / previous_close.loc[valid_log_return]
    )

    feature_frame["market_volatility_20"] = log_return.rolling(
        window=config.market_volatility_window,
        min_periods=config.market_volatility_window,
    ).std(ddof=0)

    ema_fast = close.ewm(
        span=config.market_ema_fast_window,
        adjust=False,
        min_periods=config.market_ema_fast_window,
    ).mean()
    ema_slow = close.ewm(
        span=config.market_ema_slow_window,
        adjust=False,
        min_periods=config.market_ema_slow_window,
    ).mean()

    feature_frame["market_ema_20_distance"] = _safe_ratio(
        close - ema_fast,
        close,
    )
    feature_frame["market_ema_60_distance"] = _safe_ratio(
        close - ema_slow,
        close,
    )

    valid_market_ohlc = (
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

    feature_frame["market_range_fraction"] = np.nan
    feature_frame.loc[
        valid_market_ohlc,
        "market_range_fraction",
    ] = (
        high.loc[valid_market_ohlc]
        - low.loc[valid_market_ohlc]
    ) / close.loc[valid_market_ohlc]

    close_location = pd.Series(
        np.nan,
        index=market.index,
        dtype=float,
    )

    ordinary_range = valid_market_ohlc & high.gt(low)
    close_location.loc[ordinary_range] = (
        close.loc[ordinary_range] - low.loc[ordinary_range]
    ) / (
        high.loc[ordinary_range] - low.loc[ordinary_range]
    )

    locked_market = (
        valid_market_ohlc
        & high.eq(low)
        & high.eq(close)
    )
    locked_with_previous = locked_market & previous_close.notna()
    locked_up = locked_with_previous & close.gt(previous_close)
    locked_down = locked_with_previous & close.lt(previous_close)
    locked_equal_previous = (
        locked_with_previous & close.eq(previous_close)
    )

    close_location.loc[locked_up] = 1.0
    close_location.loc[locked_down] = 0.0
    close_location.loc[locked_equal_previous] = np.nan

    feature_frame["market_close_location"] = close_location

    rolling_max = close.rolling(
        window=config.market_drawdown_window,
        min_periods=config.market_drawdown_window,
    ).max()
    feature_frame["market_drawdown_60"] = _safe_ratio(
        close,
        rolling_max,
    ) - 1.0

    close_location_out_of_bounds = (
        feature_frame["market_close_location"].notna()
        & (
            feature_frame["market_close_location"].lt(-1e-12)
            | feature_frame["market_close_location"].gt(1.0 + 1e-12)
        )
    )
    if close_location_out_of_bounds.any():
        raise AssertionError(
            "market_close_location left the [0, 1] interval."
        )

    range_negative = (
        feature_frame["market_range_fraction"].notna()
        & feature_frame["market_range_fraction"].lt(-1e-12)
    )
    if range_negative.any():
        raise AssertionError(
            "market_range_fraction contains negative values."
        )

    audit: dict[str, Any] = {
        "market_calendar_rows": len(market),
        "market_first_date": market["dEven"].min(),
        "market_last_date": market["dEven"].max(),
        "invalid_market_ohlc_rows": int((~valid_market_ohlc).sum()),
        "locked_market_rows": int(locked_market.sum()),
        "locked_market_up_rows": int(locked_up.sum()),
        "locked_market_down_rows": int(locked_down.sum()),
        "locked_market_equal_previous_rows": int(
            locked_equal_previous.sum()
        ),
        "market_close_location_missing_rows": int(
            feature_frame["market_close_location"].isna().sum()
        ),
    }

    for feature in MARKET_REGIME_NUMERIC_FEATURES:
        audit[f"{feature}_missing_rows"] = int(
            feature_frame[feature].isna().sum()
        )

    return feature_frame, audit


def build_final_feature_frame(
    labeled_train_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    market_feature_frame: pd.DataFrame,
    *,
    config: FeatureEngineeringConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build final stock and market features on train-only history."""
    config = config or FeatureEngineeringConfig()

    labeled = labeled_train_frame.copy()
    labeled["dEven"] = parse_market_date(labeled["dEven"])
    labeled = (
        labeled.dropna(subset=["dEven"])
        .sort_values("dEven", kind="stable")
        .drop_duplicates(subset=["dEven"], keep="last")
        .reset_index(drop=True)
    )

    required_labeled_columns = {
        "dEven",
        "adj_last_price",
        "RSI_14",
        "macd",
        *[f"EMA_{window}" for window in EMA_WINDOWS],
    }
    missing_labeled = sorted(required_labeled_columns - set(labeled.columns))
    if missing_labeled:
        raise KeyError(
            f"Labeled train source is missing columns: {missing_labeled}"
        )

    raw = prepare_raw_feature_source(raw_frame)

    market_features = market_feature_frame.copy()
    market_features["dEven"] = parse_market_date(
        market_features["dEven"]
    )
    missing_market_features = sorted(
        set(MARKET_REGIME_NUMERIC_FEATURES)
        - set(market_features.columns)
    )
    if missing_market_features:
        raise KeyError(
            "Market feature frame is missing columns: "
            f"{missing_market_features}"
        )
    market_features = (
        market_features[
            ["dEven", *MARKET_REGIME_NUMERIC_FEATURES]
        ]
        .dropna(subset=["dEven"])
        .sort_values("dEven", kind="stable")
        .drop_duplicates(subset=["dEven"], keep="last")
        .reset_index(drop=True)
    )

    merge_columns = [
        "dEven",
        "buy_I_Volume",
        "buy_I_Count",
        "sell_I_Volume",
        "sell_I_Count",
        "buy_N_Volume",
        "sell_N_Volume",
        "pDrCotVal",
        "priceFirst",
        "priceMax",
        "priceMin",
        "priceChange",
        "xNivInuClMresIbs",
    ]

    merged = labeled.merge(
        raw[merge_columns],
        on="dEven",
        how="left",
        validate="one_to_one",
        indicator="_raw_join",
    )
    raw_join_missing_rows = int(merged["_raw_join"].ne("both").sum())
    merged = merged.drop(columns=["_raw_join"])

    numeric_source_columns = [
        "adj_last_price",
        "RSI_14",
        "macd",
        "buy_I_Volume",
        "buy_I_Count",
        "sell_I_Volume",
        "sell_I_Count",
        "buy_N_Volume",
        "sell_N_Volume",
        "pDrCotVal",
        "priceFirst",
        "priceMax",
        "priceMin",
        "priceChange",
        "xNivInuClMresIbs",
        *[f"EMA_{window}" for window in EMA_WINDOWS],
    ]
    merged = _numeric(merged, numeric_source_columns)

    feature_frame = pd.DataFrame(
        {"dEven": merged["dEven"]},
        index=merged.index,
    )
    adjusted_last = merged["adj_last_price"]

    for window in EMA_WINDOWS:
        feature_frame[f"ema_{window}_distance"] = _safe_ratio(
            adjusted_last - merged[f"EMA_{window}"],
            adjusted_last,
        )

    feature_frame["rsi_14_centered"] = (
        merged["RSI_14"] - 50.0
    ) / 50.0

    feature_frame["macd_relative"] = _safe_ratio(
        merged["macd"],
        adjusted_last,
    )

    valid_power = (
        merged["buy_I_Volume"].gt(0)
        & merged["buy_I_Count"].gt(0)
        & merged["sell_I_Volume"].gt(0)
        & merged["sell_I_Count"].gt(0)
    )
    power_ratio = pd.Series(np.nan, index=merged.index, dtype=float)
    power_ratio.loc[valid_power] = (
        (
            merged.loc[valid_power, "buy_I_Volume"]
            / merged.loc[valid_power, "buy_I_Count"]
        )
        /
        (
            merged.loc[valid_power, "sell_I_Volume"]
            / merged.loc[valid_power, "sell_I_Count"]
        )
    )
    valid_positive_power = power_ratio.gt(0) & np.isfinite(power_ratio)
    feature_frame["log_power_of_buy"] = np.nan
    feature_frame.loc[
        valid_positive_power,
        "log_power_of_buy",
    ] = np.log(power_ratio.loc[valid_positive_power])

    daily_total_volume = (
        merged["buy_I_Volume"] + merged["buy_N_Volume"]
    )
    rolling_volume_mean = daily_total_volume.rolling(
        window=config.volume_window,
        min_periods=config.volume_window,
    ).mean()
    volume_ratio = _safe_ratio(
        daily_total_volume,
        rolling_volume_mean,
    )
    valid_volume_ratio = volume_ratio.ge(0) & np.isfinite(volume_ratio)
    feature_frame["log_volume_ratio_30"] = np.nan
    feature_frame.loc[
        valid_volume_ratio,
        "log_volume_ratio_30",
    ] = np.log1p(volume_ratio.loc[valid_volume_ratio])

    buy_total = merged["buy_I_Volume"] + merged["buy_N_Volume"]
    sell_total = merged["sell_I_Volume"] + merged["sell_N_Volume"]

    feature_frame["ho_buy_fraction"] = _safe_ratio(
        merged["buy_N_Volume"],
        buy_total,
    )
    feature_frame["ho_sell_fraction"] = _safe_ratio(
        merged["sell_N_Volume"],
        sell_total,
    )

    relative_strength_ratio = _safe_ratio(
        adjusted_last,
        merged["xNivInuClMresIbs"],
    )
    feature_frame["x_relative_strength_zscore"] = (
        _rolling_population_zscore(
            relative_strength_ratio,
            window=config.relative_strength_window,
        )
    )

    daily_return = adjusted_last.pct_change(fill_method=None)
    feature_frame["y_return_zscore"] = _rolling_population_zscore(
        daily_return,
        window=config.return_zscore_window,
    )

    positive_run_length = _positive_run_length(
        feature_frame["x_relative_strength_zscore"]
    )
    feature_frame["log_positive_rs_run_length"] = np.log1p(
        positive_run_length.astype(float)
    )

    body_ratio, body_audit = _build_body_ratio(merged)
    feature_frame["body_ratio"] = body_ratio

    market_alignment = feature_frame[["dEven"]].merge(
        market_features,
        on="dEven",
        how="left",
        validate="one_to_one",
        indicator="_market_join",
    )
    market_feature_join_missing_rows = int(
        market_alignment["_market_join"].ne("both").sum()
    )
    market_alignment = market_alignment.drop(columns=["_market_join"])

    for feature in MARKET_REGIME_NUMERIC_FEATURES:
        feature_frame[feature] = market_alignment[feature].to_numpy()

    short_columns = [f"EMA_{window}" for window in SHORT_EMA_WINDOWS]
    long_columns = [f"EMA_{window}" for window in LONG_EMA_WINDOWS]
    short = merged[short_columns]
    long = merged[long_columns]

    complete_gmma = short.notna().all(axis=1) & long.notna().all(axis=1)
    bullish = complete_gmma & short.min(axis=1).gt(long.max(axis=1))
    bearish = complete_gmma & short.max(axis=1).lt(long.min(axis=1))

    gmma_state = pd.Series(pd.NA, index=merged.index, dtype="string")
    gmma_state.loc[complete_gmma] = "mixed"
    gmma_state.loc[bullish] = "bullish"
    gmma_state.loc[bearish] = "bearish"
    feature_frame["gmma_state"] = gmma_state

    audit: dict[str, Any] = {
        "rows": len(merged),
        "raw_join_missing_rows": raw_join_missing_rows,
        "market_feature_join_missing_rows": market_feature_join_missing_rows,
        "invalid_power_source_rows": int((~valid_power).sum()),
        "power_feature_missing_rows": int(
            feature_frame["log_power_of_buy"].isna().sum()
        ),
        "volume_feature_missing_rows": int(
            feature_frame["log_volume_ratio_30"].isna().sum()
        ),
        "ho_buy_missing_rows": int(
            feature_frame["ho_buy_fraction"].isna().sum()
        ),
        "ho_sell_missing_rows": int(
            feature_frame["ho_sell_fraction"].isna().sum()
        ),
        "x_missing_rows": int(
            feature_frame["x_relative_strength_zscore"].isna().sum()
        ),
        "y_missing_rows": int(
            feature_frame["y_return_zscore"].isna().sum()
        ),
        "gmma_missing_rows": int(
            feature_frame["gmma_state"].isna().sum()
        ),
        **body_audit,
    }

    return feature_frame, audit


def final_feature_schema() -> pd.DataFrame:
    """Return the frozen semantic feature manifest."""
    rows: list[dict[str, object]] = []

    for window in EMA_WINDOWS:
        rows.append(
            {
                "feature": f"ema_{window}_distance",
                "semantic_group": "trend_location",
                "source_feature": f"EMA_{window}",
                "transformation": (
                    f"(adj_last_price - EMA_{window}) / adj_last_price"
                ),
                "unit_before": "price",
                "unit_after": "dimensionless",
                "data_type": "numeric",
                "price_basis": "adjusted",
            }
        )

    stock_rows = [
        ("rsi_14_centered", "momentum", "RSI_14", "(RSI_14 - 50) / 50", "bounded index", "dimensionless", "adjusted"),
        ("macd_relative", "momentum", "macd", "macd / adj_last_price", "price", "dimensionless", "adjusted"),
        ("log_power_of_buy", "investor_behavior", "client-type fields from collection pipeline", "ln((buy_I_Volume/buy_I_Count)/(sell_I_Volume/sell_I_Count))", "ratio", "log ratio", "not price based"),
        ("log_volume_ratio_30", "trading_activity", "raw buy volumes", "log1p(daily_total_volume / causal rolling_mean_30)", "volume ratio", "log ratio", "not price based"),
        ("ho_buy_fraction", "investor_behavior", "raw client-type fields", "buy_N_Volume/(buy_I_Volume+buy_N_Volume)", "volume", "fraction", "not price based"),
        ("ho_sell_fraction", "investor_behavior", "raw client-type fields", "sell_N_Volume/(sell_I_Volume+sell_N_Volume)", "volume", "fraction", "not price based"),
        ("x_relative_strength_zscore", "relative_market_position", "adjusted price and equal-weight index close", "current stock/index ratio z-score over 12 observations", "ratio", "z-score", "adjusted stock price"),
        ("y_return_zscore", "relative_market_position", "adj_last_price", "current standard return z-score over 12 returns", "return", "z-score", "adjusted"),
        ("log_positive_rs_run_length", "relative_market_position", "x_relative_strength_zscore", "log1p(consecutive observations with x > 0)", "observation count", "log count", "derived"),
        ("body_ratio", "price_action", "pDrCotVal, priceFirst, priceMax, priceMin, priceChange", "unadjusted signed body/range; locked-bar direction from same-day priceChange (+1/-1/0)", "unadjusted price", "dimensionless", "unadjusted"),
    ]
    for feature, group, source, transform, unit_before, unit_after, basis in stock_rows:
        rows.append(
            {
                "feature": feature,
                "semantic_group": group,
                "source_feature": source,
                "transformation": transform,
                "unit_before": unit_before,
                "unit_after": unit_after,
                "data_type": "numeric",
                "price_basis": basis,
            }
        )

    market_rows = [
        ("market_return_1", "equal-weight index close", "I_t / I_(t-1) - 1", "return"),
        ("market_return_5", "equal-weight index close", "I_t / I_(t-5) - 1", "return"),
        ("market_return_20", "equal-weight index close", "I_t / I_(t-20) - 1", "return"),
        ("market_volatility_20", "equal-weight index close", "population std of 20 causal one-day log returns", "log-return volatility"),
        ("market_ema_20_distance", "equal-weight index close", "(I_t - EMA20_t) / I_t", "dimensionless"),
        ("market_ema_60_distance", "equal-weight index close", "(I_t - EMA60_t) / I_t", "dimensionless"),
        ("market_range_fraction", "equal-weight index high/low/close", "(index_high - index_low) / index_close", "fraction"),
        ("market_close_location", "equal-weight index high/low/close", "(close-low)/(high-low); locked high==low==close: up vs previous close => 1, down => 0", "bounded location"),
        ("market_drawdown_60", "equal-weight index close", "I_t / rolling_max_60(I) - 1", "drawdown fraction"),
    ]
    for feature, source, transform, unit_after in market_rows:
        rows.append(
            {
                "feature": feature,
                "semantic_group": "market_regime",
                "source_feature": source,
                "transformation": transform,
                "unit_before": "equal-weight index level",
                "unit_after": unit_after,
                "data_type": "numeric",
                "price_basis": "market index",
            }
        )

    rows.extend(
        [
            {
                "feature": "gmma_state",
                "semantic_group": "ema_structure",
                "source_feature": "short and long EMA ribbons",
                "transformation": "bullish / bearish / mixed",
                "unit_before": "price ordering",
                "unit_after": "categorical state",
                "data_type": "categorical",
                "price_basis": "adjusted",
            },
            {
                "feature": "distance_above_confirmed_low_fraction",
                "semantic_group": "confirmed_zigzag_geometry",
                "source_feature": "Stage 04 causal ZigZag",
                "transformation": "carried from confirmed ZigZag state",
                "unit_before": "price",
                "unit_after": "fraction",
                "data_type": "numeric",
                "price_basis": "adjusted",
            },
            {
                "feature": "distance_below_confirmed_high_fraction",
                "semantic_group": "confirmed_zigzag_geometry",
                "source_feature": "Stage 04 causal ZigZag",
                "transformation": "carried from confirmed ZigZag state",
                "unit_before": "price",
                "unit_after": "fraction",
                "data_type": "numeric",
                "price_basis": "adjusted",
            },
        ]
    )

    schema = pd.DataFrame(rows)
    schema.insert(0, "feature_order", range(1, len(schema) + 1))
    schema["approved_for_pooled_model"] = True
    return schema
