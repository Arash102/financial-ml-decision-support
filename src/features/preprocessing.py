"""Causal, cross-sectionally comparable feature engineering for pooled models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


EMA_WINDOWS = (3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60, 200)

SHORT_EMA_WINDOWS = (3, 5, 8, 10, 12, 15)
LONG_EMA_WINDOWS = (30, 35, 40, 45, 50, 60)

FEATURE_ENGINEERING_SCHEMA_VERSION = "stage04_pooled_v5_pricechange_zigzag_provenance"

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
    """Normalize the immutable raw-data columns required by final features."""
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

    numeric_columns = [
        column
        for column in columns
        if column != "dEven"
    ]
    result = _numeric(result, numeric_columns)

    result = (
        result.dropna(subset=["dEven"])
        .sort_values("dEven", kind="stable")
        .drop_duplicates(subset=["dEven"], keep="last")
        .reset_index(drop=True)
    )
    return result


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
    valid = (
        numeric.notna()
        & rolling_mean.notna()
        & rolling_std.notna()
    )
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
    """
    Build the only intentionally unadjusted-price feature.

    Ordinary bar:
        (unadjusted last - unadjusted open)
        / (unadjusted high - unadjusted low)

    Locked price-limit bar:
        high == low == open == last
        +1 when raw `priceChange` is positive
        -1 when raw `priceChange` is negative
         0 when raw `priceChange` is zero

    Locked-bar direction comes directly from the same-day raw market field
    `priceChange`. No previous-day lookup and no structural-break guard are
    used for this special case.
    """
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

def build_final_feature_frame(
    labeled_train_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    *,
    config: FeatureEngineeringConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Reconstruct the frozen pooled-model feature schema on train-only history.

    The function is deterministic and causal. No target value or future event
    metadata is used to construct a feature.
    """
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
    for optional in ("structural_break", "adjustment_factor"):
        if optional in raw.columns:
            merge_columns.append(optional)

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

    # Price-level EMA values become dimensionless signed distances.
    for window in EMA_WINDOWS:
        ema = merged[f"EMA_{window}"]
        feature_frame[f"ema_{window}_distance"] = _safe_ratio(
            adjusted_last - ema,
            adjusted_last,
        )

    # Bounded momentum.
    feature_frame["rsi_14_centered"] = (
        merged["RSI_14"] - 50.0
    ) / 50.0

    # Price-unit MACD becomes dimensionless.
    feature_frame["macd_relative"] = _safe_ratio(
        merged["macd"],
        adjusted_last,
    )

    # Individual buyer/seller power from raw client-type fields.
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

    # Daily buy-side total volume and its causal 30-observation rolling mean.
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

    # Institutional fractions are reconstructed independently on buy and sell.
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

    # Corrected current-date stock-to-market relative-strength z-score.
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

    # Corrected standard one-observation return and current-date rolling z-score.
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

    # User-confirmed Iran-market microstructure feature on unadjusted prices.
    # Locked-bar direction comes directly from same-day raw `priceChange`.
    body_ratio, body_audit = _build_body_ratio(merged)
    feature_frame["body_ratio"] = body_ratio

    # GMMA state is categorical, not ordinal.
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

    rows.extend(
        [
            {
                "feature": "rsi_14_centered",
                "semantic_group": "momentum",
                "source_feature": "RSI_14",
                "transformation": "(RSI_14 - 50) / 50",
                "unit_before": "bounded index",
                "unit_after": "dimensionless",
                "data_type": "numeric",
                "price_basis": "adjusted",
            },
            {
                "feature": "macd_relative",
                "semantic_group": "momentum",
                "source_feature": "macd",
                "transformation": "macd / adj_last_price",
                "unit_before": "price",
                "unit_after": "dimensionless",
                "data_type": "numeric",
                "price_basis": "adjusted",
            },
            {
                "feature": "log_power_of_buy",
                "semantic_group": "investor_behavior",
                "source_feature": "raw client-type fields",
                "transformation": (
                    "ln((buy_I_Volume/buy_I_Count)"
                    "/(sell_I_Volume/sell_I_Count))"
                ),
                "unit_before": "ratio",
                "unit_after": "log ratio",
                "data_type": "numeric",
                "price_basis": "not price based",
            },
            {
                "feature": "log_volume_ratio_30",
                "semantic_group": "trading_activity",
                "source_feature": "raw buy volumes",
                "transformation": (
                    "log1p(daily_total_volume / causal rolling_mean_30)"
                ),
                "unit_before": "volume ratio",
                "unit_after": "log ratio",
                "data_type": "numeric",
                "price_basis": "not price based",
            },
            {
                "feature": "ho_buy_fraction",
                "semantic_group": "investor_behavior",
                "source_feature": "raw client-type fields",
                "transformation": (
                    "buy_N_Volume/(buy_I_Volume+buy_N_Volume)"
                ),
                "unit_before": "volume",
                "unit_after": "fraction",
                "data_type": "numeric",
                "price_basis": "not price based",
            },
            {
                "feature": "ho_sell_fraction",
                "semantic_group": "investor_behavior",
                "source_feature": "raw client-type fields",
                "transformation": (
                    "sell_N_Volume/(sell_I_Volume+sell_N_Volume)"
                ),
                "unit_before": "volume",
                "unit_after": "fraction",
                "data_type": "numeric",
                "price_basis": "not price based",
            },
            {
                "feature": "x_relative_strength_zscore",
                "semantic_group": "relative_market_position",
                "source_feature": "adjusted price and market index",
                "transformation": (
                    "current stock/index ratio z-score over 12 observations"
                ),
                "unit_before": "ratio",
                "unit_after": "z-score",
                "data_type": "numeric",
                "price_basis": "adjusted stock price",
            },
            {
                "feature": "y_return_zscore",
                "semantic_group": "relative_market_position",
                "source_feature": "adj_last_price",
                "transformation": (
                    "current standard return z-score over 12 returns"
                ),
                "unit_before": "return",
                "unit_after": "z-score",
                "data_type": "numeric",
                "price_basis": "adjusted",
            },
            {
                "feature": "log_positive_rs_run_length",
                "semantic_group": "relative_market_position",
                "source_feature": "x_relative_strength_zscore",
                "transformation": (
                    "log1p(consecutive observations with x > 0)"
                ),
                "unit_before": "observation count",
                "unit_after": "log count",
                "data_type": "numeric",
                "price_basis": "derived",
            },
            {
                "feature": "body_ratio",
                "semantic_group": "price_action",
                "source_feature": (
                    "pDrCotVal, priceFirst, priceMax, priceMin, priceChange"
                ),
                "transformation": (
                    "unadjusted signed body/range; locked-bar direction "
                    "from same-day priceChange (+1/-1/0)"
                ),
                "unit_before": "unadjusted price",
                "unit_after": "dimensionless",
                "data_type": "numeric",
                "price_basis": "unadjusted",
            },
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
