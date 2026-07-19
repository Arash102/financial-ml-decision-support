"""Causal unseen-period helpers for the Stage 04 breadth feature extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.stage04_breadth_extension import (
    STAGE04_BREADTH_FEATURES,
    Stage04BreadthConfig,
    build_daily_market_breadth,
    parse_market_date,
)


UNSEEN_BREADTH_SCHEMA_VERSION = (
    "stage09_v4_causal_stage04_breadth_reconstruction"
)


def prepare_symbol_breadth_observations(
    raw_path: Path,
    *,
    symbol: str,
    horizon_end: pd.Timestamp,
    config: Stage04BreadthConfig | None = None,
) -> pd.DataFrame:
    """Build previous-valid adjusted-close return signs for one symbol."""
    config = config or Stage04BreadthConfig()

    frame = pd.read_csv(
        raw_path,
        usecols=[
            config.date_column,
            config.adjusted_close_column,
        ],
        low_memory=False,
    )
    frame[config.date_column] = parse_market_date(
        frame[config.date_column]
    )
    frame[config.adjusted_close_column] = pd.to_numeric(
        frame[config.adjusted_close_column],
        errors="coerce",
    )
    frame = (
        frame.dropna(subset=[config.date_column])
        .sort_values(config.date_column, kind="stable")
        .drop_duplicates(
            subset=[config.date_column],
            keep="last",
        )
        .reset_index(drop=True)
    )
    frame = frame.loc[
        frame[config.date_column].le(horizon_end)
    ].copy()

    valid_price = (
        np.isfinite(frame[config.adjusted_close_column])
        & frame[config.adjusted_close_column].gt(0)
    )
    frame = frame.loc[valid_price].copy()
    frame["symbol_return"] = frame[
        config.adjusted_close_column
    ].pct_change(fill_method=None)
    frame["symbol"] = str(symbol)

    return frame[
        [config.date_column, "symbol", "symbol_return"]
    ].dropna(subset=["symbol_return"]).reset_index(drop=True)


def load_started_run_length(
    *,
    symbol: str,
    raw_path: Path,
    fallback_path: Path | None,
    horizon_end: pd.Timestamp,
    config: Stage04BreadthConfig | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load the original nonnegative started run length without recomputation."""
    config = config or Stage04BreadthConfig()

    candidates: list[tuple[str, Path]] = [
        ("raw_data", raw_path),
    ]
    if fallback_path is not None:
        candidates.append(("labeled_unseen_test", fallback_path))

    for source_name, source_path in candidates:
        if not source_path.exists():
            continue

        header = pd.read_csv(source_path, nrows=0)
        required = {
            config.date_column,
            config.started_column,
        }
        if not required.issubset(header.columns):
            continue

        frame = pd.read_csv(
            source_path,
            usecols=[
                config.date_column,
                config.started_column,
            ],
            low_memory=False,
        )
        frame[config.date_column] = parse_market_date(
            frame[config.date_column]
        )
        frame["started_run_length"] = pd.to_numeric(
            frame[config.started_column],
            errors="coerce",
        )
        frame = (
            frame.dropna(subset=[config.date_column])
            .sort_values(config.date_column, kind="stable")
            .drop_duplicates(
                subset=[config.date_column],
                keep="last",
            )
            .reset_index(drop=True)
        )
        frame = frame.loc[
            frame[config.date_column].le(horizon_end)
        ].copy()

        if frame["started_run_length"].lt(0).any():
            raise AssertionError(
                f"{symbol}: original started contains negative values."
            )

        return (
            frame[
                [
                    config.date_column,
                    "started_run_length",
                ]
            ],
            source_name,
        )

    raise KeyError(
        f"{symbol}: no valid original started source was found."
    )


def merge_stage04_breadth_features(
    candidate_frame: pd.DataFrame,
    *,
    started_frame: pd.DataFrame,
    breadth_frame: pd.DataFrame,
    config: Stage04BreadthConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge the five Stage 04 additions without changing candidate identity."""
    config = config or Stage04BreadthConfig()

    required_candidate = {
        "event_id",
        "symbol",
        config.date_column,
    }
    missing_candidate = sorted(
        required_candidate - set(candidate_frame.columns)
    )
    if missing_candidate:
        raise KeyError(
            f"Candidate frame missing columns: {missing_candidate}"
        )

    frame = candidate_frame.copy()
    for column in STAGE04_BREADTH_FEATURES:
        if column in frame.columns:
            frame = frame.drop(columns=[column])

    frame[config.date_column] = parse_market_date(
        frame[config.date_column]
    )
    original_event_ids = frame["event_id"].astype(str).tolist()
    original_rows = len(frame)

    enriched = frame.merge(
        started_frame,
        on=config.date_column,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    enriched = enriched.merge(
        breadth_frame[
            [
                config.date_column,
                "market_breadth_raw",
                "market_breadth_ema30",
                "market_breadth_slope5",
                "market_breadth_regime",
            ]
        ],
        on=config.date_column,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if len(enriched) != original_rows:
        raise AssertionError(
            "Candidate row count changed during breadth enrichment."
        )
    if enriched["event_id"].astype(str).tolist() != original_event_ids:
        raise AssertionError(
            "Candidate identity or order changed during breadth enrichment."
        )

    missing_counts = {
        column: int(enriched[column].isna().sum())
        for column in STAGE04_BREADTH_FEATURES
    }
    if any(value != 0 for value in missing_counts.values()):
        raise AssertionError(
            f"Candidate breadth enrichment has missing values: {missing_counts}"
        )

    if enriched["started_run_length"].lt(0).any():
        raise AssertionError(
            "Enriched started_run_length contains negative values."
        )
    if not enriched["market_breadth_raw"].between(
        -1.0,
        1.0,
        inclusive="both",
    ).all():
        raise AssertionError(
            "market_breadth_raw left the theoretical range."
        )

    audit = {
        "candidate_rows": int(len(enriched)),
        "candidate_identity_preserved": True,
        "missing_counts": missing_counts,
        "started_minimum": float(
            enriched["started_run_length"].min()
        ),
        "started_maximum": float(
            enriched["started_run_length"].max()
        ),
        "breadth_raw_minimum": float(
            enriched["market_breadth_raw"].min()
        ),
        "breadth_raw_maximum": float(
            enriched["market_breadth_raw"].max()
        ),
    }
    return enriched, audit
