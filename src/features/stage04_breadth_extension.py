"""Transactional Stage 04 causal market-breadth extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd


STAGE04_BREADTH_SCHEMA_VERSION = (
    "stage04_pooled_v8_causal_market_breadth_extension"
)

STAGE04_BREADTH_NUMERIC_FEATURES = (
    "started_run_length",
    "market_breadth_raw",
    "market_breadth_ema30",
    "market_breadth_slope5",
)

STAGE04_BREADTH_CATEGORICAL_FEATURES = (
    "market_breadth_regime",
)

STAGE04_BREADTH_FEATURES = (
    STAGE04_BREADTH_NUMERIC_FEATURES
    + STAGE04_BREADTH_CATEGORICAL_FEATURES
)


@dataclass(frozen=True)
class Stage04BreadthConfig:
    date_column: str = "dEven"
    adjusted_close_column: str = "adj_last_price"
    started_column: str = "started"
    ema_span: int = 30
    ema_min_periods: int = 30
    slope_lag_market_dates: int = 5
    transition_lower: float = -0.30
    transition_upper: float = 0.30
    output_encoding: str = "utf-8-sig"


def parse_market_date(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    numeric_like = raw.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )
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


def locate_repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (
            (candidate / "raw_data").exists()
            and (candidate / "notebooks").exists()
            and (candidate / "results").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Repository root was not found. Run from inside the project."
    )


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit_sha(repository_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def _symbol_from_candidate_path(path: Path) -> str:
    suffix = "_train_candidates.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected candidate filename: {path.name}")
    return path.name[:-len(suffix)]


def _read_dates(path: Path, date_column: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=[date_column], low_memory=False)
    return parse_market_date(frame[date_column])


def _prepare_symbol_breadth_source(
    raw_path: Path,
    *,
    symbol: str,
    horizon_end: pd.Timestamp,
    config: Stage04BreadthConfig,
) -> pd.DataFrame:
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
        .drop_duplicates(subset=[config.date_column], keep="last")
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


def build_daily_market_breadth(
    observations: pd.DataFrame,
    *,
    config: Stage04BreadthConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or Stage04BreadthConfig()

    required = {
        config.date_column,
        "symbol",
        "symbol_return",
    }
    missing = sorted(required - set(observations.columns))
    if missing:
        raise KeyError(f"Breadth observations missing columns: {missing}")

    panel = observations.copy()
    panel[config.date_column] = parse_market_date(
        panel[config.date_column]
    )
    panel["symbol_return"] = pd.to_numeric(
        panel["symbol_return"],
        errors="coerce",
    )
    panel = (
        panel.dropna(
            subset=[
                config.date_column,
                "symbol",
                "symbol_return",
            ]
        )
        .sort_values(
            [config.date_column, "symbol"],
            kind="stable",
        )
        .drop_duplicates(
            subset=[config.date_column, "symbol"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    returns = panel["symbol_return"]
    panel["breadth_sign"] = np.select(
        [returns.gt(0), returns.lt(0)],
        [1.0, -1.0],
        default=0.0,
    )

    grouped = panel.groupby(config.date_column, sort=True)
    daily = grouped["breadth_sign"].agg(
        breadth_numerator="sum",
        valid_symbol_count="count",
    )
    daily["positive_symbol_count"] = grouped["breadth_sign"].apply(
        lambda values: int((values > 0).sum())
    )
    daily["negative_symbol_count"] = grouped["breadth_sign"].apply(
        lambda values: int((values < 0).sum())
    )
    daily["unchanged_symbol_count"] = grouped["breadth_sign"].apply(
        lambda values: int((values == 0).sum())
    )
    daily = daily.reset_index()

    daily["market_breadth_raw"] = (
        daily["breadth_numerator"]
        / daily["valid_symbol_count"]
    )
    daily["market_breadth_ema30"] = (
        daily["market_breadth_raw"]
        .ewm(
            span=config.ema_span,
            adjust=False,
            min_periods=config.ema_min_periods,
        )
        .mean()
    )
    daily["market_breadth_slope5"] = (
        daily["market_breadth_ema30"]
        - daily["market_breadth_ema30"].shift(
            config.slope_lag_market_dates
        )
    )

    ema = daily["market_breadth_ema30"]
    slope = daily["market_breadth_slope5"]
    transition = ema.between(
        config.transition_lower,
        config.transition_upper,
        inclusive="both",
    )

    regime = pd.Series(
        "warmup_unavailable",
        index=daily.index,
        dtype="string",
    )
    regime.loc[ema.lt(config.transition_lower)] = "broad_decline"
    regime.loc[ema.gt(config.transition_upper)] = "broad_advance"
    regime.loc[
        transition & slope.gt(0) & ema.lt(0)
    ] = "recovery_negative"
    regime.loc[
        transition & slope.gt(0) & ema.ge(0)
    ] = "recovery_positive"
    regime.loc[
        transition & slope.lt(0)
    ] = "deterioration"
    regime.loc[
        transition & slope.eq(0)
    ] = "neutral_transition"
    daily["market_breadth_regime"] = regime

    if not daily["market_breadth_raw"].between(
        -1.0,
        1.0,
        inclusive="both",
    ).all():
        raise AssertionError(
            "market_breadth_raw left the theoretical [-1, 1] range."
        )

    count_identity = (
        daily["positive_symbol_count"]
        + daily["negative_symbol_count"]
        + daily["unchanged_symbol_count"]
    ).eq(daily["valid_symbol_count"])
    if not count_identity.all():
        raise AssertionError(
            "Breadth counts do not sum to the denominator."
        )

    audit = {
        "calendar_rows": int(len(daily)),
        "first_date": daily[config.date_column].min(),
        "last_date": daily[config.date_column].max(),
        "minimum_valid_symbols": int(
            daily["valid_symbol_count"].min()
        ),
        "maximum_valid_symbols": int(
            daily["valid_symbol_count"].max()
        ),
        "raw_missing_rows": int(
            daily["market_breadth_raw"].isna().sum()
        ),
        "ema30_missing_rows": int(
            daily["market_breadth_ema30"].isna().sum()
        ),
        "slope5_missing_rows": int(
            daily["market_breadth_slope5"].isna().sum()
        ),
        "warmup_regime_rows": int(
            daily["market_breadth_regime"]
            .eq("warmup_unavailable")
            .sum()
        ),
    }

    output_columns = [
        config.date_column,
        "positive_symbol_count",
        "negative_symbol_count",
        "unchanged_symbol_count",
        "valid_symbol_count",
        "market_breadth_raw",
        "market_breadth_ema30",
        "market_breadth_slope5",
        "market_breadth_regime",
    ]
    return daily[output_columns].copy(), audit


def _load_started_source(
    *,
    symbol: str,
    repository_root: Path,
    raw_path: Path,
    config: Stage04BreadthConfig,
) -> tuple[pd.DataFrame, str]:
    candidate_sources = [
        ("raw_data", raw_path),
        (
            "labeled_train",
            repository_root
            / "data_ready"
            / "labeled"
            / "train"
            / f"{symbol}_train_labeled.csv",
        ),
    ]

    for source_name, source_path in candidate_sources:
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


def _candidate_identity_digest(
    frame: pd.DataFrame,
) -> str:
    original_columns = [
        column
        for column in frame.columns
        if column not in STAGE04_BREADTH_FEATURES
    ]
    normalized = frame[original_columns].copy()
    hashed = pd.util.hash_pandas_object(
        normalized,
        index=True,
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _enrich_candidate_frame(
    candidate: pd.DataFrame,
    *,
    started_source: pd.DataFrame,
    breadth: pd.DataFrame,
    config: Stage04BreadthConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = candidate.copy()
    existing = [
        column
        for column in STAGE04_BREADTH_FEATURES
        if column in frame.columns
    ]
    if existing:
        frame = frame.drop(columns=existing)

    frame[config.date_column] = parse_market_date(
        frame[config.date_column]
    )
    if frame[config.date_column].isna().any():
        raise ValueError("Candidate file contains invalid dates.")

    before_rows = len(frame)
    before_digest = _candidate_identity_digest(frame)

    enriched = frame.merge(
        started_source,
        on=config.date_column,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    enriched = enriched.merge(
        breadth[
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

    if len(enriched) != before_rows:
        raise AssertionError(
            "Candidate row count changed during enrichment."
        )
    if _candidate_identity_digest(enriched) != before_digest:
        raise AssertionError(
            "Candidate identity or row order changed."
        )

    audit = {
        "candidate_rows": int(len(enriched)),
        "candidate_identity_sha256": before_digest,
        "started_missing_rows": int(
            enriched["started_run_length"].isna().sum()
        ),
        "breadth_raw_missing_rows": int(
            enriched["market_breadth_raw"].isna().sum()
        ),
        "breadth_ema30_missing_rows": int(
            enriched["market_breadth_ema30"].isna().sum()
        ),
        "breadth_slope5_missing_rows": int(
            enriched["market_breadth_slope5"].isna().sum()
        ),
        "breadth_regime_missing_rows": int(
            enriched["market_breadth_regime"].isna().sum()
        ),
    }
    return enriched, audit


def _update_approved_features(
    path: Path,
    *,
    encoding: str,
) -> tuple[pd.DataFrame, list[str]]:
    manifest = pd.read_csv(path, low_memory=False)
    if "feature" not in manifest.columns:
        raise KeyError(f"{path} has no feature column.")

    manifest = manifest.loc[
        ~manifest["feature"].isin(STAGE04_BREADTH_FEATURES)
    ].copy()

    extension = pd.DataFrame(
        {
            "feature": list(STAGE04_BREADTH_FEATURES),
            "data_type": [
                "numeric",
                "numeric",
                "numeric",
                "numeric",
                "categorical",
            ],
        }
    )

    for column in manifest.columns:
        if column not in extension.columns:
            extension[column] = pd.NA
    extension = extension[manifest.columns]

    updated = pd.concat(
        [manifest, extension],
        ignore_index=True,
    )
    if "feature_order" in updated.columns:
        updated["feature_order"] = range(1, len(updated) + 1)

    updated.to_csv(path, index=False, encoding=encoding)
    return updated, updated["feature"].astype(str).tolist()


def _update_feature_schema(
    path: Path,
    *,
    encoding: str,
) -> pd.DataFrame:
    schema = pd.read_csv(path, low_memory=False)
    if "feature" not in schema.columns:
        raise KeyError(f"{path} has no feature column.")

    schema = schema.loc[
        ~schema["feature"].isin(STAGE04_BREADTH_FEATURES)
    ].copy()

    rows = [
        {
            "feature": "started_run_length",
            "semantic_group": "original_signal_state",
            "source_feature": "started",
            "transformation": (
                "original nonnegative causal run length; "
                "no hard started filter"
            ),
            "unit_before": "observation count",
            "unit_after": "count",
            "data_type": "numeric",
            "price_basis": "not price based",
            "approved_for_pooled_model": True,
        },
        {
            "feature": "market_breadth_raw",
            "semantic_group": "cross_sectional_market_breadth",
            "source_feature": (
                "frozen-universe previous-valid adjusted-close returns"
            ),
            "transformation": (
                "(positive symbols - negative symbols) / valid symbols"
            ),
            "unit_before": "cross-sectional signs",
            "unit_after": "bounded breadth",
            "data_type": "numeric",
            "price_basis": "adjusted",
            "approved_for_pooled_model": True,
        },
        {
            "feature": "market_breadth_ema30",
            "semantic_group": "cross_sectional_market_breadth",
            "source_feature": "market_breadth_raw",
            "transformation": (
                "causal EMA span 30, adjust=False, min_periods=30"
            ),
            "unit_before": "bounded breadth",
            "unit_after": "smoothed breadth",
            "data_type": "numeric",
            "price_basis": "adjusted",
            "approved_for_pooled_model": True,
        },
        {
            "feature": "market_breadth_slope5",
            "semantic_group": "cross_sectional_market_breadth",
            "source_feature": "market_breadth_ema30",
            "transformation": (
                "EMA30_t - EMA30_(t-5 market dates)"
            ),
            "unit_before": "smoothed breadth",
            "unit_after": "breadth change",
            "data_type": "numeric",
            "price_basis": "adjusted",
            "approved_for_pooled_model": True,
        },
        {
            "feature": "market_breadth_regime",
            "semantic_group": "cross_sectional_market_breadth",
            "source_feature": (
                "market_breadth_ema30 and market_breadth_slope5"
            ),
            "transformation": (
                "broad_decline / recovery_negative / "
                "recovery_positive / deterioration / "
                "neutral_transition / broad_advance / "
                "warmup_unavailable"
            ),
            "unit_before": "breadth level and change",
            "unit_after": "categorical state",
            "data_type": "categorical",
            "price_basis": "adjusted",
            "approved_for_pooled_model": True,
        },
    ]

    extension = pd.DataFrame(rows)
    for column in schema.columns:
        if column not in extension.columns:
            extension[column] = pd.NA
    extension = extension[schema.columns]

    updated = pd.concat(
        [schema, extension],
        ignore_index=True,
    )
    if "feature_order" in updated.columns:
        updated["feature_order"] = range(1, len(updated) + 1)

    updated.to_csv(path, index=False, encoding=encoding)
    return updated


def _update_run_manifest(
    path: Path,
    *,
    approved_features: list[str],
    config: Stage04BreadthConfig,
    repository_root: Path,
    breadth_audit: dict[str, Any],
    candidate_rows: int,
    started_sources: dict[str, int],
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["approved_model_feature_count"] = len(approved_features)
    manifest["approved_model_features"] = approved_features
    manifest["feature_engineering_schema_version"] = (
        STAGE04_BREADTH_SCHEMA_VERSION
    )
    manifest["git_commit_sha"] = git_commit_sha(repository_root)
    manifest["market_breadth_extension"] = {
        "status": "experimental_train_only_feature_extension",
        "configuration": asdict(config),
        "configuration_hash": stable_hash(asdict(config)),
        "features": list(STAGE04_BREADTH_FEATURES),
        "candidate_filter_applied": False,
        "started_filter_applied": False,
        "candidate_rows_after_extension": int(candidate_rows),
        "breadth_audit": breadth_audit,
        "started_source_counts": started_sources,
        "unseen_test_used": False,
    }
    path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return manifest


def run_stage04_breadth_extension(
    repository_root: Path | str | None = None,
    *,
    config: Stage04BreadthConfig | None = None,
) -> dict[str, Any]:
    config = config or Stage04BreadthConfig()
    repository_root = locate_repository_root(
        Path(repository_root) if repository_root is not None else None
    )

    raw_root = repository_root / "raw_data"
    candidate_root = (
        repository_root
        / "data_ready"
        / "candidates"
        / "train"
    )
    manifest_root = repository_root / "results" / "manifests"
    audit_root = repository_root / "results" / "audits"
    audit_root.mkdir(parents=True, exist_ok=True)

    frozen_universe_path = (
        manifest_root / "02_frozen_universe.csv"
    )
    approved_features_path = (
        manifest_root / "04_approved_model_features.csv"
    )
    feature_schema_path = (
        manifest_root / "04_final_model_feature_schema.csv"
    )
    run_manifest_path = (
        manifest_root
        / "04_feature_and_leakage_audit_manifest.json"
    )

    required_paths = [
        raw_root,
        candidate_root,
        frozen_universe_path,
        approved_features_path,
        feature_schema_path,
        run_manifest_path,
    ]
    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Required Stage 04 inputs are missing: "
            + ", ".join(missing_paths)
        )

    universe = pd.read_csv(
        frozen_universe_path,
        low_memory=False,
    )
    if "symbol" not in universe.columns:
        raise KeyError(
            "02_frozen_universe.csv has no symbol column."
        )
    symbols = sorted(
        universe["symbol"].astype(str).unique().tolist()
    )
    if len(symbols) != 499:
        raise AssertionError(
            f"Expected 499 frozen symbols, observed {len(symbols)}."
        )

    candidate_paths = sorted(
        candidate_root.glob("*_train_candidates.csv")
    )
    candidate_map = {
        _symbol_from_candidate_path(path): path
        for path in candidate_paths
    }
    if set(candidate_map) != set(symbols):
        raise AssertionError(
            "Candidate files do not match the frozen universe."
        )

    candidate_end_dates = []
    for path in candidate_paths:
        dates = _read_dates(path, config.date_column)
        if dates.notna().any():
            candidate_end_dates.append(dates.max())
    if not candidate_end_dates:
        raise RuntimeError(
            "No valid candidate dates were found."
        )
    horizon_end = max(candidate_end_dates)
    if horizon_end > pd.Timestamp("2021-03-20"):
        raise AssertionError(
            "Stage 04 candidate horizon exceeds train cutoff."
        )

    breadth_parts = []
    source_error_rows = []
    for symbol in symbols:
        raw_path = raw_root / f"{symbol}.csv"
        try:
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)
            breadth_parts.append(
                _prepare_symbol_breadth_source(
                    raw_path,
                    symbol=symbol,
                    horizon_end=horizon_end,
                    config=config,
                )
            )
        except Exception as exc:
            source_error_rows.append(
                {
                    "symbol": symbol,
                    "file_name": raw_path.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    source_errors = pd.DataFrame(
        source_error_rows,
        columns=[
            "symbol",
            "file_name",
            "error_type",
            "error_message",
        ],
    )
    source_errors.to_csv(
        audit_root / "04_breadth_source_errors.csv",
        index=False,
        encoding=config.output_encoding,
    )
    if not source_errors.empty:
        raise RuntimeError(
            "Breadth source errors exist. Review "
            "results/audits/04_breadth_source_errors.csv"
        )

    observations = pd.concat(
        breadth_parts,
        ignore_index=True,
    )
    breadth, breadth_audit = build_daily_market_breadth(
        observations,
        config=config,
    )

    enrichment_audit_rows = []
    started_source_counts: dict[str, int] = {}
    total_candidate_rows = 0

    with TemporaryDirectory(
        prefix="stage04_breadth_",
        dir=repository_root,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)

        for symbol in symbols:
            candidate_path = candidate_map[symbol]
            raw_path = raw_root / f"{symbol}.csv"

            candidate = pd.read_csv(
                candidate_path,
                low_memory=False,
            )
            started_source, started_source_name = (
                _load_started_source(
                    symbol=symbol,
                    repository_root=repository_root,
                    raw_path=raw_path,
                    config=config,
                )
            )
            started_source_counts[started_source_name] = (
                started_source_counts.get(started_source_name, 0)
                + 1
            )

            enriched, audit = _enrich_candidate_frame(
                candidate,
                started_source=started_source,
                breadth=breadth,
                config=config,
            )
            enrichment_audit_rows.append(
                {
                    "symbol": symbol,
                    "started_source": started_source_name,
                    **audit,
                }
            )
            total_candidate_rows += len(enriched)

            enriched.to_csv(
                temporary_root / candidate_path.name,
                index=False,
                encoding=config.output_encoding,
            )

        enrichment_audit = pd.DataFrame(
            enrichment_audit_rows
        )

        for column in [
            "started_missing_rows",
            "breadth_raw_missing_rows",
            "breadth_ema30_missing_rows",
            "breadth_slope5_missing_rows",
            "breadth_regime_missing_rows",
        ]:
            if int(enrichment_audit[column].sum()) != 0:
                raise AssertionError(
                    f"Candidate enrichment has missing values in {column}."
                )

        for symbol in symbols:
            candidate_path = candidate_map[symbol]
            staged_path = temporary_root / candidate_path.name
            os.replace(staged_path, candidate_path)

    breadth.to_csv(
        audit_root / "04_daily_market_breadth.csv",
        index=False,
        encoding=config.output_encoding,
    )
    pd.DataFrame([breadth_audit]).to_csv(
        audit_root / "04_market_breadth_feature_audit.csv",
        index=False,
        encoding=config.output_encoding,
    )
    enrichment_audit.to_csv(
        audit_root / "04_breadth_candidate_enrichment_audit.csv",
        index=False,
        encoding=config.output_encoding,
    )

    approved_manifest, approved_features = (
        _update_approved_features(
            approved_features_path,
            encoding=config.output_encoding,
        )
    )
    _update_feature_schema(
        feature_schema_path,
        encoding=config.output_encoding,
    )
    run_manifest = _update_run_manifest(
        run_manifest_path,
        approved_features=approved_features,
        config=config,
        repository_root=repository_root,
        breadth_audit=breadth_audit,
        candidate_rows=total_candidate_rows,
        started_sources=started_source_counts,
    )

    if len(approved_features) != 40:
        raise AssertionError(
            f"Expected 40 approved features, observed "
            f"{len(approved_features)}."
        )

    summary = {
        "schema_version": STAGE04_BREADTH_SCHEMA_VERSION,
        "git_commit_sha": run_manifest.get("git_commit_sha"),
        "frozen_symbols": len(symbols),
        "train_horizon_end": str(horizon_end.date()),
        "daily_breadth_rows": len(breadth),
        "candidate_rows": total_candidate_rows,
        "approved_features": len(approved_features),
        "added_features": list(STAGE04_BREADTH_FEATURES),
        "hard_started_filter_applied": False,
        "hard_breadth_filter_applied": False,
        "candidate_identity_preserved": True,
        "configuration_hash": stable_hash(asdict(config)),
    }

    return {
        "summary": summary,
        "daily_breadth": breadth,
        "breadth_audit": breadth_audit,
        "enrichment_audit": enrichment_audit,
        "approved_feature_manifest": approved_manifest,
        "run_manifest": run_manifest,
        "paths": {
            "daily_breadth": (
                audit_root / "04_daily_market_breadth.csv"
            ),
            "breadth_audit": (
                audit_root
                / "04_market_breadth_feature_audit.csv"
            ),
            "enrichment_audit": (
                audit_root
                / "04_breadth_candidate_enrichment_audit.csv"
            ),
            "run_manifest": run_manifest_path,
        },
    }
