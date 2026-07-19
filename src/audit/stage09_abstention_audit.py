"""Independent audit utilities for the Stage 09 abstention-policy retest.

This module intentionally does not import the Stage 08 policy implementation or
the Stage 09 outcome helper. It independently reconstructs policy decisions,
classification metrics, and raw-price corrected event outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import yaml


AUDIT_SCHEMA_VERSION = "stage09a_v1_independent_abstention_audit"


@dataclass(frozen=True)
class FrozenPolicy:
    gate_name: str
    allowed_regimes: tuple[str, ...]
    minimum_raw_score: float
    maximum_daily_fraction: float
    minimum_signals_per_date: int
    score_column: str = "xgboost_ranking_score"
    date_column: str = "dEven"
    symbol_column: str = "symbol"
    event_id_column: str = "event_id"
    regime_column: str = "market_breadth_regime"

    def validate(self) -> None:
        if not self.gate_name:
            raise ValueError("gate_name is empty.")
        if not self.allowed_regimes:
            raise ValueError("allowed_regimes is empty.")
        if not 0.0 <= float(self.minimum_raw_score) <= 1.0:
            raise ValueError("minimum_raw_score must lie in [0, 1].")
        if not 0.0 < float(self.maximum_daily_fraction) <= 1.0:
            raise ValueError("maximum_daily_fraction must lie in (0, 1].")
        if int(self.minimum_signals_per_date) != 0:
            raise ValueError(
                "Independent audit expects minimum_signals_per_date = 0."
            )


def locate_repository_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (
            (candidate / "configs").exists()
            and (candidate / "results").exists()
            and (candidate / "src").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Repository root was not found from the current directory."
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML artifact must be a mapping: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_market_date(series: pd.Series) -> pd.Series:
    """Independently parse YYYYMMDD values and conventional date strings."""
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


def as_bool(series: pd.Series) -> pd.Series:
    """Normalize common CSV boolean encodings without truthy-string mistakes."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    result = normalized.map(mapping)
    invalid = normalized.notna() & result.isna()
    if invalid.any():
        values = sorted(normalized.loc[invalid].dropna().unique().tolist())
        raise ValueError(f"Unrecognized boolean encodings: {values}")
    return result.fillna(False).astype(bool)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def resolve_raw_data_root(
    repository_root: Path,
    paths_config: dict[str, Any],
) -> Path:
    environment_variable = str(
        paths_config.get("environment_variable", "FINML_DATA_ROOT")
    )
    external_root = os.environ.get(environment_variable)
    if external_root:
        data_root = Path(external_root).expanduser().resolve()
    else:
        data_root = repository_root

    raw_relative = str(
        paths_config["directories"]["raw_data"]
    )
    raw_root = data_root / raw_relative
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw-data directory is missing: {raw_root}")
    return raw_root


def reconstruct_policy_decisions(
    lock_frame: pd.DataFrame,
    *,
    policy: FrozenPolicy,
) -> pd.DataFrame:
    """Independently reproduce the frozen gate, threshold, ranking, and cap."""
    policy.validate()
    required = {
        policy.date_column,
        policy.score_column,
        policy.regime_column,
        policy.symbol_column,
        policy.event_id_column,
    }
    missing = sorted(required - set(lock_frame.columns))
    if missing:
        raise KeyError(f"Policy-audit columns are missing: {missing}")

    frame = lock_frame[
        [
            policy.event_id_column,
            policy.symbol_column,
            policy.date_column,
            policy.regime_column,
            policy.score_column,
        ]
    ].copy()
    frame[policy.event_id_column] = frame[
        policy.event_id_column
    ].astype(str)
    frame[policy.symbol_column] = frame[
        policy.symbol_column
    ].astype(str)
    frame[policy.date_column] = pd.to_datetime(
        frame[policy.date_column],
        errors="raise",
    ).dt.normalize()
    frame[policy.regime_column] = frame[
        policy.regime_column
    ].astype("string")
    frame[policy.score_column] = pd.to_numeric(
        frame[policy.score_column],
        errors="raise",
    )

    if frame[policy.event_id_column].duplicated().any():
        raise ValueError("Duplicate event IDs exist in inference lock.")
    if frame[policy.regime_column].isna().any():
        raise ValueError("Missing Breadth regimes exist in inference lock.")
    if not np.isfinite(
        frame[policy.score_column].to_numpy(dtype=float)
    ).all():
        raise ValueError("Nonfinite model scores exist in inference lock.")

    frame = frame.sort_values(
        [
            policy.date_column,
            policy.score_column,
            policy.symbol_column,
            policy.event_id_column,
        ],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    grouped = frame.groupby(
        policy.date_column,
        sort=False,
        observed=False,
    )
    frame["audit_daily_candidate_count"] = grouped[
        policy.event_id_column
    ].transform("size").astype(int)
    frame["audit_daily_maximum_quota"] = np.ceil(
        frame["audit_daily_candidate_count"].to_numpy(dtype=float)
        * float(policy.maximum_daily_fraction)
    ).astype(int)

    frame["audit_market_gate_pass"] = frame[
        policy.regime_column
    ].isin(list(policy.allowed_regimes))
    frame["audit_score_threshold_pass"] = frame[
        policy.score_column
    ].ge(float(policy.minimum_raw_score))
    frame["audit_policy_eligible"] = (
        frame["audit_market_gate_pass"]
        & frame["audit_score_threshold_pass"]
    )

    frame["audit_daily_eligible_count"] = (
        frame["audit_policy_eligible"]
        .groupby(frame[policy.date_column], observed=False)
        .transform("sum")
        .astype(int)
    )
    frame["audit_daily_signal_quota"] = np.minimum(
        frame["audit_daily_maximum_quota"].to_numpy(dtype=int),
        frame["audit_daily_eligible_count"].to_numpy(dtype=int),
    ).astype(int)

    eligible_rank = (
        frame["audit_policy_eligible"]
        .groupby(frame[policy.date_column], observed=False)
        .cumsum()
    )
    frame["audit_daily_eligible_rank"] = np.where(
        frame["audit_policy_eligible"],
        eligible_rank,
        0,
    ).astype(int)
    frame["audit_selected_signal"] = (
        frame["audit_policy_eligible"]
        & frame["audit_daily_eligible_rank"].gt(0)
        & frame["audit_daily_eligible_rank"].le(
            frame["audit_daily_signal_quota"]
        )
    )

    cutoffs = (
        frame.loc[
            frame["audit_selected_signal"],
            [policy.date_column, policy.score_column],
        ]
        .groupby(
            policy.date_column,
            sort=False,
            observed=False,
        )[policy.score_column]
        .min()
        .rename("audit_daily_selected_score_cutoff")
    )
    frame = frame.merge(
        cutoffs,
        left_on=policy.date_column,
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    return frame


def classification_metrics(
    labels: pd.Series,
    selected: pd.Series,
) -> dict[str, float | int]:
    y = pd.to_numeric(labels, errors="raise").astype(int).to_numpy()
    signal = as_bool(selected).to_numpy(dtype=bool)

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")

    tp = int(np.sum(signal & (y == 1)))
    fp = int(np.sum(signal & (y == 0)))
    tn = int(np.sum((~signal) & (y == 0)))
    fn = int(np.sum((~signal) & (y == 1)))
    precision = safe_divide(tp, tp + fp)
    prevalence = float(np.mean(y))
    specificity = safe_divide(tn, tn + fp)
    sensitivity = safe_divide(tp, tp + fn)

    return {
        "events": int(len(y)),
        "signals": int(signal.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "prevalence": prevalence,
        "precision_lift": float(precision - prevalence),
        "precision_ratio": safe_divide(precision, prevalence),
        "specificity": specificity,
        "sensitivity": sensitivity,
    }


def corrected_outcome_metrics(
    selected_frame: pd.DataFrame,
) -> dict[str, float | int]:
    required = {
        "meta_label",
        "corrected_event_return",
        "barrier_touched",
    }
    missing = sorted(required - set(selected_frame.columns))
    if missing:
        raise KeyError(f"Outcome metric columns are missing: {missing}")
    if selected_frame.empty:
        raise ValueError("Selected outcome frame is empty.")

    labels = pd.to_numeric(
        selected_frame["meta_label"],
        errors="raise",
    ).astype(int)
    returns = pd.to_numeric(
        selected_frame["corrected_event_return"],
        errors="raise",
    )
    finite = returns[np.isfinite(returns.to_numpy(dtype=float))]
    if len(finite) != len(selected_frame):
        raise ValueError("Nonfinite corrected returns exist.")
    if int((finite > 0.0).sum()) != int(labels.sum()):
        raise AssertionError("Return signs disagree with labels.")

    positive = finite[finite > 0.0]
    negative = finite[finite < 0.0]
    zero = finite[np.isclose(finite, 0.0, atol=1.0e-12)]
    average_win = (
        float(positive.mean()) if len(positive) else float("nan")
    )
    average_loss = (
        float(negative.mean()) if len(negative) else float("nan")
    )
    gross_profit = float(positive.sum())
    gross_loss = float(-negative.sum())
    barrier_counts = (
        selected_frame["barrier_touched"]
        .astype("string")
        .str.lower()
        .value_counts(dropna=False)
        .to_dict()
    )

    result: dict[str, float | int] = {
        "events": int(len(selected_frame)),
        "winning_events": int(labels.sum()),
        "nonwinning_events": int((labels == 0).sum()),
        "losing_events_negative_return": int(len(negative)),
        "breakeven_events": int(len(zero)),
        "win_rate": float(labels.mean()),
        "mean_corrected_event_return": float(finite.mean()),
        "median_corrected_event_return": float(finite.median()),
        "corrected_event_return_standard_deviation": float(
            finite.std(ddof=0)
        ),
        "minimum_corrected_event_return": float(finite.min()),
        "maximum_corrected_event_return": float(finite.max()),
        "average_winning_return": average_win,
        "average_losing_return": average_loss,
        "payoff_ratio": safe_divide(
            average_win,
            abs(average_loss),
        ),
        "gross_profit_sum": gross_profit,
        "gross_loss_absolute_sum": gross_loss,
        "profit_factor": safe_divide(gross_profit, gross_loss),
        "upper_barrier_events": int(barrier_counts.get("upper", 0)),
        "lower_barrier_events": int(barrier_counts.get("lower", 0)),
        "vertical_barrier_events": int(
            barrier_counts.get("vertical", 0)
        ),
    }

    if "holding_period_observations" in selected_frame.columns:
        holding = pd.to_numeric(
            selected_frame["holding_period_observations"],
            errors="coerce",
        )
        holding = holding[np.isfinite(holding.to_numpy(dtype=float))]
        result[
            "mean_label_event_holding_period_observations"
        ] = float(holding.mean())
        result[
            "median_label_event_holding_period_observations"
        ] = float(holding.median())
    return result


def values_close(
    observed: Any,
    expected: Any,
    *,
    tolerance: float = 1.0e-10,
) -> bool:
    if observed is None or expected is None:
        return observed is expected

    try:
        observed_float = float(observed)
        expected_float = float(expected)
        if math.isnan(observed_float) and math.isnan(expected_float):
            return True
        return math.isclose(
            observed_float,
            expected_float,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return str(observed) == str(expected)


def compare_metric_dicts(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    tolerance: float = 1.0e-10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, observed_value in observed.items():
        if key not in expected:
            rows.append({
                "metric": key,
                "passed": False,
                "observed": observed_value,
                "expected": "<missing>",
            })
            continue
        expected_value = expected[key]
        rows.append({
            "metric": key,
            "passed": values_close(
                observed_value,
                expected_value,
                tolerance=tolerance,
            ),
            "observed": observed_value,
            "expected": expected_value,
        })
    return rows


def independently_reconstruct_selected_outcomes(
    selected_frame: pd.DataFrame,
    *,
    raw_root: Path,
    signal_generation_end: pd.Timestamp,
    tail_end: pd.Timestamp,
    horizon: int = 30,
    upper_barrier: float = 0.15,
    lower_barrier: float = -0.15,
    tolerance: float = 1.0e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild every selected corrected outcome directly from raw prices."""
    required = {
        "event_id",
        "symbol",
        "dEven",
        "meta_label",
        "barrier_touched",
        "event_end_date",
        "original_event_return",
        "entry_adjusted_last_price",
        "outcome_window_end_date",
        "outcome_window_uses_tail",
        "day30_adjusted_last_price",
        "day30_return_reconstructed",
        "maximum_adjusted_high_30",
        "maximum_adjusted_high_date_30",
        "maximum_adjusted_high_observation_30",
        "maximum_positive_return_30",
        "corrected_event_return",
        "corrected_return_rule",
        "corrected_winner",
        "corrected_outcome_date",
        "corrected_outcome_observation",
    }
    missing = sorted(required - set(selected_frame.columns))
    if missing:
        raise KeyError(
            f"Selected-signal reconstruction columns are missing: {missing}"
        )

    source = selected_frame.copy()
    source["event_id"] = source["event_id"].astype(str)
    source["symbol"] = source["symbol"].astype(str)
    for column in [
        "dEven",
        "event_end_date",
        "outcome_window_end_date",
        "maximum_adjusted_high_date_30",
        "corrected_outcome_date",
    ]:
        source[column] = pd.to_datetime(
            source[column],
            errors="raise",
        ).dt.normalize()

    source["meta_label"] = pd.to_numeric(
        source["meta_label"],
        errors="raise",
    ).astype(int)
    source["outcome_window_uses_tail"] = as_bool(
        source["outcome_window_uses_tail"]
    )
    source["corrected_winner"] = as_bool(
        source["corrected_winner"]
    )

    audit_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    signal_end = pd.Timestamp(signal_generation_end).normalize()
    frozen_tail = pd.Timestamp(tail_end).normalize()

    for symbol, events in source.groupby(
        "symbol",
        sort=True,
        observed=False,
    ):
        raw_path = raw_root / f"{symbol}.csv"
        if not raw_path.exists():
            for event_id in events["event_id"]:
                error_rows.append({
                    "event_id": event_id,
                    "symbol": symbol,
                    "error_type": "FileNotFoundError",
                    "error_message": str(raw_path),
                })
            continue

        try:
            raw = pd.read_csv(
                raw_path,
                usecols=[
                    "dEven",
                    "adj_last_price",
                    "adj_high",
                ],
                low_memory=False,
            )
            raw["dEven"] = parse_market_date(raw["dEven"]).dt.normalize()
            raw["adj_last_price"] = pd.to_numeric(
                raw["adj_last_price"],
                errors="raise",
            )
            raw["adj_high"] = pd.to_numeric(
                raw["adj_high"],
                errors="raise",
            )
            raw = raw.loc[
                raw["dEven"].notna()
                & raw["dEven"].le(frozen_tail)
            ].copy()
            raw = (
                raw.sort_values("dEven", kind="stable")
                .drop_duplicates("dEven", keep="last")
                .reset_index(drop=True)
            )
            if raw.empty:
                raise ValueError("Raw history is empty.")
            if not np.isfinite(
                raw[
                    ["adj_last_price", "adj_high"]
                ].to_numpy(dtype=float)
            ).all():
                raise ValueError("Raw adjusted prices contain nonfinite values.")
            if (
                raw["adj_last_price"].le(0.0).any()
                or raw["adj_high"].le(0.0).any()
            ):
                raise ValueError("Raw adjusted prices contain nonpositive values.")

            date_to_position = {
                pd.Timestamp(value): int(position)
                for position, value in enumerate(raw["dEven"])
            }

            for event in events.itertuples(index=False):
                event_id = str(event.event_id)
                event_date = pd.Timestamp(event.dEven).normalize()
                try:
                    if event_date > signal_end:
                        raise AssertionError(
                            "Signal occurs after frozen signal end."
                        )
                    if event_date not in date_to_position:
                        raise KeyError("Signal-date raw row is missing.")

                    entry_position = date_to_position[event_date]
                    final_position = entry_position + int(horizon)
                    if final_position >= len(raw):
                        raise ValueError(
                            "Insufficient future observations for horizon."
                        )

                    entry_price = float(
                        raw.iloc[entry_position]["adj_last_price"]
                    )
                    future = raw.iloc[
                        entry_position + 1 : final_position + 1
                    ].copy()
                    future_high = future["adj_high"].to_numpy(dtype=float)
                    maximum_offset = int(np.argmax(future_high)) + 1
                    maximum_high = float(
                        future_high[maximum_offset - 1]
                    )
                    maximum_date = pd.Timestamp(
                        future.iloc[
                            maximum_offset - 1
                        ]["dEven"]
                    ).normalize()
                    maximum_return = maximum_high / entry_price - 1.0

                    day30 = future.iloc[-1]
                    day30_date = pd.Timestamp(
                        day30["dEven"]
                    ).normalize()
                    day30_price = float(day30["adj_last_price"])
                    day30_return = day30_price / entry_price - 1.0

                    barrier = str(
                        event.barrier_touched
                    ).strip().lower()
                    event_end = pd.Timestamp(
                        event.event_end_date
                    ).normalize()

                    if barrier == "upper":
                        corrected_return = maximum_return
                        corrected_rule = (
                            "upper_maximum_adjusted_high_return_over_"
                            "next_30_trading_observations"
                        )
                        corrected_date = maximum_date
                        corrected_observation = maximum_offset
                        if corrected_return + tolerance < upper_barrier:
                            raise AssertionError(
                                "Upper reconstructed return is below barrier."
                            )
                    elif barrier == "lower":
                        corrected_return = float(lower_barrier)
                        corrected_rule = "lower_fixed_minus_15_percent"
                        corrected_date = event_end
                        corrected_observation = int(
                            future["dEven"].le(event_end).sum()
                        )
                    elif barrier == "vertical":
                        corrected_return = day30_return
                        corrected_rule = (
                            "vertical_adjusted_last_return_on_"
                            "trading_observation_30"
                        )
                        corrected_date = day30_date
                        corrected_observation = int(horizon)
                        if not math.isclose(
                            float(event.original_event_return),
                            corrected_return,
                            rel_tol=tolerance,
                            abs_tol=tolerance,
                        ):
                            raise AssertionError(
                                "Vertical original return does not match day 30."
                            )
                        if event_end != day30_date:
                            raise AssertionError(
                                "Vertical event end does not match day 30."
                            )
                    else:
                        raise ValueError(
                            f"Unexpected barrier: {barrier}"
                        )

                    comparisons = {
                        "entry_adjusted_last_price": (
                            entry_price,
                            float(event.entry_adjusted_last_price),
                        ),
                        "outcome_window_end_date": (
                            day30_date,
                            pd.Timestamp(
                                event.outcome_window_end_date
                            ).normalize(),
                        ),
                        "outcome_window_uses_tail": (
                            bool(day30_date > signal_end),
                            bool(event.outcome_window_uses_tail),
                        ),
                        "day30_adjusted_last_price": (
                            day30_price,
                            float(event.day30_adjusted_last_price),
                        ),
                        "day30_return_reconstructed": (
                            day30_return,
                            float(event.day30_return_reconstructed),
                        ),
                        "maximum_adjusted_high_30": (
                            maximum_high,
                            float(event.maximum_adjusted_high_30),
                        ),
                        "maximum_adjusted_high_date_30": (
                            maximum_date,
                            pd.Timestamp(
                                event.maximum_adjusted_high_date_30
                            ).normalize(),
                        ),
                        "maximum_adjusted_high_observation_30": (
                            maximum_offset,
                            int(event.maximum_adjusted_high_observation_30),
                        ),
                        "maximum_positive_return_30": (
                            maximum_return,
                            float(event.maximum_positive_return_30),
                        ),
                        "corrected_event_return": (
                            corrected_return,
                            float(event.corrected_event_return),
                        ),
                        "corrected_return_rule": (
                            corrected_rule,
                            str(event.corrected_return_rule),
                        ),
                        "corrected_winner": (
                            bool(corrected_return > 0.0),
                            bool(event.corrected_winner),
                        ),
                        "corrected_outcome_date": (
                            corrected_date,
                            pd.Timestamp(
                                event.corrected_outcome_date
                            ).normalize(),
                        ),
                        "corrected_outcome_observation": (
                            corrected_observation,
                            int(event.corrected_outcome_observation),
                        ),
                    }

                    failed_fields = [
                        field
                        for field, (observed, expected) in comparisons.items()
                        if not values_close(
                            observed,
                            expected,
                            tolerance=tolerance,
                        )
                    ]
                    label_matches = (
                        int(corrected_return > 0.0)
                        == int(event.meta_label)
                    )
                    if not label_matches:
                        failed_fields.append("meta_label_sign")

                    audit_rows.append({
                        "event_id": event_id,
                        "symbol": symbol,
                        "dEven": event_date,
                        "barrier_touched": barrier,
                        "audit_entry_price": entry_price,
                        "audit_day30_date": day30_date,
                        "audit_day30_return": day30_return,
                        "audit_maximum_return_30": maximum_return,
                        "audit_corrected_event_return": corrected_return,
                        "audit_corrected_winner": bool(
                            corrected_return > 0.0
                        ),
                        "failed_field_count": int(len(failed_fields)),
                        "failed_fields": "|".join(failed_fields),
                        "passed": bool(not failed_fields),
                    })
                    if failed_fields:
                        error_rows.append({
                            "event_id": event_id,
                            "symbol": symbol,
                            "error_type": "ReconstructionMismatch",
                            "error_message": "|".join(failed_fields),
                        })
                except Exception as exc:
                    error_rows.append({
                        "event_id": event_id,
                        "symbol": symbol,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    })
        except Exception as exc:
            for event_id in events["event_id"]:
                error_rows.append({
                    "event_id": event_id,
                    "symbol": symbol,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })

    audit = pd.DataFrame(audit_rows)
    errors = pd.DataFrame(
        error_rows,
        columns=[
            "event_id",
            "symbol",
            "error_type",
            "error_message",
        ],
    )
    if not audit.empty:
        audit = audit.sort_values(
            ["dEven", "symbol", "event_id"],
            kind="stable",
        ).reset_index(drop=True)
    return audit, errors
