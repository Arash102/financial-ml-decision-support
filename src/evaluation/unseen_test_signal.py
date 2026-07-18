"""Confirmatory unseen-test signal-level evaluation helpers.

Stage 09 v3 preserves the frozen model and signal policy while making the
signal-level economic outcome definition explicit:

- upper barrier: maximum adjusted-high return over the complete next
  30 trading observations;
- lower barrier: fixed -15 percent;
- vertical barrier: adjusted-last return on trading observation 30.

The upper-barrier measure is an ex-post maximum favorable event outcome. It is
not an executable exit rule or a portfolio return.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import math

import numpy as np
import pandas as pd

from src.features.preprocessing import parse_market_date


UNSEEN_TEST_SIGNAL_SCHEMA_VERSION = (
    "stage09_v3_confirmatory_unseen_test_selected_signal_corrected_outcomes"
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def reconstruct_corrected_event_outcomes(
    frame: pd.DataFrame,
    *,
    raw_file_map: dict[str, Path],
    signal_generation_end: pd.Timestamp,
    outcome_observation_tail_end: pd.Timestamp,
    horizon_observations: int = 30,
    upper_barrier_return: float = 0.15,
    lower_barrier_return: float = -0.15,
    date_column: str = "dEven",
    symbol_column: str = "symbol",
    event_id_column: str = "event_id",
    label_column: str = "meta_label",
    barrier_column: str = "barrier_touched",
    original_return_column: str = "event_return",
    event_end_column: str = "event_end_date",
    tolerance: float = 1.0e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct corrected event-level outcomes after the inference lock.

    The function reads adjusted market prices only after scores and selections
    have been locked. Signal dates remain capped at ``signal_generation_end``.
    Later raw rows may only complete the already-defined 30-observation outcome
    window, up to ``outcome_observation_tail_end``.
    """
    required = {
        event_id_column,
        symbol_column,
        date_column,
        label_column,
        barrier_column,
        original_return_column,
        event_end_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Corrected-outcome input columns are missing: {missing}")
    if frame.empty:
        raise ValueError("Corrected-outcome input frame is empty.")
    if horizon_observations <= 0:
        raise ValueError("The outcome horizon must be positive.")
    if upper_barrier_return <= 0.0:
        raise ValueError("The upper barrier must be positive.")
    if lower_barrier_return >= 0.0:
        raise ValueError("The lower barrier must be negative.")

    signal_end = pd.Timestamp(signal_generation_end).normalize()
    tail_end = pd.Timestamp(outcome_observation_tail_end).normalize()
    if tail_end < signal_end:
        raise ValueError("Outcome tail end cannot precede signal-generation end.")

    source = frame.copy()
    source[date_column] = pd.to_datetime(
        source[date_column], errors="raise"
    ).dt.normalize()
    source[event_end_column] = pd.to_datetime(
        source[event_end_column], errors="raise"
    ).dt.normalize()
    source[label_column] = pd.to_numeric(
        source[label_column], errors="raise"
    ).astype(int)

    if source[event_id_column].duplicated().any():
        raise ValueError("Duplicate event IDs exist in corrected-outcome input.")
    if not source[label_column].isin([0, 1]).all():
        raise ValueError("Outcome labels must be binary 0/1.")
    if source[date_column].gt(signal_end).any():
        raise ValueError("A signal date occurs after the frozen signal end.")

    result_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []

    for symbol, symbol_events in source.groupby(symbol_column, sort=True):
        raw_path = Path(raw_file_map.get(str(symbol), Path("")))
        if not raw_path.exists():
            for event in symbol_events.itertuples(index=False):
                error_rows.append({
                    event_id_column: getattr(event, event_id_column),
                    symbol_column: symbol,
                    date_column: getattr(event, date_column),
                    "error_type": "FileNotFoundError",
                    "error_message": f"Raw file is missing: {raw_path}",
                })
            continue

        try:
            raw = pd.read_csv(
                raw_path,
                usecols=["dEven", "adj_last_price", "adj_high"],
                low_memory=False,
            )
            raw["dEven"] = pd.to_datetime(
                parse_market_date(raw["dEven"]), errors="coerce"
            ).dt.normalize()
            raw["adj_last_price"] = pd.to_numeric(
                raw["adj_last_price"], errors="raise"
            )
            raw["adj_high"] = pd.to_numeric(
                raw["adj_high"], errors="raise"
            )
            raw = raw.loc[
                raw["dEven"].notna() & raw["dEven"].le(tail_end)
            ].copy()
            raw = (
                raw.sort_values("dEven", kind="stable")
                .drop_duplicates("dEven", keep="last")
                .reset_index(drop=True)
            )
            if raw.empty:
                raise ValueError("Raw market history is empty after filtering.")
            if (
                not np.isfinite(raw["adj_last_price"].to_numpy(dtype=float)).all()
                or not np.isfinite(raw["adj_high"].to_numpy(dtype=float)).all()
            ):
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

            for event in symbol_events.itertuples(index=False):
                event_id = getattr(event, event_id_column)
                event_date = pd.Timestamp(
                    getattr(event, date_column)
                ).normalize()
                try:
                    if event_date not in date_to_position:
                        raise KeyError(
                            f"Raw row for signal date {event_date.date()} was not found."
                        )
                    entry_position = date_to_position[event_date]
                    final_position = entry_position + horizon_observations
                    if final_position >= len(raw):
                        available = len(raw) - entry_position - 1
                        raise ValueError(
                            f"Only {available} future observations remain; "
                            f"{horizon_observations} are required."
                        )

                    entry_price = float(
                        raw.iloc[entry_position]["adj_last_price"]
                    )
                    future = raw.iloc[
                        entry_position + 1 : final_position + 1
                    ].copy()
                    future_high = future["adj_high"].to_numpy(dtype=float)
                    maximum_offset = int(np.argmax(future_high)) + 1
                    maximum_high = float(future_high[maximum_offset - 1])
                    maximum_date = pd.Timestamp(
                        future.iloc[maximum_offset - 1]["dEven"]
                    )
                    maximum_return = maximum_high / entry_price - 1.0

                    day30_row = future.iloc[-1]
                    day30_date = pd.Timestamp(day30_row["dEven"])
                    day30_price = float(day30_row["adj_last_price"])
                    day30_return = day30_price / entry_price - 1.0

                    barrier = str(
                        getattr(event, barrier_column)
                    ).strip().lower()
                    original_return = float(
                        getattr(event, original_return_column)
                    )
                    stored_event_end = pd.Timestamp(
                        getattr(event, event_end_column)
                    ).normalize()

                    if barrier == "upper":
                        corrected_return = maximum_return
                        corrected_rule = (
                            "upper_maximum_adjusted_high_return_over_"
                            "next_30_trading_observations"
                        )
                        corrected_outcome_date = maximum_date
                        corrected_outcome_observation = maximum_offset
                        if corrected_return + tolerance < upper_barrier_return:
                            raise AssertionError(
                                "Upper event has reconstructed maximum return "
                                f"below +15%: {corrected_return:.12f}"
                            )
                    elif barrier == "lower":
                        corrected_return = lower_barrier_return
                        corrected_rule = "lower_fixed_minus_15_percent"
                        corrected_outcome_date = stored_event_end
                        corrected_outcome_observation = int(
                            (future["dEven"].le(stored_event_end)).sum()
                        )
                    elif barrier == "vertical":
                        corrected_return = day30_return
                        corrected_rule = (
                            "vertical_adjusted_last_return_on_"
                            "trading_observation_30"
                        )
                        corrected_outcome_date = day30_date
                        corrected_outcome_observation = horizon_observations
                        if not math.isclose(
                            corrected_return,
                            original_return,
                            rel_tol=tolerance,
                            abs_tol=tolerance,
                        ):
                            raise AssertionError(
                                "Vertical stored return differs from the "
                                "reconstructed day-30 return: "
                                f"stored={original_return:.12f}, "
                                f"reconstructed={corrected_return:.12f}"
                            )
                        if stored_event_end != day30_date:
                            raise AssertionError(
                                "Vertical stored event end differs from the "
                                "reconstructed day-30 date."
                            )
                    else:
                        raise ValueError(
                            f"Unexpected barrier_touched value: {barrier}"
                        )

                    corrected_winner = bool(corrected_return > 0.0)
                    label = int(getattr(event, label_column))
                    if int(corrected_winner) != label:
                        raise AssertionError(
                            "Corrected return sign disagrees with frozen label: "
                            f"return={corrected_return:.12f}, label={label}"
                        )
                    if day30_date > tail_end:
                        raise AssertionError(
                            "Outcome window exceeds the frozen observation tail."
                        )

                    result_rows.append({
                        event_id_column: event_id,
                        "entry_adjusted_last_price": entry_price,
                        "outcome_window_end_date": day30_date,
                        "outcome_window_uses_tail": bool(day30_date > signal_end),
                        "day30_adjusted_last_price": day30_price,
                        "day30_return_reconstructed": day30_return,
                        "maximum_adjusted_high_30": maximum_high,
                        "maximum_adjusted_high_date_30": maximum_date,
                        "maximum_adjusted_high_observation_30": maximum_offset,
                        "maximum_positive_return_30": maximum_return,
                        "corrected_event_return": corrected_return,
                        "corrected_return_rule": corrected_rule,
                        "corrected_winner": corrected_winner,
                        "corrected_outcome_date": corrected_outcome_date,
                        "corrected_outcome_observation": corrected_outcome_observation,
                    })
                except Exception as exc:
                    error_rows.append({
                        event_id_column: event_id,
                        symbol_column: symbol,
                        date_column: event_date,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    })
        except Exception as exc:
            for event in symbol_events.itertuples(index=False):
                error_rows.append({
                    event_id_column: getattr(event, event_id_column),
                    symbol_column: symbol,
                    date_column: getattr(event, date_column),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })

    result = pd.DataFrame(result_rows)
    errors = pd.DataFrame(
        error_rows,
        columns=[
            event_id_column,
            symbol_column,
            date_column,
            "error_type",
            "error_message",
        ],
    )
    if not result.empty:
        result = result.sort_values(event_id_column, kind="stable").reset_index(
            drop=True
        )
    return result, errors


def corrected_event_outcome_metrics(
    frame: pd.DataFrame,
    *,
    label_column: str = "meta_label",
    return_column: str = "corrected_event_return",
    holding_column: str = "holding_period_observations",
    barrier_column: str = "barrier_touched",
) -> dict[str, float | int]:
    """Summarize corrected event outcomes without portfolio claims."""
    required = {label_column, return_column, barrier_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Outcome columns are missing: {missing}")
    if frame.empty:
        raise ValueError("Outcome frame is empty.")

    labels = pd.to_numeric(frame[label_column], errors="raise").astype(int)
    returns = pd.to_numeric(frame[return_column], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("Outcome labels must be binary 0/1.")

    finite_returns = returns[np.isfinite(returns.to_numpy(dtype=float))]
    positive_returns = finite_returns[finite_returns > 0.0]
    negative_returns = finite_returns[finite_returns < 0.0]
    zero_returns = finite_returns[np.isclose(finite_returns, 0.0, atol=1.0e-12)]

    if len(finite_returns) != len(frame):
        raise ValueError("Nonfinite corrected event returns exist.")
    if int((finite_returns > 0.0).sum()) != int(labels.sum()):
        raise AssertionError("Corrected return signs disagree with labels.")

    average_gain = (
        float(positive_returns.mean()) if len(positive_returns) else float("nan")
    )
    average_loss = (
        float(negative_returns.mean()) if len(negative_returns) else float("nan")
    )
    gross_gain = float(positive_returns.sum())
    gross_loss_magnitude = float(-negative_returns.sum())
    barrier_counts = (
        frame[barrier_column].astype("string").value_counts(dropna=False).to_dict()
    )

    result: dict[str, float | int] = {
        "events": int(len(frame)),
        "winning_events": int(labels.sum()),
        "nonwinning_events": int((labels == 0).sum()),
        "losing_events_negative_return": int(len(negative_returns)),
        "breakeven_events": int(len(zero_returns)),
        "win_rate": float(labels.mean()),
        "mean_corrected_event_return": float(finite_returns.mean()),
        "median_corrected_event_return": float(finite_returns.median()),
        "corrected_event_return_standard_deviation": float(
            finite_returns.std(ddof=0)
        ),
        "minimum_corrected_event_return": float(finite_returns.min()),
        "maximum_corrected_event_return": float(finite_returns.max()),
        "average_winning_return": average_gain,
        "average_losing_return": average_loss,
        "payoff_ratio": _safe_ratio(average_gain, abs(average_loss)),
        "gross_profit_sum": gross_gain,
        "gross_loss_absolute_sum": gross_loss_magnitude,
        "profit_factor": _safe_ratio(gross_gain, gross_loss_magnitude),
        "upper_barrier_events": int(barrier_counts.get("upper", 0)),
        "lower_barrier_events": int(barrier_counts.get("lower", 0)),
        "vertical_barrier_events": int(barrier_counts.get("vertical", 0)),
    }

    if holding_column in frame.columns:
        holding = pd.to_numeric(frame[holding_column], errors="coerce")
        finite_holding = holding[np.isfinite(holding.to_numpy(dtype=float))]
        result.update({
            "mean_label_event_holding_period_observations": float(
                finite_holding.mean()
            ),
            "median_label_event_holding_period_observations": float(
                finite_holding.median()
            ),
        })

    return result


def grouped_corrected_event_outcome_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    selected_column: str | None = None,
) -> pd.DataFrame:
    """Return corrected event-outcome metrics by a deterministic group."""
    if group_column not in frame.columns:
        raise KeyError(f"Grouping column is missing: {group_column}")

    source = frame
    if selected_column is not None:
        if selected_column not in frame.columns:
            raise KeyError(f"Selection column is missing: {selected_column}")
        source = frame.loc[frame[selected_column].astype(bool)].copy()

    rows: list[dict[str, object]] = []
    for group_value, group_frame in source.groupby(group_column, sort=True):
        row = corrected_event_outcome_metrics(group_frame)
        row[group_column] = group_value
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    columns = [group_column] + [
        column for column in rows[0] if column != group_column
    ]
    return pd.DataFrame(rows)[columns]


def selected_vs_candidate_corrected_outcome_summary(
    frame: pd.DataFrame,
    *,
    selected_column: str = "selected_signal",
) -> pd.DataFrame:
    """Compare selected signals with the complete candidate population."""
    if selected_column not in frame.columns:
        raise KeyError(f"Selection column is missing: {selected_column}")

    rows: list[dict[str, object]] = []
    for population_name, population_frame in [
        ("all_unseen_test_candidates", frame),
        (
            "frozen_policy_selected_signals",
            frame.loc[frame[selected_column].astype(bool)],
        ),
    ]:
        row = corrected_event_outcome_metrics(population_frame)
        row["population"] = population_name
        rows.append(row)

    result = pd.DataFrame(rows)
    columns = ["population"] + [
        column for column in result.columns if column != "population"
    ]
    return result[columns]


# Backward-compatible aliases for any local exploratory code.
gross_event_outcome_metrics = corrected_event_outcome_metrics
grouped_gross_event_outcome_metrics = grouped_corrected_event_outcome_metrics
selected_vs_candidate_outcome_summary = (
    selected_vs_candidate_corrected_outcome_summary
)
