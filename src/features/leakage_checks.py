"""Leakage audits and causal confirmed-ZigZag state reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfirmedZigZagConfig:
    """Parameters matching the original ZigZag design intent."""

    depth: int = 10
    deviation_percent: float = 15.0

    @property
    def half_window(self) -> int:
        return self.depth // 2


def _confirmed_local_pivots(
    values: np.ndarray,
    *,
    is_high: bool,
    config: ConfirmedZigZagConfig,
) -> list[dict[str, object]]:
    """
    Detect local pivots and assign the first time each pivot is confirmed.

    A pivot uses `depth // 2` observations on each side. After the right
    local window is complete, the original design waits `depth` additional
    observations while rejecting a pivot that is broken before confirmation.
    """
    values = np.asarray(values, dtype=float)
    n_rows = len(values)
    half = config.half_window
    events: list[dict[str, object]] = []

    for pivot_index in range(half, n_rows - half):
        pivot_price = values[pivot_index]
        if not np.isfinite(pivot_price):
            continue

        left = values[pivot_index - half : pivot_index]
        right = values[pivot_index + 1 : pivot_index + half + 1]

        if (
            len(left) != half
            or len(right) != half
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
        ):
            continue

        if is_high:
            is_local_pivot = (
                not np.any(left > pivot_price)
                and not np.any(right >= pivot_price)
            )
        else:
            is_local_pivot = (
                not np.any(left < pivot_price)
                and not np.any(right <= pivot_price)
            )

        if not is_local_pivot:
            continue

        confirmation_start = pivot_index + half + 1
        confirmation_index: int | None = None

        for current_index in range(confirmation_start, n_rows):
            current_value = values[current_index]
            if not np.isfinite(current_value):
                continue

            invalidated = (
                current_value > pivot_price
                if is_high
                else current_value < pivot_price
            )
            if invalidated:
                confirmation_index = None
                break

            if current_index - confirmation_start >= config.depth:
                confirmation_index = current_index
                break

        if confirmation_index is None:
            continue

        events.append(
            {
                "pivot_type": "high" if is_high else "low",
                "pivot_index": int(pivot_index),
                "pivot_price": float(pivot_price),
                "confirmation_index": int(confirmation_index),
            }
        )

    return events


def _accept_confirmed_pivot(
    accepted: list[dict[str, object]],
    event: dict[str, object],
    *,
    deviation_percent: float,
) -> None:
    """
    Update the online accepted-pivot sequence at the event confirmation time.

    Same-type pivots may replace the latest accepted same-type pivot only when
    the newly confirmed pivot is more extreme. Opposite-type pivots are accepted
    only after the configured minimum percentage deviation.
    """
    if not accepted:
        accepted.append(dict(event))
        return

    last = accepted[-1]
    event_type = str(event["pivot_type"])
    last_type = str(last["pivot_type"])
    event_price = float(event["pivot_price"])
    last_price = float(last["pivot_price"])

    if event_type == last_type:
        more_extreme = (
            event_price > last_price
            if event_type == "high"
            else event_price < last_price
        )
        if more_extreme:
            accepted[-1] = dict(event)
        return

    if last_price <= 0:
        return

    deviation = abs(100.0 * (event_price - last_price) / last_price)
    if deviation >= deviation_percent:
        accepted.append(dict(event))


def build_confirmation_gated_zigzag_state(
    dataframe: pd.DataFrame,
    *,
    config: ConfirmedZigZagConfig,
    date_column: str = "dEven",
    high_column: str = "adj_high",
    low_column: str = "adj_low",
    close_column: str = "adj_last_price",
) -> pd.DataFrame:
    """
    Build a causal ZigZag state using confirmation timestamps directly.

    The function processes pivot confirmations in chronological order. A pivot
    cannot affect a row before its confirmation index. The resulting state is
    therefore suitable for prefix-invariance auditing.
    """
    frame = dataframe.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.sort_values(date_column, kind="stable").reset_index(drop=True)

    high = pd.to_numeric(frame[high_column], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame[low_column], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(frame[close_column], errors="coerce").to_numpy(dtype=float)
    dates = frame[date_column].to_numpy(dtype="datetime64[ns]")

    events = _confirmed_local_pivots(
        high,
        is_high=True,
        config=config,
    )
    events.extend(
        _confirmed_local_pivots(
            low,
            is_high=False,
            config=config,
        )
    )
    events.sort(
        key=lambda row: (
            int(row["confirmation_index"]),
            int(row["pivot_index"]),
            0 if row["pivot_type"] == "low" else 1,
        )
    )

    events_by_confirmation: dict[int, list[dict[str, object]]] = {}
    for event in events:
        events_by_confirmation.setdefault(
            int(event["confirmation_index"]),
            [],
        ).append(event)

    n_rows = len(frame)
    high_price = np.full(n_rows, np.nan, dtype=float)
    low_price = np.full(n_rows, np.nan, dtype=float)
    high_pivot_index = np.full(n_rows, -1, dtype=int)
    low_pivot_index = np.full(n_rows, -1, dtype=int)
    high_confirmation_index = np.full(n_rows, -1, dtype=int)
    low_confirmation_index = np.full(n_rows, -1, dtype=int)

    accepted: list[dict[str, object]] = []
    latest_high: dict[str, object] | None = None
    latest_low: dict[str, object] | None = None

    for current_index in range(n_rows):
        for event in events_by_confirmation.get(current_index, []):
            before = list(accepted)
            _accept_confirmed_pivot(
                accepted,
                event,
                deviation_percent=config.deviation_percent,
            )

            if accepted != before:
                latest_high = next(
                    (
                        pivot
                        for pivot in reversed(accepted)
                        if pivot["pivot_type"] == "high"
                    ),
                    None,
                )
                latest_low = next(
                    (
                        pivot
                        for pivot in reversed(accepted)
                        if pivot["pivot_type"] == "low"
                    ),
                    None,
                )

        if latest_high is not None:
            high_price[current_index] = float(latest_high["pivot_price"])
            high_pivot_index[current_index] = int(latest_high["pivot_index"])
            high_confirmation_index[current_index] = int(
                latest_high["confirmation_index"]
            )

        if latest_low is not None:
            low_price[current_index] = float(latest_low["pivot_price"])
            low_pivot_index[current_index] = int(latest_low["pivot_index"])
            low_confirmation_index[current_index] = int(
                latest_low["confirmation_index"]
            )

    with np.errstate(divide="ignore", invalid="ignore"):
        distance_above_low = (close - low_price) / close
        signed_distance_from_high = (close - high_price) / close
        distance_below_high = (high_price - close) / close

    invalid_close = ~np.isfinite(close) | (close <= 0)
    for values in (
        distance_above_low,
        signed_distance_from_high,
        distance_below_high,
    ):
        values[invalid_close] = np.nan
        values[~np.isfinite(values)] = np.nan

    result = pd.DataFrame(
        {
            date_column: frame[date_column],
            "confirmed_zigzag_high_price": high_price,
            "confirmed_zigzag_low_price": low_price,
            "confirmed_zigzag_high_pivot_index": high_pivot_index,
            "confirmed_zigzag_low_pivot_index": low_pivot_index,
            "confirmed_zigzag_high_confirmation_index": high_confirmation_index,
            "confirmed_zigzag_low_confirmation_index": low_confirmation_index,
            "distance_above_confirmed_low_fraction": distance_above_low,
            "signed_distance_from_confirmed_high_fraction": signed_distance_from_high,
            "distance_below_confirmed_high_fraction": distance_below_high,
        }
    )

    result["confirmed_zigzag_high_pivot_date"] = pd.NaT
    result["confirmed_zigzag_low_pivot_date"] = pd.NaT
    result["confirmed_zigzag_high_confirmation_date"] = pd.NaT
    result["confirmed_zigzag_low_confirmation_date"] = pd.NaT

    valid_high_pivot = high_pivot_index >= 0
    valid_low_pivot = low_pivot_index >= 0
    valid_high_confirmation = high_confirmation_index >= 0
    valid_low_confirmation = low_confirmation_index >= 0

    result.loc[
        valid_high_pivot,
        "confirmed_zigzag_high_pivot_date",
    ] = dates[high_pivot_index[valid_high_pivot]]

    result.loc[
        valid_low_pivot,
        "confirmed_zigzag_low_pivot_date",
    ] = dates[low_pivot_index[valid_low_pivot]]

    result.loc[
        valid_high_confirmation,
        "confirmed_zigzag_high_confirmation_date",
    ] = dates[high_confirmation_index[valid_high_confirmation]]

    result.loc[
        valid_low_confirmation,
        "confirmed_zigzag_low_confirmation_date",
    ] = dates[low_confirmation_index[valid_low_confirmation]]

    result["confirmed_zigzag_state_available"] = (
        np.isfinite(high_price)
        & np.isfinite(low_price)
    )

    return result


def build_candidate_long_mask(
    state: pd.DataFrame,
    *,
    eligible_for_modeling: pd.Series,
    threshold_fraction: float,
) -> pd.Series:
    """Build the frozen primary long-side candidate rule."""
    low_distance = pd.to_numeric(
        state["distance_above_confirmed_low_fraction"],
        errors="coerce",
    )
    high_distance = pd.to_numeric(
        state["distance_below_confirmed_high_fraction"],
        errors="coerce",
    )
    eligible = eligible_for_modeling.fillna(False).astype(bool)

    return (
        eligible
        & state["confirmed_zigzag_state_available"].fillna(False).astype(bool)
        & low_distance.ge(0.0)
        & low_distance.le(float(threshold_fraction))
        & high_distance.ge(float(threshold_fraction))
    )


def prefix_invariance_audit(
    dataframe: pd.DataFrame,
    *,
    config: ConfirmedZigZagConfig,
    positions: Iterable[int],
    date_column: str = "dEven",
) -> pd.DataFrame:
    """Recompute causal state on prefixes and compare the last row to full history."""
    frame = dataframe.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.sort_values(date_column, kind="stable").reset_index(drop=True)

    full_state = build_confirmation_gated_zigzag_state(
        frame,
        config=config,
        date_column=date_column,
    )

    compare_columns = [
        "confirmed_zigzag_high_price",
        "confirmed_zigzag_low_price",
        "confirmed_zigzag_high_confirmation_index",
        "confirmed_zigzag_low_confirmation_index",
        "distance_above_confirmed_low_fraction",
        "distance_below_confirmed_high_fraction",
    ]

    rows: list[dict[str, object]] = []

    for position in sorted(set(int(value) for value in positions)):
        if position < 0 or position >= len(frame):
            continue

        prefix_state = build_confirmation_gated_zigzag_state(
            frame.iloc[: position + 1].copy(),
            config=config,
            date_column=date_column,
        )

        prefix_row = prefix_state.iloc[-1]
        full_row = full_state.iloc[position]

        mismatch_columns: list[str] = []

        for column in compare_columns:
            left = prefix_row[column]
            right = full_row[column]

            if pd.isna(left) and pd.isna(right):
                continue

            if column.endswith("_index"):
                equal = int(left) == int(right)
            else:
                equal = bool(
                    np.isclose(
                        float(left),
                        float(right),
                        rtol=1e-12,
                        atol=1e-12,
                        equal_nan=True,
                    )
                )

            if not equal:
                mismatch_columns.append(column)

        rows.append(
            {
                "position": position,
                "event_date": frame.loc[position, date_column],
                "prefix_invariant": not mismatch_columns,
                "mismatch_columns": "|".join(mismatch_columns),
            }
        )

    return pd.DataFrame(rows)


def legacy_zigzag_source_logic_audit() -> pd.DataFrame:
    """
    Document the timing distinction observed in the supplied collection code.

    The legacy code records confirmation indices but the `new_2` distance loop
    scans `zj` pivot rows and skips the first encountered pivot. Stage 04 therefore
    does not treat the legacy columns as formally confirmation-gated.
    """
    rows = [
        {
            "check": "confirmation_index_is_computed",
            "observed": True,
            "risk": False,
            "stage_04_action": "retain design intent",
        },
        {
            "check": "confirmation_markers_are_created",
            "observed": True,
            "risk": False,
            "stage_04_action": "retain design intent",
        },
        {
            "check": "legacy_new2_loop_scans_pivot_marker_zj",
            "observed": True,
            "risk": True,
            "stage_04_action": "do not use legacy new_2 columns directly",
        },
        {
            "check": "legacy_new2_loop_explicitly_requires_confirmation_marker",
            "observed": False,
            "risk": True,
            "stage_04_action": "reconstruct confirmation-gated state",
        },
        {
            "check": "legacy_new2_loop_skips_first_pivot",
            "observed": True,
            "risk": True,
            "stage_04_action": "replace heuristic with confirmation timestamp gate",
        },
        {
            "check": "stage04_state_requires_confirmation_index_at_or_before_row",
            "observed": True,
            "risk": False,
            "stage_04_action": "required",
        },
        {
            "check": "stage04_state_is_prefix_invariance_audited",
            "observed": True,
            "risk": False,
            "stage_04_action": "required",
        },
    ]
    return pd.DataFrame(rows)
