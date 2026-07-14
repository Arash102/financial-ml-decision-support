"""Leakage-safe binary triple-barrier labeling for chronological partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierParameters:
    """Frozen parameters for binary triple-barrier event labeling."""

    scenario_name: str
    entry_column: str
    high_column: str
    low_column: str
    upper_barrier: float
    lower_barrier: float
    max_holding_period: int
    barrier_monitoring_starts: str = "next_trading_observation"
    same_bar_rule: str = "negative"
    vertical_positive_label: int = 1
    vertical_nonpositive_label: int = 0

    def validate(self) -> None:
        if not self.scenario_name:
            raise ValueError("scenario_name must be defined.")
        if not self.entry_column:
            raise ValueError("entry_column must be defined.")
        if not self.high_column:
            raise ValueError("high_column must be defined.")
        if not self.low_column:
            raise ValueError("low_column must be defined.")
        if not np.isfinite(self.upper_barrier) or self.upper_barrier <= 0:
            raise ValueError("upper_barrier must be a finite positive return.")
        if not np.isfinite(self.lower_barrier) or self.lower_barrier >= 0:
            raise ValueError("lower_barrier must be a finite negative return.")
        if self.lower_barrier <= -1:
            raise ValueError("lower_barrier must be greater than -1.")
        if self.max_holding_period <= 0:
            raise ValueError("max_holding_period must be a positive integer.")
        if self.barrier_monitoring_starts != "next_trading_observation":
            raise ValueError(
                "Only next_trading_observation monitoring is supported."
            )
        if self.same_bar_rule != "negative":
            raise ValueError(
                "Only the conservative negative same-bar rule is supported."
            )
        if self.vertical_positive_label != 1:
            raise ValueError("vertical_positive_label must be 1.")
        if self.vertical_nonpositive_label != 0:
            raise ValueError("vertical_nonpositive_label must be 0.")


def scenario_names_from_config(config: Mapping[str, Any]) -> list[str]:
    """Return configured scenario names in deterministic YAML insertion order."""
    scenarios = config.get("candidate_scenarios")
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise ValueError("candidate_scenarios must be a non-empty mapping.")
    return [str(name) for name in scenarios.keys()]


def parameters_from_config(
    config: Mapping[str, Any],
    scenario_name: str | None = None,
) -> TripleBarrierParameters:
    """Build and validate one scenario from configs/labeling.yaml."""
    if scenario_name is None:
        scenario_name = str(config["selected_scenario"])

    scenarios = config.get("candidate_scenarios")
    if not isinstance(scenarios, Mapping):
        raise ValueError("candidate_scenarios must be defined.")

    if scenario_name not in scenarios:
        raise KeyError(
            f"Unknown labeling scenario {scenario_name!r}. "
            f"Available: {list(scenarios)}"
        )

    scenario = dict(scenarios[scenario_name])
    vertical_rule = dict(config["vertical_barrier_rule"])

    parameters = TripleBarrierParameters(
        scenario_name=str(scenario_name),
        entry_column=str(config["entry_column"]),
        high_column=str(config["high_column"]),
        low_column=str(config["low_column"]),
        upper_barrier=float(scenario["upper_barrier"]),
        lower_barrier=float(scenario["lower_barrier"]),
        max_holding_period=int(scenario["max_holding_period"]),
        barrier_monitoring_starts=str(config["barrier_monitoring_starts"]),
        same_bar_rule=str(config["same_bar_rule"]),
        vertical_positive_label=int(vertical_rule["positive_return"]),
        vertical_nonpositive_label=int(
            vertical_rule["zero_or_negative_return"]
        ),
    )
    parameters.validate()
    return parameters


def all_parameters_from_config(
    config: Mapping[str, Any],
) -> dict[str, TripleBarrierParameters]:
    """Build every configured scenario."""
    return {
        name: parameters_from_config(config, name)
        for name in scenario_names_from_config(config)
    }


def _finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def _base_output_arrays(row_count: int) -> dict[str, np.ndarray]:
    return {
        "event_end_position": np.full(row_count, -1, dtype=np.int64),
        "holding_period_observations": np.full(
            row_count, -1, dtype=np.int64
        ),
        "available_future_observations": np.zeros(
            row_count, dtype=np.int64
        ),
        "upper_barrier_price": np.full(row_count, np.nan, dtype=float),
        "lower_barrier_price": np.full(row_count, np.nan, dtype=float),
        "event_end_price": np.full(row_count, np.nan, dtype=float),
        "event_return": np.full(row_count, np.nan, dtype=float),
        "label": np.full(row_count, np.nan, dtype=float),
        "barrier_touched": np.full(row_count, "none", dtype=object),
        "label_status": np.full(row_count, "unprocessed", dtype=object),
        "censoring_reason": np.full(row_count, "", dtype=object),
        "same_bar_double_touch": np.zeros(row_count, dtype=bool),
        "full_horizon_available": np.zeros(row_count, dtype=bool),
        "eligible_for_modeling": np.zeros(row_count, dtype=bool),
    }


def label_triple_barrier_partition(
    dataframe: pd.DataFrame,
    parameters: TripleBarrierParameters,
    date_column: str = "dEven",
) -> pd.DataFrame:
    """
    Label every row as an event start using only the supplied partition.

    Train and unseen-test partitions must be passed separately. Therefore, an
    event beginning in train never uses an unseen-test row. Events without an
    observed horizontal touch and without the full vertical horizon are marked
    right-censored and excluded from modeling.
    """
    parameters.validate()

    required_columns = {
        date_column,
        parameters.entry_column,
        parameters.high_column,
        parameters.low_column,
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise KeyError(f"Missing labeling columns: {missing_columns}")

    result = dataframe.copy()
    result[date_column] = pd.to_datetime(
        result[date_column], errors="coerce"
    )

    if result[date_column].isna().any():
        raise ValueError("Invalid dates remain in the labeling partition.")
    if not result[date_column].is_monotonic_increasing:
        raise ValueError("Labeling dates are not monotonically increasing.")
    if result[date_column].duplicated().any():
        raise ValueError("Labeling dates are not unique within the symbol.")

    for column in (
        parameters.entry_column,
        parameters.high_column,
        parameters.low_column,
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    dates = result[date_column].to_numpy(dtype="datetime64[ns]")
    entry_prices = result[parameters.entry_column].to_numpy(dtype=float)
    high_prices = result[parameters.high_column].to_numpy(dtype=float)
    low_prices = result[parameters.low_column].to_numpy(dtype=float)

    row_count = len(result)
    horizon = parameters.max_holding_period
    outputs = _base_output_arrays(row_count)

    for start_position in range(row_count):
        entry_price = entry_prices[start_position]
        available_future = row_count - start_position - 1

        outputs["available_future_observations"][
            start_position
        ] = available_future
        outputs["full_horizon_available"][start_position] = (
            available_future >= horizon
        )

        if not _finite_positive(entry_price):
            outputs["label_status"][
                start_position
            ] = "invalid_entry_price"
            outputs["censoring_reason"][
                start_position
            ] = "invalid_entry_price"
            outputs["event_end_position"][start_position] = start_position
            outputs["holding_period_observations"][start_position] = 0
            continue

        upper_price = entry_price * (1.0 + parameters.upper_barrier)
        lower_price = entry_price * (1.0 + parameters.lower_barrier)
        outputs["upper_barrier_price"][start_position] = upper_price
        outputs["lower_barrier_price"][start_position] = lower_price

        last_monitor_position = min(
            start_position + horizon,
            row_count - 1,
        )
        resolved = False

        for monitor_position in range(
            start_position + 1,
            last_monitor_position + 1,
        ):
            high_price = high_prices[monitor_position]
            low_price = low_prices[monitor_position]
            holding_period = monitor_position - start_position

            if (
                not _finite_positive(high_price)
                or not _finite_positive(low_price)
                or high_price < low_price
            ):
                outputs["label_status"][
                    start_position
                ] = "invalid_monitoring_price"
                outputs["censoring_reason"][
                    start_position
                ] = "invalid_high_low_before_event_resolution"
                outputs["event_end_position"][
                    start_position
                ] = monitor_position
                outputs["holding_period_observations"][
                    start_position
                ] = holding_period
                resolved = True
                break

            upper_hit = high_price >= upper_price
            lower_hit = low_price <= lower_price

            if not upper_hit and not lower_hit:
                continue

            outputs["event_end_position"][
                start_position
            ] = monitor_position
            outputs["holding_period_observations"][
                start_position
            ] = holding_period
            outputs["label_status"][start_position] = "labeled"
            outputs["eligible_for_modeling"][start_position] = True

            if upper_hit and lower_hit:
                outputs["same_bar_double_touch"][start_position] = True
                outputs["barrier_touched"][start_position] = "lower"
                outputs["label"][start_position] = 0
                outputs["event_end_price"][start_position] = lower_price
                outputs["event_return"][
                    start_position
                ] = parameters.lower_barrier
            elif lower_hit:
                outputs["barrier_touched"][start_position] = "lower"
                outputs["label"][start_position] = 0
                outputs["event_end_price"][start_position] = lower_price
                outputs["event_return"][
                    start_position
                ] = parameters.lower_barrier
            else:
                outputs["barrier_touched"][start_position] = "upper"
                outputs["label"][start_position] = 1
                outputs["event_end_price"][start_position] = upper_price
                outputs["event_return"][
                    start_position
                ] = parameters.upper_barrier

            resolved = True
            break

        if resolved:
            continue

        if available_future < horizon:
            censor_position = row_count - 1
            outputs["event_end_position"][
                start_position
            ] = censor_position
            outputs["holding_period_observations"][
                start_position
            ] = available_future
            outputs["label_status"][
                start_position
            ] = "right_censored"
            outputs["censoring_reason"][
                start_position
            ] = "partition_ended_before_vertical_barrier"
            continue

        vertical_position = start_position + horizon
        vertical_price = entry_prices[vertical_position]
        outputs["event_end_position"][
            start_position
        ] = vertical_position
        outputs["holding_period_observations"][
            start_position
        ] = horizon

        if not _finite_positive(vertical_price):
            outputs["label_status"][
                start_position
            ] = "invalid_vertical_price"
            outputs["censoring_reason"][
                start_position
            ] = "invalid_vertical_exit_price"
            continue

        vertical_return = vertical_price / entry_price - 1.0
        outputs["barrier_touched"][start_position] = "vertical"
        outputs["event_end_price"][start_position] = vertical_price
        outputs["event_return"][start_position] = vertical_return
        outputs["label"][start_position] = (
            parameters.vertical_positive_label
            if vertical_return > 0
            else parameters.vertical_nonpositive_label
        )
        outputs["label_status"][start_position] = "labeled"
        outputs["eligible_for_modeling"][start_position] = True

    event_end_dates = np.full(
        row_count,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )
    valid_end_positions = outputs["event_end_position"] >= 0
    event_end_dates[valid_end_positions] = dates[
        outputs["event_end_position"][valid_end_positions]
    ]

    result["labeling_scenario"] = parameters.scenario_name
    result["event_start_date"] = result[date_column]
    result["event_end_date"] = pd.to_datetime(event_end_dates)
    result["event_end_position"] = pd.array(
        np.where(
            valid_end_positions,
            outputs["event_end_position"],
            None,
        ),
        dtype="Int64",
    )
    result["holding_period_observations"] = pd.array(
        np.where(
            outputs["holding_period_observations"] >= 0,
            outputs["holding_period_observations"],
            None,
        ),
        dtype="Int64",
    )
    result["available_future_observations"] = outputs[
        "available_future_observations"
    ]
    result["upper_barrier_price"] = outputs["upper_barrier_price"]
    result["lower_barrier_price"] = outputs["lower_barrier_price"]
    result["event_end_price"] = outputs["event_end_price"]
    result["event_return"] = outputs["event_return"]
    result["barrier_touched"] = outputs["barrier_touched"]
    result["same_bar_double_touch"] = outputs[
        "same_bar_double_touch"
    ]
    result["full_horizon_available"] = outputs[
        "full_horizon_available"
    ]
    result["label_status"] = outputs["label_status"]
    result["censoring_reason"] = outputs["censoring_reason"]
    result["eligible_for_modeling"] = outputs[
        "eligible_for_modeling"
    ]
    result["label"] = pd.array(outputs["label"], dtype="Int64")

    return result
