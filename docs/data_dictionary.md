# Data Dictionary

## Stage 03 label fields

| Field | Meaning |
|---|---|
| `labeling_scenario` | Frozen barrier scenario used for the saved labeled dataset |
| `event_start_date` | Date on which the event starts |
| `event_end_date` | First horizontal-barrier touch date or vertical/censoring end date |
| `event_end_position` | Zero-based row position of the event end within the partition |
| `holding_period_observations` | Number of trading observations from event start to event end |
| `available_future_observations` | Number of later observations available inside the same partition |
| `upper_barrier_price` | Entry price multiplied by one plus the upper barrier |
| `lower_barrier_price` | Entry price multiplied by one plus the lower barrier |
| `event_end_price` | Barrier price or vertical exit price |
| `event_return` | Return represented by the resolved event |
| `barrier_touched` | `upper`, `lower`, `vertical`, or `none` |
| `same_bar_double_touch` | True when both horizontal barriers are touched on the same observation |
| `full_horizon_available` | Whether the complete vertical horizon exists in the same partition |
| `label_status` | `labeled`, `right_censored`, or an invalid-price status |
| `censoring_reason` | Reason an event was excluded |
| `eligible_for_modeling` | True only for valid, uncensored binary outcomes |
| `label` | Binary outcome: 1 for positive and 0 for negative |

## ZigZag fields

`zigzag_up_new_2` and `zigzag_down_new_2` remain candidate features and potential event-filter inputs. They are not used to construct the Stage 03 label. Their timing and sign conventions are audited in Notebook 04 before any filtering or meta-labeling rule is frozen.
