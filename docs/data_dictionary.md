# Data Dictionary

## Metadata

| Column | Role | Model input |
|---|---|---|
| `insCode` | Security identifier | No |
| `dEven` | Market date | No |

## Adjusted price fields

| Column | Role |
|---|---|
| `adj_open` | Adjusted opening price |
| `adj_high` | Adjusted daily high |
| `adj_low` | Adjusted daily low |
| `adj_last_price` | Adjusted reference/entry price |

## Candidate features
Candidate features are listed in `configs/columns.yaml`. Their presence in the
prepared dataset does not prove that they are leakage-free.

## Legacy audit fields

| Column | Status |
|---|---|
| `class` | Legacy target; not the new target |
| `max_price_find` | Future-derived audit field |
| `min_price_find` | Future-derived audit field |
| `after_days` | Future-derived audit field |
| `days_max_find` | Future-derived audit field |

These fields are written to a separate directory and must never enter model matrices.
