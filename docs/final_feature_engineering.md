# Final pooled-model feature engineering

Stage 04 was extended after the pooled global-model architecture was reviewed.
Raw price-level indicators are not used directly because stock price levels vary
substantially across securities.

## Final semantic inputs

There are 35 semantic inputs: 34 numeric and one categorical GMMA state.

### Trend location
Thirteen signed EMA distances:
`(adj_last_price - EMA_N) / adj_last_price`.

### Momentum
- centered RSI: `(RSI_14 - 50) / 50`
- relative MACD: `macd / adj_last_price`

### Investor behavior and activity
- `log_power_of_buy`: log ratio of average individual buyer volume to average
  individual seller volume. The collection pipeline applies an explicit `0 -> 1`
  continuity rule to the four individual client-type source fields before the
  raw-data snapshot is stored. This preserves intentionally extreme values for
  one-sided individual participation; the log transform compresses the tail.
- `log_volume_ratio_30`: log1p of daily total buy volume divided by its causal
  30-observation rolling mean.
- `ho_buy_fraction`: institutional buy volume divided by total buy volume.
- `ho_sell_fraction`: institutional sell volume divided by total sell volume.
  This corrects the legacy denominator bug.

### Relative market position
- current-date stock/index relative-strength z-score over 12 observations.
- standard adjusted-price return z-score over 12 returns.
- log1p of the consecutive run length with positive relative-strength z-score.

### Price action
`body_ratio` is the only intentionally unadjusted-price feature.

Ordinary bar:
`(pDrCotVal - priceFirst) / (priceMax - priceMin)`.

Locked price-limit bar where `high == low == open == last`:
- same-day raw `priceChange > 0`: `+1`
- same-day raw `priceChange < 0`: `-1`
- same-day raw `priceChange == 0`: `0`

The locked-bar direction is read directly from raw `priceChange`. No previous-day
lookup and no structural-break guard are used for this special case.

### EMA structure
GMMA is categorical: `bullish`, `bearish`, or `mixed`. It is not encoded as an
ordinal 1/2/3 variable.

### Confirmed ZigZag geometry
Both confirmation-gated ZigZag distances remain inputs to the meta-model.

## Explicitly removed
Raw EMA levels, raw MACD, legacy buyer-power, legacy volume ratio, legacy
`ho_buy`, legacy `ho_sell`, legacy `body`, `color`, raw `started`, legacy
`zigzag_up_new_2`, legacy `zigzag_down_new_2`, and `RSI_Signal`.


## Feature provenance split

The 26 final semantic inputs have two provenance classes:

- 24 deterministic pooled-feature inputs are reconstructed from labeled-train
  and raw same-day/historical data: 23 numeric plus categorical `gmma_state`.
- 2 confirmed ZigZag geometry inputs are carried from the frozen Stage 04
  candidate-generation state:
  `distance_above_confirmed_low_fraction` and
  `distance_below_confirmed_high_fraction`.

The final feature-engineering function does not recreate or overwrite the
confirmed ZigZag geometry.


## Data-source conventions inherited from collection

### Zero `priceChange`

For this project's market-data provider, rows with `priceChange == 0` are treated
as non-trading/closed-security observations returned by the API. The collection
pipeline removes these rows before technical-feature construction. This is a
provider-specific data-semantic rule and is not asserted as a universal rule for
all market datasets.

### Individual client-type zeros

The collection pipeline replaces zero values in:

- `buy_I_Volume`
- `buy_I_Count`
- `sell_I_Volume`
- `sell_I_Count`

with `1` before the current raw-data snapshot is stored.

The purpose is to avoid division by zero and to encode one-sided individual
participation as an extreme buyer/seller-power imbalance. The substituted `1`
is a computational continuity convention, not a literal observed volume or
participant count. `log_power_of_buy` retains this lineage and applies a natural
log transform to compress extreme ratios while preserving buyer-versus-seller
direction.


## Equal-weight market-regime extension

Stage 04 v7 adds nine causal equal-weight market-regime inputs. They are built
on one canonical market calendar pooled from repeated index fields in the frozen
symbol raw files.

See `docs/market_regime_feature_design.md`.
