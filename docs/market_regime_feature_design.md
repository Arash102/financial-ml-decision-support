# Equal-weight market-regime feature design

## Source columns

- `xNivInuClMresIbs`: equal-weight index close/current level
- `xNivInuPbMresIbs`: equal-weight index daily low
- `xNivInuPhMresIbs`: equal-weight index daily high

## Canonical market calendar

The index columns are repeated inside symbol raw files. A symbol file may omit
dates when that security did not trade. Stage 04 therefore pools repeated index
observations across all frozen symbols, checks same-date consistency, and
computes rolling market features once on one canonical market calendar.

## Frozen market features

1. `market_return_1 = I_t / I_(t-1) - 1`
2. `market_return_5 = I_t / I_(t-5) - 1`
3. `market_return_20 = I_t / I_(t-20) - 1`
4. `market_volatility_20`: population standard deviation of 20 causal one-day
   log returns
5. `market_ema_20_distance = (I_t - EMA20_t) / I_t`
6. `market_ema_60_distance = (I_t - EMA60_t) / I_t`
7. `market_range_fraction = (high - low) / close`
8. `market_close_location = (close - low) / (high - low)`
9. `market_drawdown_60 = I_t / rolling_max_60(I) - 1`

## Locked market range

When `high == low == close`:

- current close > previous market close -> `1`
- current close < previous market close -> `0`
- current close == previous market close -> missing

The equality case remains missing rather than receiving an invented direction.

## Causality

Every feature uses only current and prior canonical market observations.


## Stage 04 train-horizon scope correction

The Stage 04 canonical index calendar is truncated at the maximum date actually
present in the frozen labeled-train files before cross-file consistency testing
and before rolling market-feature calculation.

The first v7 audit accidentally tested the complete raw-data history, including
post-train dates that are outside Stage 04 feature approval. The uploaded audit
showed that every flagged cross-file difference occurred after the frozen train
cutoff; there were zero flagged index-field inconsistencies on or before
2021-03-20.

Post-train raw market-index values are not used to construct Stage 04 features.
