# Market-data collection conventions

## `priceChange == 0`

The upstream API returns rows for dates on which a security may be closed or
otherwise not actively traded. In this project, `priceChange == 0` is used by the
collection pipeline as the provider-specific indicator for those non-trading
rows, and such rows are removed before technical-feature and event construction.

This convention is specific to the retained data source and collection code. It
should not be generalized to unrelated market datasets without verifying the
provider's semantics.

## Individual client-type zero handling

Before buyer-power calculation and before the retained raw-data snapshot is
stored, the collection pipeline replaces zero values in:

- `buy_I_Volume`
- `buy_I_Count`
- `sell_I_Volume`
- `sell_I_Count`

with `1`.

The rule serves two project-specific purposes:

1. prevent division by zero;
2. encode days with one-sided individual participation as an intentionally
   extreme buyer/seller-power imbalance.

For example, when individual buyers participate but individual sellers do not,
the seller-side individual average is regularized to the continuity floor and
buyer power becomes very large. The reverse case produces a very small ratio.

The value `1` is therefore a computational continuity/sentinel convention and
must not be interpreted as a literal observation of one share or one investor.

The final pooled-model input is:

`log_power_of_buy = ln((buy_I_Volume / buy_I_Count) / (sell_I_Volume / sell_I_Count))`

using the collection-pipeline fields. The log transform compresses extreme
ratios and maps equal buyer/seller power to zero, buyer dominance to positive
values, and seller dominance to negative values.
