# Stage 10C — Causal Market Breadth and Early-Move Diagnostic

## Status

Stage 10C is an exploratory, post-hoc diagnostic. It does not retrain the
frozen XGBoost model, recalculate model scores, change the 15% causal ZigZag
candidate population, or claim a new confirmatory test.

## Questions

1. Does restricting `started` to the first three observations (`1 <= started <= 3`)
   improve the positive-label rate?
2. How does signal quality vary across the causal breadth distribution?
3. Does the transition zone behave differently when breadth is recovering versus
   deteriorating?
4. Are the relationships stable across 2021–2024?
5. Does breadth add information beyond the nine existing market-regime features?

## Breadth definition

For each market date and the frozen Stage 02 universe:

`breadth_raw = (positive symbols - negative symbols) / valid-return symbols`

A symbol is positive or negative according to its adjusted-close return from its
previous valid trading observation. Missing or suspended symbols are excluded
from that date's denominator. Unchanged symbols remain in the denominator with
zero numerator contribution.

The causal smoothed variables are:

- `market_breadth_raw`;
- `market_breadth_ema30`;
- `market_breadth_slope5`;
- `market_breadth_scaled_100`.

Recovery requires `-0.30 <= EMA30 <= 0.30` and a positive five-observation
slope. Deterioration uses the same level interval and a negative slope.

## Frozen-score variants

- baseline;
- started 1–3;
- recovery;
- deterioration control;
- started 1–3 plus recovery;
- started 1–3 plus deterioration control.

Every variant is reranked with the frozen deterministic daily Top-5% policy.
Labels and returns are joined only after the selection inputs are fixed and are
never used to filter or rank candidates.

## Interpretation

The outcome tables use the original gross Triple-Barrier event return available
for all 78,189 candidates. These are signal-level diagnostics, not executable
portfolio returns. A later retraining stage is justified only if breadth shows a
stable incremental relationship in the training period, unseen test, annual
subperiods, and correlation audit.
