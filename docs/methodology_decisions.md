# Methodology Decisions

## Stage 03 — Frozen target

- Primary target: symmetric +15% / -15% Triple Barrier.
- Maximum holding period: 30 trading observations.
- Label design and scenario diagnostics use train only.
- Unseen test is not used to select the target.
- ZigZag does not construct the target.

## Stage 04 — Feature and ZigZag policy

- Feature approval is structural and train-only.
- Target-derived, event-end, censoring, and legacy-label fields are prohibited as model inputs.
- Adjusted OHLC levels are retained for context and label reconstruction, but are not direct model features.
- Legacy `zigzag_up_new_2` and `zigzag_down_new_2` are not direct model features or filter inputs.
- The confirmed ZigZag state is reconstructed with explicit confirmation timestamps.
- Prefix invariance is an integrity requirement.
- The 15% candidate-event threshold is pre-registered; 10% and 20% are train-only sensitivity diagnostics.
- The confirmed ZigZag rule generates the primary long side.
- Within candidate long events, the frozen Triple-Barrier label is interpreted as the take/skip meta-label.
- Unseen-test outcomes remain untouched during Stage 04.


## Stage 04 data-lineage clarification

Two upstream collection conventions were explicitly retained and documented:

1. `priceChange == 0` rows are treated, under the semantics of the retained API
   and collection code, as non-trading/closed-security rows and are removed
   before technical-feature construction.
2. Zero values in the four individual client-type volume/count fields are
   replaced with `1` upstream to prevent division by zero and deliberately encode
   one-sided individual participation as an extreme buyer/seller-power imbalance.

The second rule is a computational continuity convention rather than a literal
data observation. The final model uses `log_power_of_buy` so that the extreme
ratio remains directional but its heavy tail is compressed.


## Stage 04 v7 market-regime extension

Nine explicit equal-weight market-regime features were added without changing
the Triple-Barrier labels, confirmation-gated ZigZag, or 15% candidate rule.

`market_close_location` uses:
- ordinary range: `(close - low) / (high - low)`;
- locked `high == low == close`, above previous market close: `1`;
- locked, below previous market close: `0`;
- locked, equal to previous market close: missing.


## Stage 04 v8 market-calendar scope correction

Market-index consistency and market-regime rolling features are evaluated only
inside the frozen labeled-train feature horizon. The maximum labeled-train date
is derived from the Stage 03 train files at runtime. Raw index observations
after that date are excluded before canonicalization.

This corrects an audit-scope error in v7; it does not relax the consistency
tolerance and does not change any feature formula.
## Stage 05 v2 validation-window balance revision

The original equal-trading-date validation partition produced highly unequal
candidate counts across folds. The revised primary design preserves chronology,
contiguity, anchored training, the 30-trading-day pre-validation gap, and
event-end purging, but chooses validation boundary dates from cumulative
candidate-event start-date counts.

No target label or event outcome is used to set the boundaries. Class balance
is reported only after boundary freezing.
