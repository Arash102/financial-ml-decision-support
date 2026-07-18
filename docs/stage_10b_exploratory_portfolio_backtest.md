# Stage 10B — Exploratory/Post-hoc Portfolio Backtest

## Status

Stage 10B is an exploratory, post-hoc diagnostic stage. The confirmatory Stage 10
portfolio result was observed before these rules were specified. Therefore Stage
10B cannot be presented as a second confirmatory unseen-test result.

## Frozen inputs

- Stage 09 inference lock SHA-256:
  `c29f1ec3b6d59fc5a2aa163f65b880562271f938fc7208a784ee820f5245c946`
- Frozen candidate population: 78,189 rows across 820 dates.
- Frozen XGBoost scores are reused exactly; the model is not retrained and scores
  are not recalculated.
- The 15% confirmation-gated causal ZigZag candidate rule is inherited from the
  locked Stage 09 candidate population. It is not recomputed or changed.

## Final signal filter

The existing `started` column is joined by symbol and date from
`data_ready/unseen_test`; `raw_data` is used only as a fallback when the first
source cannot provide a complete valid join.

The filter is exact:

```text
started != 0
```

Only `0` fails the filter. Every valid nonzero integer value, including `1`, `2`, `3`, and higher values, passes.

The operation order is:

1. Load the full Stage 09 inference lock.
2. Join the already-computed `started` value.
3. Retain only rows with `started != 0`.
4. Preserve the exact Stage 09 15% causal ZigZag eligibility.
5. Re-rank the remaining candidates by score descending, symbol ascending, and
   event ID ascending.
6. Apply the frozen daily top-5% rule with at least one selected signal on every
   date that still contains a candidate.
7. Enter at the next valid trading open subject to portfolio constraints.

This ordering prevents a removed `started=0` row from consuming a
place in the daily signal quota.

## Portfolio rules held fixed

- Long only, no leverage.
- Entry at next trading observation open.
- 0.5% planned equity risk per lot.
- Initial stop: 15% below buy execution price.
- Maximum portfolio planned open risk: 10%.
- Maximum gross exposure: 70%.
- Up to three independent lots per symbol in the multi-lot scenarios.
- Trailing activation at +15%, then 10% below the highest observed high with a
  net break-even floor, effective from the next symbol trading open.
- Fixed +15% take-profit is retained as a control.
- Maximum horizon: 30 symbol trading observations.
- Adverse stop first for ambiguous daily bars.
- Buy fee 0.464%, sell deduction 0.964%, and 0.20% slippage on each side.

## Exploratory scenario grid

The grid isolates liquidity and portfolio-capacity effects while holding primary
capital and 20 bp per-side slippage fixed.

### Liquidity profiles

| Profile | Maximum lot notional |
|---|---:|
| `adv01` | 1% of trailing ADV20 |
| `adv10` | 10% of trailing ADV20 |
| `adv50` | 50% of trailing ADV20 |

`adv50` is an optimistic capacity upper bound requested by the research owner.
It must not be described as a conservative executable assumption.

### Capacity profiles

| Profile | Symbols | Open lots | New lots/day | Symbol exposure |
|---|---:|---:|---:|---:|
| `current` | 10 | 20 | 3 | 10% |
| `expanded` | 20 | 40 | 5 | 20% |
| `broad` | 30 | 60 | 10 | 20% |

Each profile is evaluated with multi-lot and single-lot structures and with
trailing and fixed-take-profit exits, for 36 scenarios total.

The requested revised operating case is:

```text
liq_adv50__cap_broad__multi_lot__trailing
```

## Market regime

No market-regime filter is added in this version. The cyclicality argument is
scientifically relevant, but no exact causal formula and threshold were frozen.
Inventing one after observing Stage 10 would add an uncontrolled post-result
optimization.

## Output isolation

All files use a `10b_` prefix. Stage 10 outputs are not overwritten.

Main outputs:

- `results/predictions/10b_started_nonzero_zigzag15_filtered_inference.csv`
- `results/predictions/10b_started_nonzero_zigzag15_selected_signals.csv`
- `results/backtests/10b_scenario_summary.csv`
- `results/backtests/10b_trade_ledger.csv`
- `results/backtests/10b_signal_decisions.csv`
- `results/backtests/10b_daily_equity.csv`
- `results/manifests/10b_exploratory_portfolio_backtest_manifest.json`

## Interpretation rule

Stage 10 remains the confirmatory portfolio result. Stage 10B may diagnose and
motivate a future policy, but any winning Stage 10B specification requires a new
untouched temporal sample before it can support a confirmatory profitability
claim.
