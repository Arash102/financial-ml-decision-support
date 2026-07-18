# Stage 10 — Executable Long-Only Multi-Lot Portfolio Backtest

## Status

`stage_10_preregistered_v1`

This stage converts the frozen Stage 09 signal stream into an executable,
capital-constrained long-only portfolio simulation. It must not retrain the
model, alter the top-5% policy, use Stage 09 outcomes for selection, or tune
rules after portfolio results are observed.

## Frozen upstream lineage

- Stage 09 producer commit:
  `d7112980a69a99e0c183561bed2ff8b11b72bd33`
- Stage 09 tag:
  `milestone-09-v3-confirmatory-signal-evaluation`
- Frozen inference lock SHA-256:
  `c29f1ec3b6d59fc5a2aa163f65b880562271f938fc7208a784ee820f5245c946`
- Frozen selected signals: `4,311`
- Signal-generation period: `2021-03-21` through `2024-09-22`
- Outcome/execution tail: through `2024-10-26`

The Stage 09 selected-signal file may contain outcome columns, but Stage 10
deliberately reads only signal identity, symbol, date, score, rank, quota, and
selection status before reconstructing executions from raw adjusted OHLC data.

## Market direction

The simulated system is strictly **long-only**.

- It can buy and later sell to close a long position.
- It cannot establish short positions.
- A negative model outcome never creates an inverse trade.
- Leverage is prohibited.

## Initial capital in Iranian rials

The customer feature file contains 500 traders and reports `avg_Buy` in IRR.
It does not contain daily total wealth or cash balances. Consequently, Stage 10
uses a **customer transaction-size-anchored initial-capital proxy** rather than
claiming to observe average total trader wealth.

For each of the 500 traders, the `avg_Buy` value is retained. The primary
transaction-size anchor is the 10% two-sided trimmed mean:

- arithmetic mean `avg_Buy`: 6,094,947,160.18146 IRR
- median `avg_Buy`: 442,389,804.85 IRR
- 10% trimmed mean `avg_Buy`: 620,315,055.56125 IRR

With 0.5% planned risk per lot, a 15% stop, 0.464% buy deductions, 0.964% sell
deductions, and 0.20% slippage on each side, the planned loss fraction relative
to total entry cash is:

`0.16375777591973245`

The implied initial lot weight is:

`0.005 / 0.16375777591973245 = 0.03053290124342432`

The primary capital proxy is therefore:

`620,315,055.56125 / 0.03053290124342432 = 20,316,282,773.64711 IRR`

Sensitivity capital values are:

- median proxy: 14,488,954,106.360094 IRR
- arithmetic-mean proxy: 199,618,998,259.92108 IRR

## Execution

A signal becomes known only after its signal-day bar is complete.

- Entry: adjusted open of the next trading observation for that symbol.
- Validity: one next observation only; rejected signals are not carried forward.
- Buy slippage: primary 0.20%.
- Sell slippage: primary 0.20%.
- Quantity: positive integer, rounded down.
- Cash from an opening gap exit may be reused for new entries at that same
  opening timestamp.
- Daily OHLC ambiguity is resolved conservatively: adverse stop first.

Adjusted prices create synthetic share quantities. Return accounting is
internally consistent on the adjusted price scale, but share counts must not be
interpreted as reconstructed historical registered share quantities.

## Fees and tax

The user-supplied 1404 Tehran Stock Exchange schedule is applied uniformly:

- total buy deductions: 0.464%
- total sell deductions: 0.964%
- sell tax contained in the sell deduction: 0.500%
- non-tax sell deductions: 0.464%

The schedule is held constant over 2021–2024. This is an explicit scenario
assumption, not a reconstruction of historical fee changes. A frozen
exchange-board classification is not available in this stage, so the supplied
TSE schedule is conservatively applied to all symbols.

## Independent repeated-signal lots

A symbol may contain one net long position composed of multiple independently
tracked lots.

- Maximum three open lots per symbol.
- Each frozen repeated signal may create one new lot if all constraints pass.
- Existing lots are not replaced.
- Existing horizons are not reset.
- Each lot has its own entry, stop, trailing state, high-water mark, and
  30-observation time limit.
- The single-lot-per-symbol implementation is retained as a control scenario.

This design tests whether discarding later signals while a symbol is already
held would remove economically valuable entries.

## Risk and capacity

Primary limits:

- risk per new lot: 0.5% of current equity
- maximum planned open risk per symbol: 1.5%
- maximum planned open risk for the portfolio: 10%
- maximum symbol exposure: 10%
- maximum gross exposure: 70%
- maximum open lots: 20
- maximum open symbols: 10
- maximum open lots per symbol: 3
- maximum new lots per day: 3
- leverage: prohibited

Position quantity is the minimum permitted by risk, portfolio exposure, symbol
exposure, cash, and liquidity. A reduced position may be accepted when one of
those constraints binds, provided at least one synthetic share can be purchased.

## Planned risk release after trailing protection

Before trailing protection is executable, planned risk equals the positive
difference between entry cash cost and net proceeds at the current protective
stop.

A +15% favorable move activates trailing protection, but the trailing stop
does not become executable until the symbol's next trading open. Until then,
the lot remains `trailing_pending` and retains its planned risk.

At the next symbol trading open:

1. The proposed trailing stop is calculated as the greater of:
   - the complete net break-even quote, and
   - 90% of the highest adjusted high observed so far.
2. If the open gaps below that stop, the position exits at the open.
3. Otherwise, the trailing stop becomes executable and planned open risk is
   set to zero.
4. The released risk capacity can be allocated to new lots at the same open.

Protected lots still consume:

- cash already invested
- gross exposure
- symbol exposure
- total lot capacity
- symbol lot capacity
- distinct-symbol capacity
- liquidity capacity

Locked profit is never treated as negative risk. Therefore, one protected lot
cannot authorize the portfolio to exceed the 10% planned-risk ceiling by
offsetting other unprotected risk.

Gap, queue, and inability-to-execute risk remain real residual execution risks.
They are reported separately and are not misrepresented as zero physical risk.

## Exit policies

### Primary: activation plus trailing

- initial stop: 15% below buy execution price
- activation: +15% above buy execution price
- trailing distance: 10% below the highest observed adjusted high
- trailing floor: complete net break-even quote
- trailing effective time: next symbol trading open
- maximum horizon: observation 30 after the signal
- time exit: adjusted last price on observation 30

### Control: fixed take-profit

- stop: −15%
- take-profit: +15%
- maximum horizon: observation 30
- time exit: adjusted last price on observation 30

## Liquidity

New entry cash is limited to 1% of the trailing 20-observation average traded
value, computed with a one-observation lag.

Preferred liquidity input is a direct traded-value field such as `qTotCap`.
When no direct value field is available, adjusted last price times raw volume is
used and labelled as a proxy. This fallback can be distorted around corporate
actions and must be reported by symbol.

## Pre-registered scenario grid

The complete grid contains:

- capital: primary trimmed-mean, median, arithmetic mean
- slippage each side: 0, 0.10%, 0.20%, 0.50%
- position structure: multi-lot, single-lot
- exit style: trailing, fixed take-profit

The primary scenario is:

`primary capital + 0.20% slippage + multi-lot + trailing`

All scenarios are specified before portfolio results are inspected. The grid is
for robustness and mechanism comparison, not for selecting the best historical
configuration.

## Output files

Audit:

- `results/audits/10_initial_capital_audit.csv`
- `results/audits/10_raw_market_inventory_audit.csv`
- `results/audits/10_market_history_errors.csv`
- `results/audits/10_signal_execution_plan_errors.csv`
- `results/audits/10_portfolio_integrity_audit.csv`

Backtest:

- `results/backtests/10_signal_execution_plan.csv`
- `results/backtests/10_signal_decisions.csv`
- `results/backtests/10_trade_ledger.csv`
- `results/backtests/10_daily_equity.csv`
- `results/backtests/10_scenario_summary.csv`

Manifest:

- `results/manifests/10_portfolio_backtest_manifest.json`

## Interpretation boundaries

Stage 10 can support claims about executable simulated portfolio performance
under the frozen rules and stated assumptions. It cannot establish:

- live fill certainty in queues
- historical fee-rate fidelity before 1404
- precise market impact
- causal customer value or adoption
- actual trader total wealth
- live portfolio profitability

The benchmark source is deliberately deferred until its identity and
construction are separately frozen.


## Pre-result market-data loader hotfix (v1.1)

The first attempted execution stopped before any portfolio scenario was completed.
The diagnostic file contained 59 errors, and every error had the same cause:
at least one raw row in the affected symbol contained a nonpositive adjusted
open, high, low, or last price.

These rows cannot represent executable trading observations. The loader therefore:

- removes rows with nonfinite or nonpositive adjusted OHLC values before
  constructing the symbol execution calendar;
- does not impute, forward-fill, replace, or repair any price;
- excludes removed rows from entry timing, holding-period observation counts,
  trailing-stop processing, and ADV calculation;
- records removed-row counts by symbol in
  `10_raw_market_inventory_audit.csv`;
- still requires all 4,311 frozen selected signals to receive a complete
  next-open entry plan and 30-observation execution horizon.

The hotfix changes no model, score, signal, capital, fee, slippage, risk,
capacity, exposure, liquidity, or exit parameter. It was applied after a
pre-result loader failure and before any portfolio result was produced.
