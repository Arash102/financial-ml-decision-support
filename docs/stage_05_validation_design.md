# Stage 05 v2 — Candidate-event-count-balanced purged anchored walk-forward

## Objective

Freeze the five primary model-selection folds before Notebook 06 while reducing
the extreme validation-size imbalance observed in the first calendar-equal
design.

## What is balanced

Only the number of Stage 04 primary candidate events is used to select
validation boundary dates.

`meta_label`, positive-label fraction, event return, barrier outcome, and every
other event outcome are excluded from boundary construction.

## Calendar architecture

1. Build one frozen train-only trading calendar from Stage 03 labeled-train dates.
2. Preserve the pre-registered first validation start at 50% of train-only
   trading dates.
3. Count primary Stage 04 candidate-event starts on each frozen calendar date.
4. Inside the remaining validation horizon, choose contiguous date boundaries
   nearest cumulative 20%, 40%, 60%, 80%, and 100% candidate-event targets.
5. Keep all candidate events sharing one start date together.
6. Place a 30-trading-day conservative gap immediately before every validation
   block.
7. Retain historical training events only when their start is on or before the
   pre-gap train end and their event end is strictly before validation start.

## Why the event counts are approximate

A calendar date is indivisible. If one date contains many candidate events, the
boundary cannot split those events across folds. Therefore the deterministic
algorithm chooses the admissible calendar boundary nearest the cumulative equal-
event target.

## Why label stratification is prohibited

Temporal class imbalance may itself be evidence of a market regime. Moving
dates after observing `meta_label` would smooth away the nonstationarity that
the model must face and would make the validation design outcome-informed.

Class prevalence is therefore audited only after fold boundaries are frozen.

## Stage 06 contract

Notebook 06 must consume the Stage 05 v2 boundaries as-is. Optuna, model
families, fold metrics, and OOF results may not redesign these boundaries.
