# Stage 05 — Purged Anchored Walk-Forward Design

## Frozen input

Stage 05 uses only the primary long candidate events created by Stage 04 with
the confirmation-gated ZigZag and the pre-registered 15% filter.

The analyst manually spot-checked the reconstructed ZigZag distances against
price data before accepting the new confirmation-gated ZigZag as the project
baseline.

## Calendar design

Fold boundaries are defined on the unique train-only trading calendar collected
from the frozen labeled-train files. All candidate events sharing the same
event-start date stay on the same side of a fold boundary.

The first validation block begins after 50% of the train-only trading calendar.
The remaining calendar is divided into five contiguous validation windows of
approximately equal numbers of trading dates.

## Anchored training

For fold `k`, the training history is all eligible historical candidate events
available before that fold's validation block, subject to purge and embargo
controls. The historical training set grows forward through time.

## Conservative embargo gap

A 30-trading-day gap is placed immediately before each validation block.
Candidate events whose start dates fall inside this gap are excluded from the
training set for that fold.

This is implemented as a conservative pre-validation temporal gap. The exact
semantics are recorded in `configs/validation.yaml`.

## Event-end purging

A historical event is excluded from training when:

`event_end_date >= validation_start_date`

Therefore no retained training label interval reaches the validation period.

## Stage 05 does not fit a model

This notebook freezes calendar boundaries and audits leakage controls. Model
selection starts in Notebook 06.

## Unseen-test isolation

The unseen-test partition is not opened or used for fold construction, class
balance decisions, algorithm selection, or hyperparameter optimization.
