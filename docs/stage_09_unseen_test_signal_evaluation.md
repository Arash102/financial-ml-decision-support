# Stage 09 v2 — Confirmatory unseen-test signal evaluation with corrected event outcomes

## Purpose

Stage 09 v2 preserves the Stage 04 candidate rule, Stage 07 XGBoost model, and
Stage 08 daily top-5-percent signal policy. It does not retrain or tune anything.
The only substantive change from v1 is a reproducible and explicit definition of
the event-level return used for win rate, payoff ratio, profit factor, and mean
event outcome.

## Frozen temporal boundary

- Signal generation: 21 March 2021 through 22 September 2024.
- Outcome-observation tail: through 26 October 2024.

No signal is generated after 22 September 2024. Later raw observations are used
only to complete the already-defined 30-trading-observation outcome windows of
signals generated on or before that date.

## Blind inference lock

Candidate construction, model scoring, and daily selection are completed before
labels or event outcomes are loaded. The v2 run must reproduce the previously
audited inference-lock SHA-256 exactly:

`c29f1ec3b6d59fc5a2aa163f65b880562271f938fc7208a784ee820f5245c946`

Any different lock hash is a blocking failure.

## Corrected event-return policy

For each candidate event, entry is the adjusted last price on the signal date.
The next 30 trading observations exclude the signal-date row.

1. **Upper barrier event:** corrected return equals the maximum adjusted-high
   return over the complete 30-observation window.
2. **Lower barrier event:** corrected return is fixed at -15 percent.
3. **Vertical barrier event:** corrected return equals the adjusted-last return on
   trading observation 30.
4. A zero vertical return is not a win.

The sign of the corrected return must match the frozen binary label for every
event. This policy changes no labels; it only corrects the magnitude assigned to
upper-barrier outcomes.

## Interpretation boundary

The upper-event return uses the ex-post maximum favorable price within the
30-observation window. It measures the movement potential of a selected signal.
It is not an executable take-profit rule, a realized trade return, or a portfolio
return. Notebook 10 will require an independently pre-specified exit rule,
capital allocation, overlapping-position logic, costs, and an equity curve.

## Main v2 outputs

- `results/predictions/09_unseen_test_inference_lock.csv`
- `results/predictions/09_unseen_test_signal_evaluation.csv`
- `results/predictions/09_selected_unseen_test_signals.csv`
- `results/audits/09_unseen_test_predictive_metrics.csv`
- `results/audits/09_unseen_test_signal_classification_metrics.csv`
- `results/audits/09_unseen_test_signal_outcome_summary.csv`
- `results/audits/09_unseen_test_signal_outcomes_by_year.csv`
- `results/audits/09_unseen_test_signal_outcomes_by_barrier.csv`
- `results/audits/09_outcome_observation_tail_audit.csv`
- `results/audits/09_corrected_event_return_errors.csv`
- `results/manifests/09_unseen_test_inference_lock.json`
- `results/manifests/09_unseen_test_signal_evaluation_manifest.json`
