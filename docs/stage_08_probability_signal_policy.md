# Stage 08 v1 — Train-only probability and signal policy

## Purpose

Stage 08 freezes how the Stage 07 XGBoost score will be interpreted and how
candidate events will be converted into actionable decision-support signals.
The unseen-test candidate data and unseen-test labels are not loaded.

The legacy filename `08_unseen_test_evaluation.ipynb` is retained to avoid
renumbering the repository. Its Stage 08 implementation performs only train-only
policy design. Confirmatory unseen-test evaluation remains a later stage.

## Input evidence

The notebook consumes the 51,840 out-of-fold XGBoost predictions generated
across the five frozen Stage 05 walk-forward validation folds in Stage 06.

## Calibration decision

Three mappings are evaluated with expanding-origin temporal discipline:

1. raw identity score;
2. Platt sigmoid mapping;
3. isotonic mapping.

For target fold `k`, non-identity mappings are fitted only on OOF predictions
from folds earlier than `k`. Development uses folds 2–4. Fold 5 is an internal
confirmation fold and is excluded from method selection.

A calibration method is admissible only when:

- its mapping is nondecreasing in the raw score;
- it does not reduce fold ROC AUC by more than 0.005;
- it improves mean Brier score or mean log loss by at least 0.002.

The expected frozen result is rejection of post-hoc calibration because the
candidate mappings are temporally unstable. The raw XGBoost output is therefore
retained only as a ranking score and is not described as a calibrated
probability.

## Signal policy

The artifact uses a daily cross-sectional rank rule. Candidate events on each
date are sorted by:

1. raw XGBoost score descending;
2. symbol ascending;
3. event ID ascending.

Candidate fractions of 2.5%, 5%, 7.5%, 10%, 15%, and 20% are compared on
development folds 1–4. At least one event is selected on every date.

An eligible policy must produce at least 250 signals in every development fold
and must have nonnegative worst-fold precision lift over that fold's base rate.
Selection maximizes worst-fold precision lift, retains policies within 0.005 of
the best value, then maximizes mean precision lift and mean specificity while
preferring a lower signal fraction.

The expected frozen policy is the daily top 5% rule. Fold 5 is reported only as
an internal confirmation and is not used to choose the fraction.

## Interpretation

- No fixed probability threshold is selected.
- No probability calibrator is fitted.
- The score is used only for cross-sectional ranking.
- The rule is implementable online because each date uses only candidate scores
  available for that date.
- Economic returns, transaction costs, portfolio constraints, and unseen-test
  outcomes do not influence policy selection.

## Primary outputs

- `results/audits/08_oof_input_audit.csv`
- `results/audits/08_calibration_temporal_fold_metrics.csv`
- `results/audits/08_calibration_method_summary.csv`
- `results/audits/08_signal_policy_fold_metrics.csv`
- `results/audits/08_signal_policy_summary.csv`
- `results/audits/08_signal_policy_confirmation.csv`
- `results/audits/08_signal_policy_date_audit.csv`
- `results/predictions/08_oof_signal_policy_predictions.csv`
- `results/manifests/08_probability_signal_policy.json`
- `results/manifests/08_probability_signal_policy_manifest.json`
