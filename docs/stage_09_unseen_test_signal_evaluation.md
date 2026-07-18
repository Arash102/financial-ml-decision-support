# Stage 09 v1 — Confirmatory unseen-test signal-level evaluation

## Purpose

Stage 09 applies the already frozen Stage 07 model and Stage 08 daily ranking
policy to the untouched unseen-test period. No model, feature, calibration,
threshold, or signal-fraction decision is changed after test outcomes are opened.

## Causal feature inference

The train tail is included as historical warm-up for rolling stock features and
for the confirmation-gated ZigZag state. The test period is not treated as an
independent cold-start sequence. All feature calculations remain causal: each
row uses only information available on or before that row's date.

The equal-weight market index is reconstructed across the frozen 499-symbol
universe through 22 September 2024. Repeated index observations are collapsed
by their cross-file median on each date, matching the Stage 04 canonicalization
rule. Cross-file inconsistencies are audited rather than used to modify the
frozen feature or policy design.

## Blind inference lock

The notebook first creates candidate features, loads and hash-verifies the frozen
XGBoost pipeline, generates scores, applies the daily top-5-percent rule, and
writes an outcome-free inference lock. Only after this lock is written are label,
event-return, barrier, and holding-period columns read and joined for evaluation.

## Evaluation levels

Predictive evaluation reports ROC AUC, average precision, Brier score, log loss,
and ECE on all unseen-test candidate events. Calibration metrics are diagnostic
only because Stage 08 rejected literal probability interpretation.

Signal classification evaluation treats the frozen daily rank policy as the
binary decision rule and reports precision, precision lift, specificity, and
sensitivity.

Signal outcome evaluation reports win rate, gross label-horizon event return,
payoff ratio, profit factor, and holding-period statistics. These are overlapping
event-level outcomes. They are not portfolio returns and do not include capital
constraints, position sizing, liquidity, slippage, or transaction costs.

## Frozen policy

For each test signal date:

1. rank eligible long candidates by raw XGBoost score descending;
2. break ties by symbol ascending and event ID ascending;
3. select `max(1, ceil(0.05 × daily candidate count))`.

No fixed probability threshold or calibrator is introduced.

## Main outputs

- `results/predictions/09_unseen_test_inference_lock.csv`
- `results/predictions/09_unseen_test_signal_evaluation.csv`
- `results/predictions/09_selected_unseen_test_signals.csv`
- `results/audits/09_unseen_test_candidate_panel_audit.csv`
- `results/audits/09_unseen_test_feature_engineering_audit.csv`
- `results/audits/09_unseen_test_predictive_metrics.csv`
- `results/audits/09_unseen_test_signal_classification_metrics.csv`
- `results/audits/09_unseen_test_signal_outcome_summary.csv`
- `results/audits/09_unseen_test_signal_outcomes_by_year.csv`
- `results/audits/09_unseen_test_signal_date_audit.csv`
- `results/manifests/09_unseen_test_inference_lock.json`
- `results/manifests/09_unseen_test_signal_evaluation_manifest.json`
