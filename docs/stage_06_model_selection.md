# Stage 06 v3 — Fold-local Average Uniqueness and Complete Optuna Search

## Frozen input

Stage 06 v3 consumes:

- the Stage 04 v8 pooled feature schema: 35 semantic inputs;
- the Stage 04 primary 15% confirmation-gated long candidate population;
- the Stage 05 v2 event-count-balanced purged anchored walk-forward folds.

The unseen-test partition is not opened.

## Fold-local Average Uniqueness Weighting

For every fold, weights are recomputed from that fold's retained training events
only.

For symbol `s` and trading observation `t`, concurrency is:

`c(s,t) = number of current-fold training events for symbol s active at t`

An event is active on the inclusive interval:

`event_start_date <= t <= event_end_date`

For event `i`, raw average uniqueness is:

`u_bar(i) = mean over its active observations of 1 / c(s,t)`

The normalized training weight is:

`w(i) = u_bar(i) / mean_fold_train(u_bar)`

Consequently, each fold's normalized training weights have mean one.

Concurrency is never pooled across unrelated securities. Validation-event
presence is not used to calculate training weights. Validation metrics remain
unweighted.

Average uniqueness is applied to:

- class-balanced Logistic Regression;
- Random Forest;
- XGBoost.

The prior-probability Dummy baseline remains unweighted.

The primary design does not multiply uniqueness by event returns and does not
use sequential bootstrap. Sequential bootstrap remains a later Random Forest
robustness option.

## Preprocessing

Numeric median imputation and categorical most-frequent imputation are fitted
inside each fold's training partition. `gmma_state` is one-hot encoded
fold-locally. Only Logistic Regression receives numeric standardization.

## Optuna

Random Forest and XGBoost each receive 30 COMPLETE trials.

- Pruner: `NopPruner`
- Every trial evaluates all five frozen folds.
- Separate seeded multivariate TPE samplers are used for the two studies.
- The Random Forest search uses a fixed search-space shape by sampling
  `max_samples` in every trial and ignoring it when `bootstrap=False`.

Tree caps:

- Random Forest: `n_estimators <= 150`, `max_depth <= 15`
- XGBoost: `n_estimators <= 150`, `max_depth <= 15`

## Objective and Hierarchical Selection

The Optuna objective is equal-fold mean ROC AUC.

After 30 COMPLETE trials, the selected configuration is determined by the
pre-registered hierarchy:

1. identify the best mean fold ROC AUC;
2. keep trials no more than 0.005 below that best mean;
3. inside that band, maximize worst-fold ROC AUC;
4. then minimize fold ROC AUC standard deviation;
5. then choose the lower trial number.

The same mean-band, worst-fold, and stability logic is applied when selecting
between Random Forest and XGBoost. Threshold 0.50 is diagnostic only. Trading
threshold selection remains deferred.

## Output audits

Stage 06 writes event-level and aggregate weighting audits, full Optuna trial
histories, hierarchical trial rankings, fold metrics, validation probabilities,
best hyperparameters, and a reproducibility manifest.

Notebook 07 must consume the frozen Stage 06 decision and hyperparameters. It
must not redesign folds, recompute hyperparameters after viewing unseen-test
performance, or use unseen-test outcomes for threshold selection.
