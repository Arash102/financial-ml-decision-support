# Stage 07 v1 — Frozen full-train XGBoost training

Stage 07 consumes the frozen Stage 06 decision and fits the selected XGBoost
pipeline once on the complete eligible train-only candidate population.

## Frozen inputs

- Primary model: XGBoost, Stage 06 trial 25.
- Population: 118,464 Stage 04 primary long candidate events from 499 symbols.
- Feature schema: 35 semantic inputs, including 34 numeric features and
  categorical `gmma_state`.
- Target: `meta_label` reconstructed in Stage 03.

## Full-train Average Uniqueness

Average Uniqueness is recomputed over all eligible train candidate events.
Concurrency is calculated within symbol on inclusive event intervals and the
weights are normalized to mean one over the complete train population.
Validation and unseen-test events are not used.

## Fitting policy

Only the primary XGBoost model is fitted. Random Forest remains the challenger
but is not fitted in the primary Stage 07 run. The complete preprocessing and
model pipeline is serialized and reloaded on a deterministic probe sample.

No in-sample model-quality metric is reported. Stage 07 selects no threshold and
fits no probability calibrator. Raw XGBoost outputs may be used for ranking but
must not be interpreted as calibrated probabilities.

## Outputs

- `results/models/07_frozen_xgboost_pipeline.joblib`
- `results/models/07_frozen_xgboost_model_card.json`
- `results/folds/07_full_train_average_uniqueness_weights.csv`
- Stage 07 audit files under `results/audits`
- `results/manifests/07_fitted_feature_schema.csv`
- `results/manifests/07_frozen_model_training_manifest.json`

The binary model and event-level weights should remain local unless repository
size policy is changed explicitly.
