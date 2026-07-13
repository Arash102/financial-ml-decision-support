# Methodology Decisions

## Project objective
Develop and evaluate leakage-controlled machine-learning models for generating
stock-level decision-support signals in the Iranian capital market.

## Evidence boundary
This repository contains the stock-modeling pipeline only. Customer-profile and
trader-segmentation analyses are managed separately.

## Temporal design
- Training and model-selection signal dates: on or before `2021-03-20`
- Final unseen-test signal dates: `2021-03-21` through `2024-09-22`

The unseen test must not influence universe selection, feature selection, label
design, missing-value policy, hyperparameter optimization, algorithm selection,
threshold selection, or portfolio-rule selection.

## Candidate algorithms
- Random Forest
- XGBoost

## Baselines
- Dummy Classifier
- Logistic Regression

## Hyperparameter optimization
- Optuna
- 30 trials
- No test-set access

## Validation
- Primary: purged anchored walk-forward
- Robustness: CPCV

## Notebook 01 decisions
- Raw files are immutable source data.
- Dates are parsed and sorted chronologically.
- Duplicate dates are removed by keeping the last occurrence.
- No missing-value imputation is performed.
- Legacy target and future-derived fields are physically isolated.
- Candidate features are not assumed leakage-free.
