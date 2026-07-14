# Methodology Decisions

## Stage 03: label reconstruction and censoring

### Target definition

The target is reconstructed from adjusted price paths. The legacy `class` field is not reused. Each row is treated as a potential event start and receives an event end, outcome status, and eligibility status.

### Primary barrier scenario

The primary, pre-registered scenario is:

- upper barrier: +15%
- lower barrier: -15%
- maximum holding period: 30 trading observations
- monitoring begins on the next trading observation
- same-bar upper/lower touch: negative outcome
- unresolved event at observation 30: positive when the vertical return is greater than zero, otherwise negative
- insufficient future observations: right-censored and excluded

The primary scenario is retained from the previous strategy because it expresses a symmetric economic outcome. It is not chosen by unseen-test performance.

### Train-only sensitivity diagnostics

Notebook 03 also evaluates the following alternatives on the train partition only:

- +15% / -10%
- +10% / -10%
- +8% / -10%

These comparisons are descriptive robustness diagnostics. The notebook does not automatically switch the target scenario to improve class balance or unseen-test behavior.

### Partition isolation

Train labels are completed only with train rows. Unseen-test rows are never used to resolve a train event. The unseen-test partition is labeled independently. Events near either partition boundary that lack the full horizon and do not touch a horizontal barrier are right-censored.

### ZigZag and meta-labeling

ZigZag is not used to create the Stage 03 target. The uploaded implementation records pivot confirmation and the `zigzag_up_new_2` / `zigzag_down_new_2` construction is intended to avoid using the most recent potentially unconfirmed pivot. A formal row-level timing audit remains part of Notebook 04.

After that audit, confirmed ZigZag distance rules may define candidate long events. RF and XGBoost can then act as meta-models that decide whether to take or skip each candidate. This separation prevents the event filter from being confused with the target itself.
