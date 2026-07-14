# ZigZag Confirmation Review

## Stage 04 decision

The supplied collection code clearly contains a confirmation concept: pivot
candidates receive a `confirmation_index`, and separate confirmation markers are
created.

The audit also identified an important implementation distinction. The legacy
`zigzag_up_new_2` and `zigzag_down_new_2` distance loop scans pivot markers
(`zj`) and skips the first encountered pivot. It does not directly gate the
selected pivot by the stored confirmation marker.

Therefore Stage 04 does **not** use the legacy `new_2` columns as direct model
features or as the primary event filter.

Instead, Stage 04 reconstructs an online confirmation-gated ZigZag state:

1. detect a local high or low using the original depth logic;
2. assign the first confirmation observation after the original waiting rule;
3. process confirmations in chronological order;
4. require the minimum 15% deviation before accepting an opposite pivot;
5. update the known state only at the confirmation observation;
6. carry the latest confirmed high and low forward;
7. audit prefix invariance by recomputing sampled historical prefixes.

A pivot is never available before its confirmation observation.

## Candidate long-event rule

The primary threshold remains the pre-registered 15% Stage 03 rule:

- the event is eligible under the frozen Triple-Barrier labeling policy;
- a confirmed high and confirmed low are both available;
- `0 <= distance_above_confirmed_low_fraction <= 0.15`;
- `distance_below_confirmed_high_fraction >= 0.15`.

The 10% and 20% thresholds are train-only sensitivity diagnostics. The notebook
does not automatically select a threshold from label performance.

## Meta-labeling interpretation

The confirmation-gated ZigZag rule generates the primary long side. Within those
candidate long events, the frozen Triple-Barrier label becomes the take/skip
meta-label. RF and XGBoost are evaluated later as meta-models.
