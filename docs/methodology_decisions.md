# Methodology Decisions

## Stage 03 — Frozen target

- Primary target: symmetric +15% / -15% Triple Barrier.
- Maximum holding period: 30 trading observations.
- Label design and scenario diagnostics use train only.
- Unseen test is not used to select the target.
- ZigZag does not construct the target.

## Stage 04 — Feature and ZigZag policy

- Feature approval is structural and train-only.
- Target-derived, event-end, censoring, and legacy-label fields are prohibited as model inputs.
- Adjusted OHLC levels are retained for context and label reconstruction, but are not direct model features.
- Legacy `zigzag_up_new_2` and `zigzag_down_new_2` are not direct model features or filter inputs.
- The confirmed ZigZag state is reconstructed with explicit confirmation timestamps.
- Prefix invariance is an integrity requirement.
- The 15% candidate-event threshold is pre-registered; 10% and 20% are train-only sensitivity diagnostics.
- The confirmed ZigZag rule generates the primary long side.
- Within candidate long events, the frozen Triple-Barrier label is interpreted as the take/skip meta-label.
- Unseen-test outcomes remain untouched during Stage 04.
