# Confirmed ZigZag Source Review

Source reviewed: `collect.ipynb`.

SHA-256: `972956bae554bf1727b2ca343a0d9f785283c13f7dfeb245b6cc85b8322e7302`.

The source implementation contains these safeguards:

1. The pivot search works on a chronological view of the input series.
2. `find_confirmation` invalidates a candidate high when a later value exceeds it and invalidates a candidate low when a later value falls below it.
3. A pivot receives a separate confirmation index only after the configured depth has elapsed without invalidation.
4. The generated `zigzag_up_new_2` and `zigzag_down_new_2` features skip the most recent pivot and use an older structural point, which is intended to avoid relying on a currently unconfirmed pivot.
5. The production call uses a 15% deviation threshold.

This source review supports the intended non-repainting design, but it is not a substitute for a row-level data audit. Notebook 04 must still verify, for sampled and boundary observations, that every pivot contributing to a feature had a confirmation date no later than the feature date.

Notebook 03 therefore does not use ZigZag to define labels, select barrier parameters, or remove events.
