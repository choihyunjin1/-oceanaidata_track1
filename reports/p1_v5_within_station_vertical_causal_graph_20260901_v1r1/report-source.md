# P1 v5r1 within-station vertical causal graph

## Terminal decision

`p1_v5_within_station_vertical_causal_graph_20260901_v1r1` is a valid, exactly-once `NO_GO_RESEARCH_ONLY`. The fixed inner precision-LCB gate admitted no threshold in Q2, Q3, or Q4, so the champion-preserving union made zero additions and zero removals. This is a no-op result, not evidence that the graph representation improved the anchor.

The predecessor v5 is separately quarantined as `INVALID_TECHNICAL_TARGET_LEAKAGE_TIME_UNIT`. Its nine fits and result were not used to choose or change any v5r1 model, threshold, budget, seed, capacity, or gate.

## Frozen execution and gates

- Representation: zero horizontal edges; bidirectional adjacency only between consecutive observed layers within each station; one left-padded causal temporal convolution; one fixed graph message pass; one linear anomaly head.
- Capacity: hidden width 16, kernel 9, four epochs, weight decay 0.0001, learning rate 0.002, three fixed seeds across Q2-Q4, exactly nine fits, no sweep.
- Selection: inner-only quantiles 0.995/0.9975/0.999, maximum addition share 0.0025, minimum 25 proposals, Wilson 90% precision LCB at least 0.55. Outer tuning was zero.
- Corrected time contract: explicit `DatetimeIndex.as_unit("ns").asi8`; actual train-only inner boundaries were 2025-01-17 02:50 KST, 2025-04-04 11:00 KST, and 2025-06-21 16:00 KST for Q2-Q4 respectively.

The best inner LCB remained below 0.55 in every fold: Q2 reached 0.470208 at the 0.999 quantile, Q3 reached 0.144569 at 0.999, and Q4 reached 0.072981 at 0.995. Therefore all outer additions were zero without reading outer labels for selection.

## Historical result

Pooled candidate and anchor F1 were both 0.8604836038423319 with TP/FP/FN 12989/1146/3066. Fold F1 was 0.7784135753749013, 0.8970588235294118, and 0.9090245682315738 for Q2-Q4. Paired block-bootstrap delta CI90 was [0, 0]. Nominal and transport-adjusted expected points were both 0. Runtime was 20.907 seconds on one NVIDIA GeForce RTX 5090. Anchor removals, official/test/sample/submission/hidden access, CSV materialization, and uploads were all zero.

## Independent QA and hashes

Independent read-only QA reloaded all three sealed arrays and train labels, recomputed pooled TP/FP/FN/F1 and the add-only union, checked nine unique fresh model hashes, three distinct nanosecond boundaries, all prediction/config/runner/completion/lock hashes, and zero prohibited access. It passed. Result SHA-256 is `4a88d30c31e14b3e96caa84f8f49ee07285a2c40074a47426374da2fba7d3647`.

## Next non-duplicate axis (document only)

Audit a P1-specific causal multichannel ROCKET-style representation: fixed, non-trained dilated random kernels over past-only temperature/salinity/mask channels, summarized by PPV and maximum response, followed by one sparse linear add-only head with the same anchor firewall. This is structurally distinct from learned MS-TCN/graph encoders, label-free TS2Vec, fixed peer residuals, CAPA, event rescoring, and Group-DRO. It must receive a new semantic/data-contract preflight and preregistration before any fit; this report does not authorize or execute it.
