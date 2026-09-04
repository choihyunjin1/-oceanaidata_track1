# P1 v7 causal path cross-moment add-only cycle

## Decision

The exactly-once exploratory result is `NO_GO_EXPLORATORY_ONLY`. The deterministic path representation was genuinely distinct from P1 MiniRocket, multiscale Haar residual summaries, soft-symbol transitions, MS-TCN, the vertical graph encoder, TS2Vec, and CAPA, but its inner precision did not transport to the exposed Q2-Q4 historical surface.

The model used causal first-level increments, total and quadratic variation, and signed temperature-salinity level-2 path area over fixed 12/72/288-row windows. Kiraly and Oberhauser describe ordered sample cross-moments as sequential feature maps that converge consistently to path kernels: https://www.jmlr.org/papers/v20/16-314.html.

## Frozen protocol

Three fixed SGD-logistic seeds were fit in each of Q2-Q4, for exactly nine fits. Threshold quantiles 0.995/0.9975/0.999, maximum addition share 0.0025, and the Wilson90 precision LCB minimum 0.55 were fixed before metrics. The 0.55 gate was retained because v6 proved the requested incumbent-F1/2 recalibration scientifically unavailable: authenticated Q2 inner incumbent predictions do not exist, so lowering the gate would require leakage or an unregistered anchor refit.

All three folds selected the 0.999 inner quantile. Inner precision/LCB was 1.0/0.956853 for Q2, 1.0/0.977208 for Q3, and 0.951613/0.918641 for Q4. Despite that, Q2 and Q3 made no outer additions, while Q4 added seven S-ORS layer-3 rows and every addition was false.

## Metrics

Pooled F1 fell from 0.8604836038423319 to 0.8602841341855151, delta `-0.00019946965681683082`. Candidate TP/FP/FN were 12989/1153/3066. Q2 and Q3 deltas were zero; Q4 F1 was 0.9082513063555717, delta `-0.0007732618760021293`. The worst affected station-layer was S-ORS layer 3, delta `-0.002319826158835414`.

Paired block-bootstrap CI90 was `[-0.000587734895991166, 0]`. Nominal and transport-adjusted points were `-0.0053019817015861935` and `-0.001590594510475858`. Runtime was 27.375 seconds. Fits were 9; anchor removals, official/test/sample/submission/hidden access, CSV, and uploads were all zero.

Independent QA reloaded all sealed arrays and train labels, recomputed TP/FP/FN/F1 and the add-only union, verified nine unique model hashes and all config/runner/completion/lock hashes, and passed. Result SHA-256 is `28c89ad33d866a23858f35fa0b7a0b2cb9ae67fa8981afb4c4ac75571cbd322d`.
