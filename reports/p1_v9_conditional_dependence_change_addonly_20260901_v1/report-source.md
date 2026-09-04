# P1 v9 conditional-dependence change representation

## Decision

`p1_v9_conditional_dependence_change_addonly_20260901_v1` completed exactly once as a valid `NO_GO_EXPLORATORY_ONLY`. It is closed without retry, ridge/window/threshold retuning, or gate relaxation.

The representation used only prefix-fitted cross-layer temperature covariance, ridge precision, and partial correlation, compared with a causal 72-row-half-life exponentially weighted covariance. It did not assume a physical horizontal graph. A fold-prefix layer with no observations was fixed to a zero channel for that fold. The five anomaly-head inputs were global covariance/precision/partial-correlation change norms and layer-node precision/partial-correlation change norms.

Wu et al., AISTATS 2024, motivates looking for multivariate conditional-distribution changes that can be absent from marginal distributions. This implementation is a simplified covariance/precision representation. It is not the paper's clustering/MDL algorithm and makes no paper-derived P1 performance claim.

## Semantic and readiness gates

The prior dependence-null cycle calibrated block maxima of `abs(clean_state_decoder_signal)` plus concurrent-layer counts; it did not estimate joint covariance or precision. Clean-state CAPA used marginal segment likelihood, e150 rescored existing long-event proposals, and the frozen long-event family classified proposal segments. Therefore v9 was neither an exact nor semantic duplicate.

- Source allowlist: local README and train only; official test/hidden/sample/submission reads 0.
- ns/cutoff, future-invariance, station reset, same-marginal dependence-change response, add-only, and 9-fit tests: 6/6 PASS. Ruff: PASS.
- Label-free support: both multilayer stations passed; nonzero dependence-feature share `0.9658776937`; all five feature variances exceeded the preregistered floor.
- After making the no-prefix-observation layer rule explicit, two accepted preflights were byte-identical at SHA-256 `6890feb3985816eafe4ba782b8358111f04e51d1a8c22cb1a28c053314194a7c`. No artifact or lock existed after either. Pre-execution QA: PASS.

## Frozen selection and result

Three fixed SGD-logistic seeds per Q2/Q3/Q4 yielded 9 fits, no sweep. The same quantiles `0.995/0.9975/0.999`, maximum addition share `0.0025`, minimum 25 additions, and Wilson 90% precision LCB `0.55` were used. The 0.55 gate was not relaxed because v6 did not establish a leakage-safe replacement.

No fold had an eligible threshold. The best inner evidence was still below the gate: Q2 precision/LCB `0.4413/0.3933`, Q3 `0.3697/0.3006`, and Q4 `0.0204/0.0141`. Thus all outer predictions were the untouched anchor, based solely on inner selection.

- Runtime: 22.0 seconds; 9 unique model hashes.
- Pooled 421,032 rows: F1 `0.8604836038423319`, TP/FP/FN `12,989/1,146/3,066`, precision `0.9189246551`, recall `0.8090314544` for both anchor and candidate.
- Additions/removals/action slices: `0/0/0`.
- Q2/Q3/Q4 delta F1: all 0. Pooled delta F1 0.
- Paired block bootstrap 90% CI `[0, 0]`, 2,000 replicates, 3,089 clusters.
- Canonical nominal and transport-adjusted expected-point deltas: both 0.
- Pooled offset recall `0.6477211796`; Q2/Q3/Q4 `0.5361356932/0.7415094340/0.6872146119`.
- Long-event interior diagnostic: 81 positive runs, 15,009 interior rows, anchor=candidate recall `0.8107135719`, delta 0.

Independent lifecycle-safe QA recomputed the counts/F1/delta and long-event diagnostic, checked all seals, 9 unique model hashes, zero pre-seal target reads, exact add-only union, artifact hashes, and official/CSV/upload access 0. Result SHA-256: `620fb0402f02f628a8c1bb547a92b436ca0805541692b15976355d8a76d369f8`.

## Next axis

Standard Matrix Profile remains excluded because the existing P1 academic audit already identified subsequence z-normalization as structurally erasing level offsets. The next audit candidate is a causal nonlinear recurrence/laminar-state representation: recurrence rate, diagonal determinism, vertical laminarity, and trapping time over fixed past windows, without subsequence z-normalization. It must first be checked against P1 soft-symbolic transitions, Haar/path features, flatline rules, MS-TCN, and the P3 recurrence sibling; only P1 architectural novelty would authorize a new one-shot ID.
