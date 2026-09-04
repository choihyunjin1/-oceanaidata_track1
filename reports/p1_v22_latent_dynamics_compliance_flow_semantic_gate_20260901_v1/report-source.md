# P1 v22 latent-dynamics compliance-flow semantic gate

## Decision

`CLOSE_ZERO_FIT_SEMANTIC_OVERLAP_AND_COMPUTE_SCOPE`. No source row, target, model fit, lock, artifact, official file, CSV, or upload was touched.

Baumgartner, da Silva, and Urteaga (UAI 2026) distinguish observation-space likelihood from a goodness-of-fit test for compliance with prescribed latent dynamics. That is a valid mechanism-level distinction, not P1 performance evidence: [PMLR primary source](https://proceedings.mlr.press/v337/baumgartner26a.html).

For P1, the faithful version would require a conditional normalizing flow, a prescribed latent state transition, and a latent goodness-of-fit test. The only implementation small enough to fit the current nine-fit budget would replace the flow by a causal linear or low-rank encoder and score innovations. That collapses semantically into the already audited P1 v4 causal state-space innovation, asynchronous latent-state GP, temporally fused RPCA, and v15 Koopman/innovation closure. Keeping the nonlinear flow would instead introduce an unfixed deep architecture and a materially larger resource/support question. Neither branch is safe for a one-shot execution.

## Immutable counters

- train/README reads: `0`; target reads: `0`; fits: `0`; locks/artifacts: `0`
- official/test/sample/submission/hidden reads: `0`; CSV/uploads: `0`
- no threshold, latent dimension, flow depth, transition law, or goodness-of-fit statistic was selected from outcomes

The next audit therefore moves to a learning-objective-only axis: prefix-supervised row-level bipartite ranking on a fixed causal context basis, while retaining anchor union and the pre-existing multi-environment transport veto.
