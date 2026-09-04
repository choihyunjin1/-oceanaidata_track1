# P2 post-v38 next-axis fingerprint

## Decision

The next bounded axis is `p2_target_layer_gradient_sign_unanimity_deepset_20260901_v39`.
It is **not yet authorized or executed**. It keeps the exact v13 DeepSets forward model,
weighted SmoothL1 loss, prefix-only data with seven-day purge, seeds, 60 epochs, AdamW,
`0.8/0.2` blend, correction/action caps, and nine-fit maximum. The only proposed change is
the optimizer gradient construction: compute three same-batch weighted SmoothL1 gradients
for predeclared target layers `[2,3,4]`, average them, and retain a parameter coordinate
only when all three nonzero gradient signs agree (`abs(mean(sign(g_layer))) == 1`).

There is no threshold sweep, partial agreement, task order, projection, group reweighting,
month router, ensemble, row deletion, or Public-score selection. The v26a prospective gate
remains at least `8/9` non-harm fold×layer cells and maximum cell harm `<= +0.003 C`.

## Semantic audit

- P2 exact AND-mask / parameter-coordinate sign-consensus execution hits: zero.
- v28 PCGrad uses ordered pairwise vector projections when task-gradient dot products are
  negative. v39 would perform no projection or task ordering; it applies an elementwise
  unanimity mask to the mean of all three gradients.
- v36 Fishr matches diagonal variances of per-sample final-head gradients through a loss
  penalty. v39 would match no gradient moments and add no loss penalty.
- v18 Group-DRO changes environment risk weights; v19 V-REx changes environment-risk
  dispersion; v30 IRMv1 differentiates a scaled environment risk. v39 changes only which
  parameter coordinates receive the existing loss gradient.
- P1 v45 executed a station×quarter classification version. That cross-problem adjacency
  is disclosed. P2 v39 would use fixed target-layer regression tasks and does not transfer
  any P1 result, threshold, support decision, or performance claim.

## Primary-source boundary

Parascandolo et al., *Learning Explanations that are Hard to Vary*, ICLR 2021,
https://openreview.net/forum?id=hb1sDDSLbV, motivates retaining gradient coordinates whose
signs agree across environments. It supplies no P2 regression or competition-performance
claim. Target layers `[2,3,4]` are the existing fixed P2 prediction tasks, not selected from
v38 outcomes.

## Required preflight and stop rule

Two target-free byte-identical preflights must prove the exact unanimity formula, fixed
task order `[2,3,4]`, permutation invariance of task presentation, conflicting-coordinate
zeroing, all-agree mean-gradient identity, zero-gradient handling, finite deterministic
training, masked/future and set-permutation isolation, exact prefix/purge/science contract,
namespace zero, and access counters zero.

After one exactly-nine-fit execution, any formal/safety/v26a failure is terminal `NO_GO`.
No agreement relaxation, layer/month subdivision, optimizer/blend change, router, or
post-hoc ensemble is allowed.
