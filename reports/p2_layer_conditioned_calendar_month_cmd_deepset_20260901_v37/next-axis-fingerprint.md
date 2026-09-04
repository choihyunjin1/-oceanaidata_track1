# P2 post-v37 next-axis fingerprint

## Decision

The next bounded axis is `p2_spectral_decoupled_output_regularized_deepset_20260901_v38`.
It is **not yet authorized or executed**. The exact proposed change to v13 is one fixed,
target-independent normalized-residual output penalty:

`weighted SmoothL1(beta=1) + 0.5 * 0.01 * weighted_mean(prediction^2)`.

Everything else remains exact v13: representation, layer×calendar-month×KST-day weights,
prefix-only training with a 7-day purge, seeds, 60 epochs, AdamW, `0.8/0.2` blend,
`2.5 C` model correction clip, `0.5 C` final action cap, and at most nine fits. There is no
coefficient sweep, scheduler, router, ensemble, row deletion, official-feedback selection,
or reuse of v37 metrics to choose a slice.

## Semantic audit

- P2 exact execution hits for `spectral decoupling`, `gradient starvation output penalty`,
  `squared output penalty`, and `prediction-norm regularization`: zero.
- v27 constrains parameter operator norms; v38 would leave parameters unconstrained and
  penalize only the scalar forward output during training.
- v23 penalizes derivatives with respect to public-temperature inputs; v38 uses no input
  derivative.
- v21 changes the residual likelihood to Student-t; v32 changes the evaluation-aligned
  squared-error loss; v38 retains exact weighted SmoothL1 and adds a target-independent
  output magnitude term.
- v20/v31/v36/v37 align domains, adversarial month information, parameter-gradient
  variance, or latent moments. v38 has no environment discrepancy term.
- P1 v46 executed the classification-logit version. That cross-problem execution is fully
  disclosed; v38 is a P2 normalized-residual regression hypothesis, not a claim of
  repository-global novelty and not a transfer of P1 performance.

## Primary-source boundary

Pezeshki et al., *Gradient Starvation: A Learning Proclivity in Neural Networks*, NeurIPS
2021, motivates spectral decoupling as an output regularizer for feature-learning dynamics:
https://proceedings.neurips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html.
The paper does not establish benefit for this P2 regression task. Applying its output
penalty to a normalized residual is a local exploratory hypothesis and requires the full
blocked historical test.

## Required preflight and stopping rule

Before execution, two target-free preflights must be byte-identical and prove: exact loss
formula and coefficient `0.01`; zero-output penalty/gradient no-op; finite nonzero-output
gradient; inference prediction unchanged; masked-token and permutation isolation; exact
v13 prefix/purge/seeds/optimizer/blend/caps; v26a prospective gate hash; namespace zero;
and official/test/sample/hidden/CSV/upload counters zero.

After exactly nine fits, retain the existing formal and safety gates plus prospective v26a:
at least `8/9` fold×layer cells non-harm and every cell `delta RMSE <= +0.003 C`. Any failure
is terminal `NO_GO`; coefficient, weight decay, blend, slice, router, and post-hoc ensemble
must not be retuned.
