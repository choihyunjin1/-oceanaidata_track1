# P2 v46 sealed next-axis fingerprint

## Decision

Propose exactly one new axis, `p2_hidden_activation_layer_normalized_deepset_20260901_v46`.
This is a fingerprint only. It is not authorized or executed by this document.

## Primary-source motivation

Ba, Kiros, and Hinton, *Layer Normalization* (2016),
https://arxiv.org/abs/1607.06450, defines normalization from the hidden units of
one training case rather than from batch statistics. This is representation and
optimization motivation only; it is not evidence that the method improves P2.

## Repository-wide semantic audit

- Executed P2 `nn.LayerNorm(` / `LayerNorm(` implementations: 0 exact hits.
- v27 changes weight spectral geometry and failed technically; v41
  reparameterizes each output weight vector; v45 masks weights during training.
  None normalizes hidden activations per example.
- v20 CORAL and v37 CMD align distributions across calendar-month domains; they
  do not normalize each hidden vector inside the predictor.
- v40 activation dropout plus two-pass consistency is stochastic and has no
  activation normalization.
- BatchNorm, RMSNorm, normalization placement variants, and affine/epsilon
  variants are outside this sealed question.

## Single sealed intervention

- Preserve exact v13 data, prefix-only cutoffs plus 7-day purge, public-token
  inputs, domain-balanced layer x calendar-month x day weights, seeds
  `[20260901, 20260902, 20260903]`, 60 epochs, AdamW, weighted SmoothL1,
  champion/model blend `0.8/0.2`, raw correction cap `2.5 C`, final action cap
  `0.5 C`, and maximum 9 fits.
- Insert exactly four `torch.nn.LayerNorm(32, eps=1e-5,
  elementwise_affine=True)` modules: after each of the two hidden Linear layers
  in the shared element map and after each of the two hidden Linear layers in
  the head, always before ReLU.
- Initialize every LayerNorm gain to 1 and bias to 0. This normalization is the
  only scientific change. The resulting initial function is expected to differ
  from v13 and that difference must be recorded, not repaired.
- No BatchNorm/RMSNorm, coefficient or placement sweep, scheduler, router,
  ensemble, row deletion, official-v23-driven selection, or post-hoc blend.

## Required target-free preflight

- Exactly four LayerNorm modules with normalized shape 32, epsilon `1e-5`,
  affine gain/bias, no running-stat buffers, and the exact prescribed locations.
- Exact expected parameter count `5121`, parameter tensors `18`, buffers `0`.
- Per-example batch-composition invariance, public-layer permutation invariance,
  masked/future-token isolation, deterministic repeat, finite forward/backward,
  and exact v13 prefix/purge/action contracts.
- Two independent zero-operation preflights must be byte-identical and show
  data rows 0, fits 0, artifacts 0, official/test/sample/hidden/query rows 0,
  CSV 0, upload 0 before the exactly-once namespace may be consumed.

## Prospective evaluation and gate

Run all three sealed blocked folds and report pooled/fold/month/layer and all
nine fold x layer deltas, day-block CI90/P, action geometry, canonical nominal
and fixed transport-adjusted points, hashes, and independent QA. In addition to
the legacy guards, the prospective v26a amendment remains unchanged: at least
8/9 fold x layer cells must be non-harm and every cell must have delta RMSE no
greater than `+0.003 C`. Any failed guard is terminal `NO_GO`; no normalization
retune or retry is permitted.

## Frozen evidence

- v45 result SHA-256:
  `24f02df803b6da31df39d63c4cad8f6a2e9006c9c7ee0a36428877c7d1ac6441`
- v45 prediction SHA-256:
  `e98aeda880c8f9e9020fca3294116467ce2f00a4684e5c21b58b1affd4c1af4a`
- prospective gate amendment SHA-256:
  `c7fde8c5d9f535ab8080eb561bf082c55e5c7172117d00e12e1479f9b4417680`
- Official/test/sample/hidden/query access, CSV, and upload used to choose this
  fingerprint: 0.
