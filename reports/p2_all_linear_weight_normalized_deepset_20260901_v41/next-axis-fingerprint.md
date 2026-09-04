# P2 post-v41 next-axis fingerprint

## 결론

다음 단일 후보는 `p2_masked_token_virtual_adversarial_deepset_20260901_v42`이다.
P2 저장소 검색에서 virtual adversarial training(VAT), label-free adversarial
input direction, 또는 power-iteration input-consistency 실행은 0건이다. v42는 exact
v13의 관측된 연속 token-value 입력에만 고정된 국소 worst-direction consistency를
추가하며, 모델 구조·정답 손실·fold·blend·action은 바꾸지 않는다. 이 문서는 실행
승인이 아니며 현재 v42 fit, official access, CSV, upload는 모두 0이다.

## Primary-source boundary

Miyato, Maeda, Koyama, and Ishii, *Virtual Adversarial Training: A
Regularization Method for Supervised and Semi-Supervised Learning*, IEEE TPAMI
41(8), 2019, https://doi.org/10.1109/TPAMI.2018.2858821, motivates measuring
local conditional-distribution smoothness along a label-free adversarial input
direction. The source makes no P2, ocean-temperature, regression-RMSE, or
transport claim. Fixed-variance Gaussian predictive divergence below is a
preregistered P2 falsification, not a claim from that paper.

## Sealed scientific fingerprint

- Base: exact v13 token/context representation, architecture, prefix-only folds,
  7-day purge, layer x calendar-month x KST-day weights, seeds
  `[20260901, 20260902, 20260903]`, 60 epochs, AdamW `lr=0.001`, weight decay
  `0.0001`, weighted SmoothL1 beta 1.0, 0.8 champion + 0.2 raw correction,
  raw cap 2.5 C, final action cap 0.5 C, exactly 9 fits maximum.
- Only change: on every training minibatch, compute one label-free virtual
  adversarial direction over **observed continuous token values only** using
  normalized-feature epsilon `0.05`, finite-difference seed radius `xi=1e-6`,
  one power iteration, and fixed coefficient `1.0`; add
  `0.5 * weighted_mean((stopgrad(clean_prediction) - adversarial_prediction)^2)`.
- Target-layer one-hot, depth coordinates, presence masks, calendar/context
  covariates, missing tokens, padding and labels are never perturbed. Direction
  normalization is per row across eligible observed token values; rows with no
  eligible coordinate are exact no-ops. Clean prediction is stop-gradient only
  inside the VAT term, not the supervised loss.
- No epsilon/xi/coefficient/direction-step sweep, stochastic dropout, MixStyle,
  raw/label MixUp, SAM weight perturbation, router, ensemble, row deletion,
  Public-feedback selection or inference-time perturbation is allowed.
- Prospective v26a gate is unchanged: original gates plus at least 8/9 fold x
  layer cells non-harm and maximum cell delta RMSE <= +0.003 C.

## Exact and semantic audit

- Repository P2 searches for `virtual adversarial`, standalone token `VAT`,
  `adversarial perturbation`, and `input adversarial consistency` found 0
  executions. The two `power iteration` P2 hits are v41's explicit assertions
  that WeightNorm contains no spectral/power-iteration state.
- v23 directly penalizes the derivative of output with respect to the public
  temperature channel. v42 does not minimize a Jacobian norm; it selects one
  label-free worst local direction over all eligible observed token values and
  matches clean/adversarial predictions. v23's official aggregate is not used
  to select epsilon, coefficient, layer, fold, or slice.
- v24 SAM perturbs model parameters toward worst local loss. v42 leaves all
  parameters untouched while constructing a detached perturbation in observed
  normalized input space.
- v26 convexly mixes raw inputs and labels between rows. v42 mixes neither rows
  nor labels.
- v31 uses a layer-conditioned calendar-month discriminator and gradient
  reversal. v42 has no domain classifier or environment label.
- v40 uses two random dropout views. v42 has no dropout and uses a deterministic
  one-step adversarial input direction. It reuses the same fixed-variance
  Gaussian divergence formula only as a metric definition, not v40 stochasticity.
- v41 changes parameter coordinates with learned per-output WeightNorm. v42
  restores exact unparametrized v13 Linear weights.

## Required target-free preflight

- Prove the direction builder reads no target and perturbs only finite observed
  continuous token values; all mask/depth/context/one-hot/padding coordinates
  remain bit-identical.
- Prove epsilon `0` and coefficient `0` are exact clean-training no-ops; fixed
  epsilon produces per-row eligible-coordinate L2 norm `0.05` within `1e-6`.
- Prove one power iteration exactly, finite gradients, detached direction, clean
  stop-gradient only inside consistency, and deterministic seed replay.
- Prove future/masked-token perturbations cannot affect current predictions,
  public-layer permutation plus inverse permutation is invariant, validation
  rows never enter the perturbation builder, and inference is exact v13.
- Run two byte-identical target-free preflights with namespace 0 before one
  exactly-once execution.
- Report raw RMSE, canonical nominal and fixed transport-adjusted points,
  fold/month/layer/9-cell metrics, CI, action geometry, VAT norms/steps/hashes,
  independent QA and access counters.

## Evidence pins

- v41 result: `4c6e3659c3c4635a14a22a1151667990fc92c2ad9d8bc5190bb67e8baad2752c`
- v40 result: `b60534e132c493e76bec94e4a17fcc17b7bc85f30b3f1d8f63be318f3b3cdda5`
- v26 result: `7547e7c97344ef06d2e8107f565e7e704e2eb474bfd154787f6d82301b869281`
- v24 result: `d5d6d7fae5e6ae30278acd698516e1792f90a17ffc81234b87fc6af1141a3b6a`
- v23 result: `ddecf11bfde6ceb6cb0c814e5069bd0e8c96ee2b079020a1e2e468e84754e64b`
- v31 result: `e73dbba22d5e2f3f1b5b15583d32ec2ee4bbfe3dc924af9fa2a252a5003325a7`
- v26a prospective gate: `c7fde8c5d9f535ab8080eb561bf082c55e5c7172117d00e12e1479f9b4417680`

## Operation counters

- v42 model fits: `0`
- official/test/sample/query/hidden rows: `0`
- submission CSVs: `0`
- uploads: `0`
