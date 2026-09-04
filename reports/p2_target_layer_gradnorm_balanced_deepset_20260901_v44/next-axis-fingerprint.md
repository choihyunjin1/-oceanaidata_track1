# P2 post-v44 next-axis fingerprint

## 결론

다음 단일 후보는 `p2_all_linear_dropconnect_deepset_20260901_v45`이다.
저장소 전체 검색에서 `DropConnect`, weight dropout, Bernoulli weight mask,
stochastic weight mask 실행은 P1/P2/P3 모두 0건이다. v45는 exact v13의 다섯
Linear weight에만 학습 중 고정 확률 Bernoulli mask를 적용하는 weight-level
regularization 축이다. v40 activation dropout/consistency나 v41 WeightNorm의
변형이 아니며, 이 문서는 실행 승인이 아니다. 현재 v45 fit,
official/test/sample/query/hidden access, CSV, upload는 모두 0이다.

## Primary-source boundary

Wan, Zeiler, Zhang, Le Cun, and Fergus, *Regularization of Neural Networks
using DropConnect*, ICML/PMLR 2013,
https://proceedings.mlr.press/v28/wan13.html, motivates zeroing a random subset
of weights rather than activations during training. The source makes no P2,
ocean-profile, RMSE, layer-safety, transport, or deployment claim. The fixed
probability and guards below are a preregistered local falsification; the paper's
multi-model aggregation is not used.

## Sealed scientific fingerprint

- Base: exact v13 data, tokens/context, chronological prefix plus 7-day purge,
  layer x calendar-month x KST-day weights, seeds `[20260901,20260902,20260903]`,
  60 epochs, exact weighted SmoothL1 beta `1.0`, AdamW `lr=0.001`, weight decay
  `0.0001`, 0.8 champion + 0.2 correction, raw cap `2.5 C`, final action cap
  `0.5 C`, maximum 9 fits.
- Only change: during training forward passes, independently mask each weight in
  exactly the five v13 Linear matrices with Bernoulli keep probability `0.9`
  and use inverted scaling `mask / 0.9`. Biases are never masked. Evaluation
  uses each learned raw weight exactly once with no mask or rescaling.
- The five parameter shapes, initial state, total parameter count, task loss,
  row weights, optimizer, sampler, batch order and inference graph remain exact
  v13. There is no activation dropout, two-pass consistency, Monte-Carlo
  inference, ensemble/model aggregation, layer/month router, row deletion,
  coefficient/probability/location sweep, or Public-feedback selection.
- Mask streams are generated only from each sealed fit seed and training-step
  order. Validation/query rows never generate or influence a mask. A zero-drop
  test path must be bit-identical to exact v13, but the executed probability is
  only `0.1`.
- Prospective v26a gate remains unchanged: original gates plus at least 8/9
  fold x layer cells non-harm and maximum cell delta RMSE <= `+0.003 C`.

## Exact and semantic audit

- Repository-wide searches for `DropConnect`, `drop connect`, `weight dropout`,
  `weight-drop`, `Bernoulli weight mask`, and `stochastic weight mask` found 0
  executed candidates.
- v40 adds four activation-Dropout modules and a two-pass predictive-consistency
  loss. v45 masks parameter connections in one supervised pass, adds no module
  parameter, and has no consistency term.
- v41 WeightNorm deterministically reparameterizes each Linear as learned
  magnitude plus unit direction. v45 retains raw v13 parameters and samples a
  transient multiplicative binary mask only during training.
- v27 spectral normalization constrains operator norms through power iteration;
  v45 estimates no norm and owns no parametrization buffer.
- v24 SAM perturbs all parameters along a loss-gradient direction and performs
  a second loss evaluation. v45 uses data-independent Bernoulli masks, one loss,
  and no adversarial direction.
- v43 adds target-layer-conditioned FiLM parameters; v45 restores the exact
  unconditional v13 architecture and parameter count.
- v44 adaptively changes three target-layer scalar loss weights; v45 restores
  the single exact v13 row-weighted loss and defines no task split.
- Official v23 aggregate is not used for probability, layer, mask location,
  coefficient, stopping, fold, slice, or selection.

## Required target-free preflight

- Prove exactly five masked Linear matrices with the exact v13 shapes, biases
  unmasked, added parameters/buffers `0`, and state-identical evaluation output
  error `0` against exact v13.
- Prove training keep probability `0.9`, inverted scale, finite gradients,
  deterministic replay for the same seed/step, distinct masks for consecutive
  steps, and empirical keep share within a preregistered broad synthetic
  tolerance `[0.85,0.95]` without reading targets.
- Prove probability `0` is an exact training-forward no-op; executed probability
  remains `0.1` with no alternate candidate.
- Prove evaluation consumes no RNG and is repeat deterministic, public-token
  permutation invariance, masked/future-token isolation, prefix/purge cutoffs,
  max9 fits and action cap unchanged.
- Run two byte-identical target-free preflights with namespace 0 before any
  exactly-once execution. If approved, report RMSE, canonical nominal/fixed
  transport points, fold/month/layer/9-cell gate, CI, action geometry, mask
  counts/hashes, independent QA and access counters.

## Why v44 does not authorize adaptive repair

v44 improved pooled RMSE by `-0.049641307767 C`, nominal
`+0.622875853056` points and fixed-adjusted `+0.501193761446` points, but passed
only `6/9` fold x layer cells. Maximum harm was `+0.043823672068 C` at Jul-Aug
layer 2; Jul-Aug layer 3 and Nov-Dec layer 4 were also harmful. v45 may not use
those exposed cells to change its mask probability, location, or stopping.

## Evidence pins

- v44 result: `fe1b03997ad5f51b125c04eb0816727295e6bd9bb3869d3d0c5395c98e37eb5f`
- v43 result: `2a825dd41e8df0481295566139b77f3a588e469cb344372877f1fcea5d5ab084`
- v41 result: `4c6e3659c3c4635a14a22a1151667990fc92c2ad9d8bc5190bb67e8baad2752c`
- v40 result: `b60534e132c493e76bec94e4a17fcc17b7bc85f30b3f1d8f63be318f3b3cdda5`
- v27 technical receipt: `45140fede7d4f584005b6fd9e32bd7268c0b6d092f18f5317cd6aeeab0da8e01`
- v24 result: `d5d6d7fae5e6ae30278acd698516e1792f90a17ffc81234b87fc6af1141a3b6a`
- v26a prospective gate: `c7fde8c5d9f535ab8080eb561bf082c55e5c7172117d00e12e1479f9b4417680`

## Operation counters

- v45 model fits: `0`
- official/test/sample/query/hidden rows: `0`
- submission CSVs: `0`
- uploads: `0`
