# P2 post-v43 next-axis fingerprint

## 결론

다음 단일 후보는 `p2_target_layer_gradnorm_balanced_deepset_20260901_v44`이다.
저장소 전체 검색에서 P2 GradNorm, adaptive task-loss gradient normalization,
relative inverse training rate, 또는 gradient-norm 기반 target-layer loss-weight
학습 실행은 0건이다. v44는 exact v13 모델을 그대로 두고 target layer 2/3/4의
학습 속도 불균형만 동적으로 보정하는 objective/optimizer 축이다. FiLM이나
layer-specific head를 더 쌓는 v43 변형이 아니며, 이 문서는 실행 승인이 아니다.
현재 v44 fit, official/test/sample/query/hidden access, CSV, upload는 모두 0이다.

## Primary-source boundary

Chen, Badrinarayanan, Lee, and Rabinovich, *GradNorm: Gradient Normalization
for Adaptive Loss Balancing in Deep Multitask Networks*, ICML/PMLR 2018,
https://proceedings.mlr.press/v80/chen18a.html, motivates dynamically balancing
task-gradient magnitudes using relative training rates. The source makes no P2,
ocean-profile, RMSE, layer-safety, transport, or deployment claim. Alpha `1.5`
and the operational guards below are a single preregistered P2 falsification,
not a transferred performance claim.

## Sealed scientific fingerprint

- Base: exact v13 `VerticalDeepSet`, token/context representation, chronological
  prefix plus 7-day purge, layer x calendar-month x KST-day row weights, seeds
  `[20260901, 20260902, 20260903]`, 60 epochs, AdamW `lr=0.001`, weight decay
  `0.0001`, SmoothL1 beta `1.0`, 0.8 champion + 0.2 correction, raw correction
  cap `2.5 C`, final action cap `0.5 C`, maximum 9 fits.
- Only change: split each training minibatch's exact weighted SmoothL1 into three
  fixed target-layer tasks `[2,3,4]`. Initialize positive task weights
  `[1,1,1]`, normalize them to sum `3` after every task-weight update, and use
  their weighted sum for the unchanged model update.
- GradNorm is computed only on the first shared prediction-head matrix
  `head[0].weight`. At the first all-three-task minibatch of each fold/seed,
  seal detached initial losses `L_i(0)`. For each later all-three-task minibatch,
  compute `G_i = ||grad(w_i L_i, head[0].weight)||_2`, relative inverse rates
  `r_i = (L_i/L_i(0)) / mean_j(L_j/L_j(0))`, and minimize
  `sum_i |G_i - stopgrad(mean(G)) * r_i^1.5|` with respect to task weights only.
- Task weights use one fixed plain-SGD update `lr=0.025`, lower clamp `1e-3`,
  followed by exact sum-to-three renormalization. GradNorm gradients never update
  model parameters; the weighted task loss never updates task weights. A batch
  lacking any target layer performs the unchanged weighted-model update using
  current weights and an exact no-op task-weight update. There is no sampler
  change.
- No alpha/task-weight-LR/shared-layer/task-definition sweep, target-layer head,
  FiLM, gradient projection, sign mask, Group-DRO, router, ensemble, row
  deletion, outer-fold tuning, or Public-feedback selection is allowed.
- The prospective v26a gate remains unchanged: all original gates plus at least
  8/9 fold x layer cells non-harm and maximum cell delta RMSE <= `+0.003 C`.

## Exact and semantic audit

- Repository-wide searches for `GradNorm`, `gradient normalization for adaptive
  loss balancing`, `relative inverse training rate`, and task-weight gradient
  norm found 0 executions in P1/P2/P3.
- v18 Group-DRO upweights a worst layer-month scalar risk according to its
  absolute loss. v44 balances three target-layer learning rates through shared
  parameter gradient magnitudes; it has no worst-group maximization.
- v28 PCGrad projects one task-gradient vector away from another on negative
  dot products. v44 performs no projection and changes only three scalar task
  weights.
- v36 Fishr penalizes cross-environment variance of per-sample final-head
  gradients. v44 matches neither per-sample moments nor environments; it uses
  one aggregate norm and relative loss-decay rate per target-layer task.
- v39 zeros parameter coordinates lacking unanimous layer-gradient signs. v44
  masks no coordinate and preserves every model gradient.
- v43 conditions element features with target-layer FiLM. v44 restores the
  exact unconditional v13 architecture and introduces no layer-conditioned
  feature or head.
- Earlier layer-specific NNLS stacks blend frozen model outputs after fitting;
  v44 trains one shared predictor end-to-end and changes no post-fit stack.
- Official v23 aggregate is not used for alpha, task weights, shared layer,
  coefficient, fold, slice, stopping, or selection.

## Required target-free preflight

- Prove exact v13 architecture/parameter count/state and bit-identical inference
  before any optimization; no FiLM, target-specific head, or extra inference
  parameter may exist.
- On a synthetic three-task batch, prove initial weights `[1,1,1]`, positive
  clamp, sum-to-three renormalization, alpha `1.5`, shared matrix exactly
  `head[0].weight`, finite `G_i/r_i/targets`, and deterministic replay.
- Prove equal task losses and equal shared gradient norms yield an exact
  equal-weight fixed point within `1e-7`; a missing-task batch leaves weights
  bit-identical; task-weight backward leaves all model gradients unchanged and
  model-loss backward leaves task-weight gradients absent.
- Prove public-token permutation invariance, masked/future-token isolation,
  prefix/purge cutoffs, no validation participation in `L_i(0)`, and unchanged
  maximum fit count/action cap.
- Run two byte-identical target-free preflights with namespace 0 before any
  exactly-once execution. If approved, report raw RMSE, canonical nominal and
  fixed transport-adjusted points, folds/months/layers/9-cell gate, CI, action
  geometry, task-weight trajectories/hashes, independent QA and access counters.

## Why v43 does not justify execution by itself

v43 improved pooled RMSE by `-0.050199387782 C` with nominal `+0.629878379393`
and fixed-adjusted `+0.508196287783` points, but passed only `6/9` fold x layer
cells. Its maximum harm was `+0.029460495673 C` at Nov-Dec layer 4, with
Jul-Aug layer 2/3 also harmful. This pattern motivates an independently defined
task-balance falsification, but v44 may not use those exposed cells to set alpha,
weights, or stopping.

## Evidence pins

- v43 result: `2a825dd41e8df0481295566139b77f3a588e469cb344372877f1fcea5d5ab084`
- v39 result: `7e8678dd6d6f9e6b6d85c2a726e38f971bd880837b8fe61d1690df6d97b044ce`
- v36 result: `cabb73eb3b15051a4cd0d3d00c26ab53bc0301d6cbca1234d82460487d39954f`
- v28 result: `3674571d4912e288e25d3508fbe5a3b880764f1eb33341b871f267d81b8887d9`
- v18 result: `db25e23ad56d9f9bc3eba283cd31d706e2f0569dbdb516f919eaafed55c7ca70`
- v13 result: `1f1b486ff1cda87887075fc31a04d9f7631c891f8ae3c4ee7ddb14e54fa1d2a4`
- v26a prospective gate: `c7fde8c5d9f535ab8080eb561bf082c55e5c7172117d00e12e1479f9b4417680`

## Operation counters

- v44 model fits: `0`
- official/test/sample/query/hidden rows: `0`
- submission CSVs: `0`
- uploads: `0`
