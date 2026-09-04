# P2 post-v42 next-axis fingerprint

## 결론

다음 단일 후보는 `p2_target_layer_film_conditioned_deepset_20260901_v43`이다.
P2와 저장소 전체에서 FiLM, feature-wise linear modulation, target-conditioned
element encoder, 또는 conditional affine element modulation 실행은 0건이다. v43은
exact v13이 이미 head context로 받는 target-layer one-hot을 public-token element
embedding에도 한 번 주입해, 동일 public profile을 복원 대상 layer별로 다르게
해석할 수 있는지 검증한다. 새 데이터·새 loss·새 gate는 없다. 이 문서는 실행
승인이 아니며 v43 fit/access/CSV/upload는 0이다.

## Primary-source boundary

Perez, Strub, de Vries, Dumoulin, and Courville, *FiLM: Visual Reasoning with a
General Conditioning Layer*, AAAI 2018,
https://doi.org/10.1609/aaai.v32i1.11671, motivates feature-wise affine
transformations derived from conditioning information. The source makes no P2,
ocean-profile, RMSE, layer-safety, or transport claim.

## Sealed scientific fingerprint

- Base: exact v13 data/tokens/context, chronological prefix plus 7-day purge,
  layer x calendar-month x KST-day weights, seeds `[20260901, 20260902,
  20260903]`, 60 epochs, AdamW `lr=0.001`, weight decay `0.0001`, weighted
  SmoothL1 beta 1.0, 0.8 champion + 0.2 raw correction, raw cap 2.5 C, final
  action cap 0.5 C, maximum 9 fits.
- Only change: after the second shared element ReLU and before masked mean/max
  pooling, apply `h' = gamma(layer) * h + beta(layer)` with one bias-free
  `Linear(3,64)` driven by the existing target-layer `[2,3,4]` one-hot. The
  first 32 outputs are gamma and the last 32 beta.
- Initialize every gamma exactly `1` and every beta exactly `0`, proving the
  initial function equals exact v13 within `1e-6`. These 192 FiLM parameters are
  learned jointly with v13; no normalization, coefficient, depth-specific public
  token ID, month/station/year conditioning, sweep, router, ensemble, row
  deletion, or Public-feedback selection is allowed.
- The same gamma/beta is broadcast to all public elements in a row, so public
  token permutation invariance is retained. Target-layer one-hot remains in the
  unchanged head context; it is not removed or duplicated from the data.
- Prospective v26a gate remains unchanged: original gates plus at least 8/9
  fold x layer cells non-harm and max cell delta RMSE <= +0.003 C.

## Exact and semantic audit

- Repository-wide searches for standalone `FiLM`, `feature-wise linear
  modulation`, `conditional affine`, `target-conditioned element`, and
  `hypernetwork` found 0 executed P2/P1/P3 candidates.
- v12/v13 share one unconditional public-element MLP and expose target layer
  only after pooling. v43 keeps that geometry but allows the known target layer
  to modulate each encoded public element before pooling.
- v15 self-attention models pairwise interactions among public tokens but is
  not conditioned by the target-layer identity inside its element interaction.
- v16 depth graph passes messages using fixed physical-depth edges; v43 adds no
  edge, graph, or depth-specific public identity.
- v31 uses target layer only to define a calendar-month adversarial task. v43
  has no domain loss, discriminator, gradient reversal, or month label.
- v40/v42 add stochastic or adversarial consistency losses. v43 restores one
  exact deterministic SmoothL1 pass with no perturbation.
- v41 changes parameter coordinates for every Linear. v43 uses ordinary v13
  Linear weights plus exactly one identity-initialized conditional affine map.
- Official v23 aggregate is not used for architecture, initialization, layer,
  fold, or selection.

## Required target-free preflight

- Prove exactly one `Linear(3,64,bias=False)` FiLM generator, 192 added
  parameters, gamma-one/beta-zero initialization, and initial-function maximum
  error <= `1e-6` against state-identical v13.
- Prove one-hot layer 2/3/4 selects only its own FiLM column; changing layer
  identity can change modulation after training, while gamma=1/beta=0 is exact
  no-op.
- Prove public-token permutation invariance, masked/future-token isolation,
  finite outputs/gradients, deterministic replay, and exact unchanged inference
  structure.
- Prove no batch/layer/weight/spectral normalization, attention, month router,
  perturbation or extra loss exists.
- Run two byte-identical target-free preflights with namespace 0 before one
  exactly-once execution; report RMSE, canonical nominal/transport points,
  folds/months/layers/9-cell gate, CI, action geometry, FiLM state hashes and QA.

## Evidence pins

- v42 result: `36f0a441cdaef2b9c129e1fbab129c786bd472940b99b7803d5132ed04a4833d`
- v41 result: `4c6e3659c3c4635a14a22a1151667990fc92c2ad9d8bc5190bb67e8baad2752c`
- v40 result: `b60534e132c493e76bec94e4a17fcc17b7bc85f30b3f1d8f63be318f3b3cdda5`
- v31 result: `e73dbba22d5e2f3f1b5b15583d32ec2ee4bbfe3dc924af9fa2a252a5003325a7`
- v26a gate: `c7fde8c5d9f535ab8080eb561bf082c55e5c7172117d00e12e1479f9b4417680`

## Operation counters

- v43 model fits: `0`
- official/test/sample/query/hidden rows: `0`
- submission CSVs: `0`
- uploads: `0`
