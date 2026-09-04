# P2 post-v39 next-axis fingerprint

## 결론

`p2_predictive_dropout_consistency_deepset_20260901_v40`을 다음 단일 후보로 제안한다. P2 저장소에서 R-Drop, two-dropout predictive consistency, dropout-consistency 실행은 0건이다. v39의 gradient-coordinate unanimity를 조정하거나 재사용하지 않고, exact v13 encoder에 두 stochastic views의 predictive consistency만 추가하는 별도 regularization 축이다. 이 문서는 실행 승인이 아니며 fit, official access, CSV, upload는 0이다.

## Sealed scientific fingerprint

- Base: exact v13 data, prefix-only folds, 7-day purge, layer x calendar-month x KST-day weights, seeds `[20260901, 20260902, 20260903]`, 60 epochs, AdamW, 0.8 champion + 0.2 raw correction, raw correction cap 2.5 C, final action cap 0.5 C, maximum 9 fits.
- Only candidate package: fixed dropout probability `0.1` after each hidden ReLU; two independent dropout passes on the same labeled row; average of the two weighted SmoothL1 losses plus coefficient `1.0` times `0.5 * weighted_mean((prediction_a - prediction_b)^2)`; deterministic dropout-off inference.
- The dropout and consistency term are one inseparable stochastic-consistency intervention. There is no dropout/coefficient/layer-location sweep, router, ensemble, row deletion, label-dependent gate, or use of official v23 feedback for selection.
- Prospective gate remains the v26a contract: all original gates, at least 8/9 fold x layer cells non-harm, and every cell delta RMSE <= +0.003 C.

## Novelty and boundaries

- P2 exact/semantic execution hits: `0` for `R-Drop`, `rdrop`, `two-dropout`, `dropout consistency`, and `predictive consistency` under P2 configs/runners/reports/tests.
- v26 mixup interpolates two rows within a layer-month group; v40 never mixes rows and instead compares two stochastic submodels on the identical row.
- v24 SAM perturbs parameters adversarially; v40 applies Bernoulli hidden-unit masks and matches predictions, with no parameter-neighborhood optimization.
- v23 input-gradient regularization penalizes sensitivity to public-temperature inputs; v40 has no input derivative.
- v38 output penalty shrinks one deterministic prediction toward zero; v40 penalizes only disagreement between two stochastic predictions.
- v39 AND-mask changes optimizer gradients using cross-layer sign unanimity; v40 uses the ordinary v13 AdamW gradient of one scalar objective and no per-task gradient surgery.
- P1 v39 is disclosed as cross-problem adjacency. No P1 code, result, threshold, gate, or performance is transferred. Its primary source is used only to motivate predictive consistency.

## Primary-source boundary

Liang et al., *R-Drop: Regularized Dropout for Neural Networks*, NeurIPS 2021, https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html, motivates matching two dropout submodels. The P2 fixed-variance Gaussian regression translation to squared prediction disagreement is a preregistered local hypothesis; the paper makes no P2 or temperature-reconstruction performance claim.

## Required preflight and QA

- Prove two training passes differ under fixed dropout while rerunning the same seed is deterministic.
- Prove dropout-off inference is deterministic, public-layer permutation invariant, and invariant to future/masked-token perturbation.
- Prove the consistency term is nonnegative, zero for equal predictions, symmetric, finite, and acts on no target from validation/future rows.
- Run two byte-identical target-free preflights with namespace 0 before exactly-once execution.
- Report raw RMSE, canonical nominal and fixed transport-adjusted points, fold/month/layer and 9-cell metrics, day-block CI, action geometry, hashes, independent QA, and access counters.

## Evidence pins

- v39 result: `7e8678dd6d6f9e6b6d85c2a726e38f971bd880837b8fe61d1690df6d97b044ce`
- v39 config: `588b557bbe30ff1353388a4394e3feb4d82edfdae8bf834d1bbbb13f3c19922b`
- v39 runner: `dee0ce31c1f57d564af1852dc9601a4777f5cec72cdbcf925c1b368b0ccf85b1`
- v26a prospective gate: `c7fde8c5d9f535ab8080eb561bf082c55e5c7172117d00e12e1479f9b4417680`
- v13 result: `1f1b486ff1cda87887075fc31a04d9f7631c891f8ae3c4ee7ddb14e54fa1d2a4`
- P1 v39 adjacency config: `5e4be899586dbe7b6098ca19f8d6e2703d1d3689cb2d03a23c5abba5d46a77a3`
- P1 v39 report: `84c35143ce47e7b5947ae26001962dc3132c7d30c2cba1010cfae37be1eeb756`

## Operation counters

- model fits: `0`
- official/test/sample/hidden/query rows: `0`
- submission CSVs: `0`
- uploads: `0`
