# P2 post-v40 next-axis fingerprint

## 결론

다음 단일 후보는 `p2_all_linear_weight_normalized_deepset_20260901_v41`이다. P2와 저장소 전체에서 neural Linear weight-normalization reparameterization 실행은 0건이다. 이 후보는 v40 dropout/consistency를 재사용하지 않고 exact v13의 다섯 Linear weight vector를 길이 `g`와 단위 방향 `v / ||v||`로 분리하는 최적화 좌표계만 바꾼다. 이 문서는 실행 승인이 아니며 현재 fit, official access, CSV, upload는 0이다.

## Primary-source boundary

Salimans and Kingma, *Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks*, NIPS 2016, https://proceedings.neurips.cc/paper_files/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html, motivates decoupling each neuron's weight-vector length and direction without minibatch dependence. The source makes no P2, ocean-temperature, RMSE, or transport claim.

## Sealed scientific fingerprint

- Base: exact v13 tokens/context, prefix-only folds, 7-day purge, layer x calendar-month x KST-day weights, seeds `[20260901, 20260902, 20260903]`, 60 epochs, AdamW `lr=0.001`, weight decay `0.0001`, 0.8 champion + 0.2 raw correction, raw cap 2.5 C, final action cap 0.5 C, maximum 9 fits.
- Only change: apply `torch.nn.utils.parametrizations.weight_norm(..., dim=0)` once to each of the five exact v13 Linear modules before constructing AdamW. Biases, nonlinearities, widths, pooling, loss, batch order, inference, blend and caps remain unchanged.
- The learned magnitude is not fixed to one. No norm coefficient, power iteration, spectral bound, activation normalization, batch statistic, dropout, consistency term, sweep, scheduler, router, ensemble, row deletion, or Public-feedback selection is allowed.
- Prospective gate remains v26a: original gates plus at least 8/9 fold x layer cells non-harm and maximum cell delta RMSE <= +0.003 C.

## Exact and semantic audit

- Repository searches for `weight normalization`, `weight_norm`, `WeightNorm`, `weight-normalized`, and `Salimans Kingma` found no neural P2 execution. The only textual hits were unrelated sample-weight normalization and a P3 receipt label.
- v27 is a closed technical spectral-normalization family. It globally constrains the operator spectral norm with power iteration and a fixed norm-one contract. v41 neither imports nor retries v27, changes no v27 tolerance, performs no power iteration, and learns an independent magnitude for every output weight vector.
- v29 maintains slow Lookahead parameters and periodically interpolates optimizer states; v41 has one contemporaneous parameter state and no temporal interpolation.
- v34 centralizes gradients immediately before AdamW; v41 leaves gradients ordinary and changes only the parameterization through which they flow.
- v35 changes AdamW to RAdam; v41 retains exact AdamW.
- v40 uses Bernoulli hidden masks and a two-pass prediction loss; v41 is deterministic per input and retains one exact v13 SmoothL1 pass.
- Official v23 aggregate is not used to select this mechanism, coefficient, layer, or slice.

## Required target-free preflight

- Prove applying weight normalization to a state-identical v13 model preserves its initial function within `1e-6` before any optimizer step.
- Prove exactly five Linear modules and no bias are parametrized; each learned `g` matches the initial per-output weight-vector norm and every effective weight is finite.
- Prove no spectral-normalization buffer, power-iteration vector, global norm-one constraint, batch statistic, or activation normalization exists.
- Prove public-layer permutation invariance, masked-token isolation, deterministic inference and prefix/purge contracts.
- Run two byte-identical target-free preflights with namespace 0 before exactly-once execution.
- Report raw RMSE, canonical nominal and fixed transport-adjusted points, fold/month/layer/9-cell metrics, CI, action geometry, parametrization state hashes, independent QA and access counters.

## Evidence pins

- v40 result: `b60534e132c493e76bec94e4a17fcc17b7bc85f30b3f1d8f63be318f3b3cdda5`
- v27 closed failure report: `e1952f7a2becdd1cc4929c988e9db17d6a66618a2f3c6aa7613615b1f42663a6`
- v29 result: `2db32dedb85ab0e83f63aa791f2abf49604d4aa7e3ca9aecefaee5bc0e186e54`
- v34 result: `e08b1e30fcaaa86bbec41657857f541b29e1a7657fa4ab4629d0c390561c46fa`
- v35 result: `1e4ebf61b2ef5ac63244bf68a5723a1cffcf81e26e1fe2f0367a5be3d69ca9ff`
- v26a gate: `c7fde8c5d9f535ab8080eb561bf082c55e5c7172117d00e12e1479f9b4417680`

## Operation counters

- model fits: `0`
- official/test/sample/query/hidden rows: `0`
- submission CSVs: `0`
- uploads: `0`
