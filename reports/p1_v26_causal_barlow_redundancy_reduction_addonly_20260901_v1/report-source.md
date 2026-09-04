# P1 v26 causal Barlow redundancy-reduction add-only cycle

## Terminal decision

`NO_GO_EXPLORATORY_ONLY`, immutable and no retune. The new axis used label-free two-view cross-correlation identity training, froze the encoder, and trained only a linear supervised add-only head. It used no contrastive negatives, normal prototype, hypersphere, reconstruction target, exact degradation mask, event model, domain adversary, pairwise ranking, or outer row in training.

[Zbontar et al. (ICML 2021)](https://proceedings.mlr.press/v139/zbontar21a.html) motivate self-supervised redundancy reduction by making the cross-correlation of two augmented views approach the identity. This motivates only the objective; it is not evidence for P1 performance.

## Prospective guard amendment

The separate amendment `p1_v26_transport_guard_amendment_20260901_v1`, SHA-256 `6d0f6f21aaa72410ad84d6a42e400b94a3e681f65670f81a02d464989b592383`, was sealed before v26 metrics. It is not retroactive and does not rerun or rescue v24. New candidates must preserve both chronological halves and also include at least two distinct station-layer identities and two distinct stations, with TP>0 and precision>0.55 in every supported environment, while retaining the pooled Wilson-90 LCB and minimum additions.

## Protocol and gates

- two real preflights were byte-identical: SHA-256 `457bc75efeed8f04d4a897de979c20f84e8d1848bc0361ef568604cf14b06a03`; causal multi-lag support share `0.8544275440`; fits/targets/official/CSV/uploads all zero.
- fixed lags `0/1/6/36/144`, prefix-only robust normalization, 16-unit tanh encoder/projector, 6 pretrain epochs, 6 frozen-encoder head epochs, 3 seeds x Q2/Q3/Q4 = exactly 9 fits.
- Barlow off-diagonal coefficient `0.0051`, view dropout `0.1`, Gaussian view noise `0.05`, 80,000-row cap, fixed quantiles/budget, no sweep, retry, or outer tuning.

No threshold passed. Best Q2 precision/LCB was `0.438596/0.335712`; Q3 `0.25/0.190078`; Q4 `0.093548/0.079000`. Several lower quantiles had the newly required cell/station diversity, showing that the no-op was not caused solely by the amendment; their precision was far below the unchanged 0.55 requirement.

## Canonical metrics

| metric | value |
|---|---:|
| pooled F1 | `0.8604836038423319` |
| TP / FP / FN | `12989 / 1146 / 3066` |
| additions / anchor removals | `0 / 0` |
| raw F1 delta / CI90 | `0.0 / [0.0, 0.0]` |
| nominal / transport-adjusted points | `0.0 / 0.0` |
| long-event interior / boundary recall | `0.8107135718568859 / 0.779835390946502` |
| offset / drift recall | `0.6477211796246649 / 0.6595061728395062` |
| fits / runtime seconds | `9 / 23.031999999991967` |

All folds and station-layer slices were exact no-ops. Official/test/sample/submission/hidden reads, CSVs, uploads, outer target reads before all seals, labels in pretraining, outer rows in training, and anchor removals were zero.

## QA and hashes

- focused pytest `4/4 PASS`; Ruff `PASS`; post-terminal lifecycle/hash QA all checks `PASS`
- config `df57f36528e92d2d386b7a12f91d57614753ed1a210b375555b6712d3cd218fe`
- runner `9f76ab16a06c3107eb10c3935e425165b7436c67d093595ade24184a643556e2`
- completion `19318ad6e6bda084c87ee7513da2e03ec404898363a3cb243d1a3ceb60d13c91`
- lock `fa6bd06aeb5ec744210cf45ff14db86063d6be04c4aca4974efd51f95bc0baef`
- result `241e050c3a73d908e675be0f748b84e233b08a5adf7d456dfbb0b57af333064f`

This exact Barlow objective, causal lags, view corruption, capacity, head, and amended guard are closed.
