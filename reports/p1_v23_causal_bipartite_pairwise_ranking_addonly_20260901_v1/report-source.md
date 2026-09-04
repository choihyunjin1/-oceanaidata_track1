# P1 v23 causal bipartite pairwise-ranking add-only cycle

## Terminal decision

`NO_GO_EXPLORATORY_ONLY`, immutable and no retune. The new scientific axis was a symmetric positive-negative pairwise logistic ranking loss over a fixed causal station/layer/calendar context basis. It used no sensor values, temp/psal/depth derivatives, event morphology, recurrence, reservoir, physical-consistency features, event utility labels, or outer labels during fitting.

Gao and Zhou establish consistency conditions for pairwise AUC surrogates, including logistic loss; this motivates the objective but does not imply P1 or F1 gain: [IJCAI 2015 primary paper](https://www.ijcai.org/Proceedings/15/Papers/137.pdf).

The repository audit found no executed P1 pairwise AUC/ranking-loss implementation. The closed SupCon/top-k family learned a contrastive encoder plus soft-F1 and calibrated a groupwise top-k rate; frozen event rankers scored proposed intervals; hierarchical precision-LCB used a pointwise partially pooled logistic score; Group-DRO reweighted MS-TCN environment losses. v23 did none of these. It retained the existing add-only multi-environment veto strictly as an unchanged safety decoder.

## Protocol and gates

- train-only README/train hashes were pinned; three distinct nanosecond cutoffs and a prefix-only 85% inner boundary were used.
- two real zero-operation preflights were byte-identical: SHA-256 `7d14fa6bcd171dfc38bcbf5ee7c5296960808928a4ad5799e3be018524081a64`; support share `1.0`; fits/targets/official/CSV/uploads all `0`.
- one fixed objective, `60,000` seeded symmetric pairs per fit, 3 seeds x Q2/Q3/Q4 = exactly `9` fits, no sweep, no retry, no outer tuning.
- anchor union was mandatory and removals were prohibited. A threshold required Wilson-90 precision LCB at least `0.55`, at least two supported station-layer-chronological-half environments spanning both halves, precision above `0.55` in each, and nonzero TP in every supported environment.

No threshold passed. Q2 and Q3 inner precision was `0` for every sealed quantile. Q4's largest inner precision was `0.0892473` with LCB `0.0750375`. Every candidate had only one supported environment, below the frozen minimum of two. Therefore no outer proposal was materialized.

## Canonical metrics

| metric | value |
|---|---:|
| pooled F1 | `0.8604836038423319` |
| TP / FP / FN | `12989 / 1146 / 3066` |
| additions / anchor removals | `0 / 0` |
| raw F1 delta | `0.0` |
| bootstrap CI90 | `[0.0, 0.0]` |
| nominal / transport-adjusted points | `0.0 / 0.0` |
| long-event interior recall | `0.8107135718568859` |
| long-event boundary recall | `0.779835390946502` |
| offset / drift recall | `0.6477211796246649 / 0.6595061728395062` |
| fits / runtime seconds | `9 / 20.01600000000326` |

All fold and station-layer deltas were exactly zero; action slices are empty. Outer target reads before all seals, official/test/sample/submission/hidden reads, CSVs, uploads, and anchor removals were all zero.

## QA and hashes

- focused pytest: `4/4 PASS`; Ruff: `PASS`
- lifecycle-aware post-terminal QA: every check `PASS`
- config SHA-256: `02b87b60c7f7465f8f5d2b98947bb25f7edf9b955625e229efd91302653bbefe`
- runner SHA-256: `4b1a392d52fce2e03da7af3d40c0bdb655f64d0e7a6b28d1a8b3237e33ee8c90`
- completion SHA-256: `91370e9c30e873640349b978a72550bb8d99c62c8739476e4d41f878f2281f14`
- lock SHA-256: `0d4d1b605eade0fc534c99605516d678944ad58fc7d337ec2001595af923fef3`
- result SHA-256: `f109b74b0dbeb2c6cf50ef0bdbdaefb20ecaadb4bda2d31dd9913963447b0d64`

v23 closes this exact pairwise-context objective. Its pair count, context basis, loss, quantiles, budget, and veto must not be retuned from these results.
