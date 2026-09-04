# P1 v16 causal delay-embedding persistence — terminal report

## Decision

`NO_GO_EXPLORATORY_ONLY`. The preregistered station×layer×chronological-half transport veto rejected every fixed threshold in Q2–Q4, so the candidate was an exact anchor no-op. This family is closed; no window, delay, stride, threshold, or gate retuning is permitted.

## Deep Research and semantic audit

Perea and Harer analyze sliding-window/time-delay point clouds with persistent homology and relate maximum persistence to signal periodicity ([FoCM 2015 / arXiv](https://arxiv.org/abs/1307.6188), DOI `10.1007/s10208-014-9206-z`). v16 is deliberately narrower: causal 0-dimensional Vietoris–Rips finite death times, represented exactly by Euclidean minimum-spanning-tree edges. It does not implement their H1 periodicity score and the paper supplies no P1 performance claim.

Three prospective fingerprints were compared before execution:

- selected: delay-embedding H0 persistence, because the repository had no exact implementation/execution and its all-scale component-merger geometry differs from v10 fixed-radius recurrence and v13 endpoint visibility;
- rejected: empirical tail-copula change, because its P1 action fingerprint overlaps v9 prefix-fitted dependence change and v12 recent-vs-prefix distribution shift;
- rejected: supervised shapelets, because discriminative subsequence morphology overlaps causal CIF, MiniRocket, and v14 nearest-subsequence matching ([Grabocka et al., KDD 2014](https://doi.org/10.1145/2623330.2623613)).

The tail-copula source supports structural-break tests for extremal dependence, not this P1 recipe ([Bücher et al., Journal of Econometrics 2015](https://doi.org/10.1016/j.jeconom.2015.02.002)). Full hashes and claim boundaries are frozen in the config.

## Frozen method and validation

Each station×layer series used a prefix-only robust center/scale, cadence-gap resets, a 96-row causal window, delay 6, dimension 3, 12 points, and stride 6. Eight H0/MST summaries were passed to one fixed L2 logistic architecture with three seeds per Q2/Q3/Q4 fold (9 fits). The anchor was preserved by bitwise OR, removals were forbidden, and outer labels were read only after all three fold predictions were sealed.

The same prospective transport veto used multiple nonoverlapping inner environments. A threshold required at least two supported station×layer×half environments spanning both halves, more than 0.55 precision in every supported environment, and no supported environment with zero true positives. No threshold passed. The largest pooled inner Wilson LCB was `0.472789`-precision candidate's `0.425362` in Q3; Q2 and Q4 were substantially worse, and multiple supported cells had zero TP.

## Canonical result

- pooled anchor/candidate F1: `0.8604836038423319`; delta `0.0`; TP/FP/FN `12989/1146/3066`
- folds Q2/Q3/Q4 F1: `0.7784135753749013 / 0.8970588235294118 / 0.9090245682315738`; all deltas `0.0`
- additions / addition TP / addition precision: `0 / 0 / 0.0`; anchor removals `0`
- paired block bootstrap 90% CI: `[0.0, 0.0]`, 2,000 replicates, 3,089 clusters
- long-event interior recall: `0.8107135718568859` over 15,009 rows, delta `0.0`
- long-event boundary recall: `0.779835390946502` over 972 rows from 81 runs, delta `0.0`
- offset / drift recall: `0.6477211796246649 / 0.6595061728395062`
- nominal / transport-adjusted expected points: `0.0 / 0.0`
- runtime / fits: `75.391 s / 9`; nine unique model hashes verified
- result SHA-256: `d86318f9a6f0ac23dde396c0902be07409a0c2a51af4e136c5692c6611624eb8`
- config / runner SHA-256: `80c9f87ad3ea1f376ec257b2dcbe73f45c9d1f1d3fe5123f02f8b34bb770537e / 1cfcd42aec0bbf61fbba242768bf25328b4c8d3e07a318ce4d6b5d72a78abaad`

Focused pytest was 5/5 PASS and Ruff PASS. The two real zero-operation preflights were byte-identical at SHA-256 `9084b78abc7278001b2645c9dfa5c634c6674df75c8ae1499a5962afe0f0b3f6`; representation support was nonzero on `0.7481994474` of rows and all eight variances exceeded the frozen floor. Lifecycle independent QA recomputed scores, long-event receipts, hashes, 9 unique models, add-only geometry, and outer isolation: PASS.

Only source `README.md` and `train.csv` were read. Official/test/sample/submission/hidden reads, CSV materialization, uploads, and submissions were all zero.

## Limitation and next axis

H0 persistence measures connected-component merger scales, not loop persistence. This exact bounded family failed its prospective transport gate and should not be expanded adaptively. A genuinely separate next audit could examine a causal ordinal-pattern transition surprisal representation: it models symbolic transition dynamics rather than metric recurrence, visibility, distance distributions, or learned subsequence templates. It must first be checked against the existing soft-symbolic and recurrence registries and must not inherit any v16 outcome-driven settings.
