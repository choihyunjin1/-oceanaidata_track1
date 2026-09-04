# P1 v24 causal station-adversarial invariant add-only cycle

## Terminal decision

`NO_GO_EXPLORATORY_ONLY`, immutable and no retune. A fixed 8-unit causal encoder was trained with anomaly BCE plus a gradient-reversed station classifier. The source prefix alone supplied labels and station nuisance IDs; outer rows, outer labels, and outer domain rows were excluded from training.

Ganin et al. show that gradient reversal can promote task-discriminative but domain-indiscriminate features. v24 is only a station-nuisance domain-generalization screen, not the paper's source/target adaptation setting, and the paper makes no P1 claim: [JMLR primary source](https://www.jmlr.org/papers/v17/15-239.html).

The objective is distinct from P1 Group-DRO's worst-environment loss weighting, SupCon's class contrast, P2 CORAL's covariance matching, and v23's pairwise ranking. It used no score recalibration, event model, pairwise ranking, outer-target adaptation, or result-selected coefficient.

## Protocol and support

- two real preflights were byte-identical: SHA-256 `0cb2f41737a3bde90970712a75d562b61907103459aecebe5e014c98c3ff8fd7`; causal feature support share `1.0`; fits/targets/official/CSV/uploads all `0`.
- fixed prefix-normalized current temperature, exact-cadence backward difference, missing/gap flags; station one-hot was nuisance-only. No psal/depth, event morphology, recurrence, reservoir, or physical-consistency inputs.
- fixed adversarial coefficient `0.1`, 8 hidden units, 12 epochs, at most 80,000 prefix rows per fit, 3 seeds x Q2/Q3/Q4 = exactly `9` fits; no sweep, retry, or outer tuning.
- anchor union and the existing station-layer-chronological-half precision/transport veto were unchanged.

## Transport failure

The Q2 inner selector chose quantile `0.9975`: `141` proposals, precision `0.957447`, Wilson-90 LCB `0.919830`. Both supported inner environments were S-ORS layer 5, one in each chronological half, and each had precision `1.0`. Q3 and Q4 selected no threshold.

On the untouched Q2 outer surface, however, the action moved to G-ORS layer 1 (`96`), I-ORS layer 1 (`230`), and I-ORS layer 7 (`6`). All `332` additions were false positives. Thus station-adversarial invariance did not produce station-layer transport; it removed no anchor positives but caused a clear precision collapse. This exact objective, coefficient, basis, sampling, capacity, and decoder are closed.

## Canonical metrics

| metric | value |
|---|---:|
| pooled incumbent / candidate F1 | `0.8604836038423319 / 0.8511237795688356` |
| raw F1 delta | `-0.009359824273496353` |
| candidate TP / FP / FN | `12989 / 1478 / 3066` |
| additions / TP additions / precision | `332 / 0 / 0.0` |
| anchor removals | `0` |
| bootstrap CI90 | `[-0.013291383894181812, -0.006278461619619525]` |
| nominal / transport-adjusted points | `-0.24878779970885603 / -0.07463633991265681` |
| Q2 / Q3 / Q4 delta F1 | `-0.024687935329047228 / 0.0 / 0.0` |
| long-event interior / boundary recall | `0.8107135718568859 / 0.779835390946502` |
| offset / drift recall | `0.6477211796246649 / 0.6595061728395062` |
| fits / runtime seconds | `9 / 21.718999999997322` |

Official/test/sample/submission/hidden reads, CSVs, uploads, outer rows in training, outer labels before all seals, and anchor removals were all zero.

## QA and hashes

- focused pytest `4/4 PASS`; Ruff `PASS`; lifecycle-aware post-terminal QA all checks `PASS`
- config `6ca2904c09084ada7d7a49893e14399c8bc3a2f81f67de52db186e3032098392`
- runner `186f26fc9c7edea1eea01e78a485c10719ed19a0823f5ec74d79a2406f32fc6a`
- completion `721d4ef070567fe69a3fa7dcfdfd9eeed3bdea3998c0b1daf9275e5d608c33c0`
- lock `0cdab480f6a671da79dfb78c8fb9411e16d49f3466ddaa6c6c9ca8b2d695d10f`
- result `9915b48421f5e8b3d70e9c100286ede5a45c219cc3b5665c7dd3d7fd9974be36`
