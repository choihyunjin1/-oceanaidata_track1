# P1 v14 causal prefix-dictionary discord — terminal Deep Research report

Audience: P1 scientific/engineering review. Date: 2026-09-01 KST. Scope: local README/train and authenticated historical Q2-Q4 champion parts only; official/test/sample/submission/hidden/CSV/upload excluded.

## Direct answer

The genuinely unused v14 axis was a causal, non-subsequence-normalized prefix-dictionary approximation to Matrix Profile discord scoring. It is scientifically distinct, technically valid, and terminal `NO_GO_EXPLORATORY_ONLY`: the prospectively frozen transport-stability veto rejected every threshold before outer scoring, producing an exact champion no-op. The family must not be retuned or replayed.

[Yeh et al., Matrix Profile I](https://doi.org/10.1109/ICDM.2016.0179) defines the profile as each subsequence's nearest-neighbor distance and connects profile maxima to discords. The authors' extended paper states that a matrix-profile element is the distance from a subsequence to its nearest neighbor and that the highest profile point corresponds to a discord ([UCR author manuscript](https://www.cs.ucr.edu/~eamonn/MP_journal.pdf)). v14 uses only that mechanism: two fixed causal windows (12 and 48 rows) compare against 16 nonoverlapping past windows, returning level/derivative nearest, second-nearest, median, and motif-ratio features.

This is not a standard exact Matrix Profile reproduction. The local [P1 academic-method audit](../../P1_ACADEMIC_METHODS_SCOUT_2026-08-13.md) correctly rejects per-subsequence z-normalization because it can erase additive offsets, but explicitly leaves an unnormalized candidate-local drift variant for later. v14 therefore uses one prefix-fitted station-layer robust center/scale and never recenters or rescales individual query/reference subsequences. Its finite lagged dictionary may miss distant motifs; expanding the dictionary or windows after this result is forbidden.

## Repository semantic and negative-evidence audit

- No executed P1 Matrix Profile, discord, nearest-subsequence, or similarity-join implementation was found. The only exact P1 mention was the standard-z-normalization NO-GO and the unexecuted unnormalized variant.
- Typed duration and factorial semi-Markov decoders already exist, so the suggested HSMM route was rejected as a semantic duplicate before fit.
- v10 recurrence compares scalar states at fixed lags/radii; v14 compares whole subsequences and takes a nearest dictionary match.
- MiniRocket uses fixed convolution/PPV features; long-event rescore classifies frozen proposals. v14 does neither.
- Wavelet, MMD, visibility, Group-DRO, conformal/subspace, hierarchical precision LCB, and proposal-rescore families remain closed and were not reused.

## Prospective transport safeguard

To structurally address v12's 447-addition/0-TP transport collapse, pooled inner LCB was insufficient for sealing. Each threshold additionally had to pass station x layer x chronological-half environments: at least 10 proposals per supported environment, at least two supported environments spanning both halves, precision strictly above `0.55` in every supported environment, and no supported environment with TP `0`. Thresholds, windows, dictionary lags, gate, model, and seeds were all hashed before metrics.

Q3 illustrates the safeguard's effect: pooled inner precision LCB reached `0.7725927229367294`, but zero candidate threshold passed the environment rule. Q2/Q4 pooled maximum LCBs were approximately `0` and `0.2030`; all three folds sealed `chosen=null`.

## Execution, metrics, and QA

- Focused pytest `4 passed`; Ruff PASS; pre-execution QA PASS.
- Two real zero-operation preflights were byte-identical: `6d3394e356f90c8c46dd0e8b7a4d9c619eb494fb67d89716dbbe276889e78997`; artifact/lock absent.
- Feature nonzero share `0.7816290848789632`; all ten feature variances passed.
- Exactly 9 unique fits; runtime `37.218 s`; outer target reads before all seals `0`.
- Additions/removals `0/0`; every station/layer/quarter action count `0`.
- Pooled incumbent=candidate F1 `0.8604836038423319`, TP/FP/FN `12989/1146/3066`; precision/recall `0.9189246551114255/0.8090314543755839`.
- Fold F1 Q2/Q3/Q4 `0.7784135753749013 / 0.8970588235294118 / 0.9090245682315738`; all deltas `0`.
- Cluster bootstrap CI90 `[0,0]`; nominal/transport-adjusted points `0/0`.
- Long-event interior recall `0.8107135718568859` across 81 runs / 15,009 rows; boundary recall `0.779835390946502` across 972 boundary rows; offset/drift recall `0.6477211796246649/0.6595061728395062`; all deltas `0`.
- Official/test/sample/submission/hidden reads, CSV, uploads: `0`.
- Config/runner/completion/lock hashes: `69d6020a38a3f5f6b80700f46cac004ba75e92a283144e97b8829393c2766644`, `696a842be132dcbe28f2e85580f6c422311d3cbe0940a01813962a263b971269`, `23c423dbf0b8672797aab1ee9facae5f367a646f8d7ba26b819fd3037d83d99f`, `81635d875f7cf8a73602ebe73e42b702775eb9e767348d6222aa96ab602d61d7`.
- Lifecycle-independent QA recomputed pooled counts/F1, long-event interior and boundary recall, add-only/removal-zero, all seal hashes, nine model hashes, schema, outer isolation, and access-zero counters: `PASS`.
- Result SHA-256: `3a81d48b196256852bf564349b626cd62ca0069355389e25f5adbf91c61225a6`.

## Claim-to-source ledger

| Claim | Source | Publisher/date | URL | Access note |
|---|---|---|---|---|
| Matrix Profile stores nearest-neighbor subsequence distances; maxima encode discords | “Matrix Profile I” / extended author manuscript, Yeh et al. | IEEE ICDM 2016 / Springer journal extension 2018 | [DOI](https://doi.org/10.1109/ICDM.2016.0179), [author PDF](https://www.cs.ucr.edu/~eamonn/MP_journal.pdf) | Primary paper and author-hosted manuscript; mechanism only |
| Standard per-subsequence z-normalization may erase additive offset; unnormalized candidate-local drift variant remained unexecuted | `reports/P1_ACADEMIC_METHODS_SCOUT_2026-08-13.md` | Local P1 audit, 2026-08-13 | local repository | SHA-pinned negative evidence; not an external performance claim |
| HSMM, recurrence, MiniRocket, and proposal-rescore fingerprints already exist and differ/duplicate as recorded | SHA-pinned repository files in v14 config | Local repository, through 2026-09-01 | local repository | Full semantic fingerprint checked before execution |

Research stopped because the mechanism had primary support, exact/semantic repository duplication was bounded, the main transport risk had a prospective veto, and further Matrix Profile variants would only create result-driven window/dictionary choices.

## Next nonduplicate audit candidate

Do not reopen Matrix Profile, MMD, recurrence, visibility, or the veto thresholds. A plausible next audit-only axis is a small causal Koopman/dynamic-mode innovation representation: prefix-fit a fixed low-rank linear evolution operator, score one-step nonlinear observable innovations, and apply the same environment veto. It must first be checked against asynchronous latent-state GP, temporally fused RPCA, forecast/backcast residuals, and dynamic-factor/state-space families; any semantic overlap closes it at zero fits.
