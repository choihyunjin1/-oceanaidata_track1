# P1 v10 causal recurrence / laminar-state terminal report

- Experiment: `p1_v10_causal_recurrence_laminar_state_addonly_20260901_v1`
- Surface: `EXPLORATORY_REUSED_SURFACE`
- Decision: `NO_GO_EXPLORATORY_ONLY`; sealed inner selections chose no threshold in Q2, Q3, or Q4 because every Wilson precision LCB was below the frozen `0.55` gate. No result-driven retuning or restart was performed.
- Representation: strictly past-only, station-layer-reset causal recurrence/laminar features (lag recurrence, diagonal recurrence, laminar length, prefix-reference density/distance, persistence, state shift). Groups lacking the frozen 96-row prefix reference receive all-zero representation for that fold, preventing later rows from defining prefix authority.
- Semantic audit: not Matrix Profile (no subsequence z-normalization; prefix-reference distance retains level offsets), not the existing soft-symbolic PAA-transition scorer, hard plateau/run-length family, typed HSMM duration decoder, or v7 signed-path representation. P3 v29 was acknowledged only as cross-problem provenance; no P3 results or hyperparameters were transferred.
- Primary methodological motivation: Marwan et al., *Physics Reports* 438 (2007), DOI `10.1016/j.physrep.2006.11.001`. This experiment is a fixed P1 recipe, not a reproduction claim.

## Execution and canonical metrics

- Two accepted real zero-operation preflights were byte-identical: `6403b8d43fb198c243a54660b5d348129e7604d4620cfec5ddd3e791ecd01e95`.
- Exactly 9 distinct model fits (3 seeds x 3 Q2-Q4 folds); runtime `24.469 s`; outer target reads before all seals `0`.
- Pooled incumbent/candidate: F1 `0.8604836038423319`; TP/FP/FN `12989/1146/3066`; precision `0.9189246551114255`; recall `0.8090314543755839`.
- Additions/true-positive additions/removals: `0/0/0`; delta F1 `0.0`; June-KST delta `0.0`.
- Fold F1 (incumbent = candidate): Q2 `0.7784135753749013`, Q3 `0.8970588235294118`, Q4 `0.9090245682315738`; every fold delta `0.0`.
- Cluster bootstrap: 2,000 replicates, 3,089 clusters, 90% CI `[0.0, 0.0]`, positive probability `0.0`.
- Long-event interior recall: `0.8107135718568859` for both anchor and candidate across 81 runs / 15,009 interior rows; offset recall `0.6477211796246649`; drift recall `0.6595061728395062`.
- Nominal and transport-adjusted points: `0.0 / 0.0`.
- Official/test/sample/submission/hidden/upload/CSV access or materialization: `0`.

## Integrity and independent QA

- Focused pytest: `6 passed`.
- Ruff: `All checks passed`.
- Lifecycle-aware post-terminal QA: `PASS`; independently recomputed pooled counters/F1, long-event interior recall, all seal hashes, nine unique model hashes, add-only/removal-zero invariant, outer isolation, and access-zero counters.
- Config SHA-256: `327eed88b9ff9b4e7514dd3ff1e6cff06c964369886f5e903c99b2bb8f8a804a0`
- Runner SHA-256: `f8edf02cefbffa75d28a94376d0842f351add0dfd75aba8e1c9664d5cd7035a2`
- Completion SHA-256: `fb99c58eea8147360c29901a755ddb64698c17ceb7832f844c3cad0eeb8c6e5c2`
- Lock SHA-256: `49f1837b2853ba738c86f4d76e26115b633c158ddb5df4494c2cea6e668ffd800`
- Result SHA-256: `e7d3ba9c3049f8101d35b6e3ed1f4a1f1455ddfba7aff9b10139e6091770b28b`

The namespace and artifacts are terminal and must not be reused, resumed, or retuned.
