# P1 v12 causal kernel-MMD terminal report

- Experiment: `p1_v12_causal_kernel_mmd_shift_addonly_20260901_v1`; surface `EXPLORATORY_REUSED_SURFACE`; terminal decision `NO_GO_EXPLORATORY_ONLY`.
- Mechanism source: Gretton et al., *A Kernel Two-Sample Test*, JMLR 2012, https://www.jmlr.org/papers/v13/gretton12a.html. v12 is a fixed RBF-landmark mean-embedding approximation, not the paper's test calibration or a performance claim.
- Repository semantic audit found no executed P1 MMD/kernel mean-embedding implementation. v12 is distinct from CAPA penalized mean/variance segment likelihood, v9 cross-layer covariance/precision change, dependence-null calibration, and v11 wavelet energy. No P2/P3 result or hyperparameter was used.

Two real zero-operation preflights were byte-identical (`3dca278f2c7f0f38d93a8c60f84ac69fc542439b6bb466aa76e961f5548fcfe0`), with no artifact/lock; feature nonzero share was `0.8544275440127925` and all nine feature variances passed. Focused pytest `5 passed`, Ruff PASS, and pre-execution QA PASS preceded exactly one execution.

The run used exactly 9 distinct fits in `26.078 s`, with outer target reads before seals `0`. Q2 chose inner threshold `0.9982746839523315` (precision/LCB `0.9298245614/0.8525873358`), Q3 chose `0.9242852926254272` (`1.0/0.9772079435`), and Q4 abstained. Those apparently strong inner tails transported as `447` outer additions with `0` true positives: 332 at I-ORS/L1 in Q2 and 115 at S-ORS/L7 in Q3. This is a direct transport falsification; window, kernel bandwidth, gate, or threshold must not be retuned on these results.

- Pooled F1: `0.8604836038423319` to `0.84792897476907`, delta `-0.01255462907326188`.
- TP/FP/FN: `12989/1146/3066` to `12989/1593/3066`; removals `0`.
- Fold delta: Q2 `-0.024687935329047228`, Q3 `-0.008634951427628934`, Q4 `0.0`.
- Bootstrap (2,000 replicates, 3,089 clusters): mean `-0.012897536532714004`, CI90 `[-0.022855831906705766, -0.0050989599351835115]`, positive probability `0.0`.
- Long-event interior recall stayed `0.8107135718568859` (81 runs / 15,009 rows); offset/drift recall stayed `0.6477211796246649/0.6595061728395062`.
- Nominal / transport-adjusted points: `-0.3337069641512511 / -0.10011208924537533`.
- Official/test/sample/submission/hidden reads, CSV, uploads: `0`.

Post-terminal lifecycle QA independently recomputed all counts/F1, additions, long-event recall, add-only/removal-zero, seal hashes, nine unique model hashes, outer isolation, schema, and access-zero counters: `PASS`. Config/runner/completion/lock hashes are `b54090ed03e1ced6116eacba1ca4fcf864280c646962d092eddccce6a10fad45`, `5ca89224ba3c5163263c0a00f50b0d121877e8700413316201f91ccc175995b3`, `b354abdd0f07f582e60f9d0dd69ae63babca7cb4ba885a23e171cc1fb728963f`, and `9ca77b9ef1289e3b8cbdc9f3d710088c83a0e2a84c658f096215dbbb902c7e2a`. Result SHA-256 is `b37c556c686bded7aec346c959990f5cc7232b64ed37483705d3155c6b140c75`.
