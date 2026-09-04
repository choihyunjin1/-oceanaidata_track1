# P1 v11 causal wavelet-scattering terminal report

- Experiment: `p1_v11_causal_wavelet_scattering_addonly_20260901_v1`
- Surface: `EXPLORATORY_REUSED_SURFACE`
- Decision: `NO_GO_EXPLORATORY_ONLY`; no restart, threshold change, scale change, or result-driven retuning is authorized.
- Primary mechanism source: Bruna and Mallat, *Invariant Scattering Convolution Networks* (2012), https://arxiv.org/abs/1112.1120. v11 is a compact fixed causal Haar-modulus approximation, not a reproduction or performance claim.
- Semantic novelty: v6 explicitly listed scattering as an unused next P1 axis. v11 differs from v6 one-stage Haar/level/slope summaries by using seven first-order modulus scales, five second-order ordered modulus cascades, a cross-scale energy ratio, and 96-row energy persistence. It is also distinct from MiniRocket PPV kernels, soft-symbolic PAA transitions, learned MS-TCN, v7 signed path cross-moments, and v10 state recurrence. No P2/P3 result or hyperparameter was transferred.

## Execution and scientific outcome

- Two accepted real zero-operation preflights were byte-identical: `0c3d38381ced9d6cd1bd7b7892bf0f0731886892aeb47a80d2bfeda839aa570e`; artifact/lock remained absent. Feature nonzero share was `0.8226639165913486`, and all 14 feature variances exceeded the frozen support floor.
- Exactly 9 distinct fits (3 seeds x Q2-Q4); runtime `26.5 s`; outer target reads before every seal `0`.
- The frozen Wilson-90 precision-LCB gate rejected Q2 and Q4. Q3 chose the preregistered 0.999 inner quantile at threshold `0.9996789693832397` (inner count 116, precision `0.6982758621`, LCB `0.6243132080`).
- Pooled incumbent F1 `0.8604836038423319`, TP/FP/FN `12989/1146/3066`.
- Pooled candidate F1 `0.8598925546290498`, TP/FP/FN `13045/1241/3010`; delta F1 `-0.0005910492132821243`.
- Additions/true additions/precision/removals: `151/56/0.3708609271523179/0`. All additions occurred in Q3; Q2 and Q4 were exact no-ops.
- Fold delta F1: Q2 `0.0`, Q3 `-0.0019574298884204033`, Q4 `0.0`.
- 2,000-replicate, 3,089-cluster bootstrap: mean delta `-0.0006052069121425283`, CI90 `[-0.004248384352288553, 0.0035662254568002752]`, positive probability `0.384`.
- Long-event interior recall improved from `0.8107135718568859` to `0.814378039842761` (delta `+0.003664467985875164`, 81 runs / 15,009 interior rows), but the precision cost made row F1 negative. Offset recall remained `0.6477211796246649`; drift recall remained `0.6595061728395062`.
- Action cells: G-ORS/L1 58 additions (pooled slice delta `+0.035177389259862246`), I-ORS/L1 43 (`-0.009460042837929838`), S-ORS/L6 29 (`-0.009824628212889097`), S-ORS/L8 21 (`-0.013535767283667566`).
- Nominal / transport-adjusted expected points: `-0.015710319873044075 / -0.004713095961913222`.
- Official/test/sample/submission/hidden reads, CSV materialization, and uploads: `0`.

## Integrity and lifecycle QA

- Focused pytest: `6 passed`; Ruff: `All checks passed`.
- Independent post-terminal QA: `PASS`, recomputing pooled counts/F1, additions, long-event recall, add-only/removal-zero, all sealed NPZ hashes, nine unique model hashes, outer isolation, schema, access-zero counters, and result hashes.
- Config SHA-256: `d6e98c2c5e4855ec72ccbd5891341f58d87771c3e83d6621c1d7e5449bc153b9`
- Runner SHA-256: `70c363e28d32bf388da8b51ce201b8cb7b9244a8edf65de3e6b1262b103ceb86`
- Completion SHA-256: `5418fbae10d1bfee4ffb0d93caf84affe9f90177dfedfa550e511cd9d91613a4`
- Lock SHA-256: `899452af53366dc09ab444c541b5afeb9f8c7ee9a8acc4cb7cc340688dde1fcd`
- Result SHA-256: `cc6412b492ac6daa864071af9f64e802096fa99d5fd019fb4334f8a2b55bcfa8`

The namespace is terminal. Its Q3 action pattern is diagnostic only and must not be used to retune or select a follow-up.
