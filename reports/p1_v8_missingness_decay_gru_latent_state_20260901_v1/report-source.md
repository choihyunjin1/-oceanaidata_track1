# P1 v8 missingness-decay recurrent latent state

## Terminal decision

`p1_v8_missingness_decay_gru_latent_state_20260901_v1` is a valid exactly-once `NO_GO_EXPLORATORY_ONLY`. It is closed without retry, threshold relaxation, or retuning. The outer Q2-Q4 surface was already exploratory; no claim about an unseen test surface is made.

The architecture was a station-layer causal recurrent state using fold-local normalized temperature, salinity, and depth; natural-observation masks; elapsed time since each channel was observed; a fixed 24-hour decay toward the fold-local mean; and a one-layer width-16 GRU anomaly head. This was distinct from the prior typed HSMM, asynchronous peer-only Matérn state, MS-TCN, masked-TCN reconstruction, graph, and deterministic path families. The scientific basis for explicit mask and elapsed-time inputs is Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values*, Scientific Reports 2018, DOI `10.1038/s41598-018-24271-9`.

## Readiness and isolation

- Source allowlist: local `README.md` and `train.csv` only. README SHA-256 `cb658f...eafd`; train SHA-256 `20b656...5cd2`.
- Official test, hidden labels, sample submission, submission materialization, CSV output, and upload reads/writes: 0.
- Natural missingness receipt: 16 station-layer sequences; psal 16,725 missing/elapsed-positive rows; depth 1,130 missing/elapsed-positive rows; temp 0. Elapsed-feature variance was 0.0490483195 for psal and 0.0010531653 for depth, so the preregistered two-channel support gate passed.
- ns/cutoff, groupwise future-invariance, two-group feature reset, recurrent-state reset, finite-shape, add-only, and nine-fit tests: 9/9 PASS. Ruff: PASS.
- A first zero-operation command computed readiness but Windows cp949 could not print the non-ASCII semantic receipt. It created no artifact or lock and performed 0 fits. The stdout-only repair changed JSON console escaping. The two accepted real preflights after that repair were byte-identical at SHA-256 `7863b1a02a3807512d2400757327b81ae30e00b7258c79135df3ff211103d1a2`; artifact and lock were absent after both. Pre-execution QA passed.

## Frozen selection

Three seeds (`20260901`, `20260917`, `20260943`) were run for each of Q2, Q3, and Q4: 9 fits total, no sweep. Epochs=3, hidden width=16, AdamW learning rate=0.001, weight decay=0.0001, positive weight=12, and chunk length=8,192 were fixed before metrics. Threshold quantiles `0.995/0.9975/0.999`, maximum addition share `0.0025`, minimum 25 additions, and Wilson 90% precision LCB `0.55` were fixed. The 0.55 gate was not relaxed because the v6 metric-consistency audit could not establish a leakage-safe replacement.

All inner threshold candidates in all three folds had empirical precision 0, so no threshold passed the frozen precision-LCB gate. This is a direct lack-of-useful-proposal result, not an outer-driven choice.

## Historical result

- Runtime: 39.015 seconds on one NVIDIA GeForce RTX 5090; 9 unique model-state hashes.
- Pooled 421,032 rows: anchor and candidate F1 `0.8604836038423319`; TP/FP/FN `12,989/1,146/3,066`; precision `0.9189246551`; recall `0.8090314544`.
- Additions: 0; removed anchor positives: 0. Action geometry is therefore 0 rows, 0 runs, 0 stations, and 0 station-layer cells.
- Fold delta F1: Q2 0, Q3 0, Q4 0. Pooled delta F1: 0.
- Paired block bootstrap 90% CI: `[0, 0]` over 2,000 replicates and 3,089 clusters.
- Canonical nominal expected-point delta: 0. Transport-adjusted point delta: 0.
- Offset recall: pooled `0.6477211796`; Q2/Q3/Q4 `0.5361356932/0.7415094340/0.6872146119`.
- Long-event-interior diagnostic, not used for selection: positive 10-minute runs of at least 18 rows, excluding 6 boundary rows per side, gave 81 runs and 15,009 interior rows. Anchor and candidate recall were both `0.8107135719`, delta 0.

## Integrity and next axis

Independent lifecycle-safe QA recomputed the pooled counts/F1/delta, verified all three sealed NPZ hashes, 9 unique model hashes, exact add-only union, zero target reads before seals, immutable config/runner/completion/lock receipts, and official/CSV/upload access 0. Result SHA-256: `9e91d4c0c00a7290faaee886652ddefa9047140527843ea39fb18280aae92be2`.

The next plausible nonduplicate axis is the conditional-dependence change representation of Wu et al., AISTATS 2024, which detects changes in multivariate conditional dependence rather than only marginal level/variance changes. It must first pass a zero-fit semantic audit against prior CAPA/CPOP, e150 long-event CP rescue, and dependence-calibration families. If novel, it should use prefix-fitted conditional-dependence scores, the same add-only anchor and fixed gate, and no outer-driven threshold tuning.
