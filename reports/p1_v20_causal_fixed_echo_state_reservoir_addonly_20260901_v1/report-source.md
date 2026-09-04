# P1 v20 fixed echo-state reservoir — terminal report

Decision: `NO_GO_EXPLORATORY_ONLY`, exact anchor no-op. This family is closed with no spectral-bound, leak-rate, width, input-scale, threshold, or gate retuning.

Jaeger's echo-state report introduces a recurrent architecture whose internal weights remain fixed while only output weights are trained ([GMD Report 148, 2001](https://publica.fraunhofer.de/entities/publication/7d4a7eec-a22c-4df0-903d-93f9cd5aca02), DOI `10.24406/publica-fhg-291111`). v20 used one deterministic 16-state causal reservoir with normalized temperature and first difference, operator-norm bound 0.8, leak 0.2, and a linear readout. This source motivates the mechanism only and makes no P1 performance claim.

Repository audit found no P1 reservoir/echo-state implementation. Unlike v8 GRU and MS-TCN, no encoder weight, gate, convolution, or boundary head was trained. Unlike MiniRocket it maintains a nonlinear recurrent state, and unlike v10 recurrence it computes no distance, radius, lag-match, or run length. Evidence hashes were sealed before execution.

Focused pytest 4/4 and Ruff passed. Two real zero-operation preflights were byte-identical at SHA-256 `9ef2365fe2f658a3a84001076d1321cef436dacdbe057f86cd995e111d59f49a`; nonzero representation support was `0.8544275440` and all 16 feature variances passed. Pre-execution QA passed and the namespace was initially empty.

Exactly-once execution completed 9 fits in `39.375 s`. Every Q2–Q4 threshold failed the prospective station×layer×chronological-half transport veto; maximum inner Wilson LCBs were `0.008688 / 0.061078 / 0.090943`. Consequently no proposal reached the outer action surface.

- pooled anchor/candidate F1: `0.8604836038423319`; delta `0`; TP/FP/FN `12989/1146/3066`
- fold F1 Q2/Q3/Q4: `0.7784135753749013 / 0.8970588235294118 / 0.9090245682315738`; all deltas `0`
- additions / addition TP / removals: `0 / 0 / 0`
- paired block-bootstrap CI90: `[0,0]`, 2,000 replicates, 3,089 clusters
- long-event interior recall: `0.8107135718568859` over 15,009 rows; boundary recall `0.779835390946502` over 972 rows from 81 runs; both deltas `0`
- offset/drift recall: `0.6477211796246649 / 0.6595061728395062`
- nominal/transport-adjusted points: `0 / 0`
- result SHA-256: `7cb398bd87747c7e791694b96cd86add5da270cf01b7f4442a4fd367e98da695`
- config/runner SHA-256: `c344c73c7ddc486257b243c22945f5f877e8bb450019137463f247e8904b13ad / 23a0543140f9025e54fe36714acfbf6948562495eed987a11491b63187110196`

Lifecycle QA recomputed scores, action geometry, long-event receipts, hashes, nine unique readouts, and outer isolation: PASS. Only README/train were accessed; official/test/sample/submission/hidden reads, CSV materialization, and uploads were zero.

Next audit-only direction: a causal physics residual based on temperature–salinity–depth consistency, provided repository audit distinguishes it from matched-filter salinity gating, generic tabular models, and P2 TEOS families. No v20 result may set its formula or gate.
