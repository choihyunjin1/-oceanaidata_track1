# P2 — active task contract

Read with [organizer policy](00_ORGANIZER_DATA_POLICY.md) and the distributed P2 README.
Historical recipes and former gates are [archived](docs/archive/instructions_20260905/01_P2_MUST_READ_FIRST.md); they are not current instructions.

- Task: restore requested middle-layer temperature. Primary metric is RMSE in absolute degrees Celsius: sqrt(total SSE / scored row count), not mean fold RMSE.
- Source: P2_DATA_DIR; immutable organizer files. Official answer columns are station,layer,time,temp and must match supplied query keys/order.
- Never use hidden target-layer temperature/salinity or non-distributed observations. During pseudo-outages, mask target temperature AND salinity before computing interpolation, lag, rolling or derived features.
- Preserve stations, physical depths, seasons and contiguous outages in validation. Layer indices need not represent a constant physical depth.
- Fit residual baselines, scalers, imputers, augmentation/weights and model selection within the training fold. Maintain source-row weighting when augmentation duplicates an example.
- The deployment-relevant September–October primary assessment must not be replaced by a better-looking pooled summer/winter proxy. Report both, with exact support and units.
- Current work: [plan v2](docs/SCORE_IMPROVEMENT_PLAN_20260905_V2.md), fixed 2×2 objective/weighting comparison then the predeclared tree alternative.
- Historical bin17/Public-derived coefficients and the 0.424019°C file are not currently eligible baselines under the September 2 score-use rule. Renaming or serializing those coefficients does not clean their ancestry.
- Old audit locks and terminal results remain unchanged; no automatic reruns. New experiments have their own contracts.
- Read [handoff](AI_HANDOFF.md) before official inputs, candidate materialization or final packaging.
