# P3 uniform selective robust fallback v21

## 결론

- overall decision: **NO_GO_ALL_SELECTIVE_FALLBACKS**.
- current official champion/reference: uniform KMA alpha=.425, RMSE 0.575233m, 24.203599 points.
- P3_1_EXTREME_DISAGREEMENT_PERSISTENCE_QUARTER_FALLBACK: NO_GO; RMSE 0.783648359m; delta +0.002456835m; raw -0.038991782 points; calibrated -0.088577836; blocks 1/6; changed 31 rows.
  - episode CI90 [4.597931892852405e-05, 0.005239433096632423]; block-station CI90 [-0.0003490850492671449, 0.005346637237362928]; worst station-lead +0.018011683m.
- P3_2_EXTREME_DISAGREEMENT_THREE_COMPONENT_MEDIAN: NO_GO; RMSE 0.782410603m; delta +0.001219078m; raw -0.019347672 points; calibrated -0.068933726; blocks 1/6; changed 21 rows.
  - episode CI90 [-0.0001107923008730971, 0.0030002995750510854]; block-station CI90 [-0.0002474915238830533, 0.003066253253048219]; worst station-lead +0.008164673m.

## Interpretation boundary

EXPLORATORY_ONLY on a repeatedly exposed 182-case historical surface; no Public transport guarantee. The uniform champion is the exact no-op default. Gate calibration reads training-only inputs, removes no rows, and uses no target labels. Official test/sample/submission/hidden/CSV/upload access is all zero.
