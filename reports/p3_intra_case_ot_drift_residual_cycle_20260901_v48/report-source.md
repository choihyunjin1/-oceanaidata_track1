# P3 intra-case optimal-transport drift residual cycle v48

## 결론

- overall decision: **NO_GO_ALL_INTRA_CASE_OT_DRIFT_CANDIDATES**.
- v48 measures fixed early-to-late distribution-mass transport within each historical case and reuses no v42-v47 output.
- The repository's earlier Wasserstein use was validation-only; v48 is the first sealed P3 predictor representation. Peyre and Cuturi (2019) motivates the mechanism only; the 182-case surface remains EXPLORATORY_ONLY.
- P3_1_OTDRIFT48_RIDGE512_ADD10: NO_GO; RMSE 0.780379755m; delta -0.000811770m; raw +0.012883 points; adjusted -0.036703; blocks 3/6; worst block +0.007442354m; lead +0.001747955m; station-lead +0.003237597m; tail +0.016708407m; episode CI90 [-0.0026409673371811603, 0.0010495536643913954]; block-station CI90 [-0.0022046832904504985, 0.0007721931685993172].
- P3_2_OTDRIFT48_RIDGE2048_ADD10: NO_GO; RMSE 0.779966689m; delta -0.001224835m; raw +0.019439 points; adjusted -0.030147; blocks 5/6; worst block +0.006899887m; lead +0.001316632m; station-lead +0.002395533m; tail +0.011708026m; episode CI90 [-0.002688984652851778, 0.0002708396159676507]; block-station CI90 [-0.0025176038224143603, 0.0002517947674682666].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
