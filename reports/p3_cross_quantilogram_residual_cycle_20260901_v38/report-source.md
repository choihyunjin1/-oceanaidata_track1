# P3 directional cross-quantilogram residual cycle v38

## 결론

- overall decision: **NO_GO_ALL_CROSS_QUANTILOGRAM_CANDIDATES**.
- v38 measures fixed directional upper/lower quantile-hit correlations, not v30 conditional-bin transfer entropy and not any prior-cycle prediction.
- Han et al. (2016) motivates the statistic only; the 182-case surface remains EXPLORATORY_ONLY.
- P3_1_XQ72_RIDGE512_ADD10: NO_GO; RMSE 0.780119300m; delta -0.001072225m; raw +0.017017 points; transport-adjusted -0.032569; blocks 4/6; worst block +0.005603045m; worst lead +0.001820994m; worst station-lead +0.003712580m; worst reference-tail block +0.010788999m; episode CI90 [-0.0026072713944200188, 0.0005327344957011926]; block-station CI90 [-0.0028736930567358652, 0.0009986166967311475].
- P3_2_XQ72_RIDGE2048_ADD10: NO_GO; RMSE 0.779927477m; delta -0.001264047m; raw +0.020061 points; transport-adjusted -0.029525; blocks 5/6; worst block +0.006329714m; worst lead +0.001623817m; worst station-lead +0.002869600m; worst reference-tail block +0.008843310m; episode CI90 [-0.0026485804538055936, 0.00021239839660663974]; block-station CI90 [-0.002799569158126014, 0.0005170257489150789].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
