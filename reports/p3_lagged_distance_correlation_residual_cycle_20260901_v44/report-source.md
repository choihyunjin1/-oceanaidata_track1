# P3 lagged distance-correlation residual cycle v44

## 결론

- overall decision: **PASS_CANDIDATE_AVAILABLE**.
- v44 measures sealed nonlinear dependence between historical Hs and every other transformed channel; it reuses no v42/v43 output.
- Szekely, Rizzo and Bakirov (2007) motivates the mechanism only; the 182-case surface remains EXPLORATORY_ONLY.
- P3_1_DCOR55_RIDGE512_ADD10: PASS_STABLE; RMSE 0.778431121m; delta -0.002760403m; raw +0.043810 points; adjusted -0.005776; blocks 5/6; worst block +0.004171115m; lead +0.000703769m; station-lead +0.001729516m; tail +0.007311131m; episode CI90 [-0.004159969697271415, -0.0012580850254963081]; block-station CI90 [-0.004447423713424659, -0.0008963610276423166].
- P3_2_DCOR55_RIDGE2048_ADD10: PASS_STABLE; RMSE 0.779229988m; delta -0.001961537m; raw +0.031131 points; adjusted -0.018455; blocks 5/6; worst block +0.005591108m; lead +0.001227228m; station-lead +0.002190160m; tail +0.007310429m; episode CI90 [-0.003253651991182932, -0.0006417944581826151]; block-station CI90 [-0.003407519243717505, -0.00033419309398000633].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
