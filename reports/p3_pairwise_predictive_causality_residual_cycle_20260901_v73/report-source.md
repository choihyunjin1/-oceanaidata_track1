# P3 pairwise predictive-causality residual cycle v73

## 결론

- overall decision: **NO_GO_ALL_PAIRWISE_PREDICTIVE_CAUSALITY_CANDIDATES**.
- Fixed lag-2 incremental linear predictability is separate from discretized transfer information, static equilibrium error, and univariate Burg memory.
- Prior and official outputs were excluded; the repeatedly exposed 182-case surface is EXPLORATORY_ONLY.
- P3_1_GRANGER24_RIDGE512_ADD10: NO_GO; RMSE 0.780474559m; delta -0.000716965m; nominal score 24.214978; planning +0.011379; transport-adjusted -0.038207; blocks 5/6; worst block +0.007447104m; lead +0.002717941m; station-lead +0.003730173m; tail +0.008676076m; episode CI90 [-0.002061883384594837, 0.0006704207065220017]; block-station CI90 [-0.0022012530964434897, 0.000974388135745541].
- P3_2_GRANGER24_RIDGE2048_ADD10: NO_GO; RMSE 0.780053993m; delta -0.001137531m; nominal score 24.221652; planning +0.018053; transport-adjusted -0.031533; blocks 5/6; worst block +0.006765860m; lead +0.001917114m; station-lead +0.002705065m; tail +0.008806261m; episode CI90 [-0.0024097347297875106, 0.00021137484908702054]; block-station CI90 [-0.0025669702644258883, 0.0004480806032346973].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
