# P3 Allan/Hadamard scale-spectrum residual cycle v64

## 결론

- overall decision: **PASS_CANDIDATE_AVAILABLE**.
- Fixed Allan/Hadamard profiles encode scale-dependent block-average stability and drift suppression, not jump mass, volatility ACF, or a frequency decomposition.
- Prior outputs and official v42 feedback are excluded; the 182-case surface is EXPLORATORY_ONLY.
- P3_1_AHVAR64_RIDGE512_ADD10: PASS_STABLE; RMSE 0.775927657m; delta -0.005263867m; raw +0.083541 points; adjusted +0.033955; blocks 6/6; worst block -0.000170230m; lead +0.003024585m; station-lead +0.004527607m; tail +0.010868870m; episode CI90 [-0.007554589068968715, -0.003076984395626414]; block-station CI90 [-0.0072620617368806765, -0.003121796949381911].
- P3_2_AHVAR64_RIDGE2048_ADD10: PASS_STABLE; RMSE 0.777318594m; delta -0.003872930m; raw +0.061466 points; adjusted +0.011880; blocks 5/6; worst block +0.002133937m; lead +0.001985862m; station-lead +0.002797567m; tail +0.010027922m; episode CI90 [-0.0055080874580532456, -0.0021960127751135205]; block-station CI90 [-0.005377796368800053, -0.0021922930191951625].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
