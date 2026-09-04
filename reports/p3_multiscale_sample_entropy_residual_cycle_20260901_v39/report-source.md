# P3 multiscale sample-entropy residual cycle v39

## 결론

- overall decision: **PASS_CANDIDATE_AVAILABLE**.
- v39 measures amplitude-sensitive template complexity across fixed coarse-grained scales; it is not v29 ordinal/recurrence geometry, v33 covariance, or v38 tail-hit directionality.
- Richman/Moorman and Costa et al. motivate the statistic only; the 182-case surface remains EXPLORATORY_ONLY.
- P3_1_MSE48_RIDGE512_ADD10: PASS_STABLE; RMSE 0.779107517m; delta -0.002084008m; raw +0.033075 points; transport-adjusted -0.016511; blocks 5/6; worst block +0.005417015m; worst lead +0.002666476m; worst station-lead +0.005133784m; worst reference-tail block +0.009135727m; episode CI90 [-0.0035318944594990463, -0.0005891236519340201]; block-station CI90 [-0.003760803739106777, -0.00011821191139656412].
- P3_2_MSE48_RIDGE2048_ADD10: PASS_STABLE; RMSE 0.779091883m; delta -0.002099642m; raw +0.033323 points; transport-adjusted -0.016263; blocks 5/6; worst block +0.006154750m; worst lead +0.002287896m; worst station-lead +0.003999830m; worst reference-tail block +0.009745360m; episode CI90 [-0.003495728738518844, -0.0006492789942909856]; block-station CI90 [-0.0037541150994415463, -0.00019429375469648391].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
