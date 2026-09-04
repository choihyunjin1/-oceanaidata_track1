# P3 full horizontal-visibility graph residual cycle v37

## 결론

- overall decision: **PASS_CANDIDATE_AVAILABLE**.
- v37 uses full P3 visibility graphs, unlike P1's endpoint-only detector, v29 metric recurrence, or v36 distance filtration. No earlier prediction or feature is reused.
- Luque et al. (2009) motivates the graph mechanism only; the 182-case surface remains EXPLORATORY_ONLY.
- P3_1_HVG64_RIDGE512_ADD10: NO_GO; RMSE 0.779450966m; delta -0.001740559m; raw +0.027624 points; transport-adjusted -0.021962; blocks 4/6; worst block +0.005524220m; worst lead +0.001391835m; worst station-lead +0.003898217m; worst reference-tail block +0.009850694m; episode CI90 [-0.003247884792592792, -0.00016108723211734617]; block-station CI90 [-0.0033103556400647825, 7.823202189157814e-05].
- P3_2_HVG64_RIDGE2048_ADD10: PASS_STABLE; RMSE 0.779424233m; delta -0.001767292m; raw +0.028048 points; transport-adjusted -0.021538; blocks 5/6; worst block +0.006108872m; worst lead +0.001754255m; worst station-lead +0.003445626m; worst reference-tail block +0.008303125m; episode CI90 [-0.0031399888096453497, -0.0003233733271234214]; block-station CI90 [-0.003274344458700523, -1.7059383346090237e-05].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
