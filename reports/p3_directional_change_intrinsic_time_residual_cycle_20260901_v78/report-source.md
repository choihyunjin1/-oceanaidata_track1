# P3 directional-change intrinsic-time residual cycle v78

## 결론

- overall decision: **PASS_CANDIDATE_AVAILABLE**.
- v78 encodes alternating reversal confirmations, overshoots and durations in past-only intrinsic event time. It uses no prior prediction or official feedback.
- Finance sources motivate the event operator only; they are not ocean performance evidence. The repeatedly exposed 182-case surface is EXPLORATORY_ONLY.
- P3_1_DIRCHANGE144_RIDGE512_ADD10: PASS_STABLE; RMSE 0.775931010m; delta -0.005260515m; nominal score 24.287087; planning +0.083488; transport-adjusted +0.033902; blocks 5/6; worst block +0.000972003m; lead +0.000140173m; station-lead +0.001584017m; tail +0.011003359m; episode CI90 [-0.00706326300939174, -0.003340290875620866]; block-station CI90 [-0.006892093426343587, -0.003491804279259558].
- P3_2_DIRCHANGE144_RIDGE2048_ADD10: PASS_STABLE; RMSE 0.777642607m; delta -0.003548918m; nominal score 24.259923; planning +0.056324; transport-adjusted +0.006738; blocks 5/6; worst block +0.003291991m; lead +0.000463295m; station-lead +0.000862298m; tail +0.010214590m; episode CI90 [-0.004913233265501038, -0.002066205958745154]; block-station CI90 [-0.004953460896833711, -0.00195875269355833].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
