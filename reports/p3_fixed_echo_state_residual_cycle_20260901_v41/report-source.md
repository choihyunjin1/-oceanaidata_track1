# P3 fixed echo-state residual cycle v41

## 결론

- overall decision: **NO_GO_ALL_FIXED_ESN_CANDIDATES**.
- v41 uses one fixed nonlinear recurrent reservoir and fits only the residual readout; it reuses no v40 prediction or feature.
- Jaeger (2001) motivates the mechanism only; the 182-case surface remains EXPLORATORY_ONLY.
- P3_1_ESN96_RIDGE512_ADD10: NO_GO; RMSE 0.780735078m; delta -0.000456447m; raw +0.007244 points; transport-adjusted -0.042342; blocks 4/6; worst block +0.008935438m; worst lead +0.002517388m; station-lead +0.003895474m; tail +0.011106274m; episode CI90 [-0.0022937082090511707, 0.001408005475759611]; block-station CI90 [-0.002493382868785371, 0.0018446607245843182].
- P3_2_ESN96_RIDGE2048_ADD10: NO_GO; RMSE 0.780299014m; delta -0.000892511m; raw +0.014165 points; transport-adjusted -0.035421; blocks 5/6; worst block +0.007749228m; worst lead +0.002029604m; station-lead +0.002937638m; tail +0.009841530m; episode CI90 [-0.0023477404288648864, 0.0006354469284272693]; block-station CI90 [-0.0025772477484036827, 0.0009636320800009273].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
