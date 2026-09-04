# P3 Kramers-Moyal residual cycle v42

## 결론

- overall decision: **PASS_CANDIDATE_AVAILABLE**.
- v42 estimates fixed-bin state-conditional increment moments; it contains no transfer entropy, threshold-crossing count, recurrent state, or prior-cycle prediction.
- Siegert et al. (1998) motivates the mechanism only; the 182-case surface remains EXPLORATORY_ONLY.
- P3_1_KM80_RIDGE512_ADD10: PASS_STABLE; RMSE 0.776335632m; delta -0.004855893m; raw +0.077067 points; adjusted +0.027481; blocks 5/6; worst block +0.004513486m; lead +0.001064341m; station-lead +0.002640994m; tail +0.013235567m; episode CI90 [-0.006939343329960862, -0.0026996321360695623]; block-station CI90 [-0.006745105610765872, -0.002614637920673629].
- P3_2_KM80_RIDGE2048_ADD10: PASS_STABLE; RMSE 0.777194343m; delta -0.003997182m; raw +0.063438 points; adjusted +0.013852; blocks 5/6; worst block +0.004881764m; lead +0.000855873m; station-lead +0.001711368m; tail +0.012384106m; episode CI90 [-0.0057194316550273185, -0.002197741796864943]; block-station CI90 [-0.005541988092888394, -0.0021584990834669685].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.

## Action geometry and immutable caveat

- Exact internal leader `P3_1_KM80_RIDGE512_ADD10`: action RMS/p99/max `0.025051076 / 0.062141002 / 0.129126373 m` over all `1,092` case-lead rows.
- Fixed adverse-slice geometry: worst block `+0.004513486 m`, worst lead `+0.001064341 m`, worst station-lead `+0.002640994 m`, and worst reference-tail block `+0.013235567 m`; all preregistered slice/tail gates pass.
- All stations improve (`G-ORS -0.006189949 m`, `I-ORS -0.003499465 m`, `S-ORS -0.004714685 m`). The comparatively large maximum single-row action is retained as a deployment-risk caveat, not used to alter the sealed result or invent a post-result ceiling.
