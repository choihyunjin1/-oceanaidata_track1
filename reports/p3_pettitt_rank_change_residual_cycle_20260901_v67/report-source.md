# P3 Pettitt rank-change residual cycle v67

## 결론

- overall decision: **NO_GO_ALL_PETTITT_CANDIDATES**.
- v67 scans all internal splits with a rank cumulative statistic; it is not BOCPD, fixed early/late OT, fixed-level crossing, or adaptive record timing.
- Prior outputs and official feedback are excluded; the 182-case surface is EXPLORATORY_ONLY.
- P3_1_PETTITT64_RIDGE512_ADD10: NO_GO; RMSE 0.780650287m; delta -0.000541238m; nominal score 24.212189; raw/planning +0.008590 points; transport-adjusted -0.040996; blocks 5/6; worst block +0.006192234m; lead +0.001787148m; station-lead +0.004065593m; tail +0.007884083m; episode CI90 [-0.0021807685134886566, 0.0011856836930334465]; block-station CI90 [-0.0022029142991147168, 0.001302219616084322].
- P3_2_PETTITT64_RIDGE2048_ADD10: NO_GO; RMSE 0.780215113m; delta -0.000976412m; nominal score 24.219095; raw/planning +0.015496 points; transport-adjusted -0.034090; blocks 5/6; worst block +0.006696172m; lead +0.001761721m; station-lead +0.003102361m; tail +0.009097286m; episode CI90 [-0.002417614940685403, 0.0005339725488935521]; block-station CI90 [-0.0025231613324551093, 0.0007525926849852597].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
