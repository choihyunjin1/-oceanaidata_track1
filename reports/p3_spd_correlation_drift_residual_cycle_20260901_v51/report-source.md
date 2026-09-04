# P3 SPD correlation-drift residual cycle v51

## 결론

- overall decision: **NO_GO_ALL_SPD_CORRELATION_DRIFT_CANDIDATES**.
- v51 measures fixed early-to-late multichannel dependence-geometry drift; it imports no P2 output and reuses no v42-v50 output.
- Arsigny et al. (2006) motivates the SPD geometry only; the 182-case surface remains EXPLORATORY_ONLY.
- P3_1_SPD24_RIDGE512_ADD10: NO_GO; RMSE 0.779738652m; delta -0.001452873m; raw +0.023058 points; adjusted -0.026528; blocks 5/6; worst block +0.008744453m; lead +0.002200883m; station-lead +0.003931851m; tail +0.011399434m; episode CI90 [-0.002770955082810872, -9.158299160278983e-05]; block-station CI90 [-0.0028018555375777963, 0.0002615093021080599].
- P3_2_SPD24_RIDGE2048_ADD10: NO_GO; RMSE 0.779741801m; delta -0.001449724m; raw +0.023008 points; adjusted -0.026578; blocks 5/6; worst block +0.007305933m; lead +0.001656677m; station-lead +0.002724343m; tail +0.009957405m; episode CI90 [-0.0027095855480996577, -0.0001138168195335449]; block-station CI90 [-0.0028476827950041305, 0.0001632648124160199].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
