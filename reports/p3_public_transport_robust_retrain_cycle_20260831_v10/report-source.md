# P3 Public-transport robust retrain v10

## 결론

- calibrated PASS: **0/3**
- CSV: **0개**, upload 0
- training outliers were downweighted from train-only residuals; no validation row was deleted.

| candidate | delta RMSE | episode CI90 upper | group CI90 upper | worst station-lead | calibrated pts | PASS |
|---|---:|---:|---:|---:|---:|---|
| P3_1_WINSOR_WEIGHTED_HUBER_BASE_BLEND | 0.043353 | 0.067963 | 0.070680 | 0.100281 | -0.321906 | False |
| P3_2_WINSOR_WEIGHTED_ABSOLUTE_HGB_BASE_BLEND | 0.003154 | 0.010639 | 0.007564 | 0.011068 | -0.321906 | False |
| P3_3_WINSOR_WEIGHTED_STRONG_RIDGE_BASE_BLEND | 0.016880 | 0.027591 | 0.028020 | 0.036803 | -0.321906 | False |
