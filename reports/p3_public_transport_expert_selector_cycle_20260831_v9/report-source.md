# P3 Public-transport physical expert selector v9

## 결론

- calibrated PASS: **0/3**
- CSV: **0개**, upload 0
- v8 residual regression을 폐기하고, prior-only regime expert selection과 exact no-op fallback만 평가했다.

| candidate | delta RMSE | changed rows | episode CI90 upper | group CI90 upper | worst station-lead | calibrated pts | PASS |
|---|---:|---:|---:|---:|---:|---:|---|
| P3_1_FINE_REGIME_PHYSICAL_EXPERT_SELECTOR | 0.001892 | 75 | 0.006943 | 0.009284 | 0.024707 | -0.321906 | False |
| P3_2_COARSE_REGIME_PHYSICAL_EXPERT_SELECTOR | 0.002844 | 89 | 0.008308 | 0.010869 | 0.024564 | -0.321906 | False |
| P3_3_ADD_ONLY_PHYSICAL_EXPERT_SELECTOR | 0.003181 | 28 | 0.005815 | 0.009186 | 0.024548 | -0.321906 | False |
