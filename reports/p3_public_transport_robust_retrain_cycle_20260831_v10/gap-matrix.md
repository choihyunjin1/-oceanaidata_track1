# P3 v10 gap matrix

| gap | v10 evidence | disposition |
|---|---|---|
| Pooled historical improvement | All three candidates worsen RMSE. | FAIL |
| Episode bootstrap stability | Every 90% upper bound is positive. | FAIL |
| Station x lead safety | Worst deltas exceed +0.01 m. | FAIL |
| Public calibrated expected value | Conservative calibrated delta is -0.321906 points. | FAIL |
| Outlier handling integrity | Train-only weights; validation rows deleted = 0. | PASS |
| Official/hidden isolation | Official rows, hidden rows, CSVs, uploads = 0. | PASS |
