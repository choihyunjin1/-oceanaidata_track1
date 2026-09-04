# P3 v8 gap matrix

| Gap | Evidence before v8 | v8 treatment | Result |
|---|---|---|---|
| Historical-to-Public reversal | v5 internal gain became Public RMSE +0.015723m worse | Subtract worst observed 0.321905690-point transport residual | All candidates remained far below +0.01 calibrated points |
| Official-like event mixture | Public cases are concentrated in high-wave rising regimes | Freeze 1.5≤Hs<2.2 and 12h rise>0.2 cohort; abstain outside its robust support | 157 independent cases, 942 rows evaluated |
| Temporal leakage | Flexible residual stacks can learn the held-out fold | First window exact no-op; later windows use completed prior OOF only | 6 historical fits, no current-fold labels |
| Local subgroup harm | Pooled averages can hide station/lead reversals | station×lead worst regression ≤0.01m hard gate | Worst regression was +0.052365m, +0.028732m, +0.026098m |
| Sampling uncertainty | 157 cases and 9 station×window groups are small | Both episode and group bootstrap CI90 upper must be below zero | No candidate cleared either CI gate |
| Submission-slot risk | One P3 slot remained and previous informative candidate was harmful | PASS-only materialization | 0 CSV, 0 official rows, 0 upload |
