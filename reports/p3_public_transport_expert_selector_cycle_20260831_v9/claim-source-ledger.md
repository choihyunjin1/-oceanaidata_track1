# P3 v9 claim-source ledger

| Claim | Source | v9 use | Limitation |
|---|---|---|---|
| Deployment covariate shift can invalidate ordinary validation. | Sugiyama et al., JMLR 2007, https://jmlr.org/papers/v8/sugiyama07a.html | Evaluate only the frozen official-like high/rising historical cohort. | No formal density-ratio estimate. |
| Selective prediction can hide subgroup harm. | Shah et al., ICML 2022, https://proceedings.mlr.press/v162/shah22a.html | Exact no-op fallback plus station×lead hard guard. | Operational groups only. |
| Worst-group behavior must be audited separately from pooled error. | Sagawa et al., ICLR 2020, https://arxiv.org/abs/1911.08731 | station×fold group bootstrap. | Not group-DRO training. |
| P3 Public transport residual was -0.321905690 points in the worst observed trial. | `reports/public_transport_calibration_20260831_v1/calibration.json` | Direct hard penalty. | Empirical guardrail, not a confidence interval. |
