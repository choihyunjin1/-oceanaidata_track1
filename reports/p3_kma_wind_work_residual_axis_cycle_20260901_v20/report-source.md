# P3 v19 wind-work residual-axis v20

## 결론

- decision: **NO_GO_WIND_WORK_RESIDUAL_AXIS** (not preserved).
- candidate RMSE `0.777861102m`; vs uniform `-0.003330423m`; vs v19 `+0.000959902m`.
- expected raw score gain vs current official champion `+0.052856274`; transport-calibrated `+0.003270220`; expected score `24.256455274`.
- improved bimonth blocks vs v19 `1/6`; worst station×lead `+0.003756942m`.
- episode CI90 vs v19 `[0.00036824936361656293, 0.0015727594624965425]`; block×station CI90 `[0.00046607167653980874, 0.001600334512205304]`.
- official test/sample/submission/hidden/CSV/upload access: all 0.

## 해석 경계

This is a repeatedly exposed 182-case development surface. It is an internal comparative result, not an independent official-score guarantee. The wind-work residual axis was sealed before this candidate's outer scores were computed and no result-based retry occurred.
