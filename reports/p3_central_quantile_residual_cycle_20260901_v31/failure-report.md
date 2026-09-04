# P3 v31 terminal technical failure

## 결론

- 상태: `INVALID_TERMINAL_TECHNICAL_FAILURE`.
- 동일 ID 재실행은 금지한다. attempt lock은 소비됐고 `result.json`은 존재하지 않는다.
- 12개 quantile fit과 candidate array 조립 뒤, outer metric을 계산하기 전에 scorer adapter가 `v28.SPECS.index(spec)`에서 foreign SPEC을 찾지 못해 종료됐다.
- 따라서 pooled RMSE, block/station/lead slice, CI, gate, 예상 점수는 전혀 산출·노출되지 않았다. 이는 과학적 `NO_GO`가 아니다.
- 공식 test/sample/submission/hidden, CSV materialization, upload 접근은 모두 0이다.

## Immutable evidence

- source config SHA-256: `a59ee3f77c8b4b0775f671bd8f975644bdcbb43b2891a59b853f67a04577ffe6`
- source runner SHA-256: `f1d48419caa840894e470c57c641b6e45750867adaa8388f5e790e7e8dfab302`
- consumed attempt lock SHA-256: `b3a78646ec156bbd1bd91517241ffb22db1fda4f80871a3991b8c202048c7fb4`
- source result existence: `false`
- exception: `ValueError: tuple.index(x): x not in tuple`
- failing boundary: `run_p3_cross_wavelet_phase_residual_cycle_20260901_v28.score`, module-global `SPECS.index(spec)`.

## Recovery boundary

Fresh ID `p3_central_quantile_residual_cycle_20260901_v31r1` may change only the scorer adapter registration. Quantiles, alpha, blend, features, folds, purge, tail gates, fit budget, and all inputs remain the original source config verbatim.
