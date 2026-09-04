# P1 v10 duplicate structural preflight

## 결론

`NO_GO_DUPLICATE_STRUCTURAL_PREFLIGHT`. 동일한 20개 station-layer-day 공변량과 clipped density ratio를 사용한 sealed 실험이 이미 존재하며, daily/row/min-group ESS가 각각 `0.2196/0.2190/0.15093`으로 고정 gate `0.25/0.25/0.20`을 모두 실패했다. 관측 best ΔF1 `0.004186`도 SMOOTH_LEARNED_PROFILE 수송 기준 약 `0.004954` 및 CI90 low `-0.009429`를 넘지 못했다. ESS나 clip을 결과 기반으로 완화하지 않고 중복 fit을 생략했다. 이 preflight에서 공식 covariate, hidden truth, submission values, upload는 모두 0이다.
