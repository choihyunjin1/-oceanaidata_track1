# p2_gaussian_copula_conditional_mean_20260830_v1

## 결론

`INVALID_TERMINAL_TECHNICAL_FAILURE`이다. 과학적 성능 판정이 아니며 같은 ID는 재실행하지 않는다.

- one-shot lock: consumed
- prediction files written: 0
- outer truth metrics computed: false
- official hidden/test/sample/submission rows read: 0
- CSV/upload: 0/0

## 원인

첫 outer fold `2024_sep_oct`의 historical alpha50 proxy는 8,779개 시각 중 8,715개가 3개 target layer를 모두 가지며, 64개 시각은 2개 layer만 가진다. v1 row mapper가 모든 query 시각을 완전 3-layer profile로 가정해 첫 누락 시각에서 `KeyError`로 종료됐다.

이는 모델 성능이나 shrinkage 결과가 아니라 support-contract 구현 오류다. 첫 fold의 예측 파일이 쓰이기 전 종료됐고 점수는 계산되지 않았다.

## 허용된 수리 범위

별도 ID v2에서 모델, empirical marginal, Kendall covariance, shrinkage 후보, outer folds, gate는 그대로 유지한다. 완전 profile은 동일 예측을 사용하고 불완전 profile만 기존 P2 fallback 원칙과 같이 residual correction 0으로 둔다. 결과 기반 튜닝과 v1 재실행은 금지한다.
