# P2 causal multiscale domain-stat DeepSets v14

## 결론

상태 `EXPLORATORY_NO_GO_CAUSAL_MULTISCALE_DOMAIN_STATS`; pooled ΔRMSE `-0.039812071℃`, canonical nominal `+0.499543`점, fixed-penalty `+0.377861`점. 모든 값은 노출된 historical surface의 탐색 지표이며 official 성능 주장이 아니다.

Wild-Time은 시간창을 domain으로 평가할 필요를 보이고, EvoS는 과거 domain 통계의 다중시간창 진화를 동기로 제공한다. v14는 이들의 학습법을 복제하지 않고 public-layer-only shifted EWM 통계를 1/7/30일 고정 반감기로 사용한다. v13 결과 기반 router나 gate 완화는 없다.

official/hidden/CSV/upload=0.
