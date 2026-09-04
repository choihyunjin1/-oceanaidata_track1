# P2 v53 v52/v45c stability-score frontier audit

## 결론

상태: `NO_GO_NO_LOFO_FRONTIER_CANDIDATE_BEATS_V23_AND_SAFETY`. LOFO pooled delta RMSE `-0.049850290 C`, nominal `+0.625498` points, transport `+0.503816` points.

v23 대비 delta RMSE 차이는 `+0.002042177 C`, transport 점수 차이는 `-0.025624`다. fold-layer non-harm `7/9`, worst cell `+0.019484292 C`, 최종 gate `False`.

가중치는 각 held-out fold×layer의 결과를 보지 않고 다른 두 fold의 같은 layer에서만 고른 cross-fitted 진단이다. median deployment weight는 진단용일 뿐 새 제출 후보가 아니며, fresh deployment preflight 전 materialization하지 않는다. model fits=0; official/test/sample/baseline/query/hidden/CSV/upload=0.
