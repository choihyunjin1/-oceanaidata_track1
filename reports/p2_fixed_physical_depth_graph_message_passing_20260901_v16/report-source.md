# P2 fixed physical-depth graph message passing v16

## 결론

상태 `EXPLORATORY_NO_GO_FIXED_DEPTH_GRAPH`; pooled ΔRMSE `-0.052161845℃`, canonical nominal `+0.654502`점, fixed-penalty `+0.532820`점. 모든 값은 노출된 historical surface의 탐색 지표이며 official 성능 주장이 아니다.

Spline/PAVA 후보는 기존 functional PCA/residual/projection과 중복이라 실행하지 않았다. v16은 fixed nominal-depth graph에서 한 번만 nonlinear message를 집계하며 learned adjacency, temporal axis, season router가 없다. Gilmer et al.은 message-passing 구조 동기만 제공한다.

official/hidden/CSV/upload=0.
