# P2 layer-month V-REx v19

## 결론

상태 `EXPLORATORY_NO_GO_LAYER_MONTH_RISK_VARIANCE_REX`; pooled ΔRMSE `-0.047768059℃`, canonical nominal `+0.599371`점, fixed-penalty `+0.477689`점. 반복 노출 historical surface의 탐색 증거이며 official 성능 주장이 아니다.

v13 architecture/input/prefix/purge/seeds/epochs/blend/action cap은 고정했고, 유일한 과학 변화는 layer×month batch risk의 population variance에 고정 10.0을 곱한 V-REx 목적과 고정 WD 0.001이다. sweep/router/ensemble은 없다.

v13 대비 pooled ΔRMSE 차이 `+0.003086516℃`, v18 대비 `+0.002971810℃`다. 비교는 사후 선택이나 ensemble에 사용하지 않고 후보 원장을 갱신하는 데만 쓴다.

official/hidden/CSV/upload=0.
