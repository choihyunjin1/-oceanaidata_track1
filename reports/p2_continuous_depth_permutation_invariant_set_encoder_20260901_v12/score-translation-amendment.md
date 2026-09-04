# P2 v12 score translation amendment

## 결론

등록 gate는 PASS지만 transport safety는 NOT_READY다. pooled ΔRMSE `-0.042993523℃`의 canonical planning 환산은 명목 `+0.539463`점, 고정 penalty 후 `+0.417780`점이다.

기존 result의 raw/transport 필드는 Sep-Oct day-bootstrap CI90 상단 환산인 legacy engine field이며 pooled-delta planning 값으로 사용하지 않는다. Nov-Dec 회귀를 근거로 posthoc 라우팅하거나 gate를 바꾸지 않았다. result.json은 수정하지 않았다.
