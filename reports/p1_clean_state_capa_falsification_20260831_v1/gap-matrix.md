# P1 clean-state CAPA gap matrix

| 판단 질문 | 고정 실측 근거 | 결론 | 다음 경계 |
|---|---:|---|---|
| fixed CAPA penalty가 실제 residual에 맞는가? | 203,574 additions, precision 0.044303, pooled ΔF1 -0.727385 | 아니오. 실제 residual에서 심각하게 과민하다. | 이 실험 안에서 penalty를 재조정하지 않는다. |
| 한 fold만의 우연한 실패인가? | Q2/Q3/Q4 ΔF1 = -0.641759/-0.755856/-0.789908 | 아니오. 세 fold가 같은 방향이다. | 동일 fixed likelihood family는 종료한다. |
| Q2 신규 I-ORS layer가 실패 원인인가? | Q2 5개 group·26,062행은 zero-signal abstain, Q3/Q4 unsupported 0인데도 더 큰 하락 | 주원인이 아니다. | future validation state를 끌어오는 fallback은 계속 금지한다. |
| incumbent를 훼손했는가? | incumbent-positive removals 0 | 아니오. protected union은 계약대로 동작했다. | 다음 연구도 incumbent union을 유지한다. |
| 불확실성상 반전 가능성이 있는가? | paired cluster bootstrap CI90 [-0.779897, -0.672243], positive probability 0 | 현재 고정 후보의 반전 근거가 없다. | 제출·공식 materialization 금지. |
| 구현/산출물 오류인가? | independent QA PASS, 9 pytest PASS, Ruff PASS, 모든 seal/hash 재검산 | 아니다. 과학적 NO_GO다. | 실패 원인을 calibration 문제로 기록한다. |
| 후속 연구는 무엇이어야 하는가? | 합성 PASS와 실제 과민의 괴리 | 이론 penalty가 실제 autocorrelation/heavy tail을 반영하지 못했다. | 별도 preregistration으로 prefix-only block-max empirical null, proposal budget, false-alarm ceiling을 먼저 고정한다. |
