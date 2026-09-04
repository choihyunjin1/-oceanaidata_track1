# P3 uncertainty-aware advantage router cycle v6b

## 결론

- 엄격 내부 PASS: **0/3**
- 제출 CSV: **0개**, hidden truth 0행, upload 0
- 공통 학습: **252 fits**; outer/inner 모두 time-blocked + 78h purge
- block/worst-slice는 진단 전용이며 PASS에는 validity, pooled RMSE, episode-bootstrap CI90만 사용했다.

## 후보

| candidate | delta RMSE(m) | base cases | CI90 low | CI90 high | expected points central | PASS |
|---|---:|---:|---:|---:|---:|---|
| P3_1_ADVANTAGE_UCB_Q50 | -0.002483 | 42 | -0.005624 | +0.000737 | 24.243012 | FAIL |
| P3_2_ADVANTAGE_UCB_Q65 | -0.001123 | 13 | -0.003748 | +0.001461 | 24.221426 | FAIL |
| P3_3_ADVANTAGE_UCB_Q80 | +0.000000 | 0 | +0.000000 | +0.000000 | 24.203599 | FAIL |

## 방법과 제한

- target은 case별 `base SSE - frozen champion SSE` 연속 advantage다.
- episode bootstrap bagged Ridge의 median과 MAD, nested blocked-OOF residual quantile로 one-sided UCB를 만든다.
- UCB<0인 확실한 champion 열위 case만 base로 전환하고, 나머지는 frozen champion 그대로 둔다.
- 50/65/80% 정책은 결과 전에 봉인됐고 결과 기반 threshold 수정·재시도는 없었다.
- 점수 환산은 0.575233m/24.203599와 -15.870739 points/m의 조건부 선형 계획값이며 gate에 쓰지 않았다.
- 내부→공식 분포 이동 때문에 예상 점수의 방향과 크기는 보장되지 않는다.
