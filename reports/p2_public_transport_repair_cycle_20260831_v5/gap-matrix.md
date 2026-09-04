# Gap matrix

| 요구 | 증거 | 상태 |
|---|---|---|
| selection-matched nested time test | Sep train → Oct holdout | 충족 |
| 계절 전이 | Jul-Aug, Nov-Dec outer tests | 충족; 실패 확인 |
| layer worst-case | Oct layer 2/3/4 모두 별도 RMSE | 충족 |
| train-only robustification | 고정 Huber loss, 행 삭제 0 | 충족 |
| calibrated gain >=0.01 | -0.062379 / -0.069432 | 미충족 |
| 제출 materialization | PASS만 생성 | PASS 0이므로 CSV 0 |
| official/hidden/upload 경계 | 0 / 0 / 0 | 충족 |
