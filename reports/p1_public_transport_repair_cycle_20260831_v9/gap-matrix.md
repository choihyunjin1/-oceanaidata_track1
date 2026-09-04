# P1 v5–v9 gap matrix

| Cycle | 구조적 질문 | 실제 결과 | 판정 | 남은 gap |
|---|---|---|---|---|
| v5 | G-ORS×L1 learned score가 high-precision add-only tail을 찾는가 | 12 fits, 3 candidates, additions 0 | NO PASS | calibration support 없음 |
| v6 | arbitrary precision 대신 F1 marginal gate가 support를 되살리는가 | Q2 inner source labels all-zero, fit 0 | TECHNICAL/IDENTIFIABILITY FAILURE | earliest block positive support 없음 |
| v7 | all-layer fitting과 isotonic calibration으로 L2 drift tail을 찾는가 | 24 fits, additions 0 | NO PASS | calibrated probability가 F1/2 미달 |
| v8r1 | 절대확률 대신 fixed top-rank/day budget이 tail을 찾는가 | 6 fits, inner block LCB 음수, additions 0 | NO PASS | ranking tail도 시간 안정성 없음 |
| v9 | target-free whole CAPA proposals를 hierarchical benefit selector로 거르는가 | 6 fits, Q2 proposal TP0/FP2046, additions 0 | NO_GO_BUDGET_SUPPORT | proposal source 자체의 transportable TP support 없음 |

## 사전 필터로 전환할 항목

| 필터 | 승격 기준 | v9 |
|---|---|---|
| 최초 train-prefix support | proposal additions TP > 0 | FAIL (TP 0) |
| outer 방향성 | Q3, Q4 ΔF1 각각 >= 0 | 무개입으로 0/0 |
| pooled uncertainty | day-block LCB >= 0.0005788103 | FAIL (0) |
| Public transport | raw expected points >= 0.331905690 | FAIL (0) |
| 안전성 | add-only, removals 0, daily cap <=0.5% | PASS, 그러나 효용 없음 |
