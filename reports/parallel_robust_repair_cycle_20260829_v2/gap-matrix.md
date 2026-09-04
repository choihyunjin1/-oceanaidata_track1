# Gap matrix

| 문제 | 목표 | 실제 결과 | Gate | 상태 | 남은 결함 | 다음 조치 |
|---|---|---|---|---|---|---|
| P1 | worst-group BCE로 월·정점 강건성 개선 | pooled ΔF1 -0.01348, 최소 월 -0.02355, station share 0.802817, anchor removal 0 | FAIL | `NO_GO_Q2` | 목적함수가 precision 손실과 S-ORS 편중을 해소하지 못함 | 동일 계열 종료 |
| P2 | train-only copula 지지집합 확인 | complete timestamps 47,216, 최소 unique 2,353, duplicate/nonfinite 0 | TRAIN SUPPORT PASS | `QUERY_AUDIT_NOT_AUTHORIZED` | 공식 query support 및 복원 성능 미검증 | 별도 승인된 신규 실험에서만 진행 |
| P3 | valid-only CatBoost HPO 후 3-window confirmation | selection 138 fits, ΔRMSE -0.02286 m, gate PASS; 첫 confirmation fit 후 KeyError | selection PASS / confirmation INVALID | `TERMINAL_TECHNICAL_FAILURE` | canonical router projection과 confirmation 입력 schema 불일치 | 재실행 없이 결함 문서화; 신규 id에서 contract test 후 재등록 |

## P3 failure boundary

| 단계 | 완료 여부 | 증거 |
|---|---:|---|
| 37-fit synthetic smoke | 완료 | static preflight |
| 138-fit selection | 완료 | selection aggregate rung 74+40+24 |
| selection gate | PASS | challenger_21, ΔRMSE -0.0228625 m |
| confirmation blind fits | 1/3 후 중단 | `_fit_predict` 반환 뒤 router column selection에서 KeyError |
| blind prediction seal | 미생성 | artifact 없음 |
| truth metric / promotion | 미실행 | result 없음 |
| full refit / submission | 미실행 | CSV/upload 0 |
