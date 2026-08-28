# 주장-근거 원장

| ID | 주장 | 근거 | 신뢰도와 한계 |
|---|---|---|---|
| C1 | 현 챔피언 OOF와 공식 test 문맥의 상승파 비율은 각각 82.87%, 81.50%다 | `artifacts/p3_target_shift_retroaudit_20260828_v1/result.json` | 높음. 고정 threshold `hs_delta_12h>0.2`에 한정 |
| C2 | 파고 상태만으로 source/test를 구분한 AUC는 0.5624다 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | 높음. 고정 특징·fold의 판별력 |
| C3 | 파고+풍속 전체 AUC는 0.7261이고 가장 큰 이동은 풍속·돌풍 결측률이다 | `artifacts/p3_target_shift_retroaudit_20260828_v1/result.json` | 높음. 관측 특징에 한정 |
| C4 | 역사 OOF 풍속 완비는 126/181이며 G 66/67, I 30/46, S 30/68이다 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | 높음 |
| C5 | Hs² 후보는 공식에서 RMSE +0.001846m, 점수 -0.029302로 악화했다 | `reports/approved_parallel_execution_20260828_v9/p3_official_submission_receipt_20260828.json` | 높음. 단일 공식 probe |
| C6 | 같은 Hs² 후보는 세 재가중 로컬 표면에서 모두 개선으로 판정됐다 | v1/v2 `result.json` | 높음. 공식 방향 예측에는 실패 |
| C7 | IWCV는 covariate shift와 안정적 조건부분포 가정을 필요로 한다 | [Sugiyama 2007](https://jmlr.org/papers/v8/sugiyama07a.html) | 1차 학술 자료 |
| C8 | target support가 training support에 포함되어야 안정적 보정이 가능하다 | [Bickel 2009](https://www.jmlr.org/beta/papers/v10/bickel09a.html) | 1차 학술 자료 |
| C9 | MMD는 두 표본 분포 차이의 커널 검정이다 | [Gretton 2012](https://www.jmlr.org/beta/papers/v13/gretton12a.html) | 1차 학술 자료 |
| C10 | 직접 다중리드와 풍파 지연 창은 다음 residual pilot의 구조적 근거가 된다 | [Applied Sciences 2026](https://www.mdpi.com/2076-3417/16/15/7447); [Ocean Engineering 2022](https://www.sciencedirect.com/science/article/abs/pii/S0029801822001469) | 구조 근거. 현 데이터 성능 보장은 아님 |
| C11 | 풍속 완비 strict-OOF residual pilot의 4개 모델은 모두 챔피언보다 악화했다 | `artifacts/p3_atmosphere_complete_residual_pilot_20260828_v1/result.json` | 높음. 124사례, 3개 outer historical fold |
| C12 | 최선 Ridge wind-only도 +0.008173m이고 CI90은 0을 가로질렀다 | 같은 pilot result | 높음. 공식 분포의 성능을 직접 보장하지 않음 |
| C13 | 3 outer fold 중 2개는 nested selection이 residual scale 0을 선택했다 | 같은 pilot `selection_log` | 높음. 해당 fold의 내부 검증에서 보정 신호가 없었음 |
