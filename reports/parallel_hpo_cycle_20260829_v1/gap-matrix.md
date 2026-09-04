# Evidence gap matrix — 2026-08-29 parallel HPO cycle

| Claim family | 현재 근거 | 모순·대안 설명 | 신뢰도 | 남은 gap | 다음 조치 |
|---|---|---|---|---|---|
| P1 탐색 완결성 | 32/32 Sobol discovery, top-2 추가 seed 4, 36 fits, QA 28/28 | 더 넓은 topology/목적함수 공간은 탐색하지 않음 | 높음(봉인 범위) | MS-TCN 전체 모델 공간의 최대치 여부 | 현재 실험을 재튜닝하지 말고 독립 Group-DRO 가설로만 검토 |
| P1 승격 가능성 | 모든 월 ΔF1 양수, pooled `+0.000565637`, anchor removal 0 | 작은 양의 효과가 seed/period proxy에 한정될 수 있음 | 높음(Q2), 낮음(official transport) | Q3/Q4와 official 관계 | 고정 gate 미달이므로 confirmation/제출 금지; 신호만 기록 |
| P2 평균 개선 | pooled `-0.002041992 °C`, bootstrap q95 upper 음수 | fold별 개선 1/회귀 1/no-op 1, 한 inner 부적격 | 높음(local) | official score transport | close-family 종료; 다른 구조의 0-fit support audit만 허용 |
| P2 평가 독립성 | outer prediction pre-commitment와 NPZ hash, truth array 없음 | outer blocks는 historical exposure가 있어 완전한 새 test는 아님 | 중간–높음 | 반복 연구로 인한 연구자 자유도 | 새 가설은 사전 봉인하고 기존 outer를 selection에 재사용하지 않음 |
| P3 과학적 성능 | 없음: rung/ranking/gate 미완료 | 74 successful fits가 있어도 rung seal 전이라 부분 순위를 해석할 수 없음 | 높음(결론 없음) | valid grid의 실제 성능 | 새 ID·valid allowlist·모든 후보 one-tree smoke 후에만 실행 |
| P3 실패 원인 | terminal 문자열과 CatBoost v1.2.10 source 705–707 일치 | hardware/OOM 가설과 맞지 않음 | 매우 높음 | 다른 조합 제약 존재 가능성 | synthetic one-tree smoke로 전체 Cartesian 후보를 검증 |
| 공식 데이터 경계 | 세 실험 receipt/code path에서 official rows 0, CSV/upload 0 | P3는 OS-level file-access audit가 없음 | 높음(P1/P2), 중간–높음(P3) | 완전한 OS provenance | 다음 실행은 sandbox/file-open audit receipt 추가 검토 |
| Local→official 보정 | 이번 사이클은 official 점수를 읽거나 제출하지 않음 | 과거 경험상 local 작은 변화가 official에서 확대될 수 있음 | 측정 불가 | calibration slope/variance | 별도 공식 제출 원장과 독립 분석에서만 추정; 현재 결과에 소급 적용 금지 |

## 종료 기준

P1/P2/P3의 terminal 상태, P3 root cause, 금지 데이터 경계, 다음 실행 조건에 대해 독립
receipt 또는 1차 출처가 확보됐다. 남은 gap은 새 실험이나 공식 제출이 필요한 별도 연구
질문이므로 현재 사이클 보고서에서 추정으로 메우지 않는다.
