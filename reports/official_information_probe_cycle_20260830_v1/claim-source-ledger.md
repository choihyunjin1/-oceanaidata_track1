# 공식 정보 probe claim-source ledger

| Claim | 직접 근거 | 범위·한계 |
|---|---|---|
| 오늘 문제별 잔여량은 P1 3, P2 2, P3 2였고 제출 후 모두 0이다. | 공식 P1/P2/P3 페이지의 `오늘 남은 제출`; 브라우저 action-time snapshot | 2026-08-30 KST 일일 quota에만 해당 |
| 팀·문제·일 3회 제한, 모델 최종 제출 후 답안 추가 업로드 불가 | 공식 참가자 전용 `[제출 안내]` 공지 | 모델 제출 기한은 공지상 2026-09-07 |
| P1 G 제거/ S 제거/ G+S 제거 결과는 0.829029 / 0.833548 / 0.829029 | 공식 제출관리 2026-08-30 22:54–22:55 KST | 6자리 Public 표시; row truth 미접근 |
| P1의 표시상 e150 효과는 G에 있고 S 효과·G×S interaction은 0 | 위 세 score와 champion 0.833548의 사전 고정 finite difference | 표시 반올림 이내 효과는 식별 불가; Private 미확인 |
| P2 bin17-only가 0.430194로 새 최고, bin18-only는 0.431267 | 공식 제출관리 및 리더보드 | 6자리 RMSE; Private 미확인 |
| P2 bin17이 이득을 만들고 bin18은 미세 역기여 | `p2-official-result.json`; alpha50 0.431252, full 0.430209, 두 분해 제출; disjoint-support composition identity | 직접 MSE 증분은 -0.000911409868/+0.000012937785; 표시 반올림 residual -3.174e-8 포함 |
| P3 S/I 보정 제거는 각각 0.579102/0.578951로 악화 | 공식 제출관리 2026-08-30 22:56 KST | station별 row truth 미접근 |
| P3 MSE 기여율 G/I/S = 12.75/42.75/44.50%이며 순서는 반올림 범위에서도 S>I>G | `p3-official-result.json`; alpha=0 0.583892, full .425 0.575233, S/I ablation RMSE의 제곱 분해 및 16-corner 계산 | 표시 RMSE 기반 근사; correction support가 station별 disjoint라는 동결 recipe에 한정 |
| 최종 리더보드는 5위, 총점 81.048404 | 공식 리더보드 2026-08-30 22:57 KST | Public leaderboard snapshot |
| 후보 7개는 첫 새 점수 전에 동시 동결·QA됐다. | P1/P2/P3 manifest와 QA receipt; `pre-submit-independent-qa.json` | 로컬에 기록된 과거 vector/hash 범위에서 비중복 |
| 반복 leaderboard query는 독립 검증이 아니다. | Blum & Hardt, *The Ladder*, ICML 2015, https://proceedings.mlr.press/v37/blum15.html; Cawley & Talbot, JMLR 2010, https://www.jmlr.org/papers/v11/cawley10a.html | 대회가 Ladder를 구현했다는 뜻은 아님 |
