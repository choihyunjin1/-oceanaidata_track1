# 2026-08-31 full internal submission cycle v3

## 결론

이번 사이클은 미세 점검이 아니라 P1/P2/P3 합계 49회 실제 학습, 시간 순서 내부 테스트, 공식 입력 순서에 맞춘 제출 CSV 7개 생성을 끝낸 전체 사이클이다. 구조 QA는 7개 모두 통과했지만 내부 성능 게이트를 통과한 제출 후보는 `P1_2_HIST_GBDT_OOF_STACK_UNION`과 `P2_1_PUBLIC_PROFILE_RESIDUAL_SHALLOW` 두 개뿐이다. P3 두 후보는 내부 RMSE가 악화되어 제출 보류가 맞다. 업로드는 수행하지 않았다.

## 실행 범위

- P1: 17 fits, Q3/Q4 forward test 287,862행, 후보 3개
- P2: 24 fits, 3개 시간 차단 외부 테스트와 KST-day 2,000회 bootstrap, 후보 2개
- P3: 8 fits, 3개 historical episode 차단 테스트와 2,000회 bootstrap, 후보 2개
- hidden truth read: 0
- upload: 0

## 후보 판정

| 문제 | 후보 | 내부 변화 | 판정 | 공식 제출본 변화 |
|---|---|---:|---|---|
| P1 | P1_1_LOGISTIC_OOF_STACK_UNION | F1 -0.002204560 | FAIL | champion 대비 +684행 |
| P1 | P1_2_HIST_GBDT_OOF_STACK_UNION | F1 +0.001380986 | PASS | champion 대비 +4행, 제거 0행 |
| P1 | P1_3_EXTRA_TREES_OOF_STACK_UNION | F1 +0.000000000 | FAIL | champion 대비 +20행 |
| P2 | P2_1_PUBLIC_PROFILE_RESIDUAL_SHALLOW | RMSE -0.010242946 C | PASS | champion 대비 RMS 변화 0.076507 C |
| P2 | P2_2_PUBLIC_PROFILE_RESIDUAL_DEEP | RMSE -0.017450344 C, 1/3 fold 개선 | FAIL | fold 불안정성으로 탈락 |
| P3 | P3_1_RIDGE_KMA_RESIDUAL_STACK | RMSE +0.000631095 m | FAIL | pooled 악화, 개선확률 0.4385 |
| P3 | P3_2_CATBOOST_KMA_RESIDUAL_STACK | RMSE +0.007962204 m | FAIL | pooled 악화, 개선확률 0.1325 |

P2 shallow은 2/3 folds가 개선되었고 pooled delta의 90% bootstrap CI가 `[-0.016620511, -0.006978717] C`였다. 다만 2025 Jul-Aug fold는 `+0.013565820 C` 악화되어 계절 수송 위험을 제출 판단에 명시한다. P1 GBDT는 Q3 동률, Q4 `+0.003432580 F1`이며 add-only 4행이라 변화 폭은 작지만 방향은 일관되게 보수적이다.

## 독립 QA

원본 공식 test/test_index의 key 및 order를 제출본과 독립 재대조했다. 7개 모두 예상 행 수, exact columns, exact key order, duplicate key 0, finite, recorded SHA-256 일치를 통과했다. P1은 binary, P3는 0--30 범위도 통과했다. 내부 평가가 끝난 뒤에만 공식 covariate를 읽었으며 hidden truth는 읽지 않았다.

## 제출 우선순위

1. `P2_1_PUBLIC_PROFILE_RESIDUAL_SHALLOW`: 동일 계절 fold 개선과 bootstrap 근거가 가장 강하다.
2. `P1_2_HIST_GBDT_OOF_STACK_UNION`: 개선 폭은 작지만 add-only 4행이고 forward gate를 통과했다.
3. P3는 이번 사이클 후보를 제출하지 않는다. 남은 기회는 새로운 구조가 내부 차단 테스트를 통과할 때까지 보존한다.
