# P2 결측 조건부 검증 — 가을 단서는 있으나 첫 seed/지원범위 한계

## 결론

신규 학습 **0**, 고정 C/R 첫-seed 모델 6개를 exact 재사용하여 16.297초에 검증했다. `공개 T5 또는 S5가 비유한이면 R, 아니면 C` 단일 규칙이다. 가을 intact는 RMSE **0.465330→0.464325℃(−0.001006℃)**, 새 가을 7/14일 outage 9,035행은 **0.971357→0.787438℃**로 개선했다. 다만 pooled intact는 **0.896731→0.900894℃**로 악화했다.

봉인 결과 `RESEARCH_ONLY_NOT_READY`는 그대로 보존한다. 새 가을 3일 artificial outage에서 4행의 공개 수온 지원이 2개 미만이 되어 사전계약상 전체 scenario의 성적을 산정하지 못했기 때문이다. 이는 **인위적 입력의 기술적 지원제약**이며 과학적 실패나 실제 공식 입력 결함을 입증한 것이 아니다. 4행만 잘라 점수를 계산하지 않았고 해당 1,294행 episode 전체 점수는 미산정이다.

## 동일 비교와 성과

평가 keys 69,850행, 가을 primary 26,273행을 유지했다. 기존 2024-10-18~11-01 14일 실험은 rule의 동기가 된 development로 별도 분리했다. 추가 episode는 실행 전에 달력으로 고정한 3/7/14일 × 가을/여름/겨울이며 결과로 날짜를 고르지 않았다. 반복 노출된 historical labels에 대한 새 masking일 뿐 fresh 확증이 아니다.

| 구간 | 평가행 | C RMSE ℃ | conditional RMSE ℃ | 해석 |
|---|---:|---:|---:|---|
| intact 가을 | 26,273 | 0.465330 | 0.464325 | 작은 평균 개선 |
| intact 전체 | 69,850 | 0.896731 | 0.900894 | 악화 |
| 새 가을 7/14일 지원완료 episode | 9,035 | 0.971357 | 0.787438 | 개선, 3일 scenario 미포함을 명시 |
| 새 여름 3/7/14일 episode | 10,339 | 1.712126 | 1.830062 | 악화 |
| 겨울 지정구간 | 8,250 | 0.235603 | 0.280003 | 이미 T5/S5가 결측이므로 새 outage 효과로 해석하면 안 됨 |
| 기존 development 14일 | 6,031 | 0.465796 | 0.224904 | 기존 결과 exact 재현, 독립 새 근거 아님 |

## 자연 결측과 trigger

독립 QA가 원본의 공개 layer5만으로 trigger를 재구성했다. 목표 layer2/3/4 결측은 rule을 켜지 않는다. 자연 trigger는 전체17,165행이다.

- 가을212행: 수온만130행, 염분만32행, 둘 다50행. 수온만 결측 RMSE0.832387→0.741626, 염분만0.665735→0.830943, 둘 다0.621714→0.329325℃. 종류별 효과가 다르지만 OR rule을 바꾸지 않았다.
- 여름69행은 염분만 결측, 겨울16,884행은 모두 T5/S5 결측이다. 겨울에서 rule은 모든 행을 R로 보내며 C0.242470→R0.301095℃로 악화했다. 따라서 “항상 intact와 같은 예측”이라고 주장할 수 없다.
- 공개 T5/S5가 모두 있는52,685행에서는 conditional이 C와 exact 동일하다. 인위적 episode 밖에서도 해당 intact policy와 exact 동일하다.

## 검증·계보·비용

- [설정](../../configs/experiments/p2_missingness_conditional_validation_20260905_v3.json), [runner](../../scripts/run_p2_missingness_conditional_validation_20260905_v3.py), [seal](preregistration-seal.json), [canonical result](result.json).
- Runner SHA `5c8675f6e625c07497e1804f010e03859036d8edc777ed40579df5d4a456f73c`, config SHA `e609972396364530fcc16ebbb6665a426b3350db2324147eca45e2a6152fce9b`.
- 모델C/R 각3fold, seed20260901. old checkpoint hash, keys/truth/folds, 새 CPU 재추론과 old OOF exact 일치. old A/B runner/config/lock/results 불변.
- 신규backbone0, rulefit0, fullfit0, CPU1thread/GPU0, 공식index/sample/baseline/hidden0, CSV/upload0, 행삭제0, Git변경0.
- [독립 QA](independent-qa.json) **142checks PASS**. 수온만/염분만/both/목표열만 결측의4 cases, onset 포함/offset 제외, targettemp+psal 선제거, lag dependency0/구간밖불변을 포함한 **11 synthetic pytest PASS**, Ruff PASS.
- `validate-data`와 `analyze-data-quality` 지침에 따라 분모, 지원되지 않는 scenario, 반복노출, 계절별 반대효과를 따로 기록했다. 원본 행값은 보고서에 출력하지 않았다.

## 제출 판단 경계

현재 공식 C는 **3-seed 평균 0.455143℃ /27.622418점**이다. 이 실행의 C는 첫 seed이므로 비trigger 예측이 오늘 공식 파일과 같다고 말할 수 없다. 새 예상 공식 점수는 미산정이다. 첫 seed 최솟값만 골라 배포하지 않는다.

가을 정보가치 단서를 확인한 root가 **별도 ID의 3-seed 검증/학습**을 승인했다. 이 v3는 재실행하지 않는다. R 나머지2seeds×3fold=6 historical fits와 R full3fits, 기존 C historical9/full3의 검증 재사용이 다음 계약이다. 추가 rule/threshold 탐색이나 공식 입력 접근은 이 문서의 승인이 아니다.
