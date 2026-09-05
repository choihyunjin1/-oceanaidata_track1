# 분리 실험과 기준선 재생성 — 통합 원장

## 결론

**성능 개선 후보는 P2 프로파일 보정 하나다. P1 T–S 특징과 P3 바람 증강은 현재 고정 실험에서 개선되지 않았다.** 세 기준선 모두 새 모델 폴더에서 학습→답안 생성과 별도 PID replay를 완료했다. P2/P3는 과거 clean 답안과 SHA까지 일치했으나 P1은 불일치했다. 수치 QA, 현재 머신 재생성, 과거 답안 일치, 독립 환경 최종 패키지 검증을 구분한다. 포털 업로드 및 최종 모델 잠금은 이 작업에서 수행하지 않았다.

## 결과 표

Δ는 후보−기준이다. **F1은 양수, RMSE는 음수가 개선**이다. harm 열은 양수가 악화다. CI90과 개선 비율은 이미 결과가 나온 historical 표면에 추가한 **서술적 paired block bootstrap**이며, 새 승격 기준이나 공식 개선 확률이 아니다.

| 변경/비교군 | 평균 Δ | 서술적 CI90 | 개선 resample 비율 | worst-fold harm | 정점/층/리드 최대 harm | 원 실험 시간 | 판단 |
|---|---:|---|---:|---:|---:|---:|---|
| P1 T–S vs clean control | F1 −0.00907248 | [−0.01869584, −0.00054590] | 4.05% | +0.02520139 | +0.02721933 | 576.657s / 6 fits | 개선 없음, full 후보 미제작 |
| P2 프로파일 / 가을 primary | RMSE −0.01601246℃ | [−0.03092231, −0.00090671] | 95.70% | −0.01601246 | −0.00928461 | 35.390s / 3 residual fits | 정보가치 후보, 아래 재생성·local 후보 QA 완료 |
| P2 프로파일 / pooled | RMSE −0.00574185℃ | [−0.01444918, +0.00280442] | 84.60% | −0.00350692 | +0.00471235 | 위와 같은 실행 | 평균 개선이나 CI가 0 포함 |
| P3 wind / matched LGBM | RMSE +0.01111106m | [−0.00183346, +0.02395265] | 7.30% | +0.02766738 | +0.02669757 | 274.203s / 6 fits | 해당 증강 채택 안 함 |
| P3 wind / clean 기준 | RMSE +0.03972393m | [+0.02204928, +0.05720551] | 0.05% | +0.08103053 | +0.05259629 | 위와 같은 실행 | no-op 유지 |

P2 가을 primary는 26,273행, pooled는 69,850행이다. primary 행의 worst-fold는 가을 한 fold뿐이라는 뜻이며 일반적인 최악 계절 보장이 아니다. 가을 7일/14일 결측 스트레스 RMSE는 각각 **+0.032764℃/+0.023480℃ 악화**했다. 3일 변형의 지원 불가 4행을 삭제해 유리하게 계산하지 않고 해당 전체 시나리오를 미채점으로 남겼다. 표의 일반 fold harm과 이 결측 스트레스 위험을 함께 읽어야 한다.

## 기준 모델 재생성 검사 통과 여부

| 기준 | 배포 데이터→새 학습→답안 | 새 모델 별도 PID replay | 과거 공식 답안 exact 복원 | 독립 환경 최종 패키지 |
|---|---|---|---|---|
| P1 clean v5 | 실행 완료: 4 fits, 169,011행 | PASS | **불일치** | 미검증 |
| P2 clean v6 | PASS: 새 3-seed 학습 → 26,061행 | PASS | **SHA 동일 PASS** | 미검증 |
| P3 clean v4 | PASS: 원본부터 8 backbone + 3 router → 1,200행 | PASS | **SHA 동일 PASS** | 미검증 |

P1 새 답안 SHA `5971e145f1ac38b8ee3e34cfd302973ba7a64b8873db11c354d3331221fdb28a`는 과거 `064ef022...`와 다르다. 새 파일에 과거 공식 점수를 붙이지 않는다. v4 실패를 보존하고 canonical CPU4/순서를 맞춘 v5에서 Q4 정책은 다시 도출했으나 full 답안까지 동일하지는 않았다. 이것은 과거 답안 복사 성공이 아니라 새 모델에서 새 답안을 만든 결과다.

P3는 과거 답안 내용 읽기 없이 새 답안 SHA `6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7`가 과거 clean SHA와 정확히 일치했다. source prepare부터 training/model replay/inference까지 1,654.002초, 별도 answer replay 3.862초다. training-only QA 123/123, 전체 QA 133/133 PASS. 이는 현재 머신의 실측이며 최종 운영진 환경 PASS를 뜻하지 않는다.

P2 v6는 가드 통합 테스트 후 새 3-seed 학습 69.094초, 새 프로세스 추론 10.844초, 별도 replay 8.734초를 완료했다. 새 C 답안 SHA `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071`가 기존 clean C와 정확히 일치했다. 27-check QA PASS. v4/v5 실패 모델은 재사용하지 않았다.

## P2 local 개선 후보

재생성된 C3에 full copula를 추가로 1회 적합했다. [후보 계약과 실행법](../p2_profile_copula_deploy_20260905_v4/README.md), [후보 QA](../p2_profile_copula_deploy_20260905_v4/independent-qa.json) 참조. 전체 26,061행 별도 PID CSV replay와 독립 QA 28/28 PASS, 합성 테스트 4 PASS/Ruff PASS. 새 backbone 학습은 이 단계에서 0이다.

- local 후보: `artifacts/p2_profile_copula_deploy_20260905_v4/05_answer/submission_p2_profile_copula.csv`
- SHA-256: `3dff2982f38ef2306248e76494c8d2c95b730244ce5730ba59335418dd3f557b`
- 고정 강도 1.0; 전 26,061행이 재계산 C와 달라졌다. 공식 점수 미측정, upload 0.
- 별도 `control_C3.csv`는 기준 SHA 검사 자료이지 이 개선 후보의 제출 파일이 아니다.

기준선부터의 통합 학습 명령을 포함한 최종 portable ZIP은 아직 별도 작업이다. 후보 폴더의 copula 학습만 돌린 것을 backbone부터 scratch 학습한 증거로 주장하지 않고 선행 v6의 실제 3-fit 원장을 연결한다.

P2 v4는 새 모델 저장 뒤 자신의 SHA 읽기를 막은 가드 오류다. v5의 정정도 native `torch.save`가 Python의 open 감사 이벤트를 거치지 않는 경우를 놓쳐 첫 seed 뒤 실패했다. 두 번 모두 모델 품질 실패로 분류하지 않고 실제 소모된 첫 seed 학습을 기록한다. v6에서는 단순 경로 함수 테스트가 아니라 실제 native 저장→SHA 경로를 감사 hook이 켜진 별도 프로세스로 검증한 뒤 진행한다. P3의 OS CPU affinity 제한으로 실행이 느려졌으며 기존 추정 시간을 성공 증거로 사용하지 않는다.

## 근거와 재사용

- [P1 원 실험·47-check replay](../p1_ts_disagreement_20260905_v4/report-source.md), [P1 v4 재생성 실패](../p1_clean_regeneration_20260905_v4/report-source.md), [P1 v5 재생성](../p1_clean_regeneration_20260905_v5/report-source.md).
- [P2 원 실험·367-check QA](../p2_profile_copula_residual_20260905_v4/report-source.md), [P2 v4 기술실패](../p2_clean_regeneration_20260905_v4/failure-report.md).
- [P3 원 실험·222-check QA](../p3_wind_only_dropout_20260905_v4/report-source.md), [P3 재생성 계약](../p3_clean_regeneration_20260905_v4/README.md).
- [원자료 재계산 집계·입력 해시](comparison.json), [재계산 코드](../../scripts/summarize_isolated_trials_20260905_v4.py), [합성 테스트](../../tests/test_summarize_isolated_trials_20260905_v4.py): 6 PASS, Ruff PASS. 추가 fit/공식 입력/CSV/upload 0. 별도 검토자가 입력 해시 7개, pooled 산술·join·표 수치 일치를 확인했다.
- [P2 T5 전문가 3-seed 실패 기록](../p2_missingness_conditional_3seed_20260905_v3/report-source.md)은 기존 자산 그대로 보존한다. 같은 가설을 첫 seed 결과만으로 다시 시작하지 않는다.
- [로드맵 §5 실행 해석](roadmap-review.md). 수정된 새 평가 계약은 [독립 구현 원장](../ocean_forward_v5/implementation-ledger.md)과 함께 로컬 커밋 `6cb99f733c59478567f29c2eed1ed1e0794b0cca`로 고정했다. 설정·평가 helper·합성 테스트·원장 4개 파일만 포함하며 45 tests PASS/Ruff PASS다. 실제 train-only 지원 감사와 새 평가 실행은 아직 하지 않았다. 지금 표를 새 반기/8블록/6분기 평가로 오해하지 않는다. push는 하지 않았다.

## 불확실성과 후속 판단

P1은 KST 일 단위(정점 함께), P2는 KST 7일 단위, P3는 저장된 정점×episode 단위로 각 기존 fold 안에서 2,000회 paired resampling했다(seed 20260905). P1 254/P2 가을 10·전체 26/P3 181 clusters다. P3의 181개를 정점 간에도 서로 독립인 기상사건 181개로 해석하지 않는다. pooled confusion/SSE를 다시 합산했으며 fold 점수를 단순 평균하지 않았다. 이 CI는 반복 후보 탐색, validation 재사용, 관측되지 않은 계절 변화나 센서 분포 이동을 보정하지 못한다.

현재 원 실험 3건의 후보 선택은 변경하지 않는다. clean 재생성과 코드/상수 출처를 확인하고 평가 계약을 새 버전으로 고정했다. 다음 단계는 그 계약의 train-only 0-fit 지원 감사다. P2 후보 local 답안은 기준 재생성 통과 후 별도 full residual 1-fit으로 준비했다. CI나 가을 최고 결과를 보고 full/half 보정 강도를 다시 고르지 않았다. 기존 실패·모델·답안·attempt lock은 보존하며, 최종 portable 패키지 및 독립 환경 재현 검증은 별도로 남아 있다.
