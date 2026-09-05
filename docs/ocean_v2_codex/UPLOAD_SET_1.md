# 재생성 기준 답안 — 첫 업로드 판단표

2026-09-06 01:49 KST 갱신. **사용자 승인으로 P1 새 재생성본 채점을 완료했다: Public F1 0.777749 / 27.426319점. P2/P3도 이미 채점된 동일 SHA 답안이므로 세 재생성 파일에 공식 점수가 연결됐다.** P1 답안 업로드 1회만 수행했고, 최종 모델 지정·새 답안 생성·commit/push는 하지 않았다.

## 파일과 실제 공식 근거

| 문제 | 행 수 / validator | 실제 공식 기록 | 판단 |
|---|---|---|---|
| P1 | 169,011 / schema·key·order·binary·finite·중복 0 PASS, 별도 PID replay exact | 새 SHA `5971e145…`, 09-06 01:49 **F1 0.777749 / 27.426319점** | 기존 다른 clean SHA `064ef022…`의 F1 0.790733 대비 −0.012984, 점수 −0.345081. 개선 아님. 과거 점수를 승계하지 않음 |
| P2 | 26,061 / schema·key·order·finite·층·중복 0 PASS, 27-check QA/replay exact | 동일 SHA, 09-05 19:28 RMSE **0.455143℃ / 27.622418점** | 이미 점수 있는 재생성 fallback. 중복 업로드 불필요 |
| P3 | 1,200 / 200 cases·6 leads·schema·key·order·finite 0~30·중복 0 PASS, 133-check QA/replay exact | 동일 SHA, 09-05 19:29 RMSE **0.607183m / 23.6965점** | 이미 점수 있는 재생성 fallback. 08-25 O의 0.607071과 혼동 금지 |

P1 공식 값은 [9월 6일 채점 영수증](../../reports/p1_regenerated_baseline_official_submission_20260906_v1/receipt.json), P2/P3는 [9월 5일 브라우저 채점 영수증](../../reports/official_score_repair_submissions_20260905_v1/receipt.json)에 SHA로 연결된다. 9월 6일 로그인된 문제 카드와 제출관리의 P1 신규 채점 결과가 일치했다. P1 잔여 횟수는 3/3 → 2/3. 제출관리에서 P2/P3의 9월 5일 기록도 여전히 존재함을 확인했다. UI submission ID와 서버 SHA는 표시되지 않으므로 파일 선택 경로·로컬 검증 SHA·단일 제출 동작·직후 영수증으로 연결했다. 정확한 전체 마감 시각은 이번에 재확인하지 않았다.

세 답안의 파일 SHA를 이번 문서 작성 시 직접 다시 계산해 아래 값과 일치함을 확인했다. validator 행 수/키 검사는 동일 SHA를 대상으로 실행된 아래 canonical QA를 사용했으며, 원본 공식 값이나 숨은 정답을 다시 열지 않았다.

## P1 — 새 재생성 파일

절대경로: `C:\Users\cedis\PycharmProjects\PythonProject\artifacts\p1_clean_regeneration_20260905_v5\05_answer\P1_submission.csv`

SHA256: `5971e145f1ac38b8ee3e34cfd302973ba7a64b8873db11c354d3331221fdb28a`

- 답안 열: `station,year,layer,time,label`.
- 모델: 새 XGBoost O + LightGBM B, 80개 특징. earlier-inner 학습에서 다시 선택된 balanced_union 및 O0.2/B0.3 decoder. CPU4, inner2+full2 fits. MS-TCN·옛 router_anchor·GI 답안 패치는 없음.
- 근거: [학습/재생성 보고](../../reports/p1_clean_regeneration_20260905_v5/report-source.md), [답안 QA](../../reports/p1_clean_regeneration_20260905_v5/inference-qa.json), [독립 QA](../../reports/p1_clean_regeneration_20260905_v5/independent-qa.json).
- 이전 clean 답안과 최소 110 label 차이이며 exact changed count는 미측정. 새 모델의 저장 후 replay는 통과했지만, 동일 전체 최종 진입점의 두 번 scratch 실행·portable cleanroom은 미완료다. 현재 별도로 실행 중인 CPU2 O 부분 결정론 검사는 이 CPU4 답안의 전체 재학습 증거가 아니다.
- 사용자가 올릴 곳: [OCN-01 답안 채점](https://oceanaidata.org/app/problems/5). 위 CSV를 선택하며 `.joblib`, replay 파일, 옛 최고점 파일을 대신 선택하지 않는다.
- 제목 초안: `P1 재생성 clean O+B 기준선`. 요약: `배포 train 새 학습 O+B와 inner 선택 decoder; 과거 답안·점수 역산 보정 없음.`

## P2 — 이미 채점된 것과 동일한 재생성 파일

절대경로: `C:\Users\cedis\PycharmProjects\PythonProject\artifacts\p2_clean_regeneration_20260905_v6\05_answer\submission_p2_clean_C3.csv`

SHA256: `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071`

- 답안 열: `station,layer,time,temp`.
- 모델: v23 blockmask DeepSets C3, 3 seeds×60 epochs, 공개 프로파일 기준선 + scale 복원 잔차의 3seed 평균. bin17 anchor·copula 후속 보정·리더보드 역산 상수 없음. 새 재생성은 GPU 학습 계열이며 현재 CPU-only 내부 실험과 다른 실행 환경이다.
- 근거: [재생성 보고](../../reports/p2_clean_regeneration_20260905_v6/report-source.md), [답안 validator](../../reports/p2_clean_regeneration_20260905_v6/result.json), [독립 QA](../../reports/p2_clean_regeneration_20260905_v6/independent-qa.json).
- 기존 채점 경로: `artifacts/p2_score_repair_deploy_20260905_v1/submission_p2_v23_blockmask_3seed.csv`. 파일명은 달라도 전체 SHA가 같다.
- 대응 화면: [OCN-02](https://oceanaidata.org/app/problems/6). 현재 목록에 09-05의 동일 후보가 있다면 그대로 fallback으로 유지한다.

## P3 — 이미 채점된 것과 동일한 재생성 파일

절대경로: `C:\Users\cedis\PycharmProjects\PythonProject\artifacts\p3_clean_regeneration_20260905_v4\05_answer\submission.csv`

SHA256: `6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7`

- 답안 열: `case_id,station,lead_h,hs_pred`.
- 모델: 591개 배포 context 특징, single/multi CatBoost, 새 historical OOF에서 적합한 기존 clean router, 사전 고정 long-lead persistence shrink 0.2. Ridge alpha=10은 학습 정규화로 Public 역산 alpha와 다르다. shrink 없는 등가 평균 코어라고 표시하지 않는다. multi GPU 포함.
- 근거: [답안 QA](../../reports/p3_clean_regeneration_20260905_v4/answer-qa.json), [독립 QA](../../reports/p3_clean_regeneration_20260905_v4/independent-qa.json), [재생성 보고](../../reports/p3_clean_regeneration_20260905_v4/report-source.md).
- 기존 채점 경로: `artifacts/p3_score_repair_deploy_20260905_v1/candidates/clean_baseline.csv`. 재생성 CSV와 SHA가 같다.
- 대응 화면: [OCN-03](https://oceanaidata.org/app/problems/7). 0.607071 파일을 이 모델의 답안으로 잘못 지정하지 않는다.

## 사용자 선택과 남은 단계

1. P1 위 SHA의 **답안 채점 완료**. 9월 6일 01:49 KST 공식 F1 0.777749 / 27.426319점, 오늘 남은 횟수 2/3. 이 파일은 새 구간 모양(bracket) 실험 후보가 아니라 재생성 기준선이다. 동일 파일을 다시 제출하지 않는다.
2. P2/P3는 동일 SHA의 기존 채점 기록을 확인해 보존한다. 새 개선 후보가 나올 때만 별도 슬롯 사용을 판단한다.
3. 답안 채점과 **모델 최종 제출/잠금**은 다르다. 현재 portable package/오프라인 전체 cleanroom/동일 전체 재학습 두 회를 통과했다고 주장하지 않으며, 이 표는 모델 잠금 지시가 아니다.
4. [BASELINE_REGEN_CHECK.md](BASELINE_REGEN_CHECK.md)와 향후 CLEANROOM_RESULT를 함께 확인한다. 이 문서의 공식 점수는 새 내부 실험 선택 계수나 예상 점수 보정에 사용하지 않는다.
