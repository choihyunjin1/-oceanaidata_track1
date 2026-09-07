# Fable 최종 검토 정본 — 2026-09-07 15:12 KST 스냅샷

> **최신 상태 — 2026-09-07 20:01 KST 접수 확인:** P1 20:00 / P2 19:44 / P3 19:59, 모두 `최종 제출(모델) / 모델 · 검증 대기 / 채점중`입니다. 첨부 목록 P1 21개·P2 3개(v2)·P3 4개를 저장된 양식에서 확인했습니다. [현재 상태·재현 근거·파일 안내](CURRENT_FINAL_STATUS_20260907.md)가 아래 역사적 보류/진행 기록보다 우선합니다. 운영진 검증 통과나 서버 파일 다운로드 SHA 검증을 뜻하지 않습니다. 재제출·삭제하지 마세요.

> **15:46 정정이 우선:** [P2 최소 수정·실제 재검증 결과](P2_PORTABILITY_REPAIR_AND_AUDIT_RESPONSE_20260907.md). 답안은 같고 P2 최종 ZIP은 새 `P2_v2/P2_FINAL_REPRODUCTION_794268f1_v2.zip` (485957 bytes / SHA 36b4b0e4a0de61277133a85fefd62a6629aa1ac1c109fc12652b1d1cb745408f)로 바뀌었습니다. 아래 aab30bbe ZIP은 보존된 v1 이력입니다. 상위 로컬 안내/P1 FORM의 버전 혼용도 정정했고 최종 확인은 여전히 보류입니다. [재검토 프롬프트](FABLE_FINAL_SELECTION_RECHECK_PROMPT_20260907.md).

## 결론과 권한

남은 리더보드 답안 5건은 모두 채점되어 **P1/P2/P3 잔여 0/0/0**이다. 아래 우선 후보는 공식 최종 모델 접수 완료를 뜻하지 않는다. 사용자가 Fable 독립 검토를 요청하며 **최종 확인 버튼 직전 보류**를 지시했다. 이번 후속 승인은 GitHub 정리·commit/push·검토 프롬프트 작성에만 적용한다. 업로드·삭제·최종 확인·실행 중 학습 변경은 하지 않는다.

P2는 보류 지시 전에 첫 `모델 최종 제출` 버튼을 눌러 잠금 경고 확인창까지 열었다. **OK를 누르지 않았고 접수 완료 receipt도 없다.** 이후 읽기 재연결은 `Debugger unattached`로 실패했다. 따라서 마지막 관찰과 현재 서버 상태를 동일시하지 않는다. 재개 승인 후에도 접수 여부를 먼저 확인하며 중복 클릭하지 않는다. P1/P3 최종 버튼은 누르지 않았다.

## 1. 우선 후보와 공식 성적 — 서로 다른 SHA의 점수 승계 금지

| 문제 | 검토 대상 | 답안 SHA256 | 행 수 | Public 지표 | Public 점수 | 재현 근거 / 남은 공백 |
|---|---|---|---:|---|---:|---|
| P1 | 원형 O+B+MS-TCN | 57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687 | 169011 | F1 0.833548 | 28.909341 | whole-cold 7fit, 6323.356초 및 saved replay exact. 여섯 셀의 역사적 OOF 선택 프로그램 미복구 |
| P2 | L120 3-seed smooth7 → clip/PAVA | 794268f15a0a7ac18ecd4dc99757083e159c49d639d2a8a72df3f835414cc481 | 26061 | RMSE 0.395254℃ | 28.373869 | 빈 모델 3fit 노트북 학습→추론→후처리 및 별도 saved ZIP 노트북 exact. 창 선택은 retrospective OOF |
| P3 | CPU numeric 3-seed no-shrink | 70761affca4d3fc6f1d24ae53467e5b185b23465926ebb4b851f0300872cddbd | 1200 | RMSE 0.595521m | 23.881592 | saved ZIP/별도 PID replay exact. fresh cold 진행 중; 전체 cold 일치 미증명 |

위 Public 점수 산술 합계는 **81.164802**이며 Private 점수나 공식 적격성 인증이 아니다. 과거 외부자료·점수 역산 적합 계보는 더 높은 Public 점수여도 최종 후보에서 제외한다. 과거 제출 기록은 삭제하지 않는다.

### 남은 5회 실제 사용 내역

| 후보 | SHA 앞8 | 공식 지표 | 점수 | 판단 / 정본 receipt |
|---|---|---|---:|---|
| P3 CPU no-shrink | 70761aff | 0.595521m | 23.881592 | 우선 후보. [receipt](../../reports/p3_numeric_cpudet_noshrink_20260907_v1/official-receipt.json) |
| P3 CPU mean/router 제거 | c9fa5366 | 0.600933m | 23.795692 | no-shrink보다 낮음. [receipt](../../reports/p3_cpudet_mean_router_ablation_20260907_v1/official-receipt.json) |
| P2 smooth7 projection | 794268f1 | 0.395254℃ | 28.373869 | 이전 projection-only보다 +0.133832점. [receipt](../../reports/p2_l120_s3_smooth7_projection_20260907_v1/official-receipt.json) |
| P1 B5 | cbeb7426 | F1 0.831622 | 28.858163 | 원형보다 낮음. [두 후보 receipt](../../reports/p1_original_learning_seed_ablation_20260907_v2/official-receipts.json) |
| P1 O_slow | c38ace7a | F1 0.826087 | 28.711047 | 원형보다 낮음. 위 동일 receipt |

SHA는 제출 직전 로컬 파일 증거이며 서버가 반환한 SHA라고 주장하지 않는다. 시각 정밀도와 quota 관찰은 각 receipt를 따른다. 새로운 브라우저 조작은 이번 정리에서 하지 않았다.

## 2. 실행 중 P3와 재현 판정

2026-09-07 15:12 KST `Get-Process -Id 26996`로 python 생존을 확인했다. 시작 12:02:51, CPU 누적 39606.09초. `fresh_cold_2/progress.json`은 backbone **23/36**, stage `CPU_TRAINING`, 경과 11226.520초를 기록했다. progress 경과는 마지막 파일 갱신 시점 값이며 현재 wall time과 다를 수 있다. 총 41fit(36 backbone + 5 router), CPU4 계약을 바꾸지 않는다.

- 루트: `C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_numeric_cpudet_unbounded_v2`.
- 완료 시 기반 답안은 `2015b38750d357630d5b2e9eee32807d2961ce752e35eb2b454ee16e579dda56`와 대조하고, 같은 frozen no-shrink를 새 모델에서 재생하여 위 `70761aff…`와 대조해야 한다.
- completion_1은 검증된 prefix와 후속 학습을 합친 실행이다. 이를 두 번의 독립 fresh cold 완료로 세지 않는다.
- 과거 17:30~18:00 완료 예상은 추정일 뿐 마감이나 완료 보장이 아니다. 최신 terminal/receipt가 이 스냅샷보다 우선한다.
- 완료 후에도 사용자 제출 보류를 자동 해제하지 않는다.

## 3. 로컬 패키지와 GitHub 경계

| 문제 | 로컬 경로 | 현재 제공 범위 |
|---|---|---|
| P1 | `C:/Users/cedis/Documents/OceanFinalRelease_20260907/P1` | 원형 SOURCE_ONLY/SAVED_MODELS, ANSWER, 15개 모델 분할 ZIP 및 재조립 도구. 기존 묶음의 P2/P3는 현재 선택과 다르므로 혼용 금지 |
| P2 | `C:/Users/cedis/Documents/OceanFinalSelected_20260907/P2/P2_FINAL_REPRODUCTION_794268f1.zip` | 최상위 README·FINAL_MANIFEST·SOURCE_ONLY·SAVED_MODELS·ANSWER·EVIDENCE. 최종 폼 첨부 준비, 접수 미확인 |
| P3 | `C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_numeric_cpudet_noshrink_v2` | no-shrink ANSWER/SOURCE_ONLY/SAVED_MODELS 및 재생 증거. fresh-cold 최종 증거 대기 |

P2 최종 ZIP: **484699 bytes**, SHA256 **aab30bbe5e244098c1ef379b09077ccd4e98bd6a6cd0ccf778109a7f5282054d**. [ready/audit](../../reports/final_selected_submission_20260907/p2-final-ready.json). SOURCE_ONLY와 SAVED_MODELS는 별도 대안이며 덮어 합치지 않는다. 최상위 README/CURRENT_README가 기존 core README의 중간 답안 지시보다 우선한다. 결과 JSON의 `uploads: 0`은 해당 로컬 빌더 동작의 범위이지 이후 폼 파일 선택/답안 제출까지 0이라는 뜻이 아니다.

GitHub에는 코드·설정·작은 집계 QA/receipt·문서만 포함한다. **원자료·모델·CSV·NPZ/parquet·ZIP·캐시·logs·attempt/finish lock·credentials는 올리지 않는다.** GitHub만 읽는 Fable은 로컬 ZIP/가중치의 직접 검증을 했다고 말할 수 없고 보고서에 의존한 부분을 구분해야 한다.

## 4. Fable이 우선 검토할 근거

1. [최상위 정책](../../00_ORGANIZER_DATA_POLICY.md), [독립 검토 요청](FABLE_FINAL_SELECTION_AUDIT_PROMPT_20260907.md).
2. P1 [result](../../reports/p1_original_learning_seed_ablation_20260907_v2/result.json), [독립 QA](../../reports/p1_original_learning_seed_ablation_20260907_v2/independent-qa.json), [원형 역사적 계보 감사](../../reports/p1_historical_path_audit_20260906_v1/claim-source-ledger.md). P1 새 O_slow의 작은 Q3/Q4 내부 이득은 Public에서 전이되지 않았다. Q2 동일 MS 근거가 없어 3fold라 부르지 않는다.
3. P2 [result](../../reports/p2_l120_s3_smooth7_projection_20260907_v1/result.json), [독립 QA](../../reports/p2_l120_s3_smooth7_projection_20260907_v1/independent-qa.json), [cold/saved notebook receipt](../../reports/p2_l120_s3_smooth7_projection_20260907_v1/candidate-ready.json). 층별 ±30분 창 뒤 endpoint 방향 clip/PAVA이며, 불완전 프로필/결측 endpoint에서는 **투영만 생략**하고 이미 평활된 값을 유지한다. fold-local와 cross-fold 진단을 혼동하지 않는다.
4. P3 [no-shrink result](../../reports/p3_numeric_cpudet_noshrink_20260907_v1/result.json), [test mix](../../reports/p3_numeric_cpudet_noshrink_20260907_v1/test-mix-check.json), [공식 receipt](../../reports/p3_numeric_cpudet_noshrink_20260907_v1/official-receipt.json). test-hs0 재가중 OOF는 악화를 예상했지만 Public에서는 개선됐다. 재가중을 무조건 옳다/틀리다 판정하거나 Public로 계수를 다시 적합하지 않는다.
5. [삭제 검토 표](USER_HANDOFF_DELETION_CHECK_20260907.md). 미대조 항목과 삭제 가능 여부·재현 대상 답안 선택 규칙은 미확인으로 남긴다. 삭제 실행 승인이 아니다.

## 5. 확정하지 못한 것과 재개 조건

- 정확한 모델 마감 시각, 첨부 개수 상한, 재현 대상 답안(best/latest/선택형)과 허용오차는 미확인. 파일당 50MB는 이전 로그인 UI 실측이다.
- 일반 모델에 공식 6시간이 적용되는지 미확인. 합성 사전학습 예외 문구와 내부 목표를 일반 규정 확정으로 확대하지 않는다.
- P1 셀 상수는 08-26 로컬 OOF 근거와 고정 recipe가 있으나 원 선택 알고리즘 미복구. 이를 숨기거나 사후 알고리즘을 역사적 원본이라 만들지 않는다.
- P2 local cold/saved 일치는 실제 증거이나 OS 네트워크 차단·fresh venv 검증과 운영진 적격성 인증은 아니다.
- **사용자 검토 후 명시적 재개 지시 → 서버 접수 현황 읽기 확인 → 미해결 규정/재현 문제 판단 → 승인된 최종 클릭 → 실제 접수 receipt** 순서다. Git push는 이 단계를 대신하지 않는다.

Fable 산출물은 `docs/ocean_v2_codex/FABLE_FINAL_SELECTION_AUDIT_20260907.md`: `[문제 | 근거 파일/SHA | 확인 사실 | 결함/미확인 | 최소 수정 | 제출 전 필수 여부]`. 치명적 불일치를 먼저 적고 점수·재현·규정 준수·접수 상태를 분리한다. 새 fit·CSV 변경·업로드·삭제·최종 지정·commit/push 없이 검토한다.
