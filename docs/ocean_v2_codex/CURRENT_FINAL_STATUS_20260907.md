# 최종 제출 현재 상태 — 2026-09-07 20:01 KST

세 문제 모두 최종 모델 **서버 접수 완료, 운영진 검증 대기**입니다. 사용자 승인 후 제출했으며, 저장된 제출 양식을 읽기 전용으로 확인했습니다. 과거 감사·재개 절차서의 보류 문구는 해당 시각의 이력입니다. 중복 제출·삭제하지 않습니다.

| 문제 | 서버 접수 시각(KST) | 답안 SHA 앞8 | Public 점수 | 확인한 첨부 수 |
|---|---|---|---:|---:|
| P1 | 20:00 | 57844ef2 | 28.909341 | 21 |
| P2 | 19:44 | 794268f1 | 28.373869 | 3 |
| P3 | 19:59 | 70761aff | 23.881592 | 4 |

모두 화면 상태는 `최종 제출(모델) / 모델 · 검증 대기 / 채점중`입니다. Public 합계 81.164802는 Private 점수나 적격성 인증이 아닙니다. [접수 정본](../../reports/final_model_submission_20260907_v1/receipt.json)에 로컬 첨부 28개의 전체 SHA256·bytes·경로와 화면 대조 방법을 기록했습니다. 서버 파일 다운로드 후 SHA 대조는 하지 않았습니다.

## 제출본과 실행 안내

로컬 공통 부모는 `C:/Users/cedis/Documents/`입니다. 각 폴더의 실제 제출한 `SUBMISSION_README_20260907.md`와 `FORM_FINAL_20260907.md`를 사용합니다.

- **P1:** `OceanFinalRelease_20260907/P1/`. SOURCE_ONLY ZIP + 저장 모델 part001~015 ZIP 전부 + 재조립 manifest/tool/README + 최신 외부 README/FORM. 약 590MB의 저장 모델 ZIP을 15분할한 것으로 서로 다른 15개 후보가 아닙니다. 모든 part를 모아 재조립 도구로 복원해야 합니다.
- **P2:** `OceanFinalSelected_20260907/P2_v2/`. `P2_FINAL_REPRODUCTION_794268f1_v2.zip` (485957 B, SHA `36b4b0e4a0de61277133a85fefd62a6629aa1ac1c109fc12652b1d1cb745408f`) + 외부 README/FORM. SOURCE_ONLY와 SAVED_MODELS는 **독립 실행 대안이며 합치지 않습니다**. SOURCE_ONLY는 TRAIN → PREDICT → SMOOTH_PROJECT, 저장 모델은 SAVED_PREDICT를 실행합니다.
- **P3:** `OceanFinalDay_20260907/P3_numeric_cpudet_noshrink_v2/`. SOURCE_ONLY.zip + SAVED_MODELS.zip + 외부 README/FORM. 두 ZIP은 **같은 루트에 합칩니다**. 학습은 RUN_TRAINING → RUN_INFERENCE → REPLAY_INFERENCE, 저장 모델은 RUN_INFERENCE → REPLAY_INFERENCE입니다.

옛 OceanFinalRelease의 상위 START_HERE/RELEASE_MANIFEST와 P2 v1 ZIP은 현재 제출 안내가 아닙니다. 저장소의 날짜별 빌드/finish 스크립트는 당시 실행 증거이며, 이미 사용된 폴더에서 다시 실행하지 않습니다. 모델·답안·ZIP·배포 데이터는 Git에 올리지 않습니다.

## 재현 확인과 정정

- P1 전체 cold 6323.356초. 답안 `57844ef2…` 재현. 여섯 셀 정책의 역사적 선택 프로그램 미복구를 공개했으며, 셀 연산 재현과 선택 절차 복원을 혼동하지 않습니다.
- P2 v2는 역사적 기반 답안 SHA 불일치를 기록 후 계속하되 **현재 실행 무결성 검사**를 유지합니다. 새 cold/saved 최종 답안 exact. v2 TRAIN 179.766 / PREDICT 39.172 / SMOOTH_PROJECT 14.672 / SAVED 53.360초. ZIP 내부 마지막 시간 항목의 v1 값은 실제 제출 외부 문서에서 정정했습니다. [수정 QA](../../reports/p2_smooth7_portability_repair_20260907_v2/repair-qa.json).
- P3 전체 신규 41 fit·재사용 0, 22182.109초. 새 모델에서 기반 `2015b387…`와 최종 no-shrink `70761aff…` 모두 exact, 별도 PID·13 QA PASS. 추가 후처리/QA 14.016초. 전체 fresh cold는 **1회**이며 기존 completion_1은 부분 재사용입니다. [검증 정본](../../reports/p3_fresh_noshrink_confirmation_20260907_v1/result.json).
- 실측은 우리 PC 환경의 결과입니다. 일반 모델에 공식 6시간 규칙이 적용되는지는 미확인입니다. 이번 모델들은 배포 데이터로 직접 학습했으며 합성-only 사전학습 예외를 이용한 모델이 아닙니다.

## 남은 확인 사항

운영진 적격성·재현 검증 결과, 서버가 저장한 파일의 다운로드 SHA, 공식 허용오차는 미확인입니다. P2 CUDA 필요, 다른 GPU·새 가상환경·OS 차단망에서의 전체 재검증도 증명하지 않았습니다. 로컬 QA PASS와 최종 모델 접수는 운영진 심사 통과와 별개입니다.

[Fable 독립 재검토](FABLE_FINAL_SELECTION_RECHECK_20260907.md)와 기존 receipt는 역사적 원문을 보존했습니다. 이번 Git 정리는 최종 접수 기록·소스·집계 QA·안내를 보관하는 작업이며, 제출된 모델·ZIP·답안은 변경하지 않습니다.
