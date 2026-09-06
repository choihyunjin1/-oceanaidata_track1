# 최종 패키지 cleanroom — 최신 결과와 이력

## 2026-09-07 현재 선택본

최신 파일 선택은 [최종 패키지 안내](../FINAL_RELEASE_20260907.md), 실제 archive/노트북/해시 검증은 [최종 보고서](../../reports/final_release_20260907_v1/report-source.md)를 따른다.

- P1 원형 O+B+MS: 배포자료부터 7fit/6323.356초, 답안 `57844ef2` exact, 원형 28.909341점 파일 복원. 실제 CLI 완료, RUN_ALL notebook schema 검증은 별개.
- P2 L120: 새 ZIP 추출/빈 모델→TRAIN/PREDICT notebook/3fit/173.010초, 답안 `fee6118b` 및 모델 bytes exact. 새 환경 설치·OS 차단망 검증은 아님.
- P3 numeric: 새 ID whole-cold 12 backbone+5 router=17fit, 수치 replay까지 1,497.613초, 실제 TRAIN/PREDICT notebook 완료. 새 답안 `56e289af`는 채점본 `ff42a6a0`와 660/1,200행이 다름(최대 0.003506m, 예측 간 RMSE 0.000546m; 정답 대비 RMSE 아님). 채점 당시 저장 모델 ZIP의 실제 새 추출 추론은 6.026초/0fit/`ff42a6a0` exact. 새 cold의 공식 점수 승계·bit 결정론·운영진 허용오차 충족은 주장하지 않는다.

아래의 bracket·hmax·baseline 및 초기 NOT_RUN은 날짜별 보존 기록이며 현재 선택본이 아니다.

## 후속 P3 hmax 후보 — 09-06 whole-cold 완료

배포 `train_wave.csv`·`train_atmos.csv`부터 새 특징을 생성하고 빈 모델 폴더에서 **12 backbone + 5 router**를 학습했다. 별도 PID 학습 QA, 1,200행 추론, 다시 새 PID byte-exact 답안 replay까지 **1,611.919초(26분 52초)**였다. 답안 SHA는 `d45605922ae8ce699c07405d8361765290c39bd38ddd0123f4e2e9ccd01808c5`. 새 학습 내부 RMSE 0.68246927556은 이전 연구 실행 수치와 별도다.

[cold 보고서](../../reports/p3_forward_candidate_cold_20260906_v1/report-source.md), [root 22-check QA](../../reports/remaining_work_completion_20260906_v1/p3-pre-upload-root-qa.json). 보존본은 `artifacts/p3_forward_candidate_cold_20260906_v1/completed/P3/`. 이 폴더를 재시작하거나 lock을 삭제하지 않는다. 새 전체 GPU 학습은 1회이며 반복 전체 학습의 byte 결정론/새 OS·차단망 설치 검증은 아니다. 저장 모델 ZIP의 별도 추출 검증은 [saved 보고서](../../reports/p3_forward_saved_20260906_v1/report-source.md)를 확인한다.

## 후속 P1 bracket 후보 — 09-06 완료

오늘 채점한 `9031c84e…d93a` 후보도 [독립 portable 검증](../../reports/p1_bracket_portable_20260906_v2/report-source.md)을 완료했다. 빈 `03_model`의 새 ZIP 추출본에서 **CPU2 4회 학습 → 모델 새 PID replay → 독립47-check QA → 추론 → 별도 PID 답안 replay**가 완료됐고, **169,011행/7,014 양성/동일 답안 SHA**를 재현했다. 학습376.750초, 단계 대기 포함 전체753.030초. 기존 CPU4 모델 bytes와는 다르며, 같은 답안 재현과 같은 모델 bytes 재현을 혼동하지 않는다.

별도 장기 저장 추론 ZIP도 실제 새 폴더 추출/새 PID/학습0/29.234초로 같은 답안 SHA를 재현했다. [root 9개 hash/link 대조](../../reports/remaining_work_completion_20260906_v1/p1-package-root-check.json) 및 [공식 F1 0.785944 영수증](../../reports/p1_bracket_official_submission_20260906_v1/receipt.json)을 연결한다. 전체 CPU2 scratch 1회이며 새 머신·새 의존환경·OS 차단망 검증은 아니다. 기존 세 문제 기준 fallback은 아래 그대로 보존한다.

## 2026-09-06 04:17 KST 갱신

**P1 두 독립 재학습 PASS, P2/P3 새 학습 및 별도 PID replay PASS.** 아래 초기 NOT_RUN 기록을 현재 상태로 오해하지 않는다. 이 판정은 같은 PC/기존 의존환경의 원 저장소 밖 실행이며 운영진의 차단망·새 PC 재현 인증은 아니다.

| 문제 | 새 빈 모델 폴더→학습→답안 | 전체 재학습 반복 / 저장 모델 replay | 학습+추론 시간 | 현재 증거 |
|---|---|---|---:|---|
| P1 | 4 fits × 2, 169,011행 기준 SHA exact | 4개 모델 SHA/답안 SHA 두 실행 exact, 별도 PID replay exact | 194.765 / 215.079초 | [20-check QA](../../reports/portable_cleanroom_20260906_v1/P1/independent-qa-v2.json), [보고서](../../reports/portable_cleanroom_20260906_v1/P1/report-source.md) |
| P2 | 3 fresh fits, 26,061행 기준 SHA exact | 이번 전체학습 1회; 별도 PID replay exact | 80.203초 | [71-check QA](../../reports/portable_cleanroom_20260906_v1/P2/independent-qa.json) |
| P3 | pristine v2 run_b: 8 backbone+3 router, 1,200행 기준 SHA exact | 전체 중단 없는 재학습 1회; 별도 recovery 44/44 PASS, 두 답안/예측 exact. CBM bytes는 서로 다름 | 계산1,203.415초 / 단계 간 대기 포함1,256.163초 | [42-check QA](../../artifacts/portable_cleanroom_20260906_v1/P3/v2_run_b_completed/04_logs/receipts/independent-qa.json), [복구 비교](../../reports/portable_cleanroom_20260906_v1/P3/repetition-comparison.json) |

보존 패키지는 `artifacts/portable_cleanroom_20260906_v1/<P>/` 아래에 있다. 코드/새 모델/답안/영수증을 분리했으며 배포 소스 데이터는 동봉하지 않는다. P1/P2의 기존 채점 SHA 재현은 점수 개선을 의미하지 않는다. 진행 중인 P1 양측 학습량/특징 진단 모델을 이 패키지와 섞지 않는다. 이번 작업의 commit/push/upload/final lock은 0이다.

### 검증 범위와 남은 항목

- P3 최신 `archives_v3`의 cold-start 코드 ZIP은 빈 모델 학습용, 저장 모델 ZIP은 `archive_infer.py` 전용이다. ZIP 실제 추출 후 저장 모델 추론3.668초/학습0회로 기존 답안 SHA exact를 확인했다. run_b/recovery 예측은 exact이나 CBM bytes는 서로 다르다. recovery는 7개 성공 모델을 재사용하므로 두 차례 중단 없는 cold-start PASS라고 기록하지 않는다. 정확한 run_a 중단 원인은 terminal/OS 종료 증거 부재로 미확인이다.
- 새 머신 또는 새 dependency 설치·OS 차단망 실행 및 현재 포털 최종 첨부 요건 확인. 로컬 패키지 준비와 최종 제출 승인은 별개다.
- 개선 후보가 나오면 그 코드/모델/답안을 별도로 재현 검증하며 현재 기준 fallback을 보존한다.

## 초기 계획 기록 — 아래 NOT_RUN은 위 갱신 전 상태

2026-09-06 현재 **NOT_RUN**. 이는 성능 실패나 재생성 불가 판정이 아니라, 아래 독립 패키지 시험을 아직 수행하지 않았다는 뜻이다. 기존 보고서의 checkout 내 scratch 학습 및 저장 모델 replay를 이 시험의 PASS로 대체하지 않는다.

| 문제 | 기존 checkout 재생성 근거 | 원 저장소 없이 새 패키지의 빈 모델 폴더→학습→답안 | 시간 / 판정 |
|---|---|---|---|
| P1 | v5 새 답안 생성/replay PASS, 과거 clean SHA 불일치 | NOT_RUN | 미측정 / 미검증 |
| P2 | v6 새 학습 답안과 기존 채점 SHA 일치 | NOT_RUN | 미측정 / 미검증 |
| P3 | v4 새 학습 답안과 기존 채점 SHA 일치 | NOT_RUN | 미측정 / 미검증 |

정확한 기준 파일은 [UPLOAD_SET_1.md](UPLOAD_SET_1.md), 모델 구성과 증거 범위는 [BASELINE_REGEN_CHECK.md](BASELINE_REGEN_CHECK.md)를 따른다. 진행 중인 CPU2 내부 ablation 모델은 이 기준 답안의 최종 weights로 대체하지 않는다.

## 실행 전 남은 항목

1. 실제 존재하는 재생성 driver/필요 함수로 문제별 portable 진입점 구현. 사양의 `ocean_v2.p?` 모듈은 아직 없으므로 현 상태에서 실행 가능하다고 안내하지 않는다.
2. 개인 경로·원 저장소 import·다른 문제·외부 계보 모듈 제거, 정확한 환경/의존 버전과 데이터 해시 manifest 작성.
3. 기존 모델 폴더를 삭제하지 않고 별도의 새 빈 작업 디렉터리를 생성. 원본 배포 데이터는 불변 참조하고 업로드 ZIP에서 제외.
4. 패키지 단독 학습→모델→추론→답안 SHA 검사. 전체 학습의 두 실행 결정론과 저장 모델 replay를 구분하고, 허용오차는 비교 후 임의로 늘리지 않음.
5. 실제 총 시간 ≤6h, 적합값/상수 출처, 외부·과거 답안·리더보드 역산 의존0, 포털 첨부 요건을 검증한 뒤 이 표를 실제 receipt 링크로 갱신.

현재 이 문서를 작성하면서 추가 학습·CSV 생성·모델 삭제·커밋·푸시·업로드는 수행하지 않았다. `PACKAGING_SPEC.md`의 리터럴/JSON 의무·기대 Public 범위에 관한 오래된 문구는 수정 로드맵과 [피드백 검토](FABLE_FEEDBACK_REVIEW_20260906.md)의 출처 감사 원칙으로 대체한다.
