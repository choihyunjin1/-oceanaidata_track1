# 현재 코어 추가 학습 — 2026-09-06

## 목적과 상태

사용자 `진행하세요`에 따라 현재 좋은 적격 모델의 학습 설정을 비교하고, 이미 내부 개선된 P3 numeric-lead 후보를 답안까지 완성한다. 새 구조 탐색 자체가 목적이 아니다. 상태는 작업 배정/사전 점검 진행 중이며, 이 문서는 학습 시작이나 성능 개선 증거가 아니다. 각 문제의 사전등록·실행 receipt가 실제 수행 여부의 근거다.

### 실제 시작 확인 — 2026-09-06 13:08 KST

- P3: 13:04:17 launcher19060 / child4544 확인. 신규 full single CPU 학습부터 실행하며 이후 multi GPU와 router 학습을 수행한다. stderr tail 비어 있음. 실행 전 QA schema 오류는 fit0에서 발견됐으며 별도 실패 receipt를 보존하고 수정된 driver의 28 synthetic tests/Ruff/preflight PASS 후 새 lock으로 시작했다.
- P1: 13:06:54 launcher17636, 하위39076/실학습22312. progress상 1 fit 완료, 두 번째 fit 실행 확인. 모든 native pool 2 threads. 고정 runner SHA `4687b1e7273aeb22b686fd9acba3f7d19cd60416a6aa58f5832006cfe5a215bb`, seal SHA `938063dc9c3c75aa98624e0bf9d5603b513da96cd3e2648f2ee0c93c76fd82f1`.
- P2: 코드/검증 준비 중이며 아직 GPU 인계 전. 시작 전 상태를 실제 학습으로 세지 않는다.
- 이 시작 확인은 완료·개선·제출 성공의 증거가 아니다. 이후 상태는 문제별 terminal/QA receipt로 확인한다.

### 자원 인계와 후속 시작 — 13:16 KST

- P3 full3fits는254.056초에 완료됐고 실제 child4544/launcher19060 소멸 및 terminal을 root가 확인한 뒤 GPU0를 P2에 명시 인계했다. P3는 이후 CPU 추론/새PID 답안 replay를 완료했다.
- P2는 synthetic15/Ruff/seal/source-support 통과 후13:15:50 시작했다. launcher36708/venv39980/worker38328을 root가 확인했다. source166268행/8fold이며 B4/B8의 마지막17일 평가행0은 그대로 보고한다.
- P2 첫 예정 fit(C60/B1)은15.453초, 고정 예산 산식의 보수적 전체 예상1180.821초로90분 이하라 계속한다. 이 첫 fit을 버리거나 다시 학습하지 않는다. 실제 완료시간은 terminal에서 별도로 확정한다.

## 작업 분리와 자원

| 담당 | 새 작업 ID | 범위 | 초기 자원 |
|---|---|---|---|
| p1_next_design | p1_core_tuning_20260906_v1 | 현 O/B tree 학습량·학습률·복잡도·정규화의 제한된 비교; feature/decoder/MS 고정 | CPU2, GPU 금지 |
| p2_next_design | p2_c3_training_comparison_20260906_v1 | 과거 중복을 먼저 확인하고 동일 환경 C3 학습 설정/목적함수 비교 | CPU2 준비, GPU는 P3 종료 후 root 인계 |
| p3_next_design | p3_numeric_candidate_20260906_v1 | 완료된 numeric OOF/QA의 exact hash 계보를 재사용하고 full 학습·QA·CSV·새 PID replay | CPU2, GPU0 단독 우선 |

실측 시스템은 CPU8 logical/메모리 약63GiB/RTX5090 32GiB다. root 수치검증은 순차 CPU1 이내를 기본으로 한다. GPU 동시 학습은 금지한다. 문제별 학습 수·실측 시간을 따로 기록한다. 각 담당은 실행 전 고정 후보/fit 상한/시간 상한/출력 경계를 사전등록한다. P1/P2 초기 계획 예산은 각90분, P3는60분이며 보장 소요시간이 아니다. 대기시간과 실제 실행시간을 구분한다.

## 평가와 선택

### 사전 점검 후 확정한 범위

- P1: control/O_slow/O_regular/B_regular 네 완성정책. CPU2 동일 조건으로 baseline24 + alternative inner15 + 선택 alternative outer최대9 = 실제39~48 base fits. B의3seed는 모델3회로 센다. CPU4 기존 baseline을 새 CPU2 학습의 동등 비교로 재사용하지 않는다. B_slow는 예산상 미실험. pooled Q2/Q3/Q4를 새 계약 주평가로 사전 고정하고 옛 Q3/Q4 지표도 병기한다.
- P2: absolute Celsius MSE는 `reports/p2_objective_alignment_20260905_v2/report-source.md`에서 이미 비교됐으므로 중복 실행하지 않는다. 동일 GPU에서 C60/wd1e-4, C120/wd1e-4, C60/wd1e-3를 v5 전체8fold 첫seed로24fits 비교하고, B3 가을에서 control/선택challenger 각각 추가2seed로4fits 확인한다. 최대28fits. 전체8fold1seed와 가을3seed 앙상블을 분리 보고한다. 가을 결과로 후보 선택하므로 추가seed 확인도 독립 기간 검증이 아니며 seed 민감도 확인이다.
- P3: 기존 numeric 15backbone+8router historical은 재실행0. full numeric single700(CPU2)+multi1200(GPU0)+router1의 신규3fits; hmax591feature 유지. 별도PID 내부 replay/QA 이후에만 공식1200행 추론·전체답안 replay.
- 위 예상시간은 P1 약35~50분+QA, P2 약20~25분, P3 약10~15분이다. 과거 실측 기반 계획치이지 이번 완료시간 보장이 아니다.

- P1 주지표는 고정 평가키의 pooled TP/FP/FN F1. P2는 지원되는 가을 주평가의 absolute Celsius sqrt(SSE/n), 전체/결측 구간 별도. P3는 pooled metre sqrt(SSE/n).
- 특정 Q·정점·층·리드 악화, bootstrap CI, anchor 변화는 별도 위험이다. 평균 개선 후보를 그것만으로 자동 탈락시키지 않는다.
- 평가 구간의 산술평균으로 pooled 지표를 대체하지 않는다. 기존 평가 support를 좋은 결과가 나오도록 이동/삭제하지 않는다.
- 탐색의 학습량·임계값 선택은 inner에 한정하고 outer 비교와 구분한다. 이미 노출된 역사 검증은 retrospective이며 fresh 성능으로 표현하지 않는다.
- 내부 F1/RMSE를 공식 기대 점수로 역산하지 않는다. 실제 공식 점수는 해당 CSV SHA에만 붙인다.

## 보존 및 검증

기존 P1 b2f17 제출 대기본 및9031 fallback, P2 46d194, P3 6bfa23를 덮어쓰지 않는다. 기존 frozen source/config/model/lock/receipt와 dirty worktree를 수정하거나 재시작하지 않는다. 기존 src/p1_qc, p2_restore, p3_wave와 공식 최종 패키지도 수정하지 않는다. 새 실험은 새 경로에만 작성한다.

배포 데이터만 사용한다. hidden truth/외부 관측/과거 답안 입력/수동 행 patch/리더보드 역산 계수는0. 학습 실행 전 작은 synthetic compatibility/누출/정렬 테스트와 focused Ruff를 수행한다. 결과는 독립 수치 검산과 저장 모델 새 PID replay로 확인한다. 공식 입력은 해당 후보 내부 QA 이후 허용된 materialization에서만 접근한다.

P3는 기존 OOF를 사용할 수 있어 같은 historical fits를 반복하지 않는다. 이 reuse는 두 번째 scratch 전체 재현 증명이 아니다. full 모델과 새 답안은 고유한 학습·입력·hash receipt를 갖는다. portable whole-cold 제한은 별도 기록한다.

## 제출과 후속

일반 제출은 이미 승인됐으나 Chrome `Debugger unattached`로 현재 막혀 있다. 차단된 브라우저 재시작을 우회하지 않는다. 학습과 브라우저 대기를 분리하며, 연결 복구 후 현재 기회·기한·중복을 확인하고 검증된 정확한 파일만 제출한다. 기존 p1-28-9 heartbeat는 기존 후보 제출 담당이며 이 새 실험 완료를 대신 증명하지 않는다.

이번 작업은 commit/push/최종 모델 잠금을 하지 않는다. root는 후보별 성과·실패·추가 fit·runtime·QA·CSV 준비 여부와 남은 작업을 통합한다.
