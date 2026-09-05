# 최종 패키지 cleanroom — 로컬 독립 검증 진행

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
