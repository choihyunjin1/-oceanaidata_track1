# 다음 AI 인수인계 — 현재 상태와 근거부터

> **최신 상태 — 2026-09-07 20:01 KST 접수 확인:** P1 20:00 / P2 19:44 / P3 19:59, 모두 `최종 제출(모델) / 모델 · 검증 대기 / 채점중`입니다. 첨부 목록 P1 21개·P2 3개(v2)·P3 4개를 저장된 양식에서 확인했습니다. [현재 상태·재현 근거·파일 안내](docs/ocean_v2_codex/CURRENT_FINAL_STATUS_20260907.md)가 아래 역사적 보류/진행 기록보다 우선합니다. 운영진 검증 통과나 서버 파일 다운로드 SHA 검증을 뜻하지 않습니다. 재제출·삭제하지 마세요.

> 09-07 15:46 최우선 정정: [P2 portability v2·감사 답변](docs/ocean_v2_codex/P2_PORTABILITY_REPAIR_AND_AUDIT_RESPONSE_20260907.md). 새 cold/saved exact, 답안 SHA 794268f1 불변, 최종 ZIP 36b4b0e4로 교체 준비. 옛 v1 첨부를 확정하지 않습니다. Fable 재검토·사용자 재개 전 최종 확인 금지. P3는 25/36 진행 스냅샷이며 전체 cold 증거는 미완입니다.

## 최우선 갱신 — 09-07 15:12 KST / 제출 보류

**[최신 정본](docs/ocean_v2_codex/FINAL_SELECTION_REVIEW_HANDOFF_20260907.md)**부터 읽습니다. 우선 후보 P1 `57844ef2` / P2 `794268f1` / P3 `70761aff`; 답안 잔여 0/0/0. P2 새 whole cold/saved exact, P3 fresh cold 진행 중(15:12 PID26996 생존, 23/36 backbone). 사용자가 Fable 검토 전 최종 확인을 멈추라고 지시했습니다. P2 첫 버튼 뒤 확인창 OK는 누르지 않았고 접수 미확인; P1/P3 최종 클릭 없음. GitHub 정리 승인은 제출 재개 승인이 아닙니다. 아래 이전 선택과 미채점/잔여 표시는 역사적 이력입니다.

## 시작

[AGENTS.md](AGENTS.md)를 따라 운영진 규정과 **담당 문제만** 읽습니다.
현재 결과는 **[2026-09-07 최종 패키지 안내](docs/FINAL_RELEASE_20260907.md)**와 [최종 검증 기록](reports/final_release_20260907_v1/report-source.md)부터 읽습니다.
초기 릴리스의 역사적 선택은 P1 `57844ef2` / P2 `fee6118b` / P3 `ff42a6a0`였으며, 현재 첨부 선택에는 사용하지 않습니다.
P1 전체 7fit 재학습 답안이 28.909341점 파일과 정확히 일치하고 서버도 중복으로 확인했습니다.
P2 L120은 실제 TRAIN/PREDICT 노트북 전체 cold 재현을 완료했습니다.
P3 새 whole-cold 17fit는 완료했으나 답안 `56e289af`는 채점본 `ff42a6a0`와 660행이 다릅니다(최대 0.003506m). 공식 점수를 승계하지 않습니다. 채점 당시 모델 ZIP의 실제 새 추출 추론은 `ff42a6a0` exact PASS입니다. 재학습 허용오차/적격성은 운영진 확인 전까지 미확인으로 남깁니다.
이 문서 아래 09-06 bracket/기준선 표는 **이전 이력**입니다. 현재 파일 선택에는 사용하지 않습니다.
9월 5일 계획과 옛 승인 표시는 이력이며, 현재 실행 권한이나 최신 후보 지정으로 간주하지 않습니다.
작업 디렉터리·브랜치·HEAD·dirty 상태는 실시간으로 확인하고, 과거 문서의 commit을 현재 상태라고 가정하지 않습니다.

## 단일 근거 지도

| 확인할 것 | 근거 |
|---|---|
| 허용 데이터·사전학습·점수 사용 | [운영진 규정](00_ORGANIZER_DATA_POLICY.md) |
| 재생성 기준 답안 SHA·공식 점수 | [P1 9월 6일 receipt](reports/p1_regenerated_baseline_official_submission_20260906_v1/receipt.json); [P2·P3 9월 5일 receipt](reports/official_score_repair_submissions_20260905_v1/receipt.json) |
| source-only 패키지·재구축 결과 | [사용 안내](docs/ocean_v2_codex/PORTABLE_PACKAGE_HANDOFF_20260906.md), [cleanroom 기록](docs/ocean_v2_codex/CLEANROOM_RESULT.md) |
| 최신 P1 후보 | [bracket 후보 안내](docs/ocean_v2_codex/P1_BRACKET_CANDIDATE_HANDOFF_20260906.md): 09-06 06:39 공식 F1 0.785944 / 27.644124점, SHA9031 정확 연결 |
| 사용자 승인 후 후속 실행 | [전체 완료 결과](reports/remaining_work_completion_20260906_v1/report-source.md): P1/P2/P3 비교·cold·패키지 재생·공식 채점 완료. 기존 소비된 attempt 재실행 금지 |
| P2·P3 기술정정 이후 | [공백표](reports/remaining_work_completion_20260906_v1/gap-matrix.md): 옛 기술실패는 보존, 새 ID의 독립QA/평가 완료와 구분 |
| 반복 실패의 원인 | [자기감사](reports/research_process_self_audit_20260831_v1/report-source.md) — 외부자료 허용 등 과거 판단은 현재 규정으로 무효 |
| 검증 반복 방지 | [개발 루프](docs/AGENT_WORKFLOW.md) |
| 사용자 제공 독립 연구·ocean_v2 사양 | [계약 변경 기록](docs/ocean_v2_codex/CONTRACT.md), [초기 검토](docs/ocean_v2_codex/REVIEW_NOTES_20260905.md): 제안과 실제 실행 계약을 구분 |
| 포털 절차·잠금 경고 | [실행서](docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md) — 역사적 후보표는 재승인 아님 |

## 후보를 혼동하지 말 것

- P1 옛 0.833548은 MS-TCN/router/GI 포함 파일의 점수입니다. 재생성 기준 `5971e145…128a`는 0.777749이며 새 bracket `9031c84e…d93a`는 공식 0.785944입니다. [정확한 채점 영수증](reports/p1_bracket_official_submission_20260906_v1/receipt.json).
- 새 P1 후보는 동일한 final-inner 선택 절차로 B107 단독/threshold 0.1을 선택했습니다. 내부 수치는 독립 holdout 또는 예상 공식 점수가 아닙니다.
- 원 P1 runner의 3,600초 workflow 제한을 우회하지 않습니다. 별도 장기 어댑터·새 ZIP 추출 검증은 [portable v2 보고서](reports/p1_bracket_portable_20260906_v2/report-source.md)를 확인합니다.
- P1 train-fit 범위/셀 정책은 [독립 비교](reports/p1_trainfit_postpolicy_20260906_v1/report-source.md)에서 주평가 동률/악화로 비승격입니다. 9031 후보와 혼합하지 않습니다.
- P2 bin17와 P3 refined-public alpha 계보는 9월 2일 규정으로 재적합 대상입니다.
- P2 새 CPU copula는 [07:28 공식 대조](reports/p2_copula_official_submission_20260906_v1/receipt.json)에서 같은 CPU C3보다 개선(.489080→.475174℃)했지만 보존CUDA C3 .455143℃보다 나빴습니다. 기존기준 유지, CPU파일에 CUDA점수 승계 금지.
- P3 hmax제거는 내부평균 개선·독립134QA 이후 source-only whole-cold12+5fit/1611.919초와 저장ZIP새추출replay를 완료했습니다. d456 답안의 [08:08 공식 결과](reports/p3_hmax_official_submission_20260906_v1/receipt.json)는 .608184m으로 기존6bfa .607183m보다 나빠 기존기준을 유지합니다. 후보에 기준점수를 승계하지 않습니다.
- 옛 final_submission config/FORM/READY는 보존용입니다. 그 경로로 자동 refresh/upload하지 않습니다.
- 현재 모델이 내부 검증을 통과해도 공식 성적을 예측 사실로 붙이지 않습니다. 점수는 같은 SHA에만 연결합니다.

## 재개·종료

실행 중이면 progress/terminal, 프로세스와 오류만 확인합니다. 멈췄다면 terminal/lock/log로
정상 종료와 기술 실패를 구분하고 중복 학습을 시작하지 않습니다.
같은 코드 검사는 hash가 변하지 않았다면 검증 영수증을 재사용할 수 있지만,
새 모델/답안의 수치·schema·replay 검사는 별도로 수행합니다.

작업 결과에는 실험 ID, 학습 수/시간, 비교 단위, 실제 성과 또는 실패, 다음 판단, canonical 경로를 남깁니다.
commit/push와 대회 업로드는 사용자가 승인한 범위에서만 수행합니다.
데이터·모델·예측·lock은 Git에서 제외하고 기존 dirty 변경을 보존합니다.
[과거 상세 자산 지도](docs/archive/instructions_20260905/AI_HANDOFF.md)는 감사용입니다.
