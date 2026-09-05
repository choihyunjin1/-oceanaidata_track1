# 다음 AI 인수인계 — 현재 상태와 근거부터

## 시작

[AGENTS.md](AGENTS.md)를 따라 운영진 규정과 **담당 문제만** 읽습니다.
현재 결과는 [2026-09-06 체크포인트](docs/RESEARCH_CHECKPOINT_20260906.md)부터 읽습니다.
9월 5일 계획과 옛 승인 표시는 이력이며, 현재 실행 권한이나 최신 후보 지정으로 간주하지 않습니다.
작업 디렉터리·브랜치·HEAD·dirty 상태는 실시간으로 확인하고, 과거 문서의 commit을 현재 상태라고 가정하지 않습니다.

## 단일 근거 지도

| 확인할 것 | 근거 |
|---|---|
| 허용 데이터·사전학습·점수 사용 | [운영진 규정](00_ORGANIZER_DATA_POLICY.md) |
| 재생성 기준 답안 SHA·공식 점수 | [P1 9월 6일 receipt](reports/p1_regenerated_baseline_official_submission_20260906_v1/receipt.json); [P2·P3 9월 5일 receipt](reports/official_score_repair_submissions_20260905_v1/receipt.json) |
| source-only 패키지·재구축 결과 | [사용 안내](docs/ocean_v2_codex/PORTABLE_PACKAGE_HANDOFF_20260906.md), [cleanroom 기록](docs/ocean_v2_codex/CLEANROOM_RESULT.md) |
| 최신 P1 후보 | [bracket 후보 안내](docs/ocean_v2_codex/P1_BRACKET_CANDIDATE_HANDOFF_20260906.md): 내부 선택 F1 +0.005550642, 로컬 답안 QA 완료, 공식 미채점 |
| P2·P3 실패와 미평가 구분 | [체크포인트](docs/RESEARCH_CHECKPOINT_20260906.md): P2 copula 기술 실패, P3 numeric lead 자원 중단은 과학적 NO_GO가 아님 |
| 반복 실패의 원인 | [자기감사](reports/research_process_self_audit_20260831_v1/report-source.md) — 외부자료 허용 등 과거 판단은 현재 규정으로 무효 |
| 검증 반복 방지 | [개발 루프](docs/AGENT_WORKFLOW.md) |
| 사용자 제공 독립 연구·ocean_v2 사양 | [계약 변경 기록](docs/ocean_v2_codex/CONTRACT.md), [초기 검토](docs/ocean_v2_codex/REVIEW_NOTES_20260905.md): 제안과 실제 실행 계약을 구분 |
| 포털 절차·잠금 경고 | [실행서](docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md) — 역사적 후보표는 재승인 아님 |

## 후보를 혼동하지 말 것

- P1 옛 0.833548은 MS-TCN/router/GI 포함 파일의 점수입니다. 재생성 기준 `5971e145…128a`는 0.777749이며 새 bracket `9031c84e…d93a`는 공식 미채점입니다.
- 새 P1 후보는 동일한 final-inner 선택 절차로 B107 단독/threshold 0.1을 선택했습니다. 내부 수치는 독립 holdout 또는 예상 공식 점수가 아닙니다.
- 새 P1 runner의 3,600초 workflow 제한 때문에 완료 폴더에서 나중에 추론을 다시 호출하면 차단될 수 있습니다. 시계·lock을 우회하지 말고 후보 안내의 새 폴더 재생성 절차를 따릅니다.
- P2 bin17와 P3 refined-public alpha 계보는 9월 2일 규정으로 재적합 대상입니다.
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
