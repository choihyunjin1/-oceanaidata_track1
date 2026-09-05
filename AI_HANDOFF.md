# 다음 AI 인수인계 — 현재 상태와 근거부터

## 시작

[AGENTS.md](AGENTS.md)를 따라 운영진 규정과 **담당 문제만** 읽습니다.
현재 계획은 [2026-09-05 v2](docs/SCORE_IMPROVEMENT_PLAN_20260905_V2.md)입니다.
작업 디렉터리·브랜치·HEAD·dirty 상태는 실시간으로 확인하고, 과거 문서의 commit을 현재 상태라고 가정하지 않습니다.

## 단일 근거 지도

| 확인할 것 | 근거 |
|---|---|
| 허용 데이터·사전학습·점수 사용 | [운영진 규정](00_ORGANIZER_DATA_POLICY.md) |
| 최신 공식 답안 SHA·점수 | [9월 5일22:10 P1 추가비교](reports/conditional_validation_and_information_submission_20260905_v3/official-receipt.json); 기존당일기준은 [19:28–29 receipt](reports/official_score_repair_submissions_20260905_v1/receipt.json) |
| source-only 재구축 결과 | [실행 보고서](reports/parallel_score_repair_20260905_v1/report-source.md) |
| 최신 검증 결론 | [v3 검증·공식비교 완료](reports/conditional_validation_and_information_submission_20260905_v3/report-source.md): P1공식악화/P2첫seed개선이3seed에서미재현/P3추가제출0; 이전 [v2결과](reports/parallel_score_improvement_20260905_v2/report-source.md) |
| 반복 실패의 원인 | [자기감사](reports/research_process_self_audit_20260831_v1/report-source.md) — 외부자료 허용 등 과거 판단은 현재 규정으로 무효 |
| 검증 반복 방지 | [개발 루프](docs/AGENT_WORKFLOW.md) |
| 사용자 제공 독립 연구·ocean_v2 사양 | [실행 전 검토](docs/ocean_v2_codex/REVIEW_NOTES_20260905.md): 원문 보존/미실행, 역산근거·분할·패키징 등 정정 필요 |
| 포털 절차·잠금 경고 | [실행서](docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md) — 역사적 후보표는 재승인 아님 |

## 후보를 혼동하지 말 것

- P1 옛 0.833548은 MS-TCN/router/GI 포함 파일의 점수입니다. 오늘 두-tree 파일의 점수가 아닙니다.
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
