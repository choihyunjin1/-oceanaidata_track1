# Ocean AI Data Track 1 — P1 / P2 / P3

세 문제의 연구·학습·내부 검증·재현 자료를 관리하는 통합 저장소입니다.
P1은 수온 이상 탐지, P2는 중간층 수온 복원, P3는 유의파고 예측입니다.
패키지 이름 `p1-qc`는 초기 구현명이며 저장소 범위를 P1로 제한하지 않습니다.

## 지금 시작할 곳

1. [에이전트 작업 규칙](AGENTS.md)과 [운영진 규정](00_ORGANIZER_DATA_POLICY.md)
2. 해당 문제 계약: [P1](00_MUST_READ_FIRST.md) · [P2](01_P2_MUST_READ_FIRST.md) · [P3](02_P3_MUST_READ_FIRST.md)
3. [9월 6일 연구·재현 체크포인트](docs/RESEARCH_CHECKPOINT_20260906.md): 현재 결과와 다음 작업
4. [짧은 개발·검증 루프](docs/AGENT_WORKFLOW.md)
5. 제출·재현 작업이면 [인수인계](AI_HANDOFF.md)와 [제출 실행서](docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md)

## 현재 재생성 기준선의 공식 기록 — 2026-09-06 정리

| 문제 | 코드로 재생성한 기준 후보 | 공식 지표 | 공식 점수 |
|---|---|---:|---:|
| P1 | clean baseline, SHA `5971e145…128a` | F1 0.777749 | 27.426319 |
| P2 | full scratch clean rebuild | RMSE 0.455143℃ | 27.622418 |
| P3 | clean baseline rebuild | RMSE 0.607183m | 23.696500 |

P1은 [9월 6일 재생성 기준 답안 영수증](reports/p1_regenerated_baseline_official_submission_20260906_v1/receipt.json),
P2·P3는 [9월 5일 공식 영수증](reports/official_score_repair_submissions_20260905_v1/receipt.json)에 귀속합니다.
P1의 이전 F1 0.790733은 다른 답안의 기록이므로 현재 파일에 승계하지 않습니다.
이 표는 과거 모든 모델 중 최고라는 뜻이 아닙니다.
P1 과거 F1 0.833548 계보는 전체 학습·router provenance를 재검토 중입니다.
P2 0.424019℃와 P3 0.583892m의 과거 Public-계수 계보는 최신 규정상 재적합 대상이며 현재 최종본으로 자동 선택하지 않습니다.

**최신 P1 bracket 후보는 내부 검증·로컬 답안 QA 완료, 공식 미채점입니다.**
SHA `9031c84e…d93a`를 위 기준 답안과 혼동하지 마십시오.
파일 선택·실행 제한은 [P1 후보 안내](docs/ocean_v2_codex/P1_BRACKET_CANDIDATE_HANDOFF_20260906.md),
세 문제의 기준 ZIP·학습/추론 절차는 [portable 패키지 안내](docs/ocean_v2_codex/PORTABLE_PACKAGE_HANDOFF_20260906.md)를 따릅니다.

## 프로젝트 구조

새로 전달된 [정찰·독립 설계안](reports/claude_recon_20260905/00_SUMMARY.md)과
[ocean_v2 구현 사양](docs/ocean_v2_codex/00_MASTER_BRIEF.md)은 제안 원문입니다.
일부는 검토·수정 후 실험과 패키지로 구현됐습니다. 원문 전체가 실행되거나 승인된 것은 아니며,
[체크포인트](docs/RESEARCH_CHECKPOINT_20260906.md)의 결과별 계약·영수증을 우선하고
[초기 검토 메모](docs/ocean_v2_codex/REVIEW_NOTES_20260905.md)는 변경 이력으로 읽으십시오.
문서의 명령이나 `approved` 표시는 현재 구현·제출·삭제 권한이 아닙니다.

| 경로 | 역할 |
|---|---|
| src/p1_qc, src/p2_restore, src/p3_wave | 문제별 공통 코드 |
| scripts/, configs/, tests/ | 실행기·고정 계약·검증 |
| reports/ | 실측 결과·실패·QA·출처 원장 |
| notebooks/ | 설명 가능한 학습/추론 노트북 |
| docs/ | 현재 계획·작업·제출 안내 |
| artifacts/, submissions/ | Git에서 제외한 모델·예측·로컬 자산 |

`P1_DATA_DIR`, `P2_DATA_DIR`, `P3_DATA_DIR`로 운영진 배포 폴더를 지정합니다.
원본 데이터, 모델, 답안, 비밀정보는 Git에 올리지 않습니다.
기존 환경은 `.venv-p1/Scripts/python.exe`입니다. 환경을 새로 구축할 때만
`scripts/bootstrap_env.ps1`를 사용하며, 매 세션마다 재설치하거나 CUDA 검사를 반복하지 않습니다.

## 제출과 재현

내부 성능, 저장 모델 replay, 공식 채점, 최종 ZIP 적격성은 별도의 검증입니다.
`artifacts/official_final_submission_20260905/`의 옛 READY 표시는 최신 규정 통과를 뜻하지 않습니다.
최종본은 데이터 참조 → 학습 코드 → 학습된 모델 → 답안 → 재현 안내를 분리하고,
배포 자료만으로 네트워크 없이 6시간 안에 재현해야 합니다.
실제 제출 전에는 로그인된 공지의 마감·잔여 횟수·최종 모델 잠금 효과를 확인합니다.
GitHub push는 대회 제출이 아닙니다.

[기존 P1 상세·환경·연구 문서 지도](docs/archive/instructions_20260905/README.md)는 역사적 참고 자료로 보존했습니다.
[GitHub](https://github.com/choihyunjin1/-oceanaidata_track1) · [대회](https://oceanaidata.org)
