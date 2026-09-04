# 연구·정찰 사이클 기록 규칙

적용일: 2026-08-29 KST
최상위 규정 갱신: 2026-09-01 KST

모든 사이클은 가장 먼저 `00_ORGANIZER_DATA_POLICY.md`와
`configs/compliance/organizer_data_policy_20260901.json`을 읽고 active policy SHA를
기록한다. 배포 데이터 외 관측·재분석·예보 접근은 0이어야 하며, 외부 prediction을
상속한 계보도 사용할 수 없다. 사전학습 가중치는 `scratch`, `synthetic_only_exception`,
`forbidden_or_unknown` 중 하나로 분류하고, 합성-only 예외이면 운영진의 네 조건을
각각 증명한다.

앞으로 P1·P2·P3 정찰 또는 연구 사이클은 성공 여부와 무관하게 다음 항목을 작은 Markdown/JSON 산출물로 남긴다.

1. 기준 Git commit과 dirty-worktree 상태
2. 가설과 기존 종료축 중복 감사
3. 데이터·누출·제출 접근 경계
4. 고정 실행 계약, seed, split, purge/embargo
5. 전체·fold·station·layer·lead별 집계 결과
6. 사전 고정 gate와 최종 판정
7. 포함/제외 파일 및 재현 명령
8. 다음에 남은 정보축과 폐쇄한 계열
9. active organizer policy SHA, 배포 데이터 allowlist 및 외부자료 접근 0
10. 모델 초기화 방식과 pretrained weight provenance 판정

각 완료 사이클의 커밋 메시지 본문에는 다음을 간략히 기록한다.

- 문제와 experiment ID
- 핵심 집계 수치 또는 preflight 판정
- `GO`, `NO_GO`, `BLOCKED`, `RESEARCH_ONLY` 중 하나의 상태
- 공식 test/sample/submission 접근·CSV 생성·업로드 여부
- 관련 보고서 경로

원본 관측 행, 공식 test 값, 제출 CSV, 모델·checkpoint, 캐시 및 대용량 산출물은 커밋하지 않는다. 실패 결과도 재실행 방지와 닫힌 축 추적을 위해 집계 보고서로 보존한다.
