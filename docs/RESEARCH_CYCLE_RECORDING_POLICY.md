# 연구·정찰 사이클 기록 규칙

적용일: 2026-08-29 KST

앞으로 P1·P2·P3 정찰 또는 연구 사이클은 성공 여부와 무관하게 다음 항목을 작은 Markdown/JSON 산출물로 남긴다.

1. 기준 Git commit과 dirty-worktree 상태
2. 가설과 기존 종료축 중복 감사
3. 데이터·누출·제출 접근 경계
4. 고정 실행 계약, seed, split, purge/embargo
5. 전체·fold·station·layer·lead별 집계 결과
6. 사전 고정 gate와 최종 판정
7. 포함/제외 파일 및 재현 명령
8. 다음에 남은 정보축과 폐쇄한 계열

각 완료 사이클의 커밋 메시지 본문에는 다음을 간략히 기록한다.

- 문제와 experiment ID
- 핵심 집계 수치 또는 preflight 판정
- `GO`, `NO_GO`, `BLOCKED`, `RESEARCH_ONLY` 중 하나의 상태
- 공식 test/sample/submission 접근·CSV 생성·업로드 여부
- 관련 보고서 경로

원본 관측 행, 공식 test 값, 제출 CSV, 모델·checkpoint, 캐시 및 대용량 산출물은 커밋하지 않는다. 실패 결과도 재실행 방지와 닫힌 축 추적을 위해 집계 보고서로 보존한다.
