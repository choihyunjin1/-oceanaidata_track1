# 2026-08-30 원본 데이터 돌파구 사이클 커밋 manifest

## 포함

- 실험 계약 5개: P1 Stage-0, P2 Stage-0/Stage-1, P3 Stage-0/Stage-1 config
- 재현 코드 9개: 원본 구조 audit, Stage-0/Stage-1 독립 QA, 네 preflight/runner,
  두 Stage-1 runner, P3 masked-SSL 구현 module
- 계약 테스트 6개
- 작은 연구 근거 7개: dataset audit, Stage-0/Stage-1 QA JSON, gap matrix,
  claim-source ledger, 기술 보고서, 본 manifest

포함 파일은 모두 코드·설정·테스트·Markdown 또는 작은 aggregate JSON이다.
가장 큰 파일도 50KB 미만이며 자격증명, 비밀키, 사용자 절대경로가 없다.

## 제외

- `artifacts/` 아래 모든 immutable result, prediction, attempt lock, checkpoint, log
- P1/P2/P3 원본 training 데이터와 파생 대용량 데이터
- 공식 test/test_index/context/sample/baseline/score/hidden-answer 파일
- submission CSV와 그 밖의 CSV output
- cache, virtual environment, credential, token, `.env`

실행 receipt 자체는 저장소의 기존 `/artifacts/` ignore 정책을 유지해 제외했다.
대신 커밋되는 두 independent-QA JSON에 원 receipt SHA256, 재계산 gate,
fit/runtime, 현재 코드·설정 hash와 고정 read surface를 보존했다.

## 검증

- focused pytest: `42 passed`
- Ruff: PASS
- Stage-0 independent QA: `PASS_WITH_DISCLOSED_P1_POST_RUN_HARDENING`
- Stage-1 independent QA: `PASS_BOTH_TERMINAL_NO_GO_WITH_P2_PROVENANCE_DISCLOSURE`
- 공식 입력 접근, submission 생성, upload: 모두 0
- training row hard deletion: 모두 0
