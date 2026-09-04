# P2 DTW Cycle 1 v2r4 독립 사전 QA

## 결론

**PASS**입니다. 실행을 막는 결함은 `P0=0 / P1=0`이며, 실행에 영향을 주지 않는 문서 결함 `P2=1`을 투명하게 기록했습니다.

이 PASS는 수치 실행 결과가 좋다는 뜻이 아닙니다. 현재 고정된 0-fit 패키지가 정확한 권한·데이터 방화벽·22-slot·single-attempt 계약으로 한 번 실행 가능한 상태라는 뜻입니다. 실제 authorization은 이 보고서의 정확한 bytes/SHA, design SHA, seal SHA를 pin한 뒤에만 true로 바뀔 수 있습니다.

## 확인된 핵심 항목

- `P2_DATA_DIR/observations.csv`만 허용하며 개인 절대 fallback이 없음
- observations bytes/SHA와 held-byte parse가 일치
- 실제 KST 시간 및 filesystem mtime 순서가 nonfuture이고 design → implementation → tests → seal → preflight 순서를 만족
- p100은 durable exact `RESEARCH_GO` 전 path resolve/stat/open/hash/parse가 0
- p100 78,156행은 canonical station/time/layer/truth keyed merge로 결합
- 물리 fit 0회
- 18 inner + 1 exact + 조건부 p100 3 = 정확히 22 materialization slots
- single attempt, automatic rerun 0
- design, execution, seal, authorization, QA, numerical module이 각각 단일 held snapshot으로 hash 및 parse/compile됨
- seal A 검증 뒤 경로가 B로 바뀌어도 authorization, claim, worker, result, terminal lineage가 A digest를 유지
- authorization path가 capture 뒤 바뀌어도 판단은 처음 held bytes에 고정
- 실제 QA receipt는 PASS, bytes/SHA, design SHA, seal SHA가 모두 일치해야 함
- Windows terminal-last 성공 출판과 각 fault boundary 테스트 통과
- 공식 test/sample/submission/candidate 경로 접근·생성·업로드 0

## 테스트 및 preflight

- `P2_DATA_DIR` 명시 sealed focused suite: **66 passed, 1 skipped**
- skip 1건: 현재 Windows 계정에서 synthetic symlink 생성 불가. exact-parent/basename 및 official-token guard는 별도 테스트로 통과
- `P2_DATA_DIR` 미설정 parent 환경: 67개 중 64 pass, 1 skip, 2 fail. 두 실패는 observations readiness와 idempotence가 명시적 환경변수 없이 fail-closed한 예상 결과이며 fallback은 사용되지 않음
- Ruff: `All checks passed!`
- read-only preflight 2회 stdout byte-identical
- preflight stdout SHA256: `8d41ce33f19f8aadfc7524fc41c298c86c52477df57a18900ead31dd837baad1`
- 두 preflight 전·사이·후 static/control state 동일
- claim, journal, terminal failure, final namespace 모두 없음
- attempts, fits, materializations, scores, candidate, official reads, uploads 모두 0

## P2 문서 caveat

봉인된 closure matrix에는 현재 test 파일에 존재하지 않는 과거 함수명 5개가 총 7회 인용돼 있습니다. 영향받는 closure는 `V2-P1-01`, `V2-P1-02`, `V2R3-P1-NEW-01`입니다.

대응 관계는 다음과 같습니다.

- `...real_qa_receipt_and_lineage` → `...real_qa_receipt_and_snapshot_lineage`
- `...accepts_only_a_pinned_real_pass_receipt` → 여섯 corruption case를 검사하는 snapshot-lineage 테스트와 valid PASS를 수락하는 one-snapshot auth-swap 테스트의 조합
- `...bytes_swap_after_capture...` → `test_authorization_uses_one_snapshot_even_if_auth_path_swaps`
- `...loads_one_held_module_snapshot...` → `test_authorized_worker_compiles_retained_module_without_path_reopen`
- `...uses_captured_bytes_after_path_mutation` → `test_module_compilation_ignores_path_mutation_after_seal_snapshot`

이는 코드·테스트가 빠진 결함이 아니라 이름이 바뀐 뒤 matrix 문구가 따라가지 못한 문서 결함입니다. 실제 대응 테스트는 모두 sealed suite에서 통과했고, runner는 해당 문자열로 테스트를 dispatch하지 않습니다. 현재 matrix는 seal이 hash-pin하므로 수정하면 r4 전체를 재봉인해야 합니다. 따라서 r4를 변형하지 않고 이 QA caveat로 정정하며, 차기 문서 successor에서 명칭만 바로잡는 것이 적절합니다.

## 최종 판정

과학·leakage·권한·seal·p100 gate·22-slot·single-attempt·Windows 출판 blocker는 발견되지 않았습니다. 따라서 이 정확한 design/seal에 대해 독립 QA verdict는 `PASS`입니다. 수치 실행·실제 materialization·p100 접근·공식 경로 접근은 이번 QA에서 수행하지 않았습니다.
