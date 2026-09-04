# P2 DTW Cycle 1 v2r3 독립 사전 QA

## 결론

**NO_GO — P0 0건, P1 1건, P2 0건입니다.** v2r3는 기존 9개 결함을 모두 실질적으로 수정했고, 과학 모듈도 v2r2와 수치 코드가 동일합니다. 그러나 의미 검증을 마친 seal과 authorization·QA가 가리키는 seal을 하나의 동일한 byte snapshot으로 묶지 않아, 동시 파일 교체 시 승인 lineage와 실제 worker lineage가 갈릴 수 있습니다. 현 authorization은 계속 `false`여야 합니다.

이번 감리에서는 fit, 실제 numerical materialization, 실제 p100 resolve/stat/open/hash/parse, 공식 test/sample/submission/candidate 접근, CSV 생성 및 업로드를 전혀 수행하지 않았습니다.

## P1 차단 결함 — seal/auth 실행 lineage TOCTOU

worker는 먼저 `_verify_seal()`에서 `SEAL_PATH`의 seal A를 읽어 bundle, 13개 transitive pin, chronology와 predecessor 의미를 검증합니다. 이어 `_authorization_state()`는 이미 검증한 A의 digest를 받지 않고 `SEAL_PATH`를 다시 hash하여 authorization 및 QA의 seal hash를 확인합니다. 이후 numerical module은 처음 보관한 seal A의 module pin으로 로드됩니다.

따라서 다음 교체 순서가 가능합니다.

1. `_verify_seal()` 동안 valid seal A와 A의 정적 파일을 제시합니다.
2. 반환 직후 path를 seal B로 교체하고 B를 pin한 authorization·QA를 제시합니다.
3. `_authorization_state()`는 B hash/QA를 통과시키지만 B의 seal 의미를 그 호출에서 다시 검증하지 않습니다.
4. module path를 A bytes로 복원하면 worker는 stale in-memory seal A pin으로 A module을 실행합니다.

즉 QA는 B를 승인했는데 실제 입력·module 선택은 A가 하는 authorization-lineage 우회가 됩니다. authorization JSON, QA JSON, module이 각각 held snapshot을 쓰도록 개선된 것은 맞지만, 이 세 경계를 연결하는 seal identity가 held snapshot이 아닙니다.

관련 위치는 runner의 `_verify_design` 183–185행, `_verify_seal` 426–428행, `_authorization_state` 578/636/646행, preflight·claim·manifest의 seal 재hash 767/1211/1404행, worker/module 경계 1478–1512행입니다.

필수 수정:

- design과 seal을 각각 한 번만 열어 같은 bytes로 digest 검증과 JSON parse를 수행합니다.
- `_verify_seal()`이 parsed seal과 그 exact digest를 함께 반환합니다.
- `_authorization_state()`는 이 digest를 인자로 받아 authorization, QA pin, parsed QA report의 seal hash를 모두 같은 값과 비교합니다. 내부에서 `SEAL_PATH`를 다시 hash하지 않습니다.
- 같은 digest를 preflight, permanent claim, worker validation, result manifest와 terminal publication까지 전달합니다. 검증 후 `_sha256(SEAL_PATH)` 재호출을 제거합니다.
- digest 확인 뒤 path를 다시 여는 execution JSON도 같은 held-snapshot 원칙으로 바꿉니다.
- `_verify_seal`과 authorization 사이 seal bytes를 바꾸는 synthetic 적대 테스트를 추가하고 claim/materialization/score 전에 fail-closed되는지 검증합니다.
- v2r3는 변경하지 말고 prospective successor와 truthful fresh seal을 만듭니다.

## 기존 9개 결함 종결 확인

- **P100 keyed binding:** 78,156행 적대적 순열에서 구 방식의 52,104개 mismatch를 재현했고, 새 `fold, station, UTC-ns time, layer` one-to-one merge는 순서 불변으로 통과했습니다.
- **Windows publication:** 실제 Windows host의 production flush 경로로 임시 synthetic publication이 terminal `SUCCESS`까지 도달했고, terminal hardlink가 publication 내부 마지막 repository I/O였습니다.
- **QA receipt:** missing report 및 잘못된 bytes/hash/verdict/design/seal lineage를 모두 거부했고, QA JSON은 verified held bytes로 parse합니다.
- **module/transitive lineage:** module은 동일 held bytes로 hash·compile·exec되며 package init을 거치지 않습니다. design과 seal의 transitive pin 13개가 일치하고 predecessor terminal failure 의미도 검사합니다.
- **P100 pre-gate firewall:** durable exact slot-19 `RESEARCH_GO` 전 resolve/stat/open/hash/parse는 0입니다.
- **terminal-last/fault:** commit-ready와 모든 검증·durability 작업이 terminal link보다 먼저 끝나며 fault injection은 fail-closed입니다.
- **용어·embargo:** prediction day, trajectory start/end가 구분되고 7일 embargo 수치는 유지됩니다.
- **portable observations:** `P2_DATA_DIR/observations.csv`만 허용하며 개인 절대 fallback, 열거, parent escape, 공식-token root를 차단합니다. 기존 49,058,719-byte SHA-256 pin을 통과했습니다.
- **실제 chronology:** aware Asia/Seoul 시간과 pinned filesystem mtime이 design ≤ implementation ≤ tests ≤ seal ≤ preflight를 만족했고 미래 시각은 0개입니다.

## 독립 검증 결과

- Command-scoped `P2_DATA_DIR` focused pytest: **63 collected, 62 passed, 1 skipped, 0 failed**
- Skip 1건: 현재 Windows 계정이 synthetic symlink 생성을 허용하지 않음. parent/basename/official-token/exact-pin guard는 직접 통과
- Ruff: **PASS**
- 기본 preflight 2회: exit 0, stdout byte-identical, 공통 SHA-256 `5cc555e233fc16254c86ffb865fcd9c8593114a064c24814a2e02b9449d63c8f`
- 두 preflight 전후 정적 bytes/hash/mtime 및 control state: **동일**
- 상태: `NOT_AUTHORIZED_PENDING_INDEPENDENT_QA`, read-only, authorization false
- operation counters: 전부 **0**
- 실제 p100 filesystem access: **0**
- 실제 experiment namespace: claim/journal/OOB/final/staging 모두 **clean**
- budget: inner 18 + exact 1 + conditional p100 3 = 최대 22, fit 0, 단일 attempt, 자동·결과기반 재실행 0

## 다음 승인 조건

seal identity를 검증 시작부터 publication까지 하나의 held digest로 연결한 successor를 새로 봉인하고, 동일한 zero-fit 독립 QA를 다시 통과해야 합니다. 그 전에는 numerical execution을 승인하면 안 됩니다.
