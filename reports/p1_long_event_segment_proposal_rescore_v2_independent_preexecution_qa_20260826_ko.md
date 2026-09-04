# P1 Cycle 1 r4 독립 사전 QA

## 결론

`NO_GO`입니다. 심각도는 **P0 2건 / P1 3건 / P2 1건**입니다.

r4는 과거 v1의 수치 worker stub과 r3의 support sentinel 문제를 실제로 고쳤습니다. 9개 anchor fit + 54개 inner segment fit + 9개 outer segment fit, 총 72 fit과 21 materialization도 코드와 journal에 구현돼 있습니다. 그러나 승인 파일을 만든 뒤에도 실행이 반드시 막히는 서로 독립적인 P0가 두 건 남아 있어 r4에는 authorization을 발급할 수 없습니다.

## P0 — 승인 후에도 실행 불가능

### 1. runner → seal → authorization → runner 순환

runner는 `AUTHORIZATION_SHA256`이 64자리 hex가 아니면 실행을 거부합니다. 따라서 실제 authorization digest를 runner literal에 넣어야 합니다. 그런데 그 한 줄을 바꾸면 raw runner SHA가 바뀌고, r4 seal은 현재 placeholder runner의 raw SHA `e210aac5...c6a9b3`를 고정한 채 현재 runner raw SHA와의 일치를 요구합니다.

즉 실제 digest를 넣은 runner는 r4 seal을 통과할 수 없습니다. 새 seal을 만들더라도 authorization이 그 seal을 pin하고, 그 authorization digest를 다시 runner에 넣어야 하므로 순환이 반복됩니다.

수정은 실제 authorization digest를 코드에 내장하지 않는 것입니다. immutable runner를 유지하고 환경에서 받은 digest로 held-open authorization을 검증한 뒤 `runner → seal → 독립 QA → authorization`의 단방향 체인을 만들어야 합니다.

### 2. 승인 전후 preflight digest가 필연적으로 달라짐

sealed preflight digest에는 다음 mutable 상태가 들어갑니다.

- actual authorization 파일 존재 여부
- placeholder 또는 실제 authorization digest
- execute authorized 여부

r4 seal은 `(false, placeholder, false)` 상태의 verification SHA `92178d55...d5f6b`를 pin했습니다. 실제 승인이 생기면 `(true, 64-hex, true)`가 되므로 digest가 반드시 바뀝니다. 하지만 execute 경로는 새 digest가 r4 seal의 옛 digest와 정확히 같아야 한다고 요구합니다. 첫 번째 P0만 고쳐도 이 비교에서 다시 차단됩니다.

authorization 상태를 immutable scientific/readiness digest 밖으로 분리하고 별도 인증 receipt로 다뤄야 합니다.

## P1 — 권한·worker·label firewall 결함

### 1. cross-seal TOCTOU

authorization 검증은 held-read한 seal A를 QA lineage와 함께 인증합니다. 그 직후 parent와 worker는 같은 경로를 다시 열어 seal B를 검증하면서 A의 digest와 같다는 조건을 주지 않습니다. 경로 교체가 일어나면 authorization/QA가 인증한 seal과 실행 code map을 공급하는 seal이 달라질 수 있습니다.

held seal과 digest를 끝까지 전달하거나 두 번째 검증에 A digest를 명시적으로 강제해야 합니다.

### 2. worker snapshot inventory가 자기 자신을 인증함

worker는 snapshot manifest의 `files`가 비어 있지 않은지만 보고, manifest가 스스로 선언한 hash로 그 파일들을 확인합니다. runner가 소유한 exact 필수 inventory와 hardcoded digest 집합을 요구하지 않습니다. 이후 legacy config만 hardcoded hash로 읽고 snapshot legacy runner는 별도 hardcoded hash 검증 없이 import합니다.

따라서 crafted hidden-worker manifest 또는 검증/load 경계 교체로 미고정 legacy code가 claim 전에 실행될 수 있습니다. exact inventory equality와 executable held-byte load 또는 load 직전·직후 digest 검증이 필요합니다.

### 3. outer truth audit가 freeze 전에 실행됨

각 outer fold에서 `_accepted_interval_audit`가 all-fold `record_outer_freeze` 전에 호출됩니다. 이 함수는 outer truth와 anomaly type으로 proposal target을 만듭니다. Candidate 자체를 바꾸지는 않지만, 계약이 금지한 target-derived outer audit가 freeze 전에 발생하므로 “모든 outer prediction을 digest-journal한 뒤 한 번만 평가”한다는 firewall이 깨집니다.

freeze 전에는 target-free interval geometry와 prediction digest만 남기고, disconnected target/precision을 포함한 truth audit는 모두 freeze 이후 `_score_outer`로 옮겨야 합니다.

## P2 — 성공 출판의 좁은 모호성

worker는 모든 파일과 `0999_completed.json`을 flush한 뒤 lock unlink로 성공 commit을 끝내고, 그 다음에야 stdout의 `worker_ok` receipt를 출력합니다. unlink 직후 stdout 전에 프로세스가 종료되면 durable success는 완성돼 있지만 parent는 receipt 부재를 실패로 보고합니다. lock도 이미 없어서 실패 terminal을 기록하지 못합니다.

supervision 실패 시 lock이 없으면 completed terminal, manifest, result를 읽기 전용으로 검증하여 recovered success를 반환하는 경로가 필요합니다. Windows synthetic test로 unlink 직후 종료를 재현해야 합니다.

## 정상 확인된 항목

- exact anchor cutoffs: 2024-05-24 / 2024-08-24 / 2024-11-23
- central shelves: 2024년 6월 / 9월 / 12월
- full support positions 전체를 세 Round-B seed가 OOS predict하고 3-seed mean + 동일 postprocess 적용
- r3의 외곽 `p=1, prediction=0` sentinel 제거
- original `event_day_balanced_binary_lgbm` seeds, event/day weight, 80-feature encoder, parameters, postprocess 유지
- 같은 station·같은 time·다른 layer peer만 사용하며 G-ORS는 neutral fallback
- 24/72, 48/168, 24/72/168 bounded context와 gap/station/layer/surface 격리
- segment LightGBM parameters와 세 seed prospective 고정
- 정확히 72 fit / 21 materialization 순서와 상한 journal enforcement
- paired bootstrap와 기존 research/submission-research-only/preferred/stretch gates 연결
- r4 seal의 21개 transitive project hash와 직접 pin 불일치 0건

## 정적 검증 결과

- parent 환경, `P1_DATA_DIR` 미설정: `17 passed, 1 skipped`
- command-scoped nonofficial P1 data 경로: `18 passed`
- Ruff: `All checks passed!`
- read-only preflight 2회 stdout byte-identical
- 두 preflight verification SHA: `92178d551730af7f9910ef7d7a76f63ecef972ac06b83089411386d6597d5f6b`
- static 파일과 namespace 14개 항목의 preflight 전후 bytes/mtime/SHA: 동일
- 현재 namespace: 네 개의 immutable seal만 존재
- actual authorization, claim, fit, materialization, outer score, candidate: 모두 0
- 공식 test/sample/submission/candidate/P3 접근: 0

## 다음 조치

r4는 그대로 `NO_GO`로 보존해야 합니다. 실제 authorization을 만들거나 실행하면 안 됩니다. successor에서 P0/P1을 모두 수정하고, authorization transition, seal swap, crafted worker manifest, pre-freeze truth sentinel, post-unlink success recovery 테스트를 추가한 뒤 새 seal과 새 독립 QA를 받아야 합니다.
