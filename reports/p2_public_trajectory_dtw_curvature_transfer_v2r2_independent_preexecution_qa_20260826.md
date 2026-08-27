# P2 DTW Cycle 1 v2r2 독립 사전 QA

## 결론

**NO_GO — P0 0건, P1 2건, P2 0건입니다.** 이전 v2 독립 QA의 7개 지적은 코드와 54개 집중 테스트에서 모두 재현·종결됐습니다. 그러나 현 v2r2는 개인 절대 입력 경로와 불가능한 미래 seal 시각 때문에 실제 실행 승인에 사용할 수 없습니다. authorization은 계속 `false`여야 합니다.

이번 감리에서는 수치 materialization, 실제 p100 접근, 공식 test/sample/submission/candidate 접근, CSV 생성 및 업로드를 전혀 수행하지 않았습니다.

## 신규 차단 결함

### P1-NEW-01 — 개인 절대 observations 경로 하드코딩

`scripts/run_p2_public_trajectory_dtw_curvature_transfer_v2r2.py:57`의 `OBSERVATIONS_PATH`가 `C:/Users/cedis/Downloads/.../observations.csv`로 고정돼 있습니다. 드라이브 문자, 사용자명, 한국어 개인 소스 경로를 Python에 직접 넣지 말고 P2 입력을 `P2_DATA_DIR`로 해석하라는 `AGENTS.md:53-54` 규칙을 위반합니다. 같은 절대 경로가 seal에도 복제돼 있어 현재 패키지는 다른 정상 workspace에서 재현되지 않습니다.

필수 수정은 다음과 같습니다.

- `observations.csv`를 `P2_DATA_DIR`에서 해석합니다.
- 로컬 편의용 repository search fallback을 둘 경우 검색 결과가 0개 또는 2개 이상이면 반드시 실패시킵니다.
- observations의 기존 bytes/SHA-256 내용 핀은 입력이 실제로 바뀌지 않는 한 유지합니다.
- seal의 source locator도 이식 가능한 표현으로 바꾸고 runner/tests/seal/auth-false/closure 증거를 원자적으로 다시 고정합니다.
- 수정 후 기본 preflight에서 모든 작업 카운터 0과 p100 접근 0을 다시 확인합니다.

### P1-NEW-02 — seal 생성 시각이 실제보다 미래

`preexecution_seal.json:5`는 `created_at_kst=2026-08-26T09:15:00+09:00`라고 선언합니다. 하지만 파일 생성 시각은 `08:11:05`, 마지막 수정 시각은 `08:12:51`, 독립 관측 시각도 `08:18:24`였습니다. 같은 `+09:00` 기준에서 아직 오지 않은 시각을 이미 생성된 seal이 주장하므로 실제 provenance chronology로 인정할 수 없습니다.

필수 수정은 다음과 같습니다.

- 모든 최종 sealed input을 쓴 뒤 실제 현재 KST로 새 seal을 생성합니다.
- 영향받은 bytes/SHA-256과 bundle/transitive pins를 다시 계산합니다.
- 새 seal hash를 가리키도록 authorization false template을 다시 만들되 `authorized=false`, QA receipt null 상태를 유지합니다.
- chronology, seal, authorization, 기본 preflight와 clean namespace를 새 독립 QA에서 다시 확인합니다.

## 이전 v2 지적 7건의 종결 확인

- **P0-01 p100 canonical binding:** 78,156행 무작위·적대적 순열 테스트가 구 positional 방식의 52,104/78,156 불일치를 재현했고, 새 방식은 `fold, station, UTC-ns time, layer` one-to-one keyed merge에서 순서 불변으로 통과했습니다. 후보 점수도 동일 키로 재검증합니다.
- **P0-02 Windows 성공 publication:** 실제 Windows host에서 production `CreateFileW`/`FlushFileBuffers` 경로를 사용한 임시 synthetic publication이 terminal `SUCCESS`까지 도달했습니다. publication 내부 마지막 repository I/O는 terminal-success hardlink였고 이후 I/O 카운트는 0이었습니다.
- **P1-01 QA receipt:** missing report와 잘못된 bytes/hash/verdict/design lineage/seal lineage를 모두 거부했습니다. 실제 safe path의 pinned·parsed `PASS` receipt만 승인 검증을 통과했습니다. 자기 승인 순환은 발견되지 않았습니다.
- **P1-02 transitive seal:** numerical module은 package `__init__` 없이 pin된 파일을 직접 import합니다. design과 seal의 transitive inventory 14개가 정확히 일치하며 predecessor claim/journal은 hash뿐 아니라 terminal failure 의미까지 검증합니다.
- **P1-03 exact-GO 전 p100:** 기본 preflight에서 p100 filesystem access는 0입니다. slot 19의 durable exact `RESEARCH_GO` 전에는 path resolve/stat/open/hash/parse가 실행되지 않습니다.
- **P1-04 terminal-last:** result/manifest, 검증, 내구성 barrier, materialization accounting, durable commit-ready, terminal bytes/hash가 terminal hardlink보다 먼저 완료됩니다. fault injection도 success와 failure 충돌 없이 fail-closed입니다.
- **P2-01 용어:** `prediction_day_start`, `trajectory_start`, `trajectory_end`가 분리됐고 7일 embargo는 수치 변경 없이 `trajectory_start` 기준으로 검사됩니다.

## 검증 결과

- Focused pytest: **54 passed, 0 failed**
- Ruff: **PASS**
- 기본 runner preflight: `NOT_AUTHORIZED_PENDING_INDEPENDENT_QA`, read-only, authorization false
- 작업 카운터: attempts/fit/materializations/scores/candidate/uploads/official reads 모두 **0**
- 실제 p100 resolve/stat/open/hash/parse: **0**
- 실제 experiment namespace: claim/journal/OOB/final/staging 모두 **없음**
- 22-slot ceiling: inner 18 + exact 1 + conditional p100 3, fit 0, 단일 attempt, 자동·결과기반 재실행 0으로 고정됨

## 승인 조건

두 P1을 수정한 새 버전에서 truthful fresh seal을 만들고, authorization을 계속 false로 둔 상태에서 동일한 zero-fit 독립 QA를 다시 통과해야 합니다. 그 전에는 numerical execution을 승인하면 안 됩니다.
