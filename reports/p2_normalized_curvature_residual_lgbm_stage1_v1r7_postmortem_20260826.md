# P2 NCR-LGBM Stage 1 v1r7 사고 분석

## 결론

v1r7은 **모델의 과학적 실패가 아니라, 결과 집계 직전의 시간 단위 변환 결함으로 종료된 인프라 실패**입니다. 세 개의 사전등록 fit은 모두 완료됐지만, 후보 예측과 집계 지표는 worker 메모리에만 있었고 저장 전에 예외가 발생했습니다. 따라서 NCR의 GO/NO_GO를 판정할 수 없으며, 영구 claim과 3-fit lifetime ceiling 때문에 v1r7을 다시 실행하거나 내부 worker를 재진입시켜서는 안 됩니다.

정확한 예외는 다음과 같이 확정했습니다.

> `ValueError: paired day bootstrap requires at least two KST days`

봉인된 stderr의 SHA-256은 `dff94f558e5dad473dd65a0bfc1dc98a1cf08c4317dab77f244afb7eb70fa0ce`입니다. 동일 sealed 파일명·frame line·예외를 사용하는 0-fit traceback 재구성 결과도 같은 SHA-256이 나왔습니다. 단순한 유력 추정이 아니라 recorded stderr와의 byte-level 일치입니다.

## 무슨 일이 일어났나

사전등록 population은 2024년 9–10월의 26,273개 정렬 행이며 실제 OOF 시간은 다음과 같습니다.

- dtype: `datetime64[us, UTC]`
- UTC 범위: `2024-08-31 15:00:00`–`2024-10-31 14:50:00`
- KST 일수: 61

그러나 `align_exact_incumbent()`는 timestamp를 `.astype("int64")`로 바꾼 뒤 `_time_ns`라고 명명하고, 나중에 `pd.to_datetime(..., utc=True)`로 복원했습니다. 봉인 환경의 pandas 3.0.1에서는 원본 해상도가 microsecond이므로 정수도 microsecond 단위입니다. 복원 함수에 `unit`을 지정하지 않으면 이 값을 nanosecond로 해석합니다.

그 결과 정렬된 시간은 다음처럼 1/1000로 압축됐습니다.

- 잘못 복원된 UTC 범위: `1970-01-20 23:11:56.400000`–`1970-01-21 00:39:46.200000`
- 고유 timestamp: 8,779
- NaT: 0
- KST 일수: 1

`paired_day_bootstrap()`은 최소 2개 KST day를 요구하므로 정확히 해당 ValueError를 발생시켰습니다. challenger를 incumbent와 동일하게 둔 0-fit 재현에서도 같은 예외가 발생하므로, 이 실패는 후보 예측값·성능·shape에 의존하지 않습니다.

## 실행 증거

| 항목 | 확인값 |
|---|---:|
| claim SHA-256 | `22b739c869cb1e57b3f68fe5ef19bfa480db9530af20692f2d047ed16d4624d3` |
| journal SHA-256 | `ab250c745ca104d4b3439d3c51bdb1283c5723a72bf6becc59711ae900ebb38d` |
| journal events | 11 |
| physical fit reserved/completed | 3 / 3 |
| 실패 phase | `ALL_FITS_COMPLETED_BUILDING_RESULT` |
| worker return code | 1 |
| worker stdout | empty (`e3b0c442…b855`) |
| final directory | 없음 |
| staging directory | 0 |
| 저장된 모델/예측/집계 지표 | 0 / 0 / 0 |
| parent/worker process | 모두 종료 |

slot별 model fit elapsed time은 각각 0.391초, 0.406초, 0.407초였습니다. journal에는 각 slot의 reservation과 completion이 정확히 한 번씩 기록돼 있습니다.

## 정적 무결성

사고 후 다시 계산한 contract hash와 bundle hash는 claim과 정확히 일치합니다.

- contract: `5be45ad6198f55e39d3f2a32a70e4784345a75a4fcdcd9f88c9124591621209c`
- bundle: `e8c1c8d5537ac547c6daff16aa0690439a4f2f17797ca97a4624561a2d003452`
- runner raw: `ff8b4adaba4a68903b513a6b42695f055389cab813c4b446e27d457625eb33ac`
- runner normalized: `6341fc87ebc66eeab1e1b1bb1eacfec91cd8dcd47a28d9ad407d97111030737c`
- numerical module: `c5d88d9bbe90c4b2b03f135d434ed5f030b71732d50549aac17ced88516070a0`
- feature builder: `b23e19ec55120f6144e693f9da24ba78b85b6191c55de1a2889b0d26fd8d8ee7`

따라서 사고 뒤 source drift로 생긴 예외가 아닙니다. 결함은 실행 당시 봉인 코드와 pandas 3.0.1의 datetime resolution 의미가 맞지 않았던 데 있습니다.

## 왜 지표를 복구할 수 없는가

각 seed의 LightGBM 객체와 decoded prediction은 worker 메모리에만 존재했습니다. 집계 후 `WORKER_COMPLETED` journal event, stdout result JSON, staging, final publication 중 어느 것도 발생하지 않았습니다. 현재 남아 있는 것은 fit 완료 provenance뿐이며, 예측값이나 sufficient statistics가 아닙니다.

따라서 aggregate metrics나 후보 예측을 얻으려면 다시 모델을 fit해야 합니다. 이는 permanent no-rerun claim과 최대 3회 physical fit 계약을 위반합니다. 메모리에서 사라진 값을 추정하거나, 같은 seed라서 같을 것이라는 이유로 새 결과를 기존 실행의 결과로 간주해서도 안 됩니다.

## 다음 P2 행동

현재 사전등록된 두 후속 설계는 자동으로 실행할 수 없습니다.

- raw-Celsius group-balanced fallback은 terminal NCR scientific FAIL 또는 저장된 normalized-vs-Celsius mismatch를 trigger로 요구합니다. 이번 사고에는 둘 다 없습니다.
- DTW Cycle 1은 predecessor가 infrastructure failure로 끝나면 materialization 전에 정지하도록 명시돼 있습니다.

따라서 현재 상태는 `BLOCKED_PENDING_A_NEW_EXPLICIT_ZERO_FIT_SUPERSESSION_AND_AUTHORIZATION`입니다. NCR 재실행 대신, **DTW Cycle 1의 새 successor를 0-fit 준비하는 경로**를 우선 제안합니다. 기존 DTW는 model fit이 0이고 NCR과 구조적으로 독립적이기 때문입니다.

준비 단계는 다음 조건을 모두 지켜야 합니다.

1. 새 experiment ID를 사용하고 기존 DTW design 및 v1r7 claim/journal/receipt를 byte-for-byte 보존합니다.
2. 이번 incident receipt와 claim/journal hash를 pin하고 NCR 결과를 추정하지 않은 채 frozen exact incumbent를 anchor로 고정합니다.
3. predecessor/anchor branch 외에는 기존 DTW의 trajectory algorithm, 6 cells, 3 inner windows, 최대 22 materializations, metrics, gates, search budget을 그대로 유지합니다.
4. 새 runner/module/runtime/native/input/Git 상태를 data parse 전에 모두 hash-pin합니다.
5. exact surface timestamp의 raw↔aligned round trip, UTC 범위, 61 KST days, key equality를 0-fit regression으로 강제합니다. 정수 datetime은 명시적 unit 없이는 금지합니다.
6. chronology, source-before-query, key uniqueness, six-cell freeze, bootstrap, output allow-list, official-path firewall, exact-once claim/journal을 단위시험합니다.
7. 이 단계에서는 test/lint/strict preflight/독립 QA만 수행하며 claim, journal, materialization, fit, result, candidate, upload는 모두 0이어야 합니다.
8. 별도 사용자 승인 후에만 aggregate-only DTW 실행을 정확히 한 번 허용하고, 결과 기반 재실행·grid 확대는 금지합니다.

raw-Celsius 설계는 여전히 과학적으로 타당한 대안이지만 새 physical fit 3회가 필요하고 기존 trigger도 충족되지 않았습니다. 이번 인프라 사고를 scientific FAIL로 바꿔 해석해 자동 실행해서는 안 됩니다.

## 보존·금지 사항

- v1r7 재실행/refit/internal-worker 재진입: 금지
- v1r7 claim/journal 삭제·수정: 금지
- 이번 사고를 NCR scientific GO/NO_GO로 해석: 금지
- 공식 test/sample/submission/candidate 접근: 0
- submission 생성/업로드: 0

기계 판독 가능한 영구 receipt는 `configs/experiments/receipts/p2_normalized_curvature_residual_lgbm_stage1_v1r7_terminal_failed_postmortem.json`입니다.
