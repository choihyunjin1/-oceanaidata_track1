# P2 trajectory-DTW Cycle 1 v2 독립 실행 전 QA

## 결론

`NO_GO`입니다. 현재 패키지는 실행하면 안 됩니다. 기존 집중 테스트 40개와 ruff, 기본 read-only preflight는 통과했지만, 실제 성공 publication 경로와 locked-p100 키 결속을 공격적으로 검사하자 P0 2건, P1 4건, P2 1건이 확인됐습니다.

공식 test/sample/submission/candidate 경로는 읽지 않았고 업로드·fit·수치 materialization은 0회입니다. 실제 p100 anchor도 open/stat/hash/parse하지 않았습니다. 결함 주입은 모두 합성 데이터 또는 임시 디렉터리에서 수행했으며 실제 실험 claim/journal/OOB/final/staging 네임스페이스는 끝까지 깨끗합니다.

## 실행 차단 결함

| ID | 등급 | 결함 | 독립 증거 | 필수 수정 |
|---|---:|---|---|---|
| P0-01 | P0 | p100 fold/station을 키가 아닌 위치로 재결속 | `load_p100_anchor()`의 merged는 layer→time 순서인데 verified는 time→layer 순서입니다. 합성 78,156행 중 52,104행(66.67%)의 key→fold가 틀렸습니다. | `(UTC-ns, layer)` one-to-one merge 결과로 verified를 직접 구성하고 입력 순열 adversarial test에서 mismatch 0을 증명해야 합니다. |
| P0-02 | P0 | Windows 실제 성공 publication 불가 | destination hardlink를 `O_RDONLY`로 연 뒤 `os.fsync()`하여 sealed Windows/Python에서 첫 `result.json` 직후 `[Errno 9] Bad file descriptor`가 발생했습니다. terminal marker는 생성되지 않았습니다. | Windows에서 지원되는 flush 방법으로 교체하고 mock 없는 전체 성공 publication을 terminal까지 테스트해야 합니다. |
| P1-01 | P1 | QA receipt가 authorization 신뢰경계에 실제 결속되지 않음 | 존재하지 않는 `reports/definitely_absent_qa.json`과 `deadbeef` hash를 넣은 합성 AUTHORIZED 객체가 승인 검사에 통과했습니다. | QA report의 안전 경로·bytes·SHA·verdict·experiment/design/seal/bundle lineage를 실제 검증하고 authorization bundle에 pin해야 합니다. |
| P1-02 | P1 | seal의 전이적 완전성 부족 | worker가 미봉인 `p2_restore/__init__.py`, `data.py`, `features.py`를 실행합니다. base-v1 및 predecessor config/claim/journal도 trigger에 문자열 pin만 있고 runner가 실제 파일/의미를 재검증하지 않습니다. | 직접 sealed-file import를 쓰거나 모든 프로젝트 의존성을 pin하고, trigger 근거 파일과 terminal 의미도 preflight에서 검증해야 합니다. |
| P1-03 | P1 | exact gate 전에 p100 경로를 resolve | `_source_readiness → _source_pins → _safe_project_path(P100) → Path.resolve()`가 실행되면서도 filesystem access를 0이라고 기록합니다. | p100은 exact GO 전에는 literal pin metadata만 보유하고, safe resolve 자체도 lazy loader 안으로 옮겨야 합니다. |
| P1-04 | P1 | terminal link 뒤 fallible I/O | terminal link 직후 directory-fsync fault를 넣으면 marker/logical SUCCESS는 남고 OOB는 없는데 OSError가 전파됩니다. | 복구 가능한 단일 commit 경계를 정하고 post-commit 오류가 실행 결과를 뒤집지 않도록 해야 합니다. |
| P2-01 | P2 | `query_start` 용어 중의성 | 코드·seal은 trajectory 시작에서 7일을 빼지만 design 문장 일부는 D 00:00을 query_start라고 부릅니다. | 수치 규칙을 바꾸지 말고 `prediction_day_start`와 `trajectory_start`를 분리해 명명해야 합니다. |

## 통과한 계약

- v1과 v2의 6개 cell, RESEARCH_GO 수치 gate, SUBMISSION_GO 수치 threshold, preferred/stretch 효과 threshold는 동일합니다.
- UTC `datetime64[ns]` 왕복, 7일 embargo guard, exact observation truth 및 candidate-key 선검증, weak-support/endpoint-missing exact no-op는 코드와 focused test에서 확인했습니다.
- materialization graph는 18 inner + 1 exact + exact GO일 때만 3 p100으로 고정돼 있고 `.fit(` 호출은 없습니다.
- exact slot 19의 `RESEARCH_GO` journal append 이후에만 p100 bytes를 open/hash/parse하는 순서는 구현돼 있습니다. 다만 P1-03처럼 그 전의 path resolve까지 완전히 지연하지는 못했습니다.
- p100 slice pass 값은 literal이 아니라 fold/layer/season/distance/missingness/coverage/distance×layer inventory에서 계산합니다. 다만 P0-01 때문에 현재 fold inventory 자체가 신뢰 불가합니다.

## 재현 결과

- `python -m pytest tests/test_p2_public_trajectory_dtw_curvature_transfer_v2.py -q -p no:cacheprovider` → `40 passed`
- `python -m ruff check ...` → `All checks passed`
- 기본 runner preflight → `NOT_AUTHORIZED_PENDING_INDEPENDENT_QA`, 실제 네임스페이스 clean, 모든 operation counter 0
- 78,156행 합성 순열 probe → fold mismatch `52,104`
- 실제 Windows hardlink success-path probe → `[Errno 9] Bad file descriptor`
- 존재하지 않는 QA receipt authorization probe → 잘못 승인됨
- terminal-link 이후 fsync fault probe → marker/logical SUCCESS + 전파된 OSError

## 다음 조건

현재 안전 실행 명령은 없습니다. 위 결함을 수정한 새 revision에서 runner/module/tests/config/auth/seal의 bytes·SHA를 모두 다시 고정하고, 실제 Windows success publication 및 임의 순서 p100 key binding을 새 테스트에 포함한 뒤 독립 zero-fit 재QA를 받아야 합니다. 그 전에는 `--execute-local`을 사용하지 마십시오.

