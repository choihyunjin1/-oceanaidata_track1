# P1 Cycle 1 r5 독립 사전 실행 QA

## 결론

**NO_GO**입니다. 차단 finding은 `P0 1건 + P1 1건`, 비차단 문서·스키마 finding은 `P2 1건`입니다. 따라서 r5 actual authorization을 만들거나 72-fit 실행을 시작하면 안 됩니다.

고정 과학 설계 자체는 정적으로 잘 구현돼 있습니다. `9 anchor + 54 inner + 9 outer = 72 fits`, `3 + 9 + 9 = 21 materializations`, 원래 Round-B 3 seeds와 `event_day_balanced_binary_lgbm` anchor, 세 historical cutoff/shelf, full-support OOS 3-seed mean, 동일 postprocess, bounded 24/72/168 context, bootstrap/gates가 유지됐습니다. 그러나 실제 worker의 첫 실행 바이트와 outer-label 최초 접근 순서가 seal의 주장보다 느슨하여 승인 가능한 패키지는 아닙니다.

## 차단 finding

### P0-01 — 실제 worker bootstrap이 sealed snapshot 밖에서 먼저 실행됨

runner `1936-1993`의 exact-tree 검사는 snapshot 내부만 다룹니다. 반면 `_worker_command`(`2567-2583`)는 snapshot runner가 아니라 live `Path(__file__)`를 `sys.executable -B`로 다시 실행합니다. r5 seal의 `snapshot_static_inventory`에도 현재 runner는 없습니다.

이 때문에 두 경로가 검증보다 먼저 실행될 수 있습니다.

- parent 검증 뒤 live runner가 교체되면 교체된 코드가 worker manifest 검증 전에 실행됩니다.
- `-I/-S`가 없으므로 `PYTHONPATH`, `sitecustomize.py`, `usercustomize.py`, site-package `.pth` startup code가 runner 첫 줄보다 먼저 실행될 수 있습니다.

사후 self-hash는 이미 실행된 bootstrap을 소급 인증하지 못합니다. r5 amendment의 “substituted/extra code가 project import 전에 실패”라는 주장도 이 실제 launch path에는 성립하지 않습니다.

최소 수정은 다음과 같습니다.

- 현재 worker runner 또는 최소 bootstrap을 exact sealed snapshot inventory에 포함하고, live 경로가 아닌 snapshot 경로를 실행합니다.
- parent는 verified bootstrap에 write/delete-denying held handle을 worker import 완료까지 유지하거나 동등한 race-free held-byte launch를 사용합니다.
- worker를 `python -I -S -B <snapshot bootstrap>`로 시작하고 환경 allowlist를 강제합니다.
- 필요한 runtime package는 authorization 뒤 individually pinned path만 직접 추가하며 `.pth`를 처리하지 않습니다.
- import 전후 exact tree/hash를 다시 확인합니다.
- malicious `PYTHONPATH/sitecustomize` marker, parent 검증 뒤 live-runner swap, snapshot `.pyc`/extra/transitive-source race가 모두 marker/import/claim 전에 실패하는 negative test가 필요합니다.

### P1-01 — outer label이 all-fold prediction freeze 전에 이미 parse·보존됨

r5 amendment는 outer truth/anomaly/target-derived audit의 최초 접근을 `record_outer_freeze` 이후로 제한합니다. 하지만 readiness는 runner `1020-1024`에서 full `train.csv`를 unrestricted `pd.read_csv`로 읽고, runner `1145-1157`에서 target 열을 포함한 전체 DataFrame을 state에 보존합니다. runner `1117-1118`은 freeze 전 label 기반 left-censor audit도 호출합니다.

left-censor audit가 과거 training prefix만 계산하고 numerical `_fit_outer`가 truth를 쓰지 않는다는 점은 확인했습니다. 그래도 outer-period target 열이 이미 materialize되어 접근 가능한 상태이므로 “최초 접근이 freeze 이후”라는 구조적 firewall은 성립하지 않습니다. 현재 테스트 `701-781`은 `frozen_truth_oof_*` key만 감시하여 initial train reader를 놓칩니다.

최소 수정은 다음과 같습니다.

- pre-freeze full frame은 target-free columns만 materialize합니다.
- anchor/inner target은 outer 평가기간 row를 물리적으로 포함할 수 없는 separately sealed historical-target view로 분리합니다.
- outer-period label/anomaly bytes는 freeze 전까지 opaque하게 유지하고 state API에서도 접근 불가능하게 합니다.
- censor audit가 필요하면 historical-target view만 사용합니다.
- pre-freeze outer-period target conversion/index/column access/metric/audit를 즉시 실패시키는 reader-level sentinel test를 추가합니다.

## P2-01 — authorization exact schema 미완성

top-level key 집합은 exact 검사하지만 `schema_version` 값과 `preexecution_seal`/`independent_qa` nested exact membership/type은 parent와 worker에서 완전히 강제하지 않습니다. 외부 digest와 핵심 필드 검사가 있어 즉시 권한 우회는 재현되지 않았으므로 P2로 분류했습니다. successor에서는 exact version, nested key/type 검사와 wrong-version/extra-field negative test를 추가해야 합니다.

## r4 finding 재검증

- auth digest circularity: 독립적으로 닫힘. 실제 digest는 external env capability이고 runner bytes는 불변입니다.
- authorization 전후 preflight digest 변화: 닫힘. live auth inspection이 stable verification digest 밖으로 분리됐습니다.
- seal A/경로 seal B TOCTOU: 닫힘. seal/QA/design/amendment/module held bytes가 snapshot으로 전달됩니다.
- self-declared inventory: snapshot 내부 검사는 닫혔지만 실제 bootstrap이 inventory 밖이므로 전체 closure는 P0-01로 실패합니다.
- outer truth chronology: `_fit_outer` 내부는 닫혔지만 eager full-train target parse 때문에 전체 closure는 P1-01로 실패합니다.
- post-unlink stdout-loss recovery: 닫힘. read-only durable-success recovery와 focused crash test가 통과했습니다.

## 독립 검증 결과

- 고정 SHA 모두 일치:
  - design `31b0bde2…69563`
  - r5 amendment `bd0370c7…57563`
  - runner `e97770f6…aef84`
  - module `68b644b1…68e81`
  - tests `d6281d9b…80673`
  - auth false template `72feb636…a7b6`
  - seal r5 `20619d35…a21d`
- focused pytest:
  - parent env: `26 passed, 1 skipped` (`P1_DATA_DIR` 미설정 skip)
  - command-scoped historical P1 data dir: `27 passed`
- Ruff: `All checks passed!`
- read-only preflight 2회:
  - exit code 모두 0
  - stdout 각 17,960 bytes, SHA 모두 `707c3d62…b89e0`로 byte-identical
  - verification SHA 모두 `793f131d…4fb3`
  - seal SHA 모두 `20619d35…a21d`
  - namespace state 전후 동일
- actual authorization, claim, fit, materialization, outer score, candidate, official-path read, upload: 모두 0

공식 P1 test/sample/submission/candidate CSV와 P3는 접근하지 않았습니다. 수치 screen도 실행하지 않았습니다.

## 최종 지시

r5는 immutable `NO_GO`로 보존해야 합니다. P0/P1 수정과 adversarial negative test를 새 successor에 넣고 zero-operation 상태에서 새 seal과 독립 QA를 다시 받아야 합니다.
