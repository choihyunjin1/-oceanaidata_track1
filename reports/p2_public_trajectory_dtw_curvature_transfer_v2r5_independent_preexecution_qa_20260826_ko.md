# P2 public-trajectory DTW v2r5 독립 사전실행 QA

## 결론

**PASS**입니다. 차단 결함은 `P0=0`, `P1=0`이고 비차단 환경 한계 `P2=1`만 남았습니다. 따라서 이 보고서 자체를 SHA-256으로 고정해 외부 authorization에 정확히 연결한 뒤에만, v2r5 단일 실행을 허가할 수 있습니다. 이번 QA는 authorization, claim, materialization, score, fit을 만들거나 실행하지 않았습니다.

고정 계보는 다음과 같습니다.

- design: `c044ae23d14f85c634d8145cbd8f85b004536e378277dc91dc00097dc78f7fe4`
- execution: `32484a1118eda01d900d87f687b05766585d42cbd7c5c6c02621e28f94aa4e8e`
- authorization(false): `932ac241d8eb1457fa72f6290f21f969e46814b4ffabf71b0e489c37ad0fd06b`
- preexecution seal: `3f335f8766e9f5f79f1475a6328ee165aef3c9895b04e86910a7fca0f6c63919`
- runner: `167c730b30c49362ff0aa7bfab3c3ca1256770269d6e66dcbb377823b50d3dbb`
- numerical module: `e523d15d6113fdd4861826786be38fef97a68f0ef5732321c196bd67acf1f05f`
- tests: `6a322ca611ba1e15c0de7d4b41bd26b7abdbe6a90ac3ff93566839389da983ac`
- closure matrix: `185a143c1fb3dedbd40b12b2cde245dfba1ce0242c4e465e9f330c0824eb738e`

## 핵심 감사 결과

v2r4와 v2r5 수치 모듈을 unified diff와 top-level AST로 비교했습니다. 바뀌거나 추가된 수치 정의는 `exact_time_contract`, `exact_sparse_key_contract`, `verify_anchor_against_observations`뿐입니다. 수정 내용은 sparse exact anchor에 연속 존재를 강제하던 잘못을 제거하고, 존재하는 모든 `(time, layer)` 키가 고정 10분 격자에 속하는지 검사하며 진단값을 명시하는 것입니다. 그 밖의 수치 AST는 동일합니다.

과학 계약도 그대로입니다. 6개 셀(`d1/d3/d7 × k3/k7`), inner 3창(2024-03/05/07), exact 1창, 조건부 p100 3창, 7일 source embargo, 총 22 materialization slot, 물리 fit 0회가 유지됩니다. exact gate와 p100 gate의 모든 임계값도 v2r4와 동일합니다.

실제 고정 historical observations와 exact anchor로 claim 전에 constructor-only readiness를 실행했습니다. 결과는 다음 등록값과 전부 일치했습니다.

- observations 789,408행, full exact anchor 69,850행
- 등록 exact 26,273행 및 고유 `(time, layer)` 26,273개, 중복 0개
- 고유 timestamp 8,779개 / 전체 격자 8,784개 / union 누락 5개
- 관측 union gap 2개, off-grid 0개
- layer별 행 수: 2층 8,777 / 3층 8,774 / 4층 8,722
- layer별 전체 격자 누락: 7 / 10 / 62
- 범위: 2024-09-01 00:00 KST ~ 2024-10-31 23:50 KST, 61일
- UTC nanosecond roundtrip PASS, NaT 0
- anchor truth와 독립 observations truth의 key 기반 완전 일치 PASS
- materialize 0, score 0, fit 0, 실제 p100 anchor 접근 0

## 신뢰 경계와 실행 안전성

design/seal의 transitive static input 22개를 독립적으로 전부 재해시했고 `22/22 PASS`, inventory 동일성을 확인했습니다. design·execution·seal·authorization·QA·module은 held-byte snapshot으로 해시와 parse/compile을 같은 바이트에 묶습니다. seal path 토글, authorization path 교체, module path 교체에 대한 fail-closed 테스트도 통과했습니다.

v2r5 네임스페이스는 claim, journal, out-of-band terminal, final, staging 모두 0입니다. journal은 22개 slot의 reserve/completed/failed/skipped 상태를 단방향으로 검증하고, fit 호출은 AST 감사에서도 0개입니다. 결과 출판은 create-only hardlink, atomic replace, commit-ready, terminal-success-last 순서를 검증합니다.

exact `RESEARCH_GO`가 durable journal에 기록되기 전에는 p100 파일 경로를 resolve/stat/open/hash/parquet-parse하지 않습니다. QA 중에도 실제 p100 anchor를 접근하지 않았습니다. 공식 test/sample/submission/candidate 접근, CSV 생성, 업로드, P3 변경은 모두 0입니다.

## 직접 실행 검증

- focused pytest: **70 passed, 1 skipped**, 42.02초
- ruff: **All checks passed**
- 기본 preflight 2회: 정규화 SHA-256이 양쪽 모두 `0e740bcd85ec2315659fb05ad5505311f599aea978f5395cae8547891c977893`로 완전 동일
- 두 preflight 모두 `NOT_AUTHORIZED_PENDING_INDEPENDENT_QA`, authorization false, claim/materialization/score/fit 0

유일한 P2는 현재 Windows 계정에 합성 symlink 생성 권한이 없어 symlink-escape 테스트 1개가 skip된 점입니다. 그러나 resolver는 resolve된 `observations.csv`의 parent가 resolve된 `P2_DATA_DIR`와 정확히 같아야 통과하도록 구현되어 있고, 실제 실행 root와 observations 파일은 모두 LinkType/Target이 없는 일반 경로임을 별도 확인했습니다. 실제 고정 경로에는 영향이 없어 비차단으로 판정했습니다.

## authorization 연결 요건

외부 authorization은 이 JSON 보고서를 생성 후의 정확한 `path`, `bytes`, `sha256`으로 고정하고, 동시에 `verdict=PASS`, 위 design SHA-256, 위 seal SHA-256을 동일하게 기록해야 합니다. authorization 자체의 raw SHA-256은 실행 시 `P2_TRAJECTORY_V2R5_EXECUTION_AUTHORIZATION_SHA256`에 정확히 제공되어야 합니다. 그 전까지는 실행 금지 상태입니다.
