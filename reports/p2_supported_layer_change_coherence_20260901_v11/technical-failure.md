# P2 supported-layer change-coherence v11 — terminal technical failure

## 결론

`INVALID_ZERO_FIT_READ_ONLY_COHERENCE_SCORE_ARRAY`다. pandas `DataFrame.max(...).to_numpy(float)`가 read-only ndarray view를 반환했고, 최소 3개 공개층 support가 없는 행에 `NaN`을 in-place 대입하는 첫 단계에서 `ValueError: assignment destination is read-only`가 발생했다.

attempt lock은 소비됐고 동일 ID를 재개하지 않는다. model fit 0, prediction commitment 0, metric 0이므로 change-coherence 가설의 과학적 NO_GO가 아니다. v10과 v11은 각각 독립적인 0-fit technical INVALID다.

## 실행 전 검증과 한계

zero-operation preflight 두 출력은 byte-identical(SHA-256 `90dbd30d9df97d09a2f710aa95c0619365dca58870638dd9617cc1f86bf0c5fb`)이었고 focused pytest 6/6, Ruff, pycompile이 통과했다. 그러나 합성 테스트가 pandas 반환 배열의 writeable flag를 검사하지 않아 이 계약 오류를 잡지 못했다.

단순 수리는 `.to_numpy(...).copy()`지만, 현재 exactly-once ID와 lock에는 적용하지 않는다. 새 기술 repair ID도 이번 지시 범위에서는 실행하지 않는다.

## 다음 architecture-scale 단일 축 제안

가장 단순한 미사용 architecture-scale 축은 **continuous-depth set encoder**다. 공개 관측층을 고정 벡터로 펼치지 않고 `(actual depth, nominal depth, temperature, salinity, presence)` 원소 집합으로 받아 shared MLP로 인코딩하고 permutation-invariant mean/max pooling한다. 각 target depth embedding과 결합해 L2–L4 residual을 공동 예측하며, 최종값은 고정된 작은 blend로 챔피언을 기본 보존한다.

이 축은 fixed-grid deep/GBM stack, DINEOF/PLS/copula/GP/CatBoost/PAVA, raw residual tree, normalized-curvature Ridge와 의미적으로 다르다. depth를 연속 query로 다루고 public-layer missingness를 set cardinality로 처리하기 때문이다. 다음 사전등록은 1 architecture, 3 fixed seeds, Huber influence loss(행 삭제 0), shared encoder 32 units×2, target head 32 units×2, champion/model 0.8/0.2 fixed blend, 최대 9 fits로 제한하고 outer 결과로 구조·blend·loss를 바꾸지 않아야 한다.

승격 gate는 pooled ΔRMSE<0, 2/3 folds 개선, L2–L4 최악 악화<=0.003°C, day-block CI90 upper<0, transport-adjusted +0.01점 이상을 그대로 사용한다. 모든 historical fold는 exposed exploratory라는 한계도 유지한다.

## v7 및 접근 경계

v7 ready pack은 값·해시 재조회, materialize, upload 모두 0이다. 챔피언 보존을 기본으로 하고, v7은 강제로 정보 probe 슬롯을 쓸 때만 기존 우선순위를 유지한다.

official test/sample/baseline/score/query support/hidden/submission CSV/upload 접근은 모두 0이다.
