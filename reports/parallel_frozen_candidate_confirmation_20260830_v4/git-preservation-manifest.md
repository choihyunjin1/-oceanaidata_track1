# 2026-08-30 연구 사이클 Git 보존 manifest

## 결론

8월 30일의 재현 가능한 코드·설정·테스트·집계 보고서·공식 결과 receipt를 하나의 연구 계보로 보존한다. 원본 데이터, 제출 CSV, 예측·체크포인트, 비밀정보, 캐시, 로그와 exactly-once attempt lock은 포함하지 않는다.

## 포함 범위

- P1 frozen confirmation, sealed-evaluation, add-only precision 연구의 코드·설정·테스트·작은 집계 결과.
- P2 availability-aware sparse copula, frozen deployability, exact frozen pack, rank-1 bin 분해의 코드·설정·테스트·작은 집계 결과.
- P3 fresh-episode confirmation, sparse-GP abstention, KMA 공식 결과의 코드·설정·테스트·작은 집계 결과.
- gate/tolerance 재교정, 세 문제 통합 확인, 7개 official information probe의 의사결정·결과·독립 QA.

선별 전 untracked 파일은 124개, 1,141,205 bytes였다. 아래 제외 9개를 빼고 이 manifest를 더해 116개 파일을 stage한다.

## 명시적 제외

- attempt lock 6개: exactly-once 실행 상태를 공개 Git 이력에 넣지 않는다.
- P1 station-ablation builder/config/test 3개: 로컬 제출 archive의 개인 절대경로가 들어 있어 현재 상태로는 portable-code 계약을 충족하지 않는다. 공식 후보의 해시·점수·독립 QA는 `reports/official_information_probe_cycle_20260830_v1/`에 별도로 보존한다.
- `.gitignore`가 차단하는 `artifacts/`, `submissions/`, 원본 데이터, CSV, 모델, 예측, cache, log, credential 전부.

## 검증

- 새 Python 파일 전체 Ruff: PASS.
- 8월 30일 focused pytest: 150개 중 148 PASS.
- 제외된 2개 assertion은 exactly-once 실행 전 namespace가 비어 있어야 한다는 preflight다. 현재 namespace는 이미 terminal 결과를 보유하므로 preflight가 `FAIL`을 반환하는 것이 정상 사후 상태다. lock/result를 삭제하거나 테스트 계약을 완화하지 않았다.
- secret-pattern scan: 0건.
- 1 MiB 초과 untracked 파일: 0건.
- CSV/NPZ/Parquet/checkpoint/model 확장자: 0건.

## 해석 경계

이 커밋은 공식 Public 점수와 로컬 집계 결과를 보존하지만 Private 성능을 증명하지 않는다. 공식 test/sample/submission의 행 값과 hidden label은 포함하지 않는다.
