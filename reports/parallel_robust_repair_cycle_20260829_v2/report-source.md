# 2026-08-29 병렬 robust-repair 사이클 최종 보고

## 결론

- **P1: `NO_GO_Q2`** — Group-DRO MS-TCN은 앵커 양성을 제거하지 않았지만, 3개 월 모두와 pooled F1에서 incumbent를 이기지 못했고 정점 편중 제한도 넘었다. Q3/Q4는 실행하지 않았다.
- **P2: `TRAIN_ONLY_SUPPORT_PASS_QUERY_AUDIT_NOT_AUTHORIZED`** — 학습 자료에서 copula 축을 시험할 수 있는 지지집합은 충분하다. 그러나 공식 query support 검사는 승인 경계 밖이므로 모델 승격 근거가 아니라 다음 실험의 준비 근거다.
- **P3: `INVALID_TERMINAL_TECHNICAL_FAILURE`** — 138-fit successive halving에서 `challenger_21`이 control 대비 RMSE를 0.619297 m에서 0.596434 m로 낮추고 selection gate를 통과했다. 하지만 첫 confirmation fit 직후 frozen-router 컬럼 계약 오류로 종료되어 과학적 승격 여부는 판정할 수 없다. 자동 재실행은 금지한다.

## P1 — MS-TCN Group-DRO

- 고정 사항: 165 past-only features, `trial_18`, anchor-union decoder, weight decay 0.001, station×layer×quarter sparse-merged BCE, pooled/worst 0.5 objective.
- 실행량: Q2 seed 3개 × epoch 150 = historical fits 3, runtime 3,241.494 s.
- terminal: `NO_GO_Q2`; Q3/Q4 시작 안 함.
- winner threshold: 0.8.
- pooled ΔF1: -0.0134805.
- 최소 월별 ΔF1: -0.0235490.
- 최대 정점 changed-row share: 0.802817 > 0.8.
- anchor-positive removed rows: 0.
- 실험 경계: official rows 0, CSV 생성 false, upload false.
- 해석: recall 증가만으로 precision 손실과 S-ORS 편중을 상쇄하지 못했다. 이 고정 Group-DRO 목적함수는 폐기한다.

## P2 — train-only copula support audit

- observations.csv 789,408행만 재검사했다.
- complete historical timestamps: 47,216.
- seasonal-layer-coordinate 최소 unique: 2,353.
- duplicate station-layer-time: 0.
- nonfinite rank-Gaussian values: 0.
- 독립 재계산은 기존 JSON과 완전 일치했고 3.108 s가 걸렸다.
- model fits 0, official input rows 0, CSV 0, upload 0.
- query-dependent nearest-prefix/min-max 검사는 실행하지 않았다.
- 해석: copula 계열을 탐색할 학습 지지집합은 충분하지만, query support와 실제 복원 성능은 아직 미검증이다.

## P3 — valid-only CatBoost HPO

- synthetic smoke: CatBoost 1.2.10, 37 fits PASS.
- historical selection: 36 challengers를 36→9→3→1로 축소, rung fit count 74 + 40 + 24 = 138.
- selection winner: `challenger_21`.
- selected parameters: Plain/Depthwise, depth 9, learning rate 0.02, L2 20, RSM 0.75, Bayesian bagging temperature 0.2.
- control RMSE: 0.6192966 m.
- challenger RMSE: 0.5964341 m.
- ΔRMSE: -0.0228625 m; selection gate 전 항목 PASS.
- selection 후 첫 confirmation challenger fit 1회를 완료한 뒤 실패했다. 따라서 historical fits는 139, synthetic를 포함한 completed model fits는 176이다.
- terminal runtime: 약 25,871.4 s(7 h 11 m 11 s).
- 실패 원인: `read_frozen_router_components()`는 의도적으로 `current_hs`와 `single_prediction`을 제외한 canonical router component table을 반환한다. frozen v1 engine의 confirmation 경로가 이 projection에서 두 컬럼을 다시 선택하여 `KeyError`를 발생시켰다.
- `result.json`, confirmation seal/prediction, full refit, CSV, upload은 생성되지 않았다.
- 해석: selection 신호는 유망하지만 confirmation이 봉인되지 않았으므로 승격·제출 후보로 간주하면 안 된다. 이는 과학적 NO_GO가 아니라 integration-contract coverage failure다.

## 독립 QA

- focused pytest: 10/10 PASS.
- Ruff: 관련 P1/P2/P3 코드·테스트 전체 PASS.
- P1 terminal/q2 gate 내용 일치 및 해시 재계산 PASS.
- P2 audit 독립 재계산이 저장된 결과와 완전 일치.
- P3 rung 합계 138, gate metrics, artifact hashes, traceback, attempt-lock 소비를 재검산.
- 이 보고의 official/CSV/upload 0은 **robust-repair 실험 프로세스 범위**다. 별도의 사용자 승인 deadline submission은 이 실험의 산출물이 아니다.

## 다음 승격 판단

1. P1 Group-DRO v2는 종료한다. 같은 목적함수의 threshold·seed 재탐색은 하지 않는다.
2. P2는 query-independent support PASS만 보존한다. 별도 승인 전 공식 query 파일을 읽지 않는다.
3. P3는 자동 재실행하지 않는다. 다음 사이클이 승인될 경우에만 canonical router projection을 실제 confirmation schema와 맞추는 최소 수정과 end-to-end synthetic/canonical integration test를 먼저 만들고, 새로운 experiment id·lock·예산으로 재등록한다.
4. P3 selection의 -0.02286 m 신호는 다음 사이클 우선순위를 높이는 근거이지만, 현 사이클 승격 근거는 아니다.
