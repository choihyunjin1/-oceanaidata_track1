# P1 과적합 방지 감사와 다음 단일 실험 계약

작성일: 2026-08-13 KST  
판정: **기존 outer 결과는 연구 진단으로는 쓸 수 있으나 더 이상 독립 holdout이 아니다. 유일하게 허용한 fixed 24h peer-coherence ablation은 승격 게이트에 실패했으며 family를 종료했다.**

## 1. 감사 범위

다음 로컬 산출물을 집계 수준으로 감사했다.

- `artifacts/runs/*/metrics.json`, `selection.json`, `independent_validation.json`
- `artifacts/sequence_full_20260813/sequence_experiment.json`
- `artifacts/r1_boundary_20260813/preregistration*.json`
- `reports/P1_MODEL_SELECTION_2026-08-13.md`
- `reports/P1_FAILURE_RECON_2026-08-13.md`
- `reports/P1_BREAKTHROUGH_RESEARCH_2026-08-13.md`
- `reports/P1_ACADEMIC_METHODS_SCOUT_2026-08-13.md`

원본 관측값은 수정·복제하지 않았고, 이 감사에서는 집계 통계와 기존 artifact만 사용했다. 공식 test 정답이나 외부 관측값은 사용하지 않았다.

## 2. 선택 편향 감사 결과

현재 세 outer 구간의 라벨은 이미 반복적으로 관찰되었다.

| 항목 | 확인된 최소 노출량 | 해석 |
|---|---:|---|
| outer 결과가 저장된 모델/실험 | 11개 | tree 6, deep 2, R1 계열 3 |
| 서로 다른 모델 family | 8개 이상 | LightGBM offline/causal, augmentation, XGB, CatBoost, TCN, Patch Transformer, boundary completion |
| inner 후처리 candidate-fold 평가 | 4,860회 | 저장된 9개 CV/R1 run에서 fold당 180개 |
| deep screen candidate-fold 평가 | 72회 | 2 architecture × 12 config × 3 fold |
| R1 boundary candidate-fold 평가 | 333회 | 37개 × 3 fold × outer 결과가 노출된 3개 run |

이 수치는 저장된 artifact에서 확인 가능한 **하한**이다. smoke/aborted run, 구현 도중의 수동 비교, 저장되지 않은 탐색은 포함하지 않았다. 또한 동일 outer 행을 유형·지속시간·월·정점·층·모델 disagreement와 oracle로 다시 분석했다. 이 분석은 좋은 가설 생성 자료지만, 같은 outer 점수의 추가 상승을 독립 일반화 근거로 해석하면 selection bias가 생긴다.

R1 v1의 구현 오류로 결과가 무효화되었더라도, 그 결과를 사람이 보았다면 outer 노출 횟수는 되돌릴 수 없다. v2 역시 frozen XGB 대비 micro F1 `-0.005419`, weighted F1 `-0.004215`, paired bootstrap 90% CI `[-0.013423, +0.001692]`로 승격에 실패했다. 따라서 R1 parameter나 window를 바꾸는 후속 시도는 중단한다.

## 3. 새 독립 tail holdout은 존재하지 않는다

train의 마지막 시각은 `2025-12-10T03:00:00+09:00`이며, 마지막 고정 outer validation의 종료 범위 안이다. `2025-12-11` 이후 행은 0개다. 그러므로 지금 새 virgin shadow split을 만들 수 없다.

기존 outer를 임의로 다시 나누거나 일부 그룹을 사후 holdout이라고 이름 붙이지 않는다. 이후 outer 결과는 사전 동결된 한 실험의 one-shot **노출된 연구 진단**으로만 보고한다.

## 4. 다음 단 하나의 가설

다음 family는 `dynamic_peer_reliability_gate`, 실험 ID는 `P1_fixed24h_peer_coherence_v1`로 고정한다.

> 고정 24시간 change-coherence view 네 개를 추가하면, 기존 XGBoost가 층간 증거를 신뢰해도 되는 시기와 정상 성층·내부파로 층이 분리된 시기를 구분할 수 있다.

변경 허용 범위는 다음 네 feature뿐이다.

1. `peer_change_corr_24h`
2. `peer_pair_coverage_24h`
3. `peer_trust_gate_24h`
4. `temp_abs_peer_residual_gated_24h`

고정값은 offline, window 24시간, 최소 pair 비율 0.5다. frozen XGBoost, 기존 feature, seed, 700 trees 상한, 기존 iteration 선택 규칙, 기존 후처리 grid, plateau override, spike 보존은 바꾸지 않는다.

비교 arm은 `frozen_no_op`과 `fixed_24h_peer_coherence` 두 개지만 이는 hyperparameter search가 아니다. fixed gate는 inner 승패와 무관하게 정확히 한 번 실행한다. inner label은 각 arm의 model iteration과 기존 postprocess만 독립적으로 선택하는 nuisance selection에 사용한다. window, feature, gate threshold 또는 family 선택에는 사용하지 않는다.

## 5. 숫자로 고정한 승격 게이트

현재 frozen XGB weighted F1은 `0.8133155525620019`다. 새 후보는 다음을 모두 만족해야 한다.

- test-share weighted F1 `>= 0.8183155525620019` — 절대 `+0.005`
- paired event/day-block bootstrap 90% CI의 F1 차이 하한 `> 0`
- 세 outer fold 중 적어도 두 fold 비열화 없음
- 정점 그룹 F1 최대 하락 `<= 0.01`
- 정상 station-layer-day당 FP 상대 증가 `< 10%`
- G-ORS depth 100% masking stress와 S-ORS year-transfer stress의 F1 하락 각각 `<= 0.01`

outer가 이미 노출되었으므로 위 조건을 모두 통과해도 “독립 검증 완료”라고 표현하지 않는다. 제출 후보로 고려할 수 있는 강한 일관성 검사라는 의미만 갖는다. 공식 hidden test 점수는 사용자 승인 후 대회 제출에서만 알 수 있다.

## 6. NO-GO 중단 규칙

아래 중 하나라도 발생하면 이 family를 닫고 frozen XGB를 유지한다.

- 승격 게이트 하나라도 실패
- fixed 24h 이외 window, correlation cutoff, station/month 조건을 보고 싶어짐
- outer 결과를 본 뒤 feature 정의·임계값·postprocess를 수정하려 함
- fixed gate를 ensemble이나 새 classifier로 확장하려 함
- outer 실행 전에 label-blind 구현 오류 수정이 두 번째 필요함
- outer 실행 뒤 오류를 이유로 같은 family를 재실행하려 함

허용되는 구현 erratum은 outer 결과를 보기 전, 라벨과 무관한 재현 가능한 버그 한 건까지다. parameter·feature·가설 변경은 erratum으로 보지 않는다.

## 7. 실행 계약과 검증

사전등록 파일:

`configs/experiments/p1_next_single_hypothesis.json`

현재 canonical SHA-256:

`4271af9b4331b2f6f58db7faa3301d96c70920a6cdec41a933b4c002f133eacb`

가족별 outer 노출 ledger:

`reports/EXPERIMENT_LEDGER.jsonl`

사전등록 당시 검증 명령:

```powershell
.venv-p1\Scripts\python.exe scripts\validate_preregistration.py `
  configs\experiments\p1_next_single_hypothesis.json `
  --ledger reports\EXPERIMENT_LEDGER.jsonl
```

Family 종료 entry가 append된 현재는 같은 명령이 `family is closed; rerun is prohibited`로 실패하는 것이 정상이다. JSON 자체의 정적 스키마만 감사할 때는 `--ledger`를 생략할 수 있지만, 이는 실행 허가를 뜻하지 않는다.

검증기는 다음을 fail-closed로 강제한다.

- 고정 baseline run/OOF hash/F1
- exactly one fixed feature bundle
- no-op comparator + fixed ablation 두 arm만 허용
- adaptive search와 family selection 금지
- inner label은 iteration·기존 postprocess 선택에만 허용
- family당 outer 결과 최대 1회
- 이미 outer 결과가 있는 family 재사용 금지
- 외부자료·업로드·commit·push 권한 없음
- 존재하지 않는 shadow holdout을 available로 표시할 수 없음

2026-08-13 사전등록 검증 결과: unit test 11개 통과, Ruff 통과, validator PASS. 커밋·push·대회 업로드는 수행하지 않았다.

## 8. 현재 실행의 provenance caveat

현재 `scripts/run_stratification_ablation.py`는 fixed gate, XGBoost, augmentation off, 단일 CV 호출과 환경 override 차단을 코드로 고정한다. 그러나 실행 시작 전에 사전등록 JSON을 직접 읽어 validator receipt와 SHA를 manifest에 넣는 연결은 아직 없다.

이번 실행은 runner 시작 시 통과한 SHA `263709ca60558f994281240e3e19267916075c3177672b71a9fd88af8cd39066` 뒤에, 실험 parameter를 하나도 바꾸지 않고 “inner는 arm 선택이 아니라 nuisance selection만 한다”는 스키마 문구를 명확히 해 현재 SHA가 되었다. 따라서 결과를 사용하기 전 run manifest의 gate config, feature 네 개, backend, augmentation, seed, input SHA를 현재 사전등록과 독립 대조하고 이 clarification 이력을 함께 남긴다. 향후 runner는 outer 시작 전 validator receipt와 canonical SHA를 직접 manifest에 기록해야 한다.

## 9. 최종 실행 결과와 family 종료

실행 ID:

`20260813T205237+0900_strat_gate_fixed24h_59f6d5c6`

독립 QA SHA-256:

`a4bb1e6ba7ffce14907e88bc353b335cc093d07528812701655c4eb26cb09820`

| 지표 | Frozen XGB | Fixed 24h gate | 차이 | 게이트 |
|---|---:|---:|---:|---|
| row micro F1 | 0.860371 | 0.865011 | +0.004640 | 진단상 개선 |
| test-share weighted F1 | 0.813316 | 0.813436 | **+0.000121** | **실패: +0.005 미만** |
| 정상 FP/station-layer-day | 0.370760 | 0.265265 | -0.105495 | 통과 |
| 48시간 이상 event row recall | 0.631506 | 0.628426 | -0.003080 | 악화 |

추가 게이트 결과:

- paired bootstrap 90% CI: `[-0.001677, +0.011611]` — **실패**, 하한이 0보다 크지 않음
- 개선확률: `0.876` — 불확실성을 제거하지 못함
- fold F1 차이: Q2 `+0.019641`, Q3 `-0.009105`, Q4 `+0.005793` — 2/3 비열화 없음은 통과
- 최악 그룹: G-ORS layer 1 `-0.047746`, I-ORS layer 2 `-0.023705` — **실패**, 허용 하락 `0.01` 초과

따라서 micro F1과 FP 감소라는 긍정 신호만 골라 채택하지 않는다. weighted 개선이 사실상 0에 가깝고, bootstrap 불확실성이 0을 포함하며, 핵심 그룹 하락이 크다. `dynamic_peer_reliability_gate` family는 `NO-GO / CLOSED`다.

중단 조치:

- frozen XGBoost를 그대로 유지한다.
- 12h·48h window, correlation cutoff, 다른 feature 조합을 시도하지 않는다.
- 이 결과를 보고 station/month 규칙, calibration, ensemble을 추가하지 않는다.
- 같은 family의 outer 재실행을 금지한다.
- 이미 결정적인 게이트가 실패했으므로 추가 stress 계산으로 family를 구제하지 않는다.

종료 entry는 공개 `record_ledger_entry` API를 통해 `reports/EXPERIMENT_LEDGER.jsonl`에 append했다. 이 entry에는 initial preregistration SHA, 실행 후 문구만 명확히 한 amendment SHA, 실제 설정 변경 0건, 실패한 세 게이트와 family 종료 조치를 함께 기록했다.

종료 후 통합 확인은 사전등록·peer gate·runner 계약 테스트 19개 통과 및 Ruff 통과다. Ledger를 포함한 validator는 이제 동일 experiment 재실행과 새 experiment ID를 사용한 같은 family 재실행을 모두 의도적으로 거부한다.
