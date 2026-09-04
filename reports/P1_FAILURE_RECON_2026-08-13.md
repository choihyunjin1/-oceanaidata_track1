# P1 OOF 실패 원인 정찰 — 2026-08-13

> **연구 진단 전용 경계:** 이 보고서는 이미 공개된 train의 outer-validation 라벨을 사용한다. 아래 수치와 oracle은 원인 분석·가설 생성에만 사용할 수 있으며, 후보 선택, 임계값 조정, 앙상블 학습, 승격, 제출 결정 또는 hidden-test 성능 주장에 사용하면 안 된다.

## 기술 요약

- 동일한 421,032개 outer-validation 행과 16,055개 양성 행에서 LightGBM, XGBoost, CatBoost, 합성 증강 LightGBM, TCN, Patch Transformer의 OOF를 키·라벨 기준으로 완전히 정렬했다. 여섯 모델 모두 직접 재집계와 저장 metric이 일치했다.
- 현재 XGBoost OOF의 F1은 `0.860371`이며 TP 12,946, FP 1,093, FN 3,109다. 실패의 핵심은 threshold 근처의 무작위 행이 아니라 **장시간 지속되는 offset/drift 구간의 불완전한 범위 탐지**다. 정확한 anomaly signature 기준으로 offset/drift 계열이 FN의 2,882개(92.7%), 48시간 이상 true event가 FN의 2,393개(77.0%)를 차지한다.
- 실패는 소수 regime에 집중된다. `I-ORS/L1 offset`, `S-ORS/L2 drift`, `S-ORS/L6 offset`, `I-ORS/L1 drift` 네 교차군만 FN 1,852개(59.6%)다. FP는 2025년 6월 S-ORS에 632개(전체 FP의 57.8%)가 집중된다.
- 모델 간 보완 가능성은 있으나 단순 union은 F1 `0.560786`, intersection은 `0.795778`에 그친다. outer 라벨로 행마다 올바른 모델을 고르는 **구현 불가능한 oracle**은 F1 `0.951459`이며, 이는 상한 진단일 뿐 설계값이 아니다. XGBoost FN 중 1,755개(56.4%)는 다른 모델 하나 이상이 잡고, FP 중 947개(86.6%)는 다른 모델 하나 이상이 거부한다.
- 증강 LightGBM은 recall을 높였지만 FP가 14,345개로 폭증해 F1 `0.609332`로 하락했다. TCN은 recall `0.847400`과 장기 구간 탐지가 강하지만 FP 5,789개, Patch Transformer는 spike recall `0.2295`로 단독 승격 근거가 없다.
- 우선 돌파점은 장기 offset/drift의 **시작·끝 경계 복원**, regime-aware slow-anomaly expert, 그리고 inner split에서만 학습하는 보수적 disagreement gate다. 본 보고서로 발견된 outer subgroup이나 oracle을 직접 최적화해서는 안 된다.

## 1. 여섯 모델은 같은 population에서 비교되었다

모든 수치는 세 개의 7일 purge rolling-origin outer holdout을 합친 row-level binary 결과다. deep OOF는 `row_index`, tree OOF는 공식 key 네 개로 train에 결합했으며, population과 라벨이 모두 일치하지 않으면 스크립트가 실패하도록 했다.

| 모델 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 12,356 | 1,846 | 3,699 | 0.870018 | 0.769604 | 0.816737 |
| XGBoost | 12,946 | 1,093 | 3,109 | **0.922145** | 0.806353 | **0.860371** |
| CatBoost | 12,213 | 2,006 | 3,842 | 0.858921 | 0.760698 | 0.806831 |
| LightGBM + 합성 증강 | 13,320 | 14,345 | 2,735 | 0.481475 | 0.829648 | 0.609332 |
| TCN | **13,605** | 5,789 | **2,450** | 0.701506 | **0.847400** | 0.767582 |
| Patch Transformer | 13,082 | 3,578 | 2,973 | 0.785234 | 0.814824 | 0.799755 |

이 표는 outer 라벨을 이용한 사후 진단이다. XGBoost의 기존 선택을 재현하지만, 이 비교를 새 모델 선택이나 재승격 절차로 재사용할 수 없다.

## 2. XGBoost의 주된 FN은 긴 offset/drift의 경계 누락이다

### 유형별 recall

유형은 `+` 복합 라벨을 각 membership에 중복 집계하므로 행 합계가 전체 양성과 일치하지 않는 비가산 표다.

| 모델 | spike | noise | flatline | offset | drift |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.7541 | 0.8186 | 0.9997 | 0.6023 | 0.6752 |
| XGBoost | **0.8197** | **0.9355** | **0.9997** | 0.6492 | 0.6462 |
| CatBoost | 0.6557 | 0.8464 | **0.9997** | 0.6030 | 0.6141 |
| 증강 LightGBM | **0.8525** | 0.9008 | 0.9994 | 0.6776 | **0.7384** |
| TCN | 0.5902 | **0.9355** | 0.9717 | **0.7858** | 0.6966 |
| Patch Transformer | 0.2295 | 0.9128 | 0.9484 | 0.7734 | 0.6316 |

XGBoost의 mutually exclusive signature 기준 FN은 drift 1,429개, offset 1,324개, `noise+drift` 84개, `drift+offset` 45개다. 합계 2,882개로 전체 FN의 92.7%다. flatline은 3,153개 중 단 한 행만 놓쳤고 spike FN은 11개이므로, 현재 row-F1의 큰 개선 여지는 slow anomaly 쪽에 있다.

### true event 지속시간별 XGBoost

| 지속시간 | True events | 완전 미탐 events | 양성 행 | FN 행 | Event recall | Row recall |
|---|---:|---:|---:|---:|---:|---:|
| 10분 | 59 | 11 | 59 | 11 | 0.8136 | 0.8136 |
| 20분~3시간 미만 | 1 | 0 | 15 | 0 | 1.0000 | 1.0000 |
| 3~8시간 | 7 | 0 | 231 | 24 | 1.0000 | 0.8961 |
| 8~24시간 | 26 | 1 | 2,516 | 279 | 0.9615 | 0.8891 |
| 24~48시간 | 31 | 0 | 6,740 | 402 | 1.0000 | 0.9404 |
| 48시간 이상 | 17 | 3 | 6,494 | **2,393** | 0.8235 | **0.6315** |

48시간 이상 event는 17개 중 14개를 일부라도 잡았지만 전체 행의 63.2%만 탐지했다. 즉 장기 event를 전부 못 보는 문제보다 시작·끝 또는 약한 내부 구간을 잘라 먹는 문제가 더 크다. XGBoost probability가 `<0.05`인 FN도 2,677개(86.1%)라서, 전역 threshold를 조금 낮추는 방식만으로는 이 문제를 안전하게 해결하기 어렵다.

복합 event도 같은 경향이다. 복합 true event 9개는 모두 일부 탐지됐지만 row recall은 0.6755이고 FN 910개다. 단일 event의 row recall 0.8341보다 낮아, 중첩 유형에서 구간 완성도가 부족하다.

## 3. 실패는 특정 station-layer와 달에 집중된다

### XGBoost station-layer

| Station-layer | 양성 행 | TP | FP | FN | F1 | 전체 FP 기여 | 전체 FN 기여 |
|---|---:|---:|---:|---:|---:|---:|---:|
| I-ORS/L1 | 1,430 | 441 | 167 | **989** | 0.4328 | 15.3% | **31.8%** |
| S-ORS/L2 | 1,020 | 437 | **269** | **583** | 0.5064 | **24.6%** | **18.8%** |
| S-ORS/L6 | 1,335 | 943 | 51 | 392 | 0.8098 | 4.7% | 12.6% |
| G-ORS/L1 | 1,066 | 742 | 1 | 324 | 0.8203 | 0.1% | 10.4% |
| S-ORS/L1 | 1,571 | 1,449 | **285** | 122 | 0.8769 | **26.1%** | 3.9% |
| S-ORS/L3 | 1,328 | 1,309 | 163 | 19 | 0.9350 | 14.9% | 0.6% |

상위 네 FN group(`I-ORS/L1`, `S-ORS/L2`, `S-ORS/L6`, `G-ORS/L1`)이 전체 FN의 73.6%다. 유형을 교차하면 다음 네 조합만으로 59.6%를 설명한다.

| Station-layer × signature | 양성 행 | TP | FN | Recall | 전체 FN 기여 |
|---|---:|---:|---:|---:|---:|
| I-ORS/L1 × offset | 709 | 61 | **648** | 0.086 | 20.8% |
| S-ORS/L2 × drift | 666 | 83 | **583** | 0.125 | 18.8% |
| S-ORS/L6 × offset | 433 | 104 | 329 | 0.240 | 10.6% |
| I-ORS/L1 × drift | 512 | 220 | 292 | 0.430 | 9.4% |

이 concentration은 regime-specific expert의 근거가 되지만 동시에 강한 누출 위험이다. 이 outer subgroup에 맞춘 규칙이나 threshold는 만들지 않고, 가설만 동결한 뒤 과거 inner block에서 독립적으로 시험해야 한다.

### XGBoost 월별 실패

| 월 | TP | FP | FN | F1 | 전체 FP 기여 | 전체 FN 기여 |
|---|---:|---:|---:|---:|---:|---:|
| 2025-04 | 1,633 | 161 | 741 | 0.7836 | 14.7% | 23.8% |
| 2025-06 | 661 | **645** | 662 | **0.5029** | **59.0%** | 21.3% |
| 2025-07 | 1,791 | 46 | 667 | 0.8340 | 4.2% | 21.5% |
| 2025-11 | 2,512 | 147 | 225 | 0.9311 | 13.4% | 7.2% |

6월 FP 645개 중 632개가 S-ORS이며 전체 FP의 57.8%다. fold 기준으로도 2025 Q2가 FP 831개(76.0%)와 FN 1,450개를 포함해 F1 0.7746이고, Q3/Q4는 각각 0.9019/0.9066이다. 계절 성층·내부파·기지별 변동을 이상으로 오인하는 후보 원인이지만, 이 분석은 인과를 입증하지 않는다.

## 4. disagreement에는 신호가 있지만 raw ensemble은 실패한다

### 여섯 모델의 양성 vote 수

| 양성 vote | 행 수 | 실제 양성 행 | 실제 양성률 | 해석 |
|---:|---:|---:|---:|---|
| 0 | 384,657 | 1,354 | 0.35% | 공통 FN 잔여 |
| 1 | 17,882 | 653 | 3.65% | 단독 모델 양성은 대부분 FP |
| 2 | 3,651 | 654 | 17.91% | 낮은 신뢰 disagreement |
| 3 | 1,504 | 740 | 49.20% | 완전 경계 영역 |
| 4 | 1,059 | 635 | 59.96% | 중간 신뢰 disagreement |
| 5 | 1,427 | 1,313 | 92.01% | 높은 합의 |
| 6 | 10,852 | 10,706 | 98.65% | 공통 TP와 146개 공통 FP |

vote 수가 신뢰도 정보를 담지만 단순 union은 FP 21,674개로 F1 0.5608, intersection은 FN 5,349개로 F1 0.7958이다. 따라서 vote cut을 이 outer 표에서 직접 고르는 것도 허용되지 않으며, inner OOF에서만 calibration/gating해야 한다.

XGBoost 기준으로 다른 모델이 잡은 FN은 1,755개(56.4%), 다른 모델 하나 이상이 거부한 FP는 947개(86.6%)다. 하지만 outer 라벨로 매 행 올바른 결정을 고른 oracle F1 0.9515는 구현 불가능하고 과대 상한이다. 실현 가능한 uplift로 해석하면 안 된다.

모든 모델이 함께 틀린 행은 FN 1,354개와 FP 146개다. 공통 FN 중 1,057개(78.1%)가 48시간 이상 event에 속하고, exact signature로 offset 605개·drift 571개·`noise+drift` 84개·`drift+offset` 45개다. 이는 앙상블만으로는 남는 shared blind spot이며 새 장기 기준선/변화점 표현이 필요하다는 근거다.

## 5. 증강과 deep 모델은 specialist 가능성만 보였다

### 합성 증강

증강 LightGBM은 비증강 LightGBM보다 FN을 964개 줄였지만 FP를 12,499개 늘렸다. FP 14,345개는 536개 run으로 구성되며, 6월에만 9,377개가 발생했다. 48시간 이상 FP run도 5개·3,024행이다. 합성 이상의 크기·배경 regime·class-prior 또는 threshold calibration이 실자료 정상 변동과 맞지 않는다는 강한 정황이다. 현재 증강 결과를 ensemble에 넣거나 recall 개선 근거로 승격하면 안 된다.

### Deep OOF

- TCN은 offset recall 0.7858, drift recall 0.6966, 48시간 이상 event 17개를 모두 일부 탐지하고 해당 row recall 0.7219를 기록했다. 그러나 FP 5,789개와 precision 0.7015 때문에 단독 모델로는 부족하다.
- Patch Transformer는 offset recall 0.7734지만 spike recall 0.2295, FP 3,578개다. 장기 구간 specialist 가능성은 있어도 범용 대체 모델 근거는 없다.
- XGBoost와 TCN의 prediction disagreement는 1.87%이며 disagreement 행에서 XGBoost가 맞고 TCN이 틀린 경우가 5,947개, 반대가 1,910개다. TCN 전체를 혼합하기보다 inner-only residual gate가 필요한 이유다.

## 6. peer와 depth-missing 결과는 강하게 제한된다

- outer OOF에서 peer available은 96.11%다. no-peer 16,389행 중 16,316행이 G-ORS이고, 양성 1,066개도 모두 G-ORS에 있다. 따라서 no-peer F1 0.8203과 peer F1 0.8629의 차이를 peer feature의 독립 효과로 해석할 수 없다. station과 거의 완전히 교락됐다.
- depth missing은 750행(0.178%)뿐이며 양성은 1행이다. XGBoost는 그 한 행을 맞혔고 FP/FN은 0이지만 표본이 너무 작다. 특히 test의 G-ORS depth 전부 결측 조건에 대한 일반화 증거가 아니다.
- G-ORS no-peer FN 324개 중 drift 170개는 전부 놓쳤고 `noise+drift` 95개 중 84개를 놓쳤다. peer 없이 작동하는 장기 기준선이 별도 연구 대상이다.

## 7. 우선순위가 높은 돌파 가설

아래 순위는 **다음 실험 가설의 우선순위**이지 후보 승격 순위가 아니다. outer에서 발견된 수치로 parameter를 고르지 말고, 설정과 grid를 사전 고정한 뒤 outer train 내부의 과거 blocked split에서만 선택한다. 이미 본 outer는 이후에도 연구 진단으로만 남겨야 한다.

| 우선순위 | 가설 | 예상 효과 | 누출·과적합 위험 | 비용 | 근거와 안전한 검증 방식 |
|---:|---|---|---|---|---|
| P0 | slow-anomaly 시작·끝 경계 전용 head: CUSUM/change-point, 좌우 robust baseline 차이, event interior seed 후 제한적 boundary completion | 높음 | 중간 | 중간 | FN 92.7%가 offset/drift 계열, 77.0%가 48시간 이상. inner event 단위 grid와 event-boundary metric만 사용하고 spike singleton은 보존 |
| P1 | station/depth-regime categorical interaction을 가진 offset/drift expert와 shared model의 mixture | 높음 | **높음** | 중간 | 네 station-layer×type 조합이 FN 59.6%. outer group threshold 금지; inner fold마다 expert를 다시 fit하고 unseen category fallback 강제 |
| P2 | XGBoost 확률, TCN/Transformer 확률, rule seed, vote/uncertainty를 입력으로 한 nonnegative residual gate | 중~높음 | **매우 높음** | 중간 | XGB FN 56.4%가 타 모델에서 recoverable, FP 86.6%가 rejectable. outer oracle F1 0.9515는 상한일 뿐이며 gate 학습·threshold는 inner OOF에만 한정 |
| P3 | S-ORS 계절·성층 상태 gate로 정상 고변동 구간의 false event 억제 | 중간 | **높음** | 낮~중간 | 6월 S-ORS가 전체 FP 57.8%. 월 고정 규칙 대신 peer spread, 층간 상관, 장·단기 variance ratio로 상태를 표현하고 year-transfer로 검사 |
| P4 | 합성 증강의 amplitude·duration·regime matching과 prior correction 재설계 | 중간 | 낮~중간 | 중~높음 | recall은 증가했지만 FP가 13.1배. 합성 행 비율과 세기 분포를 fold train 정상구간 통계로만 정하고, 비증강 대비 precision guardrail 강제 |
| P5 | no-peer/G-ORS 전용 reference fallback 및 depth-mask dropout | 중간이나 불확실 | 중간 | 낮~중간 | no-peer G-ORS slow FN이 집중되지만 station과 교락. supervised G holdout 및 depth 강제 masking에서만 채택 판단 |
| P6 | spike용 차분·3점 복귀도 rule/tree specialist 유지 | 낮은 row-F1, 높은 event 가치 | 낮음 | 낮음 | XGB spike FN은 11행뿐이나 하루 1회 제출에서 singleton 제거는 위험. 다른 후처리가 spike를 지우지 않는 회귀시험 유지 |

현실적인 첫 실험 순서는 P0의 boundary representation을 tree에 추가하고, P1 expert를 별도 ablation으로 측정한 뒤, 두 모델의 inner OOF가 안정적일 때만 P2 gate를 시도하는 것이다. P3은 6월이라는 outer 월 자체를 조건으로 쓰지 않고 물리 상태 지표로 일반화해야 한다.

## 8. 범위, 정의와 방법

### 분석 population

- 세 outer fold의 합집합: 421,032행, 양성 16,055행, 2025-04-01~2025-12-10 KST
- 공식 key: `station, year, layer, time`; 중복 0, 결합 누락 0
- true event: 동일 station-layer에서 label=1이 10분 cadence로 연속되는 구간. 관측 gap과 group 경계를 넘지 않는다.
- composite row: `anomaly_type`에 둘 이상의 membership이 있는 행
- composite event: 한 true event 전체에서 합친 유형 membership이 둘 이상인 event
- FP run: 동일 station-layer에서 10분 cadence로 연속되는 `(label=0, prediction=1)` 구간
- score band: 각 모델이 저장한 probability의 고정 구간 `<.05, .05-.10, .10-.20, .20-.40, .40-.60, .60-.80, .80-.95, >=.95`; prediction은 저장된 fold별 후처리 결과를 그대로 사용

### 입력 artifact

| 모델 | OOF provenance |
|---|---|
| LightGBM | `artifacts/runs/20260813T144012+0900_cv_16e20929/oof.parquet` |
| 증강 LightGBM | `artifacts/runs/20260813T151648+0900_cv_378a4e89/oof.parquet` |
| XGBoost | `artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet` |
| CatBoost | `artifacts/runs/20260813T154234+0900_cv_378a4e89/oof.parquet` |
| TCN | `artifacts/sequence_full_20260813/oof_tcn.npz` |
| Patch Transformer | `artifacts/sequence_full_20260813/oof_patch_transformer.npz` |

각 tree run의 `metrics.json`·`manifest.json`, deep의 `sequence_experiment.json`, SHA 검증된 offline feature cache를 함께 읽었다. 원본 train SHA-256은 `20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2`다. 외부 관측값은 0건 사용했다.

### 생성 결과

- 재현 스크립트: `scripts/analyze_oof_failures.py`
- aggregate summary: `artifacts/failure_recon_20260813/research_only_failure_summary.json`
- 실행 manifest: `artifacts/failure_recon_20260813/manifest.json`
- summary SHA-256: `bd3acdfbc4542ffc8fed8d6cf3ec807e314b339a2bdcafa4b1680da148f8d44e`
- raw observation rows exported: 0
- 시각화 생략: 이 Markdown 산출물은 정확한 error count와 audit 가능한 교차표가 주목적이며, raw row나 시계열 모양을 노출하지 않기 위해 작은 근거 표를 사용했다.

## 9. 검증 보고서

### Overall assessment: Share with caveats

failure concentration과 모델 disagreement에 대한 기술적 진단으로는 재현 가능하고 집계가 검증됐다. 그러나 outer 라벨을 본 사후 분석이므로 후보 승격이나 기대 leaderboard 점수의 근거로 공유해서는 안 된다.

### 계산 spot-check

- 여섯 OOF 모두 별도 직접 재집계로 TP/FP/FN/TN이 summary와 일치했다.
- station, layer, station-layer, fold, month, anomaly signature, composite, event duration, peer, depth, score band와 교차 dimension의 모든 additive subtotal이 모델 전체 confusion matrix에 일치했다.
- true event duration/composite 두 경로가 모두 141 events·16,055 양성 행으로 일치했고, 각 모델의 event FN 합이 row FN과 일치했다.
- FP-run 행 합이 각 모델의 FP와 일치했다.
- vote-count 행 합은 421,032이고, unanimous error 1,500행은 label-aware oracle의 잔여 FN 1,354 + FP 146과 일치했다.
- summary 파일 해시가 manifest의 SHA-256과 일치했다. manifest에는 aggregate reconciliation 13개가 `passed`로 기록됐다.

### 중요한 한계

1. station-layer/month/type 교차군은 outer 라벨을 본 뒤 발견됐으므로 효과 크기가 낙관적일 수 있다.
2. event 정의는 label 연속 구간이다. 서로 다른 주입 event가 gap 없이 맞닿으면 하나로 합쳐질 수 있어 event 수와 duration 해석에 제한이 있다.
3. peer/no-peer는 G-ORS와 거의 완전히 교락됐고 depth-missing 양성 표본은 1행뿐이다.
4. 유형 recall은 복합 라벨 membership을 중복 계산하므로 additive하지 않는다.
5. probability scale은 모델과 fold마다 calibration이 다르다. score band는 오류 위치 진단이지 공통 threshold 제안이 아니다.
6. oracle은 outer 라벨을 행별로 사용한 구현 불가능한 상한이다. 모델 성능이나 예상 uplift가 아니다.
7. 공식 기준값 0.548255와 이 outer population의 수치는 직접 비교할 수 없다.

## 10. 재현 명령과 실제 검증 결과

```powershell
$env:P1_DATA_DIR = "운영진이 제공한 P1_qc_anomaly 폴더"
.venv-p1\Scripts\python.exe scripts\analyze_oof_failures.py --acknowledge-research-only
.venv-p1\Scripts\python.exe -m pytest tests\test_analyze_oof_failures.py
.venv-p1\Scripts\python.exe -m ruff check scripts\analyze_oof_failures.py tests\test_analyze_oof_failures.py
```

2026-08-13 KST 실제 실행 결과:

- analysis exit code 0, outer rows 421,032, 모델 6개
- unit tests `5 passed`
- Ruff `All checks passed`
- aggregate reconciliation `13 checks passed`, true events 141
- raw rows exported 0, external observations used 0
- 제출 업로드, 외부자료 다운로드, commit, push 모두 수행하지 않음

## 11. 남은 질문

- slow-anomaly boundary completion의 inner-only 개선이 station-layer와 fold 전부에서 유지되는가?
- S-ORS 6월 FP 집중을 월 정보 없이 stratification/variance/peer-state 지표로 재현할 수 있는가?
- G-ORS drift blind spot이 no-peer 때문인지 단층 station regime 자체 때문인지 supervised G holdout에서 분리 가능한가?
- deep probability를 calibration한 뒤에도 XGBoost가 놓친 slow rows에서 순증가가 남는가?
- 새 가설을 평가할 독립 label set이 없는 상태에서, 어떤 증거를 승격 기준으로 인정할지 사용자와 사전 합의할 필요가 있다.
