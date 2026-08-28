# P1·P2·P3 병렬 딥리서치 실행 보고

작성일: 2026-08-28 KST
범위: 로컬 연구, 구조 탐색, 사전 등록된 bounded 실행, 독립 QA
금지 범위: 공식 test/sample/submission 열람, 후보 CSV 생성, 업로드

## 결론

이번 사이클에서 **즉시 공식 제출할 후보는 없습니다.** 그러나 세 문제 모두 이전보다 병목이 더 정확하게 분해됐습니다.

1. **P1**: 실제-event donor를 사용하자 학습 지원성은 22 events/7 cells로 회복됐습니다. 하지만 calibration event recall 38.46%, row precision 22.27%로 품질 gate를 통과하지 못했습니다. 병목은 더 이상 “양성 support 없음”이 아니라 “proposal 품질과 event-level localization”입니다.
2. **P2**: α40 위 quasi-periodic uncertainty-gated residual은 안전했지만 69,850행 중 75행만 보정해 ΔRMSE가 -0.0000086°C에 그쳤습니다. 현재 GP/support gate 조합은 사실상 no-op입니다.
3. **P3**: ERA5 source 학습과 transfer viewpoint는 과학적으로 성공했습니다. source persistence 대비 -0.14090m, transfer는 matched local-only 대비 -0.02344m였고 3/3 local window를 개선했습니다. 그러나 incumbent 대비 +0.00233m 악화, S-ORS +0.02170m 악화로 solution gate는 실패했습니다.

따라서 다음 우선순위는 **P3 incumbent-preserving ERA5 residual/router > P2 two-mode vertical displacement > P1 event-level ranker/normality veto**입니다. 기존 실패 모델의 단순 재튜닝은 우선순위에서 제외합니다.

## 공통 판정 원칙

- 로컬 metric이 조금 좋아졌다는 이유만으로 공식 후보를 만들지 않습니다.
- 반대로 공식 점수와 로컬 순위의 괴리가 확인된 P2에서는 작은 양의 추세를 기록하되, effect size와 fold sign을 함께 봅니다.
- 모든 새 실험은 결과를 보기 전에 구조·분할·게이트·hash를 고정합니다.
- 이미 노출된 historical surface는 “fresh holdout”이라고 부르지 않습니다.
- 공식 33점 스케일과 로컬 물리 단위(°C, m)는 직접 같은 숫자로 비교하지 않습니다.

## P1 — 조건부 실제-event donor 합성

### 가설

전역 synthetic anomaly가 아니라 pre-Q2 실제 장기 이벤트 조각을 동일 station/layer/quarter의 정상 구간에 제한적으로 이식하면, 양성 proposal support와 cell 다양성을 동시에 늘릴 수 있습니다. CAROTS는 인과관계를 보존하거나 교란하는 증강으로 정상/이상 표현을 분리하고, TimeMIL은 sparse/local temporal pattern을 weakly supervised MIL로 다룹니다. 이 문헌은 P1에 직접적인 성능 보장을 주지 않지만 관계보존 증강과 event-level 학습의 근거를 제공합니다.

### 실행 계약

- pre-Q2 이력만 fit/calibration에 사용
- 15일 purge
- 실제 donor 최소 10 events, 3 cells, 최대 cell share 70%
- 단일 seed/설정, threshold search 0, result-based retry 0
- Q2 outer truth는 calibration 통과 전 접근 금지
- 공식/Q3/Q4 접근 금지

### 결과

| 항목 | 결과 | 판정 |
|---|---:|---|
| Train donor | 22 events / 7 cells | PASS |
| 최대 train cell share | 31.82% | PASS |
| Calibration truth | 26 events / 9 cells | PASS |
| 합성 양성 | 445 rows | 단일 설정 |
| Proposal | 83 | 기록 |
| Matched truth events | 10/26 | 38.46%, FAIL |
| Row precision | 22.27% | 기준 45%, FAIL |
| Row F1 | 24.37% | 기록 |
| Q2 truth read | 0 rows | PASS |

최종 판정은 `NO_GO_CALIBRATION`, 선택 arm은 `ZERO_ADD_NO_OP`입니다. support 문제는 해소됐으나, 넓은 proposal이 정상 구간까지 포함해 precision/recall이 동시에 부족합니다.

### 다음 P1 실험

새 generator를 더 넓히지 말고 현재 83 proposals에 event-level ranker 또는 station/layer/season normality veto를 붙이는 한 번의 실험이 우선입니다. calibration에서 precision 45%, event recall 80%를 동시에 넘지 못하면 이 donor 축을 종료합니다.

## P2 — α40 + quasi-periodic uncertainty-gated residual

### 가설

α40의 강한 계절 조건부 평균을 유지하고 작은 quasi-periodic residual만 보정하면, 내부조석의 짧은 위상 지속성과 수심 간 공분산을 포착하면서 out-of-support에서 정확히 α40으로 돌아갈 수 있습니다. GPR은 joint temperature/salinity reconstruction과 posterior uncertainty를 제공하는 선행 근거가 있고, 내부조석은 density-surface vertical displacement와 low vertical modes로 설명됩니다.

### 실행 계약

- architecture-fresh committed forward folds 3개
- hyperparameter search 0
- truth metric 전에 prediction hash 고정
- out-of-support byte-exact α40 no-op
- pooled ΔRMSE ≤ -0.003°C, CI90 upper < 0, 2/3 folds 개선
- correction RMS ≤ 0.05°C, p99 |correction| ≤ 0.20°C

### 결과

| 항목 | 결과 | 판정 |
|---|---:|---|
| 평가 rows | 69,850 | 기록 |
| α40 reference RMSE | 2.5477199871°C | 기준 |
| GP candidate RMSE | 2.5477113814°C | 기록 |
| ΔRMSE | -0.000008606°C | 효과 기준 FAIL |
| Bootstrap CI90 | [-0.000019683, -0.000002253]°C | sign PASS |
| 개선 folds | 1/3 | FAIL |
| 보정 활성 | 75 rows (0.107%) | 거의 no-op |
| Correction RMS | 0.006181°C | 안전성 PASS |

최종 판정은 `FAIL_GATE_STOP_NO_CSV_NO_RESEARCH_LOOP`입니다. posterior uncertainty와 support gate가 99.893%를 no-op으로 되돌려, 통계적 sign은 양수지만 제출 가치가 없는 effect size가 됐습니다.

### 다음 P2 실험

같은 GP의 threshold를 결과에 맞춰 완화하지 않습니다. 다음은 α40 profile의 실제 수심 기울기에 1–2개 수직변위 mode만 적용하는 물리적으로 다른 보정입니다. 내부조석 연구에서 two-mode approximation이 vertical displacement 구조를 설명한 것은 설계 근거이나, P2 성능 크기로 직접 이전하면 안 됩니다.

## P3 — ERA5 context-transfer dependency recovery

### 가설과 환경 복구

기존 v1은 raw/derived 363/363, 262,917 rows, 286 features까지 준비됐으나 CatBoost import 전에 종료돼 model fit 0회였습니다. 새 `p3_era5_context_transfer_dependency_recovery_20260828_v2`는 과학 계약을 바꾸지 않고 experiment ID와 output path만 새로 분리했습니다.

실행 전 Python 3.12.10, CatBoost 1.2.10, scikit-learn 1.9.0, NumPy 2.3.5, pandas 3.0.1, pyarrow 25.0.1을 확인했습니다. 32×286 synthetic NaN smoke에서 CatBoost 2-tree fit, deepcopy, `init_model` 4-tree continuation과 finite prediction을 검증했습니다. CPU를 사용한 이유는 CatBoost 공식 문서가 GPU 학습의 부동소수 합산 순서 비결정성을 명시하기 때문입니다.

### Source 결과

| 항목 | 결과 | 판정 |
|---|---:|---|
| Source train/held cases | 7,311 / 492 | 기록 |
| Held years | 2021, 2022, 2023 | 고정 |
| Persistence RMSE | 0.686892m | 기준 |
| ERA5 CatBoost RMSE | 0.545996m | 개선 |
| ΔRMSE | -0.140897m | PASS |
| Bootstrap CI90 | [-0.170289, -0.113398]m | PASS |
| 3 held years | 모두 개선 | PASS |

ERA5는 wave data가 대기모델과 다른 약 0.36° grid에 놓인 reanalysis이므로 point buoy와 강한 domain gap이 예상됩니다. 실제 domain classifier AUC는 0.9999999였고, 직접 pooling/pretrain-only 사용은 계속 금지됩니다.

### Local transfer 결과

| 비교 | ΔRMSE | Fold sign | CI90 / 중요 slice | 판정 |
|---|---:|---:|---|---|
| Transfer - matched local-only | -0.023440m | 3/3 개선 | CI90 [-0.040081, -0.007755] | Viewpoint PASS |
| Transfer - incumbent | +0.002325m | 1/3 개선 | CI90 [-0.019052, +0.023029] | Solution FAIL |
| G-ORS | -0.022413m | - | 개선 | 기록 |
| S-ORS | +0.021700m | - | critical cap FAIL | FAIL |
| Lead 24h | -0.024416m | - | 개선 | 기록 |

최종 판정은 `NO_GO_LOCAL_OR_VIEWPOINT_GATE`이며 실제로는 viewpoint gate PASS, incumbent solution gate FAIL입니다. ERA5 사전학습은 동일 CatBoost 계열을 유의하게 개선했지만 incumbent를 대체할 수준은 아닙니다.

### 다음 P3 실험

우선순위는 incumbent를 그대로 유지하고 ERA5가 잘한 G-ORS 및 12/18/24h residual만 보수적으로 적용하는 새 hypothesis-exposed router입니다. 이는 이번 local slice를 보고 세운 가설이므로 confirmatory local claim을 할 수 없고, 별도 sealed architecture 또는 제한된 공식 probe가 필요합니다. TimeXer는 source signal PASS/local solution FAIL 조건이 충족돼 2순위로 해제되지만 구현비용이 크므로 router 다음입니다. TimeXer는 endogenous patch와 exogenous variable을 구분해 결합한다는 점에서 단순 past-only MLP와 다른 구조입니다.

## 공식 제출 판정

| 문제 | 현재 후보 | 공식 CSV | 판정 |
|---|---|---:|---|
| P1 | 조건부 실제-event donor | 생성 안 함 | NO_GO |
| P2 | α40 + quasi-periodic residual | 생성 안 함 | NO_GO |
| P3 | ERA5 context transfer | 생성 안 함 | NO_GO |

이번 사이클의 값어치는 제출 수를 소모한 데 있지 않고, 세 문제의 다음 실험을 좁힌 데 있습니다. 공식 제출 기회는 effect size와 구조적 차이가 있는 다음 후보에 보존합니다.

## 근거와 한계

### 1차 문헌·공식 문서

1. Kim et al. (2025), CAROTS, ICML/PMLR. https://proceedings.mlr.press/v267/kim25aa.html
2. Chen et al. (2024), TimeMIL, ICML/PMLR. https://proceedings.mlr.press/v235/chen24af.html
3. Chen et al. (2023), 3D temperature/salinity anomaly reconstruction with GPR, Frontiers in Marine Science. https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1121334/full
4. Bendinger et al. (2024), internal-tide vertical structure and two-mode projection, Ocean Science. https://os.copernicus.org/articles/20/945/2024/
5. ECMWF/Copernicus, ERA5 data documentation. https://confluence.ecmwf.int/pages/viewpage.action?pageId=177484060
6. CatBoost, GPU nondeterminism documentation. https://catboost.ai/docs/en/features/training-on-gpu
7. Wang et al. (2024), TimeXer, NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html

### 한계

- P1/P2 historical folds는 저장소의 과거 연구에서 노출된 surface이며 완전 fresh holdout이 아닙니다.
- P1 문헌은 P1의 긴 interval F1과 동일한 과제가 아닙니다.
- P2 내부조석·GPR 문헌의 수치 개선을 단일 station 해커톤 점수로 이전할 수 없습니다.
- P3 ERA5 source 성공은 local incumbent 승리를 보장하지 않으며 domain AUC가 거의 1입니다.
- 공식 점수 개선량은 새 후보를 실제로 제출하기 전에는 추정할 수 없습니다.

## 재현·무결성

- P1 result SHA-256: `b32fc6df07b30d315d1d3b09add4455686660ea69ba3b10134ac7f4e0a8c58f4`
- P2 result SHA-256: `c04755750357b8613f7372f98840ba8f8df365173af46524b4be339ee362da2e`
- P3 result SHA-256: `ac92a530d230ea29c475e0b03acb7e16d577633b64401632cebb46fa4e0bbd2f`
- P3 blind seal SHA-256: `25accc81915e95bebcf4e69cd313b73520c36969b88521a186f5be214c4ba2a7`
- P3 sealed rows: 1,086; outcome column absent before truth attachment

세 실험의 코드·설정·결과·QA 경로는 동봉된 claim ledger와 independent QA receipt에 기록합니다.
