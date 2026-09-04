# 병렬 Deep Research gap matrix

| 문제 | 결정에 필요한 주장 | 현재 근거 | confidence | 모순·위험 | 다음 확인 |
|---|---|---|---|---|---|
| P1 | proposal 단위 2차 판별이 행 단위 score 재학습보다 precision을 높일 수 있다 | 기존 83 proposal 중 10개만 truth와 매칭; event-localized 학습과 엄격한 단순 baseline을 강조하는 1차 문헌 발견 | 중간 | proposal가 놓친 16개 이벤트는 ranker로 복구 불가 | ranker가 recall을 지나치게 줄이지 않는 사전 고정 학습·threshold 계약 |
| P1 | 복잡한 deep detector보다 단순 event feature ranker가 현재 작은 support에 적합하다 | ICML 2024 TAD position paper가 복잡도보다 평가·단순 baseline 검증을 강조 | 중간 | position paper는 P1의 supervised injected anomaly와 동일 과제가 아님 | 실제 support 수와 parameter budget을 연결한 실행 계약 |
| P2 | 1~2개 vertical-displacement mode가 공개층 상태에서 중간층 residual을 설명할 수 있다 | 내부파 관측에서 mode-1/mode-2 온도 변위가 물리적으로 관찰됨 | 중간 | 계절 혼합·수괴 변화는 순수 수직변위가 아니며 기존 soft gate와 중복 가능 | target-masked forward folds에서 mode correction과 exact no-op 비교 |
| P2 | profile projection 뒤 bounded modal correction이 incumbent를 안전하게 보존한다 | 기존 endpoint projection은 로컬 RMSE를 안정적으로 낮춤 | 중간-높음 | 현재 강한 adaptive v2는 이미 노출된 OOF에 맞춰짐 | 가장 강한 공통 OOF comparator와 2/3 fold·CI·개입률 gate |
| P3 | transfer와 incumbent의 cross-fitted stack/router가 단일 모델보다 나을 수 있다 | 시간예측 stacking의 cross-validated 이론·실험 근거; 기존 transfer/local delta가 이질적 | 중간-높음 | 181 episodes로 고용량 gate는 과적합 위험 | station/lead/current/prediction-difference만 쓰는 저차원 cross-fit |
| P3 | selective fallback은 불확실한 구간에서 incumbent no-op을 보존할 수 있다 | selective regression은 coverage와 error의 trade-off를 직접 다룸 | 중간 | conditional variance 추정 자체가 작은 표본에서 불안정 | leave-one-fold-out에서 coverage, worst-fold, CI, exact fallback 확인 |

## 현재 확인한 1차 출처

- Sarfraz et al., “Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?”, ICML 2024, PMLR.
- Adams and Marlin, “Learning Time Series Detection Models from Temporally Imprecise Labels”, AISTATS 2017, PMLR.
- van Haren et al., “A comparison between vertical motions measured by ADCP and inferred from temperature data”, Ocean Science 2008.
- Hasson et al., “Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting”, ICML 2023, PMLR.
- Noskov et al., “Selective Nonparametric Regression via Testing”, ACML 2023/2024 proceedings, PMLR.
- Sugiyama and Storkey, “Mixture Regression for Covariate Shift”, NeurIPS 2006.

## 미해결 핵심

1. P1 ranker는 83개 proposal을 동결하고 chronological 40/30/30 split과 15일 purge를 사용한다. 소표본 support가 부족하면 fit 전에 종료한다.
2. P2 mode는 공개층 residual에서 displacement amplitude를 직접 식별하는 tangent correction이며, 출력 PAVA와 크기 gate인 기존 구조와 구분된다. 다만 현 로컬 1순위와 같은 exposed OOF 비교임을 명시한다.
3. P3 router는 station·lead one-hot·특정 lead subset을 금지하고 continuous lead와 past-only dynamics만 사용해 사후 G-ORS/lead-24 hard-coding을 차단한다.

## 1차 근거 통합 판정

| 문제 | 선택 구조 | 제외 구조 | 실행 가치 | 남은 한계 |
|---|---|---|---|---|
| P1 | Frozen-83 recall-guarded L2 logistic event ranker | 새 generator, deep verifier, DAMP 단독 veto | 73 false proposals를 줄일 직접 시험 | 10 positive proposal이라 정식 recall 보장은 불가 |
| P2 | Public-only 1-mode thermocline-heave tangent correction | 2-mode, nonlinear depth warp, 새 PAVA | 기존 예측에 없는 물리적 displacement amplitude를 시험 | exposed OOF, 수평이류와 heave 혼동 |
| P3 | Low-capacity forward-cross-fit ERA5 advantage router | station/lead hard rule, 고용량 CatBoost/신경 MoE | incumbent no-op을 보존하며 transfer 이질성을 시험 | 181 episodes와 강한 source/local shift |

## 실행 계약의 공통 중단선

- 사전 support 또는 bit-exact incumbent/no-op preflight 실패 시 fit 없이 종료한다.
- 결과를 본 뒤 feature, threshold, mode count, blend strength, fold를 변경하지 않는다.
- 공식 입력·제출 CSV·업로드는 세 실험 모두 금지한다.
- PASS는 공식 점수 개선 보장이 아니라 제한된 공식 probe 후보 자격만 뜻한다.
