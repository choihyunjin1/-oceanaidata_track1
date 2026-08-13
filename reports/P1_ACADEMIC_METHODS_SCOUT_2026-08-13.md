# P1 학술 문헌 기반 점수 향상 정찰 — 2026-08-13

상태: 1차 출처 중심 방법론 정찰 및 다음 실험 설계. 외부 해양 관측값 다운로드·사용 0건, 코드·설정·모델 변경 없음, 대회 제출 없음.

## 1. 기술 요약

현재 최선인 양방향 offline XGBoost는 동일한 세 outer holdout에서 micro F1 0.860371을 기록했지만, 이는 공식 test 점수가 아니라 train 라벨을 사용한 로컬 OOF 추정치다. XGBoost의 FN 3,109행 중 2,882행(92.7%)이 offset/drift 계열이고 2,393행(77.0%)이 48시간 이상 이벤트에 속한다. 긴 이벤트 17개 중 14개는 적어도 한 행을 이미 탐지했다. 즉 주된 병목은 이벤트 존재 여부보다 **잡힌 씨앗을 정확한 시작·종료까지 확장하는 구간 완성도**다.

학술 문헌과 이 실패 구조를 함께 보면 다음 실험 순서가 가장 타당하다.

1. XGBoost 확률을 버리지 않고 CAPA/epidemic PELT와 semi-Markov 동적계획법으로 이상 구간의 양쪽 경계를 복원한다.
2. 후보 구간의 중앙을 가린 뒤 정상 앞·뒤 문맥으로 수온을 양방향 예측해, 실제값과 반사실적 정상 궤적의 차이를 offset·drift·noise 특징으로 만든다.
3. drift에는 1차 차분만 쓰지 않고 CPOP의 연속 piecewise-linear slope-change 특징을 추가한다.
4. 후보 구간 단위 분류기를 inner fold에서만 학습해 경계 점수, 지속시간, XGBoost 확률 요약, 반사실적 잔차, 층간 상태를 결합한다.
5. 층간 정보는 최근 상관·수직 온도구배·peer spread가 안정적인 때만 켜고, G-ORS/no-peer에는 단변량 fallback을 유지한다.

표준 Matrix Profile, 전면적 GNN, 새 대형 Transformer, 전역 합성 증강은 현재 우선순위가 낮다. 특히 표준 Matrix Profile의 subsequence별 z-normalization은 순수 level offset을 제거할 수 있어 가장 큰 실패군과 구조적으로 맞지 않는다.

## 2. 증거와 공식성 경계

### 2.1 로컬에서 확인된 사실

- 로컬 OOF population은 세 7일 purge rolling-origin outer validation의 합집합 421,032행이며 양성은 16,055행이다.
- XGBoost는 TP 12,946, FP 1,093, FN 3,109, precision 0.922145, recall 0.806353, micro F1 0.860371이다.
- 유형별 recall은 spike 0.8197, noise 0.9355, flatline 0.9997, offset 0.6492, drift 0.6462다.
- 48시간 이상 이벤트의 event hit는 14/17이지만 row recall은 0.6315다.
- TCN은 offset/drift recall을 각각 0.7858/0.6966까지 높였지만 FP 5,789행 때문에 micro F1 0.7676에 그쳤다.
- 6월 S-ORS FP가 632행으로 전체 XGBoost FP의 57.8%다. 다만 이 상관만으로 계절 성층이 원인이라고 확정할 수는 없다.

위 값의 재현 근거와 해시는 [실패 재정찰 보고서](P1_FAILURE_RECON_2026-08-13.md)에 있다. 이 값들은 hidden test의 공식 결과가 아니며, 이미 여러 차례 본 outer 라벨은 새 설정 선택에 재사용하지 않는다.

### 2.2 문헌 사실과 우리의 적용 추론

이 보고서에서 “논문이 보였다”와 “P1에서 좋아질 것이다”는 같은 문장이 아니다. 논문은 각 방법의 원래 문제·가정·실험 결과만 뒷받침한다. P1 적용법과 우선순위는 로컬 실패 구조를 바탕으로 세운 가설이며, 점수 향상을 보장하지 않는다. 모든 페널티·기간·임계값·결합 가중치는 outer train 내부의 과거 방향 inner split에서만 정한다.

## 3. 가장 직접적인 방법: 점이 아니라 구간을 예측한다

### 3.1 CAPA와 epidemic change-point

[Fisch, Eckley, Fearnhead의 CAPA](https://doi.org/10.1002/sam.11586)는 point anomaly와 평균·분산이 달라진 collective anomaly를 하나의 penalized dynamic program에서 구분한다. [Juodakis와 Marsland의 epidemic change 모델](https://doi.org/10.1007/s00362-022-01307-x)은 느린 nuisance background 속에서 유한 기간 동안 나타났다가 기준선으로 돌아오는 시작·종료 쌍을 명시적으로 다룬다. [PELT](https://doi.org/10.1080/01621459.2012.737745)는 구간 비용 합과 변화점 페널티의 전역 최적 분할을 유지하면서 후보를 pruning하며, 논문의 충분조건 아래 기대 선형 시간 복잡도를 갖는다.

P1 적용 추론:

- 현재 XGBoost의 양성 run 또는 높은 확률 행을 이벤트 내부 seed로 사용한다.
- station-layer의 gap-safe robust residual에서 mean-shift cost는 offset, variance cost는 noise, piecewise-linear regression cost는 drift 후보를 만든다.
- 시작과 종료를 서로 독립 threshold로 찾지 말고 “정상 → 이상 → 정상”의 paired interval로 평가한다.
- 공개된 지속시간 범위는 탐색 범위와 특징으로 사용하되 hard clipping은 하지 않는다. 중첩·맞닿은 이벤트가 단일 유형 최대 길이보다 길어질 수 있기 때문이다.
- CAPA/PELT 결과를 최종 라벨로 바로 쓰지 않고 boundary gain, 전·중·후 median, return-to-baseline, 지속시간, XGBoost coverage를 segment classifier의 입력으로 쓴다.

이 방법이 현재 최우선인 이유는 48시간 이상 true event 17개 중 14개에 이미 XGBoost seed가 있어, 완전히 새로운 detector보다 **seeded boundary completion**이 더 작은 탐색 문제이기 때문이다.

### 3.2 Semi-Markov 지속시간 디코더

[Sarawagi와 Cohen의 semi-Markov CRF](https://proceedings.neurips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html)는 개별 원소가 아니라 가변 길이 segment에 label을 주며, 구간 전체의 비-Markov 특징을 사용할 수 있고 정확 추론이 다항 시간임을 보였다.

P1에서는 완전한 CRF부터 학습할 필요가 없다. 다음처럼 작은 semi-Markov decoder를 먼저 구현할 수 있다.

- row emission: 고정된 XGBoost logit과 유형 전문가 score
- boundary score: CAPA/PELT/CPOP gain
- segment score: 확률 합·분위수, duration, 반사실적 잔차 평균·기울기·분산, peer gate
- transition: normal↔anomaly 전환 비용
- 보존 규칙: plateau hard override와 spike singleton은 decoder가 제거하지 못하게 고정

공개 지속시간의 10분 row 환산은 spike 1, noise 약 18~353, flatline 약 12~283, offset 48~519, drift 54~519이다. 이는 운영진이 합성에 사용한 범위이므로 구조적 prior 후보로는 정당하지만, 최종 비용·soft constraint는 inner fold에서만 고른다.

## 4. 후보 구간을 정상 궤적과 비교한다

[Jones 등 2022의 pyhydroqc 연구](https://doi.org/10.1016/j.envsoft.2022.105364)는 5~6년의 고빈도 aquatic sensor 자료와 기술자 라벨을 사용해 ARIMA, LSTM, Prophet 기반 동적 임계값을 비교했다. 해당 사례에서는 ARIMA가 일관되게 가장 좋았고, 여러 모델 결과의 집계가 탐지를 개선했다. 특히 local model의 forecast와 backcast를 섞어 며칠 길이의 이상 구간에 대한 correction estimate를 만들었다.

P1 적용 추론:

1. CAPA/PELT/XGBoost가 제안한 후보 구간과 작은 buffer를 학습 문맥에서 제거한다.
2. 앞쪽 정상 문맥으로 forward forecast, 뒤쪽 정상 문맥으로 backward forecast를 만든다.
3. 두 예측을 후보 내부 위치에 따라 가중 혼합해 반사실적 정상 수온을 만든다.
4. 실제값과 반사실적 궤적의 차이에서 다음을 계산한다.
   - offset: residual median/trimmed mean과 양 끝의 return score
   - drift: residual robust slope, monotonicity, 시작·끝 slope change
   - noise: residual과 1차 차분의 MAD/variance ratio
   - 불확실성: forward/backward disagreement와 context residual scale

첫 구현은 모든 행에 ARIMA를 반복하지 않는다. 후보 interval에만 강건 AR/ridge 또는 local linear state model을 적용해 계산량을 제한하고, 이것이 inner OOF에서 이득을 보일 때만 ARIMA를 ablation한다. P1의 최장 단일 이상 약 3.6일은 논문이 다룬 “수일” correction과 규모가 유사하지만, 다른 센서·환경의 결과이므로 성능 수치는 전이하지 않는다.

## 5. Drift는 1차 차분만으로 해결하지 않는다

[Fearnhead와 Grose의 CPOP](https://doi.org/10.18637/jss.v109.i07)은 연속 piecewise-linear 신호에서 L0 penalized dynamic programming으로 slope change를 찾는다. 논문은 원 신호를 1차 차분한 뒤 평균 변화 PELT를 적용하면 변화 위치 정보가 손실되어 성능이 나빠질 수 있음을 구체적으로 지적한다. CPOP 구현은 불규칙 관측, 이분산 잡음, 관측점과 다른 변화점 grid도 지원한다.

P1 적용 추론:

- gap-safe robust background residual의 원 형태에 CPOP-like score를 계산한다.
- 특징은 전후 slope, slope-change 크기, knot까지 거리, piecewise-linear SSE 개선, 양 끝에서 정상 slope로 복귀하는지 여부다.
- 1차 차분 특징은 spike/noise에는 유지하되 drift boundary의 유일한 표현으로 쓰지 않는다.
- 전체 구간 CPOP가 비싸면 XGBoost seed 주변 ±4~7일 candidate window와 축소한 변화점 grid에서 먼저 시험한다.

[ℓ1 trend filtering](https://doi.org/10.1137/070690274)도 piecewise-linear trend를 만들 수 있지만, L1 penalty가 한 변화를 같은 부호의 여러 변화로 표현하거나 과평활할 수 있다. filtered trend를 정상 baseline으로 빼버리면 주입 drift 자체가 사라질 수 있으므로, trend residual만 thresholding하지 않고 **추정된 trend의 knot와 slope 변화 자체**를 특징으로 보존한다.

## 6. 층간 정보는 동적으로 켜고 끈다

[Min 등 2021의 이어도 해양과학기지 연구](https://doi.org/10.4217/OPR.2021.43.4.229)는 P1과 가장 가까운 현장 근거다. 2013년 이어도 3 m, 20.5 m, 38 m의 10분 수온을 사용했고, 여름 중층에서 내부조석·관성파 때문에 10분 사이 최대 7.3°C의 정상 급변을 관측했다. 변동성이 큰 정상 사례에서 NDBC 검사는 3,964개 중 707개를 flag했지만 OOI 검사는 42개를 flag했다. 전체 중층에서도 flag 비율은 각각 4.77%와 0.30%였다. 이는 “한 층만 빠르게 변했다”는 이유만으로 이상을 판정하면 대량 오탐이 생길 수 있다는 직접 증거다.

[Smith 등 2012의 해양 센서 Bayesian 품질평가](https://doi.org/10.3390/s120709476)는 계절편차, gradient, 수치모델 잔차, 다른 깊이 센서를 evidence로 결합했고, 층간 비교는 두 센서가 pycnocline의 같은 쪽에 있을 때 특히 유효하다고 설명한다. [Diamant 등 2020의 다층 해양 관측 연구](https://doi.org/10.3390/rs12213470)는 후보 이상을 관련 센서의 다수결 또는 SVR 예측으로 검증해 spike 계열 false alarm/day를 2.15에서 0.743까지 낮췄으며, 센서 관계의 계절 변화를 별도로 고려해야 한다고 지적했다.

[Santos-Fernandez 등 2024](https://doi.org/10.1029/2023WR035707)는 수질 센서 네트워크에서 공간·시간 자기상관을 사용하는 posterior predictive distribution, mixture, HMM 등을 비교했고, 가까운 여러 센서가 실제 환경 사건과 기술적 이상을 구분하는 데 도움이 됨을 보였다. [CAPA-CC](https://doi.org/10.1214/21-AOAS1508)는 서로 상관된 다변량 계열 중 일부에 생긴 평균 구조 이상을 scalable dynamic program으로 찾는다.

하지만 P1의 같은 기지 layer는 “가까운 동일 센서”가 아니다. 수온약층과 내부파로 정상적인 층간 상관이 끊길 수 있고, layer 번호는 연도 간 고정 수심 ID도 아니다. 따라서 논문의 다변량 이득을 고정 peer 평균으로 곧바로 옮기면 안 된다.

권장 gate:

- 최근 정상 추정 구간의 robust peer correlation
- 실제 depth median으로 만든 deployment/depth regime
- 수직 온도구배와 peer spread의 장·단기 비
- peer 수, 결측 mask, 동일 시각 정렬 여부
- 공통 변화점 consensus와 타깃만 변한 정도

[de Boyer Montégut 등 2004](https://doi.org/10.1029/2004JC002378)의 혼합층 기준인 10 m 기준 ΔT=0.2°C 또는 Δσθ=0.03 kg/m³는 물리적 출발점이 될 수 있다. 그러나 P1은 희소한 고정층이고 해역도 다르므로 0.2°C를 hard threshold로 복사하지 않는다. vertical spread, 인접 `|ΔT|/|Δz|`, rolling coherence로 fold-local mixed/stratified probability를 학습하는 초기 변수로만 쓴다.

gate가 안정 상태일 때만 peer residual/CAPA-CC score를 사용하고, 불안정 성층이나 no-peer에서는 단변량 score로 되돌린다. G-ORS는 outer no-peer와 거의 완전히 교락되어 있으므로 현재 OOF의 peer/no-peer 성능 차이를 peer의 인과 효과로 해석하지 않는다.

## 7. 해양 QC 문헌이 지지하는 모델 구조

[IQuOD의 60개 자동 QC 검사 벤치마크](https://doi.org/10.3389/fmars.2022.1075510)는 서로 다른 기기·오류 모드에서 검사 성능이 달랐고, high-true-positive, low-false-positive, compromise 목적에 따라 다른 검사 집합을 권장했다. [Castelão 2021](https://doi.org/10.1016/j.cageo.2021.104803)은 여러 QC test 출력을 multivariate criterion으로 결합하는 접근을 제시했다.

P1 적용 추론:

- 이미 강한 XGBoost를 버리고 하나의 거대 모델로 교체하는 것보다 spike/flatline/noise/offset/drift에 맞는 검사 bank를 늘리고 학습형 결합기를 유지하는 편이 문헌과 로컬 결과 모두에 맞는다.
- 정점·depth regime별 동적 scale을 사용하되, outer에서 실패한 특정 월·층을 직접 규칙으로 박지 않는다.
- plateau rule은 독립 hard override로 보존한다. spike singleton도 모든 smoothing/min-run 제거에서 보호한다.
- 최종 binary row F1은 유형명이 아니라 label union에 달렸으므로 유형 전문가는 score의 다양성을 만드는 수단이지 anomaly_type 문자열 정확도를 위한 과도한 모델이 아니다.

## 8. 후보 구간 분류기와 빠른 시계열 표현

후보 수가 충분히 작아진 뒤에는 interval 단위 분류기가 유용하다. 각 후보에서 duration, boundary gain, 전·중·후 robust statistics, XGBoost score 요약, 반사실적 residual, peer 상태를 계산해 작은 XGBoost/LightGBM 또는 logistic model로 accept/reject한다.

추가 표현이 필요할 때의 순서는 다음과 같다.

- [catch22](https://doi.org/10.1007/s10618-019-00647-x): 수천 개 특징에서 계산 효율과 해석성을 갖는 22개 특징 집합. 먼저 proposal-level ablation에 적합하다.
- [MiniRocket](https://doi.org/10.1145/3447548.3467231): 빠르고 거의 결정론적인 convolutional time-series classifier. 고정 길이 후보 문맥을 분류할 때만 시험한다.
- [Hydra](https://doi.org/10.1007/s10618-023-00939-3): convolution/dictionary 계열 특징으로 MiniRocket과 결합 가능한 빠른 표현. 기본 interval 특징이 포화된 뒤 후보로 둔다.

주의: 후보 window별 centering은 signed offset mean을 없앨 수 있다. robust scale은 쓰되 level residual 채널의 부호와 크기를 보존해야 한다. candidate 생성과 classifier 학습 모두 outer train 내부에서 이루어져야 하며, outer true event를 proposal training set에 넣으면 즉시 누출이다.

## 9. 우선순위에서 내린 방법

| 방법 | 판단 | 이유 |
|---|---|---|
| 표준 Matrix Profile/FLUSS | 주력 NO-GO, 낮은 우선순위 ablation | subsequence별 z-normalization이 additive offset을 지울 수 있고 flat/noisy subsequence에도 주의가 필요하다. drift shape의 unnormalized candidate-local 변형만 나중에 시험한다. |
| 전면 GNN/GDN | 보류 | P1은 기지별 layer 수가 적고 관계가 계절·성층에 따라 사라진다. 먼저 rolling graph/gate 특징으로 이득을 검증한다. |
| 새 대형 Transformer/TCN 대체 | 보류 | 이미 full sequence deep 모델이 XGBoost보다 낮고 FP가 많다. deep score는 보수적 segment 특징 또는 inner-only gate로만 재사용한다. |
| 전역 합성 4% 재시도 | NO-GO | 기존 증강은 recall을 올렸지만 FP 14,345행으로 붕괴했다. 유형 제한·저비율·hard-negative 보호 전에는 재실행하지 않는다. |
| 전역 threshold 인하 | NO-GO | XGBoost FN의 86.1%가 probability 0.05 미만이라 작은 threshold 이동으로 복구되지 않고 FP만 증가할 가능성이 높다. |
| raw temp에 변화점 직접 적용 | NO-GO | 계절·조석·성층·내부파의 정상 변화가 변화점이 된다. 반드시 station/depth-regime별 robust background residual에 적용한다. |

## 10. 동결할 실험 순서

| 순위 | 실험 | 직접 겨냥하는 실패 | 예상 효과 | 구현 비용 | 주요 위험 | 승격 전 중단 기준 |
|---:|---|---|---|---|---|---|
| R1 | frozen XGB seed + paired CAPA/epidemic PELT boundary completion | 긴 offset/drift의 잘린 경계 | 높음 | 낮음~중간 | 정상 계절 변화의 CP 오탐 | inner weighted F1이 +0.005 미만이거나 FP/day가 10% 이상 증가 |
| R2 | 후보-only 양방향 forecast/backcast residual | 장기 내부의 낮은 score와 유형 분리 | 높음 | 중간 | 후보 중앙을 baseline이 흡수, 계산량 | R1 후보에서 offset/drift recall 개선이 없거나 uncertainty가 FP를 분리하지 못함 |
| R3 | CPOP slope bank + mean/variance CP bank | drift onset/offset, noise/offset 분리 | 높음 | 중간 | penalty 과적합, first-difference 오용 | 48h+ recall 순증분이 없거나 worst group F1이 0.01 이상 하락 |
| R4 | inner-only segment classifier + semi-Markov decoder | candidate accept/reject와 전체 구간 투영 | 높음 | 중간 | event-level 누출, duration overfit | 모든 3개 inner block 개선 실패 또는 bootstrap 90% CI 하한 ≤0 |
| R5 | stratification-aware peer/CAPA-CC gate | S-ORS 정상 고변동 FP와 peer-dependent offset | 중간 | 중간 | 성층을 이상으로 오탐, G fallback 부재 | S year-transfer 비열화 또는 정상 FPR 10% 이상 증가 |
| R6 | catch22 → MiniRocket/Hydra proposal classifier | 남은 복합·형태 차이 | 중간/불확실 | 중간 | window normalization이 offset 삭제 | 단순 interval 통계 대비 +0.002 미만 |
| R7 | deep score의 보수적 residual gate | TCN의 slow recall만 선택적 회수 | 중간/고위험 | 중간 | outer oracle 모방 | inner-only gate가 3개 중 2개 block에서 비열화하면 중단 |

“예상 효과”는 측정값이나 리더보드 점수 예측이 아니다. 문헌 적합성과 로컬 오류량을 바탕으로 한 자원 배분 등급이다.

## 11. R1~R4의 최소 구현 계약

### 11.1 Candidate 생성

1. 현재 XGBoost 학습·OOF를 동결하고 row probability를 emission으로 사용한다.
2. station-layer, `segment_id` 경계를 절대 넘지 않는 robust residual을 만든다.
3. proposal source를 각각 기록한다: XGB run, plateau, spike, CAPA mean, CAPA variance, CPOP slope.
4. proposal끼리 겹치면 무조건 union하지 않고, 작은 boundary candidate 집합과 source bitmask를 만든다.
5. 공개 duration 범위 밖 후보도 삭제하지 않고 `duration_prior_violation` 특징으로 남긴다.

### 11.2 Interval 특징

- 길이, 시작/끝 시각, gap 거리
- XGB probability의 sum, mean, max, p10/p50/p90, low/high threshold coverage
- pre/interior/post median·MAD·slope와 return-to-baseline
- CAPA/PELT/CPOP cost gain과 경계 합의 수
- forward/backward normal forecast와의 residual level/slope/variance
- peer residual, peer spread, 최근 robust correlation, 수직 구배, gate 상태
- psal/depth 변화, 결측 mask, station/depth-regime categorical

### 11.3 Nested 검증

- candidate generator의 penalty, maximum duration, context length를 inner 과거 blocked split에서만 선택한다.
- interval classifier도 해당 inner train에서 만든 후보와 label로만 적합한다.
- inner validation에서는 row micro F1, test-share weighted F1, 48h+ row recall, event hit, start/end MAE, 정상 1일당 FP를 함께 본다.
- plateau hard override와 spike singleton 보존은 모든 설정에서 고정한다.
- outer validation은 사전 동결된 한 설정의 연구 진단에 한 번만 사용한다. 반복적으로 outer 결과를 보고 grid를 바꾸지 않는다.
- 최종 승격은 기존 프로젝트의 +0.005, bootstrap 90% CI 하한 >0, 3개 fold 중 2개 이상 비열화 없음, 정점별 하락 ≤0.01 기준을 그대로 적용한다.

## 12. 계산량과 구현 선택

- 전체 `n × 최대기간(519)` 완전 탐색은 약 4억 개 구간 평가가 될 수 있어 첫 구현으로 부적절하다.
- CAPA/PELT pruning과 prefix-sum 구간 통계, XGB seed 주변 candidate boundary로 검색 공간을 줄인다.
- CPOP는 전체 10분 grid가 아니라 candidate window와 축소 grid에서 시작한다.
- 양방향 예측은 candidate-only로 실행하고 결과를 Parquet cache에 저장한다.
- 데이터 로딩은 이미 전체 CV 시간의 병목이 아니다. pandas Arrow CSV 합계 0.167초, raw Parquet 0.130초이며 strict ingestion도 최근 CV 중앙시간의 1.286%다. 자세한 수치는 [데이터 로딩 벤치마크](P1_DATA_LOADING_BENCHMARK_2026-08-13.md)를 따른다.

## 13. 한계와 강건성

- 조사 논문의 데이터, 센서, 라벨 정책, metric은 P1과 다르므로 논문의 reported score를 P1 기대 점수로 옮기지 않는다.
- P1 라벨은 실제 장애 기록이 아니라 정상 관측에 합성 주입한 교육용 라벨이다. 실제 QC 논문은 설계 근거이지 같은 데이터 생성 과정의 증거가 아니다.
- offset/drift 집중 그룹과 6월 S-ORS FP는 outer 사후 진단이다. 이를 직접 조건으로 만든 규칙은 연구 누출이므로 금지한다.
- offline CAPA/PELT/CPOP와 backcast는 미래 관측을 사용한다. 메인 offline QC에는 포함할 수 있지만 causal 결과로 보고하지 않는다.
- peer 효과는 station과 교락되어 있고 G-ORS test depth는 전부 결측이다. no-peer fallback 없는 모델은 승격하지 않는다.
- 외부 관측값과 외부 pretrained weight는 이번 정찰과 향후 R1~R4에 필요하지 않다.

## 14. 결정과 다음 단계

다음 구현 묶음은 R1 하나로 제한하는 것이 안전하다.

1. CAPA/epidemic mean·variance proposal과 CPOP-lite slope proposal을 순수 특징 생성기로 구현한다.
2. frozen XGBoost seed 기준의 boundary coverage와 계산시간을 먼저 측정한다.
3. 별도 segment classifier 없이 가장 단순한 inner-selected interval score로 순증분을 확인한다.
4. R1이 통과할 때만 R2 반사실적 forecast/backcast와 R4 classifier를 추가한다.

이 순서는 실패 시 원인을 `proposal 부족`, `경계 부정확`, `accept/reject 부족`으로 분리할 수 있으며, 여러 고위험 기법을 한 번에 섞어 outer 라벨에 맞추는 것을 방지한다.

## 15. 1차 출처 목록

### 구간·변화점·지속시간

- Fisch, Eckley, Fearnhead, “A linear time method for the detection of collective and point anomalies,” Statistical Analysis and Data Mining, 2022. [DOI](https://doi.org/10.1002/sam.11586)
- Killick, Fearnhead, Eckley, “Optimal Detection of Changepoints With a Linear Computational Cost,” JASA, 2012. [DOI](https://doi.org/10.1080/01621459.2012.737745)
- Juodakis, Marsland, “Epidemic changepoint detection in the presence of nuisance changes,” Statistical Papers, 2023. [DOI](https://doi.org/10.1007/s00362-022-01307-x)
- Fearnhead, Grose, “cpop: Detecting Changes in Piecewise-Linear Signals,” Journal of Statistical Software, 2024. [DOI](https://doi.org/10.18637/jss.v109.i07)
- Sarawagi, Cohen, “Semi-Markov Conditional Random Fields for Information Extraction,” NeurIPS, 2004. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html)
- Kovács, Li, Bühlmann, Munk, “Seeded Binary Segmentation,” Biometrika, 2023. [DOI](https://doi.org/10.1093/biomet/asac052)
- Kim, Koh, Boyd, Gorinevsky, “ℓ1 Trend Filtering,” SIAM Review, 2009. [DOI](https://doi.org/10.1137/070690274)

### 환경·해양 센서 QC

- Min et al., “Evaluation of International Quality Control Procedures for Detecting Outliers in Water Temperature Time-series at Ieodo Ocean Research Station,” Ocean and Polar Research, 2021. [DOI](https://doi.org/10.4217/OPR.2021.43.4.229)
- Smith et al., “A Bayesian Framework for the Automated Online Assessment of Sensor Data Quality,” Sensors, 2012. [DOI](https://doi.org/10.3390/s120709476)
- Diamant et al., “Cross-Sensor Quality Assurance for Marine Observatories,” Remote Sensing, 2020. [DOI](https://doi.org/10.3390/rs12213470)
- de Boyer Montégut et al., “Mixed layer depth over the global ocean,” Journal of Geophysical Research: Oceans, 2004. [DOI](https://doi.org/10.1029/2004JC002378)
- Jones, Jones, Horsburgh, “Toward automating post processing of aquatic sensor data,” Environmental Modelling & Software, 2022. [DOI](https://doi.org/10.1016/j.envsoft.2022.105364)
- Good et al., “Benchmarking of automatic quality control checks for ocean temperature profiles and recommendations for optimal sets,” Frontiers in Marine Science, 2022. [DOI](https://doi.org/10.3389/fmars.2022.1075510)
- Castelão, “A machine learning approach to quality control oceanographic data,” Computers & Geosciences, 2021. [DOI](https://doi.org/10.1016/j.cageo.2021.104803)
- Santos-Fernandez et al., “Unsupervised Anomaly Detection in Spatio-Temporal Stream Network Sensor Data,” Water Resources Research, 2024. [DOI](https://doi.org/10.1029/2023WR035707)
- Tveten, Eckley, Fearnhead, “Scalable change-point and anomaly detection in cross-correlated data with an application to condition monitoring,” Annals of Applied Statistics, 2022. [DOI](https://doi.org/10.1214/21-AOAS1508)

### 후보 구간 표현

- Lubba et al., “catch22: CAnonical Time-series CHaracteristics,” Data Mining and Knowledge Discovery, 2019. [DOI](https://doi.org/10.1007/s10618-019-00647-x)
- Dempster, Schmidt, Webb, “MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification,” KDD, 2021. [DOI](https://doi.org/10.1145/3447548.3467231)
- Dempster et al., “Hydra: Competing convolutional kernels for fast and accurate time series classification,” Data Mining and Knowledge Discovery, 2023. [DOI](https://doi.org/10.1007/s10618-023-00939-3)

## 16. 추가 질문

- 대회 운영진은 전체 test 시계열을 쓰는 양방향 offline QC를 명시적으로 허용하는가? 현재 계획과 구현은 이를 주력으로 두되 causal ablation을 분리한다.
- `anomaly_type`이 심사 참고에 실질적 영향을 주는가, 아니면 binary row F1 최적화만 우선하면 되는가?
- 최종 모델 패키지에서 R 구현이나 별도 change-point 의존성이 허용되는가? 불확실하면 CAPA/CPOP의 필요한 score만 Python으로 재구현한다.
- 9월 7일 최종 모델 지정과 예측 업로드 잠금의 정확한 시각·순서는 무엇인가?
