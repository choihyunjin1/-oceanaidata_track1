# P1-P3 병렬 딥리서치·단일 실행 결론 보고서 v4

작성 시각: 2026-08-28 KST
성격: 내부 기술 의사결정 메모
범위: 과거 결론 고정, 문제별 병렬 문헌조사, 사전등록된 bounded 로컬 실행, 독립 QA

## 결론

이번 사이클에서 새 공식 제출 후보는 만들지 않았다. P1, P2, P3의 새 구조는 모두 사전등록 gate를 통과하지 못했으며, 기존 incumbent를 유지하는 것이 맞다. 공식 test/sample/submission 경로 접근, 후보 CSV 생성, 업로드는 모두 0회다.

이 결과는 세 문제의 최대 성능에 도달했다는 뜻이 아니다. 다만 이번에 검증한 세 계열은 다음 이유로 종료한다.

- P1: 동결된 83개 제안만으로 독립 qualification ranker를 학습·판정할 양성 support가 부족하다.
- P2: public-only 1-mode thermocline-heave 보정은 강한 기존 OOF보다 미세하게 악화됐고, 활성 범위가 0.0687%에 그쳤다.
- P3: ERA5 전문가 우위 신호는 있었지만 I4에서 -0.001150m, 개입률 1.389%, CI90 상한 0으로 고정 기준을 넘지 못해 전 fold가 incumbent로 정확히 fallback됐다.

따라서 다음 연구는 P1의 제안 support 확장, P2의 시간-깊이 공동 잠재구조, P3의 과거 봉인 전문가 OOF support 확장에 집중한다. 통과하지 못한 계열의 임계값 완화나 결과 기반 재실행은 하지 않는다.

## 실행 통제

1. v3 결론을 baseline으로 고정한 뒤 문제별 가설을 하나씩 사전등록했다.
2. 각 문제는 check-only와 테스트를 먼저 통과한 뒤 정확히 한 번의 sealed/bounded 실행만 허용했다.
3. 결과를 본 뒤 split, feature, threshold, blend, cap을 바꾸지 않았다.
4. P1의 Q2 truth 및 Q3/Q4, P1-P3 공식 test/sample/submission 경로는 열지 않았다.
5. 로컬 물리 단위와 공식 종합 점수는 서로 다른 척도다. 로컬 delta를 공식 점수 차이로 환산하지 않는다.

## 통합 결과

| 문제 | 새 구조 | 상태 | 핵심 관측 | 결정 |
|---|---|---|---|---|
| P1 | 동결 83제안 recall-guarded L2 logistic event ranker | `NO_GO_SUPPORT` | qualification 양성 제안 1, matched event 1, normality coverage 55.42% | 학습 0회, Q2 truth 비개방, 정확한 no-op |
| P2 | public-only 1-mode thermocline-heave tangent correction | `FAIL_GATE_STOP_NO_CSV_NO_RESEARCH_LOOP` | RMSE 0.7683674566→0.7683786975°C, Δ +0.0000112409°C, CI90 [-0.0000090823,+0.0000427147], 활성 0.0687% | 계열 종료, incumbent 유지 |
| P3 | forward-cross-fit ERA5 advantage ridge router | `NO_GO_INNER_I4_GATE` | I4 Δ -0.001150m, CI90 [-0.003119,0], 개입 1.389%; outer 개입 0% | 3/3 fold bit-exact incumbent fallback |

## P1: 동결 제안 위 이벤트 판별기

### 연구 판단

P1의 기존 병목은 새 이상 제안을 더 만드는 것보다 83개 제안 중 73개 false proposal을 줄이는 데 있었다. Neyman-Pearson 분류는 한 오류율을 제한하면서 다른 오류를 최소화하는 의사결정 틀을 제공하고, conformal risk control은 유한 표본 위험 통제를 설명한다. 다만 두 방법 모두 충분한 calibration/qualification 사건 수가 전제다. 이벤트 단위 평가는 시간점 단위 지표가 놓치는 이벤트 검출 품질을 보완한다.

- [Tong et al., Neyman-Pearson classification](https://pubmed.ncbi.nlm.nih.gov/29423442/)
- [Angelopoulos et al., Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)
- [Huet et al., Affiliation metrics](https://arxiv.org/abs/2206.13167)

### 사전등록 구조

동결 proposal bank, 40/30/30 시간 순서 split, 15일 purge, train-only robust scaling, `LogisticRegression(C=1, class_weight=balanced)`을 고정했다. calibration threshold는 원래 recovered event의 90% 이상을 유지해야 하며, qualification에서는 precision/F1 개선, false proposal 30% 감소, matched cell 비감소를 요구했다.

### 결과

- 전체 83 proposals, positive 10, positive cells 4, matched events 10
- train 33/5/5, calibration 17/2/2, qualification 14/1/1
- purge 19 proposals
- train-only primary normality reference coverage 55.42% < 80%
- 실패 gate: qualification positive proposals >=2, qualification matched events >=2, normality coverage >=80%
- model fit 0, threshold selection 0, Q2 truth rows read 0
- historical zero-add와 Q2 anchor는 byte-equivalent no-op

### 결론과 다음 단계

83-bank 후처리 계열은 종료한다. 다음 P1 실험은 ranker가 아니라 proposal support를 늘리는 generator 구조여야 한다. qualification에 최소 2개 이상의 독립 이벤트가 확보되지 않으면 모델 학습 전에 중단한다. synthetic augmentation을 쓰더라도 실제 이벤트 group split에서 검증하고, synthetic-only 성능을 승격 근거로 사용하지 않는다.

## P2: public-only thermocline-heave 보정

### 연구 판단

온도 경도에 대한 수직 변위의 1차 접선 보정은 물리적으로 해석 가능하지만, 충분한 공공 층 범위와 안정적인 배경 경도가 필요하다. Geoffroy와 Nycander는 수직 변위를 온도 경도와 연결하면서 수평 이류가 해석을 오염시킬 수 있음을 지적한다. Bendinger et al.의 1-2 mode 설명력은 깊은 glider 환경의 사례이며 P2의 얕고 결측된 층에 그대로 전이할 수 없다. PCHIP는 단조성을 보존하고 overshoot를 피하는 보간기로 선택했다.

- [Geoffroy & Nycander, thermocline displacement](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JC018283)
- [Bendinger et al., vertical-mode structure](https://os.copernicus.org/articles/20/945/2024/)
- [SciPy PCHIP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
- [Southern Yellow Sea inversion counterexample](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023JC020317)

### 사전등록 구조

가장 강한 공통 OOF인 `p2_extrapolated_soft_gate_v2`를 결과 전에 비교 기준으로 고정했다. public layer의 계절 배경을 PCHIP로 만들고 `-dT/dz` 1-mode와 절편으로 변위를 적합했다. 최대 correction 0.20°C, correction RMS 0.05°C, p99 0.20°C를 안전 한계로 두고, ΔRMSE <= -0.003°C, CI90 상한 <0, 3 fold 중 2개 개선, 활성 비율 >=5%를 승격 조건으로 고정했다.

### 결과

- 69,850 rows, 3 historical blocks
- incumbent RMSE 0.7683674566°C
- candidate RMSE 0.7683786975°C
- ΔRMSE +0.0000112409°C 악화
- fold delta +0.0000540216 / -0.0000011482 / 0°C
- CI90 [-0.0000090823,+0.0000427147]°C, 개선확률 0.332
- 활성 48/69,850 = 0.0687%
- correction RMS 0.00168767°C, p99 0, max 0.1699665°C
- 안전 gate는 통과했지만 성능, CI, fold, 최소 활성 gate는 실패

초기 독립 QA는 `candidate-reference-correction`에서 최대 1.6879e-15°C의 부동소수점 뺄셈 차이만 검출했다. 예측, 지표, bootstrap, gate, 해시를 그대로 두고 `rtol=0, atol=2e-15`로 재검산한 recovery QA가 통과했다.

### 결론과 다음 단계

public-only 1-mode heave 계열과 임계값 완화는 종료한다. 다음 P2는 개별 시점의 물리 역산보다 public layer의 시간 문맥과 깊이 축을 함께 학습하는 depth-registered latent profile 또는 sequence-to-profile residual 모델을 우선한다. 단, 먼저 validation block의 공공층 endpoint와 계절 배경 support가 충분한지 preflight하고, 기존 0.7683674566°C comparator를 반드시 이겨야 한다.

## P3: ERA5 전문가 안전 라우터

### 연구 판단

시계열 distribution shift에서는 입력 분포의 변화가 base model 선택을 어렵게 한다. stacking과 Super Learner는 out-of-fold 예측 기반 결합의 원칙을 제공하지만, 시간 순서 문제에서는 과거에서 생성·봉인된 OOF만 사용해야 한다. Learn Then Test와 non-exchangeable risk control은 후보 선택을 평가 데이터와 분리하고 시간 가중 위험을 다루는 근거를 제공한다.

- [Dish-TS, distribution shift](https://ojs.aaai.org/index.php/AAAI/article/view/25914)
- [van der Laan et al., Super Learner](https://doi.org/10.2202/1544-6115.1309)
- [Hasson et al., stacking for time series](https://proceedings.mlr.press/v202/hasson23a.html)
- [Angelopoulos et al., Learn Then Test](https://arxiv.org/abs/2110.01052)
- [Barber et al., Non-Exchangeable Conformal Risk Control](https://arxiv.org/abs/2310.01262)

### 사전등록 구조

기존 286개 past-only feature와 transfer/incumbent 예측을 동결했다. station/source/fold/calendar/absolute time/lead one-hot을 금지하고 현재 파고 동역학, 주기, 파 에너지, 풍 입력 proxy, 연속 lead, transfer-incumbent 차이만 사용했다. `StandardScaler + Ridge(alpha=100)`로 `(y-incumbent)^2-(y-transfer)^2` 우위를 예측하고, I3 residual q90을 넘는 경우만 0.20 blend하도록 고정했다. inactive row는 incumbent와 bit-exact 동일해야 한다.

### 결과

- 지원 가능한 outer fold 1, 지원 부족 2
- Ridge fit 1, CatBoost fit 0
- I4 ΔRMSE -0.001150m, CI90 [-0.003119,0], coverage 1.389%
- 고정 I4 기준: Δ <= -0.003m, CI 상한 <0, coverage 5-50%
- I4 gate 실패로 outer 3/3 모두 fallback
- sealed outer 1,086 rows, intervention coverage 0%
- pooled/fold/station/lead delta 모두 0, CI90 [0,0]

### 결론과 다음 단계

현재 라우터는 공식 제출 가치가 없다. 다만 ERA5 우위가 일부 구간에서 같은 방향으로 나타났다는 탐색 신호는 보존한다. 다음 P3는 router의 alpha나 threshold를 완화하지 않고, frozen ERA5 expert의 forward-sealed historical OOF를 더 넓은 과거 구간에 생성해 I1-I4 support를 먼저 확장한다. support가 확보된 뒤 동일한 저용량 router를 새로운 blind window에서 한 번만 재평가한다.

## 독립 QA와 재현성

- P1 independent QA PASS; result SHA256 `8afca6fc57c7bd98e99478de235533d3c69aa7a994844a71aa466f1c61ec9f4e`
- P2 recovery independent QA PASS; result SHA256 `f9626d17833a01f0ae2095eb0eaf2a9c055a16659ac31bc002c589413af52400`
- P3 independent QA PASS; result SHA256 `aa9b4931e479c10ee7540a2b688072da433c4353375fa43a673a63946045ec3f`
- P3 sealed outer predictions SHA256 `cf98f7b507008b2150b31cbb992c699ebf88fc04fd80334b8ad7b56c09614cf7`
- 관련 pytest 14개 PASS, Python source Ruff PASS
- 공식 입력 접근, 후보 CSV 생성, 업로드 0회

## 다음 병렬 연구 우선순위

1. P1 support-first generator: 실제 이벤트 group/time split에서 qualification positive/matched event support를 먼저 확보한다.
2. P2 joint time-depth latent profile: 공공층 시간 문맥과 깊이 기저를 동시에 쓰되 strongest common OOF를 이기는지 한 번의 sealed block 평가로 본다.
3. P3 expert-support expansion: 과거 window의 forward-sealed expert OOF를 늘린 뒤 동일 router를 새 blind window에서 평가한다.
4. 공통: 로컬 물리 단위와 공식 점수의 관계는 제출 원장에 쌍으로 누적하되, 소수 사례의 배율을 일반화하지 않는다.

## 최종 의사결정

이번 사이클의 세 후보는 모두 `NO_GO`이며 공식 제출 기회를 소비하지 않는다. 연구는 실패한 계열의 임계값 완화가 아니라, 각 문제에서 확인된 support 병목을 해결하는 새 bounded 사이클로 이어간다.
