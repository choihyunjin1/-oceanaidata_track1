# P1·P2·P3 독립 정보축 Deep Research 및 실행

작성일: 2026-08-28 KST
대상: 분당독고다이 해커톤 모델 의사결정
상태: 1회 연구·실행·독립 QA 완료
공식 업로드: 0건

## 직접 답변

이번 사이클에서 공식 제출할 새 후보는 없다. 세 문제 모두 기존 미세조정과 다른 구조를 실제 구현하고 사전 고정 gate로 1회 평가했지만 승격하지 못했다.

- **P1:** 다층 temporally fused RPCA는 layer별 coverage와 동일 시각 joint coverage가 다르다는 사실 때문에 분해 window가 0개였다. incumbent와 bit-equivalent no-op으로 종료했다.
- **P2:** supervised rank-1 functional residual은 pooled RMSE를 `-0.004799°C` 개선하고 KST-day bootstrap CI90 `[-0.008506, -0.003107]`을 얻었다. 그러나 사전 목표 `-0.005°C`에 `0.000201°C` 못 미쳤고, `2025 Nov-Dec`가 `+0.008592°C` 회귀해 worst-fold `+0.003°C` 제한도 위반했다. exact 공식 α50 OOF가 아닌 `INCUMBENT_PROXY_VALIDATION` 비교이므로 공식 승격 근거로 쓰지 않는다.
- **P3:** wind-wave memory는 2024 H1 shadow의 18/24h RMSE를 `-0.011197m` 개선했지만 독립 case가 17개뿐이고 S-ORS가 `+0.005840m` 회귀했다. shadow gate에서 중단하여 outer 181-case 정답은 열지 않았다.

독립 QA는 PASS였고 공식 test/sample/submission 접근, CSV 생성, 업로드는 모두 0건이다.

## 연구 질문과 가정

1. 기존 champion과 다른 관측 메커니즘 또는 prediction direction이 실제 일반화 성능을 추가하는가?
2. exposed historical fold에서 생기는 낙관을 줄이기 위해 prediction commitment와 clustered bootstrap을 지킬 수 있는가?
3. 결과를 본 뒤 threshold·blend·fold를 바꾸지 않고 단일 구조의 반증 가능성을 유지할 수 있는가?

기존 historical folds는 여러 연구에 노출되었으므로 fresh holdout으로 주장하지 않는다. 공식 점수는 aggregate 결과일 뿐 hidden row label을 제공하지 않는다는 한계를 전제로 했다.

## 후보 선택 근거

### P1 — temporally fused robust PCA

RPCA는 관측 행렬을 low-rank background와 sparse corruption으로 분리한다. Candès et al.은 적절한 식별 조건에서 이 분리가 가능함을 보였고, Stable PCP는 작은 dense noise 하의 안정성을 다룬다. Sofuoglu와 Aviyente는 sparse 항의 temporal variation을 벌점화해 지속 이상에 맞춘다. NOAA QARTOD 역시 인접 수심·다변량 검사를 정당한 QC 정보축으로 다루지만 지역·센서별 threshold 검증을 요구한다.

이 근거를 바탕으로 I/S station의 `time × layer` 행렬에 low-rank + temporally fused sparse decomposition을 적용하고, G-ORS와 incumbent 양성은 exact no-op으로 보호했다. threshold는 prefix-normal 99.9% 분위수, 지속시간은 48–519행으로 고정했다.

### P2 — supervised rank-1 functional residual

무감독 CMFPCA가 독립 direction임에도 실패했으므로, predictor-response 연관을 직접 쓰는 supervised functional subspace가 필요했다. Zhang, Sun, Kong은 functional response에 대한 supervised principal component regression을 제안하며 무감독 PCA와 달리 response association을 이용한다. Izenman의 reduced-rank regression은 다변량 response coefficient를 낮은 rank로 제한하는 고전적 근거다.

공개 layer L1/L5–L8의 T/S를 고정 cubic B-spline df=5로 표현하고 6/24/72h past changes를 추가했다. target L2–L4의 `truth - alpha50_proxy` 공동 잔차에 `PLSRegression(n_components=1)`만 적합했다. train 97.5% leverage 밖 또는 public support 부족은 exact no-op, correction RMS `0.05°C`·absolute `0.20°C` cap, 기존 endpoint/PAVA를 마지막에 한 번만 적용했다.

### P3 — past-only wind-wave memory

ECMWF wave model과 NOAA WAVEWATCH III는 wave evolution을 풍응력·source term과 연계하는 물리적 근거를 제공한다. Li et al.은 inverse wave age가 peak phase speed와 wind-wave directional alignment를 함께 사용함을 설명한다. 다만 대회 `tp`는 peak period가 아니라 top-third mean이라는 데이터 정의 차이가 있어 hard wave-age threshold를 쓰지 않고 `phase_speed_proxy`로만 사용했다.

과거 48h의 simultaneous finite wind/wave pair에서 aligned wind power, 6/24h EWMA, memory contrast, `corr(q[t-lag], ΔHs²[t])`를 10개 값 + 10개 mask로 만들었다. 동일 train row·seed·CatBoost 설정에서 기존 591 features와 enriched 611 features를 paired fit했고, 18/24h에만 enriched-base increment의 20%를 적용했다.

## 실행 결과

| 문제 | 사전등록 후보 | 핵심 결과 | Gate 실패 이유 | 최종 판정 |
|---|---|---|---|---|
| P1 | `p1_temporally_fused_rpca_offset_drift_anchor_union_20260828_v1` | added 0, ΔF1 0, scored window 0 | calibration/validation 동일시각 multi-layer segment 부재 | `NO_GO_EXACT_NO_OUTPUT` |
| P2 | `p2_alpha50_supervised_rank1_functional_residual_20260828_v1` | ΔRMSE `-0.004799°C`, bootstrap P(improve)=`1.0` | pooled 목표 0.000201 미달, Nov-Dec `+0.008592°C` | `NO_GO_EXACT_NO_OUTPUT` |
| P3 | `p3_past_only_wind_wave_memory_regime_increment_20260828_v1` | shadow 18/24h ΔRMSE `-0.011197m` | 17 cases(<30), station min 5(<8), S-ORS 회귀 | `NO_GO_SHADOW_GATE` |

### P1 세부

- incumbent F1: `0.866900`; candidate F1: `0.866900`.
- Q2/Q3/Q4 모두 exact no-op이다.
- layer별 기간 coverage는 높았지만 서로 다른 관측 구간에 존재했다. rowwise coverage를 simultaneous coverage로 해석한 사전 조사 오류가 실험에서 반증되었다.
- 결론: 동일시각 다층 행렬을 전제로 한 RPCA family는 현재 데이터 representation에서 닫는다. 다음 P1은 asynchronous layer를 허용하는 station-layer event model 또는 single-layer change-point 증거가 필요하다.

### P2 세부

- aggregate proxy: `3.085787 → 3.080988°C`, Δ `-0.004799°C`.
- fold: Sep-Oct `-0.006859`, Jul-Aug `-0.006201`, Nov-Dec `+0.008592°C`.
- layer: L2 `-0.014291`, L3 `-0.009786`, L4 `-0.002953°C`; 모든 layer 평균은 개선했다.
- KST-day bootstrap 5,000회: CI90 `[-0.008506, -0.003107]`, P(improve)=`1.0`.
- correction active share `37.20%`, RMS `0.046425°C`, p99 `0.133561°C`.
- OAS cosine `-0.0873`, orthogonal share `99.62%`; 이전 historical correction cosine `0.0234`, orthogonal share `99.97%`.
- 해석: 구조는 독립이고 평균 개선 신호도 있지만 계절 운반성이 부족하다. exact α50 historical OOF가 없어 절대 RMSE와 공식 점수 대응을 주장할 수 없다. 다음 P2 연구는 correction 크기 재튜닝이 아니라 train-only regime/support veto가 Nov-Dec 회귀를 사전에 식별할 수 있는지 별도 preregistration해야 한다.

### P3 세부

- shadow support: 17 cases, G/I/S=`5/7/5`.
- 18h `-0.013010m`, 24h `-0.009511m`, 합계 `-0.011197m`.
- station: G `-0.020202m`, I `-0.020141m`, S `+0.005840m`.
- 10개 memory 값의 결측률은 약 `38.3–40.1%`였다.
- 해석: G/I의 방향성 있는 장기 예보 신호는 보였지만 support와 station transport가 부족하다. 결과를 보고 S만 제외하거나 threshold를 바꾸지 않았다. exact feature family는 닫고, 다음 P3은 더 넓은 독립 shadow와 pressure/spectral source-term을 포함한 별도 구조여야 한다.

## QA와 재현성

- 신규 unit/projection test: 11개 PASS.
- 신규 코드 전체 Ruff PASS.
- 독립 QA: `PASS`, 15개 계약 검사 모두 true.
- P1/P2 prediction은 truth metric 전에 hash commitment를 생성했다.
- P3는 shadow prediction seal 이후에만 shadow truth를 결합했고, shadow 실패 후 outer truth는 0행 읽었다.
- 세 artifact directory에 CSV가 없고 공식 업로드는 0건이다.
- 기존 `test_p3_corrected_repeated_forward_catboost_v2.py` 전체 실행 중 1개 테스트는 이미 존재하는 append-only artifact 때문에 실패했으며 새 코드 오류가 아니다. 신규·projection test만 격리 재실행해 11/11 PASS를 확인했다.

## 결론과 다음 단계

이번 사이클의 최고 연구 자산은 P2이다. 독립 direction·bootstrap·2개 fold·3개 layer 개선이 동시에 확인됐지만, 한 계절의 큰 회귀와 exact official comparator 부재 때문에 지금 제출하면 안 된다. P1과 P3은 각각 observation synchrony와 validation support에서 구조적 한계가 확인됐다.

다음 사이클 우선순위:

1. **P2:** 결과 기반 shrink 재튜닝을 하지 말고, correction을 적용하기 전 train-only support/regime veto 하나를 새 실험으로 사전등록한다. 성공 기준은 Nov-Dec 회귀 제거와 pooled `≤-0.005°C`를 모두 유지하는 것이다.
2. **P3:** 현재 20-feature family는 닫는다. 더 넓은 독립 shadow를 먼저 확보할 수 있을 때만 pressure tendency·directional spectral memory를 포함한 새 family를 연다.
3. **P1:** simultaneous matrix 가정을 버리고 asynchronous sensor별 event proposal을 공통 station state로 후결합하는 구조만 검토한다.

## 한계와 불일치

- 세 historical surface 모두 연구에 반복 노출되었다. bootstrap은 표본 변동을 추정하지만 adaptive selection bias를 제거하지 못한다.
- P2 local comparator는 공식 α50 exact OOF가 아니다. 따라서 `-0.004799°C`를 공식 leaderboard 점수로 변환하지 않는다.
- P3의 physical proxy에서 대회 `tp`와 연구 문헌의 spectral peak period는 동일하지 않다.
- P1 RPCA 이론은 low-rank·sparse 식별 조건과 동시 관측을 가정하며, 이 데이터의 asynchronous layer 수집은 전제에서 벗어났다.

## 검색 중단 기준

각 문제에서 권위 있는 1차 문헌, 저장소의 기존 실패축, 구현 연결점, 사전등록 1회 결과가 모두 확보되었다. 즉시 추가 구조를 결과에 맞춰 재실행하면 exposed folds에 대한 적응 선택을 키운다. 따라서 이번 Deep Research는 세 terminal 판정과 독립 QA PASS에서 중단한다.

## 주요 출처

- Candès, Li, Ma, Wright, 2009/2011, [Robust Principal Component Analysis?](https://arxiv.org/abs/0912.3599)
- Zhou, Li, Wright, Candès, Ma, 2010, [Stable Principal Component Pursuit](https://arxiv.org/abs/1001.2363)
- Sofuoglu, Aviyente, 2020, [Low-rank on Graphs plus Temporally Smooth Sparse Decomposition](https://arxiv.org/abs/2010.12633)
- NOAA/IOOS, 2020, [QARTOD Temperature and Salinity Manual v2.1](https://cdn.ioos.noaa.gov/media/2020/03/QARTOD_TS_Manual_Update2_200324_final.pdf)
- Zhang, Sun, Kong, 2024, [Supervised Principal Component Regression for Functional Responses](https://www.tandfonline.com/doi/abs/10.1080/10618600.2023.2250411)
- Izenman, 1975, [Reduced-rank regression for the multivariate linear model](https://www.sciencedirect.com/science/article/pii/0047259X75900421)
- ECMWF, 2023, [IFS CY48R1 Part VII: ECMWF Wave Model](https://www.ecmwf.int/en/elibrary/81373-ifs-documentation-cy48r1-part-vii-ecmwf-wave-model)
- NOAA/NCEP, [WAVEWATCH III documentation](https://polar.ncep.noaa.gov/waves/wavewatch/wavewatch.shtml)
- Li et al., 2024, [A Novel Sea State Classification Scheme](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JC020686)
- Pathak, Ma, Wainwright, 2022, [A new similarity measure for covariate shift](https://proceedings.mlr.press/v162/pathak22a.html)
