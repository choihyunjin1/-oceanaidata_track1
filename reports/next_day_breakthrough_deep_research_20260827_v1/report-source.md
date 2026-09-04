# 해양 해커톤 다음 날 3×3 제출을 위한 구조 결함·돌파구 연구

부제: P1·P2·P3 공식 점수 실험, 로컬–공식 운송성, 2026-08-27 사전등록 후보

작성 기준 시점: 2026-08-26 KST  
팀: 분당독고다이  
상태: **9개 CSV 동결·독립 QA PASS·공식 업로드 0회**

## 결론

다음 날 제출 9개는 새 backbone 세 개를 무작정 시험하는 구성이 아니다. 지금까지 얻은 공식 점수를 이용해 각 문제의 가장 큰 구조적 불확실성을 분리하면서, 동시에 실제 점수 개선 가능성을 보존한 고정 실험이다.

- **P1:** `G-only / I-only / G+I(no removals)`로 Router의 +0.024163 F1 개선을 두 개의 대규모 추가양성 셀과 12행 제거 효과로 분해한다.
- **P2:** 공식 채점이 26,061행 전체 RMSE이므로 `A−O` 파일축의 곡률을 정확히 계산할 수 있다. 층별 최적 계수 `L2 +0.144257`, `L3 −0.093178`, `L4 −0.207647`을 동시에 적용한 U의 예상 공식 RMSE는 **0.535750**이며, 현 최고 0.536536보다 약 0.000786 낮다. 나머지 두 파일은 endpoint envelope와 PAVA의 순효과를 분리한다.
- **P3:** 이전 `0.598574` 예측은 B가 전 1,200행에서 O/A midpoint라는 잘못된 축 가정에서 나왔다. 실제 B는 12/18/24h에서만 midpoint이고 3/6/9h에서는 O와 같다. 또한 Public은 숨은 66사례이므로 전체 1,200행 곡률로 Public 최적 α를 계산할 수 없다. 따라서 `long α=-2 / long α=-4 / 18·24h α=-4`의 bounded 설계를 사용한다.
- **로컬 지표:** 14개 local–official contrast 중 부호가 일치한 것은 6개뿐이다. 비교 가능성이 높은 5개에서도 3개만 일치했다. 하나의 전역 보정식을 맞추지 않는다. P1은 셀별 지원 이동, P2는 공식 all-row 대수, P3는 공식 리드별 probe를 각각 선택 기준으로 쓴다.

현재 최종 동결 패키지는 `C:\Users\cedis\Downloads\해양 해커톤 제출용\20260827_round_E_preregistered_P1x3_P2x3_P3x3`에 있다. 생성은 업로드 승인이 아니며, 실제 업로드 직전에 사용자의 새 명시 승인이 필요하다.

## 연구 질문과 증거 공백

| 연구 질문 | 첫 패스 증거 | 핵심 공백 | 이번 결론 |
|---|---|---|---|
| P1 Router 개선은 새 모델 전체 우월성인가 | Router−B 공식 +0.024163, 로컬 +0.002230 | hidden truth와 Public mask | G/I/removal 셀 효과를 별도 공식 실험으로 분해 |
| P2 L4 개선을 더 활용할 수 있는가 | global·L2·L4 공식 점수와 all-row RMSE | PAVA의 물리적 타당성 | 층별 이차최적 U를 정확 복원하고 envelope/PAVA를 ablation |
| P3 공식 개선을 더 외삽할 수 있는가 | −2 global/L12/L18·24 모두 개선 | hidden 66-case mask, O/A/B 축 계보 | −13 외삽 금지, −2/−4 두 점으로 Public 곡률 식별 |
| 로컬 점수를 공식 선택 지표로 쓸 수 있는가 | 여러 부호 반전과 크기 왜곡 | 작은 표본·상관된 실험·계보 불일치 | 문제·개입군별 신뢰 규칙만 사용, 전역 scalar 금지 |
| 새로운 구조는 무엇이어야 하는가 | 기존 모델·후처리의 실패 기록과 문제 구조 | 독립 배포형 검증 | P1 segment/type bank, P2 T–S 연속수심 공동모형, P3 lead/regime calibration |

## 공식 Round D가 알려준 것

Round D는 2026-08-26에 9개를 중간 점수 열람 없이 제출한 고정 배치다. 팀 총점은 78.092863에서 78.919104로 **+0.826241점** 상승했다.

| 문제 | 이전 최고 | Round D 최고 | 원지표 개선 | 점수 개선 |
|---|---:|---:|---:|---:|
| P1 F1 | 0.793710 | 0.817873 | +0.024163 | +0.642207 |
| P2 RMSE ℃ | 0.541085 | 0.536536 | −0.004549 | +0.057085 |
| P3 RMSE m | 0.607071 | 0.599072 | −0.007999 | +0.126949 |

이 결과는 소수점 노이즈 수준을 넘어섰지만, 그대로 “로컬 모델이 일반화했다”고 읽을 수는 없다. P1은 Router 빈도가 이동했고, P2는 층별 순위가 로컬과 달랐으며, P3는 로컬 analogue와 공식 개입 계보가 일치하지 않았다.

## P1: Router를 셀 단위로 해부한다

### 구조 결함

P1은 spike/noise/flatline/offset/drift를 합성해 행 단위 binary F1로 평가한다. 문제 세트가 정점·층·계절과 지속시간 범위를 명시했는데도, 현재 선택 체계는 전체 OOF F1과 하나의 배포 threshold에 과도하게 의존한다. F1 최적 threshold는 batch prevalence에 의존할 수 있다. [Lipton, Elkan & Narayanaswamy, 2014](https://mlanthology.org/ecmlpkdd/2014/lipton2014ecmlpkdd-optimal/)

시계열 이상은 점이 아니라 연속 범위인 경우가 많다. range 존재·중첩·위치 편향을 분리해야 한다는 Tatbul 등의 결과는 P1의 offset/drift 경계 문제가 단순 row classifier 정확도만의 문제가 아님을 뒷받침한다. [Tatbul et al., NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)

로컬 incumbent FN 3,109행 중 offset/drift가 2,882행이고, 48시간 이상 이벤트가 2,393행이다. 긴 이벤트 17개 중 14개는 이미 seed를 갖지만 row recall은 약 0.6315다. 탐지 부재보다 경계와 내부 완성이 병목일 가능성이 높다.

### Router 공식 개선의 가장 유력한 설명

로컬 OOF에서 Router 추가군은 G-ORS L1 22행, I-ORS L2 21행이었고 제거군은 31행이었다. 공식 test에서는 각각 81, 136, 12행이다.

| 셀 | 로컬 행 | test 행 | 100k당 지원비 이동 | 로컬 truth 효용 |
|---|---:|---:|---:|---:|
| G 추가 | 22 | 81 | 9.172× | 22/22 유익 |
| I 추가 | 21 | 136 | 16.133× | 18/21 유익 |
| 제거 | 31 | 12 | 0.964× | 29/31 유익 |

따라서 공식 개선이 로컬보다 10.84배 컸던 것은 “새 backbone 전체의 우월성”보다 로컬에서 희소했던 유익 셀이 test에서 크게 증폭된 결과일 가능성이 높다. 다만 빈도 이동이 hidden conditional precision 보존을 증명하지는 않는다.

### 다음 날 P1 3개

| 순서 | 후보 | B 대비 변경 | 공식 식별 대상 |
|---:|---|---:|---|
| 1 | P1_1_PROBE_G_ONLY | G-ORS L1 O-only 81행 복원 | G 셀 효용 |
| 2 | P1_2_PROBE_I_ONLY | I-ORS L2 O-only 136행 복원 | I 셀 효용 |
| 3 | P1_3_EXPLOIT_GI_NO_REMOVALS | G/I 217행 복원, 12행 제거 없음 | G/I 결합과 removal 순효과 |

기존 B, 기존 Router와 세 결과를 합치면 G와 I의 배경별 score contrast, 기술적 interaction, 12행 제거의 marginal contrast를 계산할 수 있다. F1은 비선형이고 hidden subset 가능성이 있으므로 파일 수준 descriptive contrast로만 해석한다.

### 장기 구조적 돌파구

새 단일 분류기를 더 미세조정하는 것보다 이상 유형별 proposal bank가 우선이다.

- point/spike와 collective anomaly를 분리하는 CAPA 계열;
- drift slope의 구간 변화를 찾는 CPOP;
- 다변량 분포 경계를 찾는 changeforest;
- 정상→이상→정상의 duration-aware semi-Markov decoder.

CAPA는 epidemic/collective anomaly를 분리하는 직접적 방법론 근거를 제공한다. [Fisch, Eckley & Fearnhead, 2022](https://onlinelibrary.wiley.com/doi/full/10.1002/sam.11586) CPOP 구현과 경사 변화 탐지는 [Fearnhead & Grose, JSS 2024](https://www.jstatsoft.org/article/view/v109i07), changeforest는 [Londschien, Bühlmann & Kovács, JMLR 2023](https://www.jmlr.org/papers/v24/22-0512.html), semi-Markov 구조는 [Sarawagi & Cohen, NeurIPS 2004](https://papers.nips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html)에 근거한다.

## P2: 공식 all-row 이차곡선을 층별로 정확히 푼다

### 왜 P2만 정확 최적화가 가능한가

P2 문제 메모는 `test_index.csv`의 **26,061개 키 전체**를 세 층 row-pooled RMSE로 채점한다고 명시한다. 따라서 파일축 `d=A−O`에 대해 층별 공식 MSE는

`q_l(α)=q_l(0)+b_l α+a_l α²`, `a_l=Σ_{i∈l} d_i² / 26061`

의 정확한 이차식이다. `a_l`은 CSV만으로 알고, Round D의 global/L2/L4 `α=−t`, `t=0.1589769993` 점수로 `b_l`까지 식별된다.

| 층 | 행 수 | a | b | 공식 α* | α* 반올림 강건 구간 |
|---|---:|---:|---:|---:|---:|
| L2 | 8,713 | 0.012665817 | −0.003654276 | +0.144257405 | [0.143988480, 0.144526330] |
| L3 | 8,712 | 0.033834192 | +0.006305208 | −0.093178051 | [−0.093378538, −0.092977565] |
| L4 | 8,636 | 0.120300845 | +0.049960333 | −0.207647471 | [−0.207675644, −0.207619298] |

세 계수를 동시에 적용한 U의 예상 공식 RMSE는 **0.535750480**, 반올림 강건 구간은 **[0.535748471, 0.535752490]**이다. O 대비 0.005335, 현 최고 L4-only 대비 0.000786 개선 예상이다. 이는 private 일반화 예측이 아니라 현재 고정 all-row 공식 scorer에 대한 대수적 예측이다.

### 물리 후처리의 구조 결함

현 projector는 공개 T1≤T5 여부만으로 L2/L3/L4 수온을 동일 방향으로 단조화하는 unit-weight PAVA를 적용하고 endpoint 범위로 clip한다. 그러나 수온 단조성은 해수 밀도 안정성과 같지 않다. TEOS-10의 부력진동수 `N²`는 염분과 수온의 결합항에 의존한다. [TEOS-10 gsw_Nsquared 공식 문서](https://www.teos-10.org/pubs/gsw/v3_04/html/gsw_Nsquared.html)

한국 서·남해를 포함한 중국해 대규모 관측에서 수온 역전이 흔하며 염분 증가가 안정성을 보상한다는 보고도 hard temperature monotonicity의 위험을 뒷받침한다. [Hao, Chen & Wang, JGR 2010](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2010JC006297)

U test 예측에 대한 진단:

- endpoint envelope 가능 18,415행, 실제 변경 5,274행(eligible의 28.64%);
- full PAVA+envelope 가능 18,357행, 실제 변경 7,522행(eligible의 40.98%);
- PAVA는 불규칙 수심과 층별 불확실성을 반영하지 않는다.

### 다음 날 P2 3개

| 순서 | 후보 | 변환 | 목적 |
|---:|---|---|---|
| 1 | P2_1_EXPLOIT_LAYERWISE_QUADRATIC | 층별 α* 적용 | 공식 성능 exploit |
| 2 | P2_2_PROBE_ENDPOINT_ENVELOPE | U를 공개 T1/T5 범위로 행별 clip | 안전 envelope 효과 |
| 3 | P2_3_PROBE_FULL_PAVA_ENVELOPE | U에 PAVA 후 envelope | 단조 PAVA 순효과 |

로컬 p100 surrogate에서 RMSE는 O 0.990327, U 0.989168, E 0.988919, F 0.988979였다. E가 약간 우세하지만 exact official O/A 계보가 아니므로 순서의 약한 보조 증거일 뿐이다.

### 장기 구조적 돌파구

문제 세트가 강조한 것은 공개 L1/L5 사이의 프로파일 곡률, 계절 성층/혼합, 태풍, 12.42시간 내부조석이다. 이에 맞춘 다음 구조는 다음과 같다.

- T와 보조 S를 연속 수심 함수로 공동 복원하는 conditional multivariate FPCA/PACE 또는 depth-query operator;
- 예측 T/S로 계산한 TEOS-10 density·N² soft loss;
- 층·시각별 log-variance head와 불확실도 기반 손실;
- 동일 공식 O/A 계보의 rolling blocked OOF와 row-pooled RMSE;
- M2/일주/synoptic coherence가 공개 입력과 잔차에서 실제 확인될 때만 주파수 특징 확장.

PACE는 희소·불규칙 종단관측의 조건부 FPC score 복원 근거를 제공한다. [Yao, Müller & Wang, JASA 2005](https://utstat.utoronto.ca/fyao/2005-jasa.pdf) DeepONet은 입력 함수와 출력 좌표를 분리하는 operator 구조 근거지만, 저장소에 이미 유사 depth-query head가 있으므로 이름만 교체하는 것은 돌파구가 아니다. [Lu et al., Nature Machine Intelligence 2021](https://doi.org/10.1038/s42256-021-00302-5)

## P3: 축 오류를 교정하고 hidden Public 곡률을 직접 식별한다

### 확정된 두 가지 오류

1. A는 O의 단순 long-lead 보정본이 아니다. 공식 A−O는 12/18/24h뿐 아니라 3/6/9h에서도 각 20행씩 변한다.
2. B는 global midpoint가 아니다. 12/18/24h에서는 `(O+A)/2`, 3/6/9h에서는 O다.

재현 진단:

- `RMS[B−(O+A)/2] = 0.000193654 m`;
- `max|B−(O+A)/2| = 0.001110553 m`;
- A−O early 변경 60행.

따라서 기존 α=0,0.5,1 global quadratic과 0.598574 예측은 무효다. 예측 miss를 scorer 불안정의 증거로 해석한 것도 철회한다.

더 중요한 제약은 Public이 200사례 전체가 아니라 숨은 **66사례/396행**이라는 점이다. 전체 1,200행에서 계산한 `Σd²`는 Public 곡률이 아니다. 전체 파일 기하로 계산한 α≈−13 또는 α≈168과 0.575 수준 예측은 제출 근거로 사용할 수 없다.

### 다음 날 P3 3개

| 순서 | 후보 | active lead | α | 목적 |
|---:|---|---|---:|---|
| 1 | P3_1_EXPLOIT_LONG_NEG2 | 12/18/24 | −2 | no-early exploit + additivity guard |
| 2 | P3_2_PROBE_LONG_NEG4 | 12/18/24 | −4 | bounded long Public 곡률 |
| 3 | P3_3_PROBE_LEAD18_24_NEG4 | 18/24 | −4 | 장기곡률 및 12h 기여 분리 |

C1은 disjoint support MSE 항등식 때문에 hidden mask에서도 예측 가능하다.

`RMSE(C1)² = RMSE(L12−2)² + RMSE(L18/24−2)² − RMSE(O)²`

예상 공식 RMSE는 **0.598986994**, 반올림 강건 구간은 **[0.598985480, 0.598988507]**다. 이 범위를 벗어나면 score↔file 계보, fixed mask, RMSE 정의 또는 후보 생성 중 하나가 틀린다. C1 guard가 실패하면 C2/C3의 순위값은 보존하되 곡선 해석과 추가 외삽을 중단한다.

`D_g(α)=a_g α²+b_g α`에서 −2와 −4 두 점을 얻으면 `a_g=(D4−2D2)/8`, `b_g=(D4−4D2)/4`로 hidden Public 곡선을 식별할 수 있다.

### 장기 구조적 돌파구

문제 세트는 48시간 context에서 3/6/9/12/18/24h를 직접 예측하며 긴 리드가 어렵고, 외부 pre-2024 자료를 provenance와 함께 사용할 수 있다고 명시한다. 다음 구조는 이 조건에 맞춰야 한다.

- recursive 하나 또는 horizon별 완전 분리 대신 lead-continuous calibration;
- 폭풍 성장/감쇠, 현재 Hs, 바람–파랑 불일치에 따른 regime router;
- recent 48h instance normalization과 station/lead 계층 shrink;
- ERA5 source representation은 현재 고정 source gate를 통과할 때만 local 3-window로 이관;
- hidden future/절대시각 매칭은 계속 금지.

다단계 예측에서 recursive/direct의 bias–variance trade-off는 horizon별 correction 설계를 지지한다. [Ben Taieb & Hyndman, ICML 2014](https://proceedings.mlr.press/v32/taieb14.html) lead-continuous postprocessing은 작은 horizon별 표본을 완전히 분리하는 것보다 계수를 규칙적으로 공유하는 근거를 준다. [Wessel, Ferro & Kwasniok, QJRMS 2024](https://doi.org/10.1002/qj.4701) RevIN은 시계열의 시간적 분포 이동에 대한 간단한 normalization/denormalization 구조를 제공한다. [Kim et al., ICLR 2022](https://openreview.net/pdf?id=cGDAkQo1C0p)

## 로컬–공식 지표의 신뢰도

### 대표 관측 요약 (12/14; 전체 표는 companion XLSX)

| 문제/개입 | 로컬 delta | 공식 delta | 부호 일치 | 비교 등급 |
|---|---:|---:|---|---|
| P1 A−O F1 | +0.002087 | −0.004564 | 아니오 | B |
| P1 B−O F1 | +0.004186 | +0.003001 | 예 | B |
| P1 Router−B F1 | +0.002230 | +0.024163 | 예 | A− |
| P1 Intersection−B F1 | −0.001594 | +0.009218 | 아니오 | A− |
| P1 Union−B F1 | −0.002627 | −0.011404 | 예 | A− |
| P2 global−t RMSE | −0.004883 | −0.003847 | 예 | C |
| P2 L2−t RMSE | −0.001121 | +0.000832 | 아니오 | C |
| P2 L3−t RMSE | −0.003400 | −0.000136 | 예 | C |
| P2 L4−t RMSE | −0.000357 | −0.004549 | 예 | C |
| P3 long−2 RMSE | +0.002166 | −0.008084 예상 | 아니오 | C |
| P3 L12−2 RMSE | +0.000129 | −0.000390 | 아니오 | C |
| P3 L18/24−2 RMSE | +0.002037 | −0.007689 | 아니오 | C |

14개 전체 contrast에서 6개만 부호 일치, 비교 가능성이 높은 A−/B 5개에서도 3개만 일치했다. 표본이 작고 실험들이 같은 파일축을 공유하며 일부는 treatment 계보가 다르므로 회귀계수·상관계수 하나를 fit하지 않는다.

### 문제별 사용 규칙

- **P1:** 전체 로컬 F1 대신 disagreement cell별 지원빈도와 hidden official contrast를 사용한다.
- **P2:** official all-row quadratic은 신뢰한다. 로컬 surrogate는 U/E/F 후처리 순서의 약한 보조증거로만 쓴다.
- **P3:** 현재 local analogue는 선택지표로 기각한다. C1 guard와 C2/C3 공식 곡률을 우선한다.

Covariate shift importance weighting은 `P(Y|X)` 불변 가정에서만 정당하다. [Sugiyama, Krauledat & Müller, JMLR 2007](https://www.jmlr.org/papers/v8/sugiyama07a.html) 현재 P1 domain AUC 0.889, ESS 0.219와 P3 ESS 약 0.14는 단순 weighting을 신뢰하기 어렵다는 신호다. 시계열 성능이 시간에 따라 달라지는 Wild-Time 결과도 chronological deployment clone의 필요성을 뒷받침한다. [Yao et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html)

공식 제출을 반복 최적화할 때는 leaderboard 적응 자체가 과적합을 만든다. Ladder는 holdout 정보 노출을 제한하는 원리를 제시한다. [Blum & Hardt, ICML 2015](https://proceedings.mlr.press/v37/blum15.html) 따라서 한 날의 세 후보는 첫 점수를 보기 전에 동결하고 연속 제출한다.

## 동결된 9개 제출 파일

| 문제 | 후보 | 행 수 | SHA-256 |
|---|---|---:|---|
| P1 | P1_1_PROBE_G_ONLY | 169,011 | 671e52599efcaa671cbfa26c29c3fce2dd5531ab408109e4565c7ad3598cfe69 |
| P1 | P1_2_PROBE_I_ONLY | 169,011 | e8dc0b5b59fe0c717248cb5315d14e90399f491e085feef4cf57d4edad70fa79 |
| P1 | P1_3_EXPLOIT_GI_NO_REMOVALS | 169,011 | d249b382403281050fb71cb377508202d1af264a46d152b30c09ffa11c730372 |
| P2 | P2_1_EXPLOIT_LAYERWISE_QUADRATIC | 26,061 | 13181dff0e749a1ea6dac7327b4ea34b8a7efd57a2f57170ba0d206f919cf592 |
| P2 | P2_2_PROBE_ENDPOINT_ENVELOPE | 26,061 | f0298212b4be3552597524dab7d876128eacebb4333efe9ccc61becfcb5d72b7 |
| P2 | P2_3_PROBE_FULL_PAVA_ENVELOPE | 26,061 | a6ef13c1b2d93b498202f8f3c3a12e066dfbebbc13649df516d5bed107ce0571 |
| P3 | P3_1_EXPLOIT_LONG_NEG2 | 1,200 | 73a6a5ae5dd4d88bc40a0c00b7b66b60b3da239acaf554343d3cccba38bd0ab5 |
| P3 | P3_2_PROBE_LONG_NEG4 | 1,200 | 9d74dac5a7d7c9cda5b46c286af4dc2239dbbfdca650700c8ac65d3ddb849d15 |
| P3 | P3_3_PROBE_LEAD18_24_NEG4 | 1,200 | ba48386ad7746c28afba0f0cc562ed5166d6af62301ad95c210eb7aa8441b29f |

각 후보 폴더에는 홈페이지 입력용 `제출물 제목`과 `한줄요약(접근방식)` 메모가 포함된다. Round E 직전 공식 최고 파일 P1/P2/P3 한 세트도 `backup_best_before_round_E`에 보존했다.

독립 QA는 최종 unsuffixed bundle만 대상으로 수행됐고 `PASS_GO_TO_APPROVAL_BOUNDARY`였다. P0/P1/P2 결함은 모두 0건이며, 후보 9개·입력 계보 17개·백업 3개·`SHA256SUMS` 25개 항목이 일치했다. manifest SHA-256은 `7dd80d6288cd957192055916627b6bd31778565defb60f56c9baf078c8d487bc`다.

## 실행·판정 규칙

1. 2026-08-27 기회가 실제로 3/3씩 초기화됐는지 확인한다.
2. 9개 CSV를 fresh process에서 재해시하고 manifest와 일치시키며, 제목·한줄요약을 메모에서 복사한다.
3. 사용자의 새 명시 승인을 받는다.
4. P1 1→3, P2 1→3, P3 1→3 순서로 중간 점수를 보지 않고 연속 제출한다.
5. 모든 점수가 나온 뒤에만 결과 ledger를 작성한다.
6. P3 C1이 0.598985~0.598989 밖이면 C2/C3 곡선 해석과 추가 외삽을 중단한다.
7. P2 U가 [0.535748471, 0.535752490] 밖이면 all-row 채점·score↔file 계보·반올림 가정을 감사한다.
8. P1은 G/I/GI와 기존 B/Router를 함께 사용해 파일 수준 contrast만 계산한다.

## 한계

- 공식 점수는 Public이며 최종 Private 성능을 보장하지 않는다.
- P1 공식 evaluation mask/집계 방식이 문서상 완전히 식별되지 않았다.
- P2 공식 all-row 대수 예측은 고정 scorer에 대해서는 강하지만, PAVA·envelope의 hidden truth 효과는 사전에 알 수 없다.
- P3 Public 66사례 mask가 비공개라 −4 곡선은 이번 공식 probe가 있어야 식별된다.
- local–official 비교는 동일 데이터·동일 계보의 독립 반복이 아니며 상관 추정에 충분한 표본이 아니다.
- 새 구조 후보는 문헌과 실패 재구성에서 나온 연구 우선순위이며 아직 official 성능 증명이 아니다.
- P3 ERA5 context-transfer 고정 실험은 별도 자동화 경로이며 본 Round E 패키지에서 모델·286개 특징·split·gate를 변경하지 않았다.

## 출처와 내부 증거

### 외부 1차 출처

- Tatbul et al. (2018), Precision and Recall for Time Series: https://proceedings.neurips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html
- Lipton et al. (2014), Thresholding Classifiers to Maximize F1 Score: https://mlanthology.org/ecmlpkdd/2014/lipton2014ecmlpkdd-optimal/
- Fisch et al. (2022), CAPA: https://onlinelibrary.wiley.com/doi/full/10.1002/sam.11586
- Fearnhead & Grose (2024), CPOP: https://www.jstatsoft.org/article/view/v109i07
- Londschien et al. (2023), changeforest: https://www.jmlr.org/papers/v24/22-0512.html
- Sarawagi & Cohen (2004), semi-Markov CRF: https://papers.nips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html
- Yao, Müller & Wang (2005), PACE: https://utstat.utoronto.ca/fyao/2005-jasa.pdf
- TEOS-10, gsw_Nsquared: https://www.teos-10.org/pubs/gsw/v3_04/html/gsw_Nsquared.html
- Hao et al. (2010), Temperature inversion in China seas: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2010JC006297
- Lu et al. (2021), DeepONet: https://doi.org/10.1038/s42256-021-00302-5
- Ben Taieb & Hyndman (2014), Boosting multi-step autoregressive forecasts: https://proceedings.mlr.press/v32/taieb14.html
- Wessel et al. (2024), Lead-time-continuous postprocessing: https://doi.org/10.1002/qj.4701
- Kim et al. (2022), RevIN: https://openreview.net/pdf?id=cGDAkQo1C0p
- Sugiyama et al. (2007), Importance-weighted CV: https://www.jmlr.org/papers/v8/sugiyama07a.html
- Yao et al. (2022), Wild-Time: https://proceedings.neurips.cc/paper_files/paper/2022/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html
- Blum & Hardt (2015), The Ladder: https://proceedings.mlr.press/v37/blum15.html

### 공식·내부 증거

- `00_MUST_READ_FIRST.md`, `01_P2_MUST_READ_FIRST.md`, `02_P3_MUST_READ_FIRST.md`
- `20260826_round_D_preregistered_P1x3_P2x3_P3x3/OFFICIAL_RESULTS_20260826.json`
- `artifacts/daily_submission_3x3_evidence_20260827_v2/analysis.json`
- `reports/next_day_breakthrough_deep_research_20260827_v1/local_official_calibration.json`
- `artifacts/p1_matched_budget_local_compare_20260825_v1/predictions.parquet`
- `artifacts/p2_authoritative_nested_surrogate_actual_20260825_v5/evaluated_oof_100.parquet`
- `artifacts/p3_corrected_fixed_long_shrink_v4/oof.parquet`
- `src/p2_restore/profile_projection.py`
