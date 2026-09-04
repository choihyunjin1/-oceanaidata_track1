# 다음 제출기회 고가치 사용 딥리서치

**팀:** 분당독고다이  
**기준 시각:** 2026-08-27 KST  
**상태:** 연구·로컬 검증 완료 / 공식 업로드 미수행  
**결정 질문:** 남은 제출 한 장을 어디에 써야 최고점 가능성과 장기 정보가치를 동시에 높일 수 있는가?

## 결론

다음 공식 검증의 1순위는 **P2-SEASONAL-OAS-TS-10-PROJECTED**이다. 이는 현재 P2 최고본 U를 90% 유지하고, 계절 국소 OAS(Oracle Approximating Shrinkage) 공분산으로 동시 관측된 공개 T/S 층에서 은닉 2·3·4층의 T/S 프로파일을 조건부 복원한 값을 10% 결합한 뒤, 기존 endpoint/PAVA 물리 투영을 적용하는 구조적 A/B probe다.

이 후보는 로컬에서 총 69,850행 기준 RMSE를 0.768367에서 0.760586으로 낮췄고(Δ −0.007782℃), 물리 투영 후 0.760308(Δ −0.008060℃)이었다. KST 날짜 단위 5,000회 paired bootstrap의 90% 구간은 [−0.012785, −0.003176]℃, 개선 확률은 99.92%였다. 다만 세 블록 중 2025 Jul–Aug는 악화했다. 따라서 상태는 **DEPLOYMENT_GO가 아니라 OFFICIAL_PROBE_ELIGIBLE**이다.

P3는 공개축 이차 최적점에서 이미 +0.240938점을 얻어 관측 경쟁자 최고와 약 0.099점 차이까지 접근했다. 남은 P3 슬롯을 −8/−12 같은 같은 축 미세조정에 쓰는 정보가치는 낮다. ERA5 고정 실험 완료 또는 완전히 다른 오류공간의 후보가 나오기 전까지 보류한다.

## 1. 공식 상태와 기회비용

| 문제 | 현재 공식 최고 | 관측 경쟁자 최고 | 관측 headroom | 다음 한 장의 판단 |
|---|---:|---:|---:|---|
| P1 | 28.901363점 | — | — | 현재 연구 범위 밖 |
| P2 | 26.611283점, RMSE 0.535727℃ | 28.602603점 | 약 +1.991점 | 구조적 OAS probe가 최우선 |
| P3 | 24.066167점, RMSE 0.583892m | 24.165230점 | 약 +0.099점 | 같은 축 추가 probe 보류 |
| 합계 | 79.578813점, 5위 | — | — | P2가 가장 큰 잔여 레버 |

공식 리더보드는 모델의 일반화 성능을 확인하는 희소 자원이다. 반복 제출로 같은 public holdout에 적응하면 선택 편향이 누적될 수 있다. 따라서 한 장은 단순 점수 기대값뿐 아니라 **서로 다른 가설을 구분하는 정보가치**도 가져야 한다. 이 원칙은 leaderboard 과적합을 다룬 [Blum & Hardt (2015)](https://proceedings.mlr.press/v37/blum15.html)와 모델 선택 편향을 분석한 [Cawley & Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html)의 경고와 일치한다.

## 2. 문제 구조에서 다시 찾은 핵심

P2는 2025-09-01~2025-10-31의 61일 동안 2·3·4층의 수온·염분을 복원하는 문제다. 같은 시각의 공개 1·5·6·7층 T/S와 은닉 구간 양쪽 경계의 2·3·4층 관측은 사용할 수 있다. 평가 행은 26,061개이며 RMSE로 채점된다.

기존 최고 계보는 `router_400 + depth_query_bitcn + lsti_style + timemixerpp_style + moment_units_scratch`와 층별 공식축 후처리다. 이미 61일 전체 마스크·2160시간 창·336시간 문맥의 BiTCN, CSDI/SSSD형 후보, 연주기 anomaly transfer, 조석/RTS residual은 로컬에서 탈락했다. 따라서 단순히 더 긴 창이나 더 큰 신경망을 추가하는 것만으로는 새로운 가설이 아니다.

새 가설은 시간 예측보다 **같은 시각의 수직 T/S 공분산**을 직접 이용한다. 최근 conditional multivariate FPCA 연구는 부분 관측된 다변량 프로파일에서 교차 공분산을 이용해 잠재 score의 조건부 평균을 구하고 전체 프로파일을 복원하는 구조를 제시한다. 비가우시안에서도 이 조건부 식은 최선 선형 예측량으로 해석된다. 다만 해당 연구는 2026년 arXiv 사전논문이고 우리 단일 관측소 61일 gap과 도메인이 다르므로, 효과 크기를 전이하지 않고 아이디어만 사용했다. [Conditional multivariate FPCA](https://arxiv.org/html/2608.05376v1)

## 3. 이번에 직접 반증한 후보

### 3.1 경계 정합 연주기 prior — 기각

이전 해의 같은 계절 T/S 프로파일을 현재 공개층과 7일 양쪽 경계에 정합하고 LightGBM으로 보정했다.

| 노출 블록 | frozen reference | 후보 | 10% 혼합 | 판정 |
|---|---:|---:|---:|---|
| 2025 May–Jun | 1.286492 | 1.824135 | 1.298818 | 악화 |
| 2025 Jul–Aug | 1.103796 | 4.924675 | 1.274677 | 큰 악화 |

두 블록 모두 10% 혼합이 악화했고 oracle α도 음수였다. 직접적인 전년 프로파일 이동은 계절 상태 전이와 깊은 층 편향을 견디지 못했다. 이 축은 공식 제출 후보에서 제외한다.

### 3.2 계절 국소 OAS 조건부 프로파일 — 제한적 통과

각 14일 bin마다 원형 day-of-year ±60일의 완전 관측 timestamp만 사용해 T/S 1·5·6·7층과 연주기 4개 harmonic을 X, T/S 2·3·4층을 Y로 하는 결합 공분산을 OAS shrinkage로 추정했다. 행마다 실제로 관측된 X만 사용해 조건부 평균을 계산한다.

| 노출 블록 | reference | OAS 단독 | 10% 혼합 | ΔRMSE |
|---|---:|---:|---:|---:|
| 2024 Sep–Oct | 0.447793 | 0.746001 | 0.433021 | −0.014772 |
| 2025 Jul–Aug | 1.053477 | 1.341220 | 1.066635 | +0.013157 |
| 2025 Nov–Dec | 0.613081 | 0.252229 | 0.550264 | −0.062817 |
| 전체 | 0.768367 | — | 0.760586 | −0.007782 |

물리 프로파일 투영 후 전체 RMSE는 0.760308로 추가 개선됐고 11,062행(15.84%)에서 투영이 실제 작동했다. 공식 gap과 계절이 같은 2024 Sep–Oct에서는 9개 주 모두 개선했고, L2 0.165682→0.164080, L3 0.353559→0.341990, L4 0.671300→0.648132으로 세 층이 모두 개선됐다.

이 결과가 중요한 이유는 OAS 단독 모델의 평균 성능이 아니라, 기존 모델과 오류가 다른 작은 방향을 제공한다는 점이다. 10% 혼합은 기존 최고본을 대부분 보존하면서 수직 T/S 공분산 신호만 추가한다.

## 4. 다음 후보의 사전 등록 사양

**후보명:** `P2-SEASONAL-OAS-TS-10-PROJECTED`

1. 기본 예측은 현재 공식 최고본 P2 U로 고정한다.
2. 공식 gap 밖 `observations.csv`만 사용해 14일 season bin별 OAS 결합 공분산을 적합한다.
3. 학습 표본은 각 bin 중심의 원형 day-of-year ±60일이며, 공식 gap timestamp는 label 학습에서 제외한다.
4. 입력은 `temp/psal`의 1·5·6·7층과 1~4차 연주기 sin/cos, 출력은 `temp/psal`의 2·3·4층이다.
5. 결측 입력은 행별 관측 부분집합으로 조건부 평균을 계산한다.
6. 최종값은 `0.90 × P2_U + 0.10 × OAS`로 고정한다. α 재탐색은 하지 않는다.
7. 기존 endpoint/PAVA 프로파일 투영을 정확히 한 번 적용한다.

## 5. 제출 전 승격 게이트

### P0 — 기술 무결성

- 26,061개 공식 키·순서가 sample과 정확히 일치한다.
- 모든 예측이 유한하고 허용 물리 범위 안에 있다.
- 학습은 공식 gap의 은닉 2·3·4층 label을 읽지 않는다.
- 현재 U와 새 CSV의 입력·코드·환경·SHA-256을 receipt에 고정한다.
- 같은 명령을 두 번 실행해 byte-identical 또는 수치 identical임을 확인한다.

### P1 — 로컬 재현

- 현재 U 계보에 동일한 OAS와 투영을 적용했을 때 세 노출 블록 총합 ΔRMSE가 음수이고, 목표 재현값 −0.0075℃ 이하이다.
- 2024 Sep–Oct에서 세 층과 9개 주의 개선 부호를 재현한다.
- KST-day paired bootstrap 90% 상한이 0보다 작다.
- U 대비 예측차 RMS가 전체에서 0.02℃ 이상이어야 한다. 너무 작은 차이는 공식 한 장을 쓸 정보가치가 없다.
- 2025 Jul–Aug 악화를 숨기지 않고 receipt에 기록한다.

이 게이트를 모두 통과해도 상태는 `OFFICIAL_PROBE_ELIGIBLE`이다. 노출 블록에 대한 반복 선택과 한 블록의 역전 때문에 `DEPLOYMENT_GO`로 표현하지 않는다.

### P2 — 공식 한 장의 사전 결정표

- **26.611283점보다 명확히 높음:** 새 P2 public best로 채택한다. 같은 결과로 α를 즉시 미세조정하지 않는다.
- **반올림 오차 수준의 동률:** 방향성 증거로만 기록하고 그날 같은 축 추가 제출은 하지 않는다.
- **악화:** 계절 OAS 축을 닫고, 남은 슬롯은 다른 모델 가족에 보존한다.

## 6. 왜 지금 이 후보가 가장 값진가

| 후보 | 점수 기대 | 구조적 새로움 | 실패 시 정보 | 다음 슬롯 권고 |
|---|---|---|---|---|
| P2 OAS 10% + projection | 작지만 양의 로컬 근거 | 높음: 동시각 수직 T/S 공분산 | 높음: profile covariance 가설 검증 | 1순위 |
| P2 경계 정합 연주기 prior | 음수 | 중간 | 이미 반증됨 | 제출 금지 |
| P3 −8/−12 | 매우 낮음 | 낮음: 같은 1D축 | 낮음 | 보류 |
| generic 대형 imputer | 불명 | 중간 | 준비·QA 비용 큼 | 다음 연구 사이클 |
| P3 ERA5 context transfer | 불명 | 높음 | 높음 | 고정 실험 완료 후 판단 |

P2의 관측 headroom이 P3보다 약 20배 크고, OAS 후보는 현재 공식축 미세조정과 다른 오류공간이다. 로컬 개선폭 자체는 +3 공식점 목표를 보장하지 않지만, 공식-로컬 괴리가 컸던 과거 경험을 감안하면 완전 탈락시킬 정도로 작은 변화도 아니다. 한 장으로 구조적 가설을 검증하고 성공하면 새 backbone의 보조 expert로 확장할 수 있다.

## 7. 다음 구조 연구 순서

OAS가 성공하면 조건부 프로파일 모듈을 고정 expert로 삼아 LSTI형 장·단기 양방향 imputer의 gating 입력으로 넣는다. LSTI는 장기 양방향 autoregressive expert와 단기 imputer를 결합한다. 다만 발표된 평균 개선율은 다른 데이터셋의 수치이며 우리 문제에 전이할 수 없다. [LSTI, TMLR/OpenReview](https://openreview.net/forum?id=9NVJ0ZgEfT)

ImputeFormer의 저랭크 projected attention은 긴 block missing에서 계산 효율이 좋은 구조다. 하지만 교통 센서 benchmark와 우리 단일 정점 61일 수직 프로파일은 다르며, 과거 full-mask BiTCN 실패 때문에 ‘긴 receptive field’만으로 승격하지 않는다. [ImputeFormer paper](https://arxiv.org/html/2312.01728v3), [official implementation](https://github.com/tongnie/ImputeFormer)

CSDI는 조건부 diffusion imputation의 강한 기준이지만, 우리 로컬 CSDI/SSSD형 실험은 이미 선택 가중치 0으로 탈락했다. 같은 family의 단순 재실행은 다음 한 장의 후보가 아니다. [CSDI, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html)

TSI-Bench는 28개 알고리즘·8개 데이터셋·여러 결측 패턴에 대한 34,804개 실험을 통해 결측 패턴과 검증 프로토콜을 맞추는 중요성을 보여준다. 따라서 다음 딥 모델은 정확한 61일 contiguous mask와 공개층 조건을 유지해야 한다. [TSI-Bench](https://arxiv.org/abs/2406.12747), [official benchmark code](https://github.com/WenjieDu/Awesome_Imputation)

## 8. 한계와 중단 조건

- 세 외부 블록 모두 이전 연구에서 노출되었으므로 완전한 fresh confirmation이 아니다.
- bootstrap은 날짜 간 변동성을 반영하지만 반복 모델 선택의 적응 편향을 제거하지 못한다.
- frozen reference는 공식 lineage OOF이고, 현재 U의 미세 공식축 보정을 완전히 재현한 것은 아니다.
- 2025 Jul–Aug의 악화는 계절 국소 공분산이 regime shift에 취약함을 보여준다.
- 로컬 RMSE Δ를 공식 점수 Δ로 직접 환산하지 않는다.
- 정확한 대회 파일명·문제명·팀명으로 GitHub를 검색했지만 신뢰할 수 있는 참가자 공개 코드는 찾지 못했다. 이는 코드가 없다는 증명이 아니다.

연구의 한계효용은 현재 충분히 낮아졌다. 더 많은 문헌만 추가하는 것보다 위 사양을 배포 경로에서 재현하고 P0/P1을 통과시키는 것이 다음 의사결정에 더 큰 정보를 준다.

## 최종 권고

1. **P2-SEASONAL-OAS-TS-10-PROJECTED를 한 장짜리 공식 구조 probe로 준비한다.**
2. P0/P1 중 하나라도 실패하면 제출하지 않는다.
3. 통과하면 P2 한 장만 사용하고 사전 결정표대로 해석한다.
4. P3 슬롯은 ERA5 완료 또는 다른 구조 후보까지 보존한다.
5. 목표는 여전히 최소 +3 공식점이지만, 이번 한 장의 목적은 ‘작은 로컬 개선을 점수화’가 아니라 다음 큰 모델에 넣을 수직 T/S 조건부 expert의 공식 유효성을 판별하는 것이다.

## 재현 경로

- OAS 실험 코드: `scripts/research_p2_oas_conditional_profile_20260827.py`
- OAS 결과: `artifacts/p2_oas_conditional_profile_20260827_v3/result.json`
- OAS OOF: `artifacts/p2_oas_conditional_profile_20260827_v3/oof.parquet`
- 경계 정합 반증 코드: `scripts/research_p2_boundary_registered_prior_20260827.py`
- 경계 정합 결과: `artifacts/p2_boundary_registered_prior_20260827_v1/result.json`
- 공식 제출 receipt: `reports/finite_horizon_submission_decision_20260827_v1/report-source.md`

