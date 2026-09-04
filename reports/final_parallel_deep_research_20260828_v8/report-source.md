# P1·P2·P3 최종 병렬 딥리서치 보고서

작성 기준: 2026-08-28 KST
팀: 분당독고다이
상태: 병렬 연구·기존 산출물·공식 점수 이력 교차검증 완료, 신규 CSV 생성·제출·업로드 없음

## 결론

1. **P1의 다음 돌파구 후보는 `exact degradation-mask Transformer + frozen-anchor union`이다.** 단순 coverage 복구, 일반 합성 이상 노출, type/duration decoder는 이미 실제 historical gate에서 반증됐다. 새 구조의 차이는 “이상 점수”를 학습하는 것이 아니라 정상 시계열에 가한 정확한 훼손 행 마스크를 직접 복원하는 데 있다.
2. **P2의 다음 후보는 `OAS40 + target-shift-weighted recent-context nonlinear thermocline-manifold residual`이다.** OAS40은 Public RMSE `0.445147°C`, `27.747847점`으로 공식 개선이 확인된 anchor다. 새 후보는 단순 수직 이동이나 저랭크 복원이 아니라 수온약층 깊이·sharpness·계절 변화라는 비선형 형상 잔차만 안전 support에 더하는 직교 probe다.
3. **P3의 현재 Hs² 후보는 지금 그대로는 제출하면 안 된다. 최종 상태는 `NO_SUBMIT_LINEAGE_MISMATCH`다.** 로컬 개선 `-0.003639m`은 old Gen6 OOF를 anchor로 계산했지만, 현 Public 최고는 `O + alpha(A-O)`, `alpha=-10.217432`의 다른 계보다. champion-matched local replay 없이 효과를 현 최고에 더할 수 없다.
4. P3가 현 champion 계보에서 같은 효과를 재현한다면 예상 이득은 중심 `+0.0578점`, 단순 CI 운반 시 `+0.0200~+0.0973점`이다. **큰 점수 돌파구는 아니지만, ERA5 외생 expert와 Hs² 에너지-space correction의 정보가치는 높다.** 계보 QA를 통과한 뒤 소멸성 일일 슬롯 1회로 검증할 후보이지, 현재 산출물을 바로 올릴 후보는 아니다.

## 현재 기준선과 근거

| 문제 | 현재 확인된 Public 최고 | 이번 연구의 핵심 판정 |
|---|---:|---|
| P1 | F1 `0.833248`, `28.901363점` | 현 계열의 coverage·합성·decoder 개선은 닫힘. exact-mask 직접 분할만 1회 실행 가치 |
| P2 | RMSE `0.445147°C`, `27.747847점` | OAS40 보존, 비선형 thermocline 잔차를 직교 probe로 검증 |
| P3 | RMSE `0.583892m`, `24.066168점` | 현 Hs² artifact는 계보 불일치로 제출 금지. champion-matched replay 후 조건부 1회 |

Public은 최종 Private가 아니다. P3 공식 README는 Public 66사례·396행, Private 134사례·804행이라고 명시한다. 반복적으로 Public을 최적화하면 holdout에 적응할 수 있으므로, 각 제출은 최고점 갱신뿐 아니라 다음 의사결정을 바꾸는 기전 검증이어야 한다. 이 위험은 반복 leaderboard 질의의 holdout 과적합을 분석한 [Blum과 Hardt의 Ladder 연구](https://proceedings.mlr.press/v37/blum15.html)와도 일치한다.

## P1 — exact degradation-mask 직접 분할

### 왜 기존 축을 닫는가

- TS2Vec-style coverage 복구는 `355,674/355,674`, coverage `100%`를 달성했지만 calibration과 qualification에서 `TP=0`, `F1=0`이었다. 짧은 segment 누락만의 문제가 아니라 representation signal이 없었다.
- NCAD 합성 long-event 노출은 inner에서 `ΔF1 +0.00703`이었지만 calibration `-0.37144`, qualification `-0.11760`; 추가한 행은 모두 false positive였다.
- typed-duration Semi-Markov는 pooled `ΔF1 +0.00243`에도 spike recall `-0.47059`, worst-cell F1 `-0.66667`로 구조적 퇴행을 만들었다.
- 따라서 coverage-first는 필수 엔지니어링 조건이고, type/duration은 decoder 우선순위가 아니다.

### 새 가설

[AnomalyBERT](https://arxiv.org/abs/2305.04468)는 입력 일부를 여러 방식으로 훼손하고 Transformer가 그 정확한 부분을 찾도록 자기지도 학습한다. 현 repo의 normal-prototype/NCAD scalar score와 다른 supervision이다. P1에는 같은 station×layer·calendar-month의 정상 donor를 고정해 spike/noise/flatline/offset/drift 훼손을 만들고, 모델이 exact row mask를 직접 복원하게 한다. 공식 평가는 raw row F1이므로 point adjustment와 정답 기반 event fill은 금지한다. PA가 순위를 심하게 왜곡할 수 있다는 별도 1차 연구는 [AAAI 2022 rigorous evaluation](https://ojs.aaai.org/index.php/AAAI/article/view/20680)에 있다.

### 한 번의 bounded 실행 계약

- 모델: relative-position Transformer 4 layers, `d=128`, window `1024`, direct binary mask head.
- 데이터: pre-Q2 historical fit/cal/qualification, 경계마다 15일 purge. Q2는 모든 inner gate 통과 뒤 outer 1회만 연다.
- 학습: seed 1개, 최대 30 epoch/10k steps, synthetic-validation raw-mask F1 best checkpoint. 결과 기반 재튜닝 없음.
- 배포: overlap probability 단순 평균, 사전고정 threshold grid 1회, smoothing/PA/truth-fill/gap-close 없음. incumbent positive는 절대 삭제하지 않는 frozen-anchor union.
- synthetic fidelity: noise/offset/drift 각각 raw F1 `≥0.80`, boundary MAE `≤6 rows`, eligible coverage `1.0`.
- real gate: cal·qualification 모두 anchor-union `ΔF1>0`, 추가 precision `>anchor_F1/2`, 놓친 long event `≥2` 및 `≥2 cells` 회수, worst-cell `ΔF1≥-0.01`, bootstrap `P(ΔF1>0)≥0.80`.
- 예상 자원: RTX 5090 4–8GB, 약 30–90분. 가장 큰 실패 위험은 synthetic→real transport다.

2순위는 [CATCH](https://openreview.net/pdf?id=m08aK3xxdJ)의 time-frequency patch/channel score, 3순위는 [TimeInf](https://arxiv.org/abs/2407.15247)의 temporal-block influence다. 둘 다 1순위 실패 후에만 연다.

## P2 — target-shift-weighted nonlinear thermocline residual

### 왜 이 축인가

OAS20→OAS40의 exposed local OOF는 악화를 예측했지만 공식 RMSE는 `0.038514°C` 개선되고 점수는 `+0.483260` 상승했다. 반면 동일 계보의 공식 prediction geometry 중심 예측은 실제 RMSE와 `0.003480°C` 차이였다. 그러므로 P2는 로컬 절대 순위로 즉시 버리지 말고, **frozen official anchor와 직교한 오차 방향인지**까지 측정해야 한다.

수온약층은 단순 수직 평행이동보다 깊이와 sharpness가 함께 변하는 비선형 구조다. equatorial Pacific의 subsurface temperature를 thermocline depth와 sharpness의 bounded hyperbolic-tangent 형태로 나타낸 [Yuan·Jin·Zhang](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020GL087848), 상층 pycnocline의 깊이·강도·두께가 계절·지역별로 달라짐을 보인 [Sérazin 등](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1120112/full), 2백만 개가 넘는 Argo profile에서 계절~장기 stratification 변동을 확인한 [Somavilla 등](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024AV001614)이 이 표현의 물리적 타당성을 뒷받침한다. 다만 이 근거를 얕은 8-level 정점 자료의 성능 보장으로 옮기지는 않는다.

### 단일 bounded pilot

- anchor: 공식 OAS40 prediction을 그대로 보존한다.
- 신규 arm: 완전 historical T/S profile에서 shared thermocline depth·sharpness와 bounded T/S shape coefficient를 만들고, 최근 48–72시간의 실제 깊이 T/S context로 coefficient를 예측한다.
- target shift: pseudo-target의 unlabeled X만 사용해 importance weight를 계산한다. [IWCV](https://jmlr.org/papers/v8/sugiyama07a.html)는 `P(X)`만 변하고 `P(Y|X)`는 같다는 조건에서 타당하다. 분포 차가 크면 weight variance가 폭주할 수 있다는 [robust importance weighting 연구](https://proceedings.mlr.press/v108/li20b.html)를 반영해 clip `[0.25,4]`, ESS `≥30%`, discriminator AUC `≤0.80`을 강제한다.
- validation: 3개 historical block mask, L2–L4 T/S 동시 mask, 7일 purge, target 값 input/update 금지.
- gate: target-weighted pooled `ΔRMSE≤-0.002°C`, 3 folds 중 2개 이상 개선, day-block bootstrap CI90 upper `≤+0.001°C`; unweighted regression `≤+0.005°C`, worst layer `≤+0.010°C`.
- 직교성: alpha20→40 공식 prediction direction과 correction cosine 절댓값 `≤0.85`; active share 10–70%, correction RMS `0.01–0.05°C`, p99 `≤0.20°C`.
- 통과 상태는 `ORTHOGONAL_PROBE_READY`이지 새 champion 확정이 아니다. 공식 한 점으로 새 direction의 residual alignment를 추정하고, 개선이면 다음 날 bounded scale 1회만 검증한다.

이미 닫힌 중복은 BayOTIDE(active `0/69,850`, posterior SD median `2.8296°C`), CMFPCA(`ΔRMSE +0.022734°C`), GP·TEOS analog·공개 tangent·기존 heavy imputer 계열이다.

## P3 — Hs² 후보의 실제 제출 가치

### 로컬 증거

고정 규칙은 18/24h에서만

`E_candidate = E_Gen6 + 0.25 × (E_ERA5 − E_Gen6)`, `E=Hs²`

를 적용하고 나머지 724/1,086행을 bit-exact Gen6로 유지했다. 전체 RMSE는 `0.7799487225→0.7763096877m`, 즉 `-0.0036390348m` 개선했다. 3 folds·3 stations가 모두 개선했고 case bootstrap CI90은 `[-0.0061285,-0.0012620]m`, local `P(improve)=0.995`였다.

Hs가 `4√m0`이므로 Hs² correction은 스펙트럼 0차 모멘트, 즉 에너지와 비례하는 공간에서 작동한다. 이는 [ECMWF wave parameter 정의](https://confluence.ecmwf.int/download/attachments/59774192/wave_parameters.pdf?version=1)와 [NOAA/NDBC 계산법](https://www.ndbc.noaa.gov/faq/wavecalc.shtml)에 부합한다. [WAVEWATCH III](https://polar.ncep.noaa.gov/waves/wavewatch/index.shtml)와 [ECMWF wave model 문서](https://www.ecmwf.int/en/elibrary/81144-ifs-documentation-cy46r1-part-vii-ecmwf-wave-model)가 파랑을 spectral action/energy balance와 wind input·dissipation·nonlinear transfer로 기술하므로, ERA5 외생 forcing이 장기 lead에서 별도 오류 방향을 제공한다는 가설은 물리적으로 합리적이다.

그러나 Hs² 하나는 주파수·방향·군속도·수심·해류·미래 wind source를 압축하며, weight `0.25`는 물리상수가 아니다. 더 결정적인 문제는 이 로컬 anchor가 old Gen6라는 사실이다.

### 현재 즉시 제출 금지 사유

현 Public champion은 12/18/24h에 `O + (-10.217432)(A-O)`를 적용한 별도 계보다. 현재 Hs² artifact는 official candidate CSV도 아니며, 현 champion-equivalent OOF에서 재평가되지 않았다. 따라서 현재 상태는 **`NO_SUBMIT_LINEAGE_MISMATCH`**다.

### 조건부 점수 시나리오

현재 RMSE `0.583892<T=0.630065` 구간의 공식 환산식은 `points=33.333960−15.872414×RMSE`다.

| champion-matched RMSE 운반 가정 | 예상 점수 변화 | 해석 |
|---:|---:|---|
| CI90 하단 효과 `0.001262m` | `+0.0200` | 작은 개선 |
| 중심 효과 `0.003639m` | `+0.0578` | 가장 단순한 1:1 시나리오 |
| CI90 상단 효과 `0.006129m` | `+0.0973` | 낙관 구간 |
| 중심 효과 2배 | `+0.1155` | 강한 transport |
| `+3점` 목표 | RMSE `-0.1890m` 필요 | local 효과의 약 51.9배, 현실적 기대 아님 |

이는 예측이 아니라 **champion-matched 재현을 조건으로 한 시나리오**다. 과거 P3 A/B correction은 local 개선이 Public 악화로, reverse 18/24는 local 악화가 Public 개선으로 뒤집혔다. 따라서 local bootstrap `0.995`를 공식 성공확률로 읽으면 안 된다.

### 공식 1회 probe 자격의 최소 5 gate

1. 현 champion의 정확한 formula·파일·SHA-256을 고정한다.
2. 동일 181 historical case에서 champion-equivalent OOF를 만들고 그 anchor 대비 Hs² 25% 효과를 다시 계산한다. Gen6 대비 숫자는 재사용하지 않는다.
3. 3/6/9/12h는 champion과 bit-exact, 18/24h만 변경됐는지 확인한다.
4. champion 대비 overall `Δ<0`, 3 folds 중 `≥2` 개선, station/lead 최대 악화 `≤0.0075m`를 다시 통과한다.
5. 공식 무정답 QA에서 1,200행·키·순서·0–30m·400 active/800 inactive·hash·재현 명령을 검증하고 weight/lead/station 규칙을 동결한다.

모두 통과하고 소멸성 일일 슬롯을 쓰지 않으면 잃으며, 기대 `+0.15점` 이상의 다른 독립 P3 후보가 없을 때만 1회 제출한다. 사전등록 해석은 `≥+0.03점`이면 Public champion 채택, `-0.03~+0.03점`이면 무정보로 보고 미세조정 금지, `≤-0.03점`이면 deployment family 종료, `≥+0.10점`이면 구조적 신호로 보존하되 Private-ready라고 부르지 않는 것이다.

## 실행 우선순위

1. **P3 lineage audit/replay** — 가장 짧고, 현재 유망 artifact를 제출 가능한지 즉시 판별한다. 현재는 CSV 생성 금지 상태다.
2. **P1 exact degradation-mask 1회** — 가장 큰 구조적 상한. synthetic fidelity가 먼저 실패하면 30–90분 안에 중단한다.
3. **P2 nonlinear thermocline residual 1회** — OAS40을 보존하는 직교 probe. target-weight/ESS gate가 실패하면 no-op으로 끝낸다.
4. P1 실패 시 CATCH, P2 실패 시 frozen-arm robust convex stack을 각각 한 번만 연다.

## 한계

- 외부 논문의 개선률은 데이터·채점법이 달라 이 대회의 점수로 환산하지 않았다.
- P1 AnomalyBERT 가설은 NCAD의 synthetic→real 실패와 긴장 관계에 있다. exact-mask supervision이 실제 차이를 만드는지가 핵심 실험 질문이다.
- P2 importance weighting은 `P(Y|X)`가 유지된다는 covariate-shift 가정이 깨지면 오히려 악화할 수 있다.
- P3 Public은 66사례뿐이고 반복 사용됐다. 작은 Public 개선은 Private 일반화 증명이 아니다.
- 이번 단계에서는 공식 test/sample/submission 값, hidden answer, 신규 submission CSV를 읽거나 만들지 않았다.

## 최종 판정

**P1과 P2에는 각각 한 번의 명확한 새 구조 실험이 남아 있다. P3 Hs²는 장기적으로 유의미한 physics-space/ERA5 신호지만, 현 산출물은 다른 champion lineage라 지금 제출할 수 없다. 먼저 champion-matched replay를 통과시키고, 그때만 작은 점수 기대와 높은 정보가치를 가진 공식 1회 probe로 승격한다.**
