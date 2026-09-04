# 다음 행동 메타 딥리서치 — 2026-08-31

## 결론

다음 컴퓨팅 슬롯은 **P1 dependence-calibration feasibility audit**에만 쓴다. 이것은 새 제출 후보를 만드는 성능 실험이 아니라, 최근 CAPA 실패의 직접 원인인 과다 제안을 입력-only 자료에서 사전 차단할 수 있는지 확인하는 `INFORMATION` lane의 0-supervised-fit 실험이다.

P2와 P3는 지금 새 모델을 돌리지 않는다. P2는 Public과 독립적인 동일 계절 61일 확인면이 없고, P3는 operational wave archive 자체는 존재하지만 익명 case를 실제 issue time과 연결하는 서명된 manifest가 없다. 두 문제 모두 알고리즘보다 새 정보 계약이 먼저다.

| 우선순위 | 문제 | 판정 | 다음 행동 | 성능·제출 주장 |
|---:|---|---|---|---|
| 1 | P1 | `EXECUTE_INFORMATION_PREFLIGHT_ONLY` | prefix residual의 의존성 보정 null calibration이 사전 false-alarm/proposal budget을 만족하는지 0-fit 감사 | 금지 |
| 2 | P3 | `STOP_NO_CASE_TIME_LINKAGE` | organizer 또는 원본 계약에서 case→UTC issue time manifest 확보 | 금지 |
| 3 | P2 | `HOLD_NO_FRESH_SAME_SEASON_SURFACE` | untouched same-season 61-day block 또는 blind evaluator 확보 | 금지 |

이 순위는 예상 점수 상승량이 아니라 **새 정보를 얻을 수 있는가, 기존 실패와 의미상 중복되지 않는가, 가장 싼 반증이 가능한가**로 정했다.

## 범위와 고정 입력

- 기준 커밋: `d4edf3d1d877e3b0f1d17b4841a310b62e4bfdd1`
- 내부 전수 원장: 48 historical family, 35 canonical group, 20 later key case, 4 workflow exception
- 최신 P1 결론: fixed Gaussian-style CAPA가 validation 421,032행 중 203,574행을 추가하고 pooled ΔF1 `-0.727384987`을 기록한 `NO_GO_RESEARCH_ONLY`
- 금지: official test/sample/submission/hidden 값 읽기, 제출 CSV 생성, upload, exposed result에 맞춘 threshold·penalty 재튜닝
- 이번 연구 중 새 model fit, prediction, CSV, upload: 모두 0

## P1 — dependence-aware calibration만 열어 둔다

### 왜 이 축만 남는가

최근 CAPA는 총 831개 collective segment를 선택했고 전체 validation의 약 48.35%를 추가행으로 만들었다. 세 fold 모두 크게 악화됐으므로 특정 분기나 unsupported layer가 원인이 아니다. 실패 메커니즘은 fixed independent-Gaussian null보다 실제 residual의 null tail이 훨씬 두꺼웠다는 것이다.

이 진단은 외부 연구와 정합적이다. Dette, Schüler, Vetter는 dependent error에서 원래 SMUCE가 change point 수를 과대추정하고 불일치할 수 있으며, long-run variance를 반영한 수정이 필요하다고 보였다. Cho와 Fryzlewicz는 고차원 시계열에서 noisy coordinate의 CUSUM을 그대로 합치면 문제가 생기므로 threshold를 넘은 성분만 sparse aggregation하는 방법을 제안했다. 반면 Gaussian setting의 FDRSeg 보장은 이 데이터의 heavy-tailed/autocorrelated residual에 그대로 이식할 수 없다.

따라서 다음 P1 실험은 기존 penalty의 수치 조정이 아니라 다음 질문 하나만 답해야 한다.

> label을 전혀 보지 않은 prefix residual block에서 temporal dependence와 cross-layer dependence를 보존한 empirical null이, 사전 고정한 false-alarm ceiling과 proposal-share ceiling을 실제로 맞출 수 있는가?

### 최소 실행 계약

1. 새 experiment id와 exactly-once lock을 사용한다.
2. incumbent, clean-state projection, window family는 그대로 두고 **calibration layer만 교체**한다.
3. label/anomaly type은 prediction seal 전 0행 접근을 유지한다.
4. prefix를 contiguous station-layer block으로 나누고 long-run variance 또는 moving-block null을 계산한다. block length는 label score가 아니라 prefix dependence diagnostic으로 고정한다.
5. 같은 station의 cross-layer vector를 함께 resample하거나 coherence rule을 사전 고정해 cross-sectional dependence를 보존한다.
6. 여러 alpha 중 좋은 것을 고르지 않는다. 사전 고정한 nominal level에서 held-out prefix pseudo-null block의 exceedance calibration만 본다.
7. 다음 중 하나면 label을 열지 않고 종료한다.
   - realized false-alarm rate가 nominal ceiling을 초과
   - proposal row share가 사전 ceiling을 초과
   - 한 station-layer cell이 proposal의 70%를 초과
   - block-length sensitivity에서 판정이 뒤집힘
8. 통과해도 `CALIBRATION_FEASIBLE_RESEARCH_ONLY`일 뿐 제출 후보가 아니다. 별도 frozen historical falsification이 필요하며 fresh label 부재 한계는 유지한다.

이 preflight는 label-free 통계가 addition precision을 증명한다고 주장하지 않는다. 제안 폭발을 사전에 걸러내는 능력만 검증한다.

## P2 — 방법이 아니라 독립 확인면이 부족하다

DINEOF 원 논문은 EOF 수와 reconstruction error를 cross-validation으로 정한다. 후속 해양 적용 연구는 실제 결측 구조를 닮은 cloud-shaped masks로 validation 값을 제거해야 error가 결측 영역을 대표한다고 설명한다. 이는 현재의 61일 연속 결측에는 random point mask보다 **mask-matched contiguous block validation**이 맞다는 근거다.

그러나 우리 저장소에서는 동일 historical block과 Public feedback이 이미 반복 선택에 사용됐다. mask-matched DINEOF가 문헌상 타당하다는 사실은 새 독립 성능 증거를 만들어 주지 않는다. bin17의 작은 공식 개선은 보존하지만, adjacent bin·rank·season을 같은 Public 결과로 다시 고르면 적응적 재선택이다.

따라서 P2의 다음 행동은 둘 중 하나뿐이다.

1. 모델·rank·season·mask를 고정하기 전에 untouched same-season 61-day label block을 확보한다.
2. 주최 측 blind evaluator를 확보한다.

둘 다 없으면 `NO_NEW_EXPERIMENT`다. 기존 bin17 champion을 유지하고 추가 fit을 하지 않는다.

## P3 — 자료는 있으나 정렬 키가 없다

NOAA EMC 문서는 GFS-Wave와 GEFS-Wave를 `YYYYMMDD/CC/wave` 구조와 00/06/12/18Z cycle로 제공한다고 명시한다. NOMADS는 실시간·단기 archive와 NCEI 장기 archive를 제공한다. 즉, 실제 발행 cycle에 기반한 wind/wave forecast feature를 만드는 외부 자료원 자체는 존재한다.

하지만 이 사실만으로 대회 case에 그 자료를 정렬할 수는 없다. 현재 내부 감사상 익명 case를 `UTC issue_time, station coordinate, model cycle, publication cutoff`에 연결하는 signed manifest가 없다. hindcast/reanalysis를 사후 valid time에 맞추는 것은 deployment-time forecast와 다른 estimand이며, leakage-free operational correction의 증거가 아니다.

또한 wave forecast 연구는 forcing wind bias를 줄여도 significant wave height가 보편적으로 좋아지지 않으며, wind error와 internal wave-model error가 서로 상쇄될 수도 있음을 보였다. 따라서 P3의 다음 행동은 alpha sweep이나 새 신경망이 아니라 **case-time linkage 계약 확보**다.

재개 최소조건은 다음과 같다.

- case별 UTC issue time과 station coordinate
- cycle publication cutoff와 사용할 수 있었던 forecast file hash
- lead 3/6/9/12/18/24h의 exact valid-time mapping
- 최소 3개 episode-disjoint block의 research confirmation, 성능 승격에는 더 넓은 독립 episode 필요

이 중 하나라도 없으면 fit 0으로 종료한다.

## 공통 연구 프로토콜

앞으로 딥리서치에는 다음 순서로 묻는다.

1. 먼저 `NO_NEW_EXPERIMENT`가 최선인지 판정한다.
2. 후보 이름보다 새로운 정보원을 명시한다.
3. candidate ledger와 model card를 이용해 exact/semantic overlap을 감사한다.
4. 30분 contract smoke와 가장 싼 반증을 먼저 설계한다.
5. 제안에 쓰이지 않은 confirmation surface가 없으면 `PERFORMANCE` lane을 금지한다.
6. success/failure/inconclusive stop rule을 결과 전에 고정한다.
7. 공식 점수 예상값을 만들지 않는다.

## 한계

- P1 calibration preflight가 통과해도 addition precision이나 공식 F1 개선은 증명되지 않는다.
- P2의 fresh surface 부재와 P3의 case-time linkage 부재는 문헌 조사로 해소할 수 없는 데이터 계약 문제다.
- Public의 작은 양수는 방향 정보이지 Private 수송 보장이 아니다.
- P1의 큰 leaderboard gap은 연구 우선순위를 높이지만 증거 기준을 낮추는 근거는 아니다.

## 출처

- Dette, Schüler, Vetter, *Multiscale change point detection for dependent data* (2018/2020): https://arxiv.org/abs/1811.05956
- Cho & Fryzlewicz, *Multiple-change-point detection for high dimensional time series via sparsified binary segmentation* (JRSS B, 2015): https://doi.org/10.1111/rssb.12079
- Li, Munk, Sieling, *FDR-Control in Multiscale Change-point Segmentation* (Electronic Journal of Statistics, 2016): https://doi.org/10.1214/16-EJS1131
- Beckers & Rixen, *EOF calculations and data filling from incomplete oceanographic datasets* (JTECH, 2003): https://hdl.handle.net/2268/4291
- Sirjacobs et al., *Reconstruction of MODIS total suspended matter time series maps by DINEOF and validation with autonomous platform data* (Ocean Dynamics, 2011): https://doi.org/10.1007/s10236-011-0425-4
- NOAA EMC, *Wave Models*: https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/wavemodels.php
- NOAA NCEP, *NOMADS Information*: https://nomads.ncep.noaa.gov/info.php?page=help
- Durrant et al., *The effect of statistical wind corrections on global wave forecasts* (Ocean Modelling, 2013): https://doi.org/10.1016/j.ocemod.2012.10.006
