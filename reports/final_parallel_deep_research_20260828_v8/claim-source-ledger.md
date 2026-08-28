# Claim–source ledger

| 핵심 주장 | 근거 | 신뢰도·한계 |
|---|---|---|
| P1 coverage 100%에서도 TP=0/F1=0 | `artifacts/p1_ts2vec_full_segment_coverage_recovery_20260828_v1/result.json` | 높음, frozen historical gate의 직접 결과 |
| P1 generic synthetic exposure와 typed-duration decoder는 안전하지 않음 | `artifacts/p1_ncad_synthetic_long_event_20260828_v1/result.json`; `artifacts/p1_typed_duration_semimarkov_v2/result.json` | 높음, 다만 exact-mask Transformer는 아직 미실행 |
| exact degradation-mask 학습은 현 repo 계열과 다른 supervision | [AnomalyBERT paper](https://arxiv.org/abs/2305.04468); [official code](https://github.com/Jhryu30/AnomalyBERT) | 구조 차이는 높음, P1 성능 전이는 중간 이하 |
| P2 OAS40은 공식 +0.483260점 | `reports/p2_submit_p1_p3_deep_research_20260828_v1/official_score_receipt.json` | Public에 대해 높음, Private 미확인 |
| thermocline depth·sharpness와 계절 stratification은 비선형 profile residual의 물리적 근거 | [GRL 2020](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020GL087848); [Frontiers 2023](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1120112/full); [AGU Advances 2026](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024AV001614) | 기전은 높음, 얕은 8-level 식별 가능성은 중간 이하 |
| importance weighting은 covariate shift 조건에서 유용하나 큰 분포 차에서는 고분산 | [JMLR IWCV](https://jmlr.org/papers/v8/sugiyama07a.html); [PMLR robust weighting](https://proceedings.mlr.press/v108/li20b.html); [PMLR domain adaptation calibration](https://proceedings.mlr.press/v108/park20b.html) | 이론 조건은 높음, P2의 P(Y|X) 불변성은 미확인 |
| P3 Hs² local correction은 3 folds·3 stations에서 개선 | `artifacts/p3_era5_longlead_energy_residual_shrink_20260828_v1/result.json` | 해당 Gen6 historical surface에서는 높음 |
| Hs²는 wave spectral moment/energy-space와 비례 | [ECMWF parameter note](https://confluence.ecmwf.int/download/attachments/59774192/wave_parameters.pdf?version=1); [NOAA/NDBC](https://www.ndbc.noaa.gov/faq/wavecalc.shtml) | 물리 관계는 높음, 고정 25% weight의 최적성은 뒷받침하지 않음 |
| 현재 P3 artifact와 Public champion은 다른 lineage | P3 experiment config/manifest; `C:/Users/cedis/Downloads/해양 해커톤 제출용/20260827_P3_REFINED_PUBLIC_OPTIMUM_READY/MANIFEST.json` | 높음, 즉시 NO_SUBMIT의 결정적 근거 |
| P3 중심 local 효과의 조건부 공식 환산은 +0.0578점 | P3 README의 below-T score rule; current champion RMSE/point; `evidence.json` | 산술은 높음, 실제 transport 예측은 낮음 |
| 반복 Public 제출은 holdout 적응 위험을 높임 | [Blum & Hardt, 2015](https://proceedings.mlr.press/v37/blum15.html) | 일반 원리는 높음, 본 대회의 정확한 과적합 크기는 미측정 |
