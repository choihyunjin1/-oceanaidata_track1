# Claim–source ledger — deadline cycle v17

확인일: 2026-08-28

| ID | 주장 | 출처 | 범위·한계 |
|---|---|---|---|
| S1 | 시계열 conformal은 serial dependence를 block 구조로 다룰 수 있음 | Chernozhukov, Wüthrich, Zhu, COLT 2018, https://proceedings.mlr.press/v75/chernozhukov18a.html | 약한 의존 등 조건 아래 근사 validity |
| S2 | group-conditional calibration은 score-homogeneous group 구성이 중요 | Martinez Gil et al., UAI 2024, https://proceedings.mlr.press/v244/martinez-gil24a.html | P1 station pooling의 동질성을 보장하지 않음 |
| S3 | cross-fitting은 nuisance와 held outcome의 same-sample bias를 줄이는 구조 | Chernozhukov et al., 2016, https://arxiv.org/abs/1608.00060 | P2가 DML estimator라는 뜻은 아님 |
| S4 | predictor와 rejector를 분리해 shift 위험을 제어할 수 있음 | Li et al., AISTATS 2024, https://proceedings.mlr.press/v238/li24g.html | P2의 exact official transport를 보장하지 않음 |
| S5 | CatBoost는 multidimensional MultiRMSE objective를 제공 | CatBoost official docs, https://catboost.ai/docs/en/concepts/loss-functions-multiregression | 구현 가능성만 지지, 성능 향상 보장 없음 |
| S6 | Hs·period·Hmax는 동일 spectral sea-state의 관련 output | ECMWF IFS Wave Model, 2020, https://www.ecmwf.int/sites/default/files/elibrary/2020/81192-ifs-documentation-cy47r1-part-vii-ecmwf-wave-model_1.pdf | 관련성이 Hs 예측 개선을 보장하지 않음 |
| S7 | multitask learning은 관련 task의 shared inductive bias를 활용 | Caruana, Machine Learning 1997, https://doi.org/10.1023/A:1007379606734 | task conflict 시 negative transfer 가능 |

## 내부 provenance

| 문제 | 정본 artifact | 핵심 확인 |
|---|---|---|
| P1 | `artifacts/p1_station_pooled_hierarchical_residual_subset_scan_anchor_union_20260828_v1/result.json` | no active fold, truth 0, anchor deletion 0 |
| P2 | `artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2/result.json` | one-shot, Nov-Dec no-op, benefit gate fail |
| P3 | `artifacts/p3_era5_joint_wave_state_multitask_transfer_20260828_v1/result.json` | fit 2, source gate fail, shadow truth 0 |
| 공식 | `https://oceanaidata.org/app/leaderboard` 및 문제 페이지 | 23:09 KST read-only snapshot; 업로드 0 |
