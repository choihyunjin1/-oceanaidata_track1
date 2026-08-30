# Claim-source ledger

| Claim | Primary or authoritative source | Evidence used | Confidence and limitation |
|---|---|---|---|
| P1 공식 목적은 row binary F1이고 anomaly type은 순위에 반영되지 않는다 | 배포 P1 README, 2026-08-13 | 다섯 유형, 지속시간, baseline 0.548255, 제출 schema | High; hidden type mix와 Public/Private 절차는 미상 |
| calibrated score에서 F1-optimal threshold는 batch와 best F1에 의존하고 특수한 경우 F*/2가 된다 | Lipton, Elkan, Narayanaswamy, *Thresholding Classifiers to Maximize F1 Score*, 2014, https://arxiv.org/abs/1402.1892 | add-only precision sanity의 수학적 방향 | High; calibration과 stationarity를 가정하며 shift 해결책은 아님 |
| point와 collective anomaly를 penalized likelihood로 함께 찾을 수 있다 | Fisch, Eckley, Fearnhead, *A linear time method for the detection of collective and point anomalies*, 2022, https://doi.org/10.1002/sam.11586 | P1 CAPA decoder의 구조적 근거 | High; 기본 이론은 univariate/독립 noise 중심 |
| cross-correlated multivariate series의 sparse/dense anomaly를 함께 모델링할 수 있다 | Tveten et al., *Scalable change-point and anomaly detection in cross-correlated data*, AOAS 2022, https://doi.org/10.1214/21-AOAS1508 | adjacent-layer joint residual의 연구 근거 | High; P1 synthetic generator와 직접 일치하지 않음 |
| KORS gradient류 QC는 thermocline 정상변동을 오탐할 수 있고 stuck/drift는 어려웠다 | Min, Jeong, Jang et al., KIOST, *Quality Control of Observed Temperature Time Series from the Korea Ocean Research Stations*, 2020, https://e-opr.org/articles/article/rQWX/ | thermocline FP kill diagnostic | High; P1 synthetic anomaly와 동일 생성과정은 아님 |
| 반복된 finite validation 선택은 selection criterion을 과적합한다 | Cawley & Talbot, JMLR 2010, https://www.jmlr.org/papers/v11/cawley10a.html | exposed Q2–Q4 결과를 confirmation으로 쓰지 않는 근거 | High |
| DINEOF는 EOF 수와 reconstruction error를 CV로 선택한다 | Beckers & Rixen, 2003, https://orbi.uliege.be/handle/2268/4291 | P2 rank 1/2 mask-matched CV | High; entry CV가 synchronized 61-day blackout을 보장하지 않음 |
| multivariate DINEOF는 보조변수를 joint EOF에 포함할 수 있다 | Alvera-Azcárate et al., JGR Oceans 2007, https://doi.org/10.1029/2006JC003660 | P2 joint target/public T/S 구조 | High; 위성자료 결과여서 vertical T/S transport는 미확인 |
| side information은 inductive matrix completion에서 새 row/column 예측을 정의한다 | Jain & Dhillon, 2013, https://arxiv.org/abs/1306.0626 | P2 depth-factor 대안 | Medium; low-rank/incoherence 가정은 별도 확인 필요 |
| forecast wind와 bulk wave output으로 wave-model error pattern을 학습할 수 있다 | Ellenson et al., *Coastal Engineering* 2020, https://doi.org/10.1016/j.coastaleng.2019.103595 | P3 regional-fetch forcing-error residual | High; 미국 연안 24h 결과라 P3 transport는 미확인 |
| GFS archive는 cycle과 exact lead forecast를 제공한다 | NOAA/NCEI GFS official page, https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast | P3 as-of archive 후보 | High; 실제 P3 dates와 publication latency는 미검증 |
| ERA5는 assimilated reanalysis이며 operational issue-time forecast가 아니다 | Hersbach et al., QJRMS 2020, https://doi.org/10.1002/qj.3803 | P3 ERA5 deployment boundary | High |
| serial dependence가 있는 forecast error는 paired/block inference가 필요하다 | Politis & Romano, JASA 1994, https://doi.org/10.1080/01621459.1994.10476870; Diebold & Mariano, JBES 1995, https://doi.org/10.1080/07350015.1995.10524599 | P3 whole-episode block inference | High; 8/12 최소 block 수를 직접 정당화하지 않음 |

## Local evidence inventory

- `reports/leaderboard_headroom_double_research_20260829_v1/leaderboard_snapshot.json`
- `reports/parallel_dataset_breakthrough_cycle_20260831_v2/leaderboard-readonly-recheck.json`
- `artifacts/validation_system_audit_20260822/p1.json`
- `artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r2_disposition/OWNER_STATIC_QA_NO_GO_20260823.json`
- `configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r2_science_projection.json`
- `reports/historical_model_reaudit_20260831_v1/candidate-ledger.json`
- `reports/next_cycle_breakthrough_preflight_20260831_v1/result.json`
- `reports/official_information_probe_cycle_20260830_v1/independent-qa.json`
