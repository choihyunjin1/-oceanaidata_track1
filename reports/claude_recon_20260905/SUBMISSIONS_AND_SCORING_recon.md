# 제출 이력·채점식·리더보드 프로빙·연구 프로세스 정찰 보고 (2026-09-05, 읽기 전용)

> 대상 경로(저장 예정): `C:\Users\cedis\PycharmProjects\PythonProject\reports\claude_recon_20260905\SUBMISSIONS_AND_SCORING_recon.md`
> 근거: 제출관리 폴더 `C:\Users\cedis\Downloads\해양 해커톤 제출용\*`(이하 `[제출]`), 저장소 `reports/*`, `docs/*`, `artifacts/official_final_submission_20260905/*`, 배포 데이터 README 3종. 원시 관측행·hidden truth·후보 CSV 값은 열지 않았고 집계·메타데이터만 사용했다.

## 0. 한 줄 요약

- 채점식은 세 문제 모두 "기준선 B→0점, 정책상수 T→70%, 완벽→100%, 구간별 선형"이며 우리는 항상 T 이상 구간에 있었다. P3 README가 상수를 명시(Public T=0.630065, Private T=0.621239)하고 저장된 공식 영수증 36점이 잔차 1.2e-5 이내로 이를 재현한다.
- 08-25~09-01 사이 약 64건 업로드(P1 22, P2 22, P3 20). 총점 78.01→81.13(표시) / 80.99(규정 준수 계보). 순위 2→4~5위.
- 현재 세 후보 중 P3(α=−10.217 공개 이차최적)이 Public 과적합 위험이 가장 크고, P2 V52의 anchor는 6개 스칼라가 공개점수로 정해졌지만 걸린 점수는 작으며, P1은 의존도가 낮다.
- 최종 패키지의 결정적 불일치: P2 패키지 답안 SHA(`64f5…`)가 채점된 SHA(`331b…`)와 다르고, GitHub 보존 manifest는 반대로 `331b…`를 패키지 결과라고 적었다. P3는 리더보드에 남은 KMA 계보(24.203599)와 패키지(24.066168)가 충돌한다.

## 1. 채점식과 한계 효용

### 1.1 도출 방법
- 저장된 (원지표, 점수) 공식 영수증 P1 13점, P2 10점, P3 13점을 순수 선형 최소제곱으로 적합(이번 정찰에서 재계산). 최대 절대 잔차 P1 1.16e-5, P2 5.2e-6, P3 9.8e-6 → 6자리 반올림 이내. 즉 관측 구간에서 **정확히 선형**.
- P2/P3 적합선은 (RMSE=0, 33.3333)을 지나고, P1 적합선은 (F1=1, 33.3333)을 지난다 → "완벽=100%" 조건과 일치. 단일 직선 가정으로 B를 역산하면 P1 B=−0.254(불가능)이므로 **B–T 구간과 T–완벽 구간의 기울기가 다른 2구간 선형**이 유일한 정합 해석이다.
- P3 README(`C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast\README.md` 42~53행)가 이를 명시: `B`=0점, `T`=문제 점수의 70%, RMSE=0은 100%, T를 넘어도 계속 증가. 상수표: 전체 B 0.769455/T 0.624165, **Public B 0.750046/T 0.630065**, **Private B 0.778838/T 0.621239**. 적합으로 얻은 T_public=0.63006이 README와 일치 → 검증 완료.
- P1/P2 README는 T를 적지 않고 "전체 평가 기준값"만 적는다(P1 규칙기반 F1 0.548255, P2 수심선형보간 RMSE 1.290264 ℃). 이를 B로 두면 T는 영수증 적합으로만 복원된다. `src/ocean_goal/meaningful_score_ledger_v9.py`에는 점수 환산 상수가 없고, 팀이 실제로 쓴 경험식은 `reports/leaderboard_headroom_double_research_20260829_v1/leaderboard_snapshot.json`의 `empirical_score_mapping`(P1 26.5781/F1, P2 −12.5475/℃, P3 −15.8716/m)이다.

### 1.2 문제별 식(우리가 위치한 T 이상 구간)
| 문제 | 식 (33.3333 = 100/3) | B (README) | T | 기울기 | 한계 효용 | 33.33까지 남은 점수 |
|---|---|---:|---:|---:|---|---:|
| P1 | pts = 33.3333·[0.7 + 0.3·(F1−T)/(1−T)] | 0.548255 | **0.6237**(적합) | +26.578 /F1 | **+0.01 F1 = +0.266점** | 4.42 (F1 0.833548→1) |
| P2 | pts = 33.3333·[0.7 + 0.3·(T−RMSE)/T] = 33.3333 − 10·RMSE/T | 1.290264 ℃ | **0.7970 ℃**(적합) | −12.548 /℃ | **−0.01 ℃ = +0.125점** | 5.32 (0.424019→0) |
| P3 Public | 동일 | 0.750046 m | **0.630065 m**(README) | −15.872 /m | **−0.01 m = +0.159점** | 9.27 (0.583892→0) |
| P3 Private | 동일, T=0.621239 | 0.778838 m | 0.621239 m | −16.097 /m | −0.01 m = +0.161점 | — |

- 산술적 headroom은 P3>P2>P1이지만 **입증된 headroom(문제별 1위와의 차)**은 08-29 스냅샷 기준 P1 3.10점, P2 0.70점, P3 0.58점(clean 0.72점)이다(`leaderboard_headroom_double_research_20260829_v1/leaderboard_snapshot.json`). P1 +0.01 F1의 가치가 가장 크고 격차도 가장 커서 **실질 레버는 P1**이다.
- P3 Private 함의: 같은 RMSE 0.583892라도 Private 식으로는 33.3333−16.097×0.583892 = **23.93점**(Public 24.07 대비 −0.13). 게다가 Private persistence B가 Public보다 3.8% 나쁘므로(0.7788 vs 0.7500) 우리 RMSE도 비례 악화한다면 ≈0.606 m → **≈23.57점**. 즉 P3는 과적합이 없어도 Private에서 0.1~0.5점 내려갈 구조다.
- Public 표본 노이즈(P3): 396행·66사례, RMSE 0.584. 오차가 iid 정규라면 SE(RMSE)≈RMSE·√(2/n)/2≈0.021 m(≈0.33점); 사례 내 6개 리드가 상관되어 유효 n≈66~130이면 SE≈0.036~0.05 m(≈0.6~0.8점). **P3 Public에서 0.001~0.01 m 차이는 통계적으로 식별 불가**하며, 08-30 이후 P3 프로브(±0.0003~0.004 m)는 모두 노이즈 안이다. P2(26,061행)는 SE≈0.002 ℃(≈0.02점)로 미세 차이 식별이 가능하다.

## 2. 전체 공식 제출 이력 (KST, 시간순)

범례: ★=당시 문제별 Public 최고 갱신, ◎=현재 리더보드/최종 패키지 반영, EXT=외부자료(KMA/ERA5) 계보(2026-09-01 규정상 비적격), (≈)=영수증에 점수가 없어 1.2절 식으로 환산. 출처 약칭: R0825=`[제출]\20260825_OFFICIAL_SCORE_RECONCILIATION.json`, RD=`[제출]\20260826_round_D_…\OFFICIAL_RESULTS_20260826.json`, RF=`[제출]\20260827_round_F_mstcn_e150_P1x3\OFFICIAL_RESULTS.md`, FH=`reports/finite_horizon_submission_decision_20260827_v1/report-source.md`, HO=`reports/HACKATHON_HANDOFF_2026-08-28.md`, DI=`[제출]\20260828_DEADLINE_INFORMATION_PROBES_READY\SET_MANIFEST.json`, PL=`reports/p1_official_component_score_ledger_20260901_v1/score-ledger.json`, IP=`reports/official_information_probe_cycle_20260830_v1/*-official-result.json`, CP=`reports/p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v3/result.json`(08-29 P2 UI 대조), FC=`reports/parallel_frozen_candidate_confirmation_20260830_v4/report-source.md`, LV=`reports/submission_ladders_internal_validation_20260831_v1/p?_?_official_result.md`, PR=`reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.md`, TR=`reports/parallel_public_transport_repair_cycle_20260831_v1/official-submission-receipt.json`, V33=`reports/p1_remove_i_ors_e150_information_probe_20260831_v33a/official-result.json`, V23=`reports/p2_v23_official_submission_20260901_v1/official-submission-receipt.json`, V52=`reports/p2_v52_official_submission_20260901_v1/official-submission-receipt.json`, V32=`reports/p1_v32g_official_submission_20260901_v1/official-submission-receipt.json`.

| # | 일시 | 문제 | 후보 ID | Public | 점수 | 계보 | 방법 한 줄 | 출처 |
|---:|---|---|---|---:|---:|---|---|---|
| 1 | 08-25 16:38 | P1 | original (2026-08-20 ready) | F1 0.790709 | 27.770778 | clean | 08-20 동결 LightGBM 계열 규칙+모델 | R0825 |
| 2 | 08-25 16:38 | P2 | original | 0.541085 | 26.544054 | clean | 08-20 동결 프로파일 복원 모델 | R0825 |
| 3 | 08-25 16:39 | P3 | original (O) | 0.607071 | 23.698280 | clean | CatBoost/router+persistence | R0825 |
| 4 | 08-25 18:23 | P1 | P1_IMPROVED_ENSEMBLE_V1 (A) | 0.786145 | 27.649453 | clean | 앙상블, 로컬 +0.0006 → 공식 악화 | R0825 |
| 5 | 08-25 18:23 | P2 | P2_CONSERVATIVE_STACK_IMPROVEMENT_V1 (A) | 0.713520 | 24.380424 | clean | 스택, 로컬 +0.073 → 공식 −0.172 | R0825 |
| 6 | 08-25 18:23 | P3 | P3_CORRECTED_FIXED_LONG_SHRINK_V4 (A) | 0.611680 | 23.625124 | clean | 장기리드 persistence shrink | R0825 |
| 7 | 08-25 18:24 | P1 | P1_EVENT_DAY_BALANCED_LGBM_FALLBACK_V1 (B) ★ | 0.793710 | 27.850529 | clean | 이벤트일 균형 LGBM | R0825 |
| 8 | 08-25 18:24 | P2 | …RESIDUAL_CORRECTION_V1_FALLBACK_BLEND50 (B) | 0.599921 | 25.805810 | clean | 0.5·O+0.5·A | R0825 |
| 9 | 08-25 18:24 | P3 | P3_FIXED_LONG_SHRINK_22P5_V1 (B) | 0.609346 | 23.662165 | clean | 12/18/24h에서 (O+A)/2 | R0825 |
| 10 | 08-26 22:00 | P1 | P1_1_EXPLOIT_DISAGREEMENT_ROUTER ★ | 0.817873 | 28.492736 | clean | B 대비 229행 선택 복원·제거(로컬 시간검증) | RD |
| 11 | 08-26 22:00 | P1 | P1_2_PROBE_INTERSECTION (O∩B) | 0.802928 | 28.095515 | clean | B-only 176행 제거 프로브 | RD |
| 12 | 08-26 22:00 | P1 | P1_3_PROBE_UNION (O∪B) | 0.782306 | 27.547435 | clean | O-only 824행 복원 프로브 | RD |
| 13 | 08-26 22:00 | P2 | P2_1_EXPLOIT_PUBLIC_QUADRATIC_GLOBAL | 0.537238 | 26.592326 | clean | O+α(A−O), α=−0.158977(공개 3점 이차) | RD |
| 14 | 08-26 22:01 | P2 | P2_2_PROBE_LAYER2_ONLY | 0.541917 | 26.533611 | clean | 같은 α를 2층만 | RD |
| 15 | 08-26 22:01 | P2 | P2_3_PROBE_LAYER4_ONLY ★ | 0.536536 | 26.601139 | clean | 같은 α를 4층만 | RD |
| 16 | 08-26 22:01 | P3 | P3_1_EXPLOIT_REVERSE_GLOBAL ★ | 0.599072 | 23.825229 | clean | α=−2 전체리드 | RD |
| 17 | 08-26 22:01 | P3 | P3_2_PROBE_LEAD12_ONLY | 0.606681 | 23.704466 | clean | α=−2 12h만 | RD |
| 18 | 08-26 22:02 | P3 | P3_3_PROBE_LEAD18_24_ONLY | 0.599382 | 23.820314 | clean | α=−2 18/24h만 | RD |
| 19 | 08-27 | P1 | P1_1_MSTCN_E150_ROUTER_UNION_ALL ★ | 0.833248 | 28.901363 | clean | 3-seed MS-TCN e150 full-train ∪ router(+333행) | RF |
| 20 | 08-27 | P1 | P1_2_MSTCN_E150_ROUTER_UNION_GS_ONLY | 0.822488 | 28.615402 | clean | G/S 정점 추가만(+253) | RF |
| 21 | 08-27 | P1 | P1_3_EXPLOIT_GI_NO_REMOVALS | 0.817968 | 28.495264 | clean | router 제거 12행 복원 | RF |
| 22 | 08-27 22:18 | P2 | P2_1_EXPLOIT_LAYERWISE_QUADRATIC (U) ★ | 0.535727 | 26.611283 | clean | 층별 α 공개 이차최적 | FH |
| 23 | 08-27 22:19 | P3 | P3_1_EXPLOIT_LONG_QUADRATIC_OPTIMUM ★ | 0.583892 | 24.066167 | clean | 12/18/24h α*=−10.235445 | FH |
| 24 | 08-27 | P2 | SEASONAL_OAS_TS10 ★ | 0.507628 | 26.963865 | clean | 0.9·U+0.1·계절 OAS 프로파일+PAVA | HO |
| 25 | 08-27 | P2 | SEASONAL_OAS_TS20 ★ | 0.483661 | 27.264587 | clean | 0.8·U+0.2·OAS | HO |
| 26 | 08-27 | P3 | long α=−12 | 0.584611 | 24.054757 | clean | 곡선 반대편 bracket | HO |
| 27 | 08-27 23:36 | P3 | P3_REFINED_PUBLIC_OPTIMUM_20260827 ★◎ | 0.583892 | 24.066168 | clean | α=−10.217432(공개 5점 재적합), +0.000001점 | `[제출]\20260827_P3_REFINED_PUBLIC_OPTIMUM_READY\MANIFEST.json` |
| 28 | 08-28 13:57 | P3 | ERA5_HS2_CHAMPION_MATCHED | 0.585738 | 24.036866 | EXT | 18/24h ERA5 전이모델 Hs² 잔차 25% | `reports/approved_parallel_execution_20260828_v9/p3_official_submission_receipt_20260828.json` |
| 29 | 08-28 | P2 | SEASONAL_OAS_TS40 ★ | 0.445147 | 27.747847 | clean | 0.6·U+0.4·OAS | `[제출]\20260828_P2_SEASONAL_OAS_TS40_…\제출정보.txt` |
| 30 | 08-28 19:03 | P2 | SEASONAL_OAS_TS50 (alpha50) ★ | 0.431252 | 27.922187 | clean | 0.5·U+0.5·OAS | `reports/p2_oas_alpha50_deployment_20260828_v13/official_score_receipt.json` |
| 31 | 08-28 23:44 | P1 | P1_1_E150_PLUS_GI_SPIKE2 ★◎ | 0.833548 | 28.909341 | clean | e150 union-all + GI spike 2행 | DI, PL |
| 32 | 08-28 23:44 | P1 | P1_2_E150_PLUS_GI_S5 | 0.833548 | 28.909341 | clean | e150 + S-ORS GI 5행(동률) | DI |
| 33 | 08-28 23:44 | P1 | P1_3_E150_PLUS_GI_ALL6 | 0.833333 | 28.903643 | clean | e150 + GI 6행 | DI |
| 34 | 08-28 23:44 | P2 | P2_1_ALPHA50_CROSSFIT_VETO_RANK1 ★ | 0.430250 | 27.934759 | clean | alpha50 + rank-1 계절 bin17·18 보정(α=1) | DI |
| 35 | 08-28 23:44 | P3 | P3_1_KMA_ALPHA20 ★ | 0.577671 | 24.164901 | EXT | 18/24h KMA 보정 20% | DI |
| 36 | 08-28 23:44 | P3 | P3_2_KMA_ALPHA40 ★ | 0.575262 | 24.203126 | EXT | KMA 40% | DI |
| 37 | 08-29 | P1 | P1_1_BOOTSTRAP_LOWER_BOUND_VETO | 0.820339 | 28.558277 | clean | e150 추가 325행 중 8행만 유지 | `[제출]\20260829_P1_MSTCN_LOWER_BOUND_VETO_PROBE_READY\OFFICIAL_RESULT.json` |
| 38 | 08-29 22:20 | P1 | P1_1_GI_SINGLE_ROW_1 | 0.833548 | 28.909341 | clean | GI 2행 중 1행(동률→행2 효과 0) | FC(p1-trial18…qa.json) |
| 39 | 08-29 22:20 | P1 | P1_2_GI_SINGLE_ROW_2 | 0.833248 | 28.901363 | clean | 다른 1행(효과 0 확인) | FC |
| 40 | 08-29 22:20 | P2 | P2_RANK1_AXIS_ALPHA_2 | 0.432244 | 27.910(≈) | clean | rank-1 벡터 강도 2배 | CP |
| 41 | 08-29 22:51 | P2 | P2_2_LAYERWISE_OOF_SHRUNK | 0.430253 | 27.935(≈) | clean | 층별 OOF 축소 강도 | CP(추론 대조) |
| 42 | 08-29 22:51 | P2 | P2_1_OFFICIAL_MSE_VERTEX_A083419 ★ | 0.430209 | 27.935277 | clean | rank-1 강도 α=0.834(공개 3점 정점) | CP, IP |
| 43 | 08-29 | P3 | KMA lead-split 프로브 2건 | 0.577577 / 0.576264 | 24.167 / 24.187(≈) | EXT | (α18,α24) 분리 강도 | `[제출]\20260829_ADAPTIVE_FINAL_PROBES_READY\SET_MANIFEST.json` official_evidence |
| 44 | 08-29 | P3 | P3_1_OFFICIAL_GEOMETRY_L18_08477_L24_00629 | 미확인(개선 아님) | — | EXT | 공개점수 분리기하 예측 0.5678 → 영수증 없음, champion 미갱신 | `reports/negative_evidence_registry_20260830_v1/report-source.md` 87행 |
| 45 | 08-30 21:19 | P3 | P3_KMA_UNIFORM_0425 ★(표시 최고) | 0.575233 | 24.203599 | EXT | KMA 42.5%(공개 3점 정점) | FC |
| 46 | 08-30 22:05 | P2 | GAUSSIAN_COPULA_V2_FROZEN | 0.442259 | 27.784078 | clean | 계절 Gaussian copula 조건부 잔차 | FC, CP |
| 47 | 08-30 22:54 | P1 | remove_g_ors_e150 | 0.829029 | 28.789240 | clean | e150 G 15행 제거 | IP |
| 48 | 08-30 22:54 | P1 | remove_s_ors_e150 | 0.833548 | 28.909341 | clean | S 238행 제거(동률) | IP |
| 49 | 08-30 22:55 | P1 | keep_i_ors_e150_only | 0.829029 | 28.789240 | clean | G+S 253행 제거 | IP |
| 50 | 08-30 | P2 | P2_1_RANK1_BIN17_ONLY ★ | 0.430194 | 27.935464 | clean | bin18 보정 제거 | IP |
| 51 | 08-30 | P2 | P2_2_RANK1_BIN18_ONLY | 0.431267 | 27.922001 | clean | bin17 보정 제거 | IP |
| 52 | 08-30 22:56 | P3 | KMA leave-S-out | 0.579102 | 24.142185 | EXT | S-ORS 보정 제거 | IP |
| 53 | 08-30 22:56 | P3 | KMA leave-I-out | 0.578951 | 24.144591 | EXT | I-ORS 보정 제거 | IP |
| 54 | 08-31 05:07 | P3 | P3_2_KMA_A18_0200_A24_0425 | 0.576589 | 24.182070 | EXT | 18h 0.2/24h 0.425 | LV |
| 55 | 08-31 05:34 | P2 | P2_3_BIN17_DROP_LAYER4 | 0.430800 | 27.927863 | clean | bin17 보정에서 4층 제외(내부 strict PASS) | LV |
| 56 | 08-31 | P1 | P1_2_HIST_GBDT_OOF_STACK_UNION | 0.833548 | 28.909341 | clean | +4행 add-only(동률) | PR |
| 57 | 08-31 | P2 | P2_2_HGB_ABSOLUTE_PROFILE | 0.431532 | 27.918675 | clean | HGB 절대 프로파일(내부 −0.0155) | PR |
| 58 | 08-31 | P2 | P2_1_PUBLIC_PROFILE_RESIDUAL_SHALLOW | 0.438464 | 27.831700 | clean | 얕은 프로파일 잔차(내부 −0.0102) | PR |
| 59 | 08-31 | P3 | P3_2_EXTRATREES_HARD_PHYSICAL_ROUTER | 0.590956 | 23.954041 | EXT(KMA champion 라우팅) | 사례별 base/KMA 라우터 | PR |
| 60 | 08-31 21:22 | P1 | P1_1_LABEL_FREE…LABEL_SHIFT_EM (v30) | 0.798819 | 27.986329 | clean | EM 유병률 이동 보정 add-only 755행, 내부 PASS → **−0.923점** | TR |
| 61 | 08-31 21:23 | P3 | P3_1_KMA_CONTINUOUS_WAVE_POWER_FACTOR (v19) | 0.589840 | 23.971758 | EXT | 연속 파워 팩터 | TR |
| 62 | 09-01 02:45 | P1 | P1_REMOVE_I_ORS_E150 (v33a) | 0.822795 | 28.623545 | clean | I 80행 제거 프로브(I 효과 +0.0108) | V33 |
| 63 | 09-01 07:52 | P2 | P2_V23_PUBLIC_TEMP_INPUT_GRADIENT_FULL_HISTORY_BLEND020 ★ | 0.424976 | 28.000939 | clean | 0.8·bin17anchor+0.2·DeepSets(v23) | V23 |
| 64 | 09-01 19:15 | P2 | P2_V52_SCORE_PRIORITY_FULL_HISTORY_BLEND020 ★◎ | 0.424019 | 28.012945 | clean | 0.8·bin17anchor+0.2·v52 3-seed DeepSets | V52 |
| 65 | 09-01 19:45 | P1 | P1_V32G_E150_CATBOOST_PRECISION_UNION | 0.832905 | 28.892255 | clean | +23행 CatBoost(내부 NO_GO 확인 제출) | V32 |

준비됐지만 미제출: Round E P1 G_ONLY/I_ONLY, Round G P2 endpoint/PAVA·P3 α=−8, `20260829_P3_KMA_LEAD_SWEEP` 02~05, P3 V5 ExtraTrees v7 재판정본, P1 V28(양성 169,011행 전부 → 붕괴 후보, 미업로드), P1 V31(출력 SHA가 champion과 동일=무변경), P1 S_ORS_LAYER6 v34a(정보 프로브, 영수증 없음), P2 V7 ExtraTrees(08-31 일일 한도 초과로 차단), P3 LEAD_CONTINUOUS score-priority(fresh 1사례 +0.0226 m 악화). 리더보드 추이: 78.01(08-25) → 78.92(08-26, 4위) → 79.58(08-27, 5위) → 81.05(08-28, 4위) → 81.05(08-30, 5위) → 표시 81.13/clean 80.99(09-01).

## 3. 리더보드 프로빙 분석과 현재 후보의 의존성

### 3.1 프로빙 관행별로 공개점수에서 추출한 것
| 관행 | 일자 | 추출한 정보/파라미터 | 현재 후보 반영 |
|---|---|---|---|
| Round C/D 사전등록 3×3(블라인드) | 08-26 | P1: O/B 양성집합 2×2 factorial(B-only 176행 −0.009, O-only 824행 −0.012 → 교집합 우월); P2: 층별 MSE 기여(L4 −0.0049, L2 +0.0009); P3: 리드별 기여(18/24h −0.0093, 12h −0.0005) | P1 router 채택 근거, P2 U의 층별 α, P3 12/18/24h 축 정의 |
| 공개 이차최적(exact algebra) | 08-26/27 | 고정 Public에서 O+α(A−O)의 MSE는 α의 정확한 2차식 → 3점으로 α* 복원. P2 전역 α*=−0.159, P2 U 층별 α, P3 α*=−10.235 → 5점 재적합 −10.217 | **P3 후보 전체**, P2 anchor의 U |
| 정보 프로브(deadline) | 08-28 | P1 GI 2/5/6행 중 spike2 선택(+0.0003 F1); P2 rank-1 bin17·18 방향 확인; P3 KMA 20/40% 단조 | P1 GI2, P2 rank-1 |
| 적응형 축 프로브 | 08-29(당일 점수 본 뒤) | P1 단일행 분해(행1이 전부, 행2=0); P2 rank-1 강도 α=2 → 정점 0.834; 층별 OOF 축소 | P2 anchor(정점은 bin17+18 기준; 이후 bin17-only가 대체) |
| 정점 ablation | 08-30 | P1 e150 성분: G +0.0045, S 0(표시), I +0.0108(09-01); P2 bin17 +/bin18 −; P3 KMA 정점 기여 G/I/S=12.75/42.75/44.5% | P1: 성분 유지 판단, P2: bin17-only anchor |
| bin17 층 팩터 사다리 | 08-31 | 4층 제외가 내부 PASS → 공식 +0.000606 ℃ 악화 | 미반영(anchor는 bin17 전체층) |
| KMA 리드 팩터 사다리 | 08-31 | (0.2,0.425) 공식 악화 | 미반영(EXT 계보 폐기) |
| P1 union/intersection | 08-26 | 상동 | 구조 근거만 |

관찰: 총 64건 중 순수 "점수 갱신 후보"는 약 20건이고 나머지는 프로브·확인·사다리다. 진짜 점수를 만든 것은 구조 변경 4건(router +0.64, e150 +0.41, OAS 10→50% 합계 +1.31, V23/V52 +0.08)과 P3 반전축(+0.37)이며, 08-29 이후 30여 건의 프로브가 만든 순증은 P2 +0.0002, P3 +0.0005점이다.

### 3.2 현재 후보별 리더보드 의존성과 Private 위험
**P1_1_E150_PLUS_GI_SPIKE2** (`57844ef2…`): e150(로컬 full-train 3-seed) ∪ router(B 대비 229행, 로컬 시간검증으로 선택 후 공식 +0.024 확인) + GI 2행. 공개점수로 정한 이산 선택은 2개(union-all vs GS-only, spike2 vs S5 vs all6)뿐이고 걸린 점수는 GI2 +0.008점. P1 README에는 Public/Private 분할이 적혀 있지 않아 분할 존재 자체가 미확인. 위험 **낮음~중간**(라우터 229행이 특정 Public 행에 맞춰졌을 가능성이 유일한 우려; F1 표본 노이즈 ≈0.004~0.006).

**P2_V52_SCORE_PRIORITY_FULL_HISTORY_BLEND020** (`331b1635…`) = 0.8·bin17 anchor + 0.2·v52 DeepSets(3-seed, 로컬 학습). anchor 계보: O →(A−O 축)→ U(층별 α 3개, 공개 이차) → alpha50(OAS 혼합비 0.5, 공개 사다리 0.1/0.2/0.4/0.5 단조) → rank-1 bin17 보정(강도 1, bin 선택은 공개 분해). **공개점수로 고정된 스칼라 ≈6개, 관련 업로드 ≈22건.** 다만 이 미세 단계들이 만든 점수는 U +0.010, rank-1 +0.013, bin17-only +0.0002, 정점 +0.0005점으로 합쳐도 ≈0.03점이라 전부 역전돼도 손실은 미미하다. 큰 몫(OAS 0→50% = +1.31점, 블렌드 +0.08점)은 공개 단조성과 사전 예측구간 일치(TS50 예측 0.4322 vs 실측 0.4313)로 뒷받침되나, OAS 혼합비 0.5가 Private 최적보다 클 가능성은 남는다(예: Private 최적 0.3이면 ≈0.1~0.3점 손실). P2 README는 8,717시각 26,061키 전체를 채점한다고 적고 Public/Private 분할 언급이 없어, **P2에 Private 분할이 없다면 위험은 0**이다. 위험 **중간(분할이 있을 경우)/없음(없을 경우)**.

**P3_REFINED_PUBLIC_OPTIMUM_20260827** (`ea65370a…`): 12/18/24h 600행에 O+α(A−O), α=−10.217. α는 오직 Public 396행의 공식점 5개(α=0, 0.5, −2 재구성, −10.235, −12)로 정확 적합한 값이며, 로컬 근거는 **반대 부호**(historical 창에서 α=−2가 +0.0022 m 악화; RD `local_vs_official.P3` 3/3 부호 반전). 원래 B 중점(α=0.5)에서 20배, α=−2 프로브에서 5배 외삽이고, O 대비 RMS 변화 ≈0.16 m·최대 ≈0.9 m(α=−2의 0.031/0.172 m × 5.1). 공개 MSE 곡선 q(α)=0.000264α²+0.0054α+0.3685에서: (i) Private의 b가 ±50% 다르면 최적 α∈[−5,−15], α=−10.2의 손실은 a·Δα²≈0.0066 MSE ≈0.006 m ≈0.09점(감내 가능); (ii) **b의 부호가 뒤집히면**(로컬 analogue가 시사) q=0.451 → RMSE 0.672 → O 대비 **−1.4점**. 반면 α=−2는 같은 시나리오에서 최악 −0.16점, 상방 +0.13점. 표본 66사례의 SE(RMSE)≈0.02~0.05 m라 α*의 표본오차도 크다. 위험 **높음**, 하방 비대칭.

## 4. 연구 프로세스 약점

1. **로컬 CV의 예측력 부재가 반복 확인됨.** 08-25: 5쌍 중 방향 일치 1(20%)(`[제출]\20260825_OFFICIAL_SCORE_RECONCILIATION_요약.md`). 08-26: P1 router 로컬 +0.0022 vs 공식 +0.024(10.8배), P2 층 순위 역전, P3 3/3 부호 역전. 08-27 `official_probe_value_deep_research`: 14개 contrast 중 부호 일치 6. 08-31: 내부 PASS 4건 → 신규 champion 0건(P2 −0.017/−0.104, P3 −0.250점), 내부 strict PASS v30 → **−0.923점**, P2_3 bootstrap P(개선)=98.8% → 공식 악화, copula 로컬 −0.0106 → 공식 +0.0121 ℃, P3 CatBoost selection −0.0229 → confirmation +0.0080 m, P1 trial18 Q2 +0.0006 → Q3/Q4 −0.0119. 예외적으로 같은 계열 nested 비교(V52 vs V23 내부 −0.00076 → 공식 −0.00096 ℃)만 맞았다. 결론: 로컬은 "같은 계열 내 소폭 비교"에만 쓸 수 있고 계열 간·구조 변경의 공식 전이는 예측 못 한다.
2. **문턱의 양극단.** 초기 승격 gate(P1 ΔF1≥+0.0255·CI90 하한 +0.012, P2 ≤−0.060 ℃, P3 ≤−0.030 m; `leaderboard_gap_research_20260826_v2`)는 실제로 점수를 만든 후보(router +0.0022, U +0.0008 ℃)를 전부 탈락시켰을 수준이고, 반대로 08-31~09-01의 "score-priority" 후보는 안정성 실패(V52 fold-layer 7/9, P3 lead-continuous fresh 1사례 악화, V32G 내부 NO_GO)를 알고도 제출했다. `public_transport_calibration v1~v3`는 n=1~6쌍의 잔차로 "penalty"를 만든 유사 정밀도다.
3. **작은·재사용된 검증면.** P3 181~182사례/1,086행·3창, P1 Q2/Q3/Q4 모두 노출, P2는 정확한 champion OOF가 없어 proxy 비교(`submission_ladders_internal_validation_20260831_v1/report-source.md` "비교 기준의 한계"). 자기감사(`research_process_self_audit_20260831_v1`)도 "선택면 과적합"을 주원인으로 지목.
4. **과잉 엔지니어링.** P1 실험 ID 155개·결과 149개·v5~v60 미세특징 사이클, P2 v8~v53 DeepSets 변형 40여 종, P3 v21~v81 기술자 사이클, 48 family 전수 재감사, 1,943개 파일 보존(`github_preservation_20260905/manifest.md`). one-shot lock이 기술 실패로 소모(ERA5 runner `catboost` 미설치, P1 r8 격리실행기)된 사례도 있다. 09-01 이후 P1 60개 사이클의 공식 반영은 0건.
5. **지표 혼용.** 내부 ΔRMSE×기울기로 "예상 점수"를 만들어 우선순위를 정하고(P3_2 +0.028점 예상 → −0.022 실측, V23 +0.53점 예상 → +0.065 실측), 이를 다시 penalty로 보정하는 순환.
6. **적응형 선택.** 08-27 OAS 10→20%, 08-28 40→50%→crossfit, 08-28 KMA 20→40, 08-29 "adaptive axis" 3건, 08-29 KMA 분리 프로브, 08-31 사다리 1→2번 순차 제출은 모두 당일 점수를 보고 다음 파일을 정한 것이다. P2/P3 업로드의 약 절반이 이 유형이며 Ladder/재사용 holdout 문제를 스스로 인용하면서도 반복했다.
7. **외부자료 계보 관리 실패.** 08-28~08-31 P3 8건이 KMA/ERA5 계보로 제출됐고 09-01 운영진 공지로 비적격 판정(`00_ORGANIZER_DATA_POLICY.md`). 표시 최고 24.203599가 리더보드에 남아 있으며 철회 절차는 "별도 문의" 상태.
8. **슬롯 낭비.** +0.000001점(refined α), +0.000473점(α=0.425), 단일행 분해 등 P3 노이즈(≈0.3점) 이하의 프로브에 슬롯을 썼고, 반대로 가장 큰 headroom인 P1 구조 변경은 08-27 e150 이후 시도되지 않았다.

## 5. 최종 패키지 상태 점검 (`artifacts/official_final_submission_20260905`, `reports/official_final_submission_20260905`, `docs/OFFICIAL_FINAL_SUBMISSION_20260905.md`)

| 항목 | 상태 | 위험/불일치 |
|---|---|---|
| P1 답안 | `57844ef2…` byte-exact, 저장 가중치 3개(각 ≈210 MB) 추론 | 업로드 파일 24개(조각 22개 + core + manifest). 포털이 다중 파일/50 MB 조각을 받는지 미확인. `01_data/derived/train_features.parquet` 4조각(≈156 MB)은 추론에 불필요할 가능성 → 제외 검토 |
| P1 재현 시간 | TRAIN notebook은 해시 감사만, 실제 3-seed×150 epoch 재학습은 장시간 | 운영진 "전체 재현 6시간 이내"(`00_ORGANIZER_DATA_POLICY.md` 21·31행)를 재학습 기준으로 적용하면 P1은 충족 근거가 없음. README에 소요시간·GPU 명시 필요 |
| **P2 답안** | 패키지 답안 `64f59fe7…`(fresh 3-fit replay, 미채점) ≠ 채점 SHA `331b1635…` | `MASTER_MANIFEST.json`의 `candidate_hash_exact: true`와 `historical_champion_hash_exact: false`가 병존해 혼동. `reports/github_preservation_20260905/manifest.md`는 "P2 SHA 331b…"를 패키지 결과로 기재 → **문서 간 모순**. CUDA 3-seed 60 epoch 학습은 재실행 시 비트 동일 보장이 없어 심사자 재현도 새 SHA가 나온다 |
| P3 답안 | `ea65370a…` byte-exact, CatBoost 체인 2개(9 파일) + 봉인 α | α=−10.217은 공개점수로 적합한 상수를 코드에 하드코딩. 심사 시 "리더보드 튜닝"으로 보일 수 있고, Private 위험(3.2절) |
| P3 리더보드 | 팀 P3 최고 표시는 KMA 24.203599(비적격) | 최종 집계가 "문제별 Public 최고"라면 비적격 점수가 집계되고, 패키지와 SHA도 다름. 철회/소명 미해결(`reports/p3_clean_incumbent_reset_20260901_v1`) |
| 마감 | 참가자 공지 2026-09-07 vs 랜딩 2026-09-30 | `configs/final_submission_portal_20260905.json` `exact_deadline_time_verified: false`. 09-07이면 잔여 2일×3회 |
| 잠금 | "모델 최종 제출"이 답안 업로드를 잠금 | 답안 업로드 → 최종 모델 순서 준수 필요(runbook 118~125행) |
| Git 기준점 | 패키지 commit `48da22f1`, core ZIP은 `95255588`에서 갱신 | 두 커밋 혼재; 저장소 URL 공개 브랜치 `codex/p1-qc` |
| QA | pytest 5, Ruff PASS, notebook 6개 오류 0, 업로드 26개 ≤50 MB, credential 0 | 형식 QA는 양호 |

## 6. 전략적 결론

전제: 실제 마감·최종 집계 규칙(문제별 Public 최고 vs 최종 지정 답안의 Private 평가)을 로그인 공지에서 먼저 확정한다. 아래는 "최종 지정 답안이 Private로 평가된다"는 가정과 잔여 2일(문제당 6회)을 기준으로 한다.

**1순위 — P3 하방 헤지(슬롯 1~2회, 연구 0).** α=−10.217 후보는 Public 최적이지만 Private 손실 꼬리(−1.4점)가 상방(+0.13점)보다 훨씬 크고 로컬 근거는 반대 부호다. 66사례 표본에 대한 통상적 수축(공개 최적의 약 50%)으로 **α≈−5**(공개 예측 RMSE ≈0.590, 24.0 점, Public 손실 ≈0.10점)를 최종 답안으로 지정하는 것이 기대 Private 점수를 높이고, 절대 안전을 원하면 **α=−2**(`57a90beb…`, 0.599072)를 쓴다. α=−5 파일은 공개 점수가 해석적으로 예측되므로 확인용 1회 업로드로 충분하다. 어떤 경우에도 KMA 계보와 P3 미세 프로브는 더 올리지 않는다. 안전 폴백: α=−2, 그다음 O(`d89e69b9…`).

**2순위 — P2 패키지 답안의 공식 점수 확보(슬롯 1회, 연구 0).** replay CSV `64f59fe7…`를 한 번 채점해 `331b…`와의 차이(예상 ±0.002 ℃, ±0.03점)를 기록한다. replay가 동률 이상이면 패키지 답안=제출 답안으로 통일해 SHA 불일치를 없애고, 나쁘면 `331b…`를 최종 답안으로 지정하고 패키지 문서에 그대로 명시한다. 동시에 `github_preservation_20260905/manifest.md`의 P2 SHA 오기와 MASTER_MANIFEST 플래그를 정정한다. 폴백: `331b…`; 블렌드 자체가 문제 되면 bin17-only `99c6925c…`(0.430194).

**3순위 — P1은 현상 유지, 남는 시간은 헤지 검증에.** P1 champion은 byte-exact 재현되고 공개 의존도가 낮으며, 남은 프로브 계열(성분 제거·add-only·라벨이동)은 전부 공식 음성으로 닫혔다. 2일 안에 새 backbone을 미사용 시간블록으로 검증할 수 없으므로 P1 슬롯은 쓰지 않는다. 폴백: e150 union-all `a52dc49c…`(0.833248, GI 행 없음). 마감이 09-30으로 확인될 때만 P1 구조 연구(유형별 검출기+구간 디코더, 2025 Q4 이후 미노출 블록 확보)를 재개한다. 이때도 로컬 gate는 "같은 계열 nested 비교"에만 쓰고, 계열 변경은 사전등록 1회 공식 확인으로만 판단한다.

**병행 행동:** (a) 운영진에 KMA 계보 8건의 처리 방식과 최종 집계 규칙을 문의하고 답을 문서화, (b) P1 24파일 업로드 가능 여부와 6시간 재현 해석을 확인, (c) 답안 업로드 완료 후에만 "모델 최종 제출"을 문제별로 클릭.

**과적합 판정 요약:** P3 α=−10.217 = Public 과적합 가능성 높음(헤지 필수), P2 V52 anchor = 공개 미세튜닝 다수이나 걸린 점수 ≈0.03점(수용), P1 = 낮음(유지).
