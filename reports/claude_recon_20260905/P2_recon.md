# P2 정찰 보고서 — 중간층 수온 복원 (2026-09-05, read-only recon)

> 대상 경로(계획): `reports/claude_recon_20260905/P2_recon.md` — plan mode 때문에 이 파일에 작성함. 아래 내용을 그대로 복사하면 됨.
> 원칙: 관측값 원자료는 일절 인용하지 않았고, 모든 수치는 집계 통계·공식 receipt·코드 인용이다. 경로는 저장소 루트 `C:\Users\cedis\PycharmProjects\PythonProject` 기준.

## 0. 한 줄 요약

- 현재 최고 `P2_V52_SCORE_PRIORITY_FULL_HISTORY_BLEND020`(Public 0.424019)은 **80% 리더보드 튜닝 앵커(bin17_anchor.csv) + 20% clean scratch DeepSets**다. 모델 기여는 행당 RMS 0.040 °C, 최대 0.5 °C.
- 앵커의 계보(O→U→OAS50→rank-1→vertex→bin17)는 공식 Public 점수로 최소 6개 스칼라 자유도를 적합한 것이며, 최종 재현 패키지에는 그 생성 코드가 없다(`07_source/scripts`에 v12/v13/v23/v50/v52 러너만 존재).
- 내부 검증 comparator(“cross-fit bin17 champion proxy”)는 2024-09~10 fold에서 RMSE **4.88 °C**(T1 단순 복사 0.59 °C보다 8배 나쁨)이고 0.5 °C action cap이 포화(p95=p99=max=0.5)해 45개 DeepSets 변형의 순위를 사실상 구분하지 못했다. 내부 Δ −0.0527 → 공식 Δ −0.00096.
- 데이터 측면: hidden 구간 test 행의 29.3%는 T5 결측(10월 T5 유효율 41%), hidden 행의 실제 `depth`는 99.3% 유효한데 파이프라인은 nominal 수심만 사용. 2024-09~10에서 “T1 복사”가 L2/L3 RMSE 0.13/0.32로 선형보간(0.46/0.92)을 압도.
- 개선 1순위는 새 모델이 아니라 **검증 comparator 교체**와 **모델 단독/고가중 앙상블**(2점 공식 점수로 추정한 모델 단독 ≈0.437, 최적 blend ≈0.43 → ≈0.4215)이다.

---

## 1. 현재 최고 파이프라인 정확한 재구성

### 1.1 산출 수식 (`scripts/final_submission_20260905/P2/predict_submission.py:57-73`)

```
model_c    = baseline_organizer + mean_seed(DeepSet_seed(tokens, mask, ctx)) * profile_scale      # :57-67
candidate  = anchor + 0.2 * clip(model_c - anchor, -2.5, +2.5)                                     # :69-71
guard: |candidate - anchor| <= 0.5 (+1e-12), finite                                                # :72
```
- `anchor` = `03_model/decision_artifacts/bin17_anchor.csv` (SHA `99c6925c…`, = 공식 제출 `P2_1_RANK1_BIN17_ONLY`, Public 0.430194) — 코드로 재생성되지 않는 고정 CSV(`contract.json` `decision_files`).
- 후처리는 위 clip/가중치뿐. **최종 blend 뒤에는 endpoint envelope/PAVA 투영이 없다**(앵커 자체는 투영됨). −5~45 °C 클리핑 코드도 없음(실제 값 16.3~29.1이라 무해).

### 1.2 입력 특징 (`scripts/final_submission_20260905/P2/p2_pipeline.py:81-143`; 원본 `src/p2_restore/features.py`, `normalized_curvature_residual.py`)

- 공개층 토큰 5개(layer 1,5,6,7,8) × 8채널 (`p2_pipeline.py:107-126`):
  `(temp−baseline)/profile_scale`, `(psal−psal_mean)/psal_scale`, `(depth−target_depth)/50`, `(nominal−target_depth)/50`, presence 4비트(temp/psal/depth/nominal). NaN→0, clip ±12. token_mask = temp∧nominal 유효(`:126`); 유효 토큰 <2면 예외(`:127`).
- 컨텍스트 11개(`:131-140`): `target_depth/50`, layer one-hot(3), `log1p(profile_scale)`, `doy_sin/cos, hour_sin/cos, m2_sin/cos`(12.42 h).
- `baseline`: 학습 시 `features.py:26-50 _nearest_public_baseline`(nominal 수심, 목표를 감싸는 두 층 선형보간, 감싸지 못하면 **가장 가까운 두 층으로 선형 외삽**); 추론 시 조직 배포 `baseline_interp.csv`(`predict_submission.py:41,52`).
- `profile_scale` (`normalized_curvature_residual.py:195-208`): `|T1−T5|`, T1·T5 중 하나라도 결측이면 공개층 range, floor 0.5 °C. 목표(정규화 곡률) = `(truth−baseline)/profile_scale` (`:211-227`).
- **시간 문맥 없음**: 동시각 단일 프로파일만 입력(과거/미래 lag 없음). 외부 자료·사전학습 0.
- 결측 처리: 토큰 마스킹 + presence 비트. T5 결측 행은 baseline이 T1(4.19 m)–T6(30.68 m) 보간, scale=range로 바뀜.

### 1.3 학습 (`train_model.py:37-123`, `p2_pipeline.py:146-275`)

- 행 선택: `build_training_features`(target 유효 ∧ nominal-baseline 유효 ∧ 공개 temp ≥2, `features.py:136-141`) 후 KST `>= 2024-05-01` (`train_model.py:63`). 결과 **166,268행**, 2024-05-08 ~ 2025-12-10(`reports/p2_v52_score_priority_deployment_20260901_v1/result.json:7-9`). 즉 2024-05~12, 2025-04~08, 2025-11~12 전부 사용(가림 전후 양쪽 포함, full-history).
- 가중치(`p2_pipeline.py:146-182`): layer×달력월 그룹(27개) 총질량 동일 → 그룹 내 KST-일 동일 → 일 내 행 동일. 정규화 후 min 0.69, max **15.4**(4월: 2025년 10일치만 존재).
- 손실: SmoothL1(β=1) 가중평균 + **0.01 × 입력기울기 L2 penalty**(관측 토큰의 temp 채널에 대한 ∂loss/∂token², `:185-197,250-255`).
- 옵티마이저 AdamW lr 1e-3, wd 1e-4, batch 4096, **60 epoch 고정, early stopping·검증 없음**, seed 20260901/02/03 (3 fit), CUDA, `use_deterministic_algorithms(warn_only=True)`.
- 아키텍처(`:35-78`, 5,889 params): element MLP 8→32→32(ReLU) 공유 → masked mean / masked max / **masked 3차 중심모멘트**(`centered³ 평균`, `:75-77`) concat + ctx(11) → head 107→32→32→1. v13(mean/max) 헤드를 먼저 만들고 3차모멘트 열을 0으로 확장해 RNG 순서를 보존(`:47-64`).

### 1.4 bin17 anchor의 정확한 계보 (모두 공식 Public 점수 기반 선택)

| 단계 | 파일/SHA | 정의 | Public RMSE |
|---|---|---|---:|
| O | `1c959f81…` (`output/2026-08-20/ready`) | Deep stack(BiTCN/LSTI/TimeMixer++/MOMENT-scratch + LGBM router) + public-state soft gate + physical projection + 고정 외삽(L2×10, L4×2). clean | 0.541085 |
| A | `3960660b…` | LightGBM 보수 스택 0.625 (local +0.073, Public −0.172) | 0.713520 |
| U | `13181dff…` | `O + α_L·(A−O)` 층별 α_L을 **공식 α=0/0.5/1 및 층별 probe(라운드 D)의 MSE 분해**로 적합 (`scripts/build_p2_p3_public_quadratic_round_g_20260827.py`, `configs/experiments/p2_rank1_bin_decomposition_probes_20260830_v1.json`) | 0.535727 |
| alpha50 | `bd550127…` | `0.5·U + 0.5·OAS 계절(14일 bin, ±60일) 조건부 프로파일` + endpoint/PAVA (`scripts/build_p2_seasonal_oas_submission_20260827.py:96-157,208-211`). α=0.5는 **공식 α=0/0.1/0.2/0.4 점수의 기하학적 bound**로 선택(`configs/experiments/p2_seasonal_oas_alpha50_deploy_20260828.json` `metric_geometry`, `src/p2_restore/metric_geometry.py`) | 0.431252 |
| rank-1 | `665485e1…` | alpha50 + supervised PLS rank-1 잔차(공개 T/S B-spline 계수+변화, `src/p2_restore/supervised_rank1_functional_residual.py`) bin 17·18 한정 | 0.430250 |
| vertex | `bf15d705…` | rank-1 강도 0.83419 = **공식 강도 0/1/2 점수 포물선 정점** | 0.430209 |
| **bin17_only** | `99c6925c…` | vertex 보정 중 bin17(9/1~9/9)만 유지, bin18 되돌림(`scripts/build_p2_rank1_bin_decomposition_probes_20260830_v1.py:72-88`) | **0.430194** |

→ 앵커 = alpha50(=0.5U+0.5OAS) + 0.834×rank-1 보정(9/1~9/9, 762행) 후 PAVA. 공식 점수로 결정된 자유도: α_L2, α_L3, α_L4(U), OAS α, rank-1 강도, bin 선택(+ 라운드 D의 층 배정) ≈ **6~7개**.

### 1.5 역사적 0.424019(`331b1635…`) vs 재현 replay(`64f59fe7…`)

- 역사적 실행(`scripts/materialize_p2_v52_score_priority_20260901_v1.py`)은 체크포인트를 저장하지 않음(`artifacts/official_final_submission_20260905/P2/02_train/TRAINING_LINEAGE.md`). 패키지 빌드 시 동일 레시피를 CUDA에서 3 fit 재학습 → seed 20260901의 최종 data loss 0.027718(역사) vs 0.027795(replay) 등 값이 달라 CSV SHA가 다름. 원인은 `warn_only=True` 비결정 CUDA 커널.
- `contract.json`: `allow_documented_replay_variance: true`, `historical_champion_hash_exact: false`. replay는 **미채점**이며 0.424019를 replay 점수로 주장하면 안 됨(`docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md` P2 선택 규칙). 모델 기여가 20%·RMS 0.04 °C이므로 replay 점수 차이는 아마 1e-4 °C 수준이지만 검증된 바 없음.

---

## 2. 검증 설계와 약점

### 2.1 설계 (`scripts/run_p2_prefix_safe_domain_balanced_deepset_20260901_v13.py:286-323`, v50/v52 config)

- 3개 outer 블록: `2024_sep_oct`(26,273행), `2025_jul_aug`(26,693), `2025_nov_dec`(16,884) = 69,850행 “truth-free scoring frame”(`artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2/scored_predictions_no_truth.parquet`).
- fold별 학습: `2024-05-01 ≤ t < fold_start − 7일`(prefix-only) → 45,935 / 119,667 / 149,384행, 3 seed → 9 fit. 배포 모델은 full-history 166,268행(검증된 구성과 다름).
- 후보 = `reference + 0.2·clip(model−reference, ±2.5)`; 지표 = pooled ΔRMSE vs reference, fold×KST-day bootstrap 5,000회, “official-like fold” = 2024_sep_oct, 점수 환산 12.5475 pt/°C, transport penalty 0.1217 pt(`configs/experiments/…_v50.json evaluation`).
- score-priority gate(v52): pooled Δ < v23의 −0.05189, 3 fold 모두 개선, 2024_sep_oct 개선, CI90 상한 <0, transport 점수 > v23. 안정성 진단(9 fold-layer 셀 중 8 non-harm)은 **7/9로 실패**했으나 “explicit risk”로 제출.

### 2.2 comparator(reference)의 실체 — 가장 큰 약점

- `run_p2_group_balanced_raw_residual_20260901_v8.py:254-274 make_reference`: `reference = PAVA(scored.reference + [bin17 ? candidate−reference : 0])`. `scored.reference`는 `run_p2_alpha50_supervised_rank1_functional_residual_20260828_v1.py:145-168 alpha50_reference` = `current_blend50`(= 8/16 `p2_state_conditional_lean_v1` LightGBM OOF) + 0.5·(**prefix-only forward 계절 OAS** − blend50) 후 PAVA.
- 재계산(파케이+observations 병합, 집계): fold RMSE **4.882 / 1.186 / 0.281**, 2024_sep_oct 층별 1.47 / 2.41 / **7.99**, 10월만 6.47, 평균 편향 −0.88 °C. 즉 2024-09~10 fold의 “챔피언 프록시”는 가을 혼합을 본 적 없는 OAS(prefix에 2024-10 이후 자료 없음)라 성층이 유지된다고 예측 → 실제 앵커(양쪽 자료로 학습, Public 0.43)와 전혀 닮지 않음. 참고: 같은 fold에서 T1 단순 복사 0.593, 선형보간 0.972.
- 결과: 후보 action이 cap에 포화(`…v52…/result.json action_geometry`: p95=p99=p999=max=0.5, active 100%). 2024_sep_oct ΔRMSE는 45개 변형 모두 −0.079~−0.081로 사실상 동일(§5 표). 변형 간 순위는 Jul-Aug/Nov-Dec의 1e-3 차이로 결정됐고 v52 vs v23 내부 차이는 −0.00076 °C.
- 운송 기록(내부 Δ → 공식 Δ): v23 −0.0519 → −0.0052; v52(대 v23) −0.00076 → −0.00096; copula v2 −0.0106 → **+0.0121**; HGB −0.0155 → **+0.0013**; shallow −0.0102 → **+0.0083**; DROP_L4 −0.00003 → **+0.0006**; conservative stack +0.0734(local) → **−0.1724**. 부호 반전 5/7. `20260825_OFFICIAL_SCORE_RECONCILIATION.json`의 결론(“local은 calibrated predictor가 아님”)이 이후에도 유지됨.

### 2.3 regime 커버리지

- hidden 9~10월: 월평균 |T1−T5| 2025-09 1.60, 2025-10 1.43(T5 유효 41%만) vs 2024-09 2.65, 2024-10 0.48. 2024 fold는 “강성층→완전 혼합” 전이를, 2025-07~08은 강성층(5~8 °C), 2025-11~12는 완전 혼합(T5 전부 결측)을 대표. **T5가 결측인 성층/전이 상태**(hidden 10월의 핵심 regime)는 어느 fold에도 없다.
- 2024 fold는 센서 기하가 다름(L4 14.87 m, L5 19.15 m, L7 49 m; 2025는 14.74/19.59/39.45 + L8 49.35). DeepSets는 수심 토큰으로 흡수하지만 “9~10월 doy 조건”은 2024 기하에서만 학습됨.

### 2.4 누출 경로 점검

- hidden truth: `src/p2_restore/data.py:90-99`가 가림 격자 26,352행 temp/psal NaN을 assert. 러너 operation counter도 hidden 0. 누출 없음.
- 가림 밖 2025 layer 2/3/4(4~8월, 11~12월) 학습 사용: 배포 데이터이므로 허용. 2025 Nov–Dec(가림 직후)를 학습에 쓰는 것은 예보가 아니라 복원 문제이므로 규정상 문제 없음.
- OAS 앵커: `build_panel`이 layer 2/3/4 temp·psal을 y로 사용하되 가림 구간 제외(`build_p2_seasonal_oas_submission_20260827.py:110-124`). 정상.
- 리더보드 피드백: 배포 데이터 외 “정보원”이지만 규정이 명시적으로 금지한 외부 관측/재분석/예보/사전학습은 아님. 다만 **Public 분할 적응**이므로 Private 위험(§4).

---

## 3. 취약점·버그·이상한 점·한계

1. **[높음] 앵커 80%가 리더보드 적합 산물이며 패키지에서 재생성 불가.** `predict_submission.py:43-45` 고정 CSV 읽기; `07_source/scripts`에 OAS/U/rank-1/quadratic 빌더 없음. 조직이 “모델 재현”을 검증하면 답안의 80%는 코드로 설명되지 않는다. Private 분할에서 §4의 미세 자유도(bin17, 0.834, 층별 α)가 노이즈일 위험.
2. **[높음] 검증 comparator 비현실 + action cap 포화** (§2.2). `…_v8.py:254-274`, `…_v13.py:325-330`. 45개 변형 실험은 사실상 판별력 0. 재사용 금지 권고.
3. **[중] 역사적 0.424019 CSV는 재현 불가, replay 미채점.** `p2_pipeline.py:218` `warn_only=True`; 체크포인트 미저장. 최종 제출 시 “역사 CSV 업로드 vs replay 업로드” 결정이 필요(런북은 역사 CSV를 권함).
4. **[중] baseline 정의 불일치(학습 vs 추론).** 조직 `baseline_interp.csv`는 `np.interp(nominal target, sorted nominal public depths)`로 26,061행 전부 재현(최대 오차 0). `features.py:41-42`는 감싸지 못하면 **선형 외삽** → T1 결측 6개 test 행에서 최대 4.25 °C 차이(RMS 0.045). 학습 target(외삽 baseline 기준)과 추론 decode(클램프 baseline 기준)가 다름. 영향은 작지만(6행, 0.2 blend) 논리 오류.
5. **[중] T5 결측 regime(테스트 7,642행, 29.3%; 10월 59%) 취급.** 이 행들은 baseline = T1–T6(4.19–30.68 m) 보간, scale = 공개층 range, PAVA/envelope no-op(`profile_projection.py:127,169-170`), 앵커 OAS는 결측 패턴별 조건부. 어떤 검증 fold도 “성층 + T5 결측” 조합을 포함하지 않음(§2.3). 유효 공개층 수 분포(test): 2층 4행, 3층 124행, 4층 7,638행, 5층 18,295행 → 2~3층 행은 baseline이 4.19~30.68 m 또는 그 이상 간격.
6. **[중] 미사용 합법 정보: hidden 행의 실제 `depth`.** observations의 가림 행 26,352개 중 99.3%가 실제 수심 보유(L2 7.08±0.81, L3 9.47±0.80, L4 14.68±0.80 m). 파이프라인은 `target_depth = nominal`(`features.py:164`, `p2_pipeline.py:103`)만 사용. 9월 성층기 수온 기울기 ~0.3~0.5 °C/m를 감안하면 ±0.8 m 편차는 층 3·4에서 수 0.1 °C 오차원. 공개층은 실제/공칭 둘 다 토큰에 있음.
7. **[중] 공개 염분 결측**: hidden 구간 psal_1 유효율 56.6%(T1은 99.2%). psal 토큰·psal_mean/scale이 행마다 다른 층 집합으로 계산됨 → 분포 이동. 학습 구간 psal_1 유효율은 별도 확인 필요.
8. **[낮] 432행 padding**: 마지막 432행 = 2026-01-01 00:00~08:50 KST(=2025-12-31 UTC 꼬리), 8층×54행, 전 변수 NaN, `year=2026`. `features.py:136` keep 조건(target 유효)으로 자동 제외되어 무해. `elapsed_days`/`year`는 계산되지만 특징에서 배제(`normalized_curvature_residual.py:357-367` assert).
9. **[낮] layer↔수심 연도 불일치의 실체**: nominal 전환이 정확히 2025-01-01 09:00 KST(=UTC 연초)에 일어남(달력연도 라벨링). 2025-01~03은 층 1,5,8만 유효(2024형 3센서 배치), 4월부터 8층 배치. 즉 “2024 layer 7(49 m)”의 물리적 후속은 2025 layer 8. 파이프라인은 layer 번호를 쓰지 않고 수심 토큰을 쓰므로 정상이나, 앵커의 OAS(`build_panel` PUBLIC=(1,5,6,7))는 **layer 번호 열**을 쓰므로 2024 temp_7(49 m)과 2025 temp_7(39 m)을 같은 변수로 취급하고 layer 8은 쓰지 않음.
10. **[낮] 가중치 왜곡**: 9·10월 그룹은 2024년 30·31일치뿐(`MODEL_MANIFEST.json` groups) → 행당 가중이 6·7월의 ~2배; 4월은 10일치로 15.4배. 소수 일에 과적합 가능.
11. **[낮] blend 후 물리 투영 없음**: 앵커는 envelope/PAVA 투영됐지만 최종 후보는 최대 0.5 °C까지 envelope 밖으로 나갈 수 있음(`predict_submission.py:69-73`).
12. **[낮] 고정 60 epoch, 검증 없음, 3 seed**: 시드 간 손실 편차(0.0277~0.0280)는 작지만 seed 분산 추정이 없음.
13. **[정보] 조직 채점 규칙**: README “QC good ∧ 동시각 공개 수온층 ≥2”만 채점(26,061/26,352). 8,717 시각 중 8,630 3층, 84 2층, 3 1층. 공식 baseline 1.290264는 하위 layer 4(1.33~1.87 수준)와 T5 결측 행이 지배.

---

## 4. 공식 제출 이력 표 + 리더보드 프로빙 의존성

### 4.1 P2 공식 업로드 전수 (KST, Public RMSE; 출처: `20260825_OFFICIAL_SCORE_RECONCILIATION.json`, 라운드 D `OFFICIAL_RESULTS_20260826.md`, `reports/finite_horizon_submission_decision_20260827_v1`, `reports/HACKATHON_HANDOFF_2026-08-28.md`, `reports/p2_oas_alpha50_deployment_20260828_v13`, `reports/deadline_submission_results_20260828_v1`, `20260829_ADAPTIVE_FINAL_PROBES_READY/SET_MANIFEST.json`, `reports/official_information_probe_cycle_20260830_v1/p2-official-result.json`, `reports/p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v3/official-result-summary.md`, `reports/parallel_internal_pass_registry_20260831_v1`, `reports/submission_ladders_internal_validation_20260831_v1/p2_3_official_result.md`, `reports/p2_v23_official_submission_20260901_v1`, `reports/p2_v52_official_submission_20260901_v1`)

| # | 일시 | 후보 (SHA 앞 8) | RMSE | 점수 | 한 줄 설명 | 계보 |
|---:|---|---|---:|---:|---|---|
| 1 | 08-25 16:38 | O `1c959f81` | 0.541085 | 26.544 | deep stack + soft gate + 물리투영 + 고정 외삽 | clean |
| 2 | 08-25 18:23 | A `3960660b` | 0.713520 | 24.380 | LightGBM 보수 스택 w=0.625 | clean, 악화 |
| 3 | 08-25 18:24 | B `d7bb0215` | 0.599921 | 25.806 | 0.5O+0.5A | clean, 악화 |
| 4 | 08-26 22:00 | QUADRATIC_GLOBAL `9cc95180` | 0.537238 | 26.592 | O+α(A−O), α=−0.159 (공식 3점 포물선) | **probe** |
| 5 | 08-26 22:01 | PROBE_LAYER2 `5507317f` | 0.541917 | 26.534 | 같은 α를 L2에만 | **probe** |
| 6 | 08-26 22:01 | PROBE_LAYER4 `98890354` | 0.536536 | 26.601 | 같은 α를 L4에만 | **probe** |
| 7 | 08-27 22:18 | U `13181dff` | 0.535727 | 26.611 | 층별 α* 동시 적용 | **probe** |
| 8 | 08-27 | OAS-TS-10 `65b754c8` | 0.507628 | 26.964 | 0.9U+0.1OAS+PAVA | OAS clean/base probe |
| 9 | 08-27 | OAS-TS-20 `f46dec79` | 0.483661 | 27.265 | α=0.2 | 〃 |
| 10 | 08-28 | OAS-TS-40 `6e28ddb8` | 0.445147 | 27.748 | α=0.4 | 〃 |
| 11 | 08-28 19:03 | alpha50 `bd550127` | 0.431252 | 27.922 | α=0.5(공식 4점 기하 bound로 선택) | **probe-selected** |
| 12 | 08-28 | CROSSFIT_VETO_RANK1 `665485e1` | 0.430250 | 27.935 | alpha50 + PLS rank-1 bin17·18 | 모델+probe base |
| 13 | 08-29 | RANK1_AXIS_ALPHA_2 `7d06cf01` | 0.432244 | – | 강도 2 축 탐침 | **probe** |
| 14 | 08-29 | MSE_VERTEX_A083419 `bf15d705` | 0.430209 | 27.935 | 강도 0.834(공식 0/1/2 포물선 정점) | **probe** |
| 15 | 08-29 | LAYERWISE_OOF_SHRUNK `dbcef773` | (기록 미발견; 챔피언 갱신 없음) | – | 층별 강도 수축 | probe |
| 16 | 08-30 22:05 | Gaussian copula v2 `f498c6e1` | 0.442259 | 27.784 | alpha50 + 계절 copula 조건부 잔차 | clean 모델, **악화** |
| 17 | 08-30 | **RANK1_BIN17_ONLY** `99c6925c` | **0.430194** | 27.935 | vertex 보정 bin17만 (=현 앵커) | **probe** |
| 18 | 08-30 | RANK1_BIN18_ONLY `0d213e97` | 0.431267 | 27.922 | bin18만 | **probe** |
| 19 | 08-31 05:34 | BIN17_DROP_LAYER4 `6cfafc36` | 0.430800 | 27.928 | bin17 보정에서 L4 제거+PAVA | **probe**, 악화 |
| 20 | 08-31 | HGB_ABSOLUTE_PROFILE | 0.431532 | 27.919 | 앵커 위 HistGB 절대손실 잔차(내부 PASS) | clean 모델, 악화 |
| 21 | 08-31 | PROFILE_RESIDUAL_SHALLOW `098b5bb2` | 0.438464 | 27.832 | 얕은 공개프로파일 잔차(내부 PASS) | clean 모델, 악화 |
| 22 | 09-01 07:52 | v23 `a6c62a8a` | 0.424976 | 28.001 | 0.8 anchor + 0.2 v23 DeepSets | 20% clean |
| 23 | 09-01 19:15 | **v52** `331b1635` | **0.424019** | **28.013** | 0.8 anchor + 0.2 v52 DeepSets | 20% clean, **현 최고** |

외부자료(ERA5/KMA/NASA/조석) 사용 P2 제출은 없음(외부 addon은 로컬 시험 후 미채택, `reports/negative_evidence_registry_20260830_v1` P2-F04). 8/16 deep stack 계보(O)는 scratch 모델(`output/2026-08-20/receipts/P2.json`). 단 “clean”은 데이터 계보 기준이며, #4~#19는 **Public 점수를 파라미터 선택에 사용**한 후보다.

### 4.2 프로빙이 추출한 정보와 현재 후보의 의존성

- 수학적 도구: 고정 축 `p(α)=p0+α·d`의 Public MSE는 α의 정확한 2차식 → 3개 점수로 정점 복원(`build_p2_p3_public_quadratic_round_g_20260827.py:87-105`); 서로소 부분집합(층/bin) 후보를 각각 제출하면 `MSE(champion)=MSE(A)+MSE(B)−MSE(base)` 항등식으로 부분별 ΔMSE 분해(`p2-official-result.json rmse_squared_decomposition`); 새 후보의 RMSE를 이미 채점된 방향들의 span으로 상·하한(`src/p2_restore/metric_geometry.py:21-86`, 반올림 16 corner). 이는 hidden truth의 **저차원 투영(내적 ⟨e, d_k⟩)을 리더보드에서 정확히 읽어내는** 절차다.
- 현 앵커가 상속한 Public 적합 파라미터: U의 α_L2/α_L3/α_L4(라운드 C/D/E, 5회 채점), OAS α=0.5(4회), rank-1 강도 0.834(3회), bin17 선택(2회), 층 유지(1회). 효과 크기: U−O −0.005, OAS −0.105(큰 효과, 물리적 신호일 가능성 큼), rank-1 −0.001, vertex −0.00004, bin17 −0.000015, DROP_L4 +0.0006(반전). **1e-3 이하 항목은 Public 표본 노이즈 수준**이며 Private에서 방향 보장이 없다. 다행히 총 이득의 대부분은 OAS(0.5)와 O 자체에 있음.
- Private 위험 정량: Public 26,061행 표시 정밀도 1e-6이라 tiny Δ도 “측정”되지만, Private가 같은 기간의 다른 행 부분집합이라면 bin17/vertex/층별 α의 기대 Private 이득 ≈ 0 ± Public 이득. 규정 위반은 아니나 조직이 “Public 과적합”을 심사하면 설명 부담이 있고, 재현 패키지의 답안 80%가 코드 밖 CSV라는 점이 더 큰 리스크.
- 두 점수(anchor 0.430194, v52 0.424019)와 action RMS 0.0401에서 2차 항등식으로 추정: 모델 단독(w=1) Public ≈ **0.437**, 최적 w≈0.43에서 ≈ **0.4215**. 즉 clean DeepSets 단독도 U(0.536)·copula(0.442)보다 훨씬 좋고, blend 가중치 0.2는 사전등록값일 뿐 최적이 아니다(단, 이 추정 자체도 Public 정보 사용이므로 “정보”로만 기록).

---

## 5. 실패한 시도 요약 (정본: `reports/negative_evidence_registry_20260830_v1/report-source.md`, `01_P2_MUST_READ_FIRST.md §7`)

| 계열 | 무엇을 | 왜 실패 | 근거 |
|---|---|---|---|
| 8/16 GBM/deep | LGBM 곡률잔차, lean-M2, phase/state router, Optuna, DeepStack(BiTCN/LSTI/TimeMixer++), CatBoost 층별, 5,000 round | 계절 간 전이 실패(튜닝 outer 악화), fitted optimism; 최선은 router/DeepStack 0.77~0.79(69,850행 proxy) | `01_P2_MUST_READ_FIRST.md:83-119` |
| soft gate / safe residual gate | 공개상태 gate로 expert convex weight | 2024-09~10 L4 강성층 셀에서 방향 반대(초과 SSE 87%), 상태 support가 블록과 교락 | `:121-133` |
| structured-mask BiTCN(가림 밖 목표층 사용) | 61일 마스크 외삽 | LOBO 가중치 0(no-op), Deep보다 불안정 | `:138-139` |
| 물리 투영/외삽 gate | envelope+PAVA(−0.0012), 외삽 gate v2(−0.006) | 채택됨(O 계보), 그러나 Public 0.541 | `:134-145` |
| conservative stack / causal residual | LGBM 스택 0.625; causal correction은 exact no-op | Public −0.172 / −0.059 | `20260825_OFFICIAL_SCORE_RECONCILIATION.json` |
| 외부·물리 addon | TEOS, tide, NASA POWER, ERA5 | 큰 악화/no-op/극미세 harm; 현재는 규정상 금지 | registry P2-F04 |
| surrogate/architecture-matched | forward surrogate, curve/L4, matched A/B | full-prefix/Public 반전 | P2-F05~F08 |
| density/annual/offset/analog | annual transfer, terminal offset, median consensus, OAS conditional(단독), day-sequence analog, RFF | gate/전 fold 악화 | P2-F10~F17 |
| dynamic low-rank/GP, BayOTIDE | uncertainty guard로 active 0 | no-op | registry |
| supervised rank-1/heave residual | PLS rank-1 | 내부 미세, fold 회귀, 활성 0.07% — 그러나 Public에서 소폭 개선 | registry, `deadline_submission_results_20260828_v1` |
| nested PLS capacity grid | 243×3, 84 fits | pooled −0.002, fold 엇갈림 | `parallel_hpo_cycle_20260829_v1` |
| Gaussian copula v1/v2 | 계절 empirical margin+Kendall latent | v1 기술 실패; v2 내부 −0.0106 → Public **+0.0121** | copula pack v3 |
| DTW trajectory transfer v2~v2r6 | 공개층 궤적 DTW 정렬로 곡률 이식 | r6 terminal blocked; 사전 QA 반복(closure matrix) 끝에 미실행/폐쇄 | `reports/p2_public_trajectory_dtw_*` |
| joint hydrographic multitask L4 (v1, r2, r3 + verifier v1~v5) | T/S 공동 멀티태스크 | 대형 계약/검증 코드(수천 줄)만 남고 결과 승격 없음 | `src/p2_restore/joint_hydrographic_*` |
| bin17 layer factor ladder | L2/L3/L4 각 제거 | L2/L3 내부 악화, L4 Public +0.0006 반전 | `submission_ladders_internal_validation_20260831_v1` |
| HGB absolute / shallow residual / PLS2 | 앵커 위 잔차 | 내부 PASS → Public +0.0013/+0.0083 | `parallel_internal_pass_registry_20260831_v1` |
| DeepSets v8~v53(45개) | pooling(3차모멘트, logmeanexp, bilinear), 정규화(IRM, VREx, DRO, Fishr, CMD, mixup, SAM, RAdam, lookahead, dropconnect, VAT, FiLM, gradnorm, PCGrad, spectral norm, LayerNorm, CrossNet, 입력기울기, Student-t, heteroscedastic…) | 모두 같은 exposed surface에서 pooled −0.040~−0.053; 2024 fold 포화(−0.079~−0.081); 순위 차이는 노이즈 | `reports/p2_*_20260901_v*/result.json` (§2.2) |

패턴: (i) 내부 이득이 Public으로 운송된 경우는 OAS 축과 v23/v52뿐, (ii) 실패의 공통 원인은 “노출된 3블록 + 비현실 comparator”에서의 선택, (iii) 시간 문맥 모델(BiTCN, M2 phase)은 항상 전이 실패했으나 comparator 문제와 분리 평가된 적이 없다.

---

## 6. 데이터 집계 관찰 (venv 실행, 집계만)

- 행수: 2024 = 7층×52,650, 2025 = 7층×52,560 + layer8 52,506, 2026 padding 8×54. 관측소 1개(S-ORS), 2024-01-01 09:00 ~ 2026-01-01 08:50 KST(UTC 정각 경계).
- nominal(m): 2024 L1~L7 = 4.18/6.87/9.83/14.87/19.15/30.21/49.05; 2025 = 4.19/7.04/9.44/14.74/19.59/30.68/39.45 + L8 49.35. 실제 depth 표준편차 0.8~1.0 m(모든 층·연도), L1 최소 0.2 m.
- 월별 temp 유효율(핵심): 2024 L2/3/4 = 1~4월 0, 5월 0.75, 6~11월 ≥0.96, 12월 0.40; L6 1~4월 0. 2025 L2/3/4 = 1~3월 0, 4월 0.31, 5~8월 ≥0.93, **9~10월 0(가림)**, 11월 1.0, 12월 0.29. **L5: 2025-10 0.41, 11~12월 0.0**. L6/L7 2025 1~3월 0, L8 1.0.
- hidden 8,784 시각: T1∧T5 유효 70.0%(6,149); 공개층 유효 수 분포 {0:58, 1:8, 2:3, 3:42, 4:2564, 5:6109}; psal_1 유효 56.6%.
- 월평균 |T1−T5|(°C): 2024 04 2.16, 05 3.92, 06 7.72, 07 3.97, 08 3.71, **09 2.65, 10 0.48**, 11 0.06; 2025 04 0.58, 05 2.65, 06 6.59, 07 8.09, 08 5.27, **09 1.60, 10 1.43(41%)**. 2025 여름 성층이 더 강하고, 9월은 2024보다 이미 약함.
- test 26,061행: 9월 12,920 / 10월 13,141; 14일 bin {17: 3,873(9/1~9/9), 18: 6,042, 19: 6,027, 20: 6,011, 21: 4,108}; T5 결측 7,642(29.3%), T1 결측 6.
- 2024-09~10 가상 가림(층별 RMSE, n≈8.7k): 선형보간(nominal) L2 0.465 / L3 0.919 / L4 1.333, pooled 0.972; **T1 복사** L2 0.135 / L3 0.323 / L4 0.968 (pooled 0.593); T5 복사 2.58/2.49/1.97. 2025-07~08 선형보간 1.109/1.760/1.872(1.616); 2025-11~12 0.352/0.651/1.311(0.869); 2025-05~06 1.098.
- 2024→2025 같은 (doy,분) 복사 RMSE(°C): 5~8월 2.0~3.5, 11월 2.8, 12월 1.5(T1 자체가 1.5~2.8 다름). `baseline25 + (truth24 − baseline24)` 잔차 이식은 5~8월 baseline 단독보다 **악화**(예: 6월 L4 1.58→3.27), 11~12월은 중립. → 2024 동일 시각 값/잔차 직접 이식은 무용; 2024는 “계절 전이의 형태”를 배우는 용도로만 유효.
- 조직 baseline = `np.interp(nominal target, sorted nominal public depths, temps)` 26,061행 정확 재현(최대 오차 0). `features.py` 방식은 26,055행 일치, 6행(T1 결측, 외삽) 최대 4.25 °C 차이.
- hidden 행 실제 depth 유효 99.3%(L2 7.08±0.81, L3 9.47±0.80, L4 14.68±0.80 m).

---

## 7. 개선 기회 (규정 준수, 기대 이득·노력 순)

1. **검증 comparator 교체 (필수 선행, 노력 낮음, 이득: 이후 모든 판단의 유효성)**. 3 fold reference를 “배포 앵커와 같은 방식으로 양쪽 자료를 쓴 OAS/anchor 아날로그”로 재구성하거나, 아예 Δ가 아닌 **모델 단독 절대 RMSE**(baseline 대비)로 평가. 2024-09~10에서 T1 복사 0.59, 선형보간 0.97이 하한 참조. action cap 0.5는 제거. 이 한 가지로 45개 변형의 진짜 순위·seed 분산이 처음 드러난다.
2. **모델 단독/고가중 앙상블 (노력 낮음, 기대 −0.002~−0.005)**. §4.2 추정상 w≈0.4 최적, 모델 단독 ≈0.437. 리더보드로 w를 다시 맞추지 말고, (1)의 새 surface에서 v52·v23·v45c·v13 등 상위 변형 × 5~10 seed 평균을 만들고 w∈{0.2,0.35,0.5}를 로컬에서 고정. seed 평균은 3→10으로만 늘려도 모델 분산의 상당 부분 제거(비용: fit당 ~25 s GPU).
3. **hidden 행 실제 depth 사용 (노력 낮음, 기대 L3/L4에서 −0.005~−0.02)**. `target_depth`에 실제 depth(결측 시 nominal)를 쓰고 baseline도 그 수심에서 재보간(조직 baseline은 nominal). 성층기 0.8 m 편차는 층 3·4에서 수 0.1 °C. 2024/2025 학습행 모두 실제 depth 보유. 물리적으로 정당하고 배포 데이터만 사용.
4. **T5 결측 regime 전용 처리 (노력 중, 기대 10월 행에서 큼)**. test 29%가 T5 결측이며 hidden 10월 T5는 41%만 유효. (a) 마지막 유효 T5(≤12 h)와 경과시간을 토큰/컨텍스트로 추가, (b) 2024-09~10 및 2025-05~08에서 T5를 **인위 마스킹**한 증강 학습(현재 학습행은 T5 거의 항상 존재 → 분포 이동), (c) T6/T7/T8 형상+T1으로 혼합층 깊이 추정. 검증은 2024 fold에 T5 마스크를 적용한 별도 fold로.
5. **파라메트릭 thermocline 프로파일 특징 (노력 중, 기대 −0.005~−0.015)**. 시각별 4~5 공개층에 `T(z)=Tb+(Ts−Tb)·σ((zc−z)/w)`(또는 2단 tanh)을 최소제곱 적합해 (Ts, Tb, zc, w, 잔차)를 컨텍스트로 넣고, 적합 곡선의 목표 수심 값을 baseline 대안 토큰으로 제공. `src/p2_restore/dynamic_sigmoid_profile.py`가 존재하나 현재 파이프라인에 미연결 — 상태 확인 후 (1)의 surface에서 재평가. 특히 “혼합층 깊이 zc가 목표 수심보다 깊으면 L2/L3≈T1” 규칙(2024-09~10 T1 복사 0.13/0.32)을 명시적으로 표현.
6. **다중 시간 규모 문맥 (노력 중, 기대 불확실)**. 동시각 프로파일만 쓰는 현 구조에 공개층의 ±1 h/±6 h/±12.42 h/±24 h/±3 d 차분과 이동 표준편차를 토큰 채널로 추가(v14의 실패는 comparator 문제와 분리 불가). 내부파·조석 위상차 복원에 직접 효과. 미래 문맥도 허용(복원 문제).
7. **2025 기하 우선 학습 (노력 낮음)**. 가중치에 “연도 = 2025” 그룹 질량 상향(예: 2025 행 1.5×)하거나 2024 학습 후 2025 4~8·11~12월로 짧은 fine-tune. 9~10월 doy 조건은 2024에서만 오므로 doy 특징을 `|T1−T5|` 기반 성층 지수로 일부 대체해 연도 편향 완화.
8. **clean fallback 후보 준비 (노력 낮음, 전략적)**. 앵커 없이 (2)의 앙상블 단독 CSV, 또는 “O + OAS(α 로컬 고정)” 등 Public 파라미터가 없는 후보를 만들어 두고, 최종 모델 제출 시 어느 쪽을 낼지 결정. Private/심사 리스크 헤지.
9. **하지 말 것**: 같은 exposed surface에서의 DeepSets 변형 추가, 리더보드 축 탐침 추가(bin/층/강도), 2024 동일시각 값 이식, copula/PLS exact recipe 재실행, ERA5 등 외부자료.

우선 실행 순서: (1) → (3)+(4b) 소규모 재학습(각 3 seed, 수 분) → (2) → (5)/(6). 제출 슬롯은 (2)의 로컬 확정 후보 1개에만 사용.
