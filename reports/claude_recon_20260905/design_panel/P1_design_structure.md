# P1 structure/physics-first 재설계: 결정론 구조 탐지기 + T–S 분리 특징 + 유형별 구간 완성 (src/ocean_v2/p1, CPU LightGBM, 반기 블록 정직 CV)

> 설계 패널(읽기 전용 설계자) 산출물, 2026-09-05. 저장소 파일·데이터는 수정하지 않았음.

## 기대 효과

현 챔피언(Public F1 0.8335 = 28.91점)은 양성의 94.8%가 학습 코드 없는 동결 CSV에서 오므로 재현검증 탈락 시 실효 가치가 0이다. 새 파이프라인의 기대치(정직 추정): C0 안전 기준선 0.79~0.83(챔피언 대비 −0.01~−0.04 F1 = −0.3~−1.1점이지만 완전 재생성·결정론·규정 준수); C1(물리 특징) +0.01~0.03; C2(구간 완성) 추가 +0.02~0.05(불확실). 현실 목표 Private F1 0.85~0.88 → 챔피언 대비 +0.4~+1.2점, 구조 상한(유형별 recall/precision 가정) F1≈0.92 → +2.3점. 근거: (1) flatline은 하드 룰로 precision≈1/recall 0.976(자연 동일값 run 최대 5행, flatline 97.6%가 ≥6행 run), test에도 27개 run 2,401행 존재; (2) 주입은 temp만 건드려 psal이 조용함(자연 >1℃ 점프의 |Δpsal| 중앙 0.22 vs 주입 진입 0.006~0.012) → 최대 FP원인 S-ORS 자연 점프(자연 쌍의 7.8%가 >0.8℃)를 분리; (3) offset은 대칭 점프쌍(|entry|,|exit|≥0.5℃ 96.9%, 부호 반대 96.9%), drift는 이탈 점프 ≥0.8℃ 96.4%(중앙 4.7℃)로 정확 경계 앵커가 있어 FN의 98.6%를 차지하는 '부분 탐지 이벤트 내부' 손실을 직접 공격; (4) 이벤트는 세그먼트 경계 3행 이내에 0건이며 test flat run도 이를 따름(1/27 vs 독립 시 ≈9). 불확실성: 완성 디코더 임계값이 32 offset/28 drift 통계에서 유도되므로 과적합 위험, test 예산 4.1% 가정은 소프트로만 사용, 1위 F1≈0.96은 이 구조가 실제 착취 가능함을 시사.

## 재사용 대상

- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\data.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\features.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\postprocess.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\rules.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\metrics.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\splits.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\models_tabular.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\change_points.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\P1\predict_submission.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\P1\p1_pipeline.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\build_official_final_submission_20260905.py
- C:\Users\cedis\PycharmProjects\PythonProject\artifacts\runs\20260813T153038+0900_cv_378a4e89\selection.json
- C:\Users\cedis\PycharmProjects\PythonProject\configs\experiments\p1_matched_budget_local_compare_20260825_v1.json
- C:\Users\cedis\PycharmProjects\PythonProject\reports\P1_FAILURE_RECON_2026-08-13.md
- C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly\README.md
- C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly\score.py
- C:\Users\cedis\.claude\plans\mellow-wishing-storm-agent-a6aaa7f9a99443372.md

# P1 설계안 (angle A: structure/physics-first) — `src/ocean_v2/p1/` — 2026-09-05

> 읽기 전용 정찰 + 원시 데이터 집계(집계값만, 관측행 미인용)로 도출. 모든 수치는 train.csv 776,706행 / test.csv 169,011행 기준. 동일 내용을 `C:\Users\cedis\.claude\plans\mellow-wishing-storm-agent-a6aaa7f9a99443372.md`에 저장함.

## 0. 이번 정찰에서 새로 확인한 구조적 사실 (설계의 근거)

| # | 사실 | 수치 | 설계 함의 |
|---|---|---|---|
| S1 | flatline은 결정론적 | 자연 동일값 연속 최대 5행(1개 run); ≥6행 자연 run 0개; flatline 행의 97.6%(6,288/6,441)가 ≥6행 동일값 run 안 | 하드 룰 `identical_run ≥ 6` → precision≈1.0, recall≈0.976. test에는 27개 run(2,401행, 전부 ≥12행, 5~11행 run 0개) |
| S2 | 이벤트는 세그먼트 경계를 절대 안 건드림 | 경계 3행 이내 양성 0/7,975행; 이벤트 최소 경계거리 4, 5% 분위 23; <100행 세그먼트의 양성 5/17,586 | edge0 행은 무조건 0; edge_dist·seg_len을 특징으로. test 자체 flat run도 1/27만 경계 3행 이내(독립이면 ≈9 기대) → test에서도 성립 |
| S3 | 단일 점프 룰은 무용 | 자연 연속쌍의 7.8%가 \|Δtemp\|>0.8℃(S-ORS 84%), 2.1%가 >2.0℃; spike 형태룰 1.5℃: TP 96/101, 자연 FP 5,012 | spike는 F1 기여 0.3%라 공격 대상 아님(FP만 회피) |
| S4 | **주입은 temp만 건드린다 (T–S 분리)** | 자연 >1℃ 점프: \|Δpsal\| 중앙 0.22, <0.02는 4.5%. 주입 offset/noise/spike 진입행: \|Δpsal\| 중앙 0.006~0.012, <0.02가 56~61%. noise 내부 psal 거칠기 0.015 vs 자연 거친 창 0.24(<0.02는 0.8%) | "psal이 조용한 temp 점프/거칠기" = 핵심 물리 특징. psal 결측 test 0.47%(train 2%) |
| S5 | offset = 대칭 점프쌍 | \|entry\|,\|exit\| ≥0.5℃ 96.9%(≥0.8: 87.5%), 부호 반대 96.9%, \|entry+exit\| 중앙 0.08(90% 2.3); 내부 \|Δ\| 중앙 0.021(자연 0.031) | 진입·이탈 점프를 경계 앵커로 → 구간 완성 |
| S6 | drift = 무음 진입 + 큰 이탈 점프 | \|entry\| 중앙 0.02; \|exit\| ≥0.8℃ 96.4%(중앙 4.7℃); 적합 총상승 중앙 4.8℃(10% 2.3, 90% 17.5); 상승 부호 +54% | 이탈 점프에서 역방향 ramp 역투영으로 시작점 복원 |
| S7 | 레벨 이동만으로는 offset 분리 불가 | own-median/T–S 회귀/peer/인접층 잔차 모두에서 자연 250행 창의 99% 분위 z ≈ 3.1~8.4 → offset 32개 중 47~72%만 초과 | 새 기준신호보다 **경계 점프 + psal 정숙 + 부호/지속시간 정합**이 판별력 |
| S8 | 계열별 고정 예산 | 23계열 전부 3.99~4.19%(예외 S-ORS 2025 L3 5.46%); 유형 구성은 계열마다 랜덤 | test 기대 양성 ≈ 4.1%×169,011 ≈ 6,930(불확실: H1 부분율은 3.89~4.76%) — 소프트 사전정보로만 |
| S9 | 이벤트 지속시간(행) | spike 1; flatline 12~283(중앙 87); noise 23~353(중앙 198); offset 48~730(중앙 248); drift 101~519(중앙 313); 복합 12개 run(양성의 2.1%) | 완성 디코더의 길이 제약(분위 기반, fitted_params) |
| S10 | 정직 fold 규모 | H1_2025 208,093행/9,911양성/장기이벤트 41+spike 21; H1_2024(S만) 108,113/4,210/27+12; H2_2025 287,862/10,685/57+43; H2_2024 172,638/7,320/37+25 | 반기 블록 4-fold, Jan–Jun 2개 fold가 1차 선택면 |

## 1. 패키지 구조 `src/ocean_v2/p1/` (신규, 자체 완결, CPU 전용)

```
src/ocean_v2/common/    paths.py(P1_DATA_DIR 해석) · hashing.py(sha256) · determinism.py(seed, thread 고정) · submission.py(스키마·키순서 검증)
src/ocean_v2/p1/
  __init__.py
  config.py      # 고정 설정 dataclass + configs/ocean_v2/p1.json 로더. 튜닝값은 전부 None → fitted_params.json에서만 주입
  data.py        # load_train/test(immutable, sha), canonical order(station,layer,time)+원위치 매핑, 세그먼트(정확 10분), 이벤트 테이블(events_from_labels)
  structure.py   # 결정론 구조 프리미티브(§2) + derive_structural_constants(train)→fitted_params 초안
  features.py    # 80개 기존 offline 특징(복사) − depth 파생 + 물리 블록 ≈35개(§3)
  cv.py          # half_year_folds(purge 21d 양측, 양성 run 통째 배정), gap_augment(음성 구간에만 절단), day-block bootstrap
  models.py      # LightGBM binary row-head + LightGBM 6-class type-head(+선택 XGBoost 2번째 계열), deterministic 파라미터, 3-seed
  decode.py      # 하드 flatline → hysteresis/min-run → edge0 제거 → 유형별 구간 완성(§4) → (옵션) 계열 예산 소프트 완화
  train.py       # CLI: 특징 캐시 → CV(OOF, 파라미터 적합) → fitted_params.json → 전량 재적합 → weights/
  predict.py     # CLI: weights+fitted_params → test 특징 → decode → CSV(+sha256, validator)
  report.py      # CV 리포트(fold/station/layer/type/월별 recall·precision, 예측률, bootstrap CI)
  audit_constants.py  # decode/features의 수치 리터럴 목록화 + fitted_params 출처 대조(재현검증 1항 대비)
configs/ocean_v2/p1.json     # 창 크기·fold 날짜·모델 고정 하이퍼·seed·thread. 임계값 없음
artifacts/ocean_v2/p1/<run>/ # features_cache.parquet(sha), oof.parquet, fitted_params.json, weights/*.txt, cv_report.{json,md}
submissions/claude_v2/p1/<cand>/  # CSV + sha + cv 요약 + validator 결과
```

### 재사용(복사·단순화) 대상
- `src/p1_qc/data.py`: `segment_timeseries`, `parse_anomaly_types`, `sha256_file`, `load_dataset`(mtime/sha 감사) → `p1/data.py`
- `src/p1_qc/features.py` `build_features`(offline, 80열, gap-aware rolling·peer·long-window) → `p1/features.py` 기저(depth_raw/nominal_depth_m/depth_regime/depth_diff 제거: G-ORS test depth 전결측 + 판별력 없음(\|Δdepth\| 자연 0.062 = offset 진입 0.066))
- `src/p1_qc/postprocess.py`: `hysteresis_threshold`, `close_short_gaps`, `remove_short_runs`, `segments_from_mask` → `p1/decode.py`
- `src/p1_qc/rules.py`: `plateau_runs` → `p1/structure.py`(flatline 하드 룰)
- `src/p1_qc/metrics.py`: `binary_counts`, `group_report`, `event_report`, `anomaly_type_recall` → `p1/report.py`
- `src/p1_qc/splits.py`: `_positive_run_ids`(양성 run 통째 배정) → `p1/cv.py`(반기 fold·양측 purge로 재작성)
- `src/p1_qc/models_tabular.py`: `lightgbm_parameters`/`xgboost_parameters`의 결정론 플래그 → `p1/models.py`
- `src/p1_qc/change_points.py`: prefix-sum mean/variance/slope gain 통계(§4 랭커 stretch에서만 참고)
- `scripts/final_submission_20260905/P1/*`, `scripts/build_official_final_submission_20260905.py`: 폴더 계약(01_data…07_source, TRAIN/PREDICT notebook, RUN_*.ps1, contract.json) 재사용. `router_anchor.csv`/`gi_spike2_patch.json`/MS-TCN 경로는 **폐기**.
- 참고 문서: 데이터 README(유형별 지속시간), `reports/P1_FAILURE_RECON_2026-08-13.md`(유형·계열별 실패 구조), `artifacts/runs/20260813T153038+0900_cv_378a4e89/selection.json`(O 레시피 후처리값 — 참고만, 재적합).

## 2. `structure.py` — 결정론 프리미티브 (정렬된 계열 배열 위, 세그먼트 내부에서만)
1. `exact_segments`: (station,layer) 내 정확 10분 run → `seg_id, seg_len, pos_in_seg, edge_dist=min(pos, len-1-pos)`.
2. `identical_run_len`: 양방향 동일값 run 길이(float 정확 일치). flatline 하드 룰 `run ≥ flat_min_run`.
3. `temp_only_jump`: `|Δtemp| ≥ τ_j` ∧ (`|Δpsal| < τ_p` ∨ psal 결측) → 부호 있는 점프 배열; 각 행에 대해 `dist_prev_tjump, dist_next_tjump, prev_tjump_val, next_tjump_val`(세그먼트 내부 한정).
4. `whiteness`: 창 w의 Δtemp lag-1 자기상관(백색잡음 차분 → −0.5), `dstd_w`(Δtemp 표준편차), `pstd_w`(Δpsal 표준편차), `dpeer_std_w`.
5. `derive_structural_constants(train)` → fitted_params 초안(출처 문자열 동봉):
   - `flat_min_run` = 자연(label 0) 동일값 run 최대 길이 + 1 (= 6)
   - `tau_jump` = offset 진입 \|Δtemp\|의 5% 분위(≈0.66)와 drift 이탈 5% 분위(≈1.24) 중 작은 값의 내림(0.5 격자) → 0.5; `tau_psal_quiet` = 주입 진입행 \|Δpsal\|의 75% 분위(≈0.02~0.03)
   - `pair_sum_kappa` = offset \|entry+exit\|/max(\|entry\|,\|exit\|)의 90% 분위; `dur_bounds[type]` = 이벤트 길이의 [min·0.8, max·1.2]
   - `series_pos_rate_prior` = 계열별 양성률 중앙값(0.041), `rate_floor` = 최소(0.0399)
   - 모두 JSON에 값+유도식 기록(리터럴 아님).

## 3. `features.py` — 특징 (총 ≈115열, float32, 라벨 미사용, 오프라인 양방향)
A. 기존 80열 중 depth 파생 4열 제외(`temp_raw, psal_raw, psal_missing, has_gap_before, day/hour sin·cos, temp_diff_1/abs/acc, psal_diff_1/abs, temp_diff_next, spike_min_abs_diff, curvature, plateau_elapsed/full_length, rolling(3/6/12/24/72h) median-resid/abs/std/diff-std/robust-z, peer_count/available/mean/resid/abs/station_std, long(7d/14d) temp_long_resid/peer_detrended/abs/reference/abs/slope_1h`).
B. 물리·구조 블록(신규):
- 구조: `flat_run_len(log)`, `edge_dist(log)`, `seg_len(log)`, `pos_in_seg_frac`
- T–S 분리: `ts_jump_ratio_1 = |Δtemp|/(|Δpsal|+ε_p)`(ε_p = 자연 소점프의 \|Δpsal\| 중앙 0.005, 데이터 유도), `ts_jump_ratio_next`, `psal_quiet_jump_prev/next`(0/1), `pstd_{6,12,36}`, `rough_ratio_w = dstd_w/(pstd_w+ε_p)`, `ts_local_resid_72h`(72h 창 robust T–S 회귀 잔차; 14d는 S7대로 약함이라 짧은 창만)
- 백색성: `d1_ac1_{12,36}`, `dstd_{6,12,36}`, `dpeer_std_{6,12}`
- 점프 앵커: `dist_prev_tjump, dist_next_tjump, prev_tjump_val, next_tjump_val, pair_sum=prev+next, pair_span=dist_prev+dist_next, bracket_level_shift`(prev~next 점프 사이 temp−14d median의 중앙값 − 양측 플랭크 72행 중앙값), `bracket_shift_sign_consistent`
- ramp: 72/144행 창 rolling 선형회귀 `slope, r2` (drift 증거), `slope_x_dist_next_tjump`(외삽 총상승 ≈ −next_tjump_val 정합)
- 인접층: `near_layer_resid`(layer±1 평균과의 차), `near_layer_resid_detrended_7d`
- 계절: `month`(정수), `station`(카테고리), `layer_category`
C. 캐시: train+test 각 1회(≈3~6분) parquet + sha. gap 증강 fold는 holdout 부분만 재계산(≈1분/fold).

## 4. 모델·디코더
**row-head**: LightGBM binary, `num_leaves 63, lr 0.03, feature_fraction 0.8, bagging 0.8/freq 1, min_child_samples 50, lambda_l2 5, max_bin 255, deterministic=True, force_row_wise=True, num_threads=8(고정 명시), seeds {20260905, 20260917, 20260929}`; 반복수는 fold 내 early-stopping(OOF logloss)의 중앙값을 fitted_params에 고정. 양성 가중 `scale_pos_weight`는 {1, 3} 중 CV. (옵션 2번째 계열: XGBoost CPU hist depth 7 lr 0.04 700it — 0813 O 레시피, 확률 평균; CV로 채택 여부)
**type-head**: LightGBM multiclass 6클래스(normal/spike/noise/flatline/offset/drift; 복합은 첫 토큰), 같은 특징·seed. 디코더의 완성 전략 선택에만 사용.
**decode(순서 고정)**:
1. `hard_flat = identical_run ≥ flat_min_run` → 1 (내부 1~2행 구멍 채움)
2. `hysteresis(p_row, high, low)` 세그먼트 내부; `min_run`(CV) 미만 제거. 단일행 spike는 `p_row ≥ spike_thr`(CV) ∧ 구조형(양측 부호반대 점프, 복귀 ≤35%, psal 정숙, 국소 스케일비 ≥ r_s)일 때만 보존 — CV에서 spike 포함/제외 자체를 선택.
3. `edge_dist == 0` → 0 (train 근거 S2; 3행 마진은 CV 옵션이되 test flat run 1건이 거리 1이므로 기본은 0행만)
4. **구간 완성**(양성 run R마다, type = argmax 평균 p_type):
   - offset: R.start에서 후방 ≤ dur_max 내 가장 가까운 temp-only 점프 J_in(부호 s), R.end에서 전방 가장 가까운 temp-only 점프 J_out(부호 −s). 채택 조건: span∈dur_bounds[offset], `|Δ_in+Δ_out| ≤ κ·max(|Δ_in|,|Δ_out|)`, bracket_level_shift 부호 = s 이고 \|shift\| ≥ τ_c(offset 레벨 이동 10% 분위, 데이터 유도), 내부 dstd 자연 수준. 채택 시 R=[J_in, J_out). 실패 시 R 유지.
   - drift: R.end 전방 가장 가까운 temp-only 점프 J_out(\|Δ\| ≥ τ_j). (temp − 인접층/자기 14d 기준) 잔차를 [t, J_out) 구간에 선형 적합하며 t를 후방 확장; R² ≥ ρ_min 유지되고 적합선의 플랭크 교차점을 시작으로. 정합 검사 `|slope·(J_out−start) + Δ_out| ≤ κ_d·|Δ_out|`. 길이 dur_bounds[drift]. 실패 시 R 유지.
   - noise: 양방향으로 `dstd_6 ≥ τ_n ∧ (pstd_6 < τ_p ∨ psal 결측)`가 지속되는 동안 확장(≤g행 끊김 브리지), 내부 구멍 전부 채움. τ_n = noise 행 dstd_6의 5% 분위(≈0.42).
   - 완성은 유형별 on/off 스위치 → CV에서 유형별로 채택.
5. (옵션 C4) 계열 예산 소프트 완화: 계열 예측률 < `rate_floor·0.6`이면 그 계열만 low 임계값을 단계적으로 낮추되 상한 `rate_floor·0.75`(=3%)까지. CV(H1_2025의 4.76% 예산 불일치가 스트레스 테스트)에서 채택 여부.
모든 임계값(high/low/min_run/spike_thr/r_s/τ_*/κ/ρ_min/g/on-off)은 `fitted_params.json`에 값+선택 근거(fold별 F1 표) 기록. LB 값은 어떤 단계에도 입력되지 않음.

## 5. 정직한 CV 설계 (`cv.py`)
- **블록**: KST 반기 4 fold — `H1_2024`(01-01~06-30, S-ORS 7층), `H2_2024`(07-01~12-31, S-ORS), `H1_2025`(01-01~06-30, S·I·G 전부), `H2_2025`(07-01~12-31 전부). train = 나머지 전부에서 `[val_start−21d, val_end+21d]` 제외(양측 purge 21일 > 오프라인 특징 지원 337h). 오프라인 QC이므로 미래→과거 학습 허용.
- **앵커 규칙**: 양성 run은 시작 시점이 속한 fold에 통째 배정(경계 이벤트 분할 금지, `_positive_run_ids` 재사용). 특징은 전체 train에 1회 계산(라벨 무관, purge로 창 겹침 차단).
- **test 정합 증강 fold**: holdout 복제본의 **음성 구간에만** 무작위 절단 삽입(행당 hazard = test 0.55% − train 0.16% ≈ 0.4%, gap 길이는 test gap 분포(중앙 60분, p90 950분)에서 표집, seed 고정) → 특징 재계산 → 장기창 결측 30~45% 상황에서 재평가. S2·test flat run 증거상 이벤트는 절단하지 않음. 학습 증강(C2+aug)에도 같은 함수로 train 1복제 추가.
- **지표**: 1차 = Jan–Jun 두 fold(H1_2024+H1_2025, 316,206행·14,121양성·68 장기이벤트) 풀링 F1; 2차 = 4-fold 풀링 F1 및 증강 fold F1; 필수 진단 = 유형별 recall(spike/noise/flatline/offset/drift), 계열별 예측률(목표 ≈4%), 월별, 경계행 양성 수, 이벤트 IoU. KST 일 블록 bootstrap 2,000회(seed 고정) CI90.
- **승격 규칙**(사전 고정): 후보 채택 = 1차 ΔF1 > 0 ∧ bootstrap P(Δ>0) ≥ 0.8 ∧ 정점별(G/I/S) 회귀 ≤ 0.02 ∧ 증강 fold에서 부호 동일. 동률·미세차는 단순한 쪽.
- Public 점수는 후보당 1회 "sanity"만: 후보의 CV F1과 Public F1 차가 0.05 초과면 버그 조사(파라미터 조정 금지).

## 6. 후보 사다리 (기대 이득은 정직 추정, ?=불확실)
| 후보 | 내용 | 기대 Private F1 | 근거 |
|---|---|---|---|
| **C0 SAFE** | 기존 80특징(−depth) + LightGBM 3-seed + hysteresis/min-run(CV) + 하드 flatline + edge0 | 0.79~0.83 | O/B 단독 Public 0.79; 하드 flatline·edge0·정직 임계값으로 소폭 상회. **완전 재생성·결정론** |
| C1 | + 물리 블록(T–S 분리·백색성·점프 앵커·구조) (+XGB 2계열 옵션) | C0 +0.01~0.03 | S3/S4: 최대 FP원(S-ORS 자연 점프)와 noise 판별을 직접 겨냥 |
| **C2** | + 유형별 구간 완성(offset 점프쌍·drift 역투영·noise 확장) | C1 +0.02~0.05 ? | FN의 98.6%가 부분탐지 이벤트 내부; S5/S6가 정확 경계 제공. 위험: S-ORS 오점프 스냅 → psal 정숙·부호·길이 정합으로 제한 |
| C2+aug | + gap 증강 학습 | +0.00~0.01 ? | test 단절 2~4배; S-ORS/G-ORS 예측률 회복 |
| C4 | + 계열 예산 소프트 완화 | +0.00~0.01 ? (하방 있음) | S8 사전정보; H1 예산 3.9~4.8% 변동이라 보수적 상한 |
| C3(stretch) | 점프쌍/이탈점프 기반 후보구간 생성 + GBDT 구간 랭커 | +0.01~0.03 ? | 시간 남을 때만 |
상한 산정(유형별 recall/precision 가정 flat 0.98/1.0, noise 0.95/0.9, offset 0.87/0.85, drift 0.85/0.85, spike 0): F1 ≈ 0.92. 현실 목표 0.85~0.88. (1위 F1≈0.96은 이 구조가 실제로 착취 가능함을 시사.)

## 7. 결정론·런타임(재현 6h 한도)
- CPU 전용, LightGBM `deterministic=True, force_row_wise=True, num_threads=8` 명시, seed 3개 고정, numpy RNG seed(증강), mergesort 정렬. GPU/딥모델 없음(MS-TCN 폐기: 비결정·자체 gate 실패·+333행).
- 예상 시간(7800X3D): 특징 캐시 train+test ≈ 5분; CV 4 fold × 3 seed × 2 head ≈ 40~70분(+증강 fold 재계산 ≈ 5분); 최종 재적합 3 seed × 2 head ≈ 10~15분; 예측 ≈ 2분. **총 ≈ 1~1.5시간**(XGB 계열 추가 시 +20분). 재현검증 4항(산출물 제거 후 재생성)에서 byte-exact 기대.
- 패키지: `03_model/weights/*.txt`(LightGBM 텍스트), `fitted_params.json`, `configs/ocean_v2/p1.json`, `07_source/src/ocean_v2`, RUN_TRAINING.ps1 / RUN_INFERENCE.ps1, README(환경·소요시간·seed·sha). `audit_constants.py` 결과 동봉.

## 8. 일정 (09-05 13:00 KST 시작, 업로드 하루 3회 = sanity 전용)
**Day 1 (09-05)**
- 13:00–15:00 `common/`, `data.py`, `structure.py`(+상수 유도), `features.py`(기존 복사+물리 블록), `cv.py`, 특징 캐시 생성.
- 15:00–17:00 `models.py`, `train.py` → **C0** CV(파라미터 적합) → `predict.py` → 검증기 → `submissions/claude_v2/p1/C0/`.
- 17:00–19:00 **C1** CV(물리 블록 on) 비교·리포트.
- 19:00 업로드 ①C0(재생성 기준선 sanity, 기대 0.79~0.83) ②C1(승격 규칙 통과 시).
- 19:00–23:00 `decode.py` 구간 완성(offset/drift/noise) 구현, 유형별 on/off CV 실행(≈1h).
**Day 2 (09-06)**
- 09:00–12:00 **C2** 결과 검토(유형별 recall, 경계 IoU, 계열 예측률), 필요 시 정합 조건 보수화 1회(사전 고정 격자만) → 업로드 ③C2.
- 12:00–16:00 C2+aug(증강 학습), C4(예산 완화) CV → 승격 규칙 → 업로드 ④최선.
- 16:00–19:00 클린룸 재현: 새 임시 폴더에 패키지만 복사 → raw 지정 → train → predict → CSV sha 대조·소요시간 기록. `audit_constants.py`.
- 19:00–22:00 최종 패키지 빌더(기존 계약 재사용) 생성, README/contract.json, 업로드 파일 ≤50MB 분할.
**Day 3 (09-07 오전)** 최종 지정 후보 확정(CV 최고 ∧ Public sanity 정상) → 답안 업로드(미업로드 시) → 사용자가 모델 최종 제출. 여유 시 C3.

## 9. 폴백
- 1차 폴백 = **C0**(완전 재생성·결정론·규정 준수). 2차 = C1(승격 통과 시). 어떤 경우에도 `router_anchor.csv` 계보(현 챔피언)는 재현검증 탈락 위험으로 최종 지정하지 않음(이미 업로드된 점수는 유지되나 모델과 불일치).

## 10. 위험과 대응
1. 정직 CV F1이 과거 OOF(0.86)보다 낮게 나옴(누출 제거·계절 이동) → 기대치이며 상대 비교만 사용.
2. 완성 디코더가 32 offset/28 drift 통계에 과적합 → 분위 기반 보수 임계값, fold별·정점별 검증, 유형별 스위치, 증강 fold 부호 일치 요구.
3. S-ORS 자연 대점프(7.8%)에 오스냅 → psal 정숙·peer 정숙·부호 반대·pair_sum 정합·길이 제약 4중 조건; 실패 시 원 run 유지(하방 = C1).
4. test 예산이 4.1%가 아닐 수 있음(H1 부분율 3.9~4.8%) → C4는 소프트·상한 3%·옵션.
5. G-ORS: peer 없음·depth 결측·train 26,503행/15이벤트 → H1_2025 fold(16,930행)로 감시; depth 특징 제거로 학습-시험 조건 일치.
6. psal 결측(I-ORS L7 train 16.8%) → 모든 T–S 특징은 NaN 허용(LightGBM), 정숙 조건은 "결측=정숙 아님·미지"로 처리해 완성 조건에서 보수적으로.
7. 시간 초과 → C2까지가 필수, C4/C3는 선택. C0는 Day 1 오후에 반드시 완성.
8. 재현검증 1항(상수 리터럴) → 모든 임계값 fitted_params 경유, audit 스크립트로 증빙.

