# P1_SPEC — 관측 수온 자동 이상 탐지 `src/ocean_v2/p1/` (설계 종합: design_panel/P1_design_*.md)

## 0. 문제와 목표
- train 776,706행(2024~2025, station/year/layer/time/temp/psal/depth + label/anomaly_type), test 169,011행(2026-01~06, 라벨 없음). 행 단위 binary F1. 제출 `station,year,layer,time,label`.
- 주입 유형(README): spike 10분, noise 3~58.8h, flatline 2~47.2h, offset 8~86.5h, drift 9~86.5h; 유효 관측의 ≈4%(계열별 3.99~4.19% 균일), 일부 중첩.
- 정찰 사실: 주입은 **temp만** 바꾼다(psal/depth 불변). 자연 연속 동일값 run 최대 4~5행 vs flatline ≥12행. offset = 대칭 점프쌍(|entry|,|exit|≥0.5℃ 96.9%, 부호 반대 96.9%), drift = 무음 진입 + 큰 이탈 점프(≥0.8℃ 96.4%), noise = dT std 2.98 vs dS 0.033. 이벤트는 세그먼트 경계 3행 이내 0건. 손실 본체 = offset/drift 이벤트 **내부 행**(FN 92.7%; 유형 recall 0.65). test는 겨울~초여름(5~6월 55%), 세그먼트 단절이 train보다 3.4배(장기창 결측 18~32%), G-ORS depth 전결측, I-ORS L3 부재. 5~9월 S-ORS 중층 자연 |dT| p99 4~5℃(FP 위험).
- 목표: CPU 결정론 GBDT + 구조 탐지·구간 완성. 안전 기준선 C0 Public 기대 0.79~0.83, 사다리 완료 시 0.85~0.88(리더 0.96은 구조 착취가 가능함을 시사).

## 1. 재사용(복사·단순화; 원본 수정 금지)
- `src/p1_qc/data.py`: `load_dataset`, `segment_timeseries`, `parse_anomaly_types`, `KEY_COLUMNS`. `src/p1_qc/features.py::build_features(mode="offline")`: 80열 중 depth 계열 4열(`depth_raw, nominal_depth_m, depth_regime, depth_diff_1/abs`) 제외 → 76열 기저. `src/p1_qc/postprocess.py`: `close_short_gaps, remove_short_runs, segments_from_mask`(히스테리시스는 벡터화 재작성). `src/p1_qc/rules.py::plateau_runs`. `src/p1_qc/metrics.py`: `binary_counts, group_report, event_report, anomaly_type_recall`. `src/p1_qc/validation.py`: `paired_block_bootstrap, normal_station_layer_day_fp`. `src/p1_qc/splits.py::_positive_run_ids`(양성 run 통째 배정). `src/p1_qc/models_tabular.py`의 결정론 파라미터 골격. 참고: `artifacts/runs/20260813T153038+0900_cv_378a4e89/selection.json`(O 레시피; 값은 재적합).
- 금지: `router_anchor.csv`, `gi_spike2_patch.json`, MS-TCN(`ms_tcn_asrf*`, `run_p1_*mstcn*`), 과거 제출 CSV, 외부 I-ORS 자료 모듈.

## 2. 패키지 구조
```
src/ocean_v2/p1/__init__.py, __main__.py   # python -m ocean_v2.p1 {audit|features|cv|train|predict|all}
config.py      # physical_constants: grid_min 10, type durations(README), edge_margin_rows 0(옵션 3), seeds, threads, block dates, purge_days 21
data.py        # 로드(sha), canonical order (station,layer,time)+원위치 매핑, 정확 10분 세그먼트, 이벤트 테이블(events_from_labels)
structure.py   # 결정론 프리미티브: 세그먼트/edge_dist, identical_run_len, temp_only_jump, whiteness, derive_structural_constants(train)->fitted_params 초안
features.py    # 76열 기저(복사) + 시간창(B) + psal 쌍둥이(C) + 단계/브래킷(D) + 노이즈(E) + 단절·밀도(G) + 인접층 + 달력
context.py     # Stage-2 문맥 특징(Stage-1 OOF 확률 집계) (사다리 C5)
cv.py          # 반기 4블록, 양측 purge 21일, 양성 run 통째 배정, gap 증강(음성 구간만), 일-블록 부트스트랩, 게이트
models.py      # LightGBM binary row-head x3 seed + XGBoost(O 레시피) x2 seed; LightGBM 6-class type-head; 결정론
decode.py      # 하드 flatline -> hysteresis/min-run -> edge0 -> 유형별 구간 완성(C4) -> (옵션) 계열 예산 소프트 완화(C6)
calibrate.py   # OOF 후처리 격자 선택 + 중첩-정직 추정, 유형·계열·월별 진단
train.py / predict.py / report.py / audit hook
```

## 3. `structure.py` 프리미티브 (정렬된 계열 배열, 세그먼트 내부)
1. `exact_segments`: (station,layer) 내 정확 10분 run → `seg_id, seg_len, pos_in_seg, edge_dist=min(pos, len−1−pos)`.
2. `identical_run_len`: 양방향 동일값 run 길이(float 정확 일치).
3. `temp_only_jump`: `|Δtemp| ≥ τ_j and (|Δpsal| < τ_p or psal 결측)` → 부호 있는 점프 배열; 각 행의 `dist_prev_tjump, dist_next_tjump, prev_tjump_val, next_tjump_val`(세그먼트 내 ffill/bfill, O(n)).
4. `whiteness`: 창 w의 Δtemp lag-1 자기상관, `dstd_w`, `pstd_w`(Δpsal std), `dpeer_std_w`.
5. `derive_structural_constants(train)` → fitted_params 초안(값 + 유도식 문자열):
   - `flat_min_run` = 자연(label 0) 동일값 run 최대 길이 + 1 (기대 6).
   - `tau_jump` = min(offset 진입 |Δtemp| 5% 분위, drift 이탈 |Δtemp| 5% 분위)를 0.5 격자로 내림(기대 0.5); `tau_psal_quiet` = 주입 진입행 |Δpsal|의 75% 분위(기대 0.02~0.03).
   - `pair_sum_kappa` = offset |entry+exit|/max(|entry|,|exit|)의 90% 분위; `dur_bounds[type]` = 이벤트 길이 [min·0.8, max·1.2]; `tau_noise` = noise 행 dstd_6의 5% 분위; `Z_BIG/A_BIG`(§4-D).
   - `series_pos_rate_prior` = 계열별 양성률 중앙값(≈0.041), `rate_floor` = 최소(≈0.0399).
   - 모두 fold train 전용으로 계산(CV) → 최종은 전체 train으로 재계산·저장.

## 4. 특징 `features.py` (float32, 라벨 미사용, 오프라인 양방향 허용)
- **A 기저 76열**(기존 offline 특징, depth 제외). 범주: station(3), layer(정수).
- **B 시간 기반 간격 허용 창**(핵심; pandas DatetimeIndex `rolling("168h", center=True, min_periods)`): 창 W∈{24h,72h,168h,336h}, min_obs {12,24,36,48}: `tb_med_W, tb_resid_W, tb_abs_resid_W, tb_mad_W, tb_z_W`; 비대칭 `tb_left_med_{24h,72h}`(과거 전용), `tb_right_med_{24h,72h}`(미래 전용), `tb_left_resid, tb_right_resid, tb_twosided_min_W=min(|left|,|right|), tb_lr_sign_agree`; `tb_diff_mad_168h`(국소 스케일) 및 `dT_local_z=|Δtemp|/(1.4826·tb_diff_mad_168h+1e-3)`.
- **C psal 쌍둥이**: `psal_tb_resid_168h, psal_tb_z_24h, psal_diff_roll_std_3h, psal_available`; `ts_slope_168h`(rolling cov/var, clip ±20), `ts_resid = tb_resid_168h − ts_slope·psal_tb_resid_168h`, `ts_abs_resid`; `ts_jump_ratio = |Δtemp|/(|Δpsal|+ε_p)`(ε_p = 자연 소점프 |Δpsal| 중앙값, fitted), `psal_quiet_jump_prev/next`, `pstd_{6,12,36}`, `rough_ratio_w = dstd_w/(pstd_w+ε_p)`, `dT_std3h_over_dS_std3h`.
- **D 단계/브래킷 변화점**: `step_k = median(temp[t..t+k−1]) − median(temp[t−k..t−1])`, k∈{3,6,18}; `abs_step_k`, `step_k_local_z`; "큰 단계" 마스크 = `|step_6_local_z| ≥ Z_BIG and |step_6| ≥ A_BIG`(Z_BIG/A_BIG = fold train 자연행 99.9 분위, A_BIG 상한 0.5℃로 clip — README 근거 기록); 좌/우 최근 큰 단계(≤120h) `left_step_val/hours, right_step_val/hours, has_bracket, bracket_closure=left+right, bracket_len_hours, bracket_level_resid`(L..R 중앙값 − 양측 플랭크 24h 중앙값 평균), `bracket_level_z`; drift형 `right_step_only, ramp_slope_72h_to_R, ramp_extrap_resid`; `resid_sign_matches_bracket`.
- **E 노이즈**: `zigzag_1h`(Δ부호 변화 비율), `diff_std3h_over_diff_mad168h`, `abs_dT_rank_168h`, `roll_range_1h`, `d1_ac1_{12,36}`.
- **F 달력·계열**: day/hour sin·cos, `month`(정수), station, layer.
- **G 단절·밀도**: `seg_len(log), pos_in_seg_frac, edge_dist(log), rows_to_gap_before/after, obs_count_±84h, obs_count_±12h, gap_before_minutes(log1p)`.
- **H peer**: 기존 5열 + `peer_resid_tb_detrended_168h`, `near_layer_resid`(layer±1), `near_layer_resid_detrended_7d`. G-ORS는 NaN.
- 캐시 parquet(입력 SHA + FEATURE_VERSION 키). 총 ≈150~190열.

## 5. 정직한 CV `cv.py`
- **블록(KST)**: H1_2024(01-01~06-30, S-ORS만), H2_2024, H1_2025(01-01~06-30, S·I·G), H2_2025(07-01~12-11). 오프라인 QC이므로 미래→과거 학습 허용.
- fold k: 검증 = 블록 k 행 + 시작 시각이 블록 k인 양성 run 전체(경계 run은 시작 블록 귀속). 학습 = 나머지에서 `[블록 시작−21일, 끝+21일]` 제외(양측 purge; 21일 > 특징 지원 337h). purge에 걸친 양성 run은 학습에서 통째 제외. 인코더·분위 상수는 fold train 전용.
- **단절 스트레스 표면**: 검증 블록 각 계열의 **음성 구간에만** test 간격 분포(행당 hazard ≈0.4%, 길이 분위(분) 60/120/950/4140/19477, seed 고정)로 인위 절단 → 특징 재계산 → 같은 fold 모델로 예측 → F1(`f1_stress`). 학습 증강(C5+aug)에도 같은 함수로 train 복제 1개 추가(가중 1).
- 지표(`oof_report.json` 필수): `f1_pooled`(4블록), **`f1_season`(H1_2024+H1_2025 풀링, 1차)**, `f1_worst_block`, 정점/층/월별 F1, 유형별 recall, 이벤트 recall·IoU, 계열별 예측률(진단; 4.1% 대비), 정상일 FP/일, 경계행 양성 수, KST-일 블록 부트스트랩 CI90(2,000회).
- **게이트(사전 등록)**: 채택 = Δf1_season > 0 and Δf1_worst ≥ −0.005 and CI90 하한 > −0.002 and Δf1_stress ≥ 0 and 정점별 회귀 ≤ 0.02. 실패 단계는 제외하고 다음 단계 진행(비누적). 하이퍼 격자 탐색 없음(고정값).
- Public은 후보당 1회 sanity: CV F1과 0.05 이상 차이면 버그 조사.

## 6. 모델 `models.py`
- **Stage-1 row-head**: LightGBM binary ×3 seed `num_leaves=63, learning_rate=0.03, n_estimators=900, min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5` + XGBoost ×2 seed(O 레시피 `max_depth=7, learning_rate=0.04, n_estimators=700, subsample=0.85, colsample_bytree=0.85, min_child_weight=20, reg_lambda=1`, hist/cpu). 샘플 가중 양성 `sqrt(neg/pos)`. 반복 수 고정(early stopping으로 검증 블록을 보지 않음; fold-4 내부 holdout(H1_2025 마지막 60일, purge 15일)에서 best_iter 1회 측정해 50% 미만이면 반값으로, 문서화). `p1` = 5모델 평균.
- **type-head**: LightGBM multiclass 6클래스(normal/spike/noise/flatline/offset/drift; 복합은 첫 토큰) 같은 특징·seed. 디코더 완성 전략 선택에만 사용.
- **Stage-2(C5)**: LightGBM ×3 `num_leaves=31, lr=0.05, n_estimators=400, min_child_samples=100, feature_fraction=0.8, lambda_l2=5`, 입력 = `context.py`(p1, 계열 내 백분위, `p1_max_±{1,6,24,72h}`, `p1_mean_±{6,24h}`, `frac_p1_ge_0.5_±24h`, `hi_left/right_hours`, `between_hi_72h`, `min(p1_max_left_24h, p1_max_right_24h)`, `p1_mean_L_to_R` + D 브래킷 + `tb_resid_168h, tb_twosided_min_72h, ts_resid, zigzag_1h`, station/layer/month, seg_len, plateau_full_length). 학습은 4-fold OOF p1 위 동일 fold; 최종 확인에서 fold별 내부 3-fold로 Stage-1 OOF를 다시 만드는 중첩 실행 1회로 낙관 편향 측정.

## 7. 디코더 `decode.py` (순서 고정; 모든 임계값은 fitted_params)
1. `hard_flat = identical_run ≥ flat_min_run` → 1(내부 1~2행 구멍 채움). 자연 FP≈0(train 검증).
2. `hysteresis(p, high, low)` 세그먼트 내부(벡터화: `cand=p≥low; run_id=cumsum(break or cand≠shift); seed=groupby(run_id).max(p≥high); label=cand&seed`), `bridge_gap_minutes`(≤120분 단절 잇기 옵션), `min_run` 미만 제거, singleton spike는 `p ≥ spike_thr and 구조형(양측 부호반대 점프, 복귀 ≤35%, psal 정숙)`일 때만 유지(포함/제외 자체를 CV).
3. `edge_dist == 0` → 0(옵션 margin 3행은 CV).
4. **유형별 구간 완성(C4)**: 양성 run R마다 type = argmax 평균 p_type:
   - offset: R.start 후방 ≤dur_max 내 가장 가까운 temp-only 점프 J_in(부호 s), R.end 전방 J_out(부호 −s). 채택 조건 span∈dur_bounds[offset] and `|Δ_in+Δ_out| ≤ κ·max(|Δ_in|,|Δ_out|)` and bracket_level 부호=s and |shift|≥τ_c and 내부 dstd 자연 수준 → R=[J_in, J_out). 실패 시 R 유지.
   - drift: R.end 전방 J_out(|Δ|≥τ_j); 잔차(temp − 인접층/자기 14d 기준)를 [t, J_out)에 선형 적합하며 t를 후방 확장, R² ≥ ρ_min 유지 시 플랭크 교차점을 시작으로; 정합 `|slope·(J_out−start)+Δ_out| ≤ κ_d·|Δ_out|`, 길이 dur_bounds[drift].
   - noise: `dstd_6 ≥ τ_n and (pstd_6 < τ_p or psal 결측)`이 지속되는 동안 양방향 확장(≤g행 끊김 브리지), 내부 구멍 채움.
   - 유형별 on/off 스위치 → CV에서 채택.
5. (옵션 C6) 계열 예산 소프트 완화: 계열 예측률 < `rate_floor·0.6`이면 그 계열만 low 임계값을 단계적으로 낮추되 상한 `rate_floor·0.75`.
- `anomaly_type` 열(선택): 하드 flat "flatline", singleton "spike", 완성 유형, 나머지 "" (채점 무관).
- `calibrate.py`: 격자 `high∈{0.25..0.70 step 0.05} × low_ratio∈{0.5,0.7,0.85} × close_gap∈{0,3,6,12} × min_run∈{3,6,12} × bridge∈{10,120} × singleton∈{off,0.8,0.9}`를 OOF 풀링 F1로 선택(≈10분, 벡터화), **중첩-정직 추정**(3블록 선택→1블록 적용, 4회 평균)을 병기.

## 8. 후보 사다리
| # | 후보 | 내용 | 기대(정직) |
|---|---|---|---|
| **C0** | `P1_v2_safe` | A 76열 + Stage-1 5모델 + 하드 flatline + edge0 + OOF 후처리 | Public 0.79~0.83 (완전 재생성·결정론) |
| C1 | `P1_v2_tb` | + B 시간창 + G 밀도 | +0.005~0.02 (test 장기창 결측 해소) |
| C2 | `P1_v2_ts` | + C psal 쌍둥이 | +0.005~0.015 |
| C3 | `P1_v2_bracket` | + D 브래킷 + E 노이즈 | +0.01~0.03 (가장 큰 레버) |
| C4 | `P1_v2_complete` | + 유형별 구간 완성 디코더 | +0.02~0.05 (불확실) |
| C5 | `P1_v2_stage2` | + Stage-2 문맥 스태킹(C4와 비교·결합) | +0.005~0.02 |
| C5+aug | 단절 증강 학습 | +0.00~0.01 |
| C6 | 계열 예산 완화 / singleton 정책 | ±0.01 (조건부) |
최종 = 게이트 통과 마지막 후보. 폴백 = C0(완전 재생성). 현 챔피언(router 계보)은 최종 지정하지 않음.

## 9. 산출물·결정론·런타임
- `models/*.txt|.json`(부스터), `fitted_params.json`(구조 상수·후처리 파라미터·완성 임계값 + 유도 근거 표), `cv/oof.parquet`(p1, p_type, fold), `cv/cv_report.{json,md}`, `TRAINING_RECEIPT.json`.
- 예상(8코어): 특징 캐시 train+test ≈ 10~15분(시간창 pandas rolling 그룹별), Stage-1 CV 4fold×5모델 ≈ 40~70분, type-head +15분, 후처리 격자 ≈ 10분, 스트레스 재계산 ≈ 5분, 최종 재적합 ≈ 15분, 예측 ≈ 3분 → **≈1.5~2h**(중첩 확인 +1.5h는 선택).
- 결정론: CPU 전용, seed·thread 고정, mergesort; `predict` 2회 SHA 동일.

## 10. 테스트 `tests/ocean_v2/test_p1.py`
purge 무겹침·양성 run 단일 fold 귀속; 합성 계열에 offset/drift 주입 → 브래킷 특징 반응·완성 디코더가 정확 경계 복원; 벡터화 히스테리시스 = 기존 루프 구현과 동일; 하드 flatline이 자연 run(≤5)에는 0; edge0 규칙; validator 통과(169,011행, 키 순서); 결정론.

## 11. 위험과 대응
5~6월 S-ORS 중층 자연 급변(국소 스케일 정규화·month·worst-block 게이트), G-ORS 검증 부족(H1_2025 72양성; depth 특징 제거로 학습-시험 조건 일치), 완성 디코더의 소표본 과적합(분위 기반 보수 임계값·유형별 스위치·증강 fold 부호 일치), psal 결측 층(NaN 허용, 정숙 조건은 "결측=미지"로 보수적), 시간 초과(C0는 Day 1에 반드시 완성; C4까지 필수, C5/C6 선택).
