# P3_SPEC — 유의파고 단기 예측 `src/ocean_v2/p3/` (설계 종합: design_panel/P3_design_*.md)

## 0. 문제와 목표
- 200개 익명 사례, 각 사례는 기준시각 이전 48h 문맥(10분 격자 289행: `step_minute=-2880..0`, 파랑열 hs/tp/hmax/wvdir은 20분 슬롯에만 값, 기상열 wspd/gust/wdir/airt/relh/caph는 10분). 예측 대상 hs(+3/6/9/12/18/24h), 지표 pooled RMSE(m), 제출 1,200행.
- 사례 선정(README): 기준시각 hs≥1.5, 6리드 정답 유효, 같은 정점 사례 간격 ≥78h(48+24+6). 정찰 확인: 사례는 **폭풍 발생 초기**에 편중(직전 3h 최소 hs<1.5 비율 0.77, 상승 85%, hs0 분위 1.52/1.55/1.64/1.86/2.39). 정시(:00) 격자에서 "first-eligible + 78h greedy"를 train에 적용하면 같은 구성이 재현된다.
- 목표: LB 상수 없이 완전 재생성되는 결정론 GBDT 앙상블. 안전 기준선 B1의 Public 기대 0.585~0.60(정직 O 0.607보다 좋고, 기존 α 후보 0.584와 동급). 사다리로 −0.005~−0.015 추가 기대.

## 1. 재사용(복사·단순화; 원본 수정 금지)
- `src/p3_wave/data.py`: `load_p3_data`, `audit_p3_data`, `build_training_grid`(정점별 10분 격자 left-merge), `build_anchor_table(grid, dense_spacing_minutes=20)`(hs≥1.5 ∧ 6리드 유효 ∧ 시작+48h → 24,360 anchor, G 9,893/I 7,312/S 7,155).
- `src/p3_wave/features.py::summarize_context`: **등가성 테스트의 기준 구현**으로만 사용(새 벡터화 구현과 공통 특징값 대조, 허용 1e-9). `wind_wave_alignment=cos(wdir−wvdir)`, `wind_input_proxy=wspd²·max(align,0)`, `steepness=hs/tp²`, `hmax_hs_ratio`, `gust_excess` 정의 유지.
- `src/p3_wave/revin_patch.py::assign_storm_episodes_from_wave`(있으면) 또는 자체 구현: episode_id = 정점별로 "hs<1.5 또는 결측"이 6h(18슬롯) 연속되면 새 episode.
- `src/p3_wave/submission.py`, `validation.py::rmse`. 금지: `loss_router.py`, `persistence_shrink.py`, `corrected_fixed_long_shrink.py`, `kma_*`, `era5_*`, `chronos*`, residual cycle 스크립트.

## 2. 패키지 구조
```
src/ocean_v2/p3/__init__.py, __main__.py   # python -m ocean_v2.p3 {audit|features|cv|train|predict|all} --config ... --out ... [--data-dir]
config.py     # dataclass <- configs/ocean_v2/p3_<cand>.json. physical_constants: hs_threshold 1.5, spacing_h 78, context_h 48, buffer_h 6, leads [3,6,9,12,18,24], clip [0,30], grid_min 10, wave_slot_min 20
data.py       # 로드·감사·격자·anchor·episode·위상 플래그·test 사례 배열화 (N,289,10) float32
features.py   # window_features(X:(N,289,10), meta) -> DataFrame; train/test 완전 동일 함수
cv.py         # 블록·purge·episode 제외·평가면 가중·지표·부트스트랩·게이트
model.py      # 리드별 잔차 GBDT 래퍼(LGBM 주, CatBoost/XGB 보조), 바람-드롭아웃 증강, seed 앙상블
calibrate.py  # OOF 리드별 a_L(승법) 산출 -> fitted_params (기본 후보는 항등 적용)
train.py / predict.py / report.py
```

## 3. 데이터 → anchor → 배열
1. `load_p3_data(data_dir)` → 6파일 SHA manifest. `build_training_grid` → 정점별 10분 격자(파랑 20분 슬롯 외 NaN 유지).
2. `build_anchor_table(dense_spacing_minutes=20)` → 24,360 anchor. 타깃 `y_L = hs(t+L) − hs0` (리드별 잔차). 추가 열:
   - `is_hourly = (minute==0)`, `hs_prev20`(직전 파랑 슬롯), `first_cross = not(hs_prev20≥1.5)`, `hs_min_3h/6h/12h/24h`(NaN 무시), `jc3 = hs_min_3h<1.5`, `run_above_1p5_h`, `rising_1h/3h`, `episode_id`, `block`(anchor_time UTC 분기: 2024Q1..2025Q2).
3. 각 anchor를 격자 슬라이스 `grid[pos−288 : pos+1]`로 (289,10) 배열화(열 순서: hs,tp,hmax,wvdir,wspd,gust,wdir,airt,relh,caph). test_context는 `case_id`별 `step_minute` 정렬로 같은 배열 구조. 메모리 ≈ 24,360×289×10×4B ≈ 280 MB.
4. 등가성 테스트(필수): 임의 anchor 30개를 test_context 형식으로 변환해 특징을 다시 계산하면 bit-동일(`np.array_equal`); `summarize_context`와 공통 통계(mean/std/min/max/slope) 값 일치.

## 4. 특징 `window_features` (절대시각·달력 특징 없음, 사례 내부만)
1. 기본 시계열 12: hs, tp, hmax, wspd, gust, caph, airt, relh, wvdir_sin/cos, wdir_sin/cos. 파생 6: energy=hs², steep=hs/tp², hmax_r=hmax/hs, wsq=wspd², align=cos(wdir−wvdir), wind_in=wsq·max(align,0).
2. 각 시계열: `current`(마지막 유한값), `age_h`(마지막 유한값까지 경과 h), lag 1/3/6/12/24h(정확 슬롯; 결측이면 NaN), 창 3/6/12/24/48h의 mean/max/min/std/slope(시간 기준 OLS), valid 비율(6h/24h).
3. 변화량: hs·wspd·gust·tp·caph의 `current − lag(1/3/6/12/24h)`.
4. **onset 위상 블록**(핵심): `run_above_1p5_h`, `hs_prev20`, `first_cross`, `jc3`, `hs_min_3/6/12/24h`, `hs_max_12/24/48h`, `hours_since_min_24h/48h`, `rise_from_min_24h`, `hours_since_max_24h/48h`, `drop_from_max_24h/48h`, `crossings_1p5_48h`, `hs_accel_3h`(최근 3h slope − 그 이전 3h slope), `hs0_over_mean24h`, `hs0_over_max48h`, `tp_change_3h/6h`, `wave_age_proxy = 1.56·tp0/max(wspd0,0.5)`, `wind_rotation_3h/6h`(각도차), `caph_min_24h`, `hours_since_caph_min_24h`, `n_valid_hs_3h`.
5. **물리 파라메트릭 블록**(사다리 B3, 설정으로 on/off): 정점별 `c_s` = train 격자에서 `hs ~ wspd_mean3h²`의 90% 분위 회귀 계수(fitted_params에 저장) → `hs_eq = c_s·wspd_mean3h²`, `growth_room = hs_eq − hs0`, `hs_ratio = hs0/hs_eq`, gust 버전, `hs_eq_trend`; `caph_slope_12h`.
6. 정점: 정수 코드(범주). 리드는 모델을 분리하므로 특징 아님. 총 ≈180~220열, NaN 유지. float32.
7. 특징 캐시: `features_cache/train_features.parquet`(anchor 메타 포함), `test_features.parquet`, 키 = (6파일 SHA, FEATURE_VERSION).

## 5. 정직한 CV (`cv.py`, 사전 등록·코드 고정)
- **블록 6개**: anchor_time UTC 분기 2024Q1, 2024Q2, 2024Q3, 2024Q4, 2025Q1, 2025Q2.
- **fold k**: 검증 = 블록 k의 모든 anchor(OOF는 24,360 전부에 존재). 학습 = `anchor_time < block_start − 78h or > block_end + 78h` 이고 `(station, episode_id)`가 검증에 없는 anchor. 검증 발자국 [−48h, +24h]과 학습 발자국 무겹침을 코드에서 assert.
- **평가면(가중치로 정의, 모두 리포트)**:
  - `S_onset`: `jc3` anchor 전부(≈5,600) 균등.
  - `S_storm`: jc3가 아닌 anchor 중 정점별 6h 간격 greedy(≈1,250) 균등.
  - **`S_dense`(1차 지표)**: `p·S_onset + (1−p)·S_storm`, p = **train 정시 anchor에 운영진 규칙(정점별 first-eligible + 78h greedy)을 복제했을 때 jc3 비율**(코드 계산, ≈0.73~0.77). test_context의 jc3 비율(≈0.765)은 진단으로만 출력·비교.
  - `S_greedy`: 정시 anchor 정점-전역 greedy 78h(≈272)의 비가중 RMSE(운영진 규칙 직접 모사, 표본 작음, 2차 지표).
  - 보고: pooled, 리드별, 정점별, 블록별, 정점×블록, 편향(truth−pred) 리드별, 동일 anchor persistence.
- **불확실성**: (station, episode) 블록 부트스트랩 1,000회(seed 고정) → 후보 간 Δ의 CI90·P(개선).
- **승격 게이트**: `ΔS_dense < 0 and P(개선) ≥ 0.90 and 6블록 중 ≥4 개선 and 정점별 악화 ≤ +0.010 and 18/24h 악화 ≤ +0.005 and ΔS_greedy ≤ +0.005`. 미통과 변형은 후보 아님. 총 후보 평가 ≤ 12회(격자 4 + 특징군 4 + 학습기/seed 4). |Δ| < 0.003은 동률 → 단순한 쪽.
- **기존 계보 재평가**(배경 1회): `artifacts/p3/features_all20_v1` 캐시 + compact 591 + CatBoost single(CPU 700 iter)을 같은 6-fold에 돌려 S_dense에서 비교·기록(새 파이프라인이 열세면 591 특징 세트를 흡수).

## 6. 모델 (`model.py`)
- **주 학습기 M1**: LightGBM 리드별 6모델, 목표 잔차. 초기 하이퍼(프로브 검증값): `objective=regression, n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=10` + `determinism.lgbm_params`. seed = `base + 100·lead_idx + k`, CV 3 seed / 최종 5 seed. HPO 격자(사전 등록 4개): (leaves 15/31) × (trees 600 lr 0.03 / trees 1000 lr 0.02).
- **보조 M2(사다리 B2)**: CatBoost CPU 리드별 RMSE, `iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=8, random_strength=0.2`. 앙상블 = 리드별 단순 평균(0.5/0.5 사전 고정).
- **선택 M3**: XGBoost per-lead `max_depth=4, n_estimators=700, lr=0.025, min_child_weight=20, subsample=0.85, colsample_bytree=0.55, reg_lambda=8, reg_alpha=1` (3멤버 등가평균 옵션).
- **학습 표본**: 24,360 anchor 전부, **가중 없음**(선정-정합/수준 가중은 CV에서 악화·무효로 측정됨). 사다리 L1에서만 3안 비교(균일 / exp(−0.45·max(hs0−1.5,0)) / 밀도비).
- **바람-드롭아웃 증강(기본 포함)**: wspd가 존재하는 학습행의 복제본에서 기상 유래 특징(wspd·gust·wdir·caph·airt·relh 및 wsq/align/wind_in/hs_eq 계열의 current/age/lag/창/변화량)을 NaN으로 두고 sample_weight w_aug 추가. w_aug ∈ {0.3, 0.5, 1.0}을 CV에서 선택해 fitted_params에 기록. 근거: I/S-ORS는 2024 기상 전무라 원모델이 "기상 NaN=2024 I/S 폭풍"을 학습(25Q2 fold에서 persistence보다 악화); test는 기상 97% 존재.
- **후처리**: `pred = hs0 + mean_seeds f_L(x)` → clip [0,30] → `%.4f`. persistence shrink·라우터·α 없음.
- **보정 `calibrate.py`**: OOF(S_dense 가중)에서 리드별 `a_L = Σw·d·r/Σw·d²`(d=pred−hs0, r=truth−hs0)를 계산해 fitted_params에 기록하되 **기본 후보는 a_L≡1**. 적용 후보 `P3_v2_calib`는 사다리 마지막(조건: CI90이 1을 배제 and 0.7≤a_L≤1.3). 가법 편향 보정은 금지(b_L=0).

## 7. 후보 사다리
| # | 후보 | 내용 | 비고 |
|---|---|---|---|
| F0 | `P3_v2_floor` | persistence(3/6/9/12h) + 정점×리드 onset 기후 평균 Δ(18/24h; fitted_params) | 최후 폴백(코드 수 줄) |
| **B1** | **`P3_v2_base`** | M1 3-seed + onset 특징 + 바람-드롭아웃 증강, a_L=1 | 안전 기준선, 업로드 #1 |
| B2 | `P3_v2_ens` | + CatBoost 멤버 0.5/0.5 (+옵션 XGB 등가) | 게이트 통과 시 |
| B3 | `P3_v2_phys` | + 물리 파라메트릭 블록 | 게이트 통과 시 |
| B4 | `P3_v2_seed5_hpo` | seed 5 + HPO 격자 최적 | Day 2 |
| L1 | 표본가중 3안 | 균일/기존 지수/밀도비 | 참고 |
| C | `P3_v2_calib` | 최종 + OOF a_L | 조건부, 마지막 |
최종 지정 = 게이트 통과한 가장 높은 번호. Public은 버그 검출(0.63 초과·persistence 근접)과 방향 확인용으로만.

## 8. 산출물·결정론·런타임
- `models/lgb_L{lead}_s{k}.txt`, `cb_L{lead}.cbm`, `fitted_params.json`(c_s, w_aug, p, a_L, 특징 목록·해시, 선택 근거 표), `cv/cv_report.{json,md}`, `TRAINING_RECEIPT.json`.
- 예상 시간(8코어): 특징 1~2분, LGBM CV 108 fit ≈ 6~8분, CatBoost CV 36 fit ≈ 25~30분, 최종 학습 ≈ 12~15분, 예측 <10 s → **≤1h**.
- 결정론: `predict` 2회 실행 SHA 동일; `train` 재실행 bit-동일(같은 머신). README에 타 머신 허용오차 |Δ|≤1e−4 m 명시.

## 9. 테스트 `tests/ocean_v2/test_p3.py`
- anchor 수 24,360, 정시 anchor 수, greedy 복제 사례 수·jc3 비율 재현(0.7±0.05), purge/episode 무겹침 assert, 특징 등가성(train 슬라이스 vs test 형식), 미래 행 미사용(anchor 이후 격자 값을 변조해도 특징 불변), validator 통과, 결정론(소규모 설정 2회 학습 SHA 동일).

## 10. 하지 말 것
persistence shrink 상수, 라우터, LB α, TabPFN, 외부자료, 분위수 블렌딩, 시퀀스 딥모델, 절대시각·월 특징, 사례 간 결합, GPU.
