# P3 v2 — 폭풍-발생(onset) 정합 검증면 + 바람-드롭아웃 증강 결정론 GBDT (규정 준수 재생성 파이프라인)

> 설계 패널(읽기 전용 설계자) 산출물, 2026-09-05. 저장소 파일·데이터는 수정하지 않았음.

## 기대 효과

[기준: 현 챔피언 ea65370a, Public 0.5839 m / 24.07점 — 그러나 α=−10.217은 LB 적합 상수라 재현검증 탈락(=0점 또는 실격) 가능성이 높음]

1) 안전 기준선 P3_v2_base(LGBM 3-seed, 바람-드롭아웃 증강, a_L=1, persistence shrink 없음):
   - Public 기대 ≈ 0.585~0.600 m (최선 추정 0.59). 근거: 기존 계보의 "shrink 제거·증폭 없음"(α=−4 등가)을 Public 이차식 q(α)=0.000264α²+0.0054α+0.3685로 환산하면 0.5925 m; 새 모델은 정직 CV에서 바람-드롭아웃 증강으로 −0.011 m, onset 특징 추가로 −0.00~−0.005 m 추가 개선. 즉 챔피언 대비 Public −0.005~+0.015 m(−0.08~+0.24점) 범위이지만 실격 위험은 0으로 내려감. 정직 O(0.607) 대비는 −0.010~−0.025 m(+0.16~+0.40점).
   - Private: B_private가 B_public보다 3.8% 나쁘므로 RMSE ≈ 0.60~0.62 m → 33.33−16.097·RMSE ≈ 23.4~23.7점. 챔피언이 살아남는 시나리오(≈23.6~23.9점, 단 부호 반전 시 −1.4점 꼬리)와 기대값이 비슷하고 하방이 훨씬 얕음. 불확실성 ±0.02 m(±0.3점) — 로컬 CV의 SE(모델 간 0.003 m)는 작지만 test 기간(2025-07 이후, 태풍기 I/S 자료 부재)로의 전이 오차가 지배함.

2) 사다리 증분(정직 CV onset-dense 기준, 부호·크기): 바람-드롭아웃 증강 −0.011 m(측정, 5/6 블록 개선), LGBM+CatBoost 2학습기 평균 −0.003~−0.006 m(추정, 미측정), 물리 특징(hs_eq=c_s·U², 성장여유) −0.000~−0.005 m(불확실), 계절/밀도비 가중 −0.000~−0.010 m(불확실, 검증면으로 판정 불가), OOF a_L shrink −0.009 m(측정)이나 Public 3점 증거(O/A/B: a 0.80→0.75에서 +0.0046 악화)와 반대라 기본형에서 제외. 선정-정합 72/28 가중은 +0.005 m(악화, SE 0.003)로 기각.

3) 종합: 규정 준수 후보의 Private 기대 점수 ≈ 23.5±0.3점. "챔피언 실격 확률"을 p라 하면 기대 이득 = p·23.5 − (1−p)·(0.2±0.5). p≥0.3만 되어도 교체가 우월.

## 재사용 대상

- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\data.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\submission.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\validation.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\features.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\event_phase.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\corrected_repeated_forward.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\models.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\P3\train_model.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\P3\predict_submission.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\P3\run_submission.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\build_official_final_submission_20260905.py
- C:\Users\cedis\PycharmProjects\PythonProject\artifacts\p3\features_all20_v1
- C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast\README.md
- C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast\score.py

# P3 v2 설계안 — "구조/물리 우선" (designer: structure), 2026-09-05

## 0. 이번 세션에서 직접 확인한 수치 (읽기 전용 집계, venv python, 각 ≤2분)

### 0.1 폭풍-발생(onset) 검증면 크기와 persistence 프로파일
| 표면 | anchor 수 | persistence RMSE (3/6/9/12/18/24h) | pooled |
|---|---:|---|---:|
| 공식 test (README) | 200 사례 | 0.546/0.701/0.706/0.760/0.890/0.946 | 0.769 |
| greedy 78h first-eligible (정점-전역) | 281 (G101/I80/S100), jc3 비율 0.77 | 0.595/0.755/0.812/0.866/0.928/0.931 | 0.823 |
| **jc3 = 직전 3h 최소 hs < 1.5 (갓 통과)** | **5,603 (G1963/I1648/S1992), 426 episode** | 0.562/0.702/0.774/0.803/0.889/0.920 | 0.784 |
| in-storm 6h 간격 (jc3 아님) | 1,247 | 0.487/0.626/0.720/0.799/1.022/1.141 | — |
| onset-dense = 0.72·jc3 + 0.28·in-storm | 6,850 | 0.542/0.682/0.759/0.802/0.929/0.987 | 0.797 |
| dense 전체 eligible | 24,360 | 0.470/0.626/0.717/0.792/1.001/1.112 | 0.816 |

- test_context에서 직접 계산: jc3 비율 **0.765**, hs0 분위(10/25/50/75/90) 1.52/1.55/1.64/1.86/2.39, 3h 상승량 중앙값 0.45 m(train jc3 0.34), wspd0 중앙값 10.7 m/s(train jc3 10.7 동일).
- jc3 anchor의 미래 형태: 리드별 평균 Δ = −0.005/−0.064/−0.113/−0.123/−0.252/−0.320 m, 24h 후 하락 70%, 6리드 중 정점 위치 히스토그램 3h 38%, 나머지 고르게 분산 → "발생 직후 곧 정점, 이후 완만 감쇠"가 평균 구조.
- 정점×리드 기후값(quarter LOQO)만으로: greedy78 24h 0.931→0.870, 18h 0.928→0.916, 그러나 3/6h는 악화(0.595→0.605, 0.755→0.774) → 기후값은 18/24h 전용 floor 후보로만 의미.

### 0.2 onset에서 정보를 가진 물리 변수 (jc3 n=5,603, Spearman with Δ6h/Δ12h/Δ24h)
wspd·gust 0.51/0.37/0.19, wspd_ch6h 0.45/0.35/0.21, wspd_ch3h 0.36/0.33/0.25, steepness hs/tp² 0.24/0.23/0.13, tp −0.21/−0.21/−0.17(장주기=너울→성장 약함), caph_ch3h 0.02/−0.10/−0.19(기압 하강→성장), hmax/hs 0.14/0.11/0.06, airt −0.12/−0.11/−0.11, **hs_slope3h 0.06/0.09/−0.04(약함!)**, run_h(1.5 초과 지속) 0.01/0.03/−0.06.
→ onset에서는 파고 모멘텀이 아니라 **바람(현재값·6h 변화)**이 핵심 정보. 21개 물리 변수 ridge 선형(quarter LOQO)만으로 jc3 pooled 0.784→0.741(−5.5%); 단 onset 전용 학습 선형모델을 greedy78의 in-storm anchor에 적용하면 12h 1.218로 폭발(외삽) → **학습은 전체 eligible anchor로, 평가만 onset 정합**이 원칙.

### 0.3 정직 CV 프로브 (6분기 블록, 78h 양방향 purge + episode 제외, LightGBM CPU 결정론, 68특징, 리드별 잔차)
| 변형 | jc3 pooled | greedy78 pooled | onset-dense pooled |
|---|---:|---:|---:|
| persistence | 0.7842 | 0.8228 | 0.7973 |
| LGBM 무가중 | 0.7163 | 0.8012 | 0.7046 |
| LGBM 기존 가중 exp(−0.45·(hs−1.5)) | 0.7197 | 0.8031 | 0.7074 |
| LGBM 선정-정합 72/28 가중 | 0.7231 | 0.8060 | 0.7091 |
| LGBM 정점 특징 제거 | — | — | 0.7057 |
| **LGBM 바람-드롭아웃 증강** | — | — | **0.6935** |
| 위 3변형 평균 | — | — | 0.6972 |
| 무가중 + OOF a_L shrink(LOBO) | — | — | 0.6957 |

- episode 블록 부트스트랩(428 episode, 300회): Δ(72/28가중 − 무가중) = +0.0047, **SE 0.0033**; Δ(모델 − persistence) = −0.087, SE 0.018. → 이 표면의 모델 간 분해능 ≈ 0.003 m(기존 181사례 표면의 1/1.5~1/2), Public SE(0.02~0.05)의 1/10.
- 블록별(onset-dense, 무가중 base): 24Q1 0.647 / 24Q2 0.615 / 24Q3 0.780 / 24Q4 0.680 / 25Q1 0.754 / **25Q2 0.734 (persistence 0.713보다 악화)**. 25Q2 정점별: G 0.697(pers 0.715) / **I 0.768(pers 0.692)** / S 0.740(pers 0.731). 바람-드롭아웃 증강 후 25Q2 0.703(I 0.740, S 0.666) → 원인은 I/S-ORS 기상이 2025년에만 존재하는 **구조적 결측 이동**(2024 I/S anchor는 wind NaN → 모델이 "NaN=2024 I/S 폭풍"을 학습). test는 기상 ~97% 존재하므로 이 증강은 test에도 직접 효과.
- 무가중 모델의 jc3 편향(truth−pred) −0.001/−0.004/−0.006/−0.012/−0.033/−0.002 → 기존 계보의 +0.07~0.14 편향은 dense 학습 자체가 아니라 특징(onset 위상 부재)+shrink 때문이었음. 선정-정합 가중은 불필요(악화).
- OOF a_L(리드별 `hs0 + a_L·(pred−hs0)`, 24Q1 제외 적합): 0.904/0.950/0.902/0.788/0.762/0.737. 로컬은 shrink를 선호하지만 Public 실측(O a=0.80 → 0.6071, A a=0.75 → 0.6117, α=−2/−10 증폭 개선)은 정반대 → 기본형은 a_L=1(항등). 상세 §4.

## 1. 패키지 구조 `src/ocean_v2/p3/` (신규, 자체완결, GPU 불사용)

```
src/ocean_v2/common/          # 경로 해석(P3_DATA_DIR), sha256, seed 고정, 런타임 기록, submission validator (p3_wave.submission 복사)
src/ocean_v2/p3/__init__.py
src/ocean_v2/p3/data.py       # load/audit/grid/anchor (p3_wave.data 복사) + episode_id + phase 플래그 + test 사례 289×10 배열화
src/ocean_v2/p3/features.py   # 벡터화 창 특징: (N,289,10) float32 배열 → (N,F) — train/test 완전 동일 함수
src/ocean_v2/p3/cv.py         # 6분기 블록, 78h purge, episode 제외, 평가면(onset/storm/dense/greedy), episode 부트스트랩, 게이트
src/ocean_v2/p3/model.py      # 리드별 잔차 GBDT(LightGBM 주, CatBoost CPU 보조), 바람-드롭아웃 증강, seed 앙상블
src/ocean_v2/p3/calibrate.py  # OOF에서 a_L 산출 → fitted_params.json (기본 후보는 항등; 파일로만 존재)
src/ocean_v2/p3/train.py      # raw → features → (cv) → full fit → artifacts/ocean_v2/p3/<candidate>/models/*.txt|*.cbm + fitted_params.json + cv_report.json
src/ocean_v2/p3/predict.py    # raw + models → submissions/claude_v2/p3/<candidate>/submission.csv + sha256 + receipt.json
src/ocean_v2/p3/__main__.py   # python -m ocean_v2.p3 {audit|features|cv|train|predict|all} --config configs/ocean_v2/p3_<candidate>.json
configs/ocean_v2/p3_base.json # 후보별 설정(특징군 on/off, seeds, 학습기, 증강 가중, 보정 on/off)
```

재사용(복사 후 단순화, 원본 미수정): `src/p3_wave/data.py`(load_p3_data, audit_p3_data, build_training_grid, build_anchor_table dense=20), `src/p3_wave/submission.py`(전부), `src/p3_wave/validation.py`(rmse, metric_slices), `src/p3_wave/corrected_repeated_forward.py`(paired_case_bootstrap → episode 블록판으로 개조), `src/p3_wave/features.py`(summarize_context는 **등가성 테스트의 기준 구현**으로만; wind_wave_alignment·wind_input_proxy 정의 유지), `src/p3_wave/event_phase.py`(run/peak 특징 정의를 벡터화 재구현), `src/p3_wave/models.py`(LightGBM deterministic 파라미터 골격), `scripts/final_submission_20260905/P3/*.py` + `scripts/build_official_final_submission_20260905.py`(01_data…07_source 폴더 계약, TRAIN/PREDICT 노트북, contract.json 생성 골격). **재사용 금지**: loss_router, persistence_shrink, corrected_fixed_long_shrink, kma_*, era5_*, chronos*, deep/timexer/tsmixer, 모든 residual cycle.

## 2. 데이터 → anchor → 특징 (data.py / features.py)

### 2.1 anchor 테이블 (README 상수만 사용: hs≥1.5, 6리드 유효, 48h 컨텍스트, 78h 간격, 6h 완충)
- `build_anchor_table(grid, dense_spacing_minutes=20)` → 24,360 eligible anchor (G 9,893 / I 7,312 / S 7,155). 타깃 = `hs(t+L) − hs0` 잔차(리드별).
- 추가 열: `prev3_min`(직전 9슬롯 min, NaN 무시), `prev6_min`, `jc3 = prev3_min<1.5`, `run_h`(1.5 이상 연속 시간), `episode_id`(정점별: 18슬롯=6h 연속 "1.5 미만 또는 결측" 발생마다 +1; README의 6h 완충과 동일 상수), `block = anchor_time의 UTC 분기`.
- test 사례: test_context를 case별 289×10 배열(열: hs,tp,hmax,wvdir,wspd,gust,wdir,airt,relh,caph)로; train anchor는 10분 grid에서 `grid[pos−288 : pos+1]` 슬라이스로 **동일 배열 구조** 생성(stride view → float32 (24,360,289,10) ≈ 280 MB).

### 2.2 특징 (하나의 numpy 함수 `window_features(X: (N,289,10)) -> DataFrame`, 절대시각·달력 특징 없음)
1. 기본 시계열 12개: hs, tp, hmax, wspd, gust, caph, airt, relh, wvdir_sin/cos, wdir_sin/cos.
2. 파생 시계열 6개: energy=hs², steep=hs/tp², hmax_r=hmax/hs, wsq=wspd², align=cos(wdir−wvdir), wind_in=wsq·max(align,0).
3. 각 시계열: `current`(마지막 유한값) + `age_h`(마지막 유한값까지 경과시간; 기존 코드에 없던 결측 경과 정보), lag 1/3/6/12/24h(정확 슬롯), 창 3/6/12/24/48h의 mean/max/min/std/slope, valid 비율(6/24h).
4. 변화량: hs·wspd·gust·tp·caph의 current−lag(1/3/6/12/24h).
5. **onset 위상 특징**(구조 활용의 핵심): run_above_1p5_h, prev3_min, prev6_min, jc3 flag, hours_since_min24/48, rise_from_min24 = hs0−min24, hours_since_max24/48, drop_from_max24/48, crossings_1p5_48h, hs_slope_accel(최근 3h slope − 그 이전 3h slope).
6. **물리 파라메트릭 특징**(fitted_params.json에 계수 저장, 학습 데이터로만 적합):
   - 풍파 평형 파고 `hs_eq_s = c_s·wspd_mean3h²` — c_s는 정점별로 train grid에서 `hs ~ wspd_mean3h²`의 90% 분위 회귀(또는 (hs/wsq)의 상위 분위 중앙값)로 산출. 파생: `growth_room = hs_eq − hs0`, `hs_ratio = hs0/hs_eq`, gust 기반 동일 세트, `hs_eq_trend = c_s·(wspd_mean3h² − wspd_mean3h²_lag3h)`.
   - 너울/풍파 구분: steep_current, steep_ch3h, tp_ch3h/6h 부호, align_current, wind_in_mean6h.
   - 기압 경향: caph_ch3h/6h/12h, caph_slope_12h.
   기대: 트리가 wsq·hs로 근사 가능하므로 −0.000~−0.005 m(불확실). CV로 on/off 판정.
7. 정점: 정수 코드 1개(범주로 사용). 리드는 모델을 분리하므로 특징 없음. 총 약 180~220개, NaN 유지(LightGBM/CatBoost 내장 처리).
- 등가성 테스트(필수): 임의 anchor 30개에 대해 train 경로(grid 슬라이스)와 "가짜 test 사례"(같은 슬라이스를 test_context 형식으로 변환) 특징이 bit-동일한지 `np.array_equal` 검사; 기존 `summarize_context`와 공통 특징(mean/std/slope 등) 값 일치 검사(허용 1e-9).

## 3. 정직 CV 설계 (cv.py) — 사전 등록, 코드로 고정

- **블록**: anchor_time UTC 분기 6개 = 2024Q1, Q2, Q3, Q4, 2025Q1, 2025Q2 (각 jc3 수 1279/618/806/993/1263/644). 계절이 모두 1~2회 포함되고, 전방 전용 fold의 "7,912 anchor로만 학습" 문제를 제거(각 fold 학습량 ≈ 전체의 75~85%, 최종 모델과 근접).
- **fold k**: 검증 = 블록 k의 **모든** eligible anchor(OOF가 24,360 전부에 존재). 학습 = `anchor_time < block_start − 78h` 또는 `> block_end + 78h` 이면서 `(station, episode_id)`가 검증에 등장하지 않는 anchor. 검증 anchor의 [−48h, +24h] 발자국이 학습 anchor 발자국과 겹치지 않음(78h ≥ 72h+6h; 코드에서 assert).
- **평가면(OOF에 가중치로 정의)**:
  - S_onset: jc3 anchor 전부(5,603) 균등.
  - S_storm: jc3가 아닌 anchor 중 정점별 greedy ≥6h 간격(1,247) 균등.
  - **S_dense(주지표)**: `p_jc3·S_onset + (1−p_jc3)·S_storm`, `p_jc3`는 test_context에서 코드가 계산(0.765; 하드코딩 금지).
  - S_greedy: 정점-전역 78h first-eligible(281) — 보고용(공식 선정 규칙의 직접 모사, 표본 작음).
  - 보고: pooled, 리드별, 정점별, 블록별, 정점×블록.
- **불확실성**: (station, episode) 블록 부트스트랩 1,000회, 후보 간 Δ의 CI90·P(개선). 측정된 SE ≈ 0.003 m(후보 간), 0.018 m(vs persistence).
- **승격 게이트(사전 등록)**: S_dense Δ<0 ∧ P(개선)≥0.90 ∧ 6블록 중 ≥4 개선 ∧ 어떤 정점도 +0.010 m 이상 악화 없음 ∧ 18/24h 리드 악화 ≤ +0.005. 게이트 미통과 변형은 후보에 넣지 않음(업로드 금지).
- **표면 노출 관리**: 총 후보 평가 횟수 ≤ 12회로 제한(HPO 격자 4 + 특징군 ablation 4 + 학습기/seed 4). 0.006 m 미만 차이는 "동률"로 취급하고 단순한 쪽 선택.
- 기존 계보 재평가(Day 1, 배경 실행 15~20분): `artifacts/p3/features_all20_v1` 캐시 + compact 591특징 + CatBoost single(CPU, 700 iter) 을 같은 6-fold에 돌려 S_dense에서 새 파이프라인과 비교. 새 파이프라인이 열세면 591특징 세트를 features.py에 흡수(summarize_context 재구현).

## 4. 모델 (model.py) — 결정론 CPU GBDT

- **주 학습기**: LightGBM 리드별 6모델 × seed 3(또는 5). 파라미터 초기값(프로브에서 검증): objective=regression, n_estimators 600, learning_rate 0.03, num_leaves 15, min_child_samples 100, subsample 0.8 (freq 1), colsample_bytree 0.6, reg_lambda 10, `deterministic=True, force_row_wise=True, num_threads=8, seed 고정`. 사전 등록 HPO 격자 4개만: (leaves 15/31) × (trees 600/1000, lr 0.03/0.02 짝) — CV 1회 ≈ 40 s.
- **보조 학습기(선택, 사다리 2단)**: CatBoost CPU RMSE 리드별, iterations 1000, lr 0.03, depth 6, l2 8, random_strength 0.2, thread_count 8, random_seed 고정 (CPU에서 seed·thread 고정 시 결정론). 앙상블 = 리드별 단순 평균(가중치 CV 적합 불필요; 0.5/0.5 사전 고정).
- **학습 표본**: 24,360 eligible anchor 전부, **가중 없음**(선정-정합/수준 가중 모두 CV에서 악화 또는 무효).
- **바람-드롭아웃 증강**(측정 −0.011 m): wspd가 존재하는 학습행의 복제본을 만들어 기상 유래 특징(wspd, gust, wdir_*, caph, airt, relh, wsq, align, wind_in, hs_eq 계열, 그 lag/창/변화량)을 전부 NaN으로 두고 가중 0.5로 추가(가중 0.3/0.5/1.0은 격자 3개로 CV 판정). 근거: I/S-ORS는 2024 기상 전무 → 원모델은 "기상 NaN=2024 I/S"를 학습(25Q2 fold에서 persistence보다 악화). test는 기상 97% 존재.
- **정점-무관 멤버**: CV에서 base와 동률(0.7057 vs 0.7046) → 단독 채택 안 함. 앙상블 다양성 목적으로도 이득 없음(3변형 평균 0.6972 > 증강 단독 0.6935) → 제외.
- **타깃/후처리**: 리드별 잔차 → `pred = hs0 + Σ_seeds f_L(x)/n_seeds` → 클립 [0, 30](README). CSV float round-trip 후 validator.
- **보정(calibrate.py)**: OOF(S_dense 가중)에서 리드별 `a_L = Σw·d·r / Σw·d²`(d=pred−hs0, r=truth−hs0)을 산출해 `fitted_params.json`에 기록하되, **기본 후보는 a_L≡1**. 이유: 로컬 a_L(0.74~0.95)은 shrink를 가리키지만 Public 실측 3점(O a=0.80 → 0.6071, A a=0.75 → 0.6117, "shrink 제거" 등가 α=−4 → 0.5925 예측)은 test 기간이 반대 방향임을 보여주며, 이는 파라미터 적합이 아니라 사전 등록 후보 간 참고(공지에서 허용)로만 사용. a_L 적용 후보(P3_v2_calib)는 사다리 마지막에 두고, Public 확인에서 base보다 0.02 m 이상 좋을 때만 고려(그 미만은 잡음).
- 하지 않는 것: persistence shrink 상수, 라우터, LB 유래 α, TabPFN·외부자료, 분위수 블렌딩, 시퀀스 딥모델(2일 안에 결정론·검증 불가).

## 5. 후보 사다리 (기대 Private 이득순, S_dense 정직 CV 수치 병기)

| # | 후보 ID | 내용 | CV 근거 | 상태 |
|---|---|---|---|---|
| F0 | P3_v2_floor | persistence(3/6/9/12h) + 정점×리드 onset 기후값(18/24h, fitted_params) | greedy78 24h −0.061, 18h −0.012; 3~12h 무변화 | 실패 시 최후 폴백(코드 6줄, 재현 100%) |
| **B1** | **P3_v2_base** | LGBM 3-seed 리드별 잔차 + 바람-드롭아웃 증강 + onset 특징, a_L=1 | S_dense 0.797→≈0.69 (−13%) | **안전 기준선, 업로드 #1** |
| B2 | P3_v2_ens | B1 + CatBoost CPU 멤버 0.5/0.5 | 추정 −0.003~−0.006 (미측정) | 게이트 통과 시 업로드 #2 |
| B3 | P3_v2_phys | B2 + 물리 파라메트릭 특징군(hs_eq, growth_room, 너울/풍파, 기압경향) | 추정 −0.000~−0.005 (불확실) | 게이트 통과 시만 |
| B4 | P3_v2_seed5_hpo | B3 + seed 5 + HPO 격자 최적 | 추정 −0.002~−0.004 | Day 2 |
| B5 | P3_v2_season | B4 + 계절 균형/밀도비 가중(train 월 분포를 test_context airt 유래 계절 비율로 재가중; 밀도비는 (hs0, slope3h, wspd0, airt) 로지스틱 분류기) | 검증면으로 정직 판정 불가(재가중 CV는 순환); 추정 −0.000~−0.010 | 선택적, Public 확인 동반 |
| C | P3_v2_calib | 최종 후보 + OOF a_L | 로컬 −0.009, Public 증거 반대 | 마지막, 조건부 |

최종 지정 규칙(사전 등록): 게이트를 통과한 가장 높은 번호 후보. Public은 "버그 검출"용(0.63 초과·persistence 근접이면 파이프라인 오류 의심)과 B1↔C 방향 확인용으로만 사용하며, 어떤 후보의 계수도 Public으로 바꾸지 않음.

## 6. 결정론·재현·런타임

- CPU 전용. LightGBM `deterministic=True, force_row_wise=True, num_threads=8`, seed 고정(리드·seed별 `seed = base + 100·lead_idx + k`); CatBoost `thread_count=8, random_seed` 고정, GPU 미사용. numpy 특징은 순수 결정론.
- 산출물: `models/lgb_L{lead}_s{k}.txt`(텍스트 덤프, 총 <20 MB), `cb_L{lead}.cbm`, `fitted_params.json`(c_s, a_L, p_jc3, 특징 목록·해시), `cv_report.json`, `submission.csv` + `sha256`. PREDICT는 저장 가중치만으로 byte-exact 재현; TRAIN 재실행 시 같은 머신·버전에서 bit-동일, 타 머신 허용오차 |Δhs| ≤ 1e−4 m를 README에 명시(재현검증 4항 대비).
- 런타임(7800X3D 8코어): 데이터 로드 5 s, 배열화+특징 ≈ 1~2분, CV(LGBM 6 fold × 6 리드 × 3 seed ≈ 108 fit) ≈ 6~8분, CatBoost CV 36 fit ≈ 25~30분(선택), 최종 학습 18 LGBM + 6 CatBoost ≈ 12~15분, 예측 <10 s. **전체 ≤ 1시간**(6시간 한도의 1/6). 상수 리터럴 감사: 1.5·78·48·6·[0,30]·289는 README 출처를 docstring에 인용; 그 외 수치는 전부 fitted_params.json.
- 규정 체크리스트: 외부자료 0, 사전학습 0, LB 유래 상수 0(grep 감사 스크립트 포함), 배포 score.py 통과, 6 파일 SHA manifest.

## 7. 일정 (2026-09-05 ~ 09-07 정오, 업로드 3회/일, 사용자 수동 업로드)

**Day 1 (09-05 잔여 ~10h)**
1. (1.5h) `ocean_v2/common` + `p3/data.py`·`features.py` 작성, 등가성 테스트, 특징 캐시(parquet) 저장.
2. (1.5h) `cv.py`·`model.py`·`train.py` 작성 → B1 CV(무가중 vs 증강 가중 0.3/0.5/1.0 = 4회) → cv_report. 병행: 기존 계보 compact-CatBoost 6-fold 재평가(배경, 20분).
3. (0.5h) B1 full fit → predict → validator → SHA → **업로드 #1 (P3_v2_base)**. 기대 Public 0.585~0.600; 0.63 초과면 버그 조사(특징 정렬·클립·키 순서).
4. (2h) CatBoost 멤버 CV(배경 30분) + 물리 특징군 CV(2회) → 게이트 통과 시 B2/B3 생성 → **업로드 #2**(B2 또는 B3 중 CV 최상), #3은 예비(버그 대응용으로 남김).
5. (1h) calibrate.py + fitted_params, 패키지 스켈레톤(01_data…07_source, RUN_*.ps1, README 초안) 생성.

**Day 2 (09-06)**
1. (2h) HPO 격자 4회 + seed 5 → B4; B5(계절/밀도비) 1~2회 CV + 재가중 CV 보고(참고용).
2. (1h) 최종 후보 확정(게이트 규칙) → **업로드 #4 (B4)**, 필요 시 #5 (B5 또는 C, 사전 등록된 비교 목적), #6 예비.
3. (3h) 최종 패키지 빌드: TRAIN/PREDICT 노트북, contract.json(candidate_sha256), README(환경·seed·소요시간·허용오차·상수 출처), 상수 리터럴 grep 감사, 6 파일 SHA. 새 임시 폴더 **클린룸 재현**(raw → train → predict → SHA 비교, 소요 ≤1h).
4. 18:00 코드 프리즈. 사용자에게 업로드/삭제 대상(KMA·ERA5 8건, LB 적합 후보) 목록 전달.

**Day 3 (09-07 오전)**: 클린룸 재현 2회차(결정론 확인, SHA 동일), 최종 답안이 이미 채점된 파일과 byte-동일함을 확인 → 사용자 답안 업로드 → 모델 최종 제출.

## 8. 위험과 완화
1. **test 기간(2025-07~) 계절·태풍기 전이**: I-ORS 2024-08~10 파랑 100% 결측, I/S 2024 기상 전무 → 태풍기 I/S 학습 근거가 G-ORS 뿐. 완화: 바람-드롭아웃 증강(측정 효과), 정점 특징은 유지하되 트리 깊이 제한, 블록별·정점별 보고에서 I-ORS 여름 성능은 "미검증"으로 명시.
2. **shrink/증폭 부호**: 로컬(a<1) vs Public(a>1) 상충. 기본형 a=1로 중립화; a_L 후보는 조건부. 어떤 경우에도 a>1 증폭 상수는 넣지 않음(데이터 근거 없음).
3. **정직 CV가 양방향(미래→과거)**: 물리 예측 문제라 추세 없음 가정; 보고서에 명시. 대안 확인으로 forward-only 2 fold(2025Q1, 2025Q2)의 부호 일치를 함께 보고.
4. **표면 재사용 과적합**: 평가 횟수 상한 12회, 0.006 m 미만 동률 처리, 게이트 사전 등록.
5. **결정론**: LightGBM 버전·스레드 고정, CatBoost CPU만; 패키지에 `pip freeze` 고정, 클린룸 2회 재현으로 SHA 동일 확인. 타 머신 오차 허용치 문서화.
6. **test 결측 특수 사례**: hs 부분 결측 53사례, wspd 전결측 1사례 → NaN 내장 처리 + age_h 특징; 증강 덕에 기상 전결측 사례도 파랑 분기로 안정 예측. 극단값 클립 [0,30] 외 추가 가드: pred < 0.3·hs0 또는 > 3·hs0+2 면 로그만 남김(값 수정 없음).
7. **시간 초과**: CatBoost·B5는 선택 항목; B1만으로도 완결 패키지가 되도록 Day 1 오후에 스켈레톤을 먼저 완성.
