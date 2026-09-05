# P1 robust-ML v2: 격자 기반 특징 + GBDT 다중시드 + 겨울/단절 정합 CV + 구간 완성 (완전 재생성 파이프라인)

> 설계 패널(읽기 전용 설계자) 산출물, 2026-09-05. 저장소 파일·데이터는 수정하지 않았음.

## 기대 효과

기준: 현 챔피언 Public F1 0.8335(재현 불가·재현검증 탈락 가능성 높음). (1) C0 안전 기준선(격자 특징 + LGBM/XGB 3+3 시드 블렌드 + CV 적합 디코더): 정직 CV(겨울+단절 표면) 0.80~0.85, Public/Private 예상 0.80~0.83 — 챔피언 대비 −0.03~0.00 F1(−0.8~0점) 이지만 규정·재현 위험 0. 근거: 동일 계보 O/B 단독 Public 0.791/0.794이며, 격자 특징(7일 창 결측 17.9%→0.9%), 겨울 fold 임계값, 3+3 시드 블렌드, 갭 브리징 디코더가 각각 +0.005~0.01 기대. (2) C1 = C0 + flank/annulus·step·ramp 특징 + 구간 완성 랭커(stage 2): 직접 계산에서 offset 행 AUC 0.635→0.91, drift 0.687→0.843 → offset/drift recall 0.65→0.75~0.85 시 전체 recall +0.05~0.10, precision 0.85 가정 시 F1 +0.02~0.04 → Private 0.84~0.87 기대(불확실, 표준오차 ±0.01). 챔피언 대비 기대 Private 효과: 점추정 +0.00~+0.03 F1(0~+0.8점), 하방 −0.02(−0.5점). (3) 이후 rung(typed head, gap 증강, 소형 결정론 TCN)은 각 +0.003~0.015(불확실). 가장 큰 가치는 '재현검증 통과 = 점수 유지' 이며, 챔피언을 유지할 경우 기대 손실은 −33.3점(실격) × P(탈락)이다.

## 재사용 대상

- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\data.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\postprocess.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\rules.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\metrics.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\validation.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\models_tabular.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\submission.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\ensemble.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\features.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\run_p1_meaningful_learning_curve_generation_v1.py
- C:\Users\cedis\PycharmProjects\PythonProject\artifacts\runs\20260813T153038+0900_cv_378a4e89\config.toml
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\common.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\build_official_final_submission_20260905.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p1_qc\change_points.py
- C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly\score.py


# P1 robust-ML v2 구현 계획 — `src/ocean_v2/p1/` (2026-09-05, 설계자 robustml)

## 0. 한 줄 요약
10분 격자(grid) 위에서 gap-내성 양방향 특징을 만들고, CPU 결정론 GBDT(LightGBM 3시드 + XGBoost 3시드)를 이벤트-일 균형 가중으로 학습, **2025-H1 겨울 블록 + 2025 Q3/Q4 + 2024-H1(S-ORS)** 4블록·21일 퍼지·test형 단절 증강 검증면에서 임계값·브리징·블렌드·구간완성 파라미터를 **코드가 적합해 `fitted_params.json`으로 저장**하는 완전 재생성 파이프라인. 리더보드 상수 0건, 동결 CSV 0건, 총 재현 시간 ≈ 2.5~3.5 h.

## 1. 설계 근거 (정찰 + 직접 계산, 모두 집계값)
| 사실 | 수치 | 설계 결정 |
|---|---|---|
| 주입 예산은 station-year-layer 전체에 대해서만 균일(3.99~4.19%); 반기별로는 0.4%~10.4% (G-ORS 2025 H1 0.43%, H2 10.4%) | agg1 | test(2026 H1) 유병률을 4.1%로 가정하지 않음. 계열별 유병률은 **진단**에만 사용(경보 기준: 예측률 <1% 또는 >12% 계열 표시) |
| 2025-H1 전 정점 = 208,093행 / 9,911양성 / 62사건(spike 21·drift 10·offset 10·noise 11·flatline 10) | agg2 | 겨울-봄 fold `W25`를 1차 선택면으로 신설 |
| psal은 주입과 무관(flatline 행 Δtemp=0 96.7% vs Δpsal=0 0.8%; 양성/음성 Δpsal 중앙값 0.008/0.011) | agg2 | psal을 '깨끗한 공변량'으로 사용: psal 잔차·step 특징, temp-특징과 동일 창 미러링 |
| 층 간 주입 독립(양성행의 동시각 peer 양성 19.7% vs 음성행 18.7%) | agg2 | peer(인접층) 잔차·step 특징 신뢰 가능 |
| offset = 깨끗한 step(peer잔차 사건내 std 0.45 vs |평균| 4.8), drift = 선형 ramp(R² 중앙값 0.94), noise 사건내 std 2.46 | agg2 | 구간 후보에 step/ramp/diff-std 적합 특징 → 랭커 |
| 동일값 연속 ≥12행: 6,288행 전부 양성(자연 FP 0), flatline 양성 6,314행의 99.6% | agg2 | flatline 하드룰(OR) 채택, 파라미터 12 = README 최소 2h |
| singleton spike 룰 단독 precision ≈3%(thr 2.0: 4,155 플래그 중 진짜 92) | agg2 | spike는 룰 ∧ 모델확률 게이트로만 허용(CV로 게이트 선택) |
| exact-cadence 세그먼트 기준 7일 창 결측 test 17.9% ↔ 격자 시간창(±84h, 25% 지원) 결측 0.9% | agg2 | **격자 기반 시간창 특징**으로 train/test 구조 shift 제거 |
| 양측 flank(annulus) 잔차: offset 행 AUC 0.635(중심 72h) → 0.910(24h/120h), drift 0.687 → 0.843 | agg3 | 장기사건 내부 행 미탐(anchor FN의 98.6%)을 겨냥한 핵심 특징군 |
| 자연 |잔차| p99.5 ≈ 5.4 °C(성층·내부파) > offset 중앙 2.7~4.8 | agg3 | 온도 단독 잔차는 부족 → peer/psal 조건부 잔차 + GBDT 결합 필수 |

## 2. 패키지 구조 — 새로 만드는 파일 (모두 `C:\Users\cedis\PycharmProjects\PythonProject\src\ocean_v2\p1\`)
```
src/ocean_v2/__init__.py
src/ocean_v2/common/{__init__,paths,hashing,determinism,submission_io}.py   # P1~P3 공용(경로 해석 P1_DATA_DIR, SHA manifest, seed/thread 고정, CSV 검증)
src/ocean_v2/p1/__init__.py
src/ocean_v2/p1/io.py          # train/test 로드(키 계약·중복·시각 파싱 검증), sample_submission 키 순서 계약
src/ocean_v2/p1/grid.py        # (station,[year],layer) → 10분 완전 격자 reindex, obs 마스크, 원행 매핑, gap 통계
src/ocean_v2/p1/features.py    # §5 특징 v2 (격자 위 계산 후 관측행으로 투영), 특징 스키마 해시
src/ocean_v2/p1/detectors.py   # 라벨-프리 하드 탐지기: identical-run(≥12, gap≤60분 허용 변형 포함), spike 후보, 변화점 강도(|Δ 6h-median|/MAD)
src/ocean_v2/p1/cv.py          # §6 블록 fold, 21일 퍼지, 사건 배정, test형 단절 증강, LOFO 임계값 추정, 블록 부트스트랩
src/ocean_v2/p1/weights.py     # event-day balanced 가중(복사), sqrt 균형 가중
src/ocean_v2/p1/models.py      # LightGBM/XGBoost 결정론 래퍼, 다중시드 학습/예측, best-iteration, 모델 직렬화(joblib+json 메타)
src/ocean_v2/p1/decode.py      # 격자 hysteresis(브리징), flatline OR, spike 게이트, min-run, 파라미터 격자 탐색(OOF 전용)
src/ocean_v2/p1/segments.py    # stage 2: 구간 후보 생성 + 구간 특징 + 랭커 학습/적용(add-only)
src/ocean_v2/p1/calibrate.py   # fitted_params.json 작성: 임계값·브리징·min-run·spike 게이트·블렌드 가중·iteration·τ2
src/ocean_v2/p1/train.py       # 전체 학습 드라이버: features → cv → calibrate → 전량 refit → 03_model/ 저장 + 리포트
src/ocean_v2/p1/predict.py     # 03_model/ 로드 → test 특징 → 확률 → decode → segments → CSV + SHA
src/ocean_v2/p1/report.py      # CV 리포트(블록별·정점/층별·유형별 recall/precision, 예측률 진단, worst-block)
src/ocean_v2/p1/cli.py, __main__.py   # python -m ocean_v2.p1 {features|cv|train|predict|validate} --config configs/ocean_v2/p1_<cand>.json
configs/ocean_v2/p1_c0.json, p1_c1.json, p1_c2.json ...   # 탐색 격자·구조 상수만(§10)
tests/ocean_v2/test_p1_{grid,features,decode,cv,segments}.py   # 합성 시계열 단위 테스트
scripts/ocean_v2/build_p1_package.py    # 기존 빌더의 scaffold를 재사용해 01_data…07_source 패키지 생성(모델을 실제로 학습·추론)
```

## 3. 재사용 vs 재작성
**복사/재사용(함수 단위로 복사해 ocean_v2 내부에 두어 패키지 독립성 유지):**
- `src/p1_qc/data.py`: `load_dataset`(사이즈/mtime·SHA 감사), `parse_anomaly_types`, `segment_timeseries`(gap 통계용).
- `src/p1_qc/postprocess.py`: `hysteresis_threshold`, `close_short_gaps`, `remove_short_runs`, `segments_from_mask` — `breaks` 배열을 격자에서 만들어 그대로 사용.
- `src/p1_qc/rules.py`: `plateau_runs`(min_run 12), `detect_singleton_spikes`(후보만).
- `src/p1_qc/metrics.py`: `micro_f1`, `evaluate_predictions`, `event_report`, `anomaly_type_recall`, `group_report`.
- `src/p1_qc/validation.py`: `paired_block_bootstrap`, `normal_station_layer_day_fp`.
- `src/p1_qc/models_tabular.py`: LightGBM/XGBoost 결정론 파라미터 세트(`deterministic=True, force_row_wise=True`, hist CPU).
- `scripts/run_p1_meaningful_learning_curve_generation_v1.py`: `_event_day_weight`, `_binary_weight`, `_lgb_parameters`, `_typed_target`, `_multiclass_weight`(rung C2).
- `src/p1_qc/submission.py`: `build_submission`, `write_submission`, `validate_submission`.
- `src/p1_qc/ensemble.py`: `fit_convex_blend`(2모델 가중 격자).
- `scripts/final_submission_20260905/common.py`, `scripts/build_official_final_submission_20260905.py`(`scaffold`, `populate_official_data`, `write_problem_docs`, zip/분할 로직) — 패키지 빌더 골격.
- 하이퍼파라미터 출발점(리더보드 무관, 로컬 계보): O = XGB depth7/lr0.04/700/mcw20/sub·col0.85 (`artifacts/runs/20260813T153038+0900_cv_378a4e89/config.toml`), B = LGBM 900/lr0.035/leaves31/min_child100/sub·col0.85/α0.2/λ1.0 + event-day 가중.

**재작성(이유):** `features.py`(exact-cadence 세그먼트 → 격자 시간창; 전구간 통계 `plateau_full_length/nominal_depth/depth_regime` 제거), `splits.py`(4-12월 3fold → 겨울 포함 4블록·21일 퍼지·양측 train·단절 증강), `pipeline.py`(inner 60일 calibration 대신 LOFO OOF 보정), MS-TCN 계열 전부(비결정·52M·자체 gate 실패 → 불채택), `router_anchor.csv`/`gi_spike2_patch.json`(동결 CSV → 폐기), `change_points.py`(과도한 일반성; stage 2는 단순 후보 생성기로 새로 작성, 필요 시 `_continuous_slope_gain` 등 개별 함수만 차용).

## 4. 데이터 표현 — 10분 격자
- 키 그룹: train은 (station, year, layer), test는 (station, layer)(연도 단일). 각 그룹의 [min t, max t]를 10분 완전 격자로 reindex(KST). 관측 없는 격자행은 temp/psal/depth NaN, `obs=0`.
- 모든 롤링 통계는 격자 행 수 = 시간 창이며 `min_periods = ceil(0.25·창행수)`(관측 수 기준) — 창이 gap을 '건너뛰는' 것이 아니라 관측이 적으면 NaN이 되는 구조(train/test 동일 규칙).
- 격자 크기: train ≈ 1.2~1.3M행, test ≈ 0.3M행 → 특징 계산 5~8분(pandas rolling median/std, 8스레드 무관 단일 프로세스, 결정론).
- 관측행으로 투영 후 원 CSV 행 순서 복원(`__original_position`).

## 5. 특징 설계 (약 140열; 모두 라벨-프리, 격자 위 centered)
A. 원시/맥락(12): temp, psal, depth_raw(§: test에서 depth가 전부 결측인 정점(G-ORS)은 train에서도 depth 계열을 NaN 마스킹 — test 입력 가용성에서 파생된 규칙, 라벨 무관), psal_missing, depth_missing, station(카테고리), layer(카테고리), day_sin/cos, hour_sin/cos, 계열 시작 후 경과일(격자).
B. 국소 차분(10): Δ1, |Δ1|, Δnext, spike_min_abs_diff(부호 반전 조건 포함), 중심 곡률, 2차 차분, Δpsal, |Δpsal|, robust z(Δ1 / 1.4826·MAD(|Δ1|, ±7d)), psal z 동일.
C. 중심 롤링(temp; 창 1h/3h/6h/12h/24h/72h/168h/336h × {median 잔차, |잔차|, robust-z(MAD), std, diff-std, range}) ≈ 48. psal에는 창 6h/24h/168h × {median 잔차, diff-std} 6.
D. **양측 flank(annulus) 잔차** — 핵심(≈ 24): (r, R) ∈ {(12h,84h), (24h,120h), (48h,168h)}에 대해 left_med = median(temp[t−R, t−r]), right_med = median(temp[t+r, t+R]); 특징: temp − mean(left,right), temp − left, temp − right, |left − right|(flank 일치), 위 잔차의 MAD-정규화(±7d), 같은 것을 psal과 peer 잔차에도 적용(step 대비 공변량 무반응 = 이상). 관측 25% 미만이면 NaN.
E. Peer(같은 정점, 같은 격자시각, 다른 층; ≈ 18): peer_mean, peer_median, 인접층(위/아래 nominal depth 순) temp 차, peer_count/available, 위 각각의 24h/168h 중심 median 잔차 및 annulus(24h/120h) 잔차, station_layer_temp_std.
F. Plateau(4): exact 동일값 run 길이(양방향 full length; gap 없음 조건), gap≤60분 허용 동일값 run 길이, 현재 run 내 위치 비율, 12행 이상 플래그.
G. 구조/가용성(8): ±12h/±24h/±7d 관측 수, 직전/직후 gap 분(격자 기준), 세그먼트 길이(exact), 세그먼트 내 위치 비율.
H. 변화점 강도(4): |Δ(6h-median)| / MAD ±7d 및 그 ±24h 내 최대·argmax 거리(stage 2 후보 생성과 공유).
→ 검증: train/test 각 열의 NaN율·분위를 리포트에 남기고 NaN율이 test에서 train의 2배 초과인 열은 경고(격자 설계로 대부분 해소).

## 6. 검증면 설계 (정직 CV)
**블록(4개, 관측 시각 KST):**
| fold | 검증 블록 | 학습 | 목적 |
|---|---|---|---|
| `W25` (1차) | 2025-01-01~2025-07-01 전 정점 (208k행, 62사건) | 나머지 전부(S-ORS 2024 전체 + 2025-07-22~12-11 전 정점) | test(2026-01~06) 계절·정점 구성 정합. 오프라인 QC이므로 미래 데이터로 학습하는 것이 정당 |
| `Q3` | 2025-07-01~10-01 | 나머지(양측) | 기존 표면과의 연속성 |
| `Q4` | 2025-10-01~12-11 | 나머지 | G-ORS 양성이 많은 블록(G 진단) |
| `S24H1` | 2024-01-01~07-01 S-ORS만 | 나머지 | 두 번째 겨울 표본(39사건), 정점 편향 억제 |
- **퍼지 21일**: 검증 블록 경계 양쪽 21일(특징 최대 창 ±7d + annulus R 7d 여유)의 행을 학습에서 제거. 
- **사건 배정(anchor 규칙)**: 양성 run은 gap≤6h 허용으로 병합해 하나의 사건(258개)으로 보고, 사건 시작 시각이 속한 블록에 통째로 배정; 경계를 넘는 사건의 반대편 행은 학습에서도 제외(이중 계상 금지). 부트스트랩 블록 = 사건 + 정상 station-layer-day(기존 함수).
- **test형 단절 증강(검증면, 필수)**: 검증 블록의 각 (station, layer) 계열에 test 타임스탬프에서 추정한 gap 밀도(914 gap/169,011행 ≈ 5.4/1,000행; train 1.56/1,000)와 gap 길이 경험분포(중앙 60분, p90 950분)를 고정 seed로 표본해 행을 제거하고 특징을 **재계산**한다(제거된 행은 채점 제외). 리포트에 `intact`와 `fragmented` 두 표를 남기되 **선택은 fragmented 기준**. test 라벨은 전혀 쓰지 않으며 타임스탬프만 사용.
- **선택 규칙(사전 등록)**: 주지표 = 4블록 row-pooled F1(fragmented). 제약: (i) 어떤 블록도 비교 대상보다 −0.005 이상 나쁘지 않을 것, (ii) offset/drift/noise/flatline recall 및 정점별 precision 리포트 필수(게이트는 아님), (iii) 동률(±0.002)이면 단순한 쪽. 
- **파라미터 보정(LOFO)**: 디코더/블렌드/τ2는 4블록 OOF 확률만으로 격자 탐색. 정직 추정치는 leave-one-fold-out(3블록 OOF로 선택 → 나머지 1블록에 적용)으로 보고하고, **배포 파라미터는 4블록 OOF 전체에서 pooled F1 최대 + worst-block 제약(최선의 worst-block −0.005 이내)**으로 결정 → `fitted_params.json`. 기존 pipeline의 60일 inner calibration은 폐기(겨울 미포함·표본 작음).
- **iteration 수**: 각 fold에서 1,500 트리 학습 후 검증 logloss 최소 iteration 기록; 정직 추정은 다른 3 fold 중앙값을 `num_iteration`으로 적용(재학습 불필요), 배포는 4 fold 중앙값.
- **진단 필수 출력**: 블록별 F1/precision/recall, 정점×층 예측률(train 계열 유병률과 대조), 유형별 recall, 사건 지속시간대별 row recall(48h+ 별도), 정상-day FP율, 21일 블록 부트스트랩 CI90(vs C0).

## 7. 모델·디코더·구간 완성
**Stage 1 (행 확률):**
- M1 LightGBM binary ×3 seeds(20260905/06/07): `num_leaves 31, lr 0.035, min_child_samples 100, subsample 0.85 (freq 1), colsample 0.85, reg_alpha 0.2, reg_lambda 1.0, n_estimators ≤1500(CV iteration), deterministic=True, force_row_wise=True, num_threads=8 고정, seeds 전부 고정`; 가중 = event-day balanced(양성: 사건길이^−1/2 정규화 × sqrt(neg/pos), 음성: 정상 station-layer-day 크기^−1/2 정규화).
- M2 XGBoost hist CPU ×3 seeds: `max_depth 7, lr 0.04, min_child_weight 20, subsample/colsample 0.85, reg_lambda 1, n_estimators ≤1200(CV), nthread=8`; 가중 = sqrt 균형.
- 블렌드 p = w·mean(M1) + (1−w)·mean(M2), w ∈ {0,0.1,…,1} OOF 디코드 후 F1로 선택(§6 규칙). (rung C3: 정점별 w를 nested로 선택.)
- 범주형: station/layer는 LightGBM categorical, XGBoost는 원핫.
**디코더(격자 위, 라벨-프리 구조 + CV 파라미터):**
1. hysteresis: high ∈ {0.10,0.15,…,0.60}, low = ratio·high, ratio ∈ {0.5,0.7,0.85}; 후보 run은 관측 없는 격자행을 `bridge_rows` ∈ {0, 6, 18, 36}(0~6h)까지 건너 이어짐(test의 짧은 gap 다수 대응; fragmented CV가 선택).
2. flatline 하드룰 OR: exact 동일값 ≥12행(자연 FP 0) + gap≤60분 허용 변형(CV로 채택 여부).
3. spike: 룰 후보 ∧ p ≥ p_spike ∈ {0.3,0.5,0.7}일 때만 singleton 보존; 그 외 `min_run` ∈ {6,12} 미만 run 제거.
4. 출력 = 0/1 라벨(anomaly_type 열은 생략해 스키마 위험 제거).
**Stage 2 — 구간 완성 랭커(C1, add-only):**
- 후보: stage-1 run(seed)마다 좌·우 각 ≤48h(288격자행) 범위에서 변화점 강도(§5H) 국소 최대 상위 8개 + run 자체 경계 → (start,end) ≤81쌍; 24h 이내 인접 run은 병합 후보 추가; 후보는 항상 seed run을 포함(add-only 보장); 최대 길이 = train 최대 사건 730행 + 10%.
- 구간 특징(≈28): 길이, 관측비율, p 통계(평균/q10/q50/min, 12h flank 평균, inside−flank), step = median(inside) − median(flank 12~60h) / flank MAD (temp·psal·peer 각각), ramp 선형적합 slope×len·R², diff-std inside/flank 비, 경계 점프 |Δ| 시작/끝, flank 일치 |left−right|, 동일값 비율, 정점/층.
- 라벨: IoU(후보, 진짜 사건) ≥ 0.6 (0.5/0.7도 CV로 비교). 학습 데이터 = 각 fold의 OOF 확률로 만든 후보(랭커도 nested: fold k 적용 시 k 제외 fold 후보로 학습). 모델 = LightGBM 3시드(leaves 15, lr 0.05, 400 트리).
- 결정: seed마다 최고 점수 후보 채택 if score ≥ τ2(격자 {0.3,…,0.8}, §6 규칙) → 라벨 = stage 1 ∪ 채택 구간. 
- 폴백 규칙(C1a, 랭커 실패 시): 잔차 부호·크기(|annulus resid| ≥ 0.5×run 중앙 |resid|) ∧ p ≥ low2(CV) 인 동안 양방향 확장.
**옵션 딥 모델(C5, 시간 남을 때만):** 격자 특징 시퀀스(2016행=14일 창, stride 1008) 위 소형 양방향 dilated 1D-CNN(≈1M params, fp32), BCE+dice, 3 seeds×≈10분(RTX 5090). 결정론: `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True, benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, 고정 seed·DataLoader worker 0, 가중치 동봉 + 재학습 경로 시 SHA 허용오차 문서화. 블렌드 가중은 CV. 채택 조건: fragmented pooled F1 +0.005 이상 & worst-block 제약.

## 8. 후보 사다리(기대 Private 이득 순, 모두 동일 CV 표면에서 비교)
| 후보 | 내용 | 기대 이득(정직 CV 기준, 불확실) | 비고 |
|---|---|---|---|
| **C0 안전 기준선** | §4 격자 + §5 A/B/C/E/F/G + M1/M2 블렌드 + 디코더(1~3) | O/B 계보의 정직 강도 이상(로컬 0.80~0.85; Public 0.80~0.83) | 폴백 후보. 첫 업로드 |
| **C1** | C0 + §5D annulus/step/ramp 특징 + §5H + Stage 2 랭커 | **+0.02~0.04** (offset/drift 내부 완성) | 1차 목표 |
| C2 | typed multiclass head(1−P(normal) 및 유형확률을 stage 2 특징·디코더 입력) | +0.003~0.01 | 학습 +25분 |
| C3 | 정점별 블렌드 가중(nested) + 정점별 high threshold(nested) | +0.00~0.01 (router 효과의 정직 대체) | 과적합 감시: LOFO 추정 필수 |
| C4 | gap 증강 학습(train 계열 50%를 test형 단절 복제로 대체) | +0.003~0.015 | CV 시간 +40% |
| C5 | 소형 결정론 TCN 블렌드 | +0.005~0.015 | 09-06 저녁 이후 여유 시에만 |
| (금지) | 유병률 강제 매칭, LB 기반 행 패치, 동결 CSV union | — | 규정·붕괴 위험 |

## 9. 결정론·런타임 예산(8코어, 재현 환경 기준)
- 모두 CPU. LightGBM `deterministic=True, force_row_wise=True, num_threads=8`, XGBoost `tree_method=hist, device=cpu, nthread=8`, 모든 seed 명시, pandas/numpy/lightgbm/xgboost 버전 pin(requirements). 스레드 수를 코드에서 고정하므로 코어 수가 달라도 결과 동일(느려질 뿐).
- 예산: 특징(train+test 격자) 6~8분 → CV 4블록 × (3 LGBM ≈4분 + 3 XGB ≈4분) ≈ 1.6~2.0h → fragmented 검증 특징 재계산 4×2분 → 디코더/블렌드 격자 탐색(OOF만) 10분 → stage 2 후보·랭커 15분 → 전량 refit 6모델 ≈25분 → 예측·디코드 5분. **합계 ≈ 2.5~3.0h (C4 포함 시 ≈3.7h) ≤ 6h.** 모든 단계 wall-clock을 `03_model/runtime.json`에 기록.
- 검증 항목 4(학습 산출물 제거 후 재생성) 대응: `RUN_TRAINING.ps1`이 CV까지 포함해 `fitted_params.json`·모델을 처음부터 재생성; `RUN_INFERENCE.ps1`은 `05_answer/P1_submission.csv` 생성 후 contract SHA 비교(동일 머신 byte-exact 확인).

## 10. 상수 정책(09-02 공지 대응)
- 코드/설정에는 (a) 탐색 격자, (b) 데이터 구조 상수만 허용하고 각 상수 옆에 근거 주석: 12행(README flatline 최소 2h), 21일 퍼지(특징 최대 창 ±7d×3), 6h 사건 병합(README 사건 간격), 최대 구간 730행(train 최대 사건), gap 분포(test 타임스탬프 경험분포·코드가 계산). 
- 적합값(임계값·ratio·bridge·min_run·p_spike·w·iteration·τ2·정점별 값)은 전부 `train.py`가 CV OOF에서 계산해 `03_model/fitted_params.json`에 저장 — 소스에 리터럴 없음. `tests/ocean_v2/test_no_magic_constants.py`로 소스 내 0.05~0.95 부동소수 리터럴을 grep 감사.
- 리더보드 점수는 어떤 파라미터·후보 선택에도 사용하지 않고 `reports/ocean_v2/p1/upload_log.md`에 sanity 기록만 남김.

## 11. 업로드 계획(하루 3회 × 2일, sanity check 전용)
- 09-06 #1: **C0**(재생성 파이프라인 산출). 기대 Public 0.80~0.83. 0.75 미만이면 파이프라인 버그(키 순서·시각 파싱·격자 투영) 의심 → 원인 수정.
- 09-06 #2: **C1**(CV 통과 시). 09-06 #3: C2 또는 C3 중 CV 최상(예비).
- 09-07 #1: 최종 후보 refit 본(패키지에서 실제 생성한 CSV; SHA 일치 확인). #2/#3: 버그 수정 재업로드 예비.
- 최종 답안 선택 규칙(사전 등록): fragmented CV pooled F1 최상 후보. Public이 CV 예측보다 0.03 이상 낮으면 '버그 신호'로만 취급(파라미터 조정 금지). Public 동률·±0.006 차이는 무시(표본 ≈30%·행당 ±0.0003).
- 폴백: C0. (기존 챔피언/e150 패키지는 재현 불가이므로 폴백 아님.)

## 12. 일정(현재 09-05 오후 KST, 모델 제출 09-07)
**D0 09-05 오후~밤(≈8h):** io/grid/features(A/B/C/E/F/G)/detectors/cv(블록·퍼지·배정·단절증강)/weights/models/decode/calibrate/train/predict/cli + 합성 단위테스트 → C0 CV 전체 실행(≈2h, 백그라운드) → CV 리포트 검토 → C0 CSV 생성·score.py 스키마 검증·SHA 기록.
**D1 09-06:** 09:00 C0 업로드(#1). 오전: §5D/H 특징 + segments.py(후보·특징·랭커·τ2) 구현, C1 CV 실행(≈2.5h). 오후: C1 결과 검토 → 업로드(#2); C2(typed head) 구현·CV → #3. 저녁: C3/C4 CV 백그라운드, 패키지 빌더(`scripts/ocean_v2/build_p1_package.py`) 작성 — 01_data(링크)/02_train(TRAIN.ipynb→train.py)/03_model/04_predict/05_answer/06_report/07_source(ocean_v2 폐쇄 의존)/RUN_*.ps1/README(환경·seed·스레드·소요시간·SHA)/contract.json.
**D2 09-07 오전:** CV 기준 최종 후보 확정 → 새 임시 폴더 클린룸 재현(raw→train→predict, ≤3.5h 계측, SHA byte-exact) → 최종 CSV 업로드(#1) → 사용자에게 패키지·폼 값·삭제 대상 제출 목록 전달 → 사용자가 답안 업로드 확인 후 모델 제출. 12:00 완성 목표, 오후는 버퍼.

## 13. 위험과 대응
- 정직 CV가 챔피언 Public(0.8335)보다 낮을 수 있음(C0 −0.03 가능): 재현·규정 리스크 제거가 우선이며 C1이 회복 목표. 사용자에게 수치로 보고.
- Stage 2 구현 지연: 4h 상한, 초과 시 C1a 확장 규칙으로 대체.
- 단절 증강 구현 오류(사건 이중 계상·특징 누출): 합성 시계열 테스트 + intact/fragmented 차이 리포트(차이 > 0.05면 점검).
- G-ORS: W25 양성 72행뿐 → G 진단은 Q4 fold 우선; test G-ORS depth 전결측은 학습 마스킹으로 정합.
- I-ORS 2025-05 공백, I-ORS L5 test 19k행(train 10k): 정점×층 예측률 진단으로 이상 확인만.
- 임계값 전이(블록별 유병률 3.9~4.8%): worst-block 제약 규칙 사전 등록, 블록별 최적 threshold 분산을 리포트.
- 결정론: 같은 머신 byte-exact 검증; 다른 머신 차이 가능성은 README에 명시(스레드 고정으로 최소화).
- 시간 초과: C4/C5는 선택적; CV 시드 수를 3으로 유지(5로 늘리지 않음).

## 14. 구현 체크리스트
[ ] grid reindex 왕복(원행 순서·개수 보존) 테스트 [ ] 특징 NaN율 train/test 표 [ ] fold 행 겹침 0·퍼지 21일 검증 [ ] 사건 배정 이중계상 0 [ ] fragmented 표면 생성 seed 고정 [ ] LOFO 정직 추정 vs 배포 파라미터 분리 저장 [ ] fitted_params.json에 모든 적합값 [ ] 소스 리터럴 감사 통과 [ ] score.py 스키마 통과 [ ] 클린룸 raw→train→predict ≤6h·SHA 일치 [ ] upload_log에 LB 사용 금지 서약·기록.

