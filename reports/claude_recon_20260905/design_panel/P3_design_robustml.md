# P3 Angle B — 선정-정합 검증면 위의 재현 가능한 GBDT 앙상블 (src/ocean_v2/p3)

> 설계 패널(읽기 전용 설계자) 산출물, 2026-09-05. 저장소 파일·데이터는 수정하지 않았음.

## 기대 효과

기준점: 현 챔피언 0.5839(Public, LB 적합 α=−10.2)은 재현 검증 탈락·Private 하방(−1.4점 꼬리) 위험이 있어 기대값 비교 대상은 '정직한 O' 0.607(Public, 23.70점)이다. 로컬 6블록·3seed 프로토타입(LightGBM 180라운드, 캐시 특징) 기준 persistence 0.8228→0.7781(가중·온셋특징) →0.7667(LOBO slope shrink) = −6.8%; 구성요소별 로컬 효과: 선정-정합 가중 −0.010±0.007, slope shrink −0.010±0.005, seed/멤버 앙상블 −0.003~−0.005(분산 감소가 주효과). 과거 로컬→Public 전이율이 낮았으므로 50% 할인하면 Public 기준 O 대비 −0.008~−0.015 m(+0.13~+0.25점), 즉 Public ≈0.59~0.60(불확실 ±0.015). Private는 persistence가 3.8% 더 어렵고 여름 사례 30%라 ≈0.60~0.62 m(23.3~23.7점) 예상. 현 챔피언의 Private 기대값(0.60~0.67, 재현 실패 시 0점)과 비교하면 기대 점수는 동등 이상, 하방 꼬리는 제거. 가장 불확실한 항목: I-ORS(70사례)에서 모델이 persistence를 거의 못 이기는 점(로컬 0.847 vs 0.817) — 정점-무관/파랑전용 멤버로 부분 완화 기대 −0.003~−0.008(미검증).

## 재사용 대상

- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\data.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\features.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\event_phase.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\models.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\validation.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\corrected_repeated_forward.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\submission.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p3_wave\sequences.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\common.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\P3\run_submission.py
- C:\Users\cedis\PycharmProjects\PythonProject\configs\final_submission_20260905.json
- C:\Users\cedis\PycharmProjects\PythonProject\artifacts\p3\features_all20_v1

# P3 설계안 (Angle B: robust-ML / validation-first) — `src/ocean_v2/p3/`

## 0. 이번 정찰에서 직접 확인한 수치 (읽기 전용, 캐시 특징 `artifacts/p3/features_all20_v1` + 원본 train_wave.csv, LightGBM 180라운드 프로토타입)

| 항목 | 값 |
|---|---|
| 운영진식 onset anchor(정점별 최초 eligible + 78h) | **281** (G101/I80/S100); 분기 블록별 65/34/38/50/56/38 (2024Q3 I-ORS는 4개뿐) |
| onset anchor hs0 분위(0/10/25/50/75/90/100) | 1.50/1.51/1.53/1.60/1.81/2.46/4.34 — test 1.50/1.52/1.55/1.64/1.87/2.39/4.45와 일치; 상승 82%, 3h 전 <1.5 = 74% |
| 확장 onset 표면(1.5 통과 후 ≤1h 모든 anchor) | **2,795** anchor / 429 episode(블록별 649/325/415/471/605/330) |
| 평균 Δhs(리드 3/6/9/12/18/24) | onset: +0.02/−0.03/−0.04/−0.06/−0.19/−0.32 vs dense 전체: −0.09/−0.18/−0.27/−0.35/−0.53/−0.65 → **학습 분포 불일치 확인** |
| persistence RMSE(onset 281) | 0.595/0.755/0.812/0.866/0.928/0.931, pooled 0.8228; 블록별 0.715/0.725/0.871/0.766/1.047/0.718 |
| 가중 비교(4블록 182사례, 동일 purge) | 균일 0.8193 · 현행 exp(−0.45·) 0.8269 · **밀도비 가중 0.8092** · onset-부분집합만 학습 0.8276 · 리드별 개별모델 0.8278 · 절대타깃 0.8281 · sqrt 완화가중 0.8188 · 정점 특징 제거 0.8077(ext 0.7313, I-ORS 0.90↔S-ORS 0.80) |
| seed 분산 | 같은 설정 seed1 0.8092 / seed2 0.8187 (182사례) → 단일 seed 비교는 잡음 |
| 6블록·3seed OOF | onset 0.7781(−5.4%), ext 0.6998(−8.3%); 리드별 0.562/0.668/0.759/0.833/0.901/0.888; 정점 G 0.685(p 0.734) / **I 0.847(p 0.817, 악화)** / S 0.808(p 0.908) |
| 사후보정 | LOBO 리드별 bias 보정 0.814→0.839, affine 0.867 (**악화**, 블록마다 편향 부호 반전) / **slope-only shrink** LOBO s=0.68~0.80(6블록 모두 <1, ext로 적합) → onset 0.7781→0.7667, ext 0.6998→0.6946 (**개선, 두 표면 일치**) |

결론: (1) 검증면은 운영진 선정규칙 재현 anchor + 확장 onset 표면, (2) 학습은 dense anchor 전체 + 밀도비 가중(부분집합 학습 X), (3) 리드 pooled 잔차 타깃, (4) 보정은 절편 없는 slope shrink만 CV로 적합, (5) 모든 비교는 3seed 이상 앙상블로.

## 1. 패키지 구조 (`src/ocean_v2/p3/`, 신규·독립)

```
src/ocean_v2/common/          # 경로 해석(P3_DATA_DIR), sha256, JSON 원자적 쓰기, seed/thread 고정, 런타임 기록 (scripts/final_submission_20260905/common.py 골격 복사)
src/ocean_v2/p3/__init__.py
src/ocean_v2/p3/__main__.py   # CLI: python -m ocean_v2.p3 {audit|features|cv|train|predict|reproduce} --config configs/ocean_v2/p3.json
src/ocean_v2/p3/data.py       # p3_wave/data.py의 load_p3_data·audit_p3_data·build_training_grid·build_anchor_table 복사 + episode_id(eligible anchor 간격 >6h면 새 episode) + hours_since_below_1p5 + select_onset_cases(운영진식 greedy) + onset_extended_mask(≤1h)
src/ocean_v2/p3/features.py   # p3_wave/features.py summarize_context 복사(정렬 검증 완료) + onset 특징 6개 + compact 선택(models.compact_feature_columns 복사) + multiprocessing(8) 추출, anchor_id 정렬로 결정론 보장, 캐시 artifacts/ocean_v2/p3/features.parquet(SHA 기록)
src/ocean_v2/p3/weights.py    # 선정-정합 밀도비 가중 (아래 §3)
src/ocean_v2/p3/cv.py         # 6블록·purge·episode 제거·S1/S2 표면·클러스터 bootstrap·리포트(JSON+MD)
src/ocean_v2/p3/models.py     # 멤버 정의(§4): lgbm_pooled, cat_pooled, lgbm_nostation, lgbm_waveonly, lgbm_lead3; 결정론 설정 고정
src/ocean_v2/p3/calibrate.py  # LOBO slope shrink 적합·수용 규칙 → fitted_params.json
src/ocean_v2/p3/train.py      # features→cv→(멤버 선택은 config에 사전등록)→전체 anchor로 최종 fit→weights/*.cbm|*.txt + fitted_params.json + cv_report.json
src/ocean_v2/p3/predict.py    # test features→멤버 예측→등가 평균→hs0 + s·Δ→[0,30] clip→CSV(+SHA); p3_wave/submission.py validate/write 복사
src/ocean_v2/p3/reproduce.py  # 새 임시 폴더에 raw만 두고 train→predict 2회 실행, SHA 동일·소요시간 기록
configs/ocean_v2/p3.json      # 블록 경계, 가중 bin, 하이퍼파라미터, seed 목록, 멤버 목록, 수용 규칙 (LB 유래 상수 0, 각 값에 provenance 주석 필드)
scripts/final_submission_v2/P3/{run_submission.py, train_model.py, predict_submission.py}  # 기존 P3/run_submission.py 계약(contract.json, 01_data…07_source) 재사용, 내부는 ocean_v2.p3 호출
```
재사용하지 않음: `loss_router.py`, `persistence_shrink.py`, `corrected_fixed_long_shrink.py`, `final_inference.py`(동결 체인), 모든 kma_/era5_/chronos 모듈, α 결합. `artifacts/p3/features_all20_v1`는 개발 중 빠른 실험용으로만 쓰고 패키지는 raw에서 재생성한다(캐시 SHA 비교로 동일성 확인).

## 2. 데이터·anchor·검증면 (정확한 규칙)

- anchor: 정점별 10분 격자(파랑 20분 left-merge), `hs≥1.5` ∧ 6개 타깃(`hs.shift(-lead*6)`) 유효 ∧ 시작+48h 이후, 20분 stride → 24,360 (기존 코드 그대로).
- 파생 메타(anchor마다): `hours_since_below_1p5`(48h 내 마지막 hs<1.5 유한값까지 시간, 없으면 48), `hs_m1h/m3h`, `hs_min_6h/12h`, `episode_id`(정점 내 eligible anchor 시간 간격 >6h이면 새 episode; 총 429).
- **S1(1차 보고 표면)** = 운영진 규칙 재현: 정점별 시간순 first-eligible + 78h 간격 greedy를 전체 학습 기간에 한 번 적용(창 경계에서 리셋 안 함) → 281 anchor. 부표: 20/40/60분 오프셋 greedy 합집합(≈328)은 감도 확인용.
- **S2(선택 표면, 저분산)** = `hours_since_below_1p5 ≤ 1h`인 모든 anchor(2,795). 보고는 anchor 등가중과 episode 등가중 둘 다.
- 블록: 달력 분기 6개 (2024Q1, Q2, Q3, Q4, 2025Q1, 2025Q2[~06-25]). 양방향(각 블록을 나머지 5블록으로 학습).
- purge: 검증 블록 b의 학습 = 다른 블록 anchor 중 (i) 같은 정점의 모든 S1∪S2 검증 anchor와 |Δt| ≥ 78h **그리고** (ii) 검증 anchor가 속한 (station, episode_id)에 속하지 않는 것. 학습 anchor 수 블록별 ≈17.2k/23.0k/21.3k/18.8k/17.6k/22.9k.
- 진단 보조: 전방 전용 3fold(2024Q4·2025Q1·2025Q2를 과거 블록만으로 학습) 수치도 같이 출력하되 선택엔 쓰지 않음; 정점별·리드별·`wspd_valid_48h>0.5`(기상 존재) 부분집합별 RMSE; 편향(truth−pred) 리드별.
- 불확실도: 클러스터 = (block×station) 18개, 또는 episode 429개 → 쌍대 ΔRMSE의 bootstrap CI90(500회).
- **수용 규칙(config에 사전등록)**: 후보 채택 조건 = S2 pooled Δ<0 ∧ S1 pooled Δ<0 ∧ S2에서 6블록 중 ≥4 개선 ∧ 어떤 정점도 S2에서 +0.010 초과 악화 없음 ∧ 모든 비교는 3seed 이상 평균. 0.003 m 미만 차이는 "동등"으로 간주하고 단순한 쪽을 택한다.

## 3. 선정-정합 가중 (`weights.py`)

- 셀 = `hours_since_below_1p5` bin [0,1),[1,3),[3,6),[6,12),[12,48] × `hs0` bin [1.5,1.6),[1.6,1.8),[1.8,2.2),[2.2,3.0),[3.0,∞).
- 참조 밀도 p_ref = S1(281) 셀 분포(코드가 학습 데이터에서 계산), 원밀도 p_dense = 24,360 셀 분포. w = clip(p_ref/p_dense, 0.05, 5) / mean. ESS ≈ 24%(≈5,800 anchor), 가중의 71%가 통과 ≤3h anchor에 실림.
- 검증된 대안: 완화(√) 가중은 악화(0.8188), 넓은 clip [0.02,20]은 동등(0.8065/ext 0.7302) → 기본 [0.05,5] 유지. bin 경계·clip은 설계 하이퍼파라미터로 config에 명시(LB와 무관).
- 검증 fold 안에서도 p_ref는 해당 fold의 학습 anchor에서만 계산(누출 방지).

## 4. 특징과 멤버 모델

특징(anchor당, 사례 내부 정보만): 기존 `summarize_context`의 compact 591개(current, lag 1/3/6/12/24/48h, 창 3/6/12/24/48h의 mean/std/delta/slope/valid, hs/wspd/caph change) + **onset 특징 6개**: `hours_since_below_1p5`, `hs_m3h`, `hs_min_6h`, `hs_min_12h`, `hs0−hs_m3h`, `hs_run_above_1p5_h`(event_phase._run_duration 복사) + peak 특징 3개(`hs_peak_12h`, `hours_since_peak_12h`, `hs_drop_from_peak_12h`; event_phase._peak_features 복사). 절대시각·월 특징 없음. 결측은 NaN 유지.

타깃: 리드 pooled 잔차 `target_L − hs0`, 입력에 `lead_h`(수치)와 `station`(범주). 리드별 개별 모델·절대 타깃은 로컬에서 열세라 기본에서 제외(3h 리드만 예외, M5).

| 멤버 | 설정 | seed | 역할 |
|---|---|---|---|
| **M1 lgbm_pooled** | LightGBM L2, lr 0.03, 500 rounds, num_leaves 15, min_child 100, feature_fraction 0.3, bagging 0.8/1, λ2 8, max_bin 63, `deterministic=True, force_row_wise=True, num_threads=8` | 5 | 주 멤버(프로토타입 근거) |
| **M2 cat_pooled** | CatBoost CPU RMSE, depth 6, iter 900, lr 0.035, l2 8, random_strength 0.2, cat=[station, lead_h], `thread_count=8` | 3 | 현 계보 백본(Public 0.607 검증된 구조), 다양성 |
| M3 lgbm_nostation | M1에서 station 제거 | 3 | I-ORS 전이(로컬 I 0.92→0.90), S-ORS는 악화하므로 단독 사용 금지 |
| M4 lgbm_waveonly | M1에서 기상 파생열 전부 제거(hs/tp/hmax/wvdir/energy/steepness만) | 3 | 2024 I/S 기상 전무 구간과 기상 결측 사례 강건성(미검증, 사다리 항목) |
| M5 lgbm_lead3 | 3h 리드 전용 모델(로컬 3h 0.577 vs pooled 0.616) | 3 | 3h 행에서만 M1과 평균(미검증) |

결합: 통과한 멤버의 **등가 평균**(가중 적합 없음). 그다음 `pred = hs0 + s_g · Δ_ens`, s_g는 §5의 CV 산출물. 마지막 [0,30] clip(발동 안 함이 정상).
하이퍼파라미터 탐색은 M1에 대해 (lr 0.03/500, 0.06/180, num_leaves 15/31) 3점만 S2로 확인(로컬에서 더 큰 모델 0.8095 vs 0.8092로 무차이 → 탐색 최소화).
딥 시퀀스 모델(GRU/TCN on 289×10, `sequences.py` 재사용)은 과거 6종 전부 열세였으므로 **시간이 남을 때만** 멤버 후보로 시험; 채택 시 `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fp32, seed 고정, 가중치 동봉, 재현 허용오차(|Δ|<1e-4 m) 명시. 최종 후보는 CPU 트리 모델만으로 byte-exact를 목표로 한다.

## 5. CV 전용 보정 (`calibrate.py`)

- 형태: 절편 없는 slope `s`만. 두 그룹(단기 3/6/9h, 장기 12/18/24h)으로 `s_short, s_long` (LOBO 결과 단기 0.78~0.92, 장기 0.58~0.73로 그룹 차이가 일관됨). 전역 단일 s(0.68~0.80)도 산출해 비교하고, 두 그룹 버전이 S1·S2 모두에서 ≥ 전역 버전일 때만 채택.
- 적합: 최종 OOF(6블록, 멤버 앙상블)의 **S2 표면**에서 `s = Σ(Δ_pred·Δ_true)/Σ(Δ_pred²)` 최소제곱. 수용 검증은 LOBO(각 블록을 빼고 적합→해당 블록 평가)로 6블록 중 ≥5 개선 및 S1 개선일 때만 활성화, 아니면 s=1.
- 배포값은 전체 6블록 OOF로 한 번 적합해 `fitted_params.json`에 저장(코드 산출물; 학습 산출물 삭제 후 재실행 시 동일하게 재생성됨 → 재현 검증 4항 대응). 정점별 s, 리드별 절편은 **금지**(LOBO에서 악화 확인).
- 근거 요약: 로컬 OOF 편향이 블록마다 부호가 바뀌어 절편형 보정은 전이되지 않지만, 예측 Δ의 과신(slope<1)은 모든 블록에서 일관.

## 6. 결정론·런타임 (전체 재현 ≤ 6h 대비 여유 큼)

- 특징 추출: 24,360 anchor 단일 프로세스 309 s(캐시 manifest) → multiprocessing 8 → ≈60~90 s; test 200사례 3 s. 출력은 anchor_id 정렬 후 float32 고정.
- CV: 6블록 × [M1 5seed(≈20 s/fit at 500 rounds) + M2 3seed(≈90 s/fit CPU) + M3/M4/M5] ≈ 6×(100+270+150) s ≈ 50분. 최종 fit(전체 24,360): ≈10분. **총 ≈1.2 h**(CPU만, 8스레드). GPU 불필요.
- 결정론: LightGBM `deterministic=True, force_row_wise=True, num_threads=8, seed 고정`; CatBoost CPU `random_seed, thread_count=8` 고정(스레드 수가 바뀌면 부동소수 순서가 달라질 수 있으므로 README에 thread_count 명시); XGBoost 미사용. `reproduce.py`가 train→predict를 2회 실행해 CSV SHA 동일성을 검사하고 소요시간을 receipt에 기록.
- 상수 리터럴 감사: config JSON 외 코드에 수치 상수 금지(clip 0/30, 리드/정점 목록, 1.5 임계 제외). s·가중 참조밀도·멤버 목록은 모두 학습 코드 산출물.

## 7. 후보 사다리 (기대 Private 이득 순, 로컬 근거 표기)

0. **R0 참조(제출 안 함)**: 현 계보 O 레시피(CatBoost single, compact 591, exp(−0.45) 가중, 12/18/24h 0.2 persistence shrink)를 같은 6블록 S1/S2에서 재평가 → 새 후보의 "정직한 강도" 비교 기준(첫날 오전 필수).
1. **P3_v2_safe (폴백)**: M2 3seed, 현행 가중, onset 특징 포함, s=1. 목표: R0와 S1/S2에서 동등 이상. 완전 재현·결정론. 예상 Public ≈0.60~0.61.
2. **P3_v2_base (기본 제출안)**: M1(5seed)+M2(3seed) 등가 평균 + 밀도비 가중 + onset 특징 + CV slope shrink(s_short/s_long). 로컬 근거: 가중 −0.010, shrink −0.010, 앙상블 −0.003~−0.005(합 ≈−0.02~−0.025 로컬, 전이 후 −0.008~−0.015 기대).
3. **P3_v2_plus**: base + M3(정점-무관) [+ M4 파랑전용] — I-ORS 목표. 수용 규칙 통과 시에만. 기대 −0.002~−0.008(불확실).
4. **P3_v2_lead3**: plus + M5(3h 전용 모델을 3h 행에서 평균). 기대 −0.002~−0.004(불확실, 3h 리드만).
5. (시간 여유 시) 딥 시퀀스 멤버 — 기대 불명, 우선순위 최하.

## 8. 일정 (현재 09-05 오후, 모델 제출 마감 09-07)

**D0 09-05 (오후~밤, ≈8h)**
- 13:00–15:00 골격: common/data/features(병렬)/weights/cv 작성, 특징 캐시 생성 및 기존 캐시와 값 일치 확인(SHA·allclose).
- 15:00–17:00 cv.py 완성(S1/S2, purge, bootstrap, 리포트), R0 참조 실행, M1·M2 6블록 실행.
- 17:00–19:00 calibrate.py(LOBO s), P3_v2_safe·P3_v2_base 생성, 제출 validator, 결정론 2회 실행 SHA 비교.
- 19:00–21:00 **업로드 1**: P3_v2_safe(sanity: O 0.607±0.01 근처여야 함). **업로드 2**: P3_v2_base. 점수는 기록만 하고 파라미터 변경 금지. 남는 1회는 보류(예비).
- 밤: M3/M4/M5 CV 실행(무인, ≈40분).

**D1 09-06 (≈10h)**
- 09:00–11:00 M3/M4/M5 결과로 사다리 판정(수용 규칙 자동 적용), P3_v2_plus/lead3 생성.
- 11:00–13:00 `reproduce.py` 클린룸: 새 폴더에 패키지+raw만 복사 → train → predict → 업로드 CSV와 SHA 비교, 총 소요시간 기록(≤6h 명시).
- 13:00–16:00 패키지(01_data…07_source, contract.json, TRAIN/PREDICT notebook 얇은 래퍼, README: 환경·seed·thread_count·소요시간·SHA), 상수 리터럴 grep 감사, 외부데이터 0 확인.
- 16:00–19:00 **업로드 3**: P3_v2_plus(또는 수용 실패 시 base 재확인용 재생성본 — SHA 동일 확인). **업로드 4**: P3_v2_lead3(수용 시). 업로드 5는 예비.
- 최종 답안 선택 규칙(사전등록): CV 수용 규칙 통과 후보 중 S2 최저. Public 점수는 참고만 하되, CV 우선 후보가 Public에서 safe보다 **0.02 m 이상** 나쁘면(표본 SE 0.02~0.05 바깥) 후보의 결함 가능성을 조사하고 그래도 원인이 없으면 다음 순위 후보로 내려간다(Public으로 계수를 바꾸지 않음).

**D2 09-07 (오전)**
- 09:00–11:00 최종 지정 답안 = 패키지 클린룸 재생성본(SHA 일치) 업로드(필요 시 **업로드 6**), 규정 위반 KMA/ERA5·LB 적합 제출 8건 삭제 목록 사용자 전달, 답안 업로드 완료 확인 후 모델 최종 제출.

## 9. 위험과 대응

1. **I-ORS 약세**(test 35%): 로컬 S1에서 모델 0.847 vs persistence 0.817. 대응: M3/M4 멤버, 정점별 진단 필수 출력; 그래도 I-ORS가 persistence보다 나쁘면 I-ORS에 한해 등가 평균 대신 M3 비중을 높이는 것을 **사전등록 후보 하나**로만 검토(정점별 s 적합은 금지).
2. **계절 불일치**: test 여름 30%·겨울 39%, 로컬 여름 블록(2024Q3) I-ORS 4사례. 전이 불확실성을 기대값에 50% 할인으로 반영; 여름 블록 성능을 별도 보고.
3. **검증 잡음**: S1 281사례 SE≈0.004~0.005, seed 간 0.01 → 모든 후보 3seed 이상, S2(2,795)로 순위, 0.003 미만 차이는 무시.
4. **2024 I/S 기상 전무**: `_valid_` 특징이 연도 대리변수가 될 수 있음 → 기상 존재 부분집합 성능 별도 보고, M4(파랑전용)로 헤지.
5. **Public 잡음**(SE 0.02~0.05 m): 업로드는 sanity check 전용, 0.02 m 미만 차이에 반응하지 않음.
6. **재현 검증**: LB 유래 상수 0, fitted_params.json은 코드가 재생성, CPU 결정론 2회 SHA 일치, 인터넷 불필요(pip 의존성은 venv 기존 것), 6h 이내(≈1.2h). CatBoost thread_count 의존성은 README에 명시하고 clean-room에서 동일 설정 사용.
7. **폴백**: 수용 규칙 통과 후보가 없거나 클린룸 실패 시 P3_v2_safe; 그것도 실패하면 M1 단독(LightGBM만, 가장 단순한 결정론 경로).
