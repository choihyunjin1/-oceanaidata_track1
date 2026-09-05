# P2_SPEC — 소청초(S-ORS) 중간층 수온 복원 `src/ocean_v2/p2/` (설계 종합: design_panel/P2_design_*.md)

## 0. 문제와 목표
- `observations.csv` 789,408행(2024-01-01 09:00 ~ 2026-01-01 08:50 KST, 10분, 층 1~8; 2024는 7층, 2025는 8층). hidden = 2025-09-01~10-31 KST의 layer 2/3/4 temp·psal(26,352행 NaN). 채점 26,061키(`test_index.csv`), 지표 RMSE(℃), 제출 `station,layer,time,temp`.
- 공개층(공칭 m): 1=4.19, 5=19.59, 6=30.68, 7=39.45(2025), 8=49.35(2025) / 2024: 1=4.18, 5=19.15, 6=30.21, 7=49.05. 목표층 2=7.04, 3=9.44, 4=14.74(2025). 조직 baseline = `np.interp(nominal target, sorted nominal public depths, temps)`(clamp; 26,061행 정확 재현 확인됨).
- 정찰 사실: hidden 행 실제 depth 99.3% 존재; test 29.3%(10/14~10/31, 7,621행)는 T5가 연속 결측이면서 강성층(T1−T6 7.8℃) — 어느 해에도 없는 regime; 9월 psal_1 연속 결측 520h. 2024-09/10 가상 가림: nominal 보간 0.971, 실제수심 보간 0.880, T1 복사 0.593, LGBM(y−T1, T5 마스크 증강) 0.563, 3목표 평균+투영 0.558. 2025-11/12(완전 혼합): T1 복사 0.068 vs GBM 0.28~0.37(성층 과대예측 위험).
- 목표: 재생성 가능한 결정론 파이프라인, 안전 기준선 R0의 Public 기대 0.43~0.46(불확실), 사다리로 0.40~0.43.

## 1. 재사용(복사·개조; 원본 수정 금지)
- `src/p2_restore/data.py`(스키마 감사, hidden NaN assert) → 포팅. `src/p2_restore/submission.py` → 포팅.
- `src/p2_restore/profile_projection.py::project_profiles_vectorized`(endpoint envelope + 3층 PAVA) → 포팅하되 deep endpoint를 "T5 유효 시 T5, 아니면 첫 유효 심층 슬롯(T6→T7→T8)"으로 일반화.
- `src/p2_restore/features.py::_nearest_public_baseline`은 **재작성**(외삽 분기 제거, clamp 보간). 시간 인코딩(doy/hour/M2) 개념 재사용.
- `src/p2_restore/dynamic_sigmoid_profile.py`의 `effective_depth`, `joint_mask_target_intervals`, `build_public_features`는 함수 단위 참고; sigmoid 적합기는 **유계 특징으로만**(예측기 금지: 다른 fold에서 발산).
- `scripts/final_submission_20260905/P2/p2_pipeline.py`의 DeepSets(5,889 param)·`build_arrays`·`train_seed` → 사다리 멤버(CPU 결정론, 실제수심)로만.
- 금지: `bin17_anchor.csv`, `metric_geometry.py`, rank-1/quadratic/OAS-α 빌더, copula/PLS/DTW/joint_hydrographic, ERA5, 기존 3-블록 proxy comparator(`run_p2_*_v8/v13`).

## 2. 패키지 구조
```
src/ocean_v2/p2/__init__.py, __main__.py   # python -m ocean_v2.p2 {audit|features|cv|train|predict|all}
config.py       # physical_constants: public_layers, target_layers, slot_depths [4,19,30,39,49], hidden_window(KST), purge_days 7, mask_days 17, envelope, clip [-5,45]
data.py         # 10분 UTC 완전 격자(105,264 ts) wide 패널: temp/psal/depth/nominal by layer -> 수심 슬롯 매핑; hidden assert; 조직 baseline 재현
masking.py      # 학습 증강용 연속블록 마스크, CV용 테스트정합 T5 마스크, 목표층 joint mask
features.py     # (시각, 목표층) 행 특징 with_t5 / without_t5 두 벌
targets.py      # resid_lin / resid_T1 / frac 인코딩·디코딩
cv.py           # 블록·purge·아웃티지 아날로그 fold·지표·Composite·부트스트랩·게이트
models.py       # LightGBM(결정론) 래퍼·다중 seed; 선택 CatBoost; 선택 DeepSets(CPU)
stack.py        # OOF NNLS 스택(합=1, 균등과 50% 수축) -> derived_constants.json
postprocess.py  # envelope 클립 + 3층 PAVA(+선택 1h 평활)
train.py / predict.py / report.py
```

## 3. 데이터 표현
1. 완전 10분 UTC 격자(결측 ts 0 확인) × 층별 temp/psal/depth/nominal. 2026-01-01 padding 432행은 target NaN으로 자동 제외.
2. **수심 슬롯**: s4=L1, s19=L5, s30=L6, s39=2025 L7, s49=2024 L7/2025 L8. 트리가 "49 m"를 연도 무관 같은 변수로 보게 함. `deepT/deepZ` = 가장 깊은 유효 슬롯.
3. 목표 수심 `zt_real` = 실제 depth(유효·>0) else nominal — 학습·추론 동일 규칙. `base_nom`(조직 baseline, clamp), `base_real`(실제 수심 보간).
4. 학습 행 = truth 유효 ∧ 공개 temp ≥2 ∧ nominal 유효(≈166k행; 2024-05~12, 2025-04~08, 2025-11~12 전부). T1 결측 test 6행은 `baseline_interp.csv` 값 출력(예외 경로, 리터럴 아님).
5. 목표층 temp/psal은 `targets.py` 외 어디서도 읽지 않음(assert). `year`, `elapsed_days` 특징 금지.

## 4. 증강 `masking.py` (학습행만, `default_rng(seed)`)
- A1 T19(+S19) **연속 블록 아웃티지**: 시작점 무작위, 길이 ∈ {1,3,7,14,20}일 균등, 학습 ts의 ≈30%가 마스킹되도록 블록 수 결정(test 10월 17일 갭 재현). 원본 행 유지 + 마스킹 복제본(가중 0.5).
- A2 S4(psal_1) 연속 블록: 길이 ∈ {2,7,21}일, ≈40%.
- A3 단기 무작위: T19 5%, S19 10%, S30/S39/S49 각 3%.
- 마스킹 후 `T19_last/T19_next/hours_since/hours_until`과 시간문맥 특징을 재계산(학습에서도 수백 h staleness 등장).

## 5. 특징 `features.py` (행 = (시각, 목표층), ≈80~100열; `with_t5`/`without_t5` 두 벌)
1. 슬롯별: `T_s, S_s, z_s(실제->nominal 대체), presT_s, presS_s`(s∈{4,19,30,39,49}).
2. 목표: `layer(int)`, `zt_nom, zt_real, zt_dev`.
3. 기준: `base_nom, base_real, T1(=T_4), deepT, deepZ, deep_slot_id, frac_z=(zt_real−z4)/(deepZ−z4), scale=max(|T1−deepT|,0.5), grad=(T1−deepT)/(deepZ−z4)`, `Tref`(T5 유효 시 T5 else T6), `dref=T1−Tref`.
4. 차분/기울기: T1−T19, T1−T30, T1−T49, T19−T30, T30−T49, S4−S19, S4−S30, 슬롯별 `T_s−base_nom`, 구간 기울기.
5. **혼합층 깊이 교차점** h05/h10/h20(공개 프로파일 구간선형에서 T1−δ 최초 도달 깊이; 없으면 deepZ+5와 플래그), `zt_real−h05/h10/h20`, `zt−MLD` 계열.
6. 시간문맥(격자 기반, 중심창, 미래 허용): T1/T19/T30 각각 Δ(±1h, ±6h, ±12h, ±24h, ±72h), rolling std 6h/12h/24h, rolling mean 24h, min/max 3일, `(T1−T30)` rolling mean 24h·24h 변화, T1 3일 선형 추세; `T19_last, T19_next, T19_hours_since, T19_hours_until, T19_last−T30`.
7. 시각: doy sin/cos(1·2조화), hour sin/cos, 조석 M2(12.4206h)/S2(12.0)/K1(23.9345)/O1(25.8193) sin/cos(UTC 초 기준).
8. 가용성: `n_public_T, presT19, presS4, presS19`.
9. 유계 sigmoid 특징(선택): 격자 최소 SSE 해의 (center, width, R²)만.
`without_t5` 벌은 T5 파생 열 제거 + h*/act/base를 T5 제외로 재계산. 누출 검사: 목표층 값을 셔플해도 특징 행렬이 바이트 동일해야 통과.

## 6. 목표 파라미터화 `targets.py` (3개 모두 학습, 디코드 후 결합)
- `resid_lin`: y − base_nom. `resid_T1`: y − T1(T1 결측 시 base_nom). `frac`: (y − T1)/(deepT − T1), |deepT−T1|<0.5 행은 학습 제외, 디코드 시 clip[−0.25, 1.25] 후 T1 + f·(deepT−T1).

## 7. 모델 `models.py`
- LightGBM 회귀(L2): `learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.8, bagging 0.8/freq 1, lambda_l2=1.0, n_estimators=600`(격자 {31,63}×{100,300}×{400,800} CV 확인) + `determinism.lgbm_params(threads=4)`. seed 목록 CV 3 / 최종 10. 3목표 × seed. 학습 sample_weight = 원본 1.0, 증강 복제 0.5.
- **T5-부재 전문가**: `without_t5` 특징만으로 같은 3목표 학습(5 seed). 예측 시 `presT19==0` 행은 전문가, 아니면 본모델(결정론 라우팅).
- 선택 CatBoost(CPU depth 8, 1,500 iter, lr 0.05) 3목표(사다리 R6a). 선택 DeepSets(CPU deterministic, 실제수심, 5 seed, 60 epoch)(R6b).
- 저장: `models/lgbm_<target>_<expert|main>_<seed>.txt` 등.

## 8. CV `cv.py` (사전 등록)
- 블록(KST, 반열림): B1 2024-05/06, B2 2024-07/08, **B3 2024-09/10**(테스트 계절 아날로그), B4 2024-11/12, B5 2025-04/05(4월은 부분), B6 2025-06/07, B7 2025-08, **B8 2025-11/12**(T5 자연 부재·혼합). 학습 = 나머지 truth 행에서 블록 경계 ±7일 purge(±1,008 ts) 제거; 증강 복제본도 원본 시각 기준 동일 규칙.
- 각 블록에 두 변형 채점: `natural`, **`testmatched`**(블록 마지막 17일 T5 연속 제거 + 자연 결측 유지). 주 지표는 testmatched.
- 아웃티지 아날로그 fold: M1 = B3 평가행에 2024-10-08~10-31 T19/S19 마스크 + 2024-09-01~09-22 S4 마스크; M2 = B7 평가행에 2025-08-15~08-31 T19 마스크. 마스킹 행의 특징·baseline·시간문맥 재계산.
- 지표(전부 저장): 블록별·층별 RMSE, 편향, seed 편차, T5 유무별, 14일 bin별; pooled; **Composite = sqrt(0.71·pooled_testmatched² + 0.29·Outage(M1)²)**(0.29 = 7,621/26,061, 코드가 test_index에서 계산). KST-일 블록 부트스트랩 2,000회 CI90.
- 비교군(같은 표면에서 필수 보고): nominal 보간, 실제수심 보간, T1 복사, DeepSets 이식본 단독.
- **게이트**: Composite 개선 and B3·B8·M1 어느 것도 +0.01℃ 이상 악화 없음 and 어떤 블록도 +0.02℃ 이상 악화 없음; |Δ| < 0.003이면 단순한 쪽. 한계 명시: 2025-09/10 자체는 어느 fold에도 없음.

## 9. 스택·후처리
- `stack.py`: 층별 base learner(목표별 seed 평균, ℃ 디코드 후) OOF에 대해 비음수·합=1 NNLS(scipy), 행 가중 균등(사다리 R5에서 shift 가중 옵션); 최종 가중 = 0.5·NNLS + 0.5·균등. `derived_constants.json` 저장, predict가 적재. **R0 기본값은 균등 평균**.
- `postprocess.py`: 시각별 3층 예측을 [min,max](T1, deepT)로 클립 + 단조 PAVA(방향 = sign(deepT−T1); T5 결측 시 T6 endpoint). 선택 R7: 1h 중심 이동평균(같은 층).

## 10. 후보 사다리
| # | 후보 | 내용 |
|---|---|---|
| **R0** | `P2_v2_safe` | 슬롯·실제수심·시간문맥 특징 + 증강 A1~A3 + 3목표 LGBM×5 seed 균등평균 + T5 전문가 + envelope/PAVA |
| R1 | `P2_v2_outage` | 마스크 길이 분포 17~20일 강화 + staleness 특징 + T5 부재 행 T1 수축 s_L(testmatched OOF로 추정; 1이면 no-op) |
| R3 | `P2_v2_stack` | 10 seed + NNLS 스택(50% 수축) |
| R4 | `P2_v2_hpo` | 소격자 HPO |
| L3 | `P2_v2_mld` | 잠재 MLD 모델(학습행 8층 프로파일로 h05 라벨, 내부 14일 블록 5-fold OOF 스태킹) |
| L4 | `P2_v2_analog` | 다른 연도 같은 doy(±7일)의 r_clim/T1_clim/d15_clim/MLD_clim 특징(없으면 NaN; B3에서 평가 불가 → guard 블록으로만 판정) |
| R6 | `P2_v2_div` | CatBoost / DeepSets(CPU) 다양성 |
| R7 | `P2_v2_smooth` | 1h 시간 평활 |
최종 = 게이트 통과한 마지막 후보; R0와 차이가 seed 편차 이내면 R0.

## 11. 산출물·결정론·런타임
- `models/*.txt|.cbm|.pt`, `derived_constants.json`(스택 가중, s_L, c 상수, 특징 목록·해시, 라운드 수), `cv/cv_report.{json,md}`, `TRAINING_RECEIPT.json`. CSV `%.5f`.
- 예상: 특징 ≈1분, CV(8블록×2변형×3목표×3seed) ≈ 8~15분, 최종 학습(3목표×10seed + 전문가) ≈ 5~10분, 예측 <1분 → **LGBM 전용 ≈ 20~30분**; CatBoost +1h, DeepSets CPU +30~45분(시간 초과 시 축소).
- 결정론: seed·thread 고정, 증강 시작점 seed 고정, 2회 실행 SHA 동일. `validate.py`에 타 머신 허용오차(max|Δ|≤1e−3℃) 검증기.

## 12. 테스트 `tests/ocean_v2/test_p2.py`
hidden 26,352행 NaN assert; 목표층 값 셔플 시 특징 불변; 조직 baseline 26,061행 정확 재현(최대오차 0); purge 무겹침; 마스크가 학습행에만 적용; envelope/PAVA 단조성; validator 통과; 결정론.

## 13. 하지 말 것
anchor CSV·OAS α·rank-1·bin17 계보, LB 축 탐침, 2024 동일시각 값 직접 이식, 비제약 sigmoid 예측, 기존 proxy comparator, GPU 학습, 외부 자료.
