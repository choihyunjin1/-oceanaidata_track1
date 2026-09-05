# P2 robust-ML 재설계 계획 (src/ocean_v2/p2) — 정직한 블록 CV + 회귀-파라미터화 GBDT 앙상블 + 연속블록 결측 증강

> 설계 패널(읽기 전용 설계자) 산출물, 2026-09-05. 저장소 파일·데이터는 수정하지 않았음.

## 기대 효과

[현 챔피언 0.424 °C(Public) 대비] 공개 점수 기준 중앙 시나리오 −0.1~−0.6점(예상 RMSE 0.43~0.47, 불확실 구간 0.40~0.52). 근거: 정직한 2024-09/10 아날로그 fold에서 선형보간 0.971 → 0.558(frac 목표 + 3-목표 평균 + envelope/PAVA)이나, 테스트의 29%(10/14~10/31 T5 연속 결측 7,621행, |T1−T6| 7~8 °C로 성층 유지)는 어떤 fold도 정확히 재현하지 못해 L4 오차가 지배적 불확실성. 그러나 (i) 챔피언은 80%가 LB-적합 고정 CSV(스칼라 6개)이고 replay SHA 불일치라 재현검증 탈락 위험이 높아 위험조정 기대값은 강하게 양(+)이며, (ii) Private(같은 기간 70% 행)에서는 챔피언의 LB-적합 미세이득(≈0.03점)이 소멸하므로 실질 격차는 더 작다. [현 계보의 정직한 강도(DeepSets 단독, 공개 추정 0.437) 대비] 동등~우세 기대: −0.01~−0.03 °C ≈ +0.1~+0.4점 — 시간문맥(±1h/±6h/±24h/3일)+물리 파라미터화 목표(frac)+연속블록 결측 증강+투영이 동시 프로파일 전용 5.9k-param 모델보다 강하고 다중 seed로 분산을 줄이기 때문. 사다리 각 단계 기대(불확실 표시): R1 연속블록 T5/psal 증강 −0.02~−0.05(결측 29% 행에 집중, 분석 fold 마스크 평가 1.24→1.10 확인, 블록 마스킹은 미측정), R2 투영 −0.01~−0.03(fold 측정 0.613→0.558, 0.380→0.369), R3 10-seed+스태킹 −0.005~−0.02(이동 블록 seed 편차 ±0.05 측정), R4 실제 수심 −0.005~−0.02(선형보간에서 −0.05~−0.19 측정, GBDT 한계효과는 더 작음), R5 test-유사도 가중 학습 ±0.01(불확실), R6 CatBoost/DeepSets 다양성 −0.005~−0.015(불확실), R7 1h 시간 평활 −0.002~−0.01(불확실).

## 재사용 대상

- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\data.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\features.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\profile_projection.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\submission.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\model.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\normalized_curvature_residual.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\P2\p2_pipeline.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\P2\train_model.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\P2\predict_submission.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\build_official_final_submission_20260905.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\build_p2_seasonal_oas_submission_20260827.py
- C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\score.py

# P2 중간층 수온 복원 — "robust-ML / validation-first" 재설계 계획 (designer: robustml, 2026-09-05)

## 0. 요약
- 새 패키지 `src/ocean_v2/p2/`를 독립 구현: 원자료 → (연속블록 결측 증강) → 특징 → 정직한 purged 블록 CV(테스트 유사도 가중 + 결측-아웃티지 아날로그 fold) → 3-목표 파라미터화 LightGBM × 다중 seed → OOF-적합 스태킹 → envelope/PAVA 투영 → CSV. 전 과정이 원자료에서 재생성되고(LGBM 전용 ≤ 20분, CatBoost/DeepSets 포함 ≤ 2시간), LB 유래 상수 0, 결정론적(고정 seed, `deterministic=True`, 스레드 수 고정).
- 이번 정찰의 읽기전용 측정(purge 7일, 600-round LGBM 단일 seed):
  - 2024-09/10 (테스트 계절 아날로그, n=26,273): 선형보간 0.971 / T1 복사 0.593 / GBDT 잔차목표 0.69 / T1-잔차목표 0.66 / **frac 목표 0.580** / 3-목표 평균 0.613 / **3-목표 평균 + envelope·PAVA 0.558**. 9월만 1.289→0.885, 10월 0.495→0.299.
  - 2025-11/12 (T5 전결측·완전혼합, n=16,884): 선형보간 0.869 / **T1 복사 0.068** / GBDT 0.30~0.54(seed 편차 ±0.05!) / 3-목표 평균+투영 0.369. OAS형 계절 ridge는 지원 없는 결측패턴에서 폭주(4.2 °C) → 선형 성분은 GBDT 보조로만, 가드 필수.
  - 2024-09/10에서 T5를 마스킹한 평가: 선형보간 2.18 / T1 복사 0.59 / GBDT(증강 없음) 1.24 / GBDT(행 단위 35% 증강) 1.10. L4가 1.7~1.9로 지배.
  - 테스트 결측 구조: **10월 T5 결측은 단일 연속 갭 ~412 h(10/14~10/31, 주 42: 79%, 43·44: 100%)**, 9월 psal_1 결측은 연속 520 h. 2025-10은 T5 결측 주에도 |T1−T6| 6.9~7.9 °C(2024-10 말 1.6과 달리 **30 m까지 강성층 유지**). → 증강은 행 단위가 아니라 **연속 다일 블록** 마스킹이어야 하며, 10월 후반 L4는 본질적으로 가장 어려운 행.
  - hidden 행 실제 depth 유효 99.97%(층별 편차 sd 0.8 m, p05/p95 ±1.3 m); 선형보간을 실제 수심으로 계산하면 블록별 −0.05~−0.19.
  - 학습 truth 월: 2024-05(3.3k)~12(1.7k), 2025-04(1.3k)~08, 2025-11, 2025-12(1.3k) = 166,268행. 2025-11/12는 T5 100% 결측. psal_1 유효율은 학습기간 ≥0.9이나 테스트 9월 27.8% → 분포이동, 증강 필요.

## 1. 재사용 vs 재작성
| 기존 자산 | 판단 |
|---|---|
| `src/p2_restore/data.py` (스키마 감사, hidden 26,352행 NaN assert) | 로직 복사(패키지 자립성 위해 import 대신 포팅) |
| `src/p2_restore/features.py` `_nearest_public_baseline` | **재작성**: 감싸지 못하면 외삽하는 버그(정찰 취약점 #4) → 조직 `baseline_interp.csv`와 동일한 clamp 보간으로 교체(26,061행 최대오차 0 확인됨) |
| `src/p2_restore/features.py` 시간 인코딩(doy/hour/M2) | 개념 재사용 |
| `src/p2_restore/profile_projection.py` `project_profiles_vectorized` | 포팅(envelope+3층 PAVA). 단 deep 끝점을 T5→"첫 유효 심층 슬롯"으로 일반화 |
| `src/p2_restore/submission.py` `build_submission/validate_submission` | 포팅 |
| `src/p2_restore/model.py` LGBM 결정론 파라미터 | 출발점 |
| `scripts/final_submission_20260905/P2/p2_pipeline.py` DeepSets(5,889 param)+`build_arrays` | 선택 rung R6 base learner로만(CPU 결정론 학습) |
| `scripts/build_official_final_submission_20260905.py`, `scripts/final_submission_20260905/*` | 패키지 골격(contract.json/manifest/07_source) 재사용, P2 부분 교체 |
| `scripts/build_p2_seasonal_oas_submission_20260827.py` OAS | LB α 없이 "계절-국소 선형 base learner"로만 검토(R6 하위, 가드 필수). 기본 후보에서는 제외 |
| 조직 `score.py` | 가상 마스킹 블록의 로컬 채점기로 함수화 복사 |
| 사용 금지 | `bin17_anchor.csv` 등 고정 CSV, `metric_geometry.py`, rank-1/quadratic 빌더, 기존 3-fold comparator(`run_p2_*_v8/v13`) |

## 2. 패키지 구조 (`C:\Users\cedis\PycharmProjects\PythonProject\src\ocean_v2\p2\`)
```
src/ocean_v2/__init__.py
src/ocean_v2/p2/__init__.py
src/ocean_v2/p2/config.py          # 모든 상수(경로·블록·purge·seed·증강비율·하이퍼) + 출처 주석. LB 유래 값 0
src/ocean_v2/p2/data.py            # 로드, 스키마/hidden NaN assert, 완전 10분 UTC 그리드(105,264 ts) 위 wide 패널, 정규 수심 슬롯 매핑
src/ocean_v2/p2/masking.py         # 연속블록 결측 증강(T19/S19, S4) — 그리드 단계에서 적용
src/ocean_v2/p2/features.py        # (시각, 목표층) 행 특징 + 시간문맥(그리드 기반, 양방향)
src/ocean_v2/p2/targets.py         # resid_lin / resid_T1 / frac 인코딩·디코딩
src/ocean_v2/p2/cv.py              # 블록·purge·아웃티지 아날로그 fold·지표·OOF 저장
src/ocean_v2/p2/shift_weights.py   # 테스트 유사도 밀도비 가중(라벨 무관)
src/ocean_v2/p2/models.py          # LightGBM(결정론) / 선택 CatBoost / 선택 DeepSets(CPU) 래퍼, 다중 seed
src/ocean_v2/p2/stack.py           # OOF 기반 NNLS 스태킹(합=1, 균등가중으로 50% 수축) 저장/적재
src/ocean_v2/p2/postprocess.py     # envelope 클립 + 3층 PAVA, 선택 1h 시간 평활
src/ocean_v2/p2/train.py           # CV → 가중치·스택 → 전체 학습 → artifacts/ (모델, json, manifest)
src/ocean_v2/p2/predict.py         # artifacts + 원자료 → 제출 CSV(+SHA, 검증)
src/ocean_v2/p2/validate.py        # 조직 score.py 동등 검증 + 키 순서
scripts/ocean_v2_p2_run.py         # CLI: --stage {cv,train,predict,all} --config configs/ocean_v2_p2.json
configs/ocean_v2_p2.json           # 실행 설정(원자료 경로, 출력 경로, rung on/off)
tests/ocean_v2_p2/test_leakage.py  # hidden NaN, 목표층 temp/psal 특징 미포함, purge 정확성, 결정론(같은 seed 2회 실행 SHA 동일)
artifacts/ocean_v2_p2/<run_id>/    # models/*.txt, stack_weights.json, shift_weights.json, cv_report.json, oof.parquet, MANIFEST.json
submissions/claude_v2/P2_*.csv     # 후보 CSV + sha256 (Git 비추적)
```
원칙: 목표층(2/3/4) temp·psal은 `targets.py` 외 어디서도 읽지 않음(assert). `year`, `elapsed_days`는 특징 금지(연도 정체성 학습 방지).

## 3. 데이터 표현
- 완전 10분 UTC 그리드(2024-01-01 09:00 KST~2026-01-01 08:50 KST, 105,264 ts; 결측 ts 0 확인)에 층별 temp/psal/depth/nominal 패널.
- **정규 수심 슬롯**(연도별 layer↔수심 불일치 해소): s4 = L1, s19 = L5, s30 = L6, s39 = 2025 L7(2024엔 없음), s49 = 2024 L7 / 2025 L8. 트리 모델이 "49 m"를 연도 무관하게 같은 변수로 보게 함.
- 목표 수심: 실제 depth(결측 시 nominal) — 학습·테스트 모두 동일 규칙(hidden 실제 depth 99.97% 유효).
- 조직 baseline 재현: `np.interp(nominal target, sorted nominal public depths, temps)`(clamp) — 학습·추론 동일. 추가로 실제 수심 버전 `base_real`.
- 학습 행 = truth 유효 ∧ 공개 temp ≥ 2 ∧ nominal 유효 (166,268행; 2026-01 padding 자동 제외).

## 4. 증강 (`masking.py`, 학습 행만, `np.random.default_rng(20260905)`)
그리드 단계에서 마스킹 → 시간문맥 특징도 결측을 본다. 원본 행 유지 + 마스킹 복제본(가중 0.5).
- A1 T19(+S19) **연속 블록** 아웃티지: 학습 기간 내 시작점 무작위, 길이 ∈ {1, 3, 7, 14, 20}일에서 균등 추출, 학습 ts의 ≈30%가 마스킹되도록 블록 수 결정. (10월 테스트 갭 17일 재현)
- A2 S4(psal_1) 연속 블록: 길이 ∈ {2, 7, 21}일, ≈40%. (9월 테스트 520 h 갭)
- A3 단기 무작위 행 마스킹: T19 5%, S19 10%, S30/S39/S49 각 3%.
- 시각별 `T19_last/T19_next`, 경과·잔여 시간(h)은 마스킹 후 재계산 → 학습에서도 최대 수백 h staleness가 등장.

## 5. 특징 (`features.py`, 행 = (시각, 목표층))
1. 슬롯별(5): `T_s, S_s, Zreal_s(→nominal 대체), presT_s, presS_s`
2. 목표: `layer(int)`, `zt_nom, zt_real, zt_dev`
3. 기준: `base_nom, base_real, T4(=T1), deepT`(첫 유효 심층 슬롯 온도), `deepZ, deep_slot_id, frac_z=(zt_real−z4)/(deepZ−z4)`, `scale=max(|T4−deepT|,0.5)`, `grad=(T4−deepT)/(deepZ−z4)`
4. 차분: `T4−T19, T4−T30, T4−T49, T19−T30, T30−T49, S4−S19, S4−S30`, 슬롯별 `T_s−base_nom`
5. MLD 프록시: 공개 프로파일 구간선형 보간에서 T4−0.5 °C, T4−1.0 °C에 처음 도달하는 수심(없으면 deepZ), `zt_real − MLD`
6. 시간문맥(그리드, 중심창, 과거·미래 모두 — 복원 문제라 허용): T4/T19/T30 각각 Δ(±1h, ±6h, ±24h), rolling std 6h, rolling mean 24h, rolling min/max 3일; `(T4−T30)` rolling mean 24h와 24h 변화; `T19_last, T19_next, T19_hours_since, T19_hours_until, T19_last−T30`
7. 시각: doy sin/cos(1·2 조화), hour sin/cos, M2(12.42 h) sin/cos
8. 가용성: `n_public_T, presT19, presS4, presS19`
총 ≈ 80개. 제외: year, elapsed_days, 목표층 값.

## 6. 목표 파라미터화 (`targets.py`) — 3개 모두 학습, CV로 결합
- `resid_lin`: y − base_nom (디코드 +base_nom)
- `resid_T1`: y − T4 (T4 결측 시 base_nom) — 혼합 regime 기본값이 "T1 복사"가 되도록
- `frac`: (y − T4)/(deepT − T4), |deepT−T4| < 0.5인 행은 학습 제외, 디코드 시 clip[−0.25, 1.25] 후 T4 + f·(deepT−T4); deepT/T4 결측 시 base_nom — 성층 강도·심층 슬롯 종류에 불변
(기존 normalized-curvature 목표는 frac의 변형이므로 별도 유지하지 않음)

## 7. 정직한 CV 설계 (`cv.py`)
- 블록(KST, 반열림): B1 2024-05/06, B2 2024-07/08, **B3 2024-09/10**, B4 2024-11/12, B5 2025-04/05, B6 2025-06/07, B7 2025-08, **B8 2025-11/12**. 각 fold 학습 = 나머지 truth 행에서 블록 경계 ±7일 purge(그리드 인덱스 기준 ±1,008 ts) 제거. 증강 복제본도 같은 규칙으로 제거(원본 시각 기준).
- **아웃티지 아날로그 fold**(같은 fold 모델을 마스킹 복제 평가셋에 적용): M1 = B3 평가행 중 2024-10-08~10-31에 T19/S19 연속 마스킹 + 2024-09-01~09-22 S4 마스킹(테스트 갭 타이밍 모사); M2 = B7 평가행에 2025-08-15~08-31 T19 마스킹. 마스킹 행의 특징·baseline·시간문맥 전부 재계산.
- 지표(전부 cv_report.json에 저장): 블록별·층별 RMSE, 편향, seed 편차; pooled RMSE; **테스트 유사도 가중 pooled RMSE**(§8); **Outage RMSE**(M1 마스킹 창 행); 사전등록 헤드라인 **Composite = sqrt(0.71·W_pooled² + 0.29·Outage²)** (0.29 = 7,621/26,061, 자료 유래 상수).
- 채택 규칙(사전등록, 코드에 고정): rung은 Composite가 개선되고 B3·B8·M1 어느 것도 +0.01 °C 이상 악화하지 않을 때만 채택. 하이퍼 격자는 소형({num_leaves 31,63}×{min_child 100,300}×{rounds 400,800}), 3 seed.
- 한계 명시: 2025-09/10 자체는 어느 fold에도 없음(2024 가을은 30 m까지 혼합됐지만 2025-10은 성층 유지). B3는 배포 모델 학습에서 2024-09/10이 빠지므로 보수적 추정.

## 8. 테스트 유사도 가중 (`shift_weights.py`, 라벨 무관)
- 라벨 없는 상태 특징(doy 제외: T4, deepT, T4−deepT, T4−T30, T4−T49, presT19, presS4, T4 rolling std 6h, hour)으로 LightGBM 이진분류(test 시각 vs 학습 시각, seed 고정, 200 round, leaves 15) → 밀도비 w = p/(1−p)·(n_train/n_test), clip[0.2, 5], 평균 1로 정규화, 시각→행 매핑.
- 용도: (i) 헤드라인 지표 가중, (ii) 스태킹 가중치 적합의 행 가중, (iii) 선택 rung R5: 학습 sample_weight(0.5·1 + 0.5·w로 완화). 모두 배포 자료로 학습 시 재계산되는 artifact(`shift_weights.json`).

## 9. 모델 (`models.py`)
- LightGBM 회귀(L2): lr 0.05, num_leaves 63, min_child_samples 100, feature_fraction 0.8, bagging 0.8/freq 1, lambda_l2 1.0, 600 round(격자로 CV 확인). `deterministic=True, force_row_wise=True, num_threads=4(고정), seed=s`. seed 목록 [20260901..20260910](CV 3개, 최종 10개). 3 목표 × seed.
- 선택 R6a CatBoost(CPU, depth 8, 1,500 iter, lr 0.05, RMSE, `thread_count=4, random_seed=s`) 같은 3 목표.
- 선택 R6b 기존 DeepSets(`p2_pipeline.py` 아키텍처, frac 목표, CPU `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(8)`, 60 epoch, 3 seed; CV는 GPU로 빠르게, 최종은 CPU 결정론).
- 저장: `models/lgbm_<target>_<seed>.txt`(model_to_string), CatBoost `.cbm`, DeepSets `.pt`.

## 10. 스태킹·후처리
- `stack.py`: 층별로 base learner(목표별 seed 평균 예측, °C 디코드 후) OOF에 대해 비음수·합=1 최소제곱(scipy nnls + 정규화), 행 가중 = shift weight; 최종 가중 = 0.5·NNLS + 0.5·균등(수축). `stack_weights.json`으로 저장, predict가 적재. 안전 기본값(R0)은 균등 평균.
- `postprocess.py`: 시각별 3층 예측을 [min,max](T4, deepT)로 클립 + 단조(PAVA, 방향 = sign(deepT−T4)) — 정찰 포팅. 선택 R7: 디코드 예측의 1 h 중심 이동평균(같은 층, 그리드 기준) — CV로 채택 여부 결정.

## 11. 재현·결정론·런타임
- 전체 재생성 = `python scripts/ocean_v2_p2_run.py --stage all`: 특징 ≈ 10 s, CV(8 블록 × 3 목표 × 3 seed × ≈5 s + M1/M2 평가) ≈ 6~8 min, 최종 학습(3 목표 × 10 seed, 4 스레드) ≈ 5 min, 예측 < 1 min → **LGBM 전용 ≈ 20 min**. CatBoost 포함 시 +1~1.5 h, DeepSets CPU 3 seed +30~45 min. 6 h 한도 대비 여유 큼.
- 결정론: 모든 RNG 고정 seed, LightGBM deterministic + 스레드 수 고정, 증강 시작점은 seed 고정, CSV는 `float_format="%.5f"`로 기록(1e-5 °C 반올림, RMSE 영향 무시), 입력·출력 SHA256을 MANIFEST.json에 기록. 기계 간 부동소수 차이 대비 `validate.py`에 허용오차 검증(max|Δ| ≤ 1e-3 °C)도 제공, README에 명시.
- 규정: 배포 자료 외 입력 0, 사전학습 0, 상수는 `config.py`에 집중·주석(출처 = 사전 고정 또는 CV 산출물). LB 점수는 어떤 파라미터에도 쓰지 않음(코드 검토로 확인).

## 12. 후보 사다리 (기대 Private 이득 순, 전부 CV 사전등록 규칙으로 채택)
- **R0 SAFE 기본 후보**: 슬롯 표현 + 실제 수심 + 시간문맥 특징, 증강 A1~A3, LGBM 3 목표(resid_lin/resid_T1/frac) × 5 seed 균등 평균, envelope+PAVA. CV 목표치: B3 ≈ 0.56~0.61, B8 ≈ 0.35~0.40, M1 ≈ 0.9~1.1. 이것만으로 현 계보의 정직한 강도(동시 프로파일 DeepSets 단독)와 동등 이상 기대.
- R1 아웃티지 특화: A1 블록 길이 분포를 테스트 갭(17일)까지 확장, `T19_last` staleness 특징, resid_T1/frac 가중 상향(스택). 기대 −0.02~−0.05(29% 행 집중).
- R2 투영·클립 변형 확인(클립만/PAVA만/둘 다) — 이미 R0에 포함, 제거 시 악화 확인.
- R3 10 seed + NNLS 스택(50% 수축). 기대 −0.005~−0.02.
- R4 하이퍼 소격자(§7). 기대 −0.005~−0.015.
- R5 shift-가중 학습(0.5 완화). 기대 ±0.01(불확실).
- R6 CatBoost / DeepSets(CPU) 다양성 추가. 기대 −0.005~−0.015(불확실).
- R7 1 h 시간 평활. 기대 −0.002~−0.01(불확실).
- 하지 않음: OAS α·rank-1·bin 보정 등 LB 계보, 2024 동시각 값 이식, 계절 ridge 단독(결측패턴 폭주), 리더보드 축 탐침.
- **Fallback**: R0 CSV(clean). 최종 지정은 Composite 최우수 rung; R0와의 CV 차이가 seed 편차 이내면 R0 유지(단순성 우선).

## 13. 일정 (마감 09-07, 업로드 3회/일, LB는 sanity check 전용)
**Day 1 (09-05)**
- H0~H2: `config/data/masking/features/targets` + `test_leakage.py`(hidden NaN, 목표층 미포함, purge, 결정론).
- H2~H4: `cv/shift_weights/models`; R0 CV 실행(3 seed) → cv_report v0(블록·층·M1/M2·Composite).
- H4~H5: `train/predict/validate`; R0 최종 학습(5 seed) → `submissions/claude_v2/P2_R0_safe.csv` + SHA. **업로드 #1 = R0**(전이 sanity: R0 공개점수가 기록 목적; 0.55 초과 등 큰 이상이면 버그 탐색).
- H5~H8: R1~R3 CV, 채택 규칙 적용, cv_report v1.
**Day 2 (09-06)**
- AM: R4~R7(선택 R6는 시간 되면), 스택 가중 적합, 후보 CSV 생성. **업로드 #2 = CV-최우수 rung**. R0 대비 공개 Δ 부호만 기록(파라미터 조정 금지).
- PM: 패키지화(`scripts/final_submission_20260905` 골격에 `src/ocean_v2` 포함, contract.json·MANIFEST·README(런타임·결정론·허용오차)), 새 venv·인터넷 차단 상태에서 `--stage all` 재생성 → SHA/허용오차 대조. 재생성 산출과 #2가 다르면 원인 수정 후 **업로드 #3 = 재생성본**(같으면 슬롯 보존, 필요 시 R0 변형 sanity에만 사용).
**09-07**: 최종 답안 = CV 규칙이 고른 rung(재생성본과 SHA 일치 확인) → 답안 업로드 후 모델 패키지 제출(잠금 순서 준수). Fallback R0 보관.

## 14. 리스크와 대응
1. 10월 후반 T5 아웃티지(29%)에서 L4 오차 1~2 °C 가능(2025-10은 30 m까지 성층 유지, 2024 아날로그와 다름) → frac 파라미터화·연속블록 증강·M1 별도 보고; 기대치에 폭넓은 불확실 구간 명시.
2. 로컬→공개 전이의 역사적 부호 반전 → 같은 계열 nested 비교만, 사전등록 채택 규칙, 업로드는 sanity 3회 이내, 파라미터 재조정 금지.
3. 이동 블록 seed 편차 ±0.05 → 최종 10 seed, 스택 수축, worst-block 게이트.
4. 기계 간 부동소수 차이로 SHA 불일치 가능 → 스레드 고정·5자리 반올림·허용오차 검증기·README 명시.
5. 시간 부족 → Day 1 종료 시점에 R0(clean, 재현 가능)가 반드시 존재하도록 우선순위 고정.
6. "과다 상수 리터럴" 검증 항목 → 상수는 config.py 집중, 층·bin별 수작업 스칼라 0, 스택·가중치는 학습 실행이 산출하는 artifact.
7. Private 분할 부재 가능성(README에 분할 언급 없음) → 설계 영향 없음(정직 CV 선택은 동일).
