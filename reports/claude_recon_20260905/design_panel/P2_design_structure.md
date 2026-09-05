# P2 구조·물리 우선 재설계 — 위치-파라미터화 thermocline 복원 파이프라인 (src/ocean_v2/p2, 완전 재생성·LB 상수 0)

> 설계 패널(읽기 전용 설계자) 산출물, 2026-09-05. 저장소 파일·데이터는 수정하지 않았음.

## 기대 효과

Private 기대효과(현 챔피언 0.424 °C·28.01점 대비): 점수 자체는 중앙값 −0.1점(범위 −0.6 ~ +0.3점; 새 파이프라인 Public RMSE 중앙 추정 0.43, 범위 0.41~0.47)으로 거의 중립이지만, 현 챔피언은 80%가 LB-적합 고정 CSV라 재현검증(09-02 공지 1항·4항) 탈락 시 P2 0점(−28점) 또는 본선 탈락 위험이 있으므로 기대값 기준으로는 압도적으로 유리하다. 근거: (1) 로컬 LOBO에서 위치-파라미터화 GBDT가 2024-09/10 블록(테스트와 가장 유사, 같은 계절 학습자료 전무)에서 pooled 0.545(선형보간 0.97, T1복사 0.593)이며 실제 배포에서는 2024-09/10 동일계절 자료가 학습에 추가되므로 개선 여지가 큼; (2) 실제 depth 사용만으로 선형기준선이 층별 0.10 °C 개선(모델에서는 그 일부 −0.02~−0.05 기대); (3) 테스트 30%인 T5 결측 행에서 전용 패턴 모델이 마스킹 평가 0.813→0.712(−0.10, pooled −0.03 상당); (4) 현 계보에서 유일하게 Public 검증된 큰 효과(OAS 계절 조건부, −0.10)는 배포자료만으로 적합되는 성분이라 그대로 멤버로 포함(혼합비는 CV 적합); (5) 재현 가능한 DeepSets(단독 ≈0.437 추정) 멤버도 CPU 결정론으로 포함해 하한을 보강. 불확실: 2025-10월 부분혼합+T5결측 regime은 어느 fold에도 없어 L4 오차의 ±0.05 °C(±0.6점) 변동은 남는다. 시간문맥·파라메트릭 thermocline 특징의 추가 이득(각 −0.00~−0.03)은 CV 통과 시에만 채택.

## 재사용 대상

- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\data.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\features.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\profile_projection.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\submission.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\normalized_curvature_residual.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\build_p2_seasonal_oas_submission_20260827.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\P2\p2_pipeline.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\final_submission_20260905\common.py
- C:\Users\cedis\PycharmProjects\PythonProject\scripts\build_official_final_submission_20260905.py
- C:\Users\cedis\PycharmProjects\PythonProject\src\p2_restore\dynamic_sigmoid_profile.py

# P2 구조·물리 우선 재설계 계획 — `src/ocean_v2/p2/` (2026-09-05, 설계자 "structure")

> 원칙: 배포 자료 → 학습 → 가중치 → 예측이 코드로 완전 재생성. 리더보드 유래 상수 0. 모든 적합값은 CV OOF에서 코드가 산출한 `fitted_params.json`. 업로드는 sanity check 전용. 아래 수치는 이번 정찰에서 raw 자료를 읽기 전용으로 집계한 값(관측 원행 인용 없음).

## 0. 한 줄 요약

- **핵심 아이디어**: 복원 대상 3층(7.0/9.4/14.7 m)의 수온을 "공개 상·하 anchor 사이의 **위치(position)** p_L = (T4 − T_L)/(T4 − T_low)"로 파라미터화하고, p_L을 물리 상태 지표(thermocline 위치 지수·심층 형상·실제 depth·계절·표층 냉각 추세)로 예측한다. 혼합층이 목표 수심보다 깊으면 자동으로 "T1 복사"가 되고(2024-09/10에서 L2/L3 오차 0.13/0.32 °C), 성층 시에는 anchor 구간 안에서만 움직여 물리적으로 안전하다.
- **성분**: (C0) 결정론 물리 성분(T4-copy, 실제-depth 선형보간, 파라메트릭 thermocline 격자적합, 확장 PAVA/envelope 투영) + (C1) 위치-파라미터화 LightGBM 2종(T5 있음/없음 패턴별, CPU 결정론) + (C2) depth-registered 계절 OAS 조건부 평균 + (C3) v52 DeepSets CPU 5-seed 이식. 셀(층×T5가용)별 비음수 블렌드 가중은 CV OOF로 적합.
- **CV**: 테스트(2025-09~10, 전이 regime, T5 결측 30%)와 라벨 무관 상태분포를 맞춘 4개 자연 블록 + 2개 T5-마스킹 블록, 7일 purge, 양방향 full-history 학습(배포와 동일 구성).
- **런타임**: 안전 후보 ≈1 h, 전체 사다리 ≈2.5 h(CPU, 8코어) ≪ 6 h. GBDT/OAS 경로는 bit-exact, DeepSets는 CPU 결정론 + 소수점 4자리 반올림.

## 1. 정찰에서 직접 계산한 근거 수치 (읽기 전용 집계)

### 1.1 블록별 regime·가용률 (KST 블록, 10분 격자)

| 블록 | n | T5 유효 | \|T1−T5\| q10/50/90 | r=(T1−T5)/(T1−T30) q10/25/50/75/90 | 비고 |
|---|---:|---:|---|---|---|
| **TEST** 2025-09-01~10-31 | 8,784 | 70.1% | 0.06/0.92/4.12 | 0.00/0.035/0.14/0.34/0.54 | T5 결측 Sep 0.1%, **Oct 58.8%**; 결측 65 run 중 1개 run 2,471행(=17일)이 94% |
| B24SO 2024-09-01~10-31 | 8,784 | 99.2% | 0.01/0.48/4.79 | −0.03/0.00/0.08/0.30/0.53 | 테스트와 가장 유사(전이) |
| B25JA 2025-07-16~08-31 | 6,768 | 100% | 2.87/5.94/9.59 | 0.27/0.37/0.50/0.62/0.72 | 강성층(9월 초 유사) |
| B25ND 2025-11-01~12-31 | 8,784 | **0%** | – | – (T1−T30 median 2.05) | 혼합+T5 전부 결측(10월 말 유사) |
| B25MJ 2025-05-01~06-30 | 8,784 | 94.6% | 0.9/4.94/7.98 | 0.54/0.75/0.90/0.97/0.99 | 얕은 thermocline |
| B24JA 2024-07-01~08-31 | 8,928 | 97.3% | 0.88/3.5/7.51 | – | 보조 |

### 1.2 성층 시(|T1−T5|>1) 층별 위치 |p_L| 중앙값 [q25,q75]

| 블록 | L2 (7 m) | L3 (9.4 m) | L4 (14.7 m) |
|---|---|---|---|
| B24SO | 0.002 [0.00,0.02] | 0.014 [0.00,0.07] | **0.192** [0.05,0.43] |
| B25JA | 0.018 [0.00,0.05] | 0.073 [0.02,0.18] | **0.505** [0.31,0.67] |
| B24JA | 0.030 | 0.117 | 0.490 |
| B25MJ | 0.049 | 0.142 | 0.708 |

→ L2는 항상 ≈T1, L3도 거의 T1; **오차의 본체는 L4의 위치**이며 여름 0.5~0.7 → 가을 0.19로 이동(혼합층 심화). 이 이동을 설명하는 공개 상태 지표(r 지수, T30/T49 형상, 표층 냉각 추세, 계절)가 설계의 중심.

### 1.3 결정론 하한과 파라미터화 비교 (LOBO, 학습 = 해당 블록±7일·가림 구간 제외 전부)

| 블록 | 선형(nominal) | 선형(**실제 depth**) | T1 복사 | 잔차-vs-선형 LGBM(+doy) | **위치-파라미터화 LGBM(+doy)** |
|---|---|---|---|---|---|
| B24SO L2/L3/L4 (pooled) | 0.462/0.917/1.333 (0.97) | 0.368/0.815/1.237 (0.88) | 0.135/0.322/0.968 (**0.593**) | 0.22/0.385/0.928 (0.593) | **0.177/0.339/0.864 (0.545)** |
| B25JA | 1.041/1.672/1.815 (1.50) | 0.985/1.622/1.767 | 0.46/1.15/3.74 (2.28) | 0.447/0.869/1.455 (1.013) | 0.412/0.834/1.467 (1.004) |
| B25ND | 0.352/0.651/1.311 (0.85) | 0.313/0.624/1.289 | **0.025/0.046/0.106 (0.068)** | 0.685/1.379/1.103 (1.094) | 0.06/0.188/0.567 (0.347) |
| B25MJ | 0.763/1.202/1.264 | 0.744/1.186/1.266 | 0.73/1.50/3.76 | – | 0.838/1.383/1.769 (1.38) |

- 실제 depth만으로 선형기준선이 층당 ≈0.10 °C 개선(B24SO). hidden 행 실제 depth는 99.3% 존재.
- 잔차-vs-선형 파라미터화는 혼합+T5결측(B25ND)에서 붕괴(0.99~1.09 vs T1복사 0.07). 위치 파라미터화는 같은 조건에서 0.347, 여전히 T1복사보다 나쁨 → **혼합 regime에서는 T1-copy 멤버가 블렌드에서 살아남아야 함**(셀별 가중의 근거).
- T5 완전 마스킹 평가(테스트 10월 모사): B24SO 0.813(T5 학습 모델) → **0.712**(T5 마스킹 학습) ; B25JA 1.55 → 1.35. 마스킹 증강을 단일 모델에 섞으면 T5 있는 행이 소폭 악화(0.545→0.552)하고 B25ND가 크게 악화(0.347→0.584; "T5결측⇒여름형" 허위연관) → **패턴별 2모델(A: T5 사용, B: T5 완전 배제)** 로 분리.
- 시간 문맥(±1/6/24 h 차분, 이동평균/표준편차)은 잔차 모델에서 B25ND 0.989→0.751, B25JA 1.013→0.973 개선, B24SO 0.593→0.665 악화 → 물리적으로 의미 있는 소수 특징(표층 3일 냉각 추세, 24 h 평균 r)만 사다리 항목으로 CV 검증.

## 2. 패키지 구조 `src/ocean_v2/p2/` (자체 완결, `p2_restore` import 없음)

```
src/ocean_v2/common/          # P1/P3와 공유: paths.py(P2_DATA_DIR), hashing.py(sha256), seeds.py, runtime.py(소요시간 기록), validator.py
src/ocean_v2/p2/
  __init__.py / __main__.py   # CLI: python -m ocean_v2.p2 {cv|train|predict|package|verify} --config configs/ocean_v2/p2.json
  config.py                   # dataclass ← configs/ocean_v2/p2.json (블록 경계·purge·격자·seed·트리수·λ·반올림 자릿수) — 전부 사전등록 상수, 점수 유래 0
  data.py                     # load/audit(재사용: p2_restore/data.py), hidden 26,352행 NaN assert, depth-registered 10분 패널 구축
  features.py                 # 시각별 공개 특징 + 시간문맥 + 목표행 특징 (아래 §3)
  physics.py                  # T4-copy, 실제-depth 선형보간, 위치 인코딩/디코딩, thermocline 격자 적합, 확장 PAVA/envelope
  oas.py                      # depth-registered 계절 OAS 조건부 평균 (재사용: build_p2_seasonal_oas_submission_20260827.py의 conditional_predict 로직)
  gbdt.py                     # LightGBM 위치모델 A/B (+선택 CatBoost), 결정론 설정, seed 평균
  deepset.py                  # v52 DeepSets 이식(재사용: final_submission_20260905/P2/p2_pipeline.py 클래스·train_seed·predict_model), CPU 결정론
  cv.py                       # 블록/purge/fold, 테스트-정합 가중, 지표(pooled·층·셀·fold), KST-일 bootstrap, OOF 저장
  blend.py                    # 셀별 NNLS(+등가중 ridge prior) → fitted_params.json
  train.py                    # 전체 라벨행으로 최종 적합 → artifacts/ocean_v2/p2/weights/
  predict.py                  # test_index → 특징 → 멤버 예측 → 블렌드 → 투영 → round(4) → CSV + receipt.json
  validate.py                 # 스키마/키순서/유한/범위/hidden NaN/상수리터럴 감사(ast로 float 리터럴 목록 출력)
  package.py                  # 01_data…07_source 골격(재사용: build_official_final_submission_20260905.py scaffold·write_problem_docs, common.py)
configs/ocean_v2/p2.json
artifacts/ocean_v2/p2/{cv/oof.parquet, cv/report.json, weights/, fitted_params.json}
submissions/claude_v2/p2/<candidate>/{P2_submission.csv, sha256.txt, cv_report.json, validator.json}
```

재사용/폐기 결정:
- **재사용(복사·단순화)**: `src/p2_restore/data.py`(로더·감사), `features.py`(wide pivot·달력 특징), `profile_projection.py`(PAVA 3점·envelope; T30 fallback으로 확장), `submission.py`(build/validate), `normalized_curvature_residual.py`의 `compute_profile_scale/encode/decode`(DeepSets 멤버 전용), `scripts/build_p2_seasonal_oas_submission_20260827.py`의 `build_panel/conditional_predict`(층번호→depth-registered로 교체), `scripts/final_submission_20260905/P2/p2_pipeline.py`(DeepSets 클래스·학습·추론·domain_balanced_weights), `scripts/final_submission_20260905/common.py`(sha/contract), `scripts/build_official_final_submission_20260905.py`(패키지 골격; `build_p2`는 새로 작성), `src/p2_restore/dynamic_sigmoid_profile.py`(시그모이드 정의 참고만; scipy least_squares 대신 벡터화 격자 적합으로 재구현).
- **폐기**: `bin17_anchor.csv`와 그 계보(U/OAS α=0.5/rank-1/vertex/bin17), `metric_geometry.py`, `scripts/run_p2_*`의 comparator·action cap, 모든 `configs/experiments/p2_*` 점수 유래 스칼라.

## 3. 데이터·특징 설계

### 3.1 depth-registered 패널 (`data.py`)
- 10분 격자 105,264 시각 × 물리 수심 열: `T4`(layer1), `T20`(layer5: 19.15 m/19.59 m), `T30`(layer6), `T39`(2025 layer7만), `T49`(2024 layer7 / 2025 layer8), 목표 `T7,T9,T15`(layer2/3/4). psal·실제 depth·nominal 동일 구조. 2024 layer7(49 m)과 2025 layer7(39 m) 혼동 제거(기존 OAS의 결함 수정).
- 실제 depth `z_* = depth if finite else nominal`(목표층 포함; hidden 행 99.3% 실제 depth 존재).
- 누출 가드: 가림 격자 26,352행 temp/psal NaN assert; 답안 파일 읽기 코드 부재; 학습 행 선택은 "목표 유효 ∧ 같은 시각 공개 수온 ≥2"(조직 채점 규칙과 동일).

### 3.2 시각별 공개 특징 (`features.py`, 라벨 무관)
1. 수온·염분·실제 depth: `T4,T20,T30,T39,T49`, `S4,S20,S30`, `z4,z20,z30,z49` + presence 비트.
2. 하부 anchor: `T_low = T20 if valid else T30`, `z_low` 대응, `den = T4 − T_low`, `t5miss` 플래그.
3. 물리 지표: `r = (T4−T20)/(T4−T30)`(T20 결측 시 NaN), `r6d = (T4−T30)/(T4−T49)`, `s6 = T4−T30`, `s56 = T20−T30`, `s6d = T30−T49`, 절대 수준 `T4, T30`.
4. **파라메트릭 thermocline 격자 적합**(`physics.py`, 결정론): 공개점 ≥3개일 때 `T(z) = Tb + (Ts−Tb)·σ((zc−z)/w)`를 격자 zc∈{2,3,…,48 m}, w∈{0.5,1,1.5,2,3,4,6,8 m}에서 (Ts,Tb)는 선형 최소제곱 폐형해로 풀고 최소 RSS 선택(105k×376 격자, numpy 벡터화 수 초). 산출: `zc_hat, w_hat, Ts, Tb, rss`, 목표 수심의 적합값 `fit_T7, fit_T9, fit_T15`, 목표 수심 대비 위치 `(zc_hat − z_t)/w_hat`. 사다리 항목(§8 L3).
5. 시간 문맥(양방향 허용, 복원 문제): `T4, T_low, T30`의 ±1 h/±6 h/±24 h 차분, 24 h 중심 이동평균−현재, 6 h 이동표준편차, **T4 3일 중심 기울기(표층 냉각→혼합층 심화)**, r의 24 h 중심 평균, T20 결측 시 ±24 h 내 최근접 유효 `T20_near`와 나이(테스트 결측 run 중앙 2스텝이라 단기 gap에 유효). 사다리 항목(§8 L2).
6. 달력: `doy sin/cos`(1차), `hour sin/cos`, `M2 sin/cos`.

### 3.3 목표행 특징
- `layer` one-hot, `z_t`(실제), `z_nominal`, `zfrac = (z_t − z4)/(z_low − z4)`.

## 4. 모델 설계

### C0 결정론 물리 성분 (`physics.py`)
- `m_copy = T4` (혼합 regime prior).
- `m_lin_real`: 실제 depth 선형보간(공개 유효점 정렬 후 `np.interp`; T20 결측 시 T30이 하부 점).
- 위치 인코딩 `pos = (T4 − T_L)/den`, 디코딩 `T̂ = T4 − clip(p̂, −0.05, 1.05)·den`; |den|<0.05 행은 학습 제외(예측은 ≈T4).
- 확장 투영: `profile_projection.py`의 PAVA(3점 단조)+envelope를 하부 endpoint `T_low`(T20 없으면 T30)로 확장 → T5 결측 행에서도 no-op이 아니게 함. 최종 블렌드 뒤에 적용.

### C1 위치-파라미터화 GBDT (`gbdt.py`) — 주 학습기
- 목표 `pos`(clip −0.2~1.2), 표본 가중 `min(den², 100)`(위치 손실 ≈ 온도 손실), 층 pooled 단일 모델(층 one-hot·z_t 포함).
- **패턴 A**(T20 유효 행 학습, T20 관련 특징 포함) / **패턴 B**(전 행을 T20 완전 배제로 학습: `T_low=T30`, r·s56·S20·T20_near 제거). 추론: 행의 T20 유효 여부로 A/B 선택. 자연 T5 결측 행(2025-11~12)은 B에만 포함.
- LightGBM: `objective=regression(L2)`, `num_leaves=31`, `learning_rate=0.03`, `min_child_samples=200`, `subsample=0.8/freq 1`, `colsample_bytree=0.8`, `reg_lambda=10`, 트리 수는 CV OOF 평균 곡선의 최소점을 200 단위로 고른 값(fitted_params에 기록; 예비 1,200), `deterministic=True, force_row_wise=True, num_threads=8`, bagging seed 3개 평균(seed 20260905, 06, 07). Bit-exact.
- 선택 멤버: CatBoost CPU(`depth 6, l2_leaf_reg 10, iterations CV`, `thread_count=8, random_seed`) 동일 목표 — 사다리 L4.

### C2 depth-registered 계절 OAS (`oas.py`)
- x = [T4,S4,T20,S20,T30,S30,T49,S49(가능 시), doy 1~4차 조화], y = [T7,S7,T9,S9,T15,S15]; 14일 bin·±60일 창·OAS 수축 공분산·결측 패턴별 조건부 평균(기존 로직 그대로, 열만 depth-registered). 실제 depth는 미사용(선형 조건부라 별도 보정 불필요; GBDT가 담당).
- CV에서는 held-out 블록±purge를 창에서 제외. **주의**: B24SO fold에서는 같은 계절 자료가 전무해 OAS가 불리(정직한 비관); 배포 시에는 2024-09~10이 창 안에 있어 Public에서 검증된 큰 효과(−0.10)가 기대됨. 이 비대칭은 블렌드 적합에서 fold 가중(§5.4)으로 다룬다.

### C3 DeepSets v52 CPU 이식 (`deepset.py`)
- 아키텍처·손실(SmoothL1 + 0.01 입력기울기 L2)·60 epoch·batch 4096·AdamW는 `p2_pipeline.py` 그대로. 변경: (a) baseline = 실제-depth 선형보간(`m_lin_real`, 학습·추론 동일 정의 → 기존 학습/추론 불일치 버그 제거), (b) 토큰 depth 채널에 실제 depth, 목표 depth도 실제, (c) profile_scale 하부 endpoint T30 fallback, (d) **CPU 전용** `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(8)`, seed 5개(20260901~05) 평균, (e) 학습 행 2024-05-01 이후 full-history + domain_balanced_weights 유지. 예상 seed당 CPU ≈1~2분.

### 블렌드 (`blend.py`)
- 셀 c ∈ {L2,L3,L4} × {T20 유효, 결측} (6셀). 멤버 k ∈ {m_copy, m_pos(A|B), m_oas, m_ds, (m_cat)}. OOF에서 `min_w ||Σ_k w_k m_k − y||²_W + λ||w − 1/K||²`, w ≥ 0, Σw = 1 (NNLS + 등가중 ridge, λ = 0.05·Σ_i W_i 사전등록). W = fold·행 가중(§5.4).
- 산출 `fitted_params.json`: 셀별 w, 트리 수, 반올림 자릿수(4), 투영 on/off. 예측 = Σ w_k m_k → 확장 PAVA/envelope → clip[−5,45] → round(4).

## 5. 정직한 CV 설계 (`cv.py`, 사전등록 → `configs/ocean_v2/p2.json`)

### 5.1 블록(KST 반개구간)과 학습 범위
| fold | 평가 블록 | purge | 학습 행 |
|---|---|---|---|
| F1 | 2024-09-01 ~ 2024-11-01 | 양쪽 7일(1,008스텝) | 블록±purge·가림구간 제외 **전 라벨행**(양방향 full-history = 배포 구성) |
| F2 | 2025-07-16 ~ 2025-09-01 | 7일 | 〃 (2024-07~08 동일계절 포함) |
| F3 | 2025-11-01 ~ 2026-01-01 | 7일 | 〃 (2024-11~12 포함; T5 자연 결측 셀) |
| F4 | 2025-05-01 ~ 2025-07-01 | 7일 | 〃 (저가중) |
| F1m | F1 평가행의 T20을 **전부 마스킹**(패턴 B 경로) | – | F1 모델 재사용(평가만) |
| F2m | F2 동일 | – | F2 모델 재사용(평가만) |

- 평가 행 = 목표 유효 ∧ 같은 시각 공개 수온 ≥2(조직 규칙). fold당 ≈17k~26k행.
- 2024-11~12·2024-07~08·2024-05~06은 학습 전용(2024 기하는 depth-registered 패널이 흡수).

### 5.2 지표
- 1차: **테스트-정합 가중 pooled RMSE**(§5.4). 2차: fold별·층별·셀별 RMSE, T1복사·실제-depth 선형·조직 nominal 선형 대비 Δ, KST-일 블록 bootstrap 1,000회 CI90. 상태-셀 표(r 5구간 × T5가용)를 항상 보고.
- action cap·Δ-vs-comparator 방식은 사용하지 않음(절대 RMSE만).

### 5.3 선택 규칙(사전등록)
후보가 incumbent를 대체하려면 (i) 가중 pooled RMSE 개선, (ii) F1·F2·F3 중 2개 이상에서 비악화(Δ ≤ +0.01), (iii) F1m·F2m 비악화, (iv) bootstrap CI90 상한 < 0. 실패하면 사다리 항목 폐기(LB 미업로드).

### 5.4 테스트-정합 fold/행 가중 (라벨 무관)
- 상태 셀: `r` 구간 {<0.02, 0.02–0.1, 0.1–0.3, 0.3–0.6, ≥0.6} × T20 {유효, 결측}(결측 시 r6d 구간 {<0.5, 0.5–0.85, ≥0.85}로 대체). 테스트 시각의 공개 상태로 `P_test(cell)`, CV 풀링 행으로 `P_cv(cell)` 계산 → 행 가중 `w = clip(P_test/P_cv, 0.2, 5)`. 이 가중을 지표·블렌드 NNLS·트리 수 선택에 공통 사용. 테스트 분포 근거: r q50 0.14, T5 결측 29.9%(§1.1).

### 5.5 현 계보의 정직한 강도 재측정
같은 surface에서 (a) 조직 nominal 선형, (b) T1 복사, (c) v52 DeepSets(원 레시피, fold별 재학습, 3 seed), (d) 기존 층번호 OAS를 평가해 표로 남김 → 새 후보와 동일 기준 비교(내부 report에 포함, 패키지 문서에도 요약).

## 6. 캘리브레이션·파라미터 적합 규칙 (LB 금지)
- 적합 대상: 블렌드 가중(6셀×K), LightGBM/CatBoost 트리 수, (선택) 위치 클립 범위 검증. 모두 `python -m ocean_v2.p2 cv`가 OOF에서 산출 → `fitted_params.json`(값·근거 fold·bootstrap CI 기록).
- 코드 내 리터럴은 물리/설계 상수만(수심 격자, 창 길이, purge 7일, seed, epoch, λ, 반올림 4) — `configs/ocean_v2/p2.json`에 각 상수의 근거 문장을 함께 적음(재현검증 1항 대비). `validate.py --audit-literals`가 소스의 float 리터럴을 열거해 config 밖 상수를 0으로 확인.
- 리더보드 점수는 후보 CSV의 "예상 범위 이탈" 감지에만 사용(예: CV 함의 대비 +0.05 °C 이상 악화 → 버그 조사). 점수에 따라 가중·bin·α를 바꾸는 행위 금지.

## 7. 결정론·런타임

| 단계 | 결정론 | 예상 시간(7800X3D 8코어) |
|---|---|---|
| 패널·특징·격자 적합 | numpy 결정론 | 3분 |
| OAS 6 bin × (4 fold+최종) | numpy/LAPACK 결정론(동일 numpy 버전 기록) | 1분 |
| LightGBM 2패턴×3 seed×(4 fold+최종) = 30 fit | bit-exact(`deterministic=True, force_row_wise=True, num_threads=8`) | ≈40분 |
| CatBoost(선택) 동일 구성 | thread_count 고정 시 결정론 | ≈40분 |
| DeepSets CPU 5 seed×(4 fold+최종) | 동일 머신 bit-exact; 타 머신 1e-6 수준 차이 가능 → round(4)+허용오차 문서화 | ≈50분 |
| 블렌드·투영·예측·검증 | 결정론 | 2분 |
| **합계** | | 안전 후보(C0+C1+C2) ≈50분, 전체 ≈2.5 h ≪ 6 h |

- 재현 검증 대응: `verify` 서브커맨드가 (1) 가중치 삭제 → 재학습 → 예측 → 제출 CSV와 비교(GBDT/OAS 경로 SHA 일치, DeepSets 포함 시 RMS 차이 ≤1e-4 °C 보고), (2) 소요시간·환경(파이썬·패키지 버전)·seed를 receipt에 기록. 조직이 SHA 완전일치를 요구하는 상황이면 **S0-det(DeepSets 제외)** 를 최종으로 지정할 수 있도록 두 후보 모두 패키지에 포함.

## 8. 후보 사다리(기대 Private 이득 순)와 업로드 계획

| 단계 | 후보 | 내용 | 기대 효과(°C, 정직 추정) | 확신 |
|---|---|---|---|---|
| **S0-det** | `P2_v2_safe_det` | C0 + C1(LGBM A/B) + C2(OAS) 셀별 블렌드 + 확장 투영. bit-exact | Public 0.43~0.48 추정(현 0.424 대비 0~+0.05) | 중 |
| **S0** | `P2_v2_safe` | S0-det + C3 DeepSets CPU 5-seed 멤버 | S0-det 대비 −0.00~−0.02 | 중 |
| L1 | `+실제 depth 검증` | (S0에 이미 포함) nominal 대비 ablation 표만 산출 | −0.02~−0.05 (L3/L4) | 중상 |
| L2 | `P2_v2_ctx` | 시간문맥 특징(3일 냉각 추세, 24 h r 평균, T20_near·나이, ±6/24 h 차분) | −0.00~−0.03 (F3형 셀에서 큼, F1 악화 가능) | 낮 |
| L3 | `P2_v2_thermo` | 파라메트릭 thermocline 격자 적합 특징(zc_hat, w_hat, fit_T_L) | −0.00~−0.02 (L4) | 낮 |
| L4 | `P2_v2_cat` | CatBoost 멤버 추가(멤버 다양성) | −0.00~−0.01 | 낮 |
| L5 | `DeepSets 마스킹 증강` | C3 학습에 T20 마스킹 복제 행(가중 0.3) | T5 결측 셀 −0.00~−0.02 | 낮 |

업로드(사용자 수동, 하루 3회, sanity check 전용):
1. **09-05 밤 U1 = S0**(DeepSets 포함, CV 완료본). 예상 범위 밖(>0.50)이면 버그 조사; 아니면 CV 결과대로 진행.
2. **09-06 U2 = CV 통과 사다리 최상 후보**(L2/L3 포함본). 3. **09-06 U3 = S0-det**(SHA-exact 폴백의 점수 확인). 남는 슬롯은 쓰지 않음.
4. **09-07**: 최종 답안은 **CV 선택 규칙(§5.3)** 으로 결정, LB는 확인만. 답안 업로드 완료 후 모델 최종 제출. 폴백 순서: U2 후보 → S0 → S0-det.

## 9. 2일 일정 (09-05 ~ 09-07 정오)

- **09-05 오후(≈5 h)**: `configs/ocean_v2/p2.json` 사전등록 → `data.py/features.py/physics.py`(패널·특징·격자 적합·확장 투영) → `cv.py`(블록·purge·가중·지표·bootstrap) → `gbdt.py/oas.py/blend.py` → `train.py/predict.py/validate.py`. 단위 점검: hidden NaN assert, 키 순서, F1에서 T1복사 0.593·실제-depth 선형 0.88 재현.
- **09-05 저녁(≈1.5 h)**: S0-det CV(≈50분) → §5.5 계보 재측정 표 → `deepset.py` 이식 후 S0 CV(≈50분, 병렬 가능) → S0 CSV·validator → **U1 업로드**. 야간: L2/L3 특징 CV 배치 실행.
- **09-06 오전**: L2/L3/L4/L5 결과 판정(§5.3) → 최상 후보 CSV → **U2**, S0-det → **U3**. 오후: `package.py`로 01_data…07_source 생성(TRAIN/PREDICT 노트북·RUN_*.ps1·README: 환경·seed·소요시간·SHA·허용오차), 상수 리터럴 감사, 6 h 예산 기록. 밤: **클린룸 재현**(새 폴더에 패키지만 복사 → raw 지정 → train → predict → SHA/RMS 비교, ≤2.5 h).
- **09-07 오전**: 클린룸 결과 반영, 최종 후보 확정(CV 기준), 문서 정합(패키지 SHA = 업로드 SHA), 답안 업로드 → 모델 최종 제출.

## 10. 리스크와 대응

1. **2025-10 부분혼합+T5결측 regime이 어느 fold에도 없음**(T5 유효 41%, |T1−T5| 1.43 vs 2024-10 0.48). 대응: 셀별 블렌드에서 T5-결측 셀은 F3(자연 결측)+F1m/F2m(마스킹)으로만 적합, L2/L3 셀은 m_copy 가중이 자연히 커짐; L4 결측 셀은 패턴 B 모델을 ridge prior로 수축. 불확실성 ±0.05 °C 인정.
2. **F1의 계절 비대칭**(같은 계절 자료 없음 → OAS·doy 저평가). 대응: fold 가중과 F2/F3(전년 동계절 존재) 증거로 균형, 보고서에 "F1 pessimistic" 명시; OAS 가중을 손으로 올리지 않음.
3. **DeepSets 타 머신 비결정성**. 대응: CPU 전용·스레드 고정·round(4)·허용오차 문서화·S0-det 폴백 동봉.
4. **시간 문맥의 fold 간 부호 불일치**(F1 악화/F3 개선). 대응: §5.3 규칙 엄수, 특징 수 최소화, 실패 시 미채택.
5. **패키징·런타임 사고**. 대응: 09-06 밤 클린룸 재현 필수, 6 h 대비 여유 2배 이상.
6. **Public 노이즈**: P2는 26,061행이라 SE≈0.002 °C로 sanity check는 유의미하나, 어떤 경우에도 점수로 파라미터를 바꾸지 않음(로그에 "확인" 결과만 기록).

## 11. 재현 검증 체크리스트
- [ ] LB 유래 상수 0(리터럴 감사 산출물 첨부) — [ ] 외부자료·사전학습 0 — [ ] hidden 라벨 NaN assert 통과 — [ ] 조직 `score.py` 스키마 통과
- [ ] CV 리포트(fold·층·셀·bootstrap·계보 재측정) 보관 — [ ] `fitted_params.json`이 CV 산출물임을 receipt로 추적 — [ ] 클린룸 raw→train→predict 재현(GBDT/OAS SHA 일치, DeepSets RMS ≤1e-4) ≤6 h — [ ] 패키지 SHA = 업로드 SHA — [ ] Git에는 코드·설정·소형 보고서만

