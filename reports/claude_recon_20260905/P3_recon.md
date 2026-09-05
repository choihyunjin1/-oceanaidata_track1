# P3 정찰 보고서 — 유의파고 예측 (2026-09-05, read-only recon)

> 이 문서는 plan 모드 제약으로 plan 파일에 작성했다. 실행이 허용되면 이 내용을 그대로
> `reports/claude_recon_20260905/P3_recon.md`에 저장하면 된다. 저장소·데이터 파일은 수정하지 않았고,
> 분석은 venv python으로 집계 통계만 계산했다(원시 관측값 미출력). 경로는 저장소 루트 기준.

## 0. 핵심 요약

1. 현재 clean incumbent `P3_REFINED_PUBLIC_OPTIMUM_20260827`(Public 0.583892 m)는 **정직한 모델 O(Public 0.607071)** 위에
   Public 리더보드 점수 3점으로 맞춘 선형축 계수 `alpha=-10.217`을 12/18/24h에 곱한 것이다. 축 `A-O`는 두 CatBoost 계보의
   차이가 아니라 사실상 **persistence shrink 0.20 vs 0.25의 차이**(corr(A−O, O−P) = −0.998, 기울기 −0.0625 = −0.05/0.8)여서,
   최종식은 `F ≈ O + 0.65·(O − P) = 1.65·O − 0.65·P` — 즉 모델이 persistence에서 벗어난 양을 **1.65배 외삽**하는 것이다.
2. 로컬 OOF(181 사례)는 정반대를 말한다: 장기리드 최적 증폭계수 k* = −0.096(축소), 리드별 12h +0.02 / 18h −0.13 / 24h −0.13.
   Public이 준 k = +0.65를 로컬에 적용하면 RMSE 0.7791 → 0.8077(+0.029 m). Private가 Public과 같은 분포면 이득(≈−0.02 m),
   로컬과 같은 분포면 손실(≈+0.03 m). 이것이 P3의 가장 큰 Private 위험이다.
3. 공식 probe(α=−2, 12h 단독 / 18·24h 단독)를 리드별로 분해하면 12h의 Public 최적 α는 ≈ −3이고, 현재 −10.2는 12h에서
   Public MSE를 +0.0026 악화시킨다(추정). **12h만 O로 되돌리면 Public ≈0.5817**로 오히려 개선되면서 LB 적합 의존이 줄어든다.
4. 검증 표면은 hs≥1.5·78h 간격·episode 분리까지 잘 모사했지만 181 사례(CI90 반폭 ≈0.003–0.005 m)라 0.001–0.005 m 후보를 분해할
   수 없고, 전방 전용 fold(1번 fold는 2024 상반기 7,912 anchor로만 학습), I/S-ORS 2024 기상 전무, I-ORS 2024-08~10 파랑 전무,
   여름 사례 비중(test 30% vs 로컬 7%) 불일치 때문에 로컬→Public 전이가 반복적으로 뒤집혔다(방향 일치 1/5, v42·Hs² 역전).
5. 학습 분포 불일치가 방치돼 있다: 학습 anchor는 hs≥1.5인 **모든** 20분 시점(24,360개, 상승 53%, 임계 갓 통과 9%, 평균 hs 2.27,
   24h 평균 변화 −0.645 m)인데 test 사례는 first-eligible 선택(상승 84%, 갓 통과 72%, 중앙값 1.64, 24h 변화 −0.32). 현재 가중치
   `exp(−0.45·max(hs−1.5,0))`는 수준만 보정한다. OOF에서 12/18h 평균 편향 +0.135/+0.139 m(모델이 감쇠를 과대예측)로 드러난다.
6. ~60개 residual cycle(v21~v81)은 전부 **부적격 KMA 0.425 champion을 reference**로 한 ridge 잔차 10% 블렌드라 clean 계보에 직접
   쓸 수 없고, 효과 크기(−0.0015~−0.005 m)가 검증 잡음 이하였다. 게이트가 지나치게 엄격했다기보다 효과가 잡음 수준이었다.
7. 규정 준수 개선 여지 중 기대값이 큰 순서: 선택-정합 학습 가중(phase-aware) 재학습 → 다중 seed 평균 → 더 큰 정합 검증 표면 →
   alpha 위험 축소(12h 복원) → 정점-무관 앙상블 멤버. TabPFN·KMA·ERA5·추가 residual cycle은 하지 않는다.

## 1. 현재 최고 파이프라인 정확한 재구성

### 1.1 데이터 → 10분 격자 → anchor (`src/p3_wave/data.py`)
- `load_p3_data`(:62-72): 6개 배포 파일만 읽음. `build_training_grid`(:197-212): 정점별 wave.time.min~max+10분의 10분 격자에
  wave(20분)·atmos(10분)를 left-merge → 20분 중간행의 파랑열은 구조적 NaN(test_context와 동일 구조; :157-159에서 검증).
- `build_anchor_table`(:215-247): anchor 조건 = `hs>=1.5`(:223) ∧ 6개 리드 target(`hs.shift(-lead*6)`, :226) 모두 존재 ∧
  시작+48h 이후(:229). `dense_spacing_minutes=20`(train_model.py:42) → stride 1 → **조건을 만족하는 모든 20분 시점**이 anchor.
  결과 24,360 anchor(G 9,893 / I 7,312 / S 7,155). 타깃은 절대 hs이며 학습 시 `target − current_hs` 잔차로 변환(validation.py:84,
  train_model.py:76-81).
- 검증용 `select_independent_validation`(:250-276): 창 안에서 정점별 시간순 first-eligible + 78h 간격 greedy(운영진 선택 규칙 모사).

### 1.2 특징 (`src/p3_wave/features.py`)
- 입력 8개 수치열 + 2개 방향열(:13-14). 방향은 sin/cos(:80-83). 파생 6개(:94-102): `wave_energy=hs²`, `hmax_hs_ratio`,
  `steepness_proxy=hs/tp²`, `gust_excess`, `wind_wave_alignment=cos(wdir−wvdir)`(둘 다 "from" 규약이라 부호 정확),
  `wind_input_proxy=wspd²·max(align,0)`. 총 18개 시계열.
- 각 시계열: `_current`(마지막 유한값, :27-29,105), lag 1/3/6/9/12/18/24/36/48h(:106-110, 10분 index로 정확 정렬; 모든 lag가
  60분 배수라 파랑 20분 슬롯과 일치), 창 1/3/6/12/24/48h 요약 10개(mean/std/min/max/q10/q50/q90/delta/slope/valid, :32-66).
  변화량 `hs/wspd/caph_change_{1,3,6,12,24}h`(:118-121). 원본 1,275개 → `compact_feature_columns`(models.py:167-208)로 591개
  (current + lag 1/3/6/12/24/48 + 창 3/6/12/24/48의 mean/std/delta/slope/valid + change; `event_` 토큰은 캐시에 없음).
- 절대시각·월·계절 특징 없음(:69-70). 사례별 독립 처리(:125-137), 사례 간 결합 없음. 결측은 NaN 유지(CatBoost 내장 처리)이고
  `_valid_` 비율이 결측 정보를 전달.

### 1.3 두 CatBoost 계보 — O(원본)와 A(축)
공통 하이퍼파라미터(train_model.py:52-63, 83-96; submissions/p3_frozen_catboost/manifest.json; run_p3_corrected_repeated_forward_catboost_v2.py:92-119):
- **single**: RMSE, iter 700, lr 0.035, depth 6, l2 8, random_strength 0.2, CPU, cat=[station, lead_h]; 입력 = 591 특징 + `current_hs_for_residual`,
  anchor×6 리드로 펼친 pooled 회귀(validation.py:62-100), 타깃 = 리드별 잔차.
- **multi**: MultiRMSE 6출력 잔차, iter 1200, lr 0.03, depth 7, l2 10, rs 0.15, **GPU Plain**(비결정적), cat=[station].
- 표본가중 `threshold_case_weights = exp(−0.45·max(hs−1.5,0))/mean`(models.py:211-216) — "public case-selection shift toward 1.5 m" 명목.
- seed: fold 20260816/17/18, full 20260817. 정점 pooled(정점은 범주 특징), 리드는 single=범주·multi=출력축.
- O 계보(2026-08-17): `submissions/p3_frozen_catboost`(전체 24,360 anchor로 학습, model.cbm b39b…, model_multi.cbm b777…) → 0.5/0.5 등가 앙상블
  → `submissions/p3_lead_long_loss_router`(12/18/24h만 softmax 라우터, 3/6/9h는 0.5/0.5/0 고정) → `submissions/p3_long_persistence_shrink`
  (12/18/24h: 0.8·routed + 0.2·persistence, persistence_shrink.py:11,52-55) = `output/2026-08-20/ready/P3_submission.csv` SHA d89e69b9.
- A 계보(2026-08-22): corrected v2(`artifacts/p3_corrected_repeated_forward_catboost_v2`, 동일 파라미터·seed 20260817 full refit, 라우터 `smooth_medium`
  고정 재적합) → `FixedLongLeadShrinkCalibrator` 0.25(corrected_fixed_long_shrink.py:18-19,46-49) = `artifacts/p3_corrected_fixed_long_shrink_v4/candidate/submission.csv`
  SHA 607f7cd4. 0.25는 2026-08-17 `artifacts/p3/long_persistence_shrink/metrics.json`의 bounded_sensitivity(0.15/0.20/0.25 → 0.7812/0.7802/0.7796)에서
  "봉인된" 값(config p3_corrected_fixed_long_shrink_v4.json:16-22).
- 라우터(loss_router.py): 입력 = station + 관측 17개(:20-38) + 두 컴포넌트의 리드별 delta/absdiff/peak/drawdown(:88-99). Ridge(α=10)로
  log(case MSE+0.05) 3성분 예측 → softmax(temperature 2×median span) → strength 0.5로 [0.5,0.5,0]과 혼합(:236-260). 최종 라우터는 182(O)/181(A) 사례
  OOF 전체로 적합. test 가중 요약(lead_long_loss_router/manifest.json): single 0.46 / multi 0.46 / persistence 0.078(p90 0.17).
- 최종 패키지(`scripts/final_submission_20260905/P3/predict_submission.py`): O 경로 :124-146(`predict_catboost_components` → `apply_saved_router` →
  `apply_long_lead_persistence_shrink`, CSV float round-trip 재현 final_inference.py:45-53), A 경로 `_axis_prediction` :34-99, 결합 :150-152
  `values[active] += alpha·(axis − original)`(active = 12/18/24h), 3/6/9h는 O와 byte-identical 검사(:153-154), [0,30] 가드(:155).
  train_model.py는 base 모델만 재학습하고 라우터/calibrator는 재학습하지 않는다(:3-6, :120-123) → TRAIN 노트북이 인증 파이프라인 전체를 scratch 재현하지는 않음.

### 1.4 alpha 산출 연대기 (모두 Public 리더보드 점수 기반)
| 일시(KST) | 후보 | 축 정의 | Public RMSE | 출처 |
|---|---|---|---:|---|
| 08-25 16:39 | O (d89e69b9) | α=0 | 0.607071 | Downloads/20260825_OFFICIAL_SCORE_RECONCILIATION.json |
| 08-25 18:23 | A (607f7cd4) | α=+1 (shrink 0.25) | 0.611680 | 같은 파일 rounds/A/P3 |
| 08-25 18:24 | B (c1be3931) | α=+0.5 (shrink 0.225) | 0.609346 | rounds/B/P3 |
| 08-26 22:01 | REVERSE_GLOBAL | α=−2, 12/18/24h | 0.599072 | round_D/OFFICIAL_RESULTS_20260826.json |
| 08-26 22:01 | LEAD12_ONLY | α=−2, 12h만 | 0.606681 | 〃 |
| 08-26 22:02 | LEAD18_24_ONLY | α=−2, 18/24h만 | 0.599382 | 〃 |
| 08-27 | LONG_QUADRATIC_OPTIMUM | α*=−10.235445 | 0.583892 (24.066167) | round_G/SET_MANIFEST.json |
| 08-27 | BRACKET_NEG12 | α=−12 | 0.584611 | 20260827_P3_REFINED_PUBLIC_OPTIMUM_READY/MANIFEST.json |
| 08-27 23:36 | REFINED (ea65370a) | α=−10.21743189862218 | 0.583892 (24.066168, +0.000001점) | 〃 |

- Round G(`scripts/build_p2_p3_public_quadratic_round_g_20260827.py:87-105`): MSE 가법성으로 α=−2 장기리드 점수를 `c1=√(L12²+L1824²−O²)`로 복원,
  (0, O²), (0.5, B²), (−2, c1²) 3점 이차식 → α*. 표시 반올림 envelope 검사(:108-130) 및 점수↔RMSE 선형환산(:207-217, 기울기 ≈15.87점/m).
  메모에 "고정 Public 표면의 정확한 선형축 최적화이며 Private 일반화 보장은 아닙니다"(:276) 명시.
- Refined(`scripts/build_p3_refined_public_optimum_20260827.py:28-48`): (0, 0.607071), (−10.235445, 0.583892), (−12, 0.584611)에 `np.polyfit` 2차 → α=−10.2174,
  예측 이득 7e-8 m. 즉 마지막 제출은 **순수 LB 미세조정**이었다.
- 참고: 선형축 위 Public MSE는 정확히 α의 2차식(잡음 없는 항등식)이므로 α*는 "Public 최적"으로서 정확하다. 문제는 오직 Public(66사례)→Private(134사례) 전이다.

### 1.5 alpha가 실제로 하는 일 (집계, 200사례 CSV 기준)
- 장기리드에서 `A−O ≈ −0.0625·(O−P)`(corr −0.998; 12/18/24h 동일) ⇒ `F−O = k·(O−P)`, k = 0.649/0.655/0.655, 잔차 std 0.011–0.0125 m
  (두 계보의 GPU 비결정성 차이가 10.2배 증폭된 몫; 작지만 순수 잡음).
- 유효식: `F = 1.65·O − 0.65·P = 1.32·R − 0.32·P`(R=라우팅 원예측) ≈ `1.22·E − 0.22·P`(E=CatBoost 등가 앙상블) — **persistence 가중치가 음수**.
- 이동량 mean|F−O|: 12h 0.148, 18h 0.177, 24h 0.184 m (max 0.60/0.79/0.88). mean(F−P): −0.143/−0.316/−0.388 vs mean(O−P) −0.087/−0.192/−0.236.
  F 최소값 0.587(O 0.794), F < 0.5·P인 행 42개. 정점별 mean(F−O): G −0.115, I −0.088, S −0.133.
- Public 기여: 0.607071 → 0.583892 = **−0.0232 m(+0.368점)** 전부 LB 적합축. 그중 약 절반(0.607→0.590, α≈−5)은 "0.2 shrink 제거"에 해당하고
  나머지 절반은 원모델을 넘어선 31% 증폭.
- 리드별 분해(공식 α=−2 probe: MSE Δ 12h −0.000473, 18/24h −0.009276, 합 −0.00975 ≈ 전체 −0.00965로 가법성 확인; 리드별 Σd² 비중 0.226/0.364/0.410):
  a12=5.96e-5, b12=3.56e-4 → **α12*≈−3.0**; a1824=2.05e-4, b1824=5.05e-3 → α1824*≈−12.3. 현재 α에서 12h 기여 **+0.0026 MSE(악화)**, 18/24h −0.0302.
  예측 Public: 12h→α0 0.5817, 12h→−3 0.5812, 전체 −8 0.5850, 전체 −5 0.5900.

## 2. 검증 설계와 약점

### 2.1 설계
- 창(validation.py:22-26; v2 config :69-86): `2024_h2_storm` 07-01~11-01, `winter_transition` 11-01~03-01, `2025_h1` 03-01~06-25. 학습 = anchor_time < 창 시작 − 78h
  (전방 전용, :39-42). 검증 = 정점별 first-eligible 78h greedy; corrected v2(corrected_repeated_forward.py:67-143)는 정점-전역 greedy + storm episode 중복 금지 +
  학습에서 같은 episode 제거(:164-209) → 181 사례/1,086행(fold 49/79/53; 정점 G67/I46/S68; fold train anchor 7,912/11,754/20,899).
- hs≥1.5 필터 모사: **예**(anchor 자체가 hs≥1.5). persistence 동일 anchor 비교: **예**(metrics.json). 로컬 리드별 persistence→final:
  3h 0.633→0.584, 6h 0.801→0.662, 9h 0.876→0.782, 12h 0.926→0.861, 18h 0.964→0.893, 24h 0.936→0.845; pooled 0.8635→0.7791(−9.8%).
  게이트(corrected_repeated_forward.py:375-415): pooled < persistence, 사례 bootstrap CI90 상한 < 0, 2/3 fold 개선, 정점/18·24h 악화 ≤0.01.
- 라우터 선택은 prequential(과거 fold만, loss_router.py:396-489), shrink 0.2/0.25는 같은 OOF 감도표에서 선택(경미한 in-sample).

### 2.2 약점
1. **표본 크기**: 181사례 → 0.001~0.005 m 효과는 CI90(±0.003~0.005) 안. 이 표면을 60회 이상 재사용(보고서 스스로 "EXPLORATORY_ONLY").
2. **전방 전용 fold**: 1번 fold 모델은 2024-01~06 7,912 anchor(I/S 기상 전무)로 학습 → 최종 모델(24,360)의 대리로 부적절; 반면 test 여름은
   최종 모델이 2024 여름 1회를 본 상태로 예측. 2024 여름 fold의 I-ORS 검증 사례 4개뿐(I-ORS 파랑 2024-08~10 100% 결측).
3. **계절 구성 불일치**: test 기준시각 airt>22℃ 30%, <12℃ 39%; 로컬 greedy anchor airt>22℃ 7%(NaN 43%). 태풍기(7~10월) 사례가 로컬 검증에 49/281뿐이고
   그마저 I-ORS 없음. 로컬 7~10월 persistence RMSE는 12h 0.995/18h 1.007로 다른 계절(0.837/0.911)보다 훨씬 어렵다.
4. **학습 분포 ≠ 선택 분포**: §0-5. 로컬 OOF 평균 편향 truth−final = +0.135(12h)/+0.139(18h)/+0.07(24h) → 모델이 감쇠를 과대예측(dense anchor 학습 효과).
   그런데 Public은 반대로 더 큰 감쇠를 요구(k=+0.65) → 로컬 표면이 test 기간을 대표하지 못한다는 강한 신호.
5. **전이 실적**: 08-25 캘리브레이션 방향 일치 1/5(20260825_OFFICIAL_SCORE_RECONCILIATION.json:93-105); Hs² 후보 로컬 3표면 개선→공식 악화;
   v42 로컬 −0.0049 → 공식 +0.0011; HPO v2 selection −0.0229 → confirmation +0.0080. Public 대비 O의 상대이득 −19%(0.750→0.607)가 로컬 −9.8%의 두 배 —
   로컬 표면이 모델 우위를 과소평가한다는 또 다른 불일치.
6. 검증 anchor 선택이 O(182, 창별 greedy 리셋)와 A(181, 정점-전역)로 미세하게 달라 두 계보의 OOF 수치가 직접 비교 불가(0.7867 vs 0.7791는 다른 표면).

## 3. 취약점·버그·이상한 점·한계 (심각도순)
1. **[HIGH] LB 적합 외삽 계수** — predict_submission.py:150-152, build_p3_refined_public_optimum_20260827.py:44-48. 1.65배 증폭; 로컬 최적은 −0.10.
   Private 기대범위 −0.02~+0.03 m. Public 66사례에서 k SE≈0.19 → k∈[0.27,1.03]도 Public과 양립. 승자의 저주로 Private 이득은 Public보다 작을 가능성 큼.
2. **[HIGH] 12h 증폭은 Public에서도 근거 없음** — round D probe 분해(§1.5): α12*≈−3, 현재 −10.2는 12h MSE +0.0026. 로컬 12h k*=+0.02(중립).
3. **[MED] 축이 정보가 아니라 shrink 차이** — A−O가 O−P와 −0.998 상관. "두 계보 결합"이라는 설명(configs/final_submission_20260905.json P3.summary)은 실질과 다름.
   10.2배 곱으로 GPU 비결정 잡음 std≈0.012 m가 제출값에 주입됨(build_p3_refined…:61-62).
4. **[MED] 학습 anchor 분포 불일치** — data.py:215-247 dense anchor + models.py:211-216 수준 가중만. phase(갓 통과 vs 폭풍 중) 미보정 → OOF 편향 +0.07~0.14 m.
5. **[MED] I/S-ORS 2024 기상 전무** — train_atmos 행수 G 78,768 / I·S 26,064(2025년만). 격자 merge(data.py:203-209)로 2024 I/S anchor의 기상 특징 전부 NaN,
   `wspd_valid_*=0`이 사실상 연도 표식. test는 기상 ~97% 존재. target-shift 재감사(reports/p3_target_shift_retroaudit_20260828_v11/report-source.md:40)도
   AUC 0.726의 주원인으로 지목. I/S의 바람-파랑 학습은 2025-01~06(겨울·봄)뿐이고 여름 바람은 G-ORS만 학습.
6. **[MED] I-ORS 태풍기 학습·검증 공백** — I-ORS hs 결측 2024-07 34%, 08~10 100%, 11 47%. test I-ORS 70사례 중 상당수가 7~10월일 것.
7. **[LOW] 계절 대리변수** — airt/relh는 사례 내부 값이라 규정상 허용이나 계절을 사실상 노출(절대 날짜 복원은 불가, 사례 간 결합 없음 features.py:125-137).
   `_valid_` 특징은 결측 구조를 학습(test hmax 결측 13% vs train 8.5%, relh 5% vs 13%로 상이).
8. **[LOW] 라우터 학습/적용 불일치** — fold 모델 OOF delta로 학습(loss_router.py:88-99), full 모델 delta에 적용(final_inference.py:150-165). 전형적 stacking 편향.
9. **[LOW] 재현성** — multi GPU Plain 비결정(train_model.py:91-94; output/2026-08-20/receipts/P3.json retrain max|Δ| 0.0049, RMSE Δ 0.0007). 단일 seed, 평균 없음.
10. **[LOW] `_current` = 마지막 유한값** — features.py:27-29,105. 결측 시 수 시간 전 값이 "현재"가 되며 경과시간 특징 없음(train/test 일관이라 누수는 아님).
    test에서 hs가 일부 결측인 사례 53개(p99 결측률 57%), wspd 전결측 사례 1개.
11. **[LOW] 1h 창 통계** — 파랑열은 10분 격자에서 1h 창에 3~4점뿐(std/slope 잡음). `_valid_`는 파랑열에서 최대 0.5. compact 집합은 1h 창을 제외하므로 영향 제한적.
12. **[INFO] 클립** — [0,30] 가드(final_inference.py:83-100, predict_submission.py:155)는 발동하지 않음(F min 0.587). step 0 hs는 200/200 존재, 최소 1.5(data.py:148-150 검증).
13. **[INFO] 규정** — O/A manifest의 입력 hash는 배포 6파일뿐(submissions/p3_frozen_catboost/manifest.json input_sha256). 외부자료·사전학습 0.
    반면 v21~v81 cycle, KMA sweep, ERA5, v42m1, v5, v19는 모두 KMA 0.425 reference(run_p3_path_signature_residual_cycle_20260901_v23.py:399 `uniform_0p425`)
    또는 ERA5 계보 → 부적격. 20260901 lead-continuous 후보(7d603e16)만 clean(미제출).
14. **[INFO] 시간 정렬** — wave 20분/atmos 10분 merge와 `step_minute` 정렬(step 0 = 파랑 슬롯, −10 = 기상만)은 train/test 일치. lag·target index 계산(lead*6, lag_h*6) 정확.
    미래 행 사용 없음(모든 창이 anchor에서 끝남; target은 anchor 테이블에서만). 절대시각은 fold 분할에만 사용.

## 4. 공식 P3 제출 이력과 LB 적합 의존성

| # | 일시(KST) | 후보 | 한 줄 설명 | Public RMSE | 점수 | 적격 |
|---|---|---|---|---:|---:|---|
| 1 | 08-25 16:39 | O `d89e69b9` | CatBoost single+multi 등가 → 장기리드 loss-router → 0.2 persistence shrink | 0.607071 | 23.698280 | clean |
| 2 | 08-25 18:23 | A `607f7cd4` (CORRECTED_FIXED_LONG_SHRINK_V4) | corrected v2 모델 + 0.25 shrink | 0.611680 | 23.625124 | clean |
| 3 | 08-25 18:24 | B `c1be3931` (FIXED_LONG_SHRINK_22P5) | O/A 장기리드 중점(α=0.5) | 0.609346 | 23.662165 | clean |
| 4 | 08-26 22:01 | REVERSE_GLOBAL `57a90beb` | α=−2 12/18/24h | 0.599072 | 23.825229 | clean(LB축) |
| 5 | 08-26 22:01 | PROBE_LEAD12_ONLY `c5ac003e` | α=−2 12h만 | 0.606681 | 23.704466 | clean(probe) |
| 6 | 08-26 22:02 | PROBE_LEAD18_24_ONLY `91ead747` | α=−2 18/24h만 | 0.599382 | 23.820314 | clean(probe) |
| 7 | 08-27 | LONG_QUADRATIC_OPTIMUM | α=−10.235445 | 0.583892 | 24.066167 | clean(LB적합) |
| 8 | 08-27 | BRACKET_NEG12 | α=−12 | 0.584611 | — | clean(LB적합) |
| 9 | 08-27 23:36 | **REFINED_PUBLIC_OPTIMUM `ea65370a`** | α=−10.2174 (현 clean incumbent) | **0.583892** | 24.066168 | clean(LB적합) |
| 10 | 08-28 13:57 | Champion-matched ERA5 Hs² `3967333b` | ERA5 잔차 보정 | 0.585738 | 24.036866 | 외부(ERA5) |
| 11 | 08-28 23:44~ | KMA_ALPHA20 | KMA 예보 균일 보정 0.2 (18/24h) | 0.577671 | — | 외부(KMA) |
| 12 | 08-29 | KMA_ALPHA40 | 균일 0.4 | 0.575262 | 24.203126 | 외부 |
| 13 | 08-29 | KMA lead-split (18h 0.6/24h 1.0), (0.8/1.0) | 리드 분리 probe | 0.577577 / 0.576264 | — | 외부 |
| 14 | 08-30 21:19 | KMA_UNIFORM_0425 `144f5e17` | 균일 0.425 (Public 최고, 부적격) | 0.575233 | 24.203599 | 외부 |
| 15 | 08-30 | KMA_LEAVE_S_ORS_OUT / LEAVE_I_ORS_OUT | 정점 절제 probe | 0.579102 / 0.578951 | — | 외부 |
| 16 | 08-31 | KMA_A18_0200_A24_0425 | 리드별 KMA | 0.576589 | 24.182070 | 외부 |
| 17 | 08-31 | V5_EXTRATREES_HARD_PHYSICAL_ROUTER | 물리 규칙 라우터 (KMA base) | 0.590956 | 23.954041 | 외부 |
| 18 | 08-31 | KMA_CONTINUOUS_WAVE_POWER_V19 | 연속 wave-power 인자 | 0.589840 | 23.971758 | 외부 |
| 19 | 09-01 07:52 | V42M1_KM80_RIDGE512_ADD10 `4ca8c020` | Kramers–Moyal 잔차 ridge 10% (KMA base) | 0.576320 | 24.186338 | 외부 |
| — | 09-01 | LEAD_CONTINUOUS_SCORE_PRIORITY_V1 `7d603e16` | clean incumbent + lead×regime ridge (예상 0.5797, fresh 1-case +0.023 악화) | 미제출 | — | clean |

출처: Downloads/20260825_OFFICIAL_SCORE_RECONCILIATION.json, round_D/OFFICIAL_RESULTS_20260826.json, 20260827_P3_REFINED_PUBLIC_OPTIMUM_READY/MANIFEST.json,
20260829_ADAPTIVE_FINAL_PROBES_READY/SET_MANIFEST.json(official_evidence), reports/approved_parallel_execution_20260828_v9/p3_official_submission_receipt_20260828.json,
reports/p3_kma_uniform_0425_official_submission_20260830_v1, reports/official_information_probe_cycle_20260830_v1/p3-official-result.json,
reports/p3_official_candidate_ledger_20260901_v1/official-lineage-audit.json, reports/p3_v42_official_submission_20260901_v1, reports/submission_ladders_internal_validation_20260831_v1.
`artifacts/one_shot_exposure_ledger.jsonl`은 P1 1건뿐. `artifacts/official_final_probe_p3_a.csv`=A(607f…)와 동일 SHA, `_o.csv`는 O와 다른 SHA(7b22cc8a, 포맷 차이 추정).
2026-08-25 round A 체크리스트의 A/P3 = 607f7cd4(=현 "axis component")가 맞다.

**LB 적합 의존성**: clean 계보 중 LB와 무관하게 만든 제출은 O/A/B 3개뿐이며 최고는 O 0.607071. 0.583892 중 0.0232 m(4%, +0.368점)는 8회의
Public 평가로 맞춘 1차원 축 계수의 산물이다. 축의 절반(α≈−5까지)은 "shrink 되돌리기"라 물리적으로 온건하지만, 나머지 절반은 모델을 넘어선 외삽이다.
KMA 계보(0.5752)는 외부 예보의 가치가 Public에서 ≈0.009 m였음을 보여주지만 현재 규정상 무의미하다. 리더 추정 0.5387 m(reports/leaderboard_clean_headroom_research_20260901_v1)
와의 격차 0.045 m는 α 조정으로 닫힐 크기가 아니다.

## 5. 실패한 시도 요약
- **residual cycle v21~v81(약 60개, 09-01 집중)**: 공통 구조 = KMA 0.425 reference의 잔차를 새 특징족(path signature, wavelet scattering, Kramers–Moyal, 엔트로피,
  visibility graph, rainflow, 정점 그래프 등) ridge(α 512/2048)로 예측해 10% 가산(`ADD10`), 182사례 6블록 cross-fit 12회 학습. 게이트(scripts/qa_p3_cycle_generic.py):
  pooled Δ<0, 5/6 블록 개선, worst block/lead/station-lead/tail 한도, episode·block-station bootstrap CI90, "transport-adjusted" 점수 = raw − 0.0496점
  (qa_…v29.py:32-33; 15.87점/m). 결과: 대부분 NO_GO(Δ −0.0015~−0.003 m), PASS_STABLE 소수(v42 −0.0049) → v42 공식 +0.0011 역전. 실패 이유: (a) 효과가 CI 폭 이하,
  (b) reference가 이미 LB 적합된 부적격 KMA 축이라 로컬 표면이 Public과 어긋남, (c) 표면 반복 노출. 게이트가 과하게 엄격한 것이 아니라 신호가 없었다.
- **KMA/ERA5/Chronos 외부 계보**(reports/p3_kma_*, p3_champion_matched_era5_*, p3_chronos2_*): 규정 변경으로 폐기. KMA α 전수 탐색(reports/p3_kma_alpha_surface_sweep_20260829_v1)
  은 cross-fit에서 전부 악화(+0.006)했으나 Public은 개선 — 로컬/Public 역전의 또 다른 사례.
- **CatBoost HPO**: v1 48점 격자 중 Ordered+Depthwise 비호환으로 75번째 fit에서 종료(failure-report.md:273-280); v2/v3 confirmation에서 challenger_21이 +0.0080 악화,
  3fold·3정점·6리드 전부 비개선(reports/p3_catboost_confirmation_contract_repair_20260830_v3/report-source.md:383). 단, 선택/확인 표면 182사례의 잡음 문제 동일.
- **딥러닝/시퀀스**: TimeXer, TSMixer, NLinear ridge, RevIN patch, causal forcing sequence, Chronos-2 전이 — 모두 재가중 표면에서 악화(target-shift gap-matrix.md:31).
- **perfect-future-wind oracle**(reports/p3_perfect_future_wind_oracle_20260829_v1): 실제 미래 바람 벡터를 넣어도 18/24h +0.0013 → "future wind/MOS 계열 종료".
  단 ridge α=1000, 123사례, KMA base 위 선형 잔차라는 약한 설계라 "바람 정보 무가치" 결론은 과잉 일반화(어차피 test에 미래 바람 없음).
- **관측 바람 direct residual pilot**(124사례): 4모델 모두 악화(+0.008). **fractional-change 타깃**(c1r1): +0.0009 악화. **선택-정합 sparse GP abstention, masked SSL,
  lead-continuous ridge**: 내부 소폭 개선/fresh 1-case 악화 → 미제출. **TabPFN**: 미설치·라이선스 미수락(사전학습 가중이라 규정 위험).
- 공통 교훈: 모든 시도가 "incumbent 위 잔차 미세보정"이었고, 학습 분포·검증 표면 정합이라는 상류 문제는 손대지 않았다.

## 6. 데이터 집계 관찰 (venv 계산, 집계만)
- train_wave 정점별 39,384행(547일×72, 결측 포함 격자). hs 결측 8.65%, tp 10.2%, hmax 8.5%, wvdir 7.9%. train_atmos G 78,768 / I 26,064 / S 26,064(2025년만);
  wspd 결측 2.0%, relh 13.2%. I-ORS hs 월별 결측: 2024-07 34%, 08~10 100%, 11 47%; G-ORS 2024-10 22%, 11 18%.
- hs 분위(5/25/50/75/90/95/99, max): G 0.34/0.59/0.97/1.56/2.33/2.93/3.83(6.83), I 0.25/0.51/0.84/1.46/2.31/2.81/3.62(5.59), S 0.26/0.45/0.73/1.27/1.97/2.53/3.61(4.86).
  hs≥1.5 비율 G 26%, I 19%, S 18.5%.
- 적격 anchor(hs≥1.5, 6타깃 유효) 24,360 = G 9,893 / I 7,312 / S 7,155. 월별(전체): 24-01 2,116 · 02 3,004 · 03 1,973 · 04 407 · 05 267 · 06 377 · 07 1,351 · 08 567 ·
  09 710 · 10 1,054 · 11 1,746 · 12 2,588 · 25-01 2,061 · 02 2,678 · 03 1,969 · 04 680 · 05 446 · 06 366. I-ORS 2024-08~10 = 0.
- persistence RMSE(리드 3/6/9/12/18/24, pooled): dense 전체 0.470/0.626/0.717/0.792/1.001/1.112 (0.816); **greedy 78h 281사례(G101/I80/S100)** 0.595/0.755/0.812/0.866/0.928/0.931 (0.823);
  공식 전체 0.546/0.701/0.706/0.760/0.890/0.946 (0.769; Public 0.750, Private 0.779). greedy 7~10월 49사례 0.514/0.802/0.778/0.995/1.007/0.965.
  정점별 greedy: G 0.56/0.60/0.62/0.76/0.91/0.88, I 0.55/0.78/0.87/0.90/0.87/0.88, S 0.66/0.87/0.93/0.93/0.99/1.02.
- anchor 위상: dense 상승(1h) 53%, 갓 통과(1h 전 <1.5) 9%, 평균 hs 2.27, mean(t−hs) 24h −0.645; greedy 82%/73%/1.81/−0.317; **test 84%/71.5%(3h 전 <1.5: 73%)**,
  step-0 hs 분위 0/10/25/50/75/90/100 = 1.50/1.52/1.55/1.64/1.865/2.39/4.45(greedy 로컬 1.51/1.53/1.60/1.81/2.46). 무작위 오프셋 greedy 40회 합집합 328 anchor(포화), 6h 간격 anchor 1,748.
- test_context: 정점별 70/70/60; step 0 hs 결측 0. 파랑 슬롯 결측 hs 3.3%, tp 5.1%, hmax 13.2%, wvdir 2.4%; 기상 ~3%(relh 5.3%). hs 일부 결측 사례 53, wspd 전결측 1.
  airt(step 0) 분위 10/25/50/75/90 = 4.7/9.1/14.2/24.3/27.9℃ → 여름형 30%, 겨울형 39%.
- 로컬 OOF(181): persistence 0.8635 / routed 0.7851 / final 0.7791; fold별 persist→final 0.841→0.715, 0.886→0.786, 0.850→0.824; 정점 G 0.793→0.727, I 0.933→0.887, S 0.881→0.751.

## 7. 개선 기회 (규정 준수, 기대이득×노력 순)
1. **선택-정합(phase-aware) 학습 가중으로 두 CatBoost 재학습** — 기대 0.005~0.02 m(불확실), 노력 낮음(캐시 `artifacts/p3/features_all20_v1` 재사용, 가중만 변경).
   운영진 선택규칙이 완전히 알려져 있으므로 학습 anchor를 (hs 수준, 갓-통과 여부, 상승/하강, episode 경과시간) 조합별 test-like 밀도/dense 밀도 비율로 중요도 가중하거나,
   갓-통과 anchor(≈2,000개) + 폭풍-중 anchor를 28% 비중으로 재표집. OOF의 +0.07~0.14 m 과감쇠 편향을 직접 겨냥. `episode_age`, `hs−1.5 통과 후 경과시간` 특징 추가는 덤.
2. **다중 seed(5×) 평균 + LightGBM 변형 1개** — 기대 0.002~0.005 m, 노력 낮음. GPU 비결정성 완화, 재현성 향상. 현재 seed 1개.
3. **더 크고 정합된 검증 표면** — 전방 전용 3fold 대신 3개월 6블록 양방향 purge(78h)+episode 제거 CV, 검증 anchor는 갓-통과 anchor 전부(≈2,000)+6h 간격 폭풍-중 anchor를
   test 구성(72/28)으로 가중, episode 블록 bootstrap. 효과 표준오차 ≈0.003→≈0.0012 m. 1·2·5의 정직한 판정에 필수. 노력 낮음.
4. **alpha 위험 축소(학습 불필요)** — (a) 12h를 O로 복원(예측 Public 0.5817, LB 의존 행 600→400); (b) 18/24h α를 −8로 헤지(Public ≈0.585, Private 하방 축소);
   (c) 유지. 권고: 최종 제출 기본형은 (a). **α를 LB로 다시 맞추지 말 것.** 새 모델 M에 k=0.65를 이식하는 것도 금지(모델·표면 특정 계수).
5. **정점-무관 앙상블 멤버** — station 범주 제거 또는 station×계절 균형 가중 멤버를 추가해 G-ORS 여름 지식을 I-ORS 여름(학습 전무)으로 전이. 노력 낮음,
   로컬 검증은 G/S 여름 fold로만 가능(한계 명시).
6. **incumbent와의 안전한 결합** — 정직 모델 M을 단독 제출(정직 Public 수치 확보) 후, 사전 고정 `0.5·M + 0.5·F(12h 복원)` 1회. 남은 업로드(3/일, 09-07 마감)에
   맞춰 [M, 0.5 블렌드, (a)] 순. 점수 보고 계수 재조정 금지.
7. **로그/비율 타깃** — 낮은 우선순위(fractional-change +0.0009 실패, 지표가 m 단위 RMSE). 시도한다면 `log(target/current)` 학습 후 `exp(μ+σ²/2)` 보정 멤버로만.
8. **물리 특징** — 이미 wind_input, steepness, alignment, tp 추세 존재. 60개 cycle이 ≤0.005 m를 반복 확인했으므로 특징 추가 단독은 후순위. 예외: 1의 위상 특징.
9. **하지 말 것** — TabPFN(실관측 사전학습 아님을 운영진이 확인하기 전엔 위험), KMA/ERA5/Chronos 후손, 182사례 표면 위 residual cycle·HPO 추가, 분위수 블렌딩(RMSE엔 평균이 최적).

## 부록: 인용 파일
scripts/final_submission_20260905/P3/{train_model.py,predict_submission.py,run_submission.py}; notebooks/final_submission_20260905/P3/{TRAIN,PREDICT}.ipynb;
configs/final_submission_20260905.json; configs/compliance/p3_clean_incumbent_20260901.json; configs/experiments/p3_corrected_fixed_long_shrink_v4.json;
artifacts/official_final_submission_20260905/P3/contract.json; src/p3_wave/{data,features,models,validation,loss_router,persistence_shrink,corrected_fixed_long_shrink,
corrected_repeated_forward,final_inference}.py; scripts/{build_p3_refined_public_optimum_20260827,build_p2_p3_public_quadratic_round_g_20260827,build_adaptive_final_probes_20260829,
run_p3_corrected_fixed_long_shrink_v4,run_p3_corrected_repeated_forward_catboost_v1/v2,run_p3_component_loss_router,build_p3_loss_router_submission,qa_p3_cycle_generic,
run_p3_path_signature_residual_cycle_20260901_v23}.py; submissions/{p3_frozen_catboost,p3_lead_long_loss_router,p3_long_persistence_shrink}/manifest.json;
artifacts/p3_corrected_repeated_forward_catboost_v2/{metrics.json,oof.parquet}; artifacts/p3/{long_persistence_shrink,lead_long_loss_router,final_ensemble_validation}/metrics.json;
reports/{p3_clean_incumbent_reset_20260901_v1,p3_official_candidate_ledger_20260901_v1,leaderboard_clean_headroom_research_20260901_v1,p3_target_shift_retroaudit_20260828_v11,
p3_perfect_future_wind_oracle_20260829_v1,p3_kma_alpha_surface_sweep_20260829_v1,p3_catboost_confirmation_contract_repair_20260830_v3,p3_catboost_ordered_hpo_20260829_v1,
p3_v42_official_submission_20260901_v1,p3_kma_uniform_0425_official_submission_20260830_v1,official_information_probe_cycle_20260830_v1}; Downloads/해양 해커톤 제출용/{20260825_*,
20260826_round_D_*,20260827_round_G_*,20260827_P3_REFINED_PUBLIC_OPTIMUM_READY,20260829_ADAPTIVE_FINAL_PROBES_READY,20260829_P3_KMA_LEAD_SWEEP_READY}; P3 README.md:46-58; score.py.
