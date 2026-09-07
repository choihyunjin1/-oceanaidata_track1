# Fable 실행 설계 — 남은 5슬롯 전부 사용 → 최종 제출 (2026-09-07 14:35 KST)

요청: `FABLE_ALL_REMAINING_SLOTS_PROMPT_20260907.md`. 이 문서는 연구·설계 문서다. 새 fit·모델 변경·CSV 생성·업로드·삭제·최종 지정·commit/push 0. 이번에 실행한 것은 기존 배포 학습 OOF(모두 이미 노출된 retrospective)와 배포 test 입력(`test_context.parquet`의 step 0 hs)만 읽는 0-fit 집계 7건(각 3~20초, 단일 스레드)이다. 원시 행·hidden·비배포 자료·credential은 읽지 않았다. `fresh_cold_2`(PID 26996, CPU4)는 progress만 읽었다(14:16 backbone 17/36).

## 결론 먼저

| 슬롯 | 주후보 (변경 하나) | 근거 요약 | 대체안 |
|---|---|---|---|
| **P2-1** | **L120 s3 + 층별 시간 이동평균(창 7스텝 = ±30분) → 기존 clip+PAVA 투영** | B3(계절 아날로그) 0.463529→0.441860 (−0.021669, 7일 블록 bootstrap CI90 [−0.02904, −0.01479], P=1.000); pooled −0.032; 8블록 중 7개 개선; T5 결측 자연행 0.302→0.273; outage 배열도 개선 | 창 13스텝(B3 −0.023, B7 악화 +0.035) |
| **P3-1** | **ff42a6a0 계보(GPU, Public 최고) + hs0 구간별 장기 리드 지속성 가중** w_long=[0.6,0.5,0.3,0.2,0.0] (hs0 구간 [1.5,1.6)/[1.6,1.7)/[1.7,1.9)/[1.9,2.4)/≥2.4) | 연도 교차 양방향 개선: 2024 선택→2025 평가 test-mix Δ −0.00347 (CI90 [−0.00716,+0.00022]) / 2025 선택→2024 Δ −0.00595 (CI90 [−0.00867,−0.00329]); 전 폴드 규칙 in-sample test-mix 0.64338→0.63541, 무가중 0.68354→0.67342 | 같은 규칙 + 단기 리드 확장(혼합 근거) |
| **P3-2** | **CPU completion_1(2015b387) 계보 + 같은 형태의 hs0 구간별 가중**(CPU OOF에서 선택, 값 동일 [0.6,0.5,0.3,0.2,0.0]) | CPU OOF 양방향 −0.00355 / −0.00440; 재현 증명 계보(빈 폴더 fresh cold 진행 중)에 같은 개선을 적용해 최종 선택 분기를 살림 | 준비된 no-shrink 70761aff(정보 가치; test-mix +0.0050 악화 예상) |
| **P1-1** | **원형 조합 그대로, B만 3-seed→5-seed 평균**(seed 2개 추가, 다른 부품 불변) | seed별 B F1 0.86248/0.86401/0.86490 vs 3-seed 0.86467: 분산 축소 여지는 작지만 존재. 위험 최소·재생성 확실. 점수 도전은 약하고 정보 가치 명시 | 원형 그대로 O만 O_slow(lr 0.02, 1,400 trees)로 교체 |
| **P1-2** | **원형 조합 그대로, O만 O_slow(lr 0.02, 1,400 trees)로 교체**(B·MS·셀·GI 불변) | 트리 단독 Q2~Q4 pooled +0.001631(core tuning), MS 결합 Q3/Q4 −0.000302 — 약한 정보 가치 후보. Public에서 MS 결합 효과를 실측 | P1 슬롯 1개 예비(무의미 중복 금지) |

**Codex 초기안 판정**: P1-A(train-OOF 결합정책 재학습) **기각** — 같은 규칙(추가 셀 = O-only 정밀도 > F1_B/2, 제거 셀 = B-only 정밀도 < F1_B/2)을 fold 하나씩 빼고 고르면 3/3 fold에서 B보다 낮다(§B-3). P2-A(C60+L120 혼합) **기각** — pooled는 개선하나 B3에서 +0.0033 악화, 중첩 선택이 방향을 뒤집는다(§B-2). P3-A no-shrink **대체안으로 강등**. P3-B router 제거 **기각** — 무가중 −0.003이나 test-mix +0.001(§B-1).

### Codex가 바로 실행할 5개 항목 (순서 = 업로드 순서)

1. **P2-1 smoothing 후보 생성**: `P2_L120_s3_proj_v2` 저장 모델 base 추론 → 층별(2/3/4) 10분 격자 중심 이동평균 창 7(min_periods=1, 예측 없는 시각은 창 계산에만 결측으로 참여) → 기존 clip+PAVA 투영(순서 고정: 평활 후 투영) → 답안 QA → 별도 PID replay → 패키지(기존 `02_code/projection/` 곁에 `smoothing/` 격리) → 업로드 준비. 새 fit 0.
2. **P3-1 조건부 shrink 후보 생성**: ff42a6a0 계보 `P3_SCORED_SAVED_MODELS_v2` 추론 경로에서 `recipe["shrink"]`의 고정 0.2 대신 hs0 구간별 가중 적용(§A 규칙 원문). hs0 = persistence 값(step 0 hs)이므로 추가 입력 없음. 단기 리드 불변. 답안 QA → replay → 패키지. 새 fit 0.
3. **P3-2**: 같은 addon을 CPU completion_1(2015b387) 저장 모델에 적용(`noshrink.py`의 `apply_long_lead_persistence_shrink` 캡처 패턴 재사용). fresh_cold_2 완료 여부와 무관하게 completion_1 모델로 생성하고, 완료 시 SHA 대조만 기록.
4. **P1-1 B 5-seed**: seeds 20260813/20260829/20260847 + 신규 2개(사전 고정, 예: 20260861/20260875). 내부 근거 = Q2/Q3/Q4 fold B fit 6개 + 원형 O·MS 고정 OOF 결합 replay(MS bridge 0-fit 도구) → pooled/fold/월별 보고 → full B 2 fit → 원형 조합 추론 → 패키지(590 MB 분할). 새 fit 8.
5. **P1-2 O_slow**: O(lr 0.02, 1,400 trees) fold 3 + full 1 = 4 fit, 나머지 P1-1과 동일 절차. P1-1과 부품이 다르므로 중복 아님.

막히면 대체 순서: P2-1 실패(패키지 fingerprint/추출 replay) → 창 13 아닌 **같은 창 7을 상위 폴더 격리로 재포장**(모델·코드 불변) → 그래도 실패면 P2 예비. P3-1 실패 → P3-2 먼저 → P3-1 재시도. P3-2 실패 → no-shrink 70761aff(준비 완료). P1-1/P1-2 실패 → 남은 P1 슬롯 예비(무의미 중복 금지). 각 후보는 사전등록(ID·입력/모델 해시·규칙 원문·비교 키·평가 표면·예산) 후 생성한다. 공식 결과를 본 뒤 그 후보의 계수를 바꾸지 않는다.

시간: 14:35 기준 P2-1·P3-1·P3-2는 0-fit이라 각 30~45분(QA·패키지 포함), P1-1·P1-2는 fit 12~15분 + 조합 replay + 590 MB 포장 각 45~60분. fresh_cold_2는 17:30~18:00 전후 완료 예상(14:16 17/36). 마감 시각 미확인이므로 P2-1 → P3-1 → P3-2 → P1-1 → P1-2 순으로 완성 즉시 업로드한다.

## A. 5슬롯 실행안

형식: `문제/슬롯 | 변경 하나 | 비교군 | 기존자산 절대경로·SHA | 선행 실패와 다른 점 | 내부평가/선택 분리 | 신규fit·실측기반예산 | 점수도전/정보가치 | 위험 | 패키지변경범위 | 대체안`

| 문제/슬롯 | 변경 하나 | 비교군 | 기존자산 | 선행 실패와 다른 점 | 내부평가/선택 분리 | 신규fit·예산 | 점수도전/정보가치 | 위험 | 패키지변경범위 | 대체안 |
|---|---|---|---|---|---|---|---|---|---|---|
| **P2-1** | L120 s3 원 예측에 층별 중심 이동평균 창 7(10분 스텝) 적용 후 clip+PAVA | 9c5fec38 (0.405920℃) | `C:\Users\cedis\Documents\OceanFinalDay_20260907\P2_L120_s3_proj_v2\` SAVED_MODELS.zip `e806d1f4…`, SOURCE_ONLY.zip `cb2fbf6c…`; OOF `artifacts/p2_c3_multiseed_completion_20260906_v1/evaluation.npz` `natural_L120` | s10(seed 확대)·C60 혼합은 B6에서만 이겼고 B3 악화. 이 변경은 **B3 −0.0217, 8블록 중 7개 개선, T5 결측·outage 배열 동시 개선**. 학습·seed·투영 규칙 불변 | 창 7은 격자 {1,7,13,19,…,73} 중 B3와 7/8 블록을 동시에 개선하는 최소 창으로 고정. 창 크기는 블록마다 최적이 달라(B1/B2는 73, B7은 1) 중첩 선택은 불안정 → **창은 사전 고정값(±30분)** 이며 그 사실을 기록. 평활은 투영 전(투영 후 평활은 B3 0.449로 열세) | 0 fit. base 추론(기존 43.7초 replay 수준) + 평활/투영 수 초 + QA·패키지 30~45분 | **점수 도전**. B7(2025-08) +0.0144 악화는 별도 위험 | test 창 가장자리 30분은 이웃이 적음(min_periods=1). test 격자 존재율 ≈ 99%로 OOF(66%)보다 유리. 10분 고주파 신호 일부 손실(진실 자체 변동 RMS 0.249℃, 예측 0.186℃) | `02_code/smoothing/` 신규 + PROJECT 단계 호출 1줄. 모델·config·기존 투영 코드 불변. manifest/ZIP 재생성·추출 replay 필수 | 창 13(B3 0.44020, B7 +0.0346) |
| **P3-1** | 장기 리드(12/18/24h) 지속성 가중을 고정 0.2 → hs0 구간별 [0.6,0.5,0.3,0.2,0.0] | ff42a6a0 (0.604351 m) | `OceanFinalRelease_20260907\P3\P3_SCORED_SAVED_MODELS_v2\` (infer.py:57 `recipe["shrink"]`), SAVED ZIP `2c9cddf4…`; OOF `artifacts/p3_numeric_lead_forward_gpu_20260906_v2/candidate_oof.parquet` | 전역 shrink 격자는 2024 선택→2025 +0.011 악화로 기각됐다. **hs0 조건부는 양방향 개선**(2024→2025 −0.00347, 2025→2024 −0.00595 test-mix). no-shrink는 test-mix +0.0050 악화 | 구간·격자 사전 고정(구간 = test hs0 분위 근거, 가중 격자 {0,…,0.6}). 배포 가중은 전 폴드 OOF에서 구간별 최소 RMSE로 선택(retrospective). 일반화 근거는 연도 교차 두 방향 | 0 fit. 저장 모델 추론 6초 + addon + QA·패키지 30~45분 | **점수 도전**. 리드 12/18/24 각각 개선, 단기 리드 불변, 5구간 모두 무가중 개선 또는 동일 | 규칙은 노출 OOF에서 고른 사후 계수(train-only). onset 축 미보정. whole-cold 재학습 불일치(660행)는 계보 고유 위험으로 그대로 | `02_code/condshrink/` 신규, `infer.py`의 shrink 호출을 규칙 함수로 교체. 모델 불변 | 단기 리드까지 확장(w_short [0.4,0.4,0.3,0.3,0.1]; 2025→2024 −0.0099 개선이나 2024→2025 −0.0002로 혼합) |
| **P3-2** | 같은 hs0 조건부 가중을 CPU completion_1 모델에 적용 | 2015b387 (0.609836 m) | `OceanFinalDay_20260907\P3_numeric_cpudet_unbounded_v2\completion_1\` 모델 36+router 5; OOF `completion_1/04_logs/candidate_oof.parquet` SHA `bede4edf…`; addon 패턴 `P3_numeric_cpudet_noshrink_v2\SOURCE_ONLY\02_code\noshrink.py` | no-shrink(w=0)는 저파고 악화. 조건부는 저파고에 더 큰 가중·고파고에 0 | CPU OOF 양방향 −0.00355 / −0.00440; 전 폴드 규칙 test-mix 0.64153→0.63422 | 0 fit. 추론 + QA·패키지 30분 | 점수 도전 + **재현 증명 계보 유지**(fresh_cold_2 완료 시 SHA 대조) | ff42 계보와 오차 상관 0.999 → Public 차이는 계보 자체(0.0055) 안에서 움직일 것. 최종 선택은 사용자 | noshrink addon과 같은 격리 방식 | 준비된 70761aff no-shrink(정보 가치) |
| **P1-1** | 원형 B 3-seed → 5-seed 평균(seed 2개 추가) | 57844ef2 (F1 0.833548) | `OceanFinalRelease_20260907\P1\SOURCE_ONLY` (`02_code/tree_recipe.json` B_parameters, `run.py`), 원형 B·O·MS 저장 모델; OOF `artifacts/p1_matched_budget_local_compare_20260825_v1/predictions.parquet` seed별 B 열 | bracket(특징 추가)·T/S·flank·decoder·셀 정책은 모두 악화 또는 Public 악화. 이 변경은 **특징·정책·임계 불변**, 분산만 축소 | Q2/Q3/Q4 fold B fit 6개 → 원형 O·MS 고정 OOF 결합 replay(`p1_core_tuning_ms_bridge` 도구, 0-fit) pooled/fold/월별 보고. 선택은 사전 고정(5-seed 등가 평균), 결과로 seed를 고르지 않음 | 8 fit(B ≈ 80~100초, CPU4·fresh_cold_2와 경합) ≈ 15분 + replay + 590 MB 포장 45~60분 | **정보 가치 위주**(seed별 F1 폭 0.0024) | 개선 폭이 Public 잡음보다 작을 수 있음. 5~6월 자연 급변 FP·장기 offset/drift FN(§B-3) 구조는 그대로 | B seed 목록·full B 2개 추가. tree_recipe.json B seeds 갱신, MS·O·셀·GI 불변 | O_slow 교체(P1-2) |
| **P1-2** | 원형 O(lr 0.04, 700 trees) → O_slow(lr 0.02, 1,400 trees) | 57844ef2 | 위와 동일 + `reports/p1_core_tuning_20260906_v1`(O_slow 트리 단독 +0.001631, MS 결합 −0.000302) | core tuning은 CPU2 새 기준선 위 비교였고 원형 조합 답안은 만들지 않았다. 이번은 **원형 부품에 O만 교체**한 실제 답안 | fold 3 fit → 원형 셀 조합(hist cells) + MS bridge replay로 pooled 보고. 임계 0.2/0.1·minrun 12 불변 | 4 fit(≈ 2~3분/fit) + replay + 포장 45~60분 | 정보 가치(트리 단독 이득이 MS 결합·Public에서 남는지) | MS 결합 내부 −0.0003 → 악화 가능. 실패 시 원형 유지 | O 모델 파일 교체, tree_recipe O 파라미터 갱신 | P1 슬롯 예비 |

**공통 규정 경계**: 배포 자료만, hidden truth 0, 공식 점수로 계수/임계 변경 0, 같은 SHA 재제출 0, 각 후보 사전등록 → 내부평가 → 답안 QA(schema/key/order/finite/중복/기존 제출 SHA 대조) → 별도 PID replay → SOURCE_ONLY/SAVED_MODELS → 업로드 → 화면 채점 receipt.

### 규칙 원문 (사전등록용)

P2-1: `pred_smooth[layer, t] = mean(pred[layer, t+k·10min] for k in −3..3 where prediction exists)`; 이후 기존 `project_profiles`(complete 2/3/4층 + finite T1·deep, clip → PAVA). 불완전 프로필·endpoint 결측 처리는 기존 규칙 그대로.

P3-1/P3-2: `hs0 = persistence(step 0 hs)`; 구간 b = digitize(hs0, [1.6, 1.7, 1.9, 2.4]) → w_long[b] = [0.6, 0.5, 0.3, 0.2, 0.0][b]; hs0 < 1.5는 b=0으로 처리(test 사례에는 없음). 리드 12/18/24: `final = (1 − w)·routed + w·persistence`, clip [0,30]. 리드 3/6/9 불변. routed·router·clip·seed 집계는 기존 코드 그대로.

## B. 오늘 실행한 최소 진단 (0-fit, 집계만)

모든 비교는 기준·후보가 같은 키·같은 분모(P3 103,602행/17,267 anchor 정확 병합, P2 166,268행 동일 npz)임을 먼저 확인했다. Public 점수는 어떤 계수에도 쓰지 않았다.

### B-1. P3

- **계보 혼합(GPU final ↔ CPU final)**: 오차 상관 0.999. 0.5 혼합은 GPU 대비 test-mix −0.00119(CI90 [−0.00171,−0.00065])이나 CPU 대비 +0.00069. 사실상 같은 모델 → 슬롯 가치 없음.
- **router 제거(single·multi 평균 후 shrink)**: 무가중 −0.00298(P 0.994)이나 test-mix +0.00103(CI90 [−0.00040,+0.00243]). no-shrink와 같은 "고파고에서만 이득" 구조 → 기각. (router는 두 arm의 평균이 아니며 53%의 행에서만 평균과 일치.)
- **hs0 조건부 shrink** (구간 [1.5,1.6)/[1.6,1.7)/[1.7,1.9)/[1.9,2.4)/≥2.4, test 비율 0.40/0.20/0.165/0.135/0.10):

| 계보 | 선택→평가 | 선택된 w_long | test-mix 평가 (후보 vs 0.2) | 무가중 평가 | bootstrap(episode 951, 1,000회) |
|---|---|---|---:|---:|---|
| GPU | 2024→2025 | [0.6,0.6,0.4,0.2,0.0] | 0.64547 vs 0.64894 | 0.66868 vs 0.68231 | rew Δ −0.00347 CI90 [−0.00716,+0.00022] P 0.936; unw Δ −0.01371 P 1.000 |
| GPU | 2025→2024 | [0.4,0.4,0.1,0.1,0.0] | 0.63282 vs 0.63883 | 0.68142 vs 0.68465 | rew Δ −0.00595 CI90 [−0.00867,−0.00329] P 1.000 |
| CPU | 2024→2025 | [0.6,0.6,0.4,0.2,0.0] | 0.64551 vs 0.64904 | 0.66724 vs 0.68097 | rew Δ −0.00355 CI90 [−0.00725,+0.00014] P 0.944 |
| CPU | 2025→2024 | [0.4,0.4,0.1,0.0,0.0] | 0.63091 vs 0.63535 | 0.68048 vs 0.68164 | rew Δ −0.00440 CI90 [−0.00739,−0.00146] P 0.986 |
| GPU | 전 폴드(배포 규칙, in-sample) | **[0.6,0.5,0.3,0.2,0.0]** | 0.63541 vs 0.64338 | 0.67342 vs 0.68354 | 리드 12/18/24: 0.6469→0.6400 / 0.7670→0.7512 / 0.8146→0.7970 (test-mix) |
| CPU | 전 폴드 | **[0.6,0.5,0.3,0.2,0.0]** | 0.63422 vs 0.64153 | 0.67145 vs 0.68132 | 구간별 무가중: 0.6098→0.6007, 0.6253→0.6197, 0.6453→0.6440, 0.6687→0.6687, 0.7452→0.7232 |

단기 리드 확장(w_short 격자 {0,…,0.4}): 2025→2024는 test-mix −0.0099(CI90 [−0.0133,−0.0067])로 크게 개선, 2024→2025는 −0.0002(CI 0 포함)로 중립 → 대체안으로만.
- **no-shrink 70761aff**: test-mix +0.005033(Codex 재현 +0.005032527) → 주후보 아님, P3-2 대체안.

### B-2. P2

| 진단 | B3 | pooled | 판정 |
|---|---:|---:|---|
| L120 s3 + 투영(기준) | 0.46353 | 1.23673 | — |
| C60+L120 0.5 혼합 + 투영 | 0.46676 (+0.0033, 7일 bootstrap CI90 [−0.0022,+0.0091], P 0.169) | 1.22616 | 기각. B3 오차 상관 0.935, 개선은 B6(−0.05)에서만. 중첩: all-but-B3 선택 → C60 단독(a=0)이 B3 0.4829로 악화; 2024 선택 0.8 → 2025 개선, 2025 선택 0.0 → 2024 악화(불안정) |
| 층별 상수 편향 보정(all-but-B3 학습 → B3) | 0.4907 | — | 기각(계절 반전: 층4 잔차 +0.216 vs B3 −0.173) |
| envelope 중점으로 수축 a=0.1 | 0.46539 | 1.22996 | 기각(B3 악화) |
| **이동평균 창 7 → 투영** | **0.44186** (−0.02167, CI90 [−0.02904,−0.01479], P 1.000) | 1.20468 (−0.03205, CI90 [−0.04237,−0.02215]) | **채택**. 블록별 Δ: B1 −0.084, B2 −0.045, B3 −0.022, B4 −0.002, B5 −0.006, B6 −0.021, **B7 +0.014**, B8 −0.014. 층별 2/3/4 모두 개선. T5 결측 자연행 0.302→0.273. outage 배열 pooled 1.290→1.256, B3 0.464→0.443 |
| 이동평균 창 13 | 0.44020 | 1.19642 | 대체안(B7 +0.035) |
| 투영 후 평활(창 7) | 0.44896 | — | 열세 → 순서는 평활→투영 |

창 선택의 한계: 블록별 최적 창이 1~73으로 갈라져 중첩 선택(all-but-B3 → 73)은 B3를 악화시킨다. 따라서 창은 "격자 중 B3와 7/8 블록을 동시에 개선하는 최소 창"으로 **사전 고정**하고, 이 선택이 노출 OOF에 근거함을 명시한다. C60에도 같은 창이 B3 0.4829→0.4521로 작동해 모델 특이 현상이 아니다.

### B-3. P1

- 원형 부품 OOF(Q2/Q3/Q4 2025, 421,032행): B 0.86467, O 0.86048, 역사적 셀 router 0.86690(fold별 +0.0004/+0.0054/−0.0002), O∪B 0.86204.
- **train-OOF 셀 정책 재학습(Codex P1-A)**: 규칙(추가 셀 = O-only 정밀도 > F1_B/2, 제거 셀 = B-only 정밀도 < F1_B/2, n≥20)을 fold 하나씩 빼고 고르면 outer F1이 B보다 낮다: q2 0.78046 vs 0.79186, q3 0.88988 vs 0.88963(동률), q4 0.91156 vs 0.91457. 셀별 불일치 정밀도가 시간에 따라 불안정(S-ORS/3 O-only 0.547, S-ORS/6 0.458 등 경계값) → **기각**. 역사적 셀의 +0.0022도 같은 표면의 선택 결과로 해석한다.
- psal 동반 이동 규칙(router 양성 run 중 최대 |dT| 지점의 |dS|가 임계 초과면 제거): FP run은 |dS|>0.1 비율 0.42, TP run 0.15로 분리력은 있으나 run 단위 제거가 TP run을 크게 잘라 LOFO에서 q3 0.895→0.825 붕괴 → **기각**.
- 월별 B: 4월 recall 0.671(FN 782, offset/drift), **6월 precision 0.700/recall 0.500**(FN = S-ORS/2 drift 333행·S-ORS/6 offset 329행 전체 미탐; FP = S-ORS/1 160·S-ORS/2 82), 7월 recall 0.712. test는 5~6월 55% → 손실 본체는 장기 offset/drift 통째 미탐이며 이번 0-fit 자산으로 고칠 수 없다. bracket이 이를 겨냥했으나 Public에서 악화.
- seed별 B F1 0.86401/0.86490/0.86248 vs 3-seed 0.86467 → P1-1(5-seed)의 기대 폭은 작다. O 변형: O_slow 트리 단독 +0.001631(core tuning), MS 결합 −0.000302 → P1-2는 정보 가치.
- 검토했으나 제안하지 않는 것: MS 제거·부분 제거(08-27~08-30 공식 음성 확인), O∪B/AND 탐침(금지), 범위 룰(철회), T/S 특징(−0.009), 양측 학습 B(평가용 개념이라 배포 변경 없음).

## C. 최종 선택과 제출 절차

### C-1. 후보별 기록 항목 (문제당 표)

Public 점수 / 내부 평균·CI·최악 블록 / 재현 증거(saved replay·빈 폴더 cold·별도 PID) / 규정 적격성(배포 자료·리터럴 출처·계수 유래) / 미확인(6시간 일반 적용, 검증 대상 답안 선택 방식). 네 축을 합산하거나 한 축으로 다른 축을 덮지 않는다.

### C-2. 선택 규칙

1. 문제별 최종 답안 = 적격·재현 QA PASS 후보 중 **Public 최고**. 리더보드 집계가 문제별 최고 Public이므로, 새 후보가 기존 최고를 넘지 못하면 기존 최고가 그대로 대표한다. 어떤 답안이 재현 검증 대상인지는 미확인이므로, 최고 Public 답안의 패키지를 최종 제출하되 **대체 후보 패키지도 보존**한다.
2. P3는 두 계보(ff42 GPU: Public 우위·whole-cold 660행 불일치 / CPU: fresh cold 대조 예정)가 남는다. P3-1·P3-2 채점 뒤 Public·재현 증거를 나란히 두고 **사용자가 선택**한다. 이 문서는 어느 쪽도 우선하지 않는다.
3. 내부 평균 개선이 있었으나 Public이 낮은 후보는 기록만 남기고 대표로 쓰지 않는다. 반대로 Public이 높은데 재현 QA를 통과하지 못한 후보는 대표로 쓰지 않는다.
4. 공식 점수로 계수·창·가중을 재조정하지 않는다. 슬롯이 남았을 때 다음 후보를 고르는 판단(모델 선택 참고)과 구분한다.

### C-3. 완료 체크리스트

- [ ] 5개 후보 각각: 사전등록 JSON(ID, 입력·모델 SHA, 규칙 원문, 격자, 평가 표면) → 내부평가 result.json → 답안 QA(26,061/1,200/169,011행, 키 순서, finite, 중복, 기존 제출 SHA와 불일치) → 별도 PID replay exact → SOURCE_ONLY/SAVED_MODELS ZIP + 추출 replay → 업로드 → 화면 채점 receipt(시각·지표·점수·SHA).
- [ ] 각 문제 남은 횟수 0을 실제 화면으로 확인(receipt 갱신).
- [ ] 최종 선택표(C-1) 작성 → 선택 버전 release 폴더: 01_data 참조 / 02_code / 03_model / 04_logs·환경·QA / 05_answer / README·FORM. 기존 `OceanFinalRelease_20260907`·fallback·실패 증거 덮어쓰지 않음.
- [ ] 파일당 50 MB: P1 SAVED 590 MB는 분할(각 ≤ 50 MB) + REASSEMBLY_MANIFEST + reassemble.py + 재조립 SHA 검증. 첨부 개수 상한은 미표시이므로 분할 수를 줄일 수 있으면 줄인다(45 MB 목표는 로컬 보수치).
- [ ] README/FORM의 답안 SHA = 채점 화면에 올린 CSV SHA = 패키지 05_answer SHA. "6시간"은 우리 PC 실측으로만 표기, 일반 적용 범위 미확인 명시.
- [ ] fresh_cold_2 완료 시 답안 SHA를 2015b387과 대조해 CPU 패키지 문서에 기록(불일치면 원인·보존).
- [ ] 과거 비적격 제출 삭제는 별도 사용자 결정(`USER_HANDOFF_DELETION_CHECK_20260907.md`). 이번 실행 목표에 삭제 승인은 포함되지 않는다.
- [ ] **모델 최종 제출은 마지막**: 세 문제 모두 답안 업로드가 끝난 뒤, 제목·요약·파일·저장소 URL 입력 → 제출 → 제출관리 접수 확인(응답이 불확실하면 재클릭 전에 접수 목록 확인).
- [ ] 마감 시각이 밝혀지면 `새 작업 중단 = 마감 − 포장·업로드 실측 − 여유`로 계산해 알린다.

## 부록. 재계산 요지 (재현용)

- P3 test-mix 가중: 구간별 `w = test 사례 비율 / OOF 행 비율`, `RMSE_w = sqrt(Σw·e²/Σw)`; station×episode paired bootstrap seed 20260907. 조건부 규칙 선택: 구간 b마다 `argmin_w RMSE(clip((1−w)·routed + w·persistence), rows in b ∩ long leads ∩ 선택 폴드)`, `routed = (final − 0.2·p)/0.8`.
- P2 평활: `pd.Series.rolling(win, center=True, min_periods=1).mean()`을 10분 완전 격자 위 층별로 적용, 예측 없는 시각은 결측으로 두고 결과도 원 예측 위치만 채움; 이후 검토 2 §3의 exact 3점 PAVA 투영. 7일 시간 블록 bootstrap 2,000회 seed 20260907.
- P1: `artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet`(label, fold) + `artifacts/p1_matched_budget_local_compare_20260825_v1/predictions.parquet`(이진 arm) + `artifacts/p1_current_router_oof_anchor_v1/anchor.parquet`(router) 병합 421,032행; psal 규칙은 배포 `train.csv`의 temp/psal 10분 차분만 사용.
