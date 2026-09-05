# P1 양측 재비교 및 학습량 진단 — canonical 연구 보고서

## 결론: 양측 평균 개선 후보 보존, 재현 QA PASS — 공식 성적은 미확인

2026-09-06 사용자가 통합 계획 실행을 승인했다. 기존 forward 실험과 공식 패키지는 그대로 보존하고
별도 `p1_tuning_twosided_20260906_v1`에서 같은 clean O/B와 B+bracket 절차를 비교한다.
**동일 H1_2025 전체 208,093행에서 F1 0.737031619 → 0.757196860, Δ+0.020165240**이다.
15 new historical fits +3 exact reused outer fits를1,443.422초(24.1분)에 완료했다.
별도PID36124(학습PID6776)의 확률·OOF·inner선택·학습곡선·CI·slice 재현 **63/63 PASS**다.
이미 연구에 노출된 기간을 fresh holdout이라고 부르지 않는다.
검증 스킬에 따라 지원 표본·수치 재현·성능 개선·공식 재현 적격성을 별도로 판정한다.

| 표준 양측 평가 | clean O/B | O + B(bracket) | ΔF1 |
|---|---:|---:|---:|
| primary H1_2025,208,093행 | 0.737031619 | 0.757196860 | +0.020165240 |
| H2_2024,172,638행 | 0.785328699 | 0.764304688 | −0.021024011 |
| H2_2025,287,862행 | 0.857021503 | 0.888307424 | +0.031285921 |
| 전체668,593행 pooled | 0.795140828 | 0.809641411 | +0.014500582 |

Primary paired-calendar-day bootstrap:181일/2,000회, seed20260906,
CI90 **[+0.005014770,+0.037606590]**, P(Δ>0)=0.993이다. 이는 과거표본의 기술적 bootstrap이며
공식 점수 상승 확률이 아니다. 새hardgate·자동승격은 없고 공식 점수 예상은 `null`이다.
Primary의TP는6,550→6,510,FP는1,313→774,FN은3,361→3,401이다. 재현율이 조금 낮아지는 대신 오탐이 줄었다.

위험도 함께 남긴다. 최대 fold 악화는 H2_2024 Δ−0.021024011, 최대 정점층 악화는
H2_2024 S-ORS L1 Δ−0.084545228(TP579동일,FP261→425)이다.
Primary 최악 calendar-day2025-02-20은864행/양성144에서F1 .165605096→0(TP13→0)이다.
전체 최악 day2025-09-30은1,367행/양성1에서1→.166666667(TP1동일,FP0→10)으로,
희소양성 분모에 민감한 day-level 수치임을 병기한다. 이 slice를 사후 제외하거나 별도정책을 고르지 않았다.

비교에서 **O는 동일한80열·같은모델**, B만80→107열로 바꾸며 `policy.select_inner`/`policies`를 그대로 쓴다.
따라서 features만 바꿔도 inner선택 정책/threshold가 달라질 수 있는 완성절차 비교다.
H2_2024 control/candidate는balanced/threshold.1, H1_2025는control O+B union vs candidate balanced.1,
H2_2025는control original.8 vs candidate balanced_union(B.6)이다. 성능을 보고 새 선택한 것이 아니다.

## 학습량 진단: 무조건 트리 수를 늘리는 근거는 약함

아래는 **fixed decoder.5 진단**이며 위 실제inner선택 완성정책 성적과 다르다. outer는700 그대로다.
서로 다른 outer의inner를 합친 값은 기술통계일 뿐 전체fold에 적용할 공통iteration 선택값이 아니다.

| arm/진단 pooled | 700 F1 | 1400 F1 | 관측 최고iteration | logloss700→1400 |
|---|---:|---:|---:|---:|
| O | .785529093 | .786378080 | 1400(ceiling) | .081737→.088466 |
| B | .781685795 | .774950348 | 700 | .101755→.111723 |
| B+bracket | .792035398 | .794085028 | 1000(F1 .794409211) | .099611→.107706 |

각 fold의관측best는 O=100/1400/1000, B=100/700/700, bracket=200/1000/1000
(H2_2024/H1_2025/H2_2025순)이다. 단일700최적 또는 일괄underfit이라고 할 수 없다.
B의700이후F1하락과 전arm의pooled logloss악화는 과도한학습/확률보정 악화와 일치하는 진단이나,
그것만으로 일반적 인과원인을 확정하지 않는다. O1400의F1증가도700대비+.000849로 작다.

## 지원과 배포 계약 위험을 분리

| fold | inner train/validation | inner unseen SL | outer train/validation | outer unseen SL |
|---|---:|---:|---:|---:|
| H2_2024 | 512843 /43626 | 0 |565539 /172638|0|
| H1_2025 | 432902 /60363 | 0 |514367 /208093|0|
| H2_2025 | 347004 /80736 |13342|444081 /287862|0|

두 측 학습지원으로 outer 미지원station-layer는0이지만 H2_2025 inner의13,342미지원행은 남아있다.
2026 배포의**year-key** lookup miss는 station-layer 지원과 별개의 계약 문제다.
이에 대한 synthetic+0fit 반사실 진단은 별도
[`p1_tuning_depth_stress_20260906_v2`](../p1_tuning_depth_stress_20260906_v2/report-source.md)에 남기며,
현재 표준 양측 결과와 모델·정책을 수정하지 않는다. 8×공동HPO는 이 계약위험 검토까지 미착수다.

## 사전 고정 계약

- 배포 train.csv만, 원본 immutable. 공식 입력/hidden/CSV/upload/Git0, CPU2/GPU0.
- outer는 기존 H2_2024/H1_2025/H2_2025와 같은 전체 검증 키다. 다른 배포 역사 시기의 학습 label을
  허용하는 **양측 retrospective 평가**이며 기존 forward 일반화 질문을 대체하지 않는다.
  양측21일purge와 전체 양성 run 배제를 적용한다. unsupported 행을 삭제하지 않는다.
- outer 학습행을 먼저 제한한다. inner는 `outer_start−21d`를 끝으로 한 직전60일이고,
  outer 학습 후보 안에서만 검증행을 선택한다. inner 학습은 다시 양측21일purge/whole-run 배제한다.
  정확 날짜는 이 산식으로 성능 노출 전에 고정하며 지원이 부족하면 날짜를 바꾸지 않고 중단한다.
- 원래80열과 B의 bracket27열(6/24/72h, flank1h)은 그대로다. 전처리/encoder/특징/rules/decoder 전에
  train/inner/outer 입력을 격리한다. 불연속 양측 구간을 하나의 연속 run으로 연결하지 않는다.
- 비교 자체의 O/B·bracket 예측은 **700트리**다. 기존 earlier-inner threshold/policy 선택 알고리즘을 사용한다.
  outer 성적을 보고 threshold나 iteration을 변경하지 않는다.
- 학습량 진단용 inner 모델만1,400트리까지 학습하고100/200/400/700/1000/1400 지점을 읽는다.
  fixed decoder threshold0.5 + 기존 low_ratio/rules/hysteresis를 고정한 pooled inner F1이 진단 primary,
  logloss는 보조다. F1동률이면 적은 iteration을 택한다. 이는 **실제 inner-selected decoder F1과 다른 진단**이다.
  최적이1,400이면 관측 ceiling이라고 표시하며 추가 연장하지 않는다. 이번 outer700에 결과를 적용하지 않는다.
- O/B native700 vs max1400의700prefix를 synthetic matrix에서 직접 학습·예측해 두 arm 모두 exact 확인했다.
  작은 합성4fit은 역사 데이터fit과 별도다. 이 검사로 prefix API/의미를 확인하며 과거 공식 답안 복원을 주장하지 않는다.
- control O/B의 모든 inner/outer 예측을 먼저 봉인하고 bracket을 진행한다.
  총18logicalfit(9inner +9outer), 정확한 기존 source/key/recipe/context/hash·확률 재현이 일치하는 outer만 재사용한다.
  예상 H2_2025 outer3fit 재사용→15new 역사fit이며, 재사용이 불가하면 새fit18개 상한이다.
- wall cap60분. 각 단계 시작/끝에서 확인하고 초과 시 partial terminal을 보존하고 종료한다.
  단일 이미 실행 중 fit을 강제 중단하지 않으므로 그 fit만큼 초과할 수 있다. 성능에 따른 조기 중단·자동 재시작 없음.
  초기 fit 실측으로 비용이 상한에 맞지 않으면 Root에 보고하며 범위를 사후 줄이지 않는다.
- primary는 같은 calendar H1_2025 전체 pooled F1. 전체3fold pooled와 fold/station-layer 위험, paired-day CI90
  (2000회, seed20260906)은 부표다. 평균 개선으로 후보 보존하며 .8hardgate나 자동 제출은 없다.

## 사전검증

합성 pytest **8 PASS**, O/B 실제 native700-prefix700 exact 포함. 양측 nested purge/outer배제,
전처리·encoder·feature·rules·decoder differential sentinel, inner pooled 산식/작은iteration tie-break를 검사했다.
Ruff의 E402는 thread환경변수를 numerical library import보다 먼저 고정하기 위한 명시적 예외로 처리했다.
실제 source 지원과 별도PID replay는 위에 기록했다. `result.json`은 실행결과 `COMPLETE_QA_PENDING`을
그대로 보존하며 후속PASS는 `independent-qa.json`으로 분리했다.
결과SHA `746c6a0a5987620bcd6fd3487ffeab617dd0ef78a605818c6dccd8f2e0610710`,
OOFSHA `37f39981e25e98a2eca5c227003ecf3aad746359a96d564e79a44f534b6ac644`,
QASHA `b27c3659123eea135b1d8702062c592ad5cab204f3952b5a2f246c4d91504871`.
공식입력/hidden/제출CSV/upload/Git0, 신규fit 재실행0이다. cleanroom전체훈련/공식답안 재현을 주장하지 않는다.

## 다음 공동탐색 — 자동 착수하지 않음

O/B 각각 기준 포함8설정의 후보표, iteration예산, fold별fit시간을 이 진단 후 제안한다.
700이 최적이라고 가정하거나 즉시16설정 탐색을 시작하지 않는다.
각 O/B 후보를 같은 nested inner에서 평가하고 동일 outer의 완성정책으로 확인해야 한다.
셀별 범위/정책은 별도사전검증 제안으로만 다루며, 학습 label0 min/max가 공식 FP0을 보장한다고 하지 않는다.

후속 설계 초안(아직 미실행/미봉인)은 다음과 같다. 각각 baseline에서 표의 한 항목만 바꾸고 seed/자료/특징/split은 고정한다.

| 설정 | O: XGBoost | B: LightGBM |
|---|---|---|
| 0 | 현 기준 depth7, min_child_weight20, lr.04 | 현 기준 leaves63, min_child_samples60, lr.035 |
| 1 | max_depth5 | num_leaves31 |
| 2 | max_depth9 | num_leaves127 |
| 3 | min_child_weight10 | min_child_samples30 |
| 4 | min_child_weight40 | min_child_samples120 |
| 5 | learning_rate.02 | learning_rate.0175 |
| 6 | learning_rate.06 | learning_rate.0525 |
| 7 | reg_lambda4 | reg_lambda4 |

- B의80열 또는107열 bank는 다음 계약 시작 전에 하나로 명시한다. 둘을 몰래 함께16설정으로 늘리지 않는다.
- 각 outer의 **자기 inner만** 사용해 iteration과 O/B 조합·decoder를 선택한다. 모든 outer의 inner를 합쳐
  global 설정을 고른 뒤 동일 outer를 평가하면 다른 fold의 outer label이 흘러들 수 있으므로 금지한다.
  현재 pooled inner best iteration은 설명용이며 이번 outer700 또는 다음 모든 outer에 자동 전이하지 않는다.
- 최대3outer×16설정=48innerfit +선택 O/B의3outer×2=6outerfit, 총54logicalfit이다.
  source/key/recipe/iteration이 일치한 현 baseline inner6fit은 재사용 가능하여 최대48newfit로 줄 수 있다.
  재사용 여부는 실제 future feature bank와 tree ceiling을 봉인한 뒤 확인한다.
- 최대1,400트리에서 현재와 같은 반복 수 후보를 inner-only로 선택하는 안을 우선한다.
  진단 best가 ceiling에 있으면 더 큰 ceiling은 별도 예산 협의 사항이며 자동 연장하지 않는다.
- O8×B8의64조합은 같은 inner 예측의 decoder bits를 재사용해 완성정책 F1로 비교할 수 있다.
  각 threshold decode는 한 번만 계산하고 union별 TP/FP/FN을 집계한다. 캐시가 기존 선택 알고리즘과 정확히 같은지
  synthetic parity를 추가한 뒤 사용한다. 모델은64쌍마다 새로 학습하지 않는다.
- 120분 cap과 Root의 패키지 CPU4 예약을 함께 고려한다. 본 실행의 fold/arm 실측fit시간과
  큰 depth/leaves 설정의 보수 비용을 기록하고, 상한이 맞지 않으면 설정을 성능 기반으로 골라 줄이지 않는다.
  자원 계약을 다시 정한 뒤 시작한다. 현재 이54logicalfit 탐색은 **0회 실행**이다.
- 셀별 train-normal 범위는 nuisance 교정 가설일 뿐, validation/official에서 FP0을 보장하지 않는다.
  다음 독립안에서는 train-only fit·미지원셀 fallback·정상 분포 이동·이상값이 정상 min/max 내부에 있는 toy를 먼저 검사하고,
  outer에서 범위나셀 정책을 선택하지 않는 nested 비교를 설계한다. 현재 모델에 clamp/gate를 추가하지 않는다.
