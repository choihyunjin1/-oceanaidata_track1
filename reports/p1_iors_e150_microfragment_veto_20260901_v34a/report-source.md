# P1 v34a I-ORS E150 microfragment veto

## 결론

`TERMINAL_NO_GO`. 길이 1–2인 I-ORS E150 파편만 제거하는 고정 후보는 Q3/Q4에서 모두 양수였지만, pooled day-block CI90 하한이 `-0.000026271`로 0보다 작아 사전등록 gate를 통과하지 못했다. 내부 중심값은 `ΔF1 +0.000283511`, 선형 예상 `+0.007535점`이지만 마지막 공식 슬롯에 쓸 만큼 강하지 않다. 공식 CSV 생성과 업로드는 금지한다.

더 근본적으로, v33a 공식 집계는 “pooled binary micro-F1 + 80행의 1→0 변경”과 수학적으로 양립하지 않는다. champion F1 `0.833548`, 양성 `6,396행`에서 80개를 모두 TP라고 가정해 삭제해도 가능한 최대 F1 하락은 `0.008571662`인데 실제 하락은 `0.010753`이다. 실제 하락은 이 이론 최대치의 `1.25448배`이며, 6자리 반올림을 허용해도 가능한 removed-TP 정수는 하나도 없다. 따라서 공식 v33a 결과로 I-ORS 80행의 TP/FP를 역산하거나 부분집합 점수를 정밀 예측할 수 없다.

## 비중복 one-shot

- 후보: `P1_IORS_E150_MICROFRAGMENT_VETO`
- discovery: Q2 I-ORS E150 addition 8개 segment 중 길이 2인 유일한 microfragment가 `0 TP / 2 FP`였다.
- action seal: Q3/Q4 truth를 붙이기 전에 station=`I-ORS`, incumbent=0, raw E150=1인 연속 segment 중 길이 `<=2`만 제거했다.
- 장구간 보존: 길이 3 이상 segment는 한 행도 제거하지 않았다.
- fit/threshold search/reselection/retry: `0/0/0/0`.

이 후보는 v33a의 all-I 80행 제거, v33b의 layer 2/3/4 제거, bootstrap-frequency veto, 다변량 logistic segment router와 동일하지 않다. 오직 미세 파편 morphology 하나만 질문하며 긴 E150 event를 bit-exact하게 보존한다. 과거 long-event proposal rescore는 infrastructure P0 때문에 scientific fit이 0회였으며, 이번 후보는 그 72-fit 패키지를 재실행하지 않았다.

## 결과

| fold | 제거 | removed TP | removed FP | ΔF1 |
|---|---:|---:|---:|---:|
| Q2 discovery diagnostic | 2 | 0 | 2 | +0.000161173 |
| Q3 outer | 11 | 3 | 8 | +0.000332296 |
| Q4 outer | 2 | 0 | 2 | +0.000217205 |
| Q3+Q4 pooled | 13 | 3 | 10 | +0.000283511 |

- removed TP share: `3/13 = 0.230769`, same-surface raw E150 F1/2 `0.453402`보다 낮음.
- day-block bootstrap: CI90 `[-0.000026271,+0.000597760]`, `P(improve)=0.936`, 164 blocks, 5,000 replicates.
- 예상 점수: center `+0.007535점`; interval을 선형 변환하면 약 `[-0.000698,+0.015888]점`.
- strict gate: pooled positive PASS, Q3/Q4 nonnegative PASS, removal precision PASS, anchor removal0 PASS, CI90-low nonnegative **FAIL**.

## 기술 복구와 QA

최초 runner는 action을 truth-blind하게 봉인한 뒤 bootstrap 결과 key를 `difference_ci90`으로 잘못 참조해 terminal technical failure가 났다. action을 재생성하거나 후보를 재선택하지 않고, 동일 sealed NPZ SHA-256 `713791ed6a030088283d7ac10e88c65ac584e55cf8fc6f31c3890bbe3ce041db`를 독립 metric replay해 위 결과를 회수했다. 독립 QA는 action-not-rebuilt, candidate-not-reselected, fit0, official/hidden/test/sample/CSV/upload0을 모두 확인해 PASS했다.

Materializer는 알고리즘 preflight만 구현돼 있으며 상태는 `BLOCKED_NO_GO`다. 공식 CSV를 읽거나 생성하는 execute 경로는 의도적으로 제공하지 않았다.

## 다음 판단

남은 공식 2회 중 하나를 이 후보에 쓰지 않는다. 우선 공식 scorer가 정말 pooled micro-F1인지, 혹은 v33a changed-row 집계가 누락됐는지 aggregate 수준에서 조정 가능한 영수증으로 확인해야 한다. 그 모순이 해소되기 전에는 v33a를 이용한 I-ORS 부분집합 TP 역산과 점수 기대값은 식별 불가능하다.
