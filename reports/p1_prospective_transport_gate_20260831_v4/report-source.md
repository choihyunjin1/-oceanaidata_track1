# P1 prospective transport gate v4 — 방법론 감사

## 결론

v28의 `max KST-day changed fraction <= 0.5%`와 `every supported station-layer delta F1 >= 0`는 **공식 pooled row-level binary F1의 hard gate로는 과도하다**. 둘 다 future-only v4부터 `diagnostic warning`으로 내린다. 다만 v28을 소급 PASS로 바꾸지는 않는다. v28의 원래 판정 `NO_GO_SAFETY_GATES`와 artifact는 그대로 보존한다.

v4는 최소 `+0.01` calibrated point, family-aware raw point gate, pooled F1, KST-day dependence-preserving bootstrap, Q3/Q4 nonnegative, anchor removal 0을 그대로 유지한다. 이 감사의 두 대상 외에 v28의 overall changed fraction과 station-layer-quarter concentration gate도 별도 감사 전까지 그대로 유지한다.

## 정확한 prospective 판정식

future candidate의 hard PASS는 다음의 논리곱이다.

```text
Level-0 validity PASS
AND pooled delta F1 > 0
AND Q3 delta F1 >= 0
AND Q4 delta F1 >= 0
AND paired KST-day bootstrap CI90 low > 0
AND bootstrap P(delta F1 > 0) >= 0.8
AND raw expected points >= 0.015383691373120248
AND calibrated expected points >= 0.01
AND anchor removals == 0
AND additions > 0
AND addition precision > incumbent F1 / 2
AND overall changed fraction <= 0.005
AND max station-layer-quarter addition concentration <= 0.5
```

그 뒤 두 diagnostic을 붙인다.

- hard PASS + diagnostic 이상 없음: `PASS_PRIMARY_NO_WARNING`
- hard PASS + max-day 또는 station-layer 경고: `PASS_PRIMARY_WITH_TRANSPORT_WARNING`
- hard gate 하나라도 실패: `NO_PASS_PRIMARY_GATE`
- lineage/leakage/access 등 Level-0 실패: `QA_BLOCKED`

따라서 diagnostic 경고는 반드시 공개하지만 pooled 판정을 뒤집지 않는다. 이 식은 [정책 JSON](../../configs/goals/p1_prospective_transport_gate_20260831_v4.json)에 봉인했다.

## 공식 metric과 목적함수 정렬

배포 문제문 보존본 `00_MUST_READ_FIRST.md`와 프로젝트 `README.md`는 P1의 순위 지표를 이상 행에 대한 **binary F1**으로 명시하고, `anomaly_type`은 순위에 반영하지 않는다고 적는다. 공식 사이트도 상세 평가 기준이 문제별 화면과 배포 데이터에 제공됨을 안내한다. 즉 공식 목적은 station/layer macro-F1이나 worst-day F1이 아니라 전체 제출 행의 pooled binary F1이다. [대회 공식 사이트](https://oceanaidata.org/)

기존 governing policy도 station/layer/window를 transport diagnostic으로 두고, official mixture·명시적 safety objective·정량화된 operational cost가 있을 때만 hard requirement로 올리도록 정했다. 현재 공식 test row 또는 hidden truth를 읽지 않았으므로 그런 추가 목적을 입증할 자료가 없다.

## 왜 max-day 0.5%가 hard veto로 과도한가

v28 결과를 보지 않고 train-only Q3/Q4 surface를 집계했다.

- 평가 surface: `287,862`행, `163` KST days
- 일별 행수: 최소 `133`, 중앙 `1,728`, 최대 `2,160`
- 동일한 0.5%를 정수 action으로 바꾸면 허용량은 `0`~`10`건이다.
- 최소일에는 1건만 바꿔도 `1/133 = 0.7518797%`여서 gate를 통과할 수 없다.

따라서 이 비율은 pooled F1 risk가 같아도 일별 denominator에 따라 action 한 건의 허용 여부가 불연속적으로 달라진다. 공식 mixture나 비용 함수에서 유도된 수치도 아니다. future v4에서는 최대/quantile, 초과 일수, action-day HHI와 최대 share를 보고하되 자동 veto로 사용하지 않는다.

## 왜 every supported station-layer >= 0이 hard veto로 과도한가

Q3/Q4에는 station-layer `16`개가 있고, 각 cell positive prevalence는 `0.007332`~`0.103938`, 약 `14.18배` 범위다. 모든 cell의 관측 point estimate가 0 이상이어야 한다는 조건은 pooled 행별 목적에 단순 guard를 더하는 것이 아니라 **최악 그룹 목적을 추가**한다.

Sagawa et al.의 group DRO는 사전 정의 그룹의 worst-case loss를 최소화하는 별도의 minimax 문제로 정의한다. 평균위험을 평가하는 문제에서 worst-group 목적을 도입하려면 group robustness 자체가 명시적 목표여야 한다. 또한 작은 그룹의 generalization gap이 다르므로 group adjustment가 필요하다고 설명한다. 따라서 16개 point estimate의 최소값을 uncertainty·multiplicity 없이 hard zero에 비교하는 것은 공식 pooled F1과 정렬되지 않는다. [Sagawa et al., ICLR 2020](https://arxiv.org/abs/1911.08731)

future v4는 모든 station-layer의 row/positive/TP/FP/FN, delta F1, 가능한 dependence-aware uncertainty와 worst point를 계속 보고한다. unseen/unsupported deployment category는 Level-0 blocker다. 다만 알려진 official mixture나 safety loss가 없는 한 관측된 slice 음수 하나로 pooled hard PASS를 뒤집지 않는다.

## pooled uncertainty와 Public transport risk는 유지한다

시간 의존 자료에서 IID row bootstrap은 부적절할 수 있다. Politis와 Romano의 stationary bootstrap은 연속 관측 블록을 이용해 weakly dependent time series의 표준오차와 confidence region을 구성하는 근거를 제공한다. 이 프로젝트는 그 원칙에 맞춰 joint KST-day block을 유지한다. [Politis & Romano, JASA 1994](https://doi.org/10.1080/01621459.1994.10476870)

Public probe는 local→Public transport가 1:1이 아님을 보여 준다.

- 2026-08-28 P1 세 후보의 Public F1은 `0.833548`, `0.833548`, `0.833333`이었다.
- 2026-08-31 내부 `+0.001380986 F1` 후보는 공식 best와 동률이었다.

그러나 이 aggregate probes 어디에도 universal max-day cap이나 every-slice nonregression이 Public F1을 예측한다는 증거는 없다. 따라서 v3의 same-problem transport penalty `0.005383691점`과 calibrated `+0.01점`은 유지하되, 두 unsupported surrogate veto와 혼합하지 않는다. v3 자체는 P1 official pair `n=1`인 provisional floor이며 완전한 confidence interval이 아니다.

## 비소급성과 selection-bias 방지

v28은 이미 관측된 development result다. 그 수치에 맞춰 threshold를 새로 정하거나 v28을 PASS로 바꾸면 selection criterion을 exposed surface에 맞추는 셈이다. Cawley와 Talbot은 finite-sample model-selection criterion을 반복 최적화하면 selection bias와 성능평가 과적합이 생길 수 있음을 보인다. 따라서 v4는 **정책 이후 새로 봉인되는 candidate에만** 적용한다. [Cawley & Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)

v28 수치는 이 감사의 대상이지만 v4의 numeric threshold 원천이 아니다. v28의 pooled improvement, day maximum, worst slice 수치는 policy JSON의 threshold로 들어가지 않았고 독립 QA가 이를 확인한다.

## 한계와 다음 사용법

- official test mixture와 hidden truth를 읽지 않았으므로 station/layer가 실제 안전·운영 hard requirement인지 확인할 수 없다.
- Public transport calibration v3는 P1 pair `n=1`이다. 동일 문제 official pairs가 3개 쌓이기 전 penalty 완화는 금지되어 있다.
- v4는 v28을 구제하지 않는다. 다음 candidate가 **새로 사전등록·봉인**될 때만 적용한다.
- warning이 있는 PASS는 “slice가 안전하다”는 뜻이 아니다. 공식 pooled 목적을 통과했으나 transport risk가 남았다는 정확한 상태다.

## 실행·접근 경계

- model fits: `0`
- historical train rows read: `421,032` (aggregate만 보존)
- official row reads: `0`
- hidden truth reads: `0`
- submission CSV: `0`
- uploads: `0`
- v28 retroactive reclassification: `0`

수치 집계와 검증 결과는 [train-only-distribution.json](train-only-distribution.json)과 [independent-qa.json](independent-qa.json)에 있다.
