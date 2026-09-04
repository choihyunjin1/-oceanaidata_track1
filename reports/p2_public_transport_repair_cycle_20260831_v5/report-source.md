# P2 Public transport repair cycle 20260831 v5

## 결론

`COMPLETE_NO_PASS`다. 사전 고정한 두 Huber correction router를 12 fits로 실제 학습·chronological blocked 내부 채점했으나, 보정 후 기대 점수 개선이 각각 -0.062379점과 -0.069432점으로 `>= +0.01점` gate를 통과하지 못했다. 제출 CSV와 업로드는 모두 0건이다.

| 후보 | Oct nested ΔRMSE | Jul-Aug ΔRMSE | Nov-Dec ΔRMSE | raw 기대점수 | 보정 기대점수 | 결론 |
|---|---:|---:|---:|---:|---:|---|
| P2_V5_GLOBAL_HUBER_ROUTER | -0.006766982 | +0.026342902 | +0.047665922 | +0.059303 | -0.062379 | FAIL |
| P2_V5_LAYER_HUBER_ROUTER | -0.006432885 | +0.026747987 | +0.001345667 | +0.052250 | -0.069432 | FAIL |

두 후보 모두 공식 계절을 모사한 2024 Oct nested holdout과 layer 2/3/4에서 개선했지만, 계절 전이에서 무너졌다. 특히 Jul-Aug 악화가 약 +0.026°C라 “공식 유사 fold 하나의 개선”만으로는 승격할 수 없다는 직전 공식 실패의 교훈이 다시 확인됐다.

## 설계와 검증

- v4 HGB의 sealed historical OOF correction만 meta-feature로 사용했다. 공식 점수는 학습 label로 사용하지 않았다.
- 전역 Huber와 층별 Huber를 결과 확인 전에 config에 고정했다. Huber M-estimation을 사용했으며 이상 행 삭제는 하지 않았다.
- 2024 Sep를 train, 2024 Oct를 selection-matched nested test로 두고, 이후 2025 Jul-Aug 및 Nov-Dec를 strictly chronological outer tests로 사용했다.
- Oct 일자 block bootstrap 90% 상단을 raw 기대 개선으로 변환한 뒤, 공식 결과에서 관측된 최악 수송 잔차 0.121682092점을 차감했다.
- inclusive gate는 calibrated expected point delta `>= 0.01`, 최소 2/3 outer 개선, Oct 개선, worst outer ΔRMSE `<=0.002`, Oct의 모든 층 개선의 논리곱이다.

## 근거와 한계

Cawley & Talbot은 동일 자료에서 선택과 평가를 반복할 때 발생하는 selection bias 때문에 nested evaluation이 필요함을 보였다. Sugiyama 등은 covariate shift에서 표준 CV가 편향될 수 있음을 보였으므로 공식 계절/선택 패턴과 시간 순서를 맞춘 검증을 우선했다. Huber의 M-estimation은 오염에 강한 손실의 근거이며, 데이터 행의 사후 삭제를 정당화하지는 않는다.

이번 matching은 공식 query의 hidden target을 보지 않고 시간·layer 구조만 모사한다. empirical transport penalty는 confidence bound가 아니라 현재 3개 공식 관측 중 최악 잔차를 재사용한 보수적 guardrail이다.

## 출처

- Cawley & Talbot (2010), JMLR, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*: https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf
- Sugiyama, Krauledat & Müller (2007), JMLR, *Covariate Shift Adaptation by Importance Weighted Cross Validation*: https://jmlr.csail.mit.edu/papers/volume8/sugiyama07a/sugiyama07a.pdf
- Huber (1964), *Robust Estimation of a Location Parameter*: https://doi.org/10.1214/aoms/1177703732
