# Public 수송 보정 게이트 v1

## 결론

새 후보는 내부 성능을 점수로 바꾼 원시 기대값에서 직전 공식 제출의 최악 수송 잔차를 차감한 뒤에도 `0.01점 이상`이어야 한다. 이 기준에서 필요한 원시 기대 개선 하한은 P1 `0.015384점`, P2 `0.131682점`, P3 `0.331906점`이다.

이 하한은 공식식이나 통계적 신뢰구간이 아니다. 직전 내부→Public 역전을 그대로 반복하지 않기 위한 경험적 방화벽이다. 현재 대응 관측은 P1 1개, P2 3개, P3 2개뿐이므로 후보의 메커니즘이 달라져도 이 수치를 절대적 진실로 보지 않고, 시간 블록·그룹/episode·nested selection 검증과 함께 사용한다.

## 계산

`transport residual = actual Public point delta - pre-submission central expected point delta`로 정의했다. 문제별 최악 잔차는 P1 `-0.005384`, P2 `-0.121682`, P3 `-0.321906점`이다. 따라서 `raw expected delta + worst residual >= 0.01`을 통과 규칙으로 고정한다.

## 연구 근거

유한 검증 표본에서 모델 선택 기준 자체를 반복 최적화하면 선택 편향이 실제 알고리즘 차이와 비슷한 크기까지 커질 수 있다. 따라서 후보 생성과 평가를 분리하고 nested 또는 frozen 평가를 사용한다 ([Cawley & Talbot, JMLR 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html)).

시간·계층 의존 데이터에는 무작위 분할 대신 block 검증이 필요하다. 시계열 예측에서는 blocked CV가 시간 의존을 보존하면서 모델 선택을 안정화할 수 있고 ([Bergmeir & Benítez, Information Sciences 2012](https://doi.org/10.1016/J.INS.2011.12.028)), 구조화된 생태 자료에서도 의존 구조가 있으면 block CV가 권고된다 ([Roberts et al., Ecography 2017](https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf)).

일반 CV의 분산은 fold 간 상관 때문에 과소추정될 수 있으며 nested CV가 불확실성 추정에 더 적절하다 ([Bates, Hastie & Tibshirani, JASA 2024](https://arxiv.org/abs/2104.00673)). Test covariate 분포가 다를 때는 가중 conformal 접근도 가능하지만 밀도비를 정확히 알거나 추정해야 한다는 조건이 있다 ([Tibshirani et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)). 따라서 현재 데이터에서는 conformal 숫자만으로 PASS를 선언하지 않는다.

## 적용 규칙

- P1: station×layer×quarter와 event 길이 블록, frozen threshold, expected gain `>=0.01점`.
- P2: 공식과 같은 Sep–Oct 계절 전이를 외부 fold로 두고 layer별 worst-case 및 nested selection 적용.
- P3: station×lead×episode 블록과 78시간 독립 사례 구조를 보존하고 high-wave selection shift를 별도 평가.
- Public 점수는 수송 오차 교정에만 쓰며 개별 공식 행의 정답 라벨처럼 역학습하지 않는다.

## 한계

공식 피드백은 후보 수준 집계치뿐이며 Public 행별 정답은 없다. 계산에는 이번 4건 외에 같은 날 앞서 제출한 P2 bin17 layer4와 P3 KMA lead-factor 영수증도 포함했다. 그래도 residual penalty는 후보 메커니즘과 상관된 오차를 분리하지 못한다. 후속 후보가 실제 Public에서 성공하면 새 관측으로 이 보정을 갱신해야 한다.
