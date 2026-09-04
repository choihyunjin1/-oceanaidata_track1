# P1 v18 repository-wide duplicate audit

## 결론

`P1_1_CAUSAL_SOFT_SYMBOLIC_TRANSITION_LINEAR_ADDONLY`의 정확한 실행은 저장소에서
발견되지 않았다. 따라서 v18은 **비중복 0-fit 사전봉인 후보**다. 아직 historical
truth, 공식 입력, hidden truth를 읽지 않았고 attempt lock·CSV·upload는 모두 0이다.

## 정확한 봉인 구조

- 각 station-layer 시점에서 temperature의 과거 24시간만 사용한다.
- 10분 backward-as-of grid, 최대 age 20분, 각 grid point의 과거 72시간
  median/MAD로 정규화한다.
- 12개 2시간 PAA block을 고정 5-center Gaussian soft membership으로 바꾼다.
- 12×5 unary membership, 인접 block 11×5×5 transition product, 12 coverage를
  합쳐 정확히 347개 연속 feature를 만든다.
- incumbent-negative row에만 L2 LogisticRegression `C=0.1`을 적합하고
  `p>=0.99`만 add-only union한다. Q2→Q3, Q2+Q3→Q4의 정확히 2 fits다.
- hard SAX bin, feature selection, window/center/bandwidth/threshold grid는 없다.

## 저장소 검색과 근접 계열 판정

검색 범위는 `configs/experiments`, `scripts`, `src/p1_qc`,
`reports/historical_model_reaudit_20260831_v1`,
`reports/negative_evidence_registry_20260830_v1`이다. 검색어는 `SAX`,
`symbolic aggregate`, `SFA`, `BOSS`, `WEASEL`, `bag-of-pattern`,
`dictionary transform`, `soft symbol`, `ordinal pattern`이었다.

| 근접 계열 | 저장소 증거 | 비중복 판단 |
|---|---|---|
| v17 MiniRocket-lite | fixed random convolution의 PPV 512개 | v18은 convolution·bias quantile 없이 저주파 PAA soft transition을 사용 |
| multiscale offset/drift | 고정 Haar/slope/curvature 29-feature bank | v18은 24h symbolic transition 분포이며 Haar contrast를 쓰지 않음 |
| TS2Vec/SupCon/TCN | 학습된 temporal encoder | v18 representation parameter는 0이고 linear head만 학습 |
| CAPA/CUSUM/RPCA/GP | change interval 또는 latent residual proposal | v18은 interval proposal·change penalty·latent state를 쓰지 않음 |
| v15 spline / v16 GCE | 현재 row의 causal tabular projection | v18은 24h shape-transition representation을 새로 구성 |

## 1차 출처와 적용 한계

- [Lin et al., SAX original paper](https://www.cs.ucr.edu/~eamonn/SAX.pdf)는
  시계열을 저차원 symbolic representation으로 바꾸는 근거다.
- [Schäfer and Leser, WEASEL](https://arxiv.org/abs/1701.07681)은 sliding-window
  symbolic features와 linear classifier가 긴 시계열에서 계산 가능한 설계임을
  보인다.
- [Cawley and Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)은
  반복된 model selection의 과적합 위험 때문에 v18의 모든 자유도를 결과 전에
  고정해야 한다는 근거다.

문헌은 P1 성능을 보장하지 않는다. 특히 원래 SAX/WEASEL은 hard symbols와
분류용 feature selection을 포함할 수 있지만, v18은 Public 수송 안정성을 위해
그 둘을 쓰지 않는 **연속 soft-symbol 변형**이다. 이 변형의 유효성은 사전등록된
historical outer block에서만 반증 가능하다.

## 후속 실행 gate

historical 실행은 별도 승인 전 0이다. 승인되더라도 고정 2 fits만 허용하며,
Q3/Q4 각각 비악화, pooled 양수, day-block CI90 lower > 0, 개선확률 ≥ 0.8,
addition precision > incumbent F1/2, removals 0, 일별 changed ≤ 0.5%,
station-layer-quarter concentration ≤ 50%, raw points ≥ 0.131682092,
수송 보정 후 ≥ 0.01을 모두 통과해야 한다. 하나라도 실패하면 exact v18을 닫고
center·bandwidth·segment·threshold를 조정하지 않는다.
