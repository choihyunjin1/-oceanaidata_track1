# P2 continuous-depth permutation-invariant set encoder 20260901 v12

## 결론

상태: `EXPLORATORY_PASS_REQUIRES_FRESH_CONFIRMATION`. exposed historical surface의 탐색 결과이며 fresh confirmation이 아니다.

| pooled ΔRMSE | Sep-Oct | Jul-Aug | Nov-Dec | nominal points | transport points | gate |
|---:|---:|---:|---:|---:|---:|---|
| -0.042993523 | -0.079386794 | -0.007186197 | +0.179735810 | +0.539463 | +0.603386 | PASS |

## 새 구조와 중복 배제

공개층 `(depth, value, mask)` token마다 같은 element MLP를 적용하고 masked mean/max로 가환 집계했다. 과거 continuous-depth TCN은 시간축 convolution과 order-sensitive fixed public-layer vector를 사용하므로 의미 중복이 아니다. Heavy-model scout의 vertical set encoder는 제안만 있었고 실행 receipt가 없다.

Deep Sets (Zaheer et al., NeurIPS 2017)의 가환 pooling 원칙을 구현했고 공개층 순열 단위 테스트를 통과했다. Set Transformer의 attention은 작은 표본에서 새 sweep이 되므로 사용하지 않았다.

행 삭제, early stopping, outer-fold 튜닝, 결과 적응 재시도는 모두 0이다. champion 80%를 보존하고 단일 frozen model 평균을 20%만 혼합했다.

## 경계

세 historical block은 이미 노출됐으므로 이 결과는 후보 폐쇄/우선순위용 탐색 증거다. 공식 test/sample/baseline/score/query/hidden rows=0, CSV=0, upload=0.
