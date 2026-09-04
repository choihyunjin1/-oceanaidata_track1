# P2 compact same-time Set Transformer v15

## 결론

상태 `EXPLORATORY_NO_GO_COMPACT_SET_TRANSFORMER`; pooled ΔRMSE `-0.051231934℃`, canonical nominal `+0.642834`점, fixed-penalty `+0.521152`점. 모든 값은 노출된 historical surface의 탐색 지표이며 official 성능 주장이 아니다.

v12/v13의 독립 element map 뒤 pooling과 달리, v15는 같은 timestamp의 public-depth tokens 사이에 1 block/2 heads self-attention을 적용한다. time axis와 positional encoding은 없으며 public-layer permutation equivariance/invariance를 단위검사했다. Set Transformer 논문은 구조 동기만 제공하고 P2 성능을 보증하지 않는다.

official/hidden/CSV/upload=0.
