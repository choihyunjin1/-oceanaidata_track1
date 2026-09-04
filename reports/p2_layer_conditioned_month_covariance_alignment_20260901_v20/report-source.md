# P2 layer-conditioned month covariance alignment v20

## 결론

상태 `EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION`; pooled ΔRMSE `-0.051840385℃`, canonical nominal `+0.650469`점, fixed-penalty `+0.528787`점. 반복 노출 historical surface의 탐색 증거이며 official 성능 주장이 아니다.

v13 architecture/input/prefix/purge/seeds/epochs/blend/action cap은 고정했고, 유일한 과학 변화는 같은 target layer 안의 calendar-month latent covariance에 고정 1.0 Deep-CORAL penalty를 적용한 것이다. cross-layer alignment, sweep, router, ensemble은 없다.

v13 대비 pooled ΔRMSE 차이 `-0.000985810℃`, v18 대비 `-0.001100516℃`, v19 대비 `-0.004072326℃`다. 비교는 후보 원장용이다.

official/hidden/CSV/upload=0.
