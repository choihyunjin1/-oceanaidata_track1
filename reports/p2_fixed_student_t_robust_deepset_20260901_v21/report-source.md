# P2 fixed Student-t robust DeepSets v21

## 결론

상태 `EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION`; pooled ΔRMSE `-0.050482138℃`, canonical nominal `+0.633426`점, fixed-penalty `+0.511744`점. 반복 노출 historical surface의 탐색 증거이며 official 성능 주장이 아니다.

v13 architecture/input/domain-balanced weights/prefix/purge/seeds/epochs/blend/action cap은 고정했다. 유일한 변화는 SmoothL1을 ν=4, normalized scale=1의 고정 Student-t location NLL로 교체한 것이다. row deletion, df/scale learning, sweep, router, ensemble은 없다.

v13 대비 ΔRMSE `+0.000372437℃`, v18 대비 `+0.000257732℃`, v20 대비 `+0.001358247℃`다. 비교는 원장용이다.

official/hidden/CSV/upload=0.
