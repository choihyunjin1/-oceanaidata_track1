# P2 v47 pooled-profile/context CrossNet DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_POOLED_CONTEXT_CROSS_NETWORK`. pooled delta RMSE `-0.050714628 C`, canonical nominal `+0.636343` points, transport `+0.514661` points.

prospective fold x layer gate `False`, non-harm `6/9`, max cell `+0.034562759 C`.

Exact v13의 masked mean/max public-profile summary와 11 context를 합친 75차원 벡터에 identity-initialized one-layer DCN feature cross만 추가했다. v45/v45c/v46 비교는 terminal 후 ledger 진단만 수행했다. selection, retune, router, ensemble, official/test/sample/hidden/query/CSV/upload는 0이다.

## 최종 판단

9 fits를 정확히 한 번 수행했으며 pooled 성능은 `-0.050714628 C`로 좋아졌다. 그러나 fold×layer prospective gate는 `6/9` non-harm에 그쳤고, 최악 cell은 2025 Nov-Dec layer 4의 `+0.034562759 C`였다. 2025-11 month도 `+0.003306041 C`로 legacy `+0.003 C` 허용치를 넘었다. 따라서 이 family는 `NO_GO`로 동결하며 depth, cross coefficient, seed, blend, router를 재조정하지 않는다.

v47의 pooled delta는 v45보다 `-0.001312525 C`, v45c보다 `-0.000136279 C` 더 낮지만 안전 gate를 통과하지 못했다. stochastic-confirmed v45/v45c DropConnect family를 P2 내부 안전 후보로 유지한다.

독립 재계산은 `40/40 PASS`, focused pytest `7/7 PASS`, Ruff 및 py_compile `PASS`이다. 공식/test/sample/submission/hidden/query/CSV/upload 접근은 모두 0이다.
