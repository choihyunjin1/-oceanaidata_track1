# P2 v48 diagonal-bilinear profile-pooling DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_DIAGONAL_BILINEAR_PROFILE_POOLING`. pooled delta RMSE `-0.051106051 C`, canonical nominal `+0.641255` points, transport `+0.519573` points.

prospective fold x layer gate `False`, non-harm `6/9`, max cell `+0.016979827 C`.

Exact v13의 masked mean/max summary에 token embedding의 masked diagonal second moment 32개를 추가했다. 새 head columns는 zero-init되어 initial function이 v13과 같다. v45/v45c/v46/v47 비교는 terminal 후 ledger 진단만 수행했다. selection, retune, router, ensemble, official/test/sample/hidden/query/CSV/upload는 0이다.

## 최종 판단

9 fits를 정확히 한 번 수행했다. Pooled delta는 `-0.051106051 C`, canonical nominal `+0.641255`, fixed transport-adjusted `+0.519573` points로 지금까지의 내부 aggregate 중 강했다. 세 fold, 여섯 month, 세 layer의 aggregate는 모두 non-harm이었고 CI90도 `[-0.082123417, -0.037485063]`이었다.

그러나 prospective fold×layer gate는 `6/9` non-harm에 그쳤다. Jul-Aug layer 2/3이 각각 `+0.016979827 C`, `+0.012511371 C`, Nov-Dec layer 4가 `+0.011601568 C` 악화되어 허용치 `+0.003 C`를 넘었다. 따라서 v48은 `NO_GO`로 동결하고 bilinear rank, normalization, blend, seed, router를 재조정하지 않는다. v45/v45c DropConnect family를 P2 안전 후보로 유지한다.

독립 재계산은 `40/40 PASS`이고 공식/test/sample/submission/hidden/query/CSV/upload 접근은 모두 0이다.
