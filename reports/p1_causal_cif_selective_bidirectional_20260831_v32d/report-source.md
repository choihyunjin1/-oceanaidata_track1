# P1 causal CIF selective bidirectional v32d

## 결론

`NO_GO`다. 봉인된 v32b CIF 확률을 threshold `0.5`로 이진화한 뒤 각 forward fold의 incumbent disagreement를 확률 margin 순으로 최대 `0.5%` 교체했지만, Q3와 Q4 모두 크게 악화됐다. 이 후보는 제출하지 않으며 threshold, top-k, decoder를 바꾼 재시도도 하지 않는다.

## Exactly-once 결과

- model fits: `0`
- runtime: `11.0055042s`
- action seal truth reads: `0`
- Q3 changes: `883 / 176,738 = 0.004996096`
- Q4 changes: `555 / 111,124 = 0.004994421`
- pooled reference F1: `0.902917024`
- pooled candidate F1: `0.834116658`
- pooled delta F1: `-0.068800366`
- Q3 delta F1: `-0.062744494`
- Q4 delta F1: `-0.077114276`
- raw expected points: `-1.828584441`
- calibrated expected points: `-1.833968133`
- dependent KST-day bootstrap CI90: `[-0.094108618, -0.047687588]`
- bootstrap probability improved: `0.0`

## Error geometry

- additions: `901`
- true-positive additions: `0`
- false-positive additions: `901`
- addition precision: `0.0`
- removals: `537`
- incumbent true-positive removals: `531`
- incumbent false-positive removals: `6`
- maximum station-layer-quarter change concentration: `0.265646732`
- maximum KST-day changed fraction: `0.060012634`

상위 margin이 anomaly correctness와 반대 방향으로 정렬됐다. Q3 선택은 883건 모두 false-positive addition이었고, Q4 제거는 537건 중 531건이 incumbent true positive였다. 따라서 add-only abstention이 단순 decoder 제약 때문이라는 가설은 반증됐다.

## Gate 및 경계

변경률과 집중도 gate만 통과했다. pooled/Q3/Q4, CI, 개선확률, expected-points, TP-removal, addition-precision gate가 실패했으므로 최종 상태는 `NO_GO`다.

- official / hidden / test / sample reads: `0 / 0 / 0 / 0`
- submission CSV / upload / retry: `0 / 0 / 0`
- action seal: `52fa1e917a5d0d2d1e1010f1618a272df3104d7d5d6cd307588241adbb00bedb`
- sealed action mask: `d2c6f49241bf5a6caeddf0a3bbff9962dc54c5c2a2ba3c590108ce0d5e52831d`
- result: `1d893372d31c2c2dcaf1a880f4a23609d3523b779f9631e97fcd64796e407911`

Post-run verification: focused pytest `4 PASS`, Ruff `PASS`, pycompile `PASS`.
