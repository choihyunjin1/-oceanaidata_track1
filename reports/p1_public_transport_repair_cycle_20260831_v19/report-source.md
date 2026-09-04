# P1 v19 G-ORS causal one-step run extension

## 결론

`NO_GO_INTERNAL`. 사전 고정한 G-ORS-only one-step rule은 Q3/Q4에서 13행을 추가했고 4 TP/9 FP를 만들었다. pooled ΔF1은 `-0.000185318`, raw 예상 점수는 `-0.004925394`, family penalty 반영값은 `-0.010309086`으로 모두 승격 기준에 미달했다. 모델 fit은 0, 결정론적 historical scoring은 정확히 1회였다.

## 핵심 결과

| 항목 | 결과 |
|---|---:|
| reference F1 | `0.906803720` |
| candidate F1 | `0.906618402` |
| pooled ΔF1 | `-0.000185318` |
| Q3 ΔF1 | `-0.000317202` |
| Q4 ΔF1 | `0.000000000` |
| 추가행 TP / FP | `4 / 9` |
| marginal precision / LCB90 | `0.307692 / 0.141611` |
| bootstrap P(improve) | `0.142` |
| bootstrap CI90 | `[-0.000510699, 0.000100558]` |
| G-ORS layer 1 ΔF1 | `-0.001684428` |
| raw / calibrated expected points | `-0.004925394 / -0.010309086` |

changed share `0.00004516`과 G-ORS day 최대 추가 2행은 안전 범위를 통과했고 I-ORS/S-ORS는 bit-exact였다. 그러나 효능, 수송, precision, bootstrap, G support gate가 모두 실패했다.

## 해석

공식 G-ORS 15행 factor가 양수였다는 aggregate 결과는 이 규칙의 rowwise 유효성을 보장하지 않았다. v12의 전역 trailing rule에서 관찰된 G slice도 raw E150 comparator로 바꾼 v19에서 재현되지 않았다. 따라서 station을 G로 제한하는 것만으로 boundary-extension 계열의 낮은 marginal precision이 해결되지 않는다.

이 결과는 fresh confirmation이 아니다. v12 G slice와 Public G factorial을 보고 support를 고정한 adaptive development replay다. cadence, span, station 또는 bootstrap gate를 결과 뒤 바꾸지 않으며 이 exact 조합을 종료한다.

## 보존 경계

- official test key/value read: `0`
- current champion row read: `0`
- hidden truth read: `0`
- submission CSV: `0`
- upload: `0`
- automatic retry: `0`
