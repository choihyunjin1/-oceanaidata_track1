# P1 nested S-layer ablation v33c

## 결론

**TERMINAL NO_GO / 재시도 금지**. Q2는 prior가 없어 abstain했고, Q3는 Q2만, Q4는 Q2+Q3만 사용해 제거 layer를 선택했다. 공식·hidden·test·sample 접근과 CSV·upload는 모두 0이다.

| fold | delta F1 vs raw E150 | removed | removed TP | removed FP |
|---|---:|---:|---:|---:|
| 2025_q2 | 0.000000000 | 0 | 0 | 0 |
| 2025_q3 | 0.000000000 | 0 | 0 | 0 |
| 2025_q4 | 0.000000000 | 0 | 0 | 0 |

- Q3+Q4 pooled delta F1: `0.000000000`
- Q3+Q4 day-block CI90: `[0.000000000, 0.000000000]`
- 개선 확률: `0.000000`
- 예상 점수 delta: `0.000000000`
- Q3 선택 layer: `[]`
- Q4 선택 layer: `[]`
- full deployment 선택 layer: `[]`
- independent QA: `PASS`
- fit count: `0`; retry: `0`

Full deployment layer set은 Q2-Q4 historical prefix에 같은 support>=10 및 marginal precision < incumbent F1/2 규칙을 정확히 한 번 적용해 계산했다. 실제 공식 materialization은 수행하지 않았다.
