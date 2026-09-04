# P2 prefix-safe domain-balanced DeepSets 20260901 v13

## 결론

상태: `EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION`. pooled ΔRMSE `-0.050854575℃`, canonical nominal `+0.638099`점, transport `+0.516417`점.

fold ΔRMSE: Sep-Oct `-0.080432908`, Jul-Aug `-0.019007142`, Nov-Dec `-0.008989588`.

각 fold는 validation 시작 7일 전까지만 학습했고, layer×calendar-month×KST-day에 동일 총 질량을 주었다. architecture/blend/seeds는 v12와 같고 결과 적응 routing, threshold search, row deletion은 없다. official/hidden/CSV/upload=0.
