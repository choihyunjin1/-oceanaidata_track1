# P2 v23 public-temperature input-gradient DeepSets

## 결론

상태: `EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION`. pooled ΔRMSE `-0.051892467 C`, canonical nominal `+0.651122`점, transport `+0.529440`점.

fold ΔRMSE: Sep-Oct `-0.080424162`, Jul-Aug `-0.026067387`, Nov-Dec `-0.009395235`.

v13 architecture/domain weights/prefix+7d purge/seeds/epochs/blend/action cap을 고정하고, observed public-temperature token에 대한 loss-input-gradient L2 penalty(lambda=0.01)만 추가했다. Ross and Doshi-Velez (AAAI 2018, DOI 10.1609/aaai.v32i1.11504)는 representation 동기만 제공하며 P2 성능 근거가 아니다. row deletion/parameter sweep/router/official/hidden/CSV/upload=0.
