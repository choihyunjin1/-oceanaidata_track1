# P2 v26 layer-month group-preserving MixUp DeepSets

## 결론

상태: `EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION`. pooled ΔRMSE `-0.048727632 C`, canonical nominal `+0.611411`점, transport `+0.489729`점.

fold ΔRMSE: Sep-Oct `-0.076323712`, Jul-Aug `-0.020800352`, Nov-Dec `-0.011568032`.

v13 model/loss/domain weights/prefix+7d purge/seeds/epochs/blend/action cap을 고정하고 같은 target-layer×calendar-month 안의 training rows에 alpha=0.2 MixUp만 적용했다. Zhang et al. (ICLR 2018)은 vicinal training 동기만 제공하며 P2 성능 근거가 아니다. cross-group mixing/alpha sweep/router/ensemble/row deletion/official/hidden/CSV/upload=0.
