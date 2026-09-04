# P2 v24 sharpness-aware domain-balanced DeepSets

## 결론

상태: `EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION`. pooled ΔRMSE `-0.049469328 C`, canonical nominal `+0.620718`점, transport `+0.499036`점.

fold ΔRMSE: Sep-Oct `-0.071802205`, Jul-Aug `-0.043474490`, Nov-Dec `-0.018310319`.

v13 architecture/loss/domain weights/prefix+7d purge/seeds/epochs/blend/action cap을 고정하고, rho=0.05 vanilla SAM parameter-neighborhood optimizer geometry만 추가했다. Foret et al. (ICLR 2021)는 optimizer 동기만 제공하며 P2 성능 근거가 아니다. ASAM/rho sweep/scheduler/router/ensemble/row deletion/official/hidden/CSV/upload=0.
