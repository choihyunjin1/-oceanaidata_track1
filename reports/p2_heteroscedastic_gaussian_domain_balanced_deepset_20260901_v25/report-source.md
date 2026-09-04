# P2 v25 heteroscedastic Gaussian domain-balanced DeepSets

## 결론

상태: `EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION`. pooled ΔRMSE `-0.045023193 C`, canonical nominal `+0.564930`점, transport `+0.443248`점.

fold ΔRMSE: Sep-Oct `-0.072438277`, Jul-Aug `-0.008631749`, Nov-Dec `-0.030935597`.

v13 shared element/pooling/hidden head/domain weights/prefix+7d purge/seeds/epochs/blend/action cap을 고정하고 final mean+conditional log-variance head와 bounded Gaussian NLL만 추가했다. inference는 mean만 사용한다. Kendall and Gal (NeurIPS 2017)은 learned attenuation 동기만 제공하며 P2 성능 근거가 아니다. variance router/abstention/sweep/row deletion/official/hidden/CSV/upload=0.
