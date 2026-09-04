# P1 v44 causal Fishr gradient-variance transport screen

## Decision

`CLOSE_ZERO_FIT_INSUFFICIENT_STATION_LAYER_PERIOD_SUPPORT`.

The repo-wide read-only fingerprint found no Fishr or environment gradient-variance matching execution in P1/P2/P3. The proposed intervention was distinct from P1 v24 station-adversarial removal, closed Group-DRO worst-risk weighting, P2 CORAL latent covariance alignment, IRMv1 scalar optimality, V-REx environment mean-risk variance, station/layer partial pooling, and density-ratio weighting. The primary source only motivates matching domain-level variances of loss gradients; it does not establish this P1 recipe or expected performance: Rame, Dancette, and Cord, *Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization*, ICML 2022, https://proceedings.mlr.press/v162/rame22a.html.

## Frozen science and support veto

Before any target read or fit, v44 sealed the existing eight-feature causal temperature-state representation, a small 8-unit head, weighted BCE plus a fixed `1.0` last-layer per-sample gradient-variance penalty, station x layer x calendar-quarter source environments, three seeds/three planned fits (maximum nine), the unchanged quantiles `[.995,.9975,.999]`, v28 cross-quarter gate, and v33 rejected-candidate action persistence. Environment one-hot bits were reserved for the training penalty and excluded from inference.

The train-only, label-free source check used only `station`, `layer`, and `time`. At the fixed pre-Q2 fit boundary there were 294,278 rows, three stations, 11 station-layer identities, and 29 station-layer-quarter environments. The smallest environment had only 54 rows, below the preregistered 100-row information-value floor (median 12,878; maximum 15,361). The floor was not relaxed after seeing this result. Therefore no labels, optimizer, Q2/Q3/Q4 surface, action bundle, or lock were opened.

An early zero-operation call exposed two technical preflight issues before this terminal: the semantic-decision token did not match the shared engine's accepted constant, and pandas native datetime resolution could make a naive integer cutoff ambiguous. The token was normalized without changing science, and all cutoff paths/tests were forced through the repository nanosecond conversion with a distinct-range assertion. Both occurred with zero fit, zero target access, and no namespace consumption.

## Verification and next axis

Focused pytest passed `6/6`; Ruff passed. Config/runner/test SHA-256 are `e4dd6e1e...7b8a512`, `6190acff...94ebd68`, and `b7d3eb92...7eb1145`. Official/test/sample/submission/hidden reads, CSV creation, uploads, anchor removals, fits, and optimizer steps are all zero.

The first follow-on idea, an environment-support intersection with deterministic abstention, was rejected immediately as a semantic duplicate: clean-state CAPA already zero-signals unsupported station-layer groups, and a prior transport repair already uses station-concentration abstention. The next genuinely different candidate to audit is a causal cross-environment gradient-sign agreement mask motivated by Parascandolo et al., *Learning Explanations that are Hard to Vary* (ICLR 2021, https://openreview.net/forum?id=hb1sDDSLbV). Unlike Fishr variance matching, Group-DRO loss reweighting, or a domain adversary, it would update only parameter coordinates whose station-by-calendar-quarter gradient signs agree prospectively; it must still pass a repository-wide AND-mask/sign-agreement fingerprint and target-free station-quarter plus layer-diversity support before any seal or fit. This report neither seals nor authorizes it.
