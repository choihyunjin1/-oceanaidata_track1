# P1 v18 state-dependent drift/diffusion — zero-fit terminal report

Decision: `CLOSE_ZERO_FIT_INSUFFICIENT_CONDITIONAL_MOMENT_SUPPORT`. No model fit, target evaluation, lock, artifact, outer score, CSV, or upload occurred. The fixed support gate failed identically on both real preflight attempts, so bin count, minimum support, variance floor, or gate must not be retuned under this ID.

## Primary-source and novelty boundary

Siegert, Friedrich, and Peinke show that drift and diffusion terms of a stochastic-system description can be estimated from empirical conditional moments ([Physics Letters A 1998](https://arxiv.org/abs/cond-mat/9803250), DOI `10.1016/S0375-9601(98)00283-7`). This motivated only a fixed finite-lag, state-quantile-bin approximation. It does not establish that P1 temperature is a Markov diffusion or imply anomaly-detection performance.

The bounded axis differed from v15's lifted conditional-mean evolution operator and the latent-state GP by fitting no latent trajectory, kernel, cross-layer dynamics, or observable lift. It also differed from ordinary rolling variance because its conditional drift and diffusion scale were fitted only on the prefix for each preceding-state bin and then frozen. Exact negative-evidence hashes are in the config.

## Failed support receipt

- source rows inspected: `776706`, README/train only
- frozen minimum nonzero-feature share: `0.5`
- observed nonzero-feature share: `0.4384426024776428` — FAIL
- eight observed feature variances: `[0.0007873827, 0.0058447341, 0.3893090785, 0.3482993543, 17.4111480713, 10.9511470795, 9.2154493332, 7.3619647026]`; variance floor `1e-6` passed
- repeated real preflight outcome: identical `CLOSE_ZERO_FIT_INSUFFICIENT_CONDITIONAL_MOMENT_SUPPORT`
- namespace after failure: artifact absent, lock absent

The representation lacked the preregistered coverage because many cadence-gap segments did not contain 192 prefix rows before their own reset. Relaxing that gate or borrowing later rows would alter the sealed architecture and is forbidden.

Focused tests were 4/4 PASS and Ruff PASS before preflight. Official/test/sample/submission/hidden reads, target reads, fits, CSV creation, upload, and submission were all zero.

## Next independent audit

A possible next axis is causal local extrema persistence in amplitude space using prefix-fitted prominence/return-time statistics, but it must be rejected if the repository fingerprint shows equivalence to peak rules, horizontal visibility, recurrence, or event proposals. No automatic execution is authorized by this report.
