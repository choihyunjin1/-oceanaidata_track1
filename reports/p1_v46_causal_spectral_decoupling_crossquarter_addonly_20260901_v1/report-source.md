# P1 v46 causal spectral decoupling

## Terminal decision

`NO_GO_CROSS_QUARTER_TRANSPORT_VETO_Q4_UNOPENED` at `PRE_Q2_CALIBRATION_GATE`. The fixed coefficient and family are frozen without retuning.

## Science and target-free readiness

Repository-wide search found no executed spectral-decoupling, gradient-starvation output penalty, squared-logit penalty, or logit-norm regularizer in P1/P2/P3. Parameter weight decay and P2 spectral normalization act on weights/operator norms; focal/GCE reweight target losses; R-Drop matches stochastic predictions; VIB penalizes latent KL. v46 instead replaced weight decay with one fixed target-independent output penalty, `weighted BCE + 0.5 x 0.01 x mean(logit^2)`, while leaving the eight past-only features, seeds, epochs, inference scores, thresholds, v28 gate, v33 persistence, and anchor-union decoder unchanged. Pezeshki et al., *Gradient Starvation: A Learning Proclivity in Neural Networks* (NeurIPS 2021, https://proceedings.neurips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html), motivates the mechanism but makes no P1 performance claim.

The label-free prefix conditioning check used 294,278 rows and no targets. All eight standardized features were finite. Covariance eigenvalues ranged from 0.000734509 to 3.864571, condition number was 5261.434, and fixed-seed initial logit variances were 0.014756, 0.045027, and 0.019083. All values passed the sealed conditioning gates. Two zero-operation preflights were byte-identical at 4,867 bytes with SHA-256 `df00047188d4a21c5407a9934c08d6d432c4416493a51c629b96fbfefc520f07`; no lock, artifact, target, or official surface had been opened. Focused pytest passed 6/6 and Ruff passed.

## Exactly-once result

Three fits completed in 5.656 seconds, 144 optimizer steps each. Mean spectral penalties were 0.000840748, 0.001523133, and 0.000750595, proving the fixed term was active.

- q=.999: 56/57 TP, precision 0.982456, Wilson90 LCB 0.925111, but only S-ORS/L5 chronological half 2 was supported.
- q=.9975: 132/141 TP, precision 0.936170, LCB 0.893427, but both supported halves came only from S-ORS/L5.
- q=.995: 238/281 TP, precision 0.846975, LCB 0.808354 and three stations appeared, but supported G-ORS/L1 had 0/31 TP, triggering the unchanged zero-TP veto.

No threshold was chosen. Q2/Q3 transport targets and Q4 target remained unopened. Raw F1 delta, nominal points, and transport-adjusted points are zero under the no-action terminal; Q4 actions and anchor removals are zero. Before any Q2 target read, v33 preserved the three label-blind Q2 action vectors with counts 332, 253, and 64; bundle SHA-256 is `dcf0062328ba8e50e6683a545f6fe649fc91fb23aa9d183083c3326ba9e9c56c`.

Post-terminal lifecycle QA passed 20/20; focused pytest again passed 6/6 and Ruff passed. Official, test, sample submission, hidden, CSV, submission, and upload access remained zero.

## Next distinct mechanism

The next candidate to fingerprint is causal hidden-feature MixStyle across source station-quarter environments: mix only hidden feature mean/scale statistics during training, never labels, targets, thresholds, or inference context. It must be distinguished from P2 MixUp, CORAL, RevIN/context normalization, station adversarial training, and existing augmentation/consistency families, and must pass a target-free multi-environment support contract before sealing. This report does not authorize it.
