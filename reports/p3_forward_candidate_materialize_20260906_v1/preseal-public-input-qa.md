# Pre-seal public-input integrity amendment

2026-09-06, before this materializer's first full fit or official read. The historical runners/configs are unchanged.

Independent P1 static review identified that training-source hashes did not identify the public context/index used by inference. The unsealed materializer now hashes only `test_context.parquet` and `test_index.csv` at the approved inference phase, checks those hashes again after inference, records them in the answer receipt, and requires matching inputs for fresh-process answer replay. Parsed data columns remain allowlisted; sample and hidden inputs are never allowed. No official files were read to implement or test this amendment.

Verification: materializer synthetic tests 4 PASS; Ruff PASS. These tests use stub model predictions, with 0 real model fits. Earlier `code-qa.json` remains a historical snapshot rather than the current materializer hash. The actual full-fit seal will record the final runner SHA-256.

The GPU full fit's one-hour limit is checked before/after the fit, not enforced as an operating-system hard timeout. Parent process monitoring remains necessary. This amendment does not change model parameters, features, coefficients, targets, sample weights, or selection rules.

Further pre-seal checks: full-model QA now inspects the saved router's hmax/target-free columns as well as native base-model feature names. CSV serialization explicitly uses LF (`lineterminator="\n"`). A new fully synthetic 200-case/1,200-row adapter test exercises key/order, public-input hash receipts, LF bytes, and the replay branch using simulated distinct PIDs. Final focused result: 5 PASS and Ruff PASS, 0 real model fits/official reads. Simulated PID testing does not substitute for the required actual fresh-process replay after full training.
