# P1 v36 technical-invalid closure

## Decision

`p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1` is closed as `INVALID_TECHNICAL_PROVENANCE_WRAPPER_HASH_OMITTED`.

The exactly-once namespace consumed three fits and stopped at the pre-Q2 calibration gate, so Q2/Q3/Q4 target reads, Q4 actions, anchor removals, official/test/sample/submission/hidden reads, CSV, and uploads were all zero. Those scientific receipts are not used for selection or family conclusions because the immutable provenance contract is incomplete.

## Exact fault

- Actual GCE wrapper SHA-256: `1f11885d944131761f0efe6a66449e31f39c0b6ba7a4b0eb90ecbef626b369ee`.
- Result `hashes.runner`: `a04bcee6504ee68e1cfc3fc55728ed9d0228e7a5f8589f9b04e8e5be3f30fe47`.
- The recorded value equals the imported v34 shared execution engine, not the wrapper that defines the GCE loss, network, objective guards, and namespace patch.

The inherited lifecycle QA also returned PASS because it revalidated the same shared-engine path. Independent QA therefore overrides that verdict to FAIL. Existing lock, result, preflight, and target-free v33 bundle remain immutable; none may be resumed, rescued, or reused for scientific adjudication.

## Preparation checks retained as technical receipts

- Focused pytest 6/6 and Ruff passed before and after execution.
- Two zero-operation preflights were byte-identical, 4,397 bytes, SHA-256 `eed54abb44beda5453393a48afe3cdd7f7554779f97ac50786c75e51b92d2446`.
- Result SHA-256: `a25ea2df245aae2d9f7c40d49e7d4f7c4db8262941aaec08d919169a7b70c7cc`.
- Config SHA-256: `b63ede32776cc2ddaee240ccbd96833eb34fadcd5b3f045be2f5d9fe80a98017`.
- Lock SHA-256: `393f36d8c2f69ff115e05353b65525b83261f5af8797fc6aa30b22305626271a`.

The minimum safe repair, if separately authorized, is a fresh v36r1 namespace that records and revalidates both wrapper and shared-engine hashes and prevents the imported `_configure` from replacing wrapper identity. v36 itself must never be retried.
