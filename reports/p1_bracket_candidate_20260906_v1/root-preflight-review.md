# Root preflight review

PASS for the run3 snapshot; start authorized after the final synthetic suite passed. This is a pre-training implementation review, not model quality or official eligibility certification.

- Generated runner SHA256: `909ce7f112fd60b34d42da2206e2c27a270feac6b3367509d79ecd9f94ea985b`.
- Manifest SHA256: `ba3f9b9f1e3d0642df0aaa05eb8822f7356e28383ef916bfb45c3f97952629c4`.
- `preflight-tests.xml`: 17 tests, 0 failures/errors/skips, 24.776 seconds. Builder/templates, source extraction, test implementations and generated import ordering reviewed directly by root.
- Exact bracket source hash matches the previously validated research runner. Synthetic parity covers both arm matrices, encoders, weights, rules and policy selection.
- O80/B107 arm separation is used in training, model replay and inference. Both models assert 700 trees; CPU4/GPU0, four fits, frozen final-inner dates and train-only statistics remain unchanged.
- Partitions precede feature computation; whole terminal positive-run handling only examines labels before the training cutoff. Extra bracket context is bounded to 37 hours and exact-cadence segments. This does not make the inherited decoder's full-segment dependency finite.
- Missing model replay or independent training QA blocks official reads. Sample keys alone are allowed at inference; no historical answer SHA/positive-count target is prescribed.
- Root found and reproduced an isolated-entrypoint import-order failure in zero-fit run2 (`python -I .../run.py --help`). Fixed before training in run3; new ordinary/isolated entrypoint tests both pass. Prior staging folders remain preserved, and this repair consumed zero fits.
- Root training QA is separately implemented in `scripts/qa_p1_bracket_candidate_20260906_v1.py`: recompute train populations/keys/target digest, check actual model resources and unchanged O hashes, independently enumerate threshold/union choices and confusion metrics. It must finish before local answer materialization.

Existing baseline packages, original source/data and previously sealed experiments were not modified. No upload, final-model lock, commit or push is authorized by this review.
