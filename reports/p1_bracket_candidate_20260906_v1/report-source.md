# P1 bracket-only local candidate

## Conclusion

**Local candidate preparation is complete: four new fits, fresh-process model replay, root training QA49/49, complete answer replay and root answer QA26/26 all PASS.** The preserved answer is `artifacts/p1_bracket_candidate_20260906_v1/package/05_answer/P1_submission.csv`,169011 rows/7014 positives, SHA256 `9031c84ea72dfa4294406dd995525e89e8975a76983e9f9d7a7b2ba74dbad93a`. No upload, official scoring, final-model lock or Git action occurred in this task. Existing baseline package, sealed experiments and original data are unchanged.

The training candidate is O80 plus B107 (the exact validated27 bracket features only). **Fresh final-inner selection chose balanced B107 alone at threshold0.1**; the final answer is not an O/B ensemble. O was nevertheless retrained and verified under the four-fit contract. This is not a depth-fallback experiment. Neither observed outer thresholds nor an expected positive count was transferred.

## Measured results and scope

Same final-inner population:123372 rows/4445 positives, training467282 rows, fixed dates below. Baseline receipt: `artifacts/portable_cleanroom_20260906_v1/P1/run_a/train_result.json`; candidate receipt: `artifacts/p1_bracket_candidate_20260906_v1/package/train_result.json`.

| Final-inner policy selection | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|
| Unchanged baseline: balanced_union | 0.8946535135793448 | 3673 | 93 | 772 |
| Bracket candidate: balanced, threshold0.1 | 0.9002041551579201 | 3748 | 134 | 697 |
| Difference | +0.0055506415785753 | +75 | +41 | −75 |

This interval selected both policies: the improvement is an **internal selection result**, not fresh holdout evidence or an expected official-score gain. The original O arm's inner/full model SHA exactly matches the baseline (`3cd6c8f…ddd33`, `567ea9c…1df3b`), supporting that its training path remained unchanged. B's extra features and consequent inner-selected policy may change predictions; the effect is not a depth-only causal claim.

After the candidate was frozen and fully generated, root compared its answer with the preserved fallback SHA `5971e145…1db28a`:1417 rows differ (1018 zero→one,399 one→zero; G-ORS100/I-ORS629/S-ORS688). These are **differences, not verified corrections**. Hidden truth was never accessed, and the comparison was not used to tune or select thresholds.

| Stage | PID | Seconds | Result |
|---|---:|---:|---|
| Four fresh fits | 40208 | 254.907 | Complete; official input0 |
| Model reload/internal-probability replay | 33056 | 105.828 | All4 exact; official input0 |
| Root independent train-only QA | 41404 | 24.219 |49/49 PASS; official input0 |
| Local official-observation/sample-key inference | 41884 | 25.938 |169011-row CSV |
| Complete answer replay | 39568 | 25.906 | Exact bytes/SHA |
| Root final answer QA | 8552 | See root receipt |26/26 PASS |

Answer replay completed589.317 seconds after the training start, including inter-stage waiting, within3600 seconds. One complete training workflow was run; saved-model replay is not a second full retraining. Each inference process read169011 official observation rows and169011 sample-key rows after the QA gate; sample prediction values, hidden rows, external observations, inference fits and uploads remained0.

## Preserved evidence and limits

New persistent root: `artifacts/p1_bracket_candidate_20260906_v1/package/`.38 source-package files and3 supplemental documentation/QA files were copied into this previously nonexistent directory; every copied file's SHA was checked, mismatches0. No caches or distributed observations were copied. The original temp workflow was preserved, not moved or overwritten.

- `train_result.json`: SHA `958f35f640c0c45d557f56ea0235026c102a8881b4c8abd2ee19d28ebcdaa7ec`.
- `03_model/frozen_recipe.json`: SHA `2c7c0cd68eb87ce3c4d331fae10f3cb4a3bd74c86aeb658bc6972d952936b9b7`.
- `06_docs/model-replay-qa.json`: SHA `fd724e1c857cacc09af76422c5e4e14a209f9df6e90e6bfc8a91a511e070a9f9`.
- `06_docs/independent-training-qa.json`: SHA `4b9650fd0bd82082923a9594dd076176eeb736da34415285557cd8f1b6fdfe25`.
- `06_docs/answer-replay-qa.json`: SHA `bf79d83d57113e2aa92b87c7d1e30af6b17016d157a7833c9ba2d6e23854c057`.
- `06_docs/root-answer-qa.json`: byte-exact copy of this report folder's root receipt, SHA `6757af4f5b42eb37c98b9da9483a4c4eb1e72b4ae3076156a133ea61ca758b8c`.
- `06_docs/qa_training_independent.py`: byte-exact root QA helper, SHA `2bd9cd4c85754e9dbbe114057de22e584bf31c82dc3f6b95a8fea1198262dde3`.
- `06_docs/REGENERATION_AND_LIMITS.md`: new-empty-folder4-fit instructions. The completed runtime's3600-second clock expires from the original training start, so **the completed folder is not a permanent saved-model inference adapter**. Replaying it tomorrow is blocked. No clock/lock bypass is allowed; durable inference-only packaging is separate work.

Future official comparison is root's decision. Existing fallback remains intact; clean-machine verification, two independent full rebuilds, a new ZIP and final-model lock are not claimed.

## Frozen contract

- New empty OS-temp package: `C:/Users/cedis/AppData/Local/Temp/ocean_p1_bracket_candidate_20260906_v1_run3/`. Earlier `preflight`/`run1`/`run2` staging folders had zero fits and are left untouched; only `run3` is designated for this workflow. A pre-fit import-path issue found in root review was corrected before training; general and isolated `-I` entrypoints now pass from a different working directory.
- Four fits only: inner B then O; full O then B. CPU4, GPU0, fixed700 trees. Entire workflow wall cap3600 seconds, including inter-stage review, with no automatic retry.
- Final-inner `[2025-07-12T00:00+09:00, 2025-09-10T00:00+09:00)`; training strictly before2025-06-21,21-day purge and unchanged whole trailing-positive-run removal. Expected467282 training rows/123372 evaluation rows. Partition selection occurs before statistics, encoders, features, rules and decoder.
- O uses the unchanged80 features. B uses those same80 plus bracket windows6/24/72 hours with1-hour exterior flanks (27 columns; maximum bracket radius37h). Whole allowed-partition dependencies of the existing decoder are not presented as bounded by the purge.
- Same station/year/layer depth lookup, numerical NaN retained, unseen encoder categories−1; no new fallback or forced unknown-category code.
- Final-inner F1 selects the policy; it is not a fresh holdout score. Full-model first4096 sorted train-feature probabilities are in-sample determinism probes only, with no quality metric.

## Preflight evidence

17 focused synthetic tests PASS, zero failed/skipped; Ruff PASS for builder/tests and the generated run/bracket modules. Tests cover exact research-to-portable O/B feature matrices, encoder maps, balancing weights, rules and policy selection;37h/gap boundaries; partition sentinel isolation; terminal positive-run exclusion; missing-depth semantics; decoder equivalence; independent F1 arithmetic; empty-folder protection; phase/network/repository denial; blocking official reads before independent QA; and independent-working-directory normal/isolated entrypoints.

Frozen hashes:

- Candidate config: `8d62f469eafedee60ee8de698ff3e3f9ebd309760c6ad15abef749aba0aa838c`
- Builder: `57683321d301fc4193b618387a56b929c1fad3c9f8b751e07996f59b76fc5fd4`
- Runtime template: `8fa0e6bfcd7774b8d33597a70abeab53686daf324aa48293423600b8a71628e4`
- Focused tests: `dafbc01ccbbe545ac6708b28a616bb1cc1b09acf9cf83f1ee4cbaa4d9f863bd0`
- Exact bracket-source runner: `dc111189f3f58a8f01ec59634ee6bb51ed63928399efe592fc032d5317828e93`
- Generated `02_code/run.py`: `909ce7f112fd60b34d42da2206e2c27a270feac6b3367509d79ecd9f94ea985b`
- Generated source manifest: `ba3f9b9f1e3d0642df0aaa05eb8822f7356e28383ef916bfb45c3f97952629c4`

## Completed sequence

1. Root preflight review/start; four fresh fits in empty03_model — completed.
2. Separate-process `model-qa`: reload all four new models; verify all final-inner probabilities, policy/thresholds and independent F1; recompute full in-sample probe probabilities. Hash training result/recipe/model files. Compare unchanged O model hashes with baseline metadata, without opening old model or answer contents.
3. Root independent train-only QA writes `06_docs/independent-training-qa.json`, linked to the exact training result/recipe/model-replay hashes. Missing or mismatched receipts block official input reads.
4. Authorized local test-observation/sample-key inference creates `05_answer/P1_submission.csv` for169011 rows; another PID recomputes the full answer and requires exact bytes plus schema/key/order/unique/binary/finite checks.
5. Readiness is reported separately from scientific improvement, official scoring and final-model lock. No upload or Git was performed in this task.

## Evidence for preparing, not promising, this candidate

The earlier two-sided comparison and separately replayed unknown-year stress retained all primary rows and favored bracket relative to the same-condition baseline, with worsened station-layer slices. See `reports/p1_tuning_twosided_20260906_v1/report-source.md` and `reports/p1_tuning_depth_stress_20260906_v2/report-source.md`. These retrospective results do not prove future or official improvement. The new fallback remains excluded. This workflow does not run8-fit double regeneration,8×HPO, or a clean-machine certificate.
