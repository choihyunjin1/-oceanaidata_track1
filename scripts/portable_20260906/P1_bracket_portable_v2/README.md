# P1 bracket portable v2

Two ZIPs serve different purposes. Neither contains source observations, existing answers, sample prediction values or hidden labels. Supply the unchanged organizer dataset with `P1_DATA_DIR`; use Python3.12.10 and pinned `02_code/requirements.txt`. Network operations are denied by a Python audit hook; this is not OS-level isolation or a fresh-environment certificate.

## Saved-model ZIP

`P1_bracket_saved_v2.zip` contains the original CPU4-trained full O/B models, exact recipe and validated code plus a separate clock-independent `archive_infer.py`. Unzip to a new folder, then run:

```powershell
python -I <extracted-P1>/02_code/archive_infer.py --official-approved --output-root <new-empty-replay-folder>
```

This uses CPU2 for inference only and performs0 training fits. Every invocation requires a new output folder. It never edits or bypasses the historical runner, attempt lock or clock. It checks the saved source/model/recipe hashes, reads official observations and sample keys only, computes the entire answer and compares its SHA with `9031c84ea72dfa4294406dd995525e89e8975a76983e9f9d7a7b2ba74dbad93a`. That SHA is verification metadata, never an optimization target. Model-source eligibility and actual official score are separate evidence.

## Cold-start code ZIP

`P1_bracket_cold_v2.zip` has empty03_model and05_answer, no previous lock or PASS receipts, and an explicit new CPU2 resource variant of the original four-fit workflow. Unzip to a new folder and run from P1:

```powershell
python -I 02_code/run.py train
python -I 02_code/run.py model-qa
python 06_docs/qa_training_independent.py <this-new-P1-folder> --data <organizer-P1-dataset>
python -I 02_code/run.py infer
python -I 02_code/run.py verify
```

Exactly four fresh CPU2/GPU0 fits: final-inner B/O then full O/B,700 trees, same80/107 features, final-inner2025-07-12~09-10 and training cutoff06-21 with21-day purge. Policy and thresholds are rederived on that inner interval; no prior answer/model is an input. No label0 natural-variation deletion/downweighting, depth fallback, cell policy or new postprocessing is applied.

Model serialization hashes may differ from CPU4 because resource settings are part of the model. Independent QA reports those differences without forcing the old model hash; final answer equality against9031 must be reported separately. A mismatch does not permit additional fitting or threshold adjustment. The new workflow retains its own3600-second cap; existing completed folders must not be restarted. A full cold run and a saved-model replay are distinct claims. Check the accompanying execution receipt for which tests actually completed.

Folders:01_data external pointer only;02_code portable source;03_model trained weights/encoders/statistics/recipe;04_logs new internal replay probes;05_answer computed CSV;06_docs QA and scope. Do not upload internal probes or redistribute the organizer data. Final upload/model lock is performed separately by the root operator.
