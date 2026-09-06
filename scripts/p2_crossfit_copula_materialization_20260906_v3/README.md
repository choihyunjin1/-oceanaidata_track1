# P2 C3 + fixed profile-copula candidate — standalone CPU training

This directory is built only if the completed paired8fold internal evaluation and independent QA retain at least one strict B3 September–October natural-input RMSE improvement. This is not an official-score guarantee. Whole8fold, natural missingness, additional17day outage risks and empty winter outage support must be read with the research report. Correction strength stays1.0; there is no row patch, Public-score coefficient, hidden truth, external observation, pretrained weight or prior answer dependency.

## Local data and reproduction

Organizer `observations.csv`, `test_index.csv` and `sample_submission.csv` remain outside the package. Set `P2_DATA_DIR` to that distributed folder. `01_data` is a reference location, not a redistributed dataset. Obtain/install the exact dependencies locally; this package has no network download code. Existing local dependency environment is not proof of a clean offline wheel install.

Start from a new package copy with empty `03_model` and `05_answer` and no attempt lock; never delete a completed/failed attempt. The package contains `02_code`, frozen`config.json`, and requirements. All models, scalers/covariances/CDFs and calibration outputs come from this invocation's distributed training data.

From an already completed package, `python 02_code/create_empty_copy.py NEW_DISJOINT_DIRECTORY` copies only code/config/README/requirements and creates empty output folders. Change into that new directory before running the commands below. Existing models, answers, calibration arrays and attempts are neither copied nor deleted.

```powershell
$env:P2_DATA_DIR = 'organizer-distributed-P2-directory'
python -I -B 02_code/run.py RUN_TRAINING
python -I -B 02_code/run.py VERIFY_TRAINING
python -I -B 02_code/run.py RUN_INFERENCE
python -I -B 02_code/run.py REPLAY
```

CPU2 threads; GPU disabled. Shared C3 backbone: three seeds20260901/02/03,60epochs, original blockmask augmentation/domain-balanced normalized-Huber+gradient-penalty recipe. If crossfit is retained, six additional inner backbones provide out-of-fold calibration from two source-only selected full calendar months and7day purges. One small covariance/CDF fit per retained arm. Thus in-sample only4fits, crossfit only10fits, both11fits. The training plan is frozen before fitting; full inner dates are recorded in config, not chosen from full-fit error. Maximum training3600s; actual training, verification and inference/replay times are separate receipts. This package's CPU full C3 can differ from the preserved CUDA C3 fallback. Compare SHA; do not transfer the fallback's official score to another file.

The full-training in-sample/saved-model replay checks are reproducibility/integrity checks, not new holdout performance. Submission quality evidence is the earlier paired8fold internal comparison. Fresh-process whole-batch replay remains bit exact. No opposite-surface tolerance changes output values.

For a newly extracted completed ZIP, `python -I -B 02_code/run.py REPLAY --replay-receipt zip-extraction-replay.json` creates a separate verification receipt without replacing the original replay record. This tests saved-model inference after extraction, not another full retraining. The archive excludes original data and train-only calibration/replay arrays; a new empty training run recreates those arrays before `VERIFY_TRAINING`.

## Files and upload

- `01_data`: references only; no source data in ZIP.
- `02_code`: local-only numerical primitives and runner; no original repository import required.
- `03_model`: newly fitted shared C3 models, optional inner models, covariance/CDF models.
- `04_logs`: training/QA/inference/replay receipts and train-only local calibration arrays.
- `05_answer/P2_insample_full.csv` or `P2_crossfit_full.csv`: only retained arm files exist after successful inference. `P2_C3_control.csv` is the same-run CPU comparator, not automatically the old fallback.
- `06_docs`: sealed manifest and lineage. No model-final lock or portal upload is performed by these commands.

Answer schema is `station,layer,time,temp`, exactly26061 unique query keys, sample order, finite Celsius. On OCN-02 `/app/problems/6`, choose the exact candidate agreed in the current handoff, not the control by accident. Parent/root verifies current quota/deadline and handles upload. Model-final-submission locking is distinct authorization. Generated source/models/CSV/NPZ are local-only, not Git assets. Hidden truth is never read.
