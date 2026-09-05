# P1 bracket-only candidate — not an uploaded or locked final model

This independent folder starts with empty `03_model` and `05_answer`. It uses only the distributed P1 data via `P1_DATA_DIR`; data and existing model/answer contents are not copied.

1. `python 02_code/run.py train` — exactly four CPU4 fits: final-inner B/O then full O/B, 700 trees each. The final-inner policy and thresholds are fitted again, not transferred from historical outer results.
2. `python 02_code/run.py model-qa` — a separate process reloads all four new models and recomputes internal predictions/selection plus an in-sample full-training probe. No official input is allowed.
3. Independent reviewer writes `06_docs/independent-training-qa.json` with `status: PASS`, a different `pid`, exact `train_result_sha256`, `recipe_sha256`, and `model_replay_qa_sha256`. Missing or mismatched review blocks inference before official reads.
4. `python 02_code/run.py infer` — only after both QA receipts, read distributed test observations and sample **keys only**, produce `05_answer/P1_submission.csv`.
5. `python 02_code/run.py verify` — another process recomputes the complete answer and requires exact CSV bytes/hash, row count, order, unique keys and binary values.

Set `P1_DENY_REPO` to the original repository path to deny access to it during runtime (the Python environment directory remains permitted). Network access is denied by the phase guard. Run each command from this new package, using its pinned `02_code/requirements.txt` environment. The 3600-second workflow cap starts at training and includes QA/inference/replay; no automatic retry or second full training is provided.

`01_data` is a documented external data pointer, `02_code` source snapshot, `03_model` newly trained models/encoders/statistics/recipe, `04_logs` internal numeric replay probes, `05_answer` the generated answer, and `06_docs` receipts. Never upload the internal probes or treat the final-inner selection score as an independent holdout. The fixed full-train probe has no quality score. This is one four-fit rebuild plus model/answer reload replay, not two full retraining runs or a clean-machine certificate. Root decides submission and final-model lock separately.
