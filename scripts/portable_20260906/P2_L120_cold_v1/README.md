# P2 L120 — independent cold training and prediction

This package reconstructs the fixed L120 three-seed candidate from the organizer-distributed observations. It is not a new recipe or the highest official score. The prior candidate SHA is `fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d`; a different cold output must be reported, never tuned to match it.

## Inputs, stages and output

Set `P2_DATA_DIR` to the distributed `P2_profile_restore` directory. No dataset is included in this archive. Use the configured Python/CUDA environment with the exact package versions listed in requirements.txt. Installation is an environment preparation step before an offline reproduction run. No network download or pretrained weight is needed at execution time.

The package separates `01_data` source references, `02_code` self-contained Python, initially empty `03_model`, `04_logs` consumed locks/QA, initially empty `05_answer`, and `06_docs` provenance. Do not copy existing weights, answers or OOF arrays into a cold package.

From this package directory, the notebook TRAIN cell should execute this argument list:

```python
[sys.executable, "-I", "-B", "02_code/stages.py", "TRAIN"]
```

It creates three new models and replays all 166,268 source-training rows per model in another process. This replay is not a validation score. Only after TRAIN succeeds, the PREDICT cell executes:

```python
[sys.executable, "-I", "-B", "02_code/stages.py", "PREDICT"]
```

PREDICT reads official key columns, uses only the three locally trained models, creates the candidate, repeats complete inference in another process, and verifies schema/order/finite/hash. It does not fit a model or upload anything. The whole two-stage cold run has a 1,800-second elapsed-time cap, so do not leave a long manual pause between the cells. Each stage is consumed exactly once. Preserve a failure and use a separately authorized new run directory; never delete a lock to retry.

Alternatively execute both actual notebooks, each in its own kernel, using `python -I -B 02_code/execute_notebooks.py --package . --timeout 1800`. This is the same TRAIN then PREDICT sequence, not an additional run: choose either this command or the manual cells, never both on a consumed package. Source notebooks remain unchanged; executed notebook copies and receipts are saved in `06_docs/executed_notebooks`. Kernel communication is local; the numerical child processes deny network connections.

Submit only `05_answer/submission_p2_L120_3seed.csv` for problem P2 (OCN-02): columns `station,layer,time,temp`, 26,061 rows, UTF-8/LF. The replay CSV is a verification copy, not a second candidate. Verify `04_logs/independent-qa.json` and `04_logs/cold-terminal.json` first. Portal upload and final-model lock are separate authorized actions, not included in these scripts.

## Fixed training and evidence

C3 11-context DeepSet; 120 epochs; AdamW wd0.0001/lr0.001; batch4096; normalized Huber with fixed domain weighting and gradient penalty0.01; fixed blockmask; seeds20260901/02/03; CPU2/GPU0. No projection, copula, routing, post-fit calibration, Public-derived coefficient, old answer or external observation is used. Both target temperature and salinity are masked before feature construction. Trainable weights are initialized anew and torch.load is denied during training.

Historical Sep-Oct B3 remains primary. The first study's B3 three-seed mean improved by0.00477927°C, but its injected-outage interval and natural T5-missing slices worsened. Additional all-eight-fold three-seed evidence is a separate secondary report, not a replacement primary or a new independently selected winner.

Previous actual full-three-seed training took121.672 seconds, and training/inference/replays/QA together169.368 seconds on the configured CUDA environment. This new cold package's actual receipts take precedence. Exact original answer SHA and new-process replay are separate checks; neither promises an official score gain.

For an outside-repository test, set `P2_DENY_REPO` to the original research repository and extract this package elsewhere. The bootstrap denies its file access except the interpreter's installed runtime directory if that runtime resides there. This exception permits installed third-party libraries, not repository source/models/data. Numeric core/base/run files are unchanged local copies. The Python network/file audit guard is not an OS sandbox. Reusing the installed environment is not a fresh-venv test; no fresh-venv success is claimed.
