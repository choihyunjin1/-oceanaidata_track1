# P1 tree union + MS-TCN portable candidate

P1 only. This preserves the frozen `b2f17f5cda8030cb3d97fbb504e6babb6aef8ba7fe555092901479677af0625e` policy and does not include a new model search. No official score is transferred to a different regenerated file.

Structure: `01_data` references the organizer dataset (not redistributed), `02_code` contains a closed source snapshot, `03_model` contains newly trained or explicitly archived models, `04_logs` records stages, `05_answer` holds generated answers, `06_docs` holds evidence. Set `P1_DATA_DIR` to the distributed `P1_qc_anomaly` directory. Do not provide hidden/original external observations.

## Commands

Run with the pinned Python environment from the extracted package. `--data <directory>` optionally overrides `P1_DATA_DIR`.

```text
python -I 02_code/run.py train --gpu-approved
python -I 02_code/run.py infer
python -I 02_code/run.py verify
```

`TRAIN.ipynb` and `PREDICT.ipynb` invoke these commands in separate processes; they do not implement a different model. The saved-model archive refuses `train`; use its `infer` and `verify` commands. The cold archive requires an empty `03_model` and has no old models, answers, OOF, or feature caches as training inputs. Use a new extracted folder for another attempt. Existing outputs are never overwritten.

## Exact frozen training recipe and time

The new cold contract is **11 fits**, not the old research CLI's 31 fits: exactly the original Q4 earlier-inner O1/B3 four fits, original inner-only threshold/selector calculation, full released-train O1/B3 four fits, and original full-train MS-TCN three seeds. The actual tree arm is fixed `union`, not the fitted cell router. Tree features remain the active train-only year-depth dictionary and 168-hour plateau cap, 80 columns; no bracket, range rule, fallback change, or current tuning candidate is included.

Tree CPU4, original MS CPU2 plus exclusively allocated CUDA device 0/bfloat16; no silent CPU/fp32 replacement. MS width512,165features,3seeds,e150 on the original300epoch schedule,2048row windows,decoder high0.8/low0.4/snap12/min19. Root must allocate the GPU before training or MS inference. The current source MS already includes subtype and boundary heads.

One new cold clock includes source preparation,11fits,model replay/QA,inference,and answer replay: maximum21,600seconds. It does not reset or alter any expired old research clock. Saved-model inference uses an independent1,800second per-command watchdog and is not evidence of new training. A completed cold folder is not a perpetual inference service; use a separately built saved archive for long-term replay.

The Q4 selector is reconstructed using the exact original earlier-inner split, whole-positive-run ownership/exclusion,21-day purge,and complete permitted-partition feature context. Other historical folds are not refitted or used as training inputs. This packaging contraction does not create fresh validation. Inner scores are calibration evidence only. Exact saved replay and newly trained answer exact restoration are separate checks; any mismatch is preserved and cannot trigger coefficient changes or repeated training.

## Submission and limitations

After all local QA passes, the answer is `05_answer/P1_submission.csv`:169,011rows and `station,year,layer,time,label`,integer0/1,in sample-key order. Sample prediction values and hidden truth are never parsed. This package does not upload, spend leaderboard attempts, or lock a final model. The source ZIP is for empty-model reconstruction; the saved ZIP is for model-backed replay. Do not confuse their evidence.

No external or pretrained observations/weights are used. All models are trained on the released dataset. Source snapshots retain original code unchanged, including import-only helper definitions; only the documented adapter entry points execute. Runtime prohibits network calls at Python audit boundaries; this is not an OS-level air-gap verification. Pinned installed library versions are supplied; an offline wheelhouse, fresh OS, and different-GPU reproducibility are not claimed tested.

Current status before execution: CODE_PREPARATION; source-only cold, saved ZIP replay, and six-hour full completion are not yet PASS. Read the generated `06_docs` receipts for actual completion and hashes.
