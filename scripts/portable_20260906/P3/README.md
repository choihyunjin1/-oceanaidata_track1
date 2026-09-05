# P3 portable clean reproduction

This is P3 only: significant wave height, six leads (3/6/9/12/18/24 hours).
It is **not** the old refined-public alpha recipe and **not** an equal-only ensemble.
Train from organizer P3 data: residual CatBoost single + multi, a newly fitted chronological OOF loss router, fixed 0.2 long-lead persistence shrink. No pretrained weights or external observations.

## Layout

- 01_data: reference/hash metadata only; organizer raw data must not be redistributed.
- 02_code: the complete local implementation, fixed config, recipe and requirements.
- 03_model: empty before training; newly trained historical/full model files.
- 04_logs: own validation data and aggregate execution receipts.
- 05_answer: local reproduction answer, generated only after training/replay.
- 06_docs: function provenance and this usage guide.

Use a copy of the pristine package for each independent regeneration. Never delete or reset an attempt lock or reuse a nonempty model directory to rerun. Historical OOF/models/answers are not accepted inputs.
The current validation recipe is the fixed three-window clean baseline; this package is not the separately stopped CPU-only five-fold numeric-lead experiment.

## Environment and commands

Python 3.12.10, packages pinned in 02_code/requirements.txt, NVIDIA CUDA-compatible GPU.
Create/install the environment while networking is allowed; the execution code needs no network.
For an offline organizer machine, provision these pinned dependencies beforehand (an offline wheel bundle is not included or certified here).

Set P3_DATA_DIR to the organizer-provided P3_wave_forecast folder. No personal path is embedded.
From **any current directory**, use the Python executable of that environment:

~~~text
python -I <package>/02_code/run.py --prepare
python -I <package>/02_code/run.py --RUN_TRAINING --gpu-approved
python -I <package>/02_code/run.py --replay
python -I <package>/02_code/audit.py --training-only
python -I <package>/02_code/run.py --RUN_INFERENCE --official-approved
python -I <package>/02_code/run.py --verify-answer --official-approved
python -I <package>/02_code/audit.py
~~~

RUN_TRAINING can perform prepare itself if it has not run. The explicit sequence lets the coordinator finish CPU feature preparation before allocating the GPU. Each command is a fresh process. Allocate the GPU exclusively; CPU model/BLAS threads are 2.
The flags acknowledge local compute/data authorization, not a portal upload or final-model lock.

Training directly reads only train_wave.csv/train_atmos.csv and recreates all 24,360 training features, six historical CatBoost fits, two full-data fits and three routers. It does not read previous caches, model assets or answer files.
Before authorized inference, replay and training-only audit must pass. Inference then reads test_context.parquet's public context columns and test_index.csv key columns only. It never opens sample labels, hidden truth, baseline answers or other datasets.

## Reproduction claims

Each execution checks the six-hour budget including prepare, training, replay and first inference.
Prior on-machine evidence is about 27.6 minutes with GPU multi; new measured times belong to the new receipts, not this estimate.
GPU training can vary. Compare each run's new CSV SHA and new predictions exactly, report differences; do not modify the recipe or widen tolerances to claim a match.
A fresh process replay of one saved model is distinct from two independent empty-model training runs.
Python audit hooks reject network requests and out-of-package model/data inputs; this is not proof of OS-level network isolation or clean organizer hardware verification.

## Answer and portal choice

The only P3 answer is 05_answer/submission.csv, schema case_id,station,lead_h,hs_pred,
1,200 unique keys in the original test_index order, finite 0–30 metres.
The historical clean reference SHA used **only as a QA scalar** is
6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7.
A differing SHA is not the same officially scored candidate and must not inherit its score.
This task does not upload or lock anything. If separately authorized, the answer is for P3 / OCN-03 at https://oceanaidata.org/app/problems/7. Final-model reproduction attachments require a separate current portal-limit/locking review. This directory is not evidence of successful official final submission.

## Constants and data policy

Full provenance is in 06_docs/source-provenance.json; origin paths are historical relative metadata only, never runtime imports.
The recipe's original fixed config SHA is e5c2eff7bc9fcd44759d0bc30d965c86eca10c038807f186b1379131aed9b169.
Single: 700 trees/depth6/lr0.035/l2=8. Multi: 1,200/depth7/lr0.03/l2=10 GPU.
Historical seeds 20260816/17/18; full seed 20260817. Loss router Ridge10, temperature2, strength0.5; short-lead route remains equal single/multi; long leads receive the fitted router then 0.2 persistence shrink. The 0.5 baseline mixture is not a description of the full final policy.
Only organizer-distributed observations are eligible for training, validation, selection and postprocessing. Public-score-inverse coefficients and external observation ancestry are excluded. No raw data redistribution.
