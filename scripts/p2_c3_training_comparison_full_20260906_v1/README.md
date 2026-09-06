# P2 L120 three-seed candidate (not the current official best)

This isolated package trains three new models from distributed observations only. It does not load historical models, OOF predictions, old answers, bin17/Public-calibration coefficients, external data or hidden truth. Exact policy: 120 epochs, AdamW wd0.0001, lr0.001, batch4096, gradient penalty0.01, fixed domain weights/blockmask, 11-context C3, three seeds20260901/02/03, GPU0/CPU2. No projection or routing is added.

01_data is a source reference only; 02_code contains locally copied core/base/driver; 03_model starts empty and contains newly trained model weights; 04_logs holds locks/provenance/QA; 05_answer holds candidate/replay CSV; 06_docs contains immutable historical result/QA/replay evidence.

Set P2_DATA_DIR to the distributed P2_profile_restore directory. With the already configured numerical environment, run python -I -B 02_code/run.py execute. This consumes a new full-cycle lock: RUN_TRAINING (3 fresh fits), TRAIN_REPLAY (separate-PID all166268 training predictions per model, not a holdout score), RUN_INFERENCE (official key columns after training QA), REPLAY (third/fourth process exact wholeCSV), FINAL_QA. Do not run consumed stages again. The existing portable base.py main is not the entrypoint.

The final file is 05_answer/submission_p2_L120_3seed.csv, station/layer/time/temp,26061 rows,UTF-8/LF,12 significant digits. Verify 04_logs/independent-qa.json and terminal_result.json before handing it to root. 05_answer/replay_p2_L120_3seed.csv is a replay check, not another candidate. No portal upload is performed here.

Historical evidence: B3 all3 natural .488284326→.483505057°C; additional-outage wholefold .538760112→.563601402°C and interval .445059242→.561965765°C are worse. Full model preparation is an information-value candidate, not proof of an official score gain. Current best C3 SHA46d194...c071 remains untouched.

Estimated full GPU training: approximately137 seconds from the previous full3 60epoch68.391-second run, plus source preparation/inference/replays. Actual receipts take precedence. The entire current cycle has a30minute wallcap. No fresh-venv or outside-repository cold-run claim is made. Copied numerical core has no research-repository import; Python allowlist/network denial is not an OS sandbox. Original distributed dataset ZIP is not bundled.
