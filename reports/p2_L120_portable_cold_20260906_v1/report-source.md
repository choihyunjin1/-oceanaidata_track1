# P2 L120: actual source-only ZIP cold/notebook reproduction PASS

**Completed: a newly extracted, outside-repository package trained three L120 models from empty03_model and produced the exact existing candidate. Actual TRAIN and PREDICT notebooks both passed.** The26061-row CSV SHA is `fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d`, and another process reproduced the entire CSV exactly. All three model-file hashes also match the earlier L120 full-training models. No tuning was performed to obtain this agreement.

This proves cold reproduction in the configured environment, not an official score improvement. The existing C60 fallback remains preserved. L120's B3primary internal mean improved slightly, but its all8secondary mean worsened; see the [complete same-resource three-seed study](../p2_c3_multiseed_completion_20260906_v1/report-source.md). This packaging task performed no portal upload, final-model lock or Git operation.

## Permanent assets and roles

Permanent verified directory: `C:/Users/cedis/Documents/OceanFinalCandidates_20260906/P2_L120_verified_v2/`.

| Asset in its parent directory | Size | SHA256 | Role |
|---|---:|---|---|
| P2_L120_SOURCE_ONLY_v2.zip |64078bytes|ecf76bd4d51ca2085c8ca3e08f6ffbf1f4ea0dcd8ee9055e2f1fdffa7ee60015|Fresh empty-model TRAIN→PREDICT reconstruction; no data/old models/answers|
| P2_L120_SAVED_PRESERVATION_v1.zip |392742bytes|968813b9d90f330d8cafa095f8613fc15f37ca0f583a571d9b8ab849cb11e678|Three trained models,exactanswer,code/notebooks and JSONprovenance; **preservation only**|

The saved ZIP is **not a verified long-term saved-inference interface**. Existing stage locks and1800-second elapsed clock are consumed and were not erased or reset. Use a new extraction of the source-only ZIP for a separately authorized new cold run; do not attempt TRAIN again in the completed folder. A future saved-model inference adapter requires a new output/receipt and separate validation.

Choose `05_answer/submission_p2_L120_3seed.csv` only for P2/OCN-02, columnsstation,layer,time,temp. The duplicate `replay_p2_L120_3seed.csv` is not a second candidate. `READ_FIRST_CURRENT_STATUS.md` in the permanent folder explicitly states official-score-not-recorded-by-this-task, B3improvement/all8decline, and the two archive roles. Source archives and the completed run are also preserved under `artifacts/p2_L120_portable_cold_20260906_v1/`; the original temporary extraction is retained.

## Actual execution, not a saved-model-only shortcut

| Stage | Actual numerical PID | Runtime | Verified scope |
|---|---:|---:|---|
| Scratch training |36892|122.235s|Three new120epoch models,empty03_model,source-only,torch.load denied|
| Training replay/QA |32548|14.672s|All166268source-training rows ×3models,exactnormalizedpredictions,19checksPASS|
| Official-key inference |25720|11.453s|AftertrainingQA,26061orderedkeys,sourcepublicfeatures/ownmodels only|
| FullCSVreplay |2216|11.313s|FreshPID,entire26061rowanswer,exactSHA|
| FinalQA |41536|includedbelow|16checksPASS;schema/keys/order/finite/LF/source/model/QAhashes|

The complete cold stage clock was173.010296seconds, below1800seconds. The actual two-notebook execution took174.375seconds, including separate kernel orchestration; its PREDICTreceipt elapsed value is cumulative, not an additional174seconds. TRAIN/PREDICT source notebooks are unmodified; executed copies and role receipts are preserved in06_docs/executed_notebooks. The generator's structural/synthetic tests are separate from this real numerical run.

Resource contract: GPU0/CPU2,threefits once,seeds20260901/02/03,120epochs,wd0.0001. The numerical core/base/run/config bytes equal the approved L120 full package. No hidden/external values,old models/answers/OOF,Public-derived coefficients,projection or routing enter training. Additional historical study28fits and this cold3fits are distinct: originalcomparison28 +originalfull3 +completion28 +cold3 =62actual research/deployment/reproduction fits. The active C60/L120 historical table uses48models, not62.

GPU was released after the five numerical process IDs had exited. The remaining preservation/archive audit is CPU-only and performs0fits/0official-input reads.

## Data/access and reproducibility boundaries

The actual source-only ZIP was extracted outside the original repository into a new directory, with03_model/04_logs/05_answer verified present and empty before TRAIN. P2_DATA_DIR points to the organizer-distributedP2_profile_restore folder. P2_DENY_REPO configured the bootstrap to reject original-repository source/model/data file access in the numerical subprocesses; the already installed interpreter runtime directory was explicitly excepted. A synthetic actual-file-read rejection test passed before execution.

Training's officialaccessrows/CSVwritten are0. After trainingQA, inference and replay parse only officialindex/samplekey columns. Sampletemperaturevalue parsing is0; finalintegrityQA also hashes entire samplefilebytes, which is distinct from parsing its value column. Nohidden truth is accessed. No rawsource data is bundled.

Environment: numpy2.3.5,pandas3.0.1,torch2.13.0+cu130,threadpoolctl3.6.0; notebook dependencies are listed in the sourcepackage requirements. Installed-environment reuse is **not** a fresh-venv test. Python audit-based network/file denial is **not** an OS-level offline sandbox; Jupyter kernel communication is separate from the guarded numerical child processes.

## Technical issue preserved, not a failed training retried

The first ZIP included files but omitted empty directories. Actual extraction detected this before any model fit or training lock. The first ZIP and extraction remain preserved. The corrected v2ZIP adds empty-directory entries only, preserving all18file bytes and the same PACKAGE_MANIFEST SHA `3457f9735b6b337dff7d1b8ee09f6a9df10351fbd6040877f90c72dc6d168476`. The [pre-fit record](preregistration.md) records the failure, correction and originalZIPhash. Synthetic archive/extract coverage was added; focused6/6tests and RuffPASS. Actualcoldtraining was run once, not restarted.

## Independent preservation checks and exact provenance

[Independent archive QA](independent-archive-qa.json) passed152checks: everycopiedfile equals the outside-run origin,packagepins unchanged,3modelSHA/QA/answerchains,actual executednotebook cells,sourceZIPemptydirectories/noexcludeddata,and byte verification of every savedZIPmember. SavedZIP44files exclude2raw.log files andthe duplicate replayCSV; OOF/probes/cache/rawdata are absent. JSONQA,consumedlocks and originalclock remain preserved. The permanentdirectory retains all original files, including those omitted from thearchive.

- Training/modelmanifestSHA: `88a4595b6943e883812876d9cef647323f84159ee5dea7c7b9aad494004e3447`.
- TrainingQA: `eb3760303511bb0f7d787e804058ba3a47f2b3e3bdbf083f7c6e56a16edb5a97`.
- Inference: `7c045746c8052fbc8e8beebe85fb8b2e332eb48645a0b7ef5c242f49a0581e91`.
- FullCSVreplay: `61aa62582da0672e296d5383cce02b885a941a753cd612ffeafaa72de3416361`.
- Final16QA: `edd1d11ea80305dccd3fb540c708322ae4a3c7ac90687ec542f0c57976580376`.
- Coldterminal: `f0fc8e80c5c12a677aa5ccd11adf9f00588cba5d20b29bf275ed61fa4df060f4`.
- Archive152QA: `5b6f763f45139c7d43d12577d45ba92fcc0a2aedb209bc09be255645fb4f2271`.

Cold/model/source-package readiness is PASS. Officialscore,finalsubmissionlock,freshvenv/OS-offline validation,and a reusable long-term saved-inference interface are separate and not claimed here. [The saved-inference follow-up design](long-term-saved-inference-design.md) is a proposal only; no additional GPU or inference run is started.
