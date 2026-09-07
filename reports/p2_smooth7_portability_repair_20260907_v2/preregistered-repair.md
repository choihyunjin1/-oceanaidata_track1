# P2 portability repair v2 — no scientific change

User requested correction before a second Fable review. No portal action or final confirmation authorized by this repair. Existing v1 packages/CSV/receipts and active P3 training remain unchanged.

Scope: in a NEW v2 package remove only the requirement that fresh base predictions equal the historical fee6118b SHA. Record actual/expected/equality instead, requiring this run's independent integrity QA and current answer hash match. Keep source, dataset, model provenance, recipe, schema, keys/order, finite checks, scalar PAVA, replay and attempt protections. CUDA remains required; no CPU fallback or cross-GPU numerical claim.

Numerical invariants: L120 three seeds, 120 epochs, same learned architecture, same smooth7 and endpoint clip/PAVA. No new model selection, window/seed/coefficient change or Public fitting. New fits: exactly 3 in a new SOURCE_ONLY cold notebook replay; saved-model path 0 fit. Existing source-only core manifest stays identical if its files do; new addon manifest and final ZIP manifest must be regenerated.

Validation before use: synthetic historical-SHA mismatch continues with valid current QA; altered-after-QA, failed/empty/missing QA must reject; old/new smoothing exact on synthetic values. Execute newly extracted SAVED_PREDICT and cold TRAIN → PREDICT → SMOOTH_PROJECT top-to-bottom, separate PID infer/replay, then compare final bytes to official candidate 794268f15a0a7ac18ecd4dc99757083e159c49d639d2a8a72df3f835414cc481. A different answer must be reported, not assigned the official score. Do not overwrite or retry consumed attempts after failure.

Old final ZIP aab30bbe5e244098c1ef379b09077ccd4e98bd6a6cd0ccf778109a7f5282054d is retained. New root: C:/Users/cedis/Documents/OceanFinalDay_20260907/P2_L120_s3_smooth7_proj_v2; final wrapper root: C:/Users/cedis/Documents/OceanFinalSelected_20260907/P2_v2. New package must receive Fable review and explicit user resume before portal confirmation.
