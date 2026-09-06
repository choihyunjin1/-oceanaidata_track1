# Final release verification — 2026-09-07

Purpose: package the already selected P1 original recipe, P2 L120 and P3 numeric candidates; no new score search.

P3 whole-cold hypothesis: its frozen numeric recipe can run from an empty model directory through internal QA and answer replay without archived answers or repository data. Comparator answer SHA256: ff42a6a08c76f0d58ed2f3a9ea31a08819ada5fa6e9007af942fe0b891937960. Compare bytes and, if different, aggregate numerical differences only; never tune to that answer or its official score.

- Fresh destination: `C:/Users/cedis/Documents/OceanFinalRelease_20260907/P3_cold_validation/P3_numeric_cold_v1`.
- Source: existing sealed P3_numeric_source_v1.zip. Safe path/CRC/source-only inventory PASS before extraction.
- Execute TRAIN then PREDICT notebooks in fresh kernels; 12 CatBoost backbone fits + 5 router fits, CPU 2 threads, exclusive GPU 0, total wall cap 21600 seconds starting at prepare.
- Existing P3_numeric_execution_v1 prepared-only directory is preserved, not restarted: its six-hour clock has expired.
- Selection surface and parameters remain frozen. Prior OOF is retrospective, not new validation.
- Success: training QA, fresh-process answer replay, schema/key/order/finite/hash checks and measured runtime. A different cold answer cannot inherit the scored comparator's official score.
- Failure: preserve receipt/logs, do not restart automatically; retain scored candidate and accurately report the cold-reproduction blocker.
- P1/P2 already completed cold runs are not retrained. Source notebook validation is distinguished from actual notebook execution.
- Raw data, weights, answers, caches, execution logs and ZIPs remain local. Git receives reviewed source/tests/configs/aggregate reports/guides only. No official upload or final-model designation in this task.
