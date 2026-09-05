# P1 — active task contract

Read with [organizer policy](00_ORGANIZER_DATA_POLICY.md) and the distributed P1 README.
Historical audits and former gates are [archived](docs/archive/instructions_20260905/00_MUST_READ_FIRST.md); they are not current instructions.

- Task: row-wise binary anomaly detection for temperature; salinity/depth are supporting observations. Primary metric is pooled binary F1 from summed TP/FP/FN, not mean fold F1.
- Source: P1_DATA_DIR; immutable organizer files. Official schema: station,year,layer,time,label. Training labels and unknown official labels must remain separate.
- Distributed complete-series offline QC permits bidirectional context unless superseded by a written rule. Keep centered windows inside continuous station/layer segments; purge by the full feature/decoder dependency. Causal evaluation is a separate ablation.
- Use blocked chronological evaluation, not random rows. Fit preprocessing, depth contracts, threshold and decoder on inner training only; do not choose them on outer/official feedback.
- Do not force official prevalence, infer per-row truth from score changes, or deploy fixed answer-row patches.
- Check that depth/year/layer handling has the same meaning in train, inner, outer and deployment. An unseen year must not silently erase a feature.
- Record per-period/station diagnostics, but old all-month-positive, zero-anchor-removal and fixed +3-point gates do not automatically govern a new experiment. Follow its predeclared contract.
- Current approved work: [score-improvement plan v2](docs/SCORE_IMPROVEMENT_PLAN_20260905_V2.md), P1-A depth contract and P1-B eligible historical model audit.
- The historical 0.833548 F1 file is not proof that a reduced two-tree model reproduces that score. Use exact-SHA receipts and full training lineage.
- Official input access and materialization require the current task's scope and internal QA. Final submission requirements are in [the handoff](AI_HANDOFF.md).
