# P1 v54 historical promotable-candidate registry audit

## Decision

`NO_OVERLOOKED_PROMOTABLE_P1_CANDIDATE` (0 fit).

The frozen pre-v54 corpus contains 470 P1 terminal result/report/QA/manifest metadata documents: 374 JSON files (372 parseable) and 96 Markdown files, 155 top-level experiment IDs, and 149 result documents. The ordered `path|bytes|sha256` corpus digest is `1f0343e12fbdfdb9a8a6e9a0f16b9bdfb9c9f2309d91eed17e4bff74032dda13`. Candidate CSVs, official input rows, sample files, hidden truth, and raw submission values were not read.

## Funnel

Nine lineages had either an explicit valid internal/diagnostic PASS or a positive pooled/outer historical result worth checking. Six fail their unchanged contemporaneous promotion gate: the learning-curve candidate failed effect/CI/slice gates; the original MS-TCN failed pooled Q3-Q4 confirmation; the bootstrap lower-bound PASS authorized shadow audit only; v28 failed day and station-layer safety gates; v33a was an information probe rather than promotion PASS; and v34a failed its CI gate and metric-consistency check.

Three lineages passed the promotion gate that actually governed them and retained exact reproducible commitments: `p1_full_improvement_cycle_v1`, `P1_2_HIST_GBDT_OOF_STACK_UNION`, and `p1_public_transport_repair_cycle_20260831_v30`. Their exact committed SHA-256 values all already occur in the durable official-submission ledger. They are therefore neither overlooked nor nonduplicate candidates. Official aggregate receipts were used only for SHA membership and submitted/not-submitted status; no official performance field influenced this decision.

Counts: 9 reviewed; 3 passed gate B; 6 failed gate B; 6 had exact reproducible commitments; 3 gate-B passes were exact submitted SHA duplicates; 0 gate-B passes remained unsubmitted and nonduplicate; 0 satisfied A+B+C+D.

No candidate pointer or fresh confirmation contract is emitted because no candidate qualifies. Creating one by weakening a historical gate, ignoring an existing submitted SHA, or reinterpreting official aggregates would violate the audit contract.

## Operations

- candidate selection/materialization/runner/preflight/lock/fit/optimizer: all 0
- candidate CSV and raw submission value reads: 0
- official test/sample-submission/hidden reads: 0
- uploads: 0
- v53 remains frozen; no previous terminal result is changed
