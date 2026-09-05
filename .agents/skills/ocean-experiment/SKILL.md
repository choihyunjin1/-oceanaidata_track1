---
name: ocean-experiment
description: Run or resume Ocean AI P1/P2/P3 experiments in this repository, checking candidate ancestry and reusing exact validation receipts. Use for model experiments, progress checks or candidate QA, not unrelated work.
---

# Ocean experiment routing

Follow root AGENTS.md and the user's current scope. This skill adds no approval gate.

1. Read organizer policy and the active problem contract once. The current plan and exact config/receipt define the branch and budget.
2. Progress checks need only process metadata, progress/terminal existence and relevant error tail, not renewed research/tests/training.
3. Before fitting, identify the mechanism, comparator, available targets, split/purge, metric/unit, equivalent prior results and next branch. External/Public-inverse ancestry is ineligible.
4. Test the synthetic parameter/schema/path contract before GPU allocation. Root assigns one GPU owner; workers own disjoint new files.
5. Run focused code checks using scripts/agent_verify.py or matching receipts. Code-test PASS is not model/candidate approval.
6. Separate internal metric, independent QA, replay, official score and final-package status. Link one canonical result instead of repeatedly copying metrics.

Details: [development loop](../../../docs/AGENT_WORKFLOW.md).
Do not edit installed third-party skills, frozen runners or historical receipts during routine work.
