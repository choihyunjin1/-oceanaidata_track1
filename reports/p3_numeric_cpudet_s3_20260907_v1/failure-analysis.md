# P3 CPU 3-seed: 4-hour budget exhaustion, no candidate

## Conclusion

The first cold terminated at **2026-09-07 09:14:27 KST** when its preregistered 14,400-second wall cap expired. This is an **incomplete execution / runtime-budget failure**, not a scientific NO_GO, proof of poor prediction quality, or a failed two-cold determinism comparison. No candidate CSV or completed package was generated. No second cold was started. Automatic restart was not performed; heartbeat `p3-cpu-cold-qa` was paused.

## Independently checked evidence

- Supervisor terminal: `TERMINAL_TECHNICAL_FAILURE`, outer `CalledProcessError`, elapsed **14,405.344s**.
- Inner `cold_1.stderr.log`: the `cpudet.py train` subprocess exited **124**. The copied `run.py:watchdog` uses `threading.Timer(remaining, lambda: os._exit(124))`; the active runner installs this watchdog using the prepare start time and the fixed 14,400s budget. This identifies the underlying timeout, rather than treating the outer notebook exception as the cause.
- Completed and saved backbone fits: **29/36** (15 single, 14 multi). All 29 saved file hashes match fit receipts. The last completed model is `Q2_2025/20260819_single.cbm`, 700 iterations. The next scheduled fit is that fold/seed's 1,200-iteration multi model; it did not produce a completed receipt/model.
- The last completed fit receipt is at **13,967.968s**. Thus approximately **432.032s** remained for the next multi, before six full-data fits, five routers, and downstream QA/inference.
- The preceding same-fold multi fits took **1,068.969s** and **1,075.297s**, demonstrating that the remaining wall budget was insufficient even for a comparable next fit. This is a runtime observation, not a retrospective change to the sealed budget.
- Historical folds were incomplete; full-data model fits **0/6**, router fits **0/5**, completed cold runs **0/2**. The 29 count excludes the separately recorded synthetic compatibility smoke fits.
- `cpudet-training.json` absent; `05_answer/submission.csv` absent; `cold_2` absent. No final internal RMSE, exact-replay certification, or official score can be claimed.
- All **25** source pins matched; the config hash matched the build. Config SHA256: `f43f3240a993ee6f3557e4a64ea980f9ca16cde2cd68fc52f3ad991d8b3cf599`.
- Supervisor terminal SHA256: `5d1424ee3ec27d7cad2fe54b471e3191ec4721d815f4352f25d973cd3d45aab8`.

The raw terminal, model files, logs, notebooks and attempt locks remain unchanged under `C:\Users\cedis\Documents\OceanFinalDay_20260907\P3_numeric_cpudet_s3_v1`. This report does not copy raw rows or weights into Git.

## Scope and next decision

The selected 4-hour CPU budget underestimated the cost of the 36-backbone plus 5-router pipeline. The synthetic smoke validated parameter compatibility and small repeatability, not full-scale runtime feasibility. This is an execution-planning failure to acknowledge, not a reason to conclude that CPU or three-seed models cannot improve score.

Do not resume this consumed attempt, increase its budget in place, use incomplete models as a full ensemble, or start cold 2. Any changed-budget/smaller-design attempt needs a new explicit decision, identity and runtime estimate. No such experiment was started here. P3 shrink removal remains deferred until the separately required round-1 decision is resolved.

P2's completed projection candidate and P1's original fallback are unaffected. Uploads, deletions, final designation, commit and push: **0** in this monitoring action. No new model code was changed; the already-passed focused tests were not repeated for this read-only timeout diagnosis. Full production QA/package execution cannot run because their completion prerequisites are absent.
