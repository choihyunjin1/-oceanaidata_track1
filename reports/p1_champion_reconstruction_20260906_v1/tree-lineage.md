# Tree lineage — historical24 + full4 COMPLETE; historical QA171 and full QA31 PASS

Conclusion: the preregistered O/B union merits candidate consideration; the generic cell router does not improve the primary diagnostic. Historical training completed 24 fits in 657.652 seconds; fresh PID 22800 replayed every probability and policy bit exactly (36 checks, 153.813 seconds). Independent QA PID 39748 passed 171 checks in 49.109 seconds. The root-approved separate full4 completed in 166.543 seconds (worker37984); fresh PID39912 replayed both full-context in-sample probe probabilities exactly in 21.546 seconds. Full independent QA PID4696 passed31 checks in 4.469 seconds. Official/hidden/CSV/upload remain zero. No duplicate model training was launched by this worker.

| F1 / identical keys | O1 | B3 | union | inner-fitted router |
|---|---:|---:|---:|---:|
| Q2: 133170 rows (not all H1) | .772157 | .766745 | .759895 | .778904 |
| Q3: 176738 rows | .848996 | .857801 | .872588 | .847379 |
| Q4: 111124 rows | .912333 | .908069 | .912866 | .907842 |
| Primary Q3+Q4: 287862 rows | .875598 | .878668 | .889264 | .872615 |
| All three: 421032 rows | .839026 | .840071 | .843459 | .839967 |

Primary union minus B: +.010595685 F1, CI90 [+.004312615,+.018423320], descriptive P(improvement)=1.0 over 163 paired KST days/2000 resamples. It adds 216 TP and 40 FP, removes 216 FN. All-three union delta is only +.003388059, CI90 [−.005557280,+.012893661]; Q2 worsens −.006850025. Primary router delta is −.006053015 (CI includes zero), so it is not promoted.

Risk is retained, not gated away: union's all-three station-layer worst is S-ORS/L1 −.033166009; primary station-layer worst S-ORS/L5 −.008629215 (15083 rows,274 positives). Worst primary day is 2025-11-19, −.5 F1 on only one positive among 2074 rows; this tiny positive denominator must accompany that number. Router primary I-ORS/L3 worsens −.065122201. These risk slices are descriptive aggregation of the same frozen OOF, not new policy selection.

This is a retrospective reconstruction diagnostic, not an exact restoration of the historical 28.9-point answer and not a replacement of the existing H1 evaluation contract. It gives no numerical official-score forecast. Current train-only year-depth behavior, including unseen-year fallback, remains unchanged.

Focused tree synthetic15/15 and QA synthetic11/11/Ruff PASS: [tree tests](tree-synthetic.xml), [QA tests](tree-qa-synthetic-final.xml), [seal](tree-seal.json). The first QA invocation stopped on Windows path separator normalization only; [preserved technical note](tree-qa-first-attempt.json). Only the new QA helper changed; frozen training/code/results and successful model replay were not repeated or changed. The validate-data skill drove explicit denominator, inner-selection and retrospective-claim checks.

Canonical local evidence: `artifacts/p1_champion_reconstruction_20260906_v1/tree_historical/`.

- terminal SHA `00e00d2b11ec1547619a943aa31a496a8d9c2f3afe9c649bb4708d395267e8b2`
- independent QA SHA `caf780df37ee3ef180f86d649ff124154eb18709451eb291afee652bb4203403`
- replay SHA `618e4f7045ad5377145f115f50fdeadf14c2620bc6a407701957c7c38e34f45b`
- OOF SHA `939681a0fd4595dab4af0e329a6373da24ecae43302647a9cf310ce9093013a2`

Full-model evidence: `artifacts/p1_champion_reconstruction_20260906_v1/tree_full/`.

- terminal SHA `c8d8f9e6671fd0caf5a325e90a28c4985442006cf3f04a44c3d6f3a2b7d26ad6`
- independent QA SHA `c3620cf2335a7c6099379992818a35445f1e19c6c68c5be34878291dad64fb64`
- replay SHA `313abdc7eda347b23f886cc3fa3a0fbba1b58e09eaee9694b5a74010a83bbdcb`
- exact Q4-inner selector SHA `5db4106c757be1cc3efc24d3c78fa12d5e2bae1c09a0d9066aea1c532dc1214a`
- full O seed20260813 SHA `6b41b66c31bdf038e9b5ef16617633cae94083f1bdf0a05a800be4fd47a2775e`
- full B seed20260813 SHA `afa50c41acaf334feea7d3bab1119fd895307bc22eafa063ca899de8565a28bf`
- full B seed20260829 SHA `e035b818f3975a883215d3d2eb64fa3c6f123039e269864c96972eb3d4db045d`
- full B seed20260847 SHA `8655601ee0a299c5410e359905fd1f0e9e28b39d26a6b77e8eb26f3a3f856c7f`

The whole path is 28 new model fits total, not 28+extra calibrations. Full models use all776706 organizer training rows/32126 positives. Full QA synthetic3/3/Ruff PASS is recorded in `tree-full-qa-synthetic.xml`. This is one full all-training fit path plus saved-model replay, not two independent full retrainings, not a clean-room portable package, and not an official answer or submission.

## Fixed reconstruction

- O: one XGBoost, seed 20260813; 700 trees, .04 learning rate, depth 7, min-child-weight 20, .85 row/column subsampling. Original pure fit/weights: `src/p1_qc/pipeline.py:200`, `:259`, `:648`; model defaults `src/p1_qc/models_tabular.py:76`.
- B: three LightGBM seeds 20260813/20260829/20260847, arithmetic probability mean; 700/.035/leaves63/min-child60/.85 row-column sampling/alpha .2/lambda1. Exact event-length / normal station-layer-KST-day weights and seed parameters are reused from `scripts/run_p1_meaningful_learning_curve_generation_v1.py:407` and `:468`. The legacy full CLI is never executed. Historical full B recipe receipt: `artifacts/p1_round_b_full_deployment_fit_contract_20260825_v2/`; it recorded 3 fits in 83.98 seconds on CPU8, not a current CPU4 runtime promise.
- Current clean 80 features are retained: train-only year-depth and spike scale, 168-hour plateau cap, features/encoder/rules isolated by allowed partition. The original same-year batch-depth median and uncapped plateau totals are NOT restored. Full allowed-partition decoder dependence is explicit; 21-day purge alone is not a bounded-context proof.
- Historical hardcoded router cells and fixed answer-row patches are excluded. New global thresholds and generic station-layer B/O/AND/OR choices are selected using earlier inner labels only. Threshold ties prefer the higher threshold; cell ties prefer B, then O, AND, OR. Unseen cells fall back to B. Zero-normal-label range filtering is not added.

## Evaluation and budget

Q2/Q3/Q4 chronological, 21-day purge, preceding 60-day inner; positive runs belong to the interval of their start, including end-boundary continuation, and crossing runs are excluded from training. Each inner and outer fits O1+B3: 24 fits total. O/B/union/router share exact outer keys; no unsupported rows are dropped.

Primary: Q3+Q4 pooled F1; Q2 separately (only part of H1), all-three pooled, fold and station-layer risks are secondary. All periods have been exposed previously and are not fresh confirmations. Paired KST-day bootstrap 2000 / seed20260906 is descriptive, no .8 hard gate. Outer labels never select thresholds or cells.

Separate full command, only after root review and new execution approval: exactly four all-training fits; reuse the last Q4 inner selector to avoid four additional calibration fits. This selector reflects the past Q4-inner distribution, not a newly selected latest full-training window. Historical cap 90 minutes; combined historical/full workflow cap six hours; CPU4/GPU0. Model-fit budget counts attempts before fitting. A process timeout/technical failure is terminal with no automatic restart.

## Reproduction scope

Preflight records source/config/package hashes and the actual post-import thread environment with no dataset reads. Historical execution uses only a hash-checked `P1_DATA_DIR/train.csv`; local model and OOF artifacts are retained for independent QA, never committed. Existing full models/answers are not inputs. The separate full phase emits only new models and a deterministic in-sample feature probe, not an official answer. The executed `replay` command recomputed training statistics/encoder, predicted with saved models, and compared exact probabilities/selector/bits in another process. Official materialization is absent. Full execution required the independently produced `independent-qa.json` containing `status: PASS` and the exact historical `terminal_result_sha256`.

Worker self-deadline and parent-owned process-tree timeout are both present. Inherited library thread environment is recorded explicitly; each model receives CPU4. Frozen originals remain unchanged.

## Historical commands (already completed; do not rerun existing paths)

```powershell
$env:P1_DATA_DIR = '<immutable organizer P1 directory>'
.venv-p1/Scripts/python.exe -I scripts/p1_champion_reconstruction_20260906_v1/tree.py historical --seal reports/p1_champion_reconstruction_20260906_v1/tree-seal.json --output artifacts/p1_champion_reconstruction_20260906_v1/tree_historical
.venv-p1/Scripts/python.exe -I scripts/p1_champion_reconstruction_20260906_v1/tree.py replay --seal reports/p1_champion_reconstruction_20260906_v1/tree-seal.json --output artifacts/p1_champion_reconstruction_20260906_v1/tree_historical
```

These historical/replay commands are completed; do not rerun their existing paths. Root separately approved full4 in `tree_full`, now also complete; no existing attempt is restarted. The fixed Q4-inner thresholds used by that phase are O .2/B .15, not the old pooled-OOF B .2 threshold. No historical hardcoded cells, answer patches, official prevalence constraints or score inversion are used. Any future official adapter must explicitly choose its already-defined arm and pass separate local answer/schema/replay QA; this report does not authorize that action.
