# MS-TCN e150 reconstruction: implementation ready, training not started

The new component reconstructs the frozen e150 MS-TCN **proposal** from organizer
`train.csv` only. It is not an already reproduced champion, and its historical
union F1 cannot be transferred to a newly trained tree anchor. No organizer row,
old model, prediction array or official input was opened while implementing this
component. Source, configuration, aggregate receipts and file metadata were read.

## Entry points and exact boundary

- New implementation: `scripts/p1_champion_reconstruction_20260906_v1/mstcn.py`.
- Contract-only: `python mstcn.py contract` (no data/model read).
- Approved future full training: `python mstcn.py train --source-root <research-root>
  --train-csv <organizer-P1>/train.csv --output <entirely-new-run>
  --training-approved --gpu-approved`.
- Immediately after training, new PID: `python <new-run>/02_code/mstcn.py replay
  --output <new-run>`. Both stages share the original 21,600-second deadline.
- Composition API: `predict_proposal(frame, model_dir=<new-run>/03_model)` returns
  `(input-order keys, frozen .8 proposal bits, source/model/encoder/replay/key-hash
  receipt)`. The caller must authorize the frame separately. It opens no official
  CSV, sample label, anchor CSV, patch JSON or prior probability artifact.
- `predict_frame` additionally exposes the raw three-seed mean in memory. Neither
  entry point fits an encoder/model or applies the independent tree union.

Use a fresh process without a foreign already-imported `p1_qc` package. The new
run snapshots the exact nine dependency files and this module; executing-module
SHA, dependency/model/encoder SHA and fresh-process replay are checked before
inference. Old `03_model/weights`, old cache, and old deployment outputs are not
fallback locations. `03_model` is an explicit required argument, addressing the
previous package's disconnected `retrained_from_scratch` versus `weights` paths.

## Frozen numerical recipe

| Item | Exact recipe / provenance |
|---|---|
| Architecture | MS-TCN++/ASRF, width 512, 165 inputs, 52,568,587 trainable parameters |
| Seeds | 20260827, 20260839, 20260863; fresh initialization for each |
| Optimizer | AdamW, LR .0003, weight decay .0001, clip norm 1 |
| Schedule | Stop at epoch 150, retain 300-epoch cosine denominator, 10-epoch warmup, minimum LR .000003 |
| Batch | 64, accumulation 1; expected 1,707 selected windows, 27 steps/epoch, 4,050 actual steps/seed |
| Precision | Original CUDA bf16 autocast; no fp32 replacement, EMA or SWA |
| Resource | CPU 2, exclusively assigned cuda:0 only after root approval |
| Windowing | 2,048 rows, stride 512, gap/year/station/layer-separated, right zero padding plus valid-mask loss |
| Output | Raw float32 three-seed mean first, type conditioning .25 on noise/offset/drift, high .8 / low .4, snap radius 12, minimum 19 rows, no maximum |

The original model's convolutions are symmetric, and the retained feature
projection includes **bounded offline/two-sided context**, not “165 past-only
features.” `nominal_depth_m`, `depth_regime`, `plateau_full_length` and
`plateau_count` are excluded by the exact original dependency classifier.

`configs/p1.toml`, with environment overrides disabled, is essential: its rolling
windows are 3/6/12/24/48/72/168 hours and depth width is 2.0. The bare FeatureConfig
default has only five rolling windows and is not equivalent. The original feature
builder produces 80 columns; its exact classifier retains 74 numerical columns.
Training-only median/IQR, missing flags, valid/gap, station/layer and current-depth
quantile vocabularies make 165 channels. Actual historical vocabulary inventory is
3 stations, 8 layers and 4 depth tokens. Labels/anomaly types never reach feature
generation; targets are attached afterwards, with `fillna('').astype(str)`.

The old helper `_fit_seed` is deliberately not called: it consults completed state
and requires a holdout. Old `_load_surfaces` also reads test/cache/sample/anchor and
is excluded. Only `_config_for_capacity`, encoder/window/model/loss/schedule,
`_train_epoch`, `predict_encoded`, type conditioning and the original decoder are
reused. All nine original source/config hashes must remain byte-identical.

## Selection ancestry, not a fresh validation claim

`configs/experiments/p1_mstcn_checkpoint_diagnostic_20260827_v2.json` (SHA-256
`437c5c0aa2d2c1508518c48c0a469aa2ae08b9b927a918fc7c7144784c2c0d0c`)
records the Q2 radius-one plateau rule: maximize the minimum epoch−5/epoch/epoch+5
delta F1, with deterministic ties, then the registered all-positive Q2 monthly
gate. The selected width512/e150/.8 neighbors 145/150/155 had Q2 deltas
.07658753494/.07540016180/.07421657248. This is a local, adaptive selection, not a
Public-score inverse and not an absent-selection placeholder.

Its subsequent fixed e150 Q3/Q4 result is explicitly retrospective, not virgin
promotion evidence. Historical router-union pooled F1 was .9029170235 →
.9068037200 (delta +.0038866965), Q3 +.0172087891, Q4 −.0154410616, 90% interval
[−.0131476659, +.0211437692]. That old router is not today's newly trained clean
tree anchor. A same-key fresh-tree-plus-fixed-proposal comparison is a separate
root-owned audit, not a score inherited by this component. No official score is
inverted to choose any coefficient or threshold.

## Historical checkpoint / OOF availability and safe replay

The following are **receipt-declared artifact hashes**, with file existence/size
checked but model/NPZ content not loaded or rehashed by this implementation audit.
Root's separate historical proposal audit must verify hashes before using them.

| Surface | Rows / ordered key SHA | Stored probabilities/proposal SHA |
|---|---|---|
| Q2 original qualification grid | 133,170 / `1df6ae044d7e1c9a3aee473de289fe3e96320e115562221507bdec74ecf6239e` | `867d4b25d968ce4231179181e05eee95cda04d76689c03c1514b907e21dd1f02` |
| Q3 diagnostic v2 | 176,738 / `8e4b225fc912a335383ebdcffd2c5a5304020f80acd5e52d53ff1b893248baec` | `402a8abe977c17d62cc93cf847b37a6fe69ab650a13510d95c0f39dccbfc908f` |
| Q4 diagnostic v2 | 111,124 / `4a91345f68decb5e25d02c721ae53063b0807c3e5b3acdd970093ca36d857f6a` | `ce482f126b099273a468747473c55944d98e9fe9fa25c882dced1bb57909a76e` |

Q2 data/receipt are under `artifacts/p1_incumbent_preserving_mstcn_asrf_v2/`;
Q3/Q4 are under `artifacts/p1_mstcn_checkpoint_diagnostic_20260827_v2/`.
No Q2 epoch150 model state was found in the original artifact root's file
inventory; do not claim checkpoint replay where only the sealed grid exists.
Q2 is the original selection surface, not a three-seed full deployment ensemble.

Q3/Q4 have three e150 states (each 210,338,479 bytes), in seed order above:

- Q3: `4f1e69f753fa2caacd9617e96a2e72b8fa32f4834ac890289b2ecf61239856a5`,
  `53c33403a8f12ac2572cc020133471c9960d8bb392f23d9f6d5d4c53374e574c`,
  `ac953160850139b52228a1f27c93bf66c9e2e44609f21f6bac73a45fefda912b`.
- Q4: `7201479a6fe903dd877f90d2a972224fdc17cf0807bc73035abcfdb04b8aa7da`,
  `149386d5dbb5b9924d735e10d8d63ddc52fa920559861c8562c279429d13262b`,
  `a33eb530fdf458c5a91985b3c83e2ad5ad0767507f5ce01e658ffef789f489c3`.

Separately rehashed encoder JSON: Q3
`2ac20d775b011f97e710370d9dbfa8a1444c8dd80fbee2bf95f04f64bad4a9ba`, Q4
`1e5edfc4e9f581d4205c491a72f397c68cef95fb46aaa54290bbb9459319b24e`.
The blinded NPZ inventory contains arrays, not station/year/layer/time keys.
Keys therefore need the exact receipt-bound historical membership, not a simple
calendar-quarter filter. In particular Q3 extends through 2025-10-01 10:40 UTC.

For a genuine historical model replay: reconstruct raw features with the same
bounded projection, use the saved fold-local encoder (never the full encoder),
apply the exact original membership/order before windowing, load each matching
seed/epoch state, perform the same three-seed raw mean and .8 decoder, and compare
the proposal column only. Do not read/use old candidate/anchor predictions. Q3
training maximum is 2025-06-09 14:50 UTC; Q4 is 2025-09-09 14:50 UTC. Both have
504h10m separation, exceeding the registered 337h combined feature support by
167h10m. Exact train/holdout key hash and encoder fit IDs must match; a new tree
trained with a different cutoff may be compared as a completed policy only with
that difference disclosed, not called an identical model contrast.

## Reproducibility and runtime limits

The original full-deployment runner SHA is
`5f13a0633137c07583c5e113c39f42d13d5c6b007c8a7d612a58309140a288eb`.
Its three full training receipts report 2,191.1374 + 2,192.7703 + 2,200.3211 =
6,584.2288 seconds, or **109.74 minutes** on the recorded GPU. This excludes fresh
raw feature generation and the new independent replay. CPU2 is a changed host
resource setting, not a proved same-time guarantee. Full3 + preparation + replay
share a six-hour clock; a process timer exits at that deadline and preserves a
resource-stop receipt. Failed/interrupted locks are never reused or deleted.

The old Sept5 package's completed operation was certified weight/provenance audit,
not an empty-folder MS-TCN retraining measurement. The observed tree regeneration
mismatch does not prove MS-TCN deterministic failure. Conversely, this module's
manual seed and original bf16 mode do not prove repeated scratch-training bytes
will match. It records actual torch/CUDA/GPU/cuDNN/TF32/determinism settings without
silently changing them. A newly saved training-derived two-window probe is checked
for all row/boundary/type outputs of all three seeds in a distinct PID. This proves
saved-state replay only, not repeated full retraining or new-union F1.

The Python audit-hook/network guard is a code-boundary test, not an OS air-gap
claim. Runtime scratch stays in the new run. Binary weights and training-derived
probe inputs/predictions belong in ignored local artifacts, never Git. This module
does not itself form a final portable model-only submission archive: its current
replay-gated inference still requires its own probe/validation hash inventory.

## Implemented checks / current status

Focused synthetic tests: **20 PASS**, including same raw features after synthetic
label perturbation, 165-channel saved encoder round trip, gap/padding target
alignment, exact source pins and 300-horizon schedule, approval/no-restart gates,
source-only API call inventory, and a separate process using copied source under
the I/O guard. That process runs one tiny width8 CPU synthetic AdamW step followed
by native torch serialization/hash/load and exact prediction comparison; it is
not one of the three candidate fits and consumes no GPU/organizer data.

Actual full candidate fits: **0/3**. Official rows/CSV/upload: **0**. Historical
NPZ/model content reads in this component implementation: **0**. Training remains
pending explicit root GPU allocation and review; the code being ready does not
constitute model or final submission readiness.
