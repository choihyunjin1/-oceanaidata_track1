# P1 MS-TCN Sobol HPO — execution report

Status: `NO_GO_PRECONFIRM`

The one-shot RTX 5090 run completed cleanly at source commit
`8e0dc9ab22a713596425375985a6fe04f878e325` after 26,183.483 seconds
(7.273 hours). It executed all 32 sealed Sobol discovery fits and exactly four
predeclared additional-seed fits for the top two recipes. The fixed
preconfirmation gate stopped the experiment before Q3/Q4, so there were 36
fits in total and no confirmatory fits.

## Selected Q2 recipe

- Trial: `trial_18`; checkpoint rule: final epoch 150 (no checkpoint file persisted)
- Seeds: `20260827`, `20260839`, `20260863`; threshold: `0.8`
- Width/batch: `512` / `64`; dropout: `0.273120475653559`
- Learning rate / weight decay: `0.000123695720217236` / `0.000114100507635706`
- Soft Dice / temporal smoothing / tied boundary-type weights:
  `1.29710821748465` / `0.183114585420117` / `0.394524332042783`
- Stage weights: `[1, 1, 1, 1]`

The Q2 three-seed control F1 was `0.867675735908627`; the selected recipe F1
was `0.868241372947038`, for pooled delta `+0.000565637038412`. Monthly deltas
were `+0.001005635359418` (April), `+0.000294857304021` (May), and
`+0.001336782663924` (June), with zero frozen-anchor positive removals.

The recipe passed strict positive monthly deltas and the zero-removal check,
but failed the predeclared pooled-delta gate of `>= +0.003`. The decision was
therefore `STOP_BEFORE_CONFIRMATION`; `q3_q4_training_started` is false. The
selection used Q2 historical data only. No result-based grid change, rerun, or
restart was authorized.

## Seals and independent QA

- Config SHA-256: `c9ff0cee7add360b37b303a283732107246df2f079046f9f814e37026814316b`
- Runner SHA-256: `155e7719472d55e3c0f3f291264726378e749fb68a9d1d3816f085ae86817401`
- Sealed design SHA-256: `98a52582b883460888d736cfe75327db6dd55ada395d796b51c5034b56e723ed`
- Discovery blind NPZ SHA-256: `3a08659c9a57ecab609c4a76d0a1bdd649d8ce5d78a5cc795e76ea8b39fd0ced`
- Top-two three-seed blind NPZ SHA-256: `6e342a5dbc850cc0174d061e3dbccaea5d4e186a3cfa01a9d119711663d775b6`
- Preconfirm gate SHA-256: `5dfb9d76b6ab05f1689cd054dd04c7b85803798c39f5b0b5ea1c23e6c84f73fe`
- Aggregate SHA-256: `fb42fe28abd4590ac4c5affeacef4a497e792f647d54f88f93f91f743d4fcf98`
- Independent QA SHA-256: `437ee74d04bd468eac34305827ec99c1fd6fdd27b696a5b44a718052f7a8cc46`

Independent artifact-only verification passed all 28 checks: identity and
hash pins, 32-point pre-fit seal, both blind NPZ receipt SHA/byte matches,
exact fit counts, all 36 epoch/non-finite/checkpoint receipts,
terminal/aggregate identity, gate consistency, absent confirmatory artifacts,
and forbidden-output guards. PyTorch used two intra-op CPU threads and one
inter-op thread.

Official test/sample/submission rows read: `0`. CSV created: `false`. Upload
performed: `false`. Persisted checkpoint files: `0`. The artifact namespace
contains JSON receipts/histories and two blind NPZ files only.
