# P3 perfect-future-wind oracle — one-shot record

## Decision

`CLOSE_PREDICTED_FUTURE_WIND_AND_MOS_FAMILY`.

The historical-only oracle made the six-lead pooled RMSE worse by `+0.001339855m`
(`0.782538046m -> 0.783877901m`). Its paired whole-case bootstrap CI90 was
`[-0.001793608, +0.004342590]m`. All seven preregistered oracle checks failed, so the
pooled multi-output future-wind forecast and frozen-KMA MOS were not executed.

## Frozen contract and provenance

- Repository base: `de1392076f15e3d08b6ab361760950eba880ddad`.
- Scientific contract: `reports/dual_engine_breakthrough_20260829_v1/report-source.md`,
  P3 lines 74–83 at the base commit.
- Membership: exact intersection of frozen KMA OOF, sealed champion OOF, and frozen
  validation keys: `179` historical cases and `1,074` rows.
- Case counts: folds `48/78/53`; stations `G-ORS=65`, `I-ORS=46`, `S-ORS=68`.
- Frozen KMA prediction: the already-sealed `candidate_final` column. Control adds the
  fixed low-dimensional current/past wave state. Treatment adds the six actual future
  local wind-vector deltas.
- Active leads: `18h, 24h`; `3/6/9/12h` are byte-exact no-op predictions in both arms.
- Folds and `78h` embargo are fixed. The control-only inner leave-one-fold-out loss chose
  ridge alpha `1000` in every outer fold; the same fold alpha was used by both arms.
- Ridge candidates, features, missing-data rule, bootstrap seeds, and every gate were
  registered in the config before the attempt lock was consumed. No result-based grid
  or rerun occurred.

The historical atmosphere record has long missing intervals. A future vector was usable
only when both the past-only current vector and that future pair were finite. This covered
`123` complete six-lead cases; usable rows by lead were `124/124/124/124/123/124` for
`3/6/9/12/18/24h`. The preregistered fallback was exact zero delta (wind persistence),
with no interpolation and no missingness indicator. All 179 cases remained in the fixed
surface. This coverage limitation weakens any claim about fully observed local wind, but
it cannot rescue this exact oracle from its explicit gate failure.

## Aggregate result

| Slice | Control RMSE (m) | Treatment RMSE (m) | Delta (m) |
|---|---:|---:|---:|
| pooled six-lead | 0.782538046 | 0.783877901 | +0.001339855 |
| fold 2024_h2_storm | 0.712535596 | 0.717175100 | +0.004639504 |
| fold winter_transition | 0.794908941 | 0.794478357 | -0.000430584 |
| fold 2025_h1 | 0.823669708 | 0.824936292 | +0.001266584 |
| station G-ORS | 0.736259391 | 0.739611633 | +0.003352241 |
| station I-ORS | 0.876310319 | 0.877711396 | +0.001401077 |
| station S-ORS | 0.757842375 | 0.757270180 | -0.000572196 |
| lead 18h | 0.898082298 | 0.903761769 | +0.005679471 |
| lead 24h | 0.846830031 | 0.848221788 | +0.001391757 |

The four short leads each have exact zero delta. Only `1/3` folds and `1/3` stations
improved. The worst station-by-lead delta was `+0.011200734m`.

| Oracle gate | Required | Observed | Pass |
|---|---:|---:|:---:|
| pooled six-lead delta | `<= -0.006m` | `+0.001339855m` | no |
| paired case CI90 upper | `< 0` | `+0.004342590m` | no |
| improved folds | `>= 2/3` | `1/3` | no |
| improved stations | `>= 2/3` | `1/3` | no |
| lead 18h | non-degrade | `+0.005679471m` | no |
| lead 24h | non-degrade | `+0.001391757m` | no |
| worst station-by-lead | `<= +0.003m` | `+0.011200734m` | no |

## Seals and independent QA

- Attempt lock SHA-256: `4f6906a84588b3295b4a005828be4d790e78d27a472bfd494dc1abe89c37be01`.
- Oracle prediction SHA-256: `391974502507ce2ca267add413895e190ad5b26a593c51d64215ca5fc8b34d52`.
- Oracle seal SHA-256: `e5aa6e5f40a5b3812cc23e16d58483e3aa8c40094f6af1caaf160b9ae39e434b`.
- Prediction parquet contains keys plus two predictions only; no Hs target column. It was
  written with exclusive-create semantics, hashed, sealed, and reloaded before the
  designated metric was computed.
- Independent QA passed `19/19` checks. It independently reattached historical truth,
  replayed all aggregate/fold/station/lead/station-by-lead metrics, reran the fixed
  5,000-draw whole-case bootstrap, reproduced all seven failures, verified the short-lead
  frozen-KMA no-op, and confirmed that no conditional artifact exists.

## Data boundary and transport

Only the historical `train_atmos.csv` and the five hash-pinned historical parquet
artifacts in the config were opened. Official test context/index, sample, baseline,
submission, scoring values, and absolute official times were not read or inferred. There
was no external-period matching, source mutation, CSV generation, model persistence,
upload, commit, or push. Raw blind prediction rows remain only under the ignored
`artifacts/` path.

Local-to-official transport is therefore closed for this family: even perfect observed
future local wind with the fixed persistence fallback did not improve the historical OOF
surface. A deployable noisy wind forecast cannot satisfy the preregistered prerequisite,
so no official paired A/B is authorized. The only material gap is the long historical
atmosphere missingness noted above; addressing it would define a different information
contract and is not a rerun of this closed family.

## Commands actually used

```powershell
$env:P3_DATA_DIR='<immutable P3 historical source root>'
.\.venv-p1\Scripts\python.exe scripts\run_p3_perfect_future_wind_oracle_20260829_v1.py --preflight
.\.venv-p1\Scripts\ruff.exe check scripts\run_p3_perfect_future_wind_oracle_20260829_v1.py tests\test_p3_perfect_future_wind_oracle_20260829_v1.py
.\.venv-p1\Scripts\python.exe -m pytest tests\test_p3_perfect_future_wind_oracle_20260829_v1.py -q
.\.venv-p1\Scripts\python.exe scripts\run_p3_perfect_future_wind_oracle_20260829_v1.py --execute
.\.venv-p1\Scripts\python.exe scripts\qa_p3_perfect_future_wind_oracle_20260829_v1.py
```

The `--execute` command is recorded for provenance and must not be run again in this
workspace: the exclusive attempt lock has been consumed.
