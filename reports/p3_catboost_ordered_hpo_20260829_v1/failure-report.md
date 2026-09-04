# P3 CatBoost ordered HPO v1 — terminal failure report

## Terminal decision

- Status: **INVALID / TERMINAL TECHNICAL FAILURE**.
- The one-shot attempt is consumed and must not be rerun under experiment ID
  `p3_catboost_ordered_hpo_20260829_v1`.
- No scientific or gate conclusion can be drawn. No rung completed, no ranking was sealed,
  and no selection or confirmation metric was produced.
- Root cause: the preregistered 48-point Cartesian grid contained 12 CatBoost-incompatible
  `boosting_type=Ordered` plus `grow_policy=Depthwise` candidates. Parameter construction in
  static preflight did not exercise CatBoost's fit-time compatibility validation.

## Execution accounting

| Item | Sealed finding |
|---|---|
| Attempt lock | Present; `rerun_forbidden=true` |
| First control fit | 18.0547 s at rung 300 / first selection fold |
| Last emitted checkpoint | 70/98 rung-300 fits at 6,878.5 s |
| Successful fits before exception | **74** |
| Failed attempt | **75th**, `challenger_37`, first selection fold |
| Result / aggregate | Not created |
| Selection gate | Not evaluated |
| Confirmation fits / gate | 0 / not evaluated |

The 74 successful-fit count is deterministically reconstructed from the frozen candidate-major,
fold-minor loop: two control fits plus two fits for each of `challenger_01` through
`challenger_36`. CatBoost rejected `challenger_37` while preparing the first `.fit`; therefore
the failed attempt did not complete a model fit and did not increment the runner's completed-fit
counter.

Last successful combination (`challenger_36`, second selection fold):

```text
boosting_type=Ordered, grow_policy=SymmetricTree, depth=9,
bootstrap_type=MVS, subsample=0.8,
learning_rate=0.04, l2_leaf_reg=8.0, random_strength=0.5, rsm=1.0,
thread_count=6
```

Failed combination (`challenger_37`, first selection fold):

```text
boosting_type=Ordered, grow_policy=Depthwise, depth=5,
bootstrap_type=Bayesian, bagging_temperature=0.2,
learning_rate=0.02, l2_leaf_reg=20.0, random_strength=0.1, rsm=0.75,
thread_count=6
```

Terminal exception, verbatim:

```text
_catboost.CatBoostError: catboost/private/libs/options/catboost_options.cpp:759: Ordered boosting is not supported for nonsymmetric trees.
```

## Boundary audit

- Official `test_context.parquet`, `test_index.csv`, and `sample_submission.csv`: **0 rows read**.
  Evidence is the sealed preflight receipt (`official_rows_read=0`) plus control-flow inspection:
  execution failed inside historical rung-300 selection before any confirmation path, and the
  three official basenames are explicitly forbidden by the runner. No OS-level file-access audit
  was available, so this is a code-path and receipt proof rather than an operating-system audit.
- Submission/upload attempts: **0**. The runner never reached a result or deployment path.
- CSV files written in the dedicated artifact tree: **0**.
- Dedicated artifact files: exactly two — `static_preflight.json` and `one_shot/attempt.lock`.
  No raw rows, prediction file, model, result JSON, or aggregate JSON exists.
- Source data was not modified. Config, grid, resource amendment, lock, and the two execution
  artifacts were not changed during post-mortem work.

## Seals and hashes

Execution-time seals preserved in `static_preflight.json`:

| Object | SHA-256 |
|---|---|
| Grid | `98710828d73af2ecb44f0db64cd369ff4504e4eca27587b793e85c403150571e` |
| Resource amendment | `70742c1a1ded515ace51aedc495bd5f959e4b6063918c9b669288834c5f9b079` |
| Authorized config | `d162ece22da5c8461c4a1352c8249b9210ebec995cb54e7cbcc4085d0fdf0adc` |
| Executed runner | `b275f0a2679b22e572de8b51c17a4335d3998a784b76ae9e5d87f7193208e6f0` |
| Executed tests | `9acb9de0647dc37a40ac9e09261baa7d3dfcebb7333364f66bf6ab099e855c21` |
| Contract module | `4e726bbfd4df6fe328c9198e5995061227bd202683c52d7b950362b19b29e249` |

Attempt-lock file SHA-256:
`51327c0d724f679219b316c39c4749e450e031f8eb2c3507252c608b41ba9ec7`.
Its payload seals the same config/grid hashes and `rerun_forbidden=true`.

## Post-mortem fail-fast safeguard

After terminal failure and without changing the frozen execution config/grid/lock/artifacts, the
runner's static grid check was patched to reject any `Ordered` candidate whose grow policy is not
`SymmetricTree` before fit. A regression test now requires the frozen grid to stop specifically at
`challenger_37`; the wrong-token test also verifies that a pre-existing attempt lock remains
byte-identical.

Post-mortem implementation hashes (these were **not** the executed hashes):

| Object | SHA-256 |
|---|---|
| Patched runner | `f7b8e2c362f778225ebf8eb1d4cc705a2e8b3701dac2a07fec27b1040c483802` |
| Patched tests | `03cddb646d4614e8ce8a8e643668c3b6398acdbca243f6d0e6edeaa15f56a173` |

QA:

```powershell
.\.venv-p1\Scripts\python.exe -m ruff check scripts/run_p3_catboost_ordered_hpo_20260829_v1.py tests/test_p3_catboost_ordered_hpo_20260829_v1.py
.\.venv-p1\Scripts\python.exe -m pytest -q tests/test_p3_catboost_ordered_hpo_20260829_v1.py
```

Observed: Ruff passed; dedicated pytest passed 7/7. Neither QA command performs a model fit.
