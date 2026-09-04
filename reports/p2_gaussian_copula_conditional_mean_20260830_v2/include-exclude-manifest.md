# Include / exclude manifest

## Included in Git

- sealed v1 full config and v2 completion-only overlay
- empirical-margin/Kendall-copula implementation
- v1 runner, v2 support-contract repair runner, independent QA runner
- focused contract tests
- v1 technical-failure report
- v2 conclusion-first report and small independent QA JSON
- negative-evidence registry update

## Excluded from Git

- `observations.csv` and every raw/source data file
- immutable alpha50 proxy parquet and other artifact parquet
- v1/v2 attempt locks
- prediction NPZ files and scored prediction parquet
- prediction commitment and result JSON under ignored `artifacts/`
- official test/sample/submission inputs, submission CSV, uploads, credentials, caches, logs

The report preserves small aggregate metrics and hashes; it does not preserve row-level truth or predictions.
