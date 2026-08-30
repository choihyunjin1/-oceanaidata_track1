# Include / exclude manifest

## Included in this cycle

- deterministic generator: `scripts/build_historical_model_reaudit_20260831_v1.py`
- focused tests: `tests/test_build_historical_model_reaudit_20260831_v1.py`
- exhaustive 48-family ledger and four-grain cross-check
- 14 generated model-family reuse cards
- status taxonomy, technical report, claim-source ledger, gap matrix
- independent QA and validated/rendered analytical artifact

## Explicitly excluded

- raw training or ERA5 data
- official test/sample/submission CSV or values
- hidden labels and Private results
- predictions, checkpoints, parquet/npz outputs, caches, logs
- credentials, tokens, `.env` files
- attempt locks
- pre-existing untracked P1 station-ablation config/script/test and six attempt locks

The excluded pre-existing files were neither modified nor staged.
