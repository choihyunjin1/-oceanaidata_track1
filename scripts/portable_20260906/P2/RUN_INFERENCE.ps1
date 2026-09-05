$ErrorActionPreference = 'Stop'
python -I -B (Join-Path $PSScriptRoot '02_code/run.py') RUN_INFERENCE
if ($LASTEXITCODE -ne 0) { throw 'P2 inference failed; preserve this package and receipts.' }
