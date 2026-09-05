$ErrorActionPreference = 'Stop'
python -I -B (Join-Path $PSScriptRoot '02_code/run.py') RUN_TRAINING
if ($LASTEXITCODE -ne 0) { throw 'P2 training failed; preserve this package and receipts.' }
