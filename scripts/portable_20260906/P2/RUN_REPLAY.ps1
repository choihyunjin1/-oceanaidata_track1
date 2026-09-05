$ErrorActionPreference = 'Stop'
python -I -B (Join-Path $PSScriptRoot '02_code/run.py') REPLAY
if ($LASTEXITCODE -ne 0) { throw 'P2 replay failed; preserve this package and receipts.' }
