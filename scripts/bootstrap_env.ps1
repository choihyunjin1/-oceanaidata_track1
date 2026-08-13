$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-p1\Scripts\python.exe"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' is required. Install Python.Python.3.12 version 3.12.10 first."
}

if (-not (Test-Path -LiteralPath $Python)) {
    py -3.12 -m venv (Join-Path $ProjectRoot ".venv-p1")
}

& $Python -m pip install --upgrade pip==26.2.1 setuptools==84.0.0 wheel==0.48.0
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements-dl.txt")
& $Python -m pip install --no-deps -e $ProjectRoot
& $Python -m pip check
& $Python (Join-Path $PSScriptRoot "smoke_cuda.py")
