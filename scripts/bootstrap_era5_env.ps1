[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$EnvironmentDirectory = Join-Path $ProjectRoot ".venv-era5"
$Era5Python = Join-Path $EnvironmentDirectory "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements-era5.txt"
$Runner = Join-Path $PSScriptRoot "run_p2_era5_primary_scaffold.py"
$SourceDirectory = Join-Path $ProjectRoot "src"

function Assert-NativeSuccess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Step
    )

    if ($LASTEXITCODE -ne 0) {
        throw "ERA5 bootstrap failed during: $Step"
    }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' is required. Install Python 3.12 first."
}

if (-not (Test-Path -LiteralPath $Era5Python -PathType Leaf)) {
    & py -3.12 -m venv $EnvironmentDirectory
    Assert-NativeSuccess -Step "isolated Python 3.12 environment creation"
}

& $Era5Python -c "import sys; assert sys.version_info[:2] == (3, 12)"
Assert-NativeSuccess -Step "Python version check"

& $Era5Python -m pip install --upgrade pip==26.2.1
Assert-NativeSuccess -Step "pip installation"

& $Era5Python -m pip install --requirement $Requirements
Assert-NativeSuccess -Step "ERA5 dependency installation"

& $Era5Python -m pip check
Assert-NativeSuccess -Step "dependency consistency check"

& $Era5Python -c "import cdsapi, netCDF4, pyarrow, xarray"
Assert-NativeSuccess -Step "ERA5 import check"

$PreviousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
try {
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $SourceDirectory, "Process")
    & $Era5Python $Runner --mode preflight
    Assert-NativeSuccess -Step "credential-safe ERA5 preflight"
}
finally {
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $PreviousPythonPath, "Process")
}

Write-Host "ERA5 isolated runtime is ready at .venv-era5. No ERA5 data was downloaded."
