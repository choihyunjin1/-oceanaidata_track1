from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
REQUIREMENTS = REPO_ROOT / "requirements-era5.txt"
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_era5_env.ps1"


def _requirement_lines() -> set[str]:
    return {
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_era5_runtime_has_only_the_pinned_minimum_stack() -> None:
    assert _requirement_lines() == {
        "cdsapi==0.7.7",
        "numpy==2.3.5",
        "pandas==3.0.1",
        "pyarrow==25.0.1",
        "xarray==2026.7.0",
        "netCDF4==1.7.4",
    }


def test_bootstrap_is_isolated_path_safe_and_download_free() -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")
    normalized = script.replace("/", "\\")

    assert '.venv-era5"' in normalized
    assert ".venv-p1" not in normalized
    assert "requirements-era5.txt" in script
    assert "$PSScriptRoot" in script
    assert "[System.IO.Path]::GetFullPath" in script
    assert "Test-Path -LiteralPath" in script
    assert re.search(r"[A-Za-z]:\\Users\\", script) is None

    assert "run_p2_era5_primary_scaffold.py" in script
    assert "--mode preflight" in script
    assert "--execute-download" not in script
    assert "--execute-anonymous-smoke" not in script


def test_bootstrap_never_reads_or_prints_the_credential() -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")

    assert "CDSAPI_KEY" not in script
    assert "Get-ChildItem Env:" not in script
    assert "Start-Transcript" not in script
    assert "Set-PSDebug" not in script
    assert 'GetEnvironmentVariable("PYTHONPATH", "Process")' in script
    assert 'SetEnvironmentVariable("PYTHONPATH", $PreviousPythonPath, "Process")' in script


def test_era5_environment_is_git_ignored() -> None:
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/.venv-era5/" in ignore


def test_bootstrap_has_valid_powershell_syntax() -> None:
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell parser is unavailable")
    command = (
        "$tokens = $null; $errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "$env:ERA5_BOOTSTRAP_PATH, [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )
    environment = os.environ.copy()
    environment["ERA5_BOOTSTRAP_PATH"] = str(BOOTSTRAP)
    completed = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
