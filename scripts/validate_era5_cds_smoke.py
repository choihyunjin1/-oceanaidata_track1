"""Validate an existing ERA5 CDS smoke response without credentials or network access."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from p2_restore.era5_cds import validate_cds_smoke
from p2_restore.era5_manifest import (
    build_cds_smoke_validation_receipt,
    sha256,
    write_receipt,
)
from p2_restore.era5_request import build_smoke_chunk

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT / "external_data/quarantine/era5_cds_sors_smoke/smoke_sors_3x3_20240901_24h.nc"
)
DEFAULT_RECEIPT = (
    REPO_ROOT / "artifacts/p2_era5_primary_scaffold_v1/cds_smoke_validation_receipt.json"
)


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "netCDF4"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _code_hashes() -> dict[str, str]:
    paths = {
        "validator": REPO_ROOT / "src/p2_restore/era5_cds.py",
        "field_preflight": REPO_ROOT / "src/p2_restore/era5_preflight.py",
        "request_builder": REPO_ROOT / "src/p2_restore/era5_request.py",
        "manifest": REPO_ROOT / "src/p2_restore/era5_manifest.py",
        "runner": Path(__file__).resolve(),
    }
    return {name: sha256(path) for name, path in paths.items()}


def run(input_path: Path, receipt_path: Path) -> dict[str, object]:
    report = validate_cds_smoke(input_path, expected_chunk=build_smoke_chunk())
    receipt = build_cds_smoke_validation_receipt(
        report.public_dict(),
        dependency_versions=_versions(),
        code_sha256=_code_hashes(),
    )
    write_receipt(receipt_path, receipt)
    return {
        "status": "complete",
        "network_action_taken": False,
        "raw_file_modified": False,
        "receipt": str(receipt_path),
        "validation": report.public_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(run(args.input, args.receipt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
