"""Independent read-only QA for the local official-final package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "artifacts" / "official_final_submission_20260905"
MAX_UPLOAD = 50_000_000
FORBIDDEN_DATA_BASENAMES = {
    "train.csv",
    "test.csv",
    "observations.csv",
    "test_index.csv",
    "sample_submission.csv",
    "baseline_interp.csv",
    "baseline_persistence.csv",
    "test_context.parquet",
    "train_wave.csv",
    "train_atmos.csv",
}
SECRET_PATTERNS = {
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_notebook(path: Path) -> dict[str, Any]:
    notebook = nbformat.read(path, as_version=4)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    execution_counts = [
        cell.get("execution_count") for cell in notebook.cells if cell.cell_type == "code"
    ]
    return {
        "path": path.name,
        "code_cells": len(execution_counts),
        "all_code_cells_executed": all(value is not None for value in execution_counts),
        "error_outputs": len(errors),
    }


def stream_parts_hash(package: Path) -> list[dict[str, Any]]:
    manifest = json.loads(
        (package / "P1" / "model_parts" / "MANIFEST.json").read_text(encoding="utf-8")
    )
    results: list[dict[str, Any]] = []
    for model in manifest["models"]:
        digest = hashlib.sha256()
        total = 0
        for part in model["parts"]:
            path = package / "P1" / "model_parts" / part["filename"]
            if path.stat().st_size != part["bytes"] or sha256_file(path) != part["sha256"]:
                raise RuntimeError(f"P1 model part drift: {path.name}")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
                    total += len(block)
        results.append(
            {
                "filename": model["filename"],
                "bytes": total,
                "sha256": digest.hexdigest(),
                "exact": total == model["bytes"] and digest.hexdigest() == model["sha256"],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    package = args.package_root.resolve()
    master = json.loads((package / "MASTER_MANIFEST.json").read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    checks["master_status"] = master["status"] == "LOCAL_READY_EXACT_NOT_UPLOADED"
    checks["atomic_problem_directories"] = master["atomic_problem_directories"] is True
    checks["notebooks_executed"] = master["notebooks_executed"] is True
    checks["clean_policy"] = all(
        master["policy"][name] == 0
        for name in (
            "external_observation_rows",
            "external_reanalysis_rows",
            "external_forecast_rows",
            "pretrained_weight_files_loaded",
            "hidden_truth_rows_read",
        )
    )
    receipt_results: dict[str, Any] = {}
    for problem in ("P1", "P2", "P3"):
        contract = json.loads((package / problem / "contract.json").read_text(encoding="utf-8"))
        receipt = json.loads(
            (package / problem / "outputs" / "receipt.json").read_text(encoding="utf-8")
        )
        output = package / problem / "outputs" / f"{problem}_submission.csv"
        receipt_results[problem] = {
            "status": receipt["status"],
            "rows": receipt["rows"],
            "sha256": sha256_file(output),
            "contract_sha_exact": sha256_file(output)
            == contract["candidate_sha256"]
            == receipt["sha256"],
            "executed_notebook": check_notebook(
                package / problem / f"{problem}_final_submission.executed.ipynb"
            ),
        }
    checks["problem_receipts"] = receipt_results

    forbidden = [
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.is_file() and path.name.lower() in FORBIDDEN_DATA_BASENAMES
    ]
    checks["forbidden_source_data_files"] = forbidden
    checks["forbidden_source_data_absent"] = not forbidden

    secret_hits: list[dict[str, str]] = []
    for path in package.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py",
            ".md",
            ".json",
            ".txt",
            ".ps1",
            ".ipynb",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                secret_hits.append({"file": str(path.relative_to(package)), "pattern": name})
    checks["secret_hits"] = secret_hits
    checks["secret_scan_pass"] = not secret_hits

    upload_results: list[dict[str, Any]] = []
    for row in master["upload_files"]:
        path = package / "upload" / row["filename"]
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
        upload_results.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "under_50mb": path.stat().st_size <= MAX_UPLOAD,
                "sha256_exact": sha256_file(path) == row["sha256"],
                "zip_integrity": bad is None,
            }
        )
    checks["upload_files"] = upload_results
    checks["p1_model_reassembly"] = stream_parts_hash(package)
    pass_value = (
        all(value for key, value in checks.items() if isinstance(value, bool))
        and all(
            item["contract_sha_exact"]
            and item["executed_notebook"]["all_code_cells_executed"]
            and item["executed_notebook"]["error_outputs"] == 0
            for item in receipt_results.values()
        )
        and all(
            item["under_50mb"] and item["sha256_exact"] and item["zip_integrity"]
            for item in upload_results
        )
        and all(item["exact"] for item in checks["p1_model_reassembly"])
    )
    result = {
        "schema_version": "ocean.official_final_submission.qa.20260905.v1",
        "status": "PASS" if pass_value else "FAIL",
        "package_root": str(package),
        "checks": checks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if not pass_value:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
