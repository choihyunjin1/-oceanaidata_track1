"""Standalone exact materializer for the frozen clean P2 final candidate."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    ContractError,
    load_contract,
    require_file,
    sha256_file,
    verify_official_files,
    write_json,
)

KEYS = ["station", "layer", "time"]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"station": "string", "time": "string"})


def preflight(data_dir: str | Path, package_dir: str | Path = ".") -> dict:
    package = Path(package_dir).resolve()
    contract = load_contract(package)
    official = verify_official_files(data_dir, contract)
    candidate_path = require_file(
        package / "assets" / "frozen_v52_candidate.csv",
        contract["candidate_sha256"],
        "P2 frozen v52 candidate",
    )
    anchor_path = require_file(
        package / "assets" / "bin17_anchor.csv",
        contract["anchor_sha256"],
        "P2 bin17 anchor",
    )
    index = _read(Path(data_dir) / "test_index.csv")
    sample_keys = _read(Path(data_dir) / "sample_submission.csv")[KEYS]
    candidate = _read(candidate_path)
    anchor = _read(anchor_path)
    if list(candidate.columns) != contract["expected_columns"]:
        raise ContractError("P2 candidate schema drift")
    if not (len(index) == len(candidate) == len(anchor) == contract["expected_rows"]):
        raise ContractError("P2 row-count drift")
    if not index[KEYS].equals(candidate[KEYS]) or not index[KEYS].equals(anchor[KEYS]):
        raise ContractError("P2 official-index/candidate/anchor key order differs")
    if not index[KEYS].equals(sample_keys):
        raise ContractError("P2 sample key order differs")
    values = pd.to_numeric(candidate["temp"], errors="coerce").to_numpy(float)
    anchor_values = pd.to_numeric(anchor["temp"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(anchor_values).all():
        raise ContractError("P2 candidate or anchor contains non-finite values")
    if float(np.max(np.abs(values - anchor_values))) > 0.500000000001:
        raise ContractError("P2 frozen 0.5 C action cap drift")
    return {
        "status": "PREFLIGHT_PASS",
        "candidate_id": contract["candidate_id"],
        "rows": len(candidate),
        "columns": list(candidate.columns),
        "key_order_exact": True,
        "official_input_hashes_ok": True,
        "package_atomic": True,
        "official": official,
    }


def materialize(
    data_dir: str | Path,
    package_dir: str | Path = ".",
    output_path: str | Path = "outputs/P2_submission.csv",
) -> dict:
    package = Path(package_dir).resolve()
    contract = load_contract(package)
    preflight(data_dir, package)
    source = package / "assets" / "frozen_v52_candidate.csv"
    target = Path(output_path)
    if not target.is_absolute():
        target = package / target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    candidate = _read(target)
    values = pd.to_numeric(candidate["temp"], errors="raise").to_numpy(float)
    actual = sha256_file(target)
    if actual != contract["candidate_sha256"]:
        raise ContractError(f"P2 final SHA drift: {actual}")
    receipt = {
        "status": "READY_EXACT_NOT_UPLOADED",
        "candidate_id": contract["candidate_id"],
        "rows": len(candidate),
        "columns": list(candidate.columns),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "key_order_exact": True,
        "sha256": actual,
        "candidate_hash_exact": True,
        "package_atomic": True,
        "lineage": "organizer_distributed_data_only_scratch_models",
        "caveat": "Exact mode freezes the deployed scratch-ensemble output; full 3-fit retraining source is bundled for audit.",
    }
    write_json(package / "outputs" / "receipt.json", receipt)
    return receipt
