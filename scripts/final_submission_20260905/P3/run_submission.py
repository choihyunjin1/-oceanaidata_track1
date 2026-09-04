"""Standalone exact materializer for the frozen clean P3 final candidate."""

from __future__ import annotations

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

KEYS = ["case_id", "station", "lead_h"]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"case_id": "string", "station": "string"})


def preflight(data_dir: str | Path, package_dir: str | Path = ".") -> dict:
    package = Path(package_dir).resolve()
    contract = load_contract(package)
    official = verify_official_files(data_dir, contract)
    original_path = require_file(
        package / "assets" / "original_component.csv",
        contract["original_component_sha256"],
        "P3 original component",
    )
    axis_path = require_file(
        package / "assets" / "axis_component.csv",
        contract["axis_component_sha256"],
        "P3 axis component",
    )
    index = _read(Path(data_dir) / "test_index.csv")
    original = _read(original_path)
    axis = _read(axis_path)
    if (
        list(original.columns) != contract["expected_columns"]
        or list(axis.columns) != contract["expected_columns"]
    ):
        raise ContractError("P3 component schema drift")
    if not (len(index) == len(original) == len(axis) == contract["expected_rows"]):
        raise ContractError("P3 row-count drift")
    if not index[KEYS].equals(original[KEYS]) or not index[KEYS].equals(axis[KEYS]):
        raise ContractError("P3 official-index/component key order differs")
    return {
        "status": "PREFLIGHT_PASS",
        "candidate_id": contract["candidate_id"],
        "rows": len(original),
        "columns": list(original.columns),
        "key_order_exact": True,
        "official_input_hashes_ok": True,
        "package_atomic": True,
        "official": official,
    }


def materialize(
    data_dir: str | Path,
    package_dir: str | Path = ".",
    output_path: str | Path = "outputs/P3_submission.csv",
) -> dict:
    package = Path(package_dir).resolve()
    contract = load_contract(package)
    preflight(data_dir, package)
    original = _read(package / "assets" / "original_component.csv")
    axis = _read(package / "assets" / "axis_component.csv")
    values_o = pd.to_numeric(original["hs_pred"], errors="raise").to_numpy(float)
    values_a = pd.to_numeric(axis["hs_pred"], errors="raise").to_numpy(float)
    values = values_o.copy()
    active = original["lead_h"].isin(contract["active_leads"]).to_numpy()
    values[active] += float(contract["alpha"]) * (values_a[active] - values_o[active])
    if not np.array_equal(values[~active], values_o[~active]):
        raise ContractError("P3 short-lead no-op drift")
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 30.0:
        raise ContractError("P3 finite/physical guard failed")
    candidate = original.copy()
    candidate["hs_pred"] = values
    target = Path(output_path)
    if not target.is_absolute():
        target = package / target
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    actual = sha256_file(target)
    if actual != contract["candidate_sha256"]:
        raise ContractError(f"P3 final SHA drift: {actual}")
    receipt = {
        "status": "READY_EXACT_NOT_UPLOADED",
        "candidate_id": contract["candidate_id"],
        "rows": len(candidate),
        "columns": list(candidate.columns),
        "changed_rows": int(active.sum()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "key_order_exact": True,
        "sha256": actual,
        "candidate_hash_exact": True,
        "package_atomic": True,
        "lineage": "organizer_distributed_data_only_scratch_models",
        "caveat": "Original component is frozen because its historical saved-weight replay differs by at most 0.0048767 m; the axis component replays exactly from bundled weights.",
    }
    write_json(package / "outputs" / "receipt.json", receipt)
    return receipt
