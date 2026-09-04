"""Standalone exact materializer for the frozen clean P1 final candidate."""

from __future__ import annotations

import json
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

KEYS = ["station", "year", "layer", "time"]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"station": "string", "time": "string"})


def preflight(data_dir: str | Path, package_dir: str | Path = ".") -> dict:
    package = Path(package_dir).resolve()
    contract = load_contract(package)
    official = verify_official_files(data_dir, contract)
    anchor_path = require_file(
        package / "assets" / "e150_anchor.csv",
        contract["anchor_sha256"],
        "P1 e150 anchor",
    )
    patch_path = require_file(
        package / "assets" / "gi_spike2_patch.json",
        contract["patch_sha256"],
        "P1 GI spike2 patch",
    )
    test = _read(Path(data_dir) / "test.csv")
    anchor = _read(anchor_path)
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    if list(anchor.columns) != contract["expected_columns"]:
        raise ContractError("P1 anchor schema drift")
    if len(anchor) != contract["expected_rows"] or len(test) != contract["expected_rows"]:
        raise ContractError("P1 row-count drift")
    if not test[KEYS].equals(anchor[KEYS]):
        raise ContractError("P1 official-test and anchor key order differ")
    if len(patch["rows"]) != 2:
        raise ContractError("P1 patch must contain exactly two rows")
    return {
        "status": "PREFLIGHT_PASS",
        "candidate_id": contract["candidate_id"],
        "rows": len(anchor),
        "columns": list(anchor.columns),
        "key_order_exact": True,
        "official_input_hashes_ok": True,
        "package_atomic": True,
        "official": official,
    }


def materialize(
    data_dir: str | Path,
    package_dir: str | Path = ".",
    output_path: str | Path = "outputs/P1_submission.csv",
) -> dict:
    package = Path(package_dir).resolve()
    contract = load_contract(package)
    preflight(data_dir, package)
    anchor = _read(package / "assets" / "e150_anchor.csv")
    patch = json.loads((package / "assets" / "gi_spike2_patch.json").read_text(encoding="utf-8"))
    candidate = anchor.copy()
    index = pd.MultiIndex.from_frame(candidate[KEYS])
    if not index.is_unique:
        raise ContractError("P1 anchor keys are not unique")
    locations: list[int] = []
    for row in patch["rows"]:
        key = tuple(row[name] for name in KEYS)
        location = index.get_indexer([key])[0]
        if location < 0:
            raise ContractError(f"P1 patch key missing: {key}")
        locations.append(int(location))
    if len(set(locations)) != 2 or not candidate.loc[locations, "label"].eq(0).all():
        raise ContractError("P1 patch is not a two-row add-only change")
    candidate.loc[locations, "label"] = 1
    labels = pd.to_numeric(candidate["label"], errors="raise").to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ContractError("P1 labels must be binary")
    target = Path(output_path)
    if not target.is_absolute():
        target = package / target
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    actual = sha256_file(target)
    if actual != contract["candidate_sha256"]:
        raise ContractError(f"P1 final SHA drift: {actual}")
    receipt = {
        "status": "READY_EXACT_NOT_UPLOADED",
        "candidate_id": contract["candidate_id"],
        "rows": len(candidate),
        "columns": list(candidate.columns),
        "positive_rows": int(labels.sum()),
        "changed_rows": 2,
        "key_order_exact": True,
        "sha256": actual,
        "candidate_hash_exact": True,
        "package_atomic": True,
        "lineage": "organizer_distributed_data_only_scratch_models",
    }
    if receipt["positive_rows"] != contract["expected_positive_rows"]:
        raise ContractError("P1 positive-row count drift")
    write_json(package / "outputs" / "receipt.json", receipt)
    return receipt
