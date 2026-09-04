"""Fail-closed P3 runner: two verified CatBoost/router chains -> CSV."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
from common import (
    ContractError,
    load_contract,
    verify_official_files,
    verify_package_files,
    write_json,
)

KEYS = ["case_id", "station", "lead_h"]
EXPECTED_LEADS = {3, 6, 9, 12, 18, 24}


def _load_predictor(package: Path):
    path = package / "04_predict" / "predict_submission.py"
    spec = importlib.util.spec_from_file_location("p3_final_predictor", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load P3 predictor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(data_dir: str | Path, package_dir: str | Path = ".") -> dict:
    package = Path(package_dir).resolve()
    data = Path(data_dir).resolve()
    contract = load_contract(package)
    official = verify_official_files(data, contract)
    models = verify_package_files(package, contract, "model_files")
    index = pd.read_csv(
        data / "test_index.csv",
        dtype={"case_id": "string", "station": "string"},
    )
    sample = pd.read_csv(
        data / "sample_submission.csv",
        usecols=KEYS,
        dtype={"case_id": "string", "station": "string"},
    )
    if len(index) != contract["expected_rows"] or not index[KEYS].equals(sample[KEYS]):
        raise ContractError("P3 official schema/key/order contract failed")
    if set(index["lead_h"].astype(int)) != EXPECTED_LEADS or index[KEYS].duplicated().any():
        raise ContractError("P3 lead/key uniqueness contract failed")
    return {
        "status": "PREFLIGHT_MODEL_CHAIN_PASS",
        "candidate_id": contract["candidate_id"],
        "rows": len(index),
        "columns": contract["expected_columns"],
        "key_order_exact": True,
        "official_input_hashes_ok": True,
        "package_atomic": True,
        "verified_model_files": len(models),
        "official": official,
    }


def materialize(
    data_dir: str | Path,
    package_dir: str | Path = ".",
    output_path: str | Path = "05_answer/P3_submission.csv",
) -> dict:
    package = Path(package_dir).resolve()
    preflight(data_dir, package)
    receipt = _load_predictor(package).predict(data_dir, package, output_path)
    if receipt["status"] != "READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED":
        raise ContractError("P3 predictor did not reach ready state")
    write_json(package / "05_answer" / "receipt.json", receipt)
    return receipt
