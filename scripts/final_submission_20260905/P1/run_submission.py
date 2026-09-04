"""Fail-closed P1 runner: verified trained checkpoints -> official CSV."""

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

KEYS = ["station", "year", "layer", "time"]


def _load_predictor(package: Path):
    path = package / "04_predict" / "predict_submission.py"
    spec = importlib.util.spec_from_file_location("p1_final_predictor", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load P1 predictor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(data_dir: str | Path, package_dir: str | Path = ".") -> dict:
    package = Path(package_dir).resolve()
    data = Path(data_dir).resolve()
    contract = load_contract(package)
    official = verify_official_files(data, contract)
    models = verify_package_files(package, contract, "model_files")
    derived = verify_package_files(package, contract, "derived_files")
    decisions = verify_package_files(package, contract, "decision_files")
    test = pd.read_csv(data / "test.csv", usecols=KEYS)
    sample = pd.read_csv(data / "sample_submission.csv", usecols=KEYS)
    if len(test) != contract["expected_rows"] or not test.equals(sample):
        raise ContractError("P1 official schema/key/order contract failed")
    if test.duplicated().any():
        raise ContractError("P1 official keys are not unique")
    return {
        "status": "PREFLIGHT_MODEL_CHAIN_PASS",
        "candidate_id": contract["candidate_id"],
        "rows": len(test),
        "columns": contract["expected_columns"],
        "key_order_exact": True,
        "official_input_hashes_ok": True,
        "package_atomic": True,
        "verified_model_files": len(models),
        "verified_derived_files": len(derived),
        "verified_decision_files": len(decisions),
        "official": official,
    }


def materialize(
    data_dir: str | Path,
    package_dir: str | Path = ".",
    output_path: str | Path = "05_answer/P1_submission.csv",
) -> dict:
    package = Path(package_dir).resolve()
    preflight(data_dir, package)
    receipt = _load_predictor(package).predict(data_dir, package, output_path)
    if receipt["status"] != "READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED":
        raise ContractError("P1 predictor did not reach ready state")
    write_json(package / "05_answer" / "receipt.json", receipt)
    return receipt
