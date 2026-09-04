"""Fail-closed P2 runner: freshly trained v52 checkpoints -> official CSV."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    ContractError,
    load_contract,
    verify_official_files,
    verify_package_files,
    write_json,
)

KEYS = ["station", "layer", "time"]


def _load_predictor(package: Path):
    path = package / "04_predict" / "predict_submission.py"
    spec = importlib.util.spec_from_file_location("p2_final_predictor", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load P2 predictor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(data_dir: str | Path, package_dir: str | Path = ".") -> dict:
    package = Path(package_dir).resolve()
    data = Path(data_dir).resolve()
    contract = load_contract(package)
    official = verify_official_files(data, contract)
    models = verify_package_files(package, contract, "model_files")
    decisions = verify_package_files(package, contract, "decision_files")
    index = pd.read_csv(data / "test_index.csv", dtype={"station": "string", "time": "string"})
    sample = pd.read_csv(
        data / "sample_submission.csv",
        usecols=KEYS,
        dtype={"station": "string", "time": "string"},
    )
    anchor = pd.read_csv(
        package / "03_model" / "decision_artifacts" / "bin17_anchor.csv",
        dtype={"station": "string", "time": "string"},
    )
    if not (len(index) == len(sample) == len(anchor) == contract["expected_rows"]):
        raise ContractError("P2 row-count contract failed")
    if not index[KEYS].equals(sample[KEYS]) or not index[KEYS].equals(anchor[KEYS]):
        raise ContractError("P2 official/anchor schema-key-order contract failed")
    values = pd.to_numeric(anchor["temp"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ContractError("P2 anchor contains non-finite values")
    return {
        "status": "PREFLIGHT_MODEL_CHAIN_PASS",
        "candidate_id": contract["candidate_id"],
        "rows": len(index),
        "columns": contract["expected_columns"],
        "key_order_exact": True,
        "official_input_hashes_ok": True,
        "package_atomic": True,
        "verified_model_files": len(models),
        "verified_decision_files": len(decisions),
        "official": official,
    }


def materialize(
    data_dir: str | Path,
    package_dir: str | Path = ".",
    output_path: str | Path = "05_answer/P2_submission.csv",
) -> dict:
    package = Path(package_dir).resolve()
    preflight(data_dir, package)
    receipt = _load_predictor(package).predict(data_dir, package, output_path)
    if receipt["status"] != "READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED":
        raise ContractError("P2 predictor did not reach ready state")
    write_json(package / "05_answer" / "receipt.json", receipt)
    return receipt
