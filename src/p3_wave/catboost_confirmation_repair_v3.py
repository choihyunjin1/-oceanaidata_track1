"""Fail-closed schema contract for the P3 CatBoost confirmation-only repair."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p3_wave.kma_source_meta import PAIR_KEYS, ROUTER_COLUMNS, read_frozen_router_components

EXPERIMENT_ID = "p3_catboost_confirmation_contract_repair_20260830_v3"
SUPPLEMENTAL_COLUMNS = (*PAIR_KEYS, "current_hs", "single_prediction")
CONTRACT_COLUMNS = (*PAIR_KEYS, "current_hs", "single_prediction", *ROUTER_COLUMNS)


class ConfirmationContractError(RuntimeError):
    """Raised before model fitting when the frozen confirmation contract differs."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_fingerprint(frame: pd.DataFrame) -> str:
    payload = {
        "columns": list(frame.columns),
        "dtypes": [str(value) for value in frame.dtypes],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_selection_receipt(
    receipt: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    if receipt.get("experiment_id") != "p3_catboost_valid_hpo_20260829_v2":
        raise ConfirmationContractError("selection receipt experiment id changed")
    if receipt.get("selected_candidate_id") != frozen.get("selected_candidate_id"):
        raise ConfirmationContractError("frozen selected candidate changed")
    if receipt.get("selected_parameters") != frozen.get("selected_parameters"):
        raise ConfirmationContractError("frozen selected parameters changed")
    observed_iterations = [int(value) for value in receipt.get("selected_best_iterations", [])]
    expected_iterations = [int(value) for value in frozen.get("selected_best_iterations", [])]
    if observed_iterations != expected_iterations:
        raise ConfirmationContractError("frozen selected best iterations changed")
    if int(np.floor(np.median(observed_iterations))) != int(frozen.get("selected_iteration", -1)):
        raise ConfirmationContractError("frozen selected iteration rule changed")
    gate = receipt.get("gate", {})
    if gate.get("pass") is not True or not all(gate.get("checks", {}).values()):
        raise ConfirmationContractError("frozen v2 selection gate is not a complete pass")
    return {
        "selected_candidate_id": str(receipt["selected_candidate_id"]),
        "selected_iteration": int(frozen["selected_iteration"]),
        "selection_gate_pass": True,
        "selection_delta_rmse_m": float(receipt["metrics"]["delta_rmse_m"]),
    }


def read_confirmation_contract(path: str | Path) -> pd.DataFrame:
    """Join only the two missing frozen columns to the canonical router projection."""

    supplemental = pd.read_parquet(path, columns=list(SUPPLEMENTAL_COLUMNS))
    if list(supplemental.columns) != list(SUPPLEMENTAL_COLUMNS):
        raise ConfirmationContractError("supplemental confirmation column order changed")
    if supplemental.duplicated(list(PAIR_KEYS)).any():
        raise ConfirmationContractError("supplemental confirmation keys are duplicated")
    numeric = supplemental[["current_hs", "single_prediction"]].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or (numeric < 0.0).any()
        or (numeric[:, 1] > 30.0).any()
    ):
        raise ConfirmationContractError("supplemental confirmation values are invalid")

    router = read_frozen_router_components(path)
    contract = supplemental.merge(
        router,
        on=list(PAIR_KEYS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(contract) != len(supplemental) or len(contract) != len(router):
        raise ConfirmationContractError("supplemental and canonical router keys differ")
    contract = contract.loc[:, list(CONTRACT_COLUMNS)]
    validate_confirmation_contract(contract)
    return contract


def validate_confirmation_contract(frame: pd.DataFrame) -> dict[str, Any]:
    if list(frame.columns) != list(CONTRACT_COLUMNS):
        raise ConfirmationContractError("confirmation contract column order changed")
    if frame.duplicated(list(PAIR_KEYS)).any():
        raise ConfirmationContractError("confirmation contract keys are duplicated")
    if any("target" in str(column).lower() or "truth" in str(column).lower() for column in frame):
        raise ConfirmationContractError("truth column leaked into confirmation contract")
    numeric_columns = ["current_hs", "single_prediction", *ROUTER_COLUMNS]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ConfirmationContractError("confirmation contract contains non-finite values")
    lead_sets = frame.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if not lead_sets.map(lambda values: values == (3, 6, 9, 12, 18, 24)).all():
        raise ConfirmationContractError("confirmation case does not contain exactly six leads")
    return {
        "rows": int(len(frame)),
        "cases": int(frame[["fold", "anchor_id"]].drop_duplicates().shape[0]),
        "folds": sorted(frame["fold"].astype(str).unique().tolist()),
        "schema_sha256": schema_fingerprint(frame),
    }


def build_single_blind(
    contract_fold: pd.DataFrame, challenger_prediction: pd.DataFrame
) -> pd.DataFrame:
    challenger_columns = ["anchor_id", "station", "lead_h", "prediction"]
    if not set(challenger_columns).issubset(challenger_prediction.columns):
        raise ConfirmationContractError("challenger prediction schema is incomplete")
    challenger = challenger_prediction.loc[:, challenger_columns].rename(
        columns={"prediction": "challenger_single_prediction"}
    )
    if challenger.duplicated(["anchor_id", "station", "lead_h"]).any():
        raise ConfirmationContractError("challenger prediction keys are duplicated")
    control = contract_fold.loc[:, list(SUPPLEMENTAL_COLUMNS)].rename(
        columns={"single_prediction": "control_single_prediction"}
    )
    single = control.merge(
        challenger,
        on=["anchor_id", "station", "lead_h"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    expected = [
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "current_hs",
        "control_single_prediction",
        "challenger_single_prediction",
    ]
    single = single.loc[:, expected]
    if len(single) != len(control) or len(single) != len(challenger):
        raise ConfirmationContractError("control and challenger confirmation keys differ")
    numeric = single[
        ["current_hs", "control_single_prediction", "challenger_single_prediction"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (numeric[:, 1:] < 0.0).any() or (numeric[:, 1:] > 30.0).any():
        raise ConfirmationContractError("single blind predictions are invalid")
    return single


__all__ = [
    "CONTRACT_COLUMNS",
    "EXPERIMENT_ID",
    "SUPPLEMENTAL_COLUMNS",
    "ConfirmationContractError",
    "build_single_blind",
    "read_confirmation_contract",
    "schema_fingerprint",
    "sha256_file",
    "validate_confirmation_contract",
    "validate_selection_receipt",
]
