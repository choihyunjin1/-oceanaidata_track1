#!/usr/bin/env python3
"""Confirmation-only recovery of the frozen P3 CatBoost v2 winner."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from p3_wave.catboost_confirmation_repair_v3 import (  # noqa: E402
    EXPERIMENT_ID,
    ConfirmationContractError,
    build_single_blind,
    read_confirmation_contract,
    sha256_file,
    validate_confirmation_contract,
    validate_selection_receipt,
)
from p3_wave.catboost_ordered_hpo import (  # noqa: E402
    apply_frozen_kma_alpha,
    evaluate_confirmation_gate,
    metric_deltas,
    paired_case_bootstrap,
)
from p3_wave.kma_source_meta import (  # noqa: E402
    PAIR_KEYS,
    ROUTER_COLUMNS,
    integrate_frozen_router,
)
from p3_wave.models import compact_feature_columns  # noqa: E402
from p3_wave.revin_patch import assign_storm_episodes_from_wave  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/experiments/p3_catboost_confirmation_contract_repair_20260830_v3.json"
V2_CONFIG = ROOT / "configs/experiments/p3_catboost_valid_hpo_20260829_v2.json"
V1_CONFIG = ROOT / "configs/experiments/p3_catboost_ordered_hpo_20260829_v1.json"
FROZEN_ENGINE_PATH = ROOT / "scripts/run_p3_catboost_ordered_hpo_20260829_v1.py"
CONTRACT_PATH = ROOT / "src/p3_wave/catboost_confirmation_repair_v3.py"
TEST_PATH = ROOT / "tests/test_p3_catboost_confirmation_contract_repair_20260830_v3.py"
FORBIDDEN_BASENAMES = {
    "test_context.parquet",
    "test_index.csv",
    "sample_submission.csv",
    "baseline_persistence.csv",
    "score.py",
}

SPEC = importlib.util.spec_from_file_location("p3_catboost_frozen_v1_engine", FROZEN_ENGINE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen P3 HPO engine")
ENGINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENGINE)


def _contained(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ConfirmationContractError(f"path escapes repository: {relative}") from exc
    if path.name.lower() in FORBIDDEN_BASENAMES:
        raise ConfirmationContractError(f"official path is forbidden: {path.name}")
    return path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_CONFIG.resolve():
        raise ConfirmationContractError("only the canonical v3 config is allowed")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ConfirmationContractError("v3 experiment id changed")
    if config["data_boundary"] != {
        "historical_train_only": True,
        "official_rows_read": 0,
        "official_files_forbidden": [
            "test_context.parquet",
            "test_index.csv",
            "sample_submission.csv",
            "baseline_persistence.csv",
            "score.py",
        ],
        "absolute_official_time_reconstruction_allowed": False,
        "external_evaluation_period_matching_allowed": False,
        "csv_output_allowed": False,
        "submission_or_upload_allowed": False,
        "source_mutation_allowed": False,
    }:
        raise ConfirmationContractError("v3 data boundary changed")
    if config["frozen_v2"]["selection_search_rerun_allowed"] is not False:
        raise ConfirmationContractError("v2 selection search cannot be rerun")
    if config["execution"] != {
        "allowed_mode_now": "one-shot-execute",
        "actual_authorized": True,
        "future_execute_token": "ROOT_APPROVED_P3_CATBOOST_CONFIRMATION_REPAIR_20260830_V3",
        "search_fit_count": 0,
        "model_fit_count_in_static_preflight": 0,
        "official_rows_read_in_static_preflight": 0,
        "csv_write_count": 0,
    }:
        raise ConfirmationContractError("v3 execution boundary changed")
    if int(config["confirmation"]["challenger_fit_count"]) != 3:
        raise ConfirmationContractError("v3 challenger confirmation fit count changed")
    if int(config["confirmation"]["maximum_historical_fit_count"]) != 6:
        raise ConfirmationContractError("v3 maximum fit budget changed")
    return config


def load_frozen_v2(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = config["frozen_v2"]
    if sha256_file(V2_CONFIG) != frozen["config_sha256"]:
        raise ConfirmationContractError("frozen v2 config hash changed")
    v2 = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    if v2.get("extends") != str(V1_CONFIG.relative_to(ROOT)).replace("\\", "/"):
        raise ConfirmationContractError("frozen v2 base path changed")
    if sha256_file(V1_CONFIG) != v2.get("extends_sha256"):
        raise ConfirmationContractError("frozen v1 base config hash changed")
    base = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    merged = _deep_merge(base, v2)
    selection_path = _contained(frozen["selection_path"])
    if sha256_file(selection_path) != frozen["selection_sha256"]:
        raise ConfirmationContractError("frozen v2 selection artifact hash changed")
    receipt = json.loads(selection_path.read_text(encoding="utf-8"))
    return merged, receipt


def _source_paths(source: dict[str, Any], data_dir: Path | None) -> dict[str, Path]:
    names = ("feature_columns", "train_features", "train_anchors", "frozen_router_oof", "frozen_kma_oof")
    paths = {name: _contained(source["inputs"][name]["path"]) for name in names}
    if data_dir is not None:
        root = data_dir.resolve(strict=True)
        relative = source["inputs"]["train_wave"]["relative_path"]
        if Path(relative).name != "train_wave.csv":
            raise ConfirmationContractError("historical wave filename changed")
        paths["train_wave"] = root / relative
    return paths


def _verify_source_hashes(source: dict[str, Any], paths: dict[str, Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ConfirmationContractError(f"missing frozen historical input: {name}")
        digest = sha256_file(path)
        if digest != source["inputs"][name]["sha256"]:
            raise ConfirmationContractError(f"frozen historical input hash changed: {name}")
        result[name] = digest
    return result


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _exclusive_json(path: Path, payload: Any) -> None:
    _exclusive_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))


def _exclusive_parquet(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, compression="zstd")
    _exclusive_bytes(path, buffer.getvalue())


def static_preflight(config_path: Path, data_dir: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    source, selection = load_frozen_v2(config)
    selection_audit = validate_selection_receipt(selection, config["frozen_v2"])
    paths = _source_paths(source, data_dir)
    input_hashes = _verify_source_hashes(source, paths)

    feature_file = pq.ParquetFile(paths["train_features"])
    anchor_file = pq.ParquetFile(paths["train_anchors"])
    frozen_columns = json.loads(paths["feature_columns"].read_text(encoding="utf-8"))
    observed_columns = compact_feature_columns(
        [name for name in feature_file.schema_arrow.names if name not in {"anchor_id", "station"}]
    )
    if observed_columns != frozen_columns or len(observed_columns) != 591:
        raise ConfirmationContractError("frozen 591-column feature contract changed")
    if feature_file.metadata.num_rows != 24360 or anchor_file.metadata.num_rows != 24360:
        raise ConfirmationContractError("frozen feature or anchor row count changed")

    contract = read_confirmation_contract(paths["frozen_router_oof"])
    contract_audit = validate_confirmation_contract(contract)
    expected_folds = sorted(row[0] for row in config["confirmation"]["windows"])
    if contract_audit["folds"] != expected_folds:
        raise ConfirmationContractError("confirmation fold names changed")
    if contract_audit["rows"] != config["confirmation"]["expected_rows"]:
        raise ConfirmationContractError("confirmation row count changed")
    if contract_audit["cases"] != config["confirmation"]["expected_cases"]:
        raise ConfirmationContractError("confirmation case count changed")

    anchors = pd.read_parquet(
        paths["train_anchors"], columns=["anchor_id", "station", "anchor_time", "current_hs"]
    )
    anchor_lookup = anchors.set_index("anchor_id")
    anchor_rows = anchor_lookup.loc[contract["anchor_id"]]
    if not anchor_rows["station"].astype(str).reset_index(drop=True).eq(
        contract["station"].astype(str).reset_index(drop=True)
    ).all():
        raise ConfirmationContractError("confirmation station differs from frozen anchors")
    if not np.allclose(
        anchor_rows["current_hs"].to_numpy(dtype=np.float64),
        contract["current_hs"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ConfirmationContractError("confirmation current_hs differs from frozen anchors")

    train_membership: dict[str, int] | str = "PENDING_EXPLICIT_DATA_DIR"
    if "train_wave" in paths:
        wave = pd.read_csv(paths["train_wave"], usecols=["station", "time", "hs"])
        episode_anchors = assign_storm_episodes_from_wave(anchors, wave)
        windows = {row[0]: row for row in config["confirmation"]["windows"]}
        train_membership = {}
        for fold_name, block in contract.groupby("fold", sort=False, observed=True):
            validation_ids = np.sort(block["anchor_id"].unique().astype(np.int64))
            train_ids = ENGINE._confirmation_train_ids(
                episode_anchors, validation_ids, windows[str(fold_name)][1]
            )
            train_membership[str(fold_name)] = int(len(train_ids))

    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_GUARDED" if data_dir is not None else "STATIC_CONTRACT_PASS_DATA_DIR_PENDING",
        "selection": selection_audit,
        "confirmation_contract": contract_audit,
        "confirmation_train_anchors": train_membership,
        "input_hashes": input_hashes,
        "implementation_hashes": {
            "runner": sha256_file(Path(__file__)),
            "contract": sha256_file(CONTRACT_PATH),
            "tests": sha256_file(TEST_PATH),
            "frozen_engine": sha256_file(FROZEN_ENGINE_PATH),
        },
        "execution_boundary": {
            "selection_search_fit_count": 0,
            "model_fit_count": 0,
            "official_rows_read": 0,
            "csv_files_written": 0,
            "attempt_lock_created": False,
        },
    }


def execute_confirmation(
    config_path: Path, data_dir: Path, authorization_token: str | None
) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config(config_path)
    if authorization_token != config["execution"]["future_execute_token"]:
        raise ConfirmationContractError("execute authorization token differs")
    preflight = static_preflight(config_path, data_dir)
    if preflight["status"] != "READY_GUARDED":
        raise ConfirmationContractError("canonical preflight did not reach READY_GUARDED")

    source, selection = load_frozen_v2(config)
    paths = _source_paths(source, data_dir)
    outputs = {
        name: _contained(relative)
        for name, relative in config["outputs"].items()
        if name != "artifact_dir"
    }
    consumed = [name for name, path in outputs.items() if name != "static_preflight_receipt" and path.exists()]
    if consumed:
        raise ConfirmationContractError(f"one-shot v3 output already exists: {sorted(consumed)}")
    _exclusive_json(
        outputs["attempt_lock"],
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(config_path),
            "selection_sha256": config["frozen_v2"]["selection_sha256"],
            "selection_search_rerun": False,
            "maximum_historical_fit_count": config["confirmation"]["maximum_historical_fit_count"],
            "rerun_forbidden": True,
        },
    )

    features = pd.read_parquet(paths["train_features"])
    anchors = pd.read_parquet(paths["train_anchors"])
    wave = pd.read_csv(paths["train_wave"], usecols=["station", "time", "hs"])
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    feature_columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    contract = read_confirmation_contract(paths["frozen_router_oof"])
    kma = pd.read_parquet(
        paths["frozen_kma_oof"],
        columns=["fold", "anchor_id", "station", "lead_h", "calibrated_source"],
    )
    parameters = dict(config["frozen_v2"]["selected_parameters"])
    iterations = int(config["frozen_v2"]["selected_iteration"])
    windows = {row[0]: row for row in config["confirmation"]["windows"]}
    blind_blocks: list[pd.DataFrame] = []
    fit_receipts: list[dict[str, Any]] = []

    for fold_name in [row[0] for row in config["confirmation"]["windows"]]:
        contract_fold = contract.loc[contract["fold"].astype(str).eq(str(fold_name))].copy()
        validation_ids = np.sort(contract_fold["anchor_id"].unique().astype(np.int64))
        train_ids = ENGINE._confirmation_train_ids(
            anchors, validation_ids, windows[str(fold_name)][1]
        )
        challenger, best_iteration, elapsed = ENGINE._fit_predict(
            parameters,
            iterations,
            features,
            anchors,
            feature_columns,
            train_ids,
            validation_ids,
        )
        single = build_single_blind(contract_fold, challenger)
        canonical_router = contract_fold.loc[:, [*PAIR_KEYS, *ROUTER_COLUMNS]]
        integrated = integrate_frozen_router(single, canonical_router)
        if not np.allclose(
            integrated["control_final"],
            integrated["incumbent_final"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ConfirmationContractError("frozen control router reconstruction changed")
        blind_blocks.append(integrated)
        fit_receipts.append(
            {
                "fold": str(fold_name),
                "train_cases": int(len(train_ids)),
                "validation_cases": int(len(validation_ids)),
                "iterations": iterations,
                "observed_best_iteration": int(best_iteration),
                "elapsed_seconds": float(elapsed),
            }
        )
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "stage": "confirmation_fold_fit_complete",
                    "fold": str(fold_name),
                    "completed_fits": len(fit_receipts),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(fit_receipts) != 3:
        raise ConfirmationContractError("confirmation did not complete exactly three fits")
    integrated = pd.concat(blind_blocks, ignore_index=True)
    blind = integrated.merge(
        kma, on=["fold", "anchor_id", "station", "lead_h"], validate="one_to_one"
    )
    if len(blind) != config["confirmation"]["expected_rows"]:
        raise ConfirmationContractError("confirmation KMA intersection row count changed")
    blind["control_prediction"] = apply_frozen_kma_alpha(
        blind["control_final"], blind["calibrated_source"], blind["lead_h"]
    )
    blind["challenger_prediction"] = apply_frozen_kma_alpha(
        blind["challenger_final"], blind["calibrated_source"], blind["lead_h"]
    )
    blind = blind[
        [
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "control_prediction",
            "challenger_prediction",
        ]
    ].sort_values(["fold", "anchor_id", "station", "lead_h"], kind="mergesort")
    _exclusive_parquet(outputs["confirmation_blind_predictions"], blind)
    seal = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "confirmation_before_truth_metric",
        "prediction_sha256": sha256_file(outputs["confirmation_blind_predictions"]),
        "row_count": int(len(blind)),
        "columns": list(blind.columns),
        "truth_columns_present": False,
        "config_sha256": sha256_file(config_path),
        "selection_sha256": config["frozen_v2"]["selection_sha256"],
    }
    _exclusive_json(outputs["confirmation_seal"], seal)

    sealed = pd.read_parquet(outputs["confirmation_blind_predictions"])
    truth_blocks: list[pd.DataFrame] = []
    anchor_lookup = anchors.set_index("anchor_id")
    for lead in [3, 6, 9, 12, 18, 24]:
        block = sealed.loc[
            sealed["lead_h"].eq(lead), ["fold", "anchor_id", "station", "lead_h"]
        ].copy()
        block["target_hs"] = anchor_lookup.loc[block["anchor_id"], f"target_{lead}"].to_numpy()
        truth_blocks.append(block)
    truth = pd.concat(truth_blocks, ignore_index=True)
    evaluated = sealed.merge(
        truth, on=["fold", "anchor_id", "station", "lead_h"], validate="one_to_one"
    )
    metrics = metric_deltas(evaluated)
    bootstrap = paired_case_bootstrap(
        evaluated,
        replicates=config["confirmation"]["bootstrap"]["replicates"],
        seed=config["confirmation"]["bootstrap"]["seed"],
    )
    gate = evaluate_confirmation_gate(metrics, bootstrap, config["confirmation"]["gate"])
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "CONFIRMATION_GATE_PASS" if gate["pass"] else "CONFIRMATION_GATE_FAIL_HPO_CLOSED",
        "preflight": preflight,
        "frozen_selection": {
            "selected_candidate_id": selection["selected_candidate_id"],
            "selected_iteration": iterations,
            "selection_sha256": config["frozen_v2"]["selection_sha256"],
            "selection_search_fit_count": 0,
        },
        "confirmation": {
            "metrics": metrics,
            "paired_case_bootstrap": bootstrap,
            "gate": gate,
            "blind_prediction_sha256": seal["prediction_sha256"],
            "fit_count": len(fit_receipts),
            "fit_receipts": fit_receipts,
        },
        "runtime_seconds": time.monotonic() - started,
        "full_refit_fit_count": 0,
        "official_rows_read": 0,
        "csv_files_written": 0,
        "submission_or_upload_attempted": False,
    }
    _exclusive_json(outputs["result"], result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static-preflight", "execute"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--authorization-token")
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.mode == "execute":
        if args.data_dir is None:
            parser.error("--data-dir is required for execute")
        result = execute_confirmation(config_path, args.data_dir, args.authorization_token)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    receipt = static_preflight(config_path, args.data_dir)
    if args.write_receipt:
        config = load_config(config_path)
        _exclusive_json(_contained(config["outputs"]["static_preflight_receipt"]), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
