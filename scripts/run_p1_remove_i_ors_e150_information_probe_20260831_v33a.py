"""Exactly-once historical test of removing only I-ORS raw-E150 additions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1 as e150_source  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v13 as truth_source  # noqa: E402

EXPERIMENT_ID = "p1_remove_i_ors_e150_information_probe_20260831_v33a"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
KEYS = ["station", "year", "layer", "time"]
FOLDS = ["2025_q2", "2025_q3", "2025_q4"]


class ContractError(RuntimeError):
    """Raised when the frozen v33a contract changes."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def resolve_source(spec: dict[str, str]) -> Path:
    path = (ROOT / spec["path"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ContractError(f"historical source unavailable: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise ContractError(f"historical source hash changed: {path}")
    return path


def load_contract() -> tuple[dict[str, Any], dict[str, Path]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    checks = {
        "schema": config["schema_version"] == "p1.remove_i_ors_e150_information_probe.20260831.v33a",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "candidate": config["candidate"] == "P1_REMOVE_I_ORS_E150_INFORMATION_PROBE",
        "station": config["decoder"]["station"] == "I-ORS",
        "threshold_absent": config["decoder"]["threshold"] is None,
        "top_k_absent": config["decoder"]["top_k"] is None,
        "fit0": config["decoder"]["model_fits"] == 0,
        "folds": config["validation"]["diagnostic_folds"] == FOLDS,
        "primary": config["validation"]["primary_outer_folds"] == FOLDS[1:],
        "official_zero": config["authorization"]["official_reads"] == 0,
        "hidden_zero": config["authorization"]["hidden_truth_reads"] == 0,
        "csv_zero": config["authorization"]["submission_csv_created"] == 0,
        "upload_zero": config["authorization"]["uploads"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"v33a config contract failed: {checks}")
    paths = {name: resolve_source(config["historical_sources"][name]) for name in ["anchor", "truth", *FOLDS]}
    return config, paths


def build_truth_blind_action(anchor_frame: pd.DataFrame, paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    incumbent = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    raw_e150 = np.empty(len(anchor_frame), dtype=np.int8)
    receipts: list[dict[str, Any]] = []
    for fold in FOLDS:
        mask = anchor_frame["fold"].eq(fold).to_numpy()
        archive = e150_source._select_archive_arrays(fold, paths[fold])
        raw = np.asarray(archive["candidate"], dtype=np.int8)
        if raw.shape != (int(mask.sum()),) or not np.isin(raw, [0, 1]).all():
            raise ContractError(f"raw E150 fold alignment failed: {fold}")
        raw_e150[mask] = raw
        receipts.append({"fold": fold, "rows": int(mask.sum()), "raw_e150_positives": int(raw.sum()), "truth_reads_before_seal": 0})
    if not np.all(raw_e150[incumbent == 1] == 1):
        raise ContractError("raw E150 does not preserve an incumbent positive")
    removal = (
        anchor_frame["station"].astype(str).eq("I-ORS").to_numpy()
        & (incumbent == 0)
        & (raw_e150 == 1)
    )
    candidate = raw_e150.copy()
    candidate[removal] = 0
    if np.any(candidate[incumbent == 1] == 0) or np.any(candidate > raw_e150):
        raise ContractError("v33a action is not an exact incumbent-preserving removal")
    return raw_e150, candidate, removal, receipts


def seal_action(config: dict[str, Any], paths: dict[str, Path], raw_e150: np.ndarray, candidate: np.ndarray, removal: np.ndarray, receipts: list[dict[str, Any]]) -> dict[str, Any]:
    sealed_path = ARTIFACT / "sealed_action.npz"
    np.savez_compressed(sealed_path, raw_e150=raw_e150, candidate=candidate, removal=removal)
    seal = {
        "schema_version": "p1.remove_i_ors_e150.action_seal.v33a",
        "experiment_id": EXPERIMENT_ID,
        "rule": config["decoder"],
        "receipts": receipts,
        "rows": int(len(candidate)),
        "removals": int(removal.sum()),
        "raw_e150_sha256": sha256_array(raw_e150),
        "candidate_sha256": sha256_array(candidate),
        "removal_sha256": sha256_array(removal.astype(np.uint8)),
        "sealed_npz_sha256": sha256_file(sealed_path),
        "source_hashes": {name: sha256_file(path) for name, path in paths.items() if name != "truth"},
        "truth_reads_before_action_seal": 0,
        "official_reads": 0,
        "hidden_truth_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0
    }
    write_json_new(ARTIFACT / "action-seal.json", seal)
    return seal


def metric_block(truth: np.ndarray, reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray, removal: np.ndarray) -> dict[str, Any]:
    removed = mask & removal
    reference_f1 = float(f1_score(truth[mask], reference[mask]))
    candidate_f1 = float(f1_score(truth[mask], candidate[mask]))
    return {
        "rows": int(mask.sum()),
        "reference_f1": reference_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": candidate_f1 - reference_f1,
        "removed_e150_additions": int(removed.sum()),
        "removed_true_positives": int((removed & (truth == 1)).sum()),
        "removed_false_positives": int((removed & (truth == 0)).sum())
    }


def evaluate(frame: pd.DataFrame, reference: np.ndarray, candidate: np.ndarray, removal: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    by_fold = {fold: metric_block(truth, reference, candidate, frame["fold"].eq(fold).to_numpy(), removal) for fold in FOLDS}
    all_mask = frame["fold"].isin(FOLDS).to_numpy()
    primary_mask = frame["fold"].isin(FOLDS[1:]).to_numpy()
    pooled_all = metric_block(truth, reference, candidate, all_mask, removal)
    pooled_primary = metric_block(truth, reference, candidate, primary_mask, removal)
    bootstrap = truth_source.base.day_bootstrap(frame, reference, candidate, config)
    raw_points = pooled_primary["delta_f1"] * float(config["score"]["points_per_f1"])
    changed = frame.loc[primary_mask & removal, ["station", "layer", "fold"]].copy()
    local_day = pd.to_datetime(frame.loc[primary_mask & removal, "time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    changed["day"] = local_day.to_numpy()
    def maximum_share(columns: list[str]) -> float:
        if not len(changed):
            return 0.0
        return float(changed.groupby(columns, observed=True).size().max() / len(changed))
    removed = primary_mask & removal
    tp_removed = int((removed & (truth == 1)).sum())
    fp_removed = int((removed & (truth == 0)).sum())
    information_value_positive = bool(removal.sum() > 0)
    return {
        "name": config["candidate"],
        "fit_count": 0,
        "reference": "raw_E150",
        "by_fold": by_fold,
        "pooled_all_q2_q3_q4": pooled_all,
        "pooled_primary_q3_q4": pooled_primary,
        "removed_true_positives_primary": tp_removed,
        "removed_false_positives_primary": fp_removed,
        "removed_false_positive_share_primary": fp_removed / int(removed.sum()) if removed.any() else None,
        "raw_expected_points_delta": raw_points,
        "day_block_bootstrap": bootstrap,
        "concentration": {
            "station_max_share": maximum_share(["station"]),
            "station_layer_max_share": maximum_share(["station", "layer"]),
            "station_layer_fold_max_share": maximum_share(["station", "layer", "fold"]),
            "day_max_share": maximum_share(["day"]),
            "unique_days": int(changed["day"].nunique()) if len(changed) else 0
        },
        "information_value_positive": information_value_positive,
        "scientific_interpretation": "This isolates the only unseparated I-ORS contribution in the official station factorial; it is not a promotion claim.",
        "official_materializer_preparation_allowed": information_value_positive
    }


def execute() -> dict[str, Any]:
    # Independent decision notes may pre-create REPORT; only a result/lock consumes
    # this metric-only namespace.
    if ARTIFACT.exists() or (REPORT / "result.json").exists():
        raise FileExistsError("v33a exactly-once namespace already exists")
    started = time.perf_counter()
    config, paths = load_contract()
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json_new(ARTIFACT / "attempt_lock.json", {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "model_fits": 0,
        "retry_budget": 0,
        "official_reads": 0,
        "hidden_truth_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0
    })
    anchor_frame = pd.read_parquet(paths["anchor"], columns=[*KEYS, "fold", "current_router_prediction"])
    raw_e150, candidate, removal, receipts = build_truth_blind_action(anchor_frame, paths)
    action_seal = seal_action(config, paths, raw_e150, candidate, removal, receipts)
    historical, candidate = truth_source.attach_truth(anchor_frame, candidate)
    record = evaluate(historical, raw_e150, candidate, removal, config)
    result = {
        "schema_version": "p1.remove_i_ors_e150_information_probe.result.v33a",
        "experiment_id": EXPERIMENT_ID,
        "status": "INFORMATION_VALUE_POSITIVE" if record["information_value_positive"] else "NO_ACTION_NO_GO",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 0,
        "candidate": record,
        "action_seal": action_seal,
        "operations": {"official_reads": 0, "hidden_truth_reads": 0, "test_reads": 0, "sample_reads": 0, "submission_csv_created": 0, "uploads": 0, "retries": 0},
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "attempt_lock_sha256": sha256_file(ARTIFACT / "attempt_lock.json"),
            "action_seal_sha256": sha256_file(ARTIFACT / "action-seal.json"),
            "sealed_action_sha256": sha256_file(ARTIFACT / "sealed_action.npz")
        }
    }
    write_json_new(ARTIFACT / "result.json", result)
    write_json_new(REPORT / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-metric-only", action="store_true")
    args = parser.parse_args()
    if not args.execute_metric_only:
        raise SystemExit("--execute-metric-only required")
    try:
        result = execute()
    except Exception as exc:
        failure = {"experiment_id": EXPERIMENT_ID, "status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "official_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0}
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            write_json_new(ARTIFACT / "terminal_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
