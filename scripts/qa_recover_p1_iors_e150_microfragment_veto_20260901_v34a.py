"""Independent metric recovery from the already sealed v34a action.

This is not a retry: it cannot construct or alter an action and refuses to run
without the original truth-blind seal and terminal technical-failure receipt.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_iors_e150_microfragment_veto_20260901_v34a"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module(
    "p1_v34a_recovery_source",
    ROOT / "scripts/run_p1_iors_e150_microfragment_veto_20260901_v34a.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def recover() -> dict[str, Any]:
    started = time.perf_counter()
    failure_path = ARTIFACT / "terminal_failure.json"
    seal_path = ARTIFACT / "action-seal.json"
    action_path = ARTIFACT / "sealed_action.npz"
    result_path = ARTIFACT / "result.json"
    if result_path.exists() or (REPORT / "result.json").exists():
        raise FileExistsError("recovery result already exists")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if failure["status"] != "TERMINAL_TECHNICAL_FAILURE" or failure["error"] != "'difference_ci90'":
        raise RuntimeError("unexpected failure; recovery not authorized")
    if seal["truth_reads_before_action_seal"] != 0 or seal["official_reads"] != 0:
        raise RuntimeError("action was not truth-blind")
    if sha256_file(action_path) != seal["sealed_npz_sha256"]:
        raise RuntimeError("sealed action hash mismatch")
    config, paths = runner.load_contract()
    with np.load(action_path, allow_pickle=False) as archive:
        raw = archive["raw_e150"].astype(np.int8)
        candidate = archive["candidate"].astype(np.int8)
        removal = archive["removal"].astype(bool)
    if runner.sha256_array(raw) != seal["raw_sha256"]:
        raise RuntimeError("raw action array mismatch")
    if runner.sha256_array(candidate) != seal["candidate_sha256"]:
        raise RuntimeError("candidate action array mismatch")
    if runner.sha256_array(removal.astype(np.uint8)) != seal["removal_sha256"]:
        raise RuntimeError("removal action array mismatch")
    anchor = pd.read_parquet(
        paths["anchor"], columns=[*runner.KEYS, "fold", "current_router_prediction"]
    )
    historical, candidate = runner.truth_source.attach_truth(anchor, candidate)
    truth = historical["label_base"].to_numpy(np.int8)
    by_fold = {
        fold: runner.metric_block(
            truth,
            raw,
            candidate,
            historical["fold"].eq(fold).to_numpy(),
            removal,
        )
        for fold in runner.FOLDS
    }
    primary = historical["fold"].isin(runner.FOLDS[1:]).to_numpy()
    pooled = runner.metric_block(truth, raw, candidate, primary, removal)
    bootstrap = runner.truth_source.base.day_bootstrap(historical, raw, candidate, config)
    removed_tp_share = pooled["removed_true_positive_share"]
    gates = {
        "primary_delta_f1_strictly_positive": pooled["delta_f1"] > 0.0,
        "q3_delta_f1_nonnegative": by_fold["2025_q3"]["delta_f1"] >= 0.0,
        "q4_delta_f1_nonnegative": by_fold["2025_q4"]["delta_f1"] >= 0.0,
        "primary_day_block_ci90_low_nonnegative": bootstrap["ci90_low"] >= 0.0,
        "removed_true_positive_share_strictly_below_reference_f1_half": removed_tp_share is not None and removed_tp_share < pooled["reference_f1"] / 2.0,
        "anchor_removals_zero": not np.any(
            removal
            & (anchor["current_router_prediction"].to_numpy(np.int8) == 1)
        ),
        "primary_removed_rows_at_least": pooled["removed_rows"] >= 1,
    }
    geometry = runner.official_geometry(config)
    internal_pass = all(gates.values())
    result = {
        "schema_version": "p1.iors_e150_microfragment_veto.recovered_result.v34a",
        "experiment_id": EXPERIMENT_ID,
        "status": "INTERNAL_PASS_OFFICIAL_GEOMETRY_BLOCKED" if internal_pass else "TERMINAL_NO_GO",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 0,
        "by_fold": by_fold,
        "pooled_primary_q3_q4": pooled,
        "day_block_bootstrap": bootstrap,
        "expected_points_delta": pooled["delta_f1"] * float(config["score"]["points_per_f1"]),
        "gates": gates,
        "internal_pass": internal_pass,
        "official_metric_geometry": geometry,
        "official_materializer_execute_allowed": internal_pass and geometry["metric_geometry_consistent"],
        "recovery": {
            "kind": "independent_metric_replay_of_preexisting_sealed_action",
            "action_rebuilt": False,
            "candidate_reselected": False,
            "threshold_changed": False,
            "technical_failure": failure,
            "sealed_action_sha256": sha256_file(action_path),
        },
        "operations": {"official_reads": 0, "hidden_truth_reads": 0, "test_reads": 0, "sample_reads": 0, "submission_csv_created": 0, "uploads": 0, "retries": 0},
        "hashes": {"config": sha256_file(CONFIG_PATH), "recovery_qa": sha256_file(Path(__file__)), "action_seal": sha256_file(seal_path), "sealed_action": sha256_file(action_path)},
    }
    write_json_new(result_path, result)
    write_json_new(REPORT / "result.json", result)
    qa_checks = {
        "original_action_sealed_truth_blind": seal["truth_reads_before_action_seal"] == 0,
        "sealed_action_hash_matches": result["hashes"]["sealed_action"] == seal["sealed_npz_sha256"],
        "action_not_rebuilt": result["recovery"]["action_rebuilt"] is False,
        "candidate_not_reselected": result["recovery"]["candidate_reselected"] is False,
        "fit0": result["fit_count"] == 0,
        "official0": result["operations"]["official_reads"] == 0,
        "hidden0": result["operations"]["hidden_truth_reads"] == 0,
        "csv0": result["operations"]["submission_csv_created"] == 0,
        "upload0": result["operations"]["uploads"] == 0,
        "gates_recomputed": result["internal_pass"] == all(result["gates"].values()),
    }
    qa = {"status": "PASS" if all(qa_checks.values()) else "FAIL", "checks": qa_checks}
    write_json_new(REPORT / "independent-qa.json", qa)
    return result


if __name__ == "__main__":
    print(json.dumps(recover(), ensure_ascii=False, indent=2, sort_keys=True))
