"""Exactly-once historical adjudication of the S-ORS layer-6 E150 split probe."""

from __future__ import annotations

import argparse
import hashlib
import json
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

import run_p1_public_transport_repair_cycle_20260831_v13 as truth_source  # noqa: E402

EXPERIMENT_ID = "p1_s_ors_layer6_information_probe_20260901_v34a"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
KEYS = ["station", "year", "layer", "time"]
FOLDS = ["2025_q2", "2025_q3", "2025_q4"]


class ContractError(RuntimeError):
    """Raised when a frozen source or action contract drifts."""


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
        raise ContractError(f"source unavailable: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise ContractError(f"source hash changed: {path}")
    return path


def load_contract() -> tuple[dict[str, Any], dict[str, Path]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    freeze = config["selection_freeze"]
    checks = {
        "schema": config["schema_version"] == "p1.s_ors_layer6_information_probe.20260901.v34a",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "candidate": config["candidate"] == "P1_REMOVE_S_ORS_LAYER6_E150_INFORMATION_PROBE",
        "station": freeze["selected_station"] == "S-ORS",
        "layer": freeze["selected_layer"] == 6,
        "threshold_none": freeze["threshold"] is None,
        "top_k_none": freeze["top_k"] is None,
        "fit0": freeze["model_fits"] == 0,
        "retry0": freeze["retries"] == 0,
        "folds": config["validation"]["diagnostic_folds"] == FOLDS,
        "primary": config["validation"]["primary_outer_folds"] == FOLDS[1:],
        "hidden0": config["authorization"]["hidden_truth_reads"] == 0,
        "sample0": config["authorization"]["sample_reads"] == 0,
        "auto_retry0": config["authorization"]["automatic_retries"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"config contract failed: {checks}")
    sources = {name: resolve_source(spec) for name, spec in config["historical_sources"].items()}
    v33c = json.loads(sources["v33c_result"].read_text(encoding="utf-8"))
    layer6 = v33c["candidate"]["full_deployment_selection"]["layer_statistics"]["6"]
    official = json.loads(sources["official_factorial_receipt"].read_text(encoding="utf-8"))
    if layer6["support"] != 767 or abs(layer6["marginal_precision"] - 0.5645371577574967) > 1e-15:
        raise ContractError("frozen layer-6 selection evidence changed")
    if official["factorial_arithmetic"]["marginals_f1_displayed"]["add_S_given_G_present"] != 0.0:
        raise ContractError("official S-neutral parent fact changed")
    return config, sources


def build_action(frame: pd.DataFrame, raw_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(raw_path, allow_pickle=False) as archive:
        incumbent = np.asarray(archive["incumbent"], dtype=np.int8)
        reference = np.asarray(archive["raw_e150"], dtype=np.int8)
    if incumbent.shape != (len(frame),) or reference.shape != (len(frame),):
        raise ContractError("historical raw arrays are misaligned")
    removal = (
        frame["station"].astype(str).eq("S-ORS").to_numpy()
        & frame["layer"].eq(6).to_numpy()
        & (incumbent == 0)
        & (reference == 1)
    )
    candidate = reference.copy()
    candidate[removal] = 0
    if np.any(candidate[incumbent == 1] == 0) or np.any(candidate > reference):
        raise ContractError("action is not an incumbent-preserving ablation")
    return incumbent, reference, candidate, removal


def metric_block(truth: np.ndarray, reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray, removal: np.ndarray) -> dict[str, Any]:
    removed = mask & removal
    ref_f1 = float(f1_score(truth[mask], reference[mask]))
    cand_f1 = float(f1_score(truth[mask], candidate[mask]))
    return {
        "rows": int(mask.sum()),
        "reference_f1": ref_f1,
        "candidate_f1": cand_f1,
        "delta_f1": cand_f1 - ref_f1,
        "removed_rows": int(removed.sum()),
        "removed_true_positives": int((removed & (truth == 1)).sum()),
        "removed_false_positives": int((removed & (truth == 0)).sum()),
    }


def evaluate(frame: pd.DataFrame, reference: np.ndarray, candidate: np.ndarray, removal: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    by_fold = {
        fold: metric_block(truth, reference, candidate, frame["fold"].eq(fold).to_numpy(), removal)
        for fold in FOLDS
    }
    all_mask = frame["fold"].isin(FOLDS).to_numpy()
    primary_mask = frame["fold"].isin(FOLDS[1:]).to_numpy()
    pooled_all = metric_block(truth, reference, candidate, all_mask, removal)
    pooled_primary = metric_block(truth, reference, candidate, primary_mask, removal)
    bootstrap = truth_source.base.day_bootstrap(frame, reference, candidate, config)
    gates = {
        "pooled_q3_q4_positive": pooled_primary["delta_f1"] > 0.0,
        "q3_nonnegative": by_fold["2025_q3"]["delta_f1"] >= 0.0,
        "q4_nonnegative": by_fold["2025_q4"]["delta_f1"] >= 0.0,
        "ci90_low_positive": bootstrap["ci90_low"] > 0.0,
    }
    return {
        "reference": "raw_E150",
        "rule": "remove S-ORS layer 6 raw-E150 additions only",
        "fit_count": 0,
        "by_fold": by_fold,
        "pooled_q2_q3_q4": pooled_all,
        "pooled_q3_q4": pooled_primary,
        "day_block_bootstrap_q3_q4": bootstrap,
        "development_expected_points_delta": pooled_primary["delta_f1"] * float(config["score"]["points_per_f1"]),
        "strict_performance_gates": gates,
        "strict_performance_pass": all(gates.values()),
        "information_value": {
            "eligible": int(removal.sum()) > 0,
            "reason": "The officially net-neutral S component is split at the single frozen historically weakest layer; this does not claim expected improvement.",
            "official_parent_component_rows": 238,
        },
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("exactly-once namespace already exists")
    started = time.perf_counter()
    config, sources = load_contract()
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "fit_count": 0,
        "retry_budget": 0,
        "official_candidate_reads": 0,
        "hidden_truth_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    write_json_new(ARTIFACT / "attempt_lock.json", lock)
    frame = pd.read_parquet(sources["anchor"], columns=[*KEYS, "fold", "current_router_prediction"])
    _, reference, candidate, removal = build_action(frame, sources["raw_seal"])
    np.savez_compressed(ARTIFACT / "sealed_action.npz", reference=reference, candidate=candidate, removal=removal)
    seal = {
        "experiment_id": EXPERIMENT_ID,
        "rows": len(frame),
        "historical_removals": int(removal.sum()),
        "reference_sha256": sha256_array(reference),
        "candidate_sha256": sha256_array(candidate),
        "removal_sha256": sha256_array(removal.astype(np.uint8)),
        "truth_reads_before_action_seal": 0,
        "official_candidate_reads": 0,
        "hidden_truth_reads": 0,
    }
    write_json_new(ARTIFACT / "action_seal.json", seal)
    historical, candidate = truth_source.attach_truth(frame, candidate)
    record = evaluate(historical, reference, candidate, removal, config)
    status = "PERFORMANCE_GATE_PASS" if record["strict_performance_pass"] else "INFORMATION_PROBE_ONLY_PERFORMANCE_GATE_FAIL"
    result = {
        "schema_version": "p1.s_ors_layer6_information_probe.result.v34a",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 0,
        "candidate": record,
        "operations": {
            "historical_truth_reads_after_seal": len(historical),
            "official_candidate_reads": 0,
            "hidden_truth_reads": 0,
            "sample_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
            "retries": 0,
        },
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "attempt_lock_sha256": sha256_file(ARTIFACT / "attempt_lock.json"),
            "action_seal_sha256": sha256_file(ARTIFACT / "action_seal.json"),
            "sealed_action_sha256": sha256_file(ARTIFACT / "sealed_action.npz"),
        },
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
        failure = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_candidate_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            write_json_new(ARTIFACT / "terminal_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
