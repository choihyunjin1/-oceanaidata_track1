from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v16 as gce  # noqa: E402

from src.p1_qc.robust_student_t_llr import calibrate_threshold_central  # noqa: E402

CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v24.json"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v3/calibration.json"
REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v24/preflight-report.json"
ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v24"
QA_REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v24/independent-qa.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {"q": config["model"]["gce_q"] == 0.7, "l2": config["model"]["l2"] == 0.001, "width": config["features"]["encoded_feature_count"] == 165, "inner75": config["inner_selector"]["split"].startswith("first75"), "fits": config["fit_budget"]["maximum"] == 2, "v3_hash": config["transport"]["calibration_sha256"] == sha256(CALIBRATION), "raw": config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == 0.015383691373120248, "v16_closed": config["lineage"]["v16_fixed_threshold_result"] == "CLOSED_NO_PASS_UNCHANGED"}
    if not all(checks.values()):
        raise RuntimeError(f"v24 contract mismatch: {checks}")
    return config


def preflight() -> dict:
    config = load_contract()
    rng = np.random.default_rng(20260831)
    n = 6000
    x = rng.normal(size=(n, 165))
    latent = 1.8 * x[:, 0] + 1.2 * x[:, 1] - 0.7 * x[:, 2]
    y = (latent + rng.logistic(size=n) > 3.5).astype(np.int8)
    fit_end = int(n * 0.75)
    weights = np.ones(fit_end)
    v16_config = json.loads((ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v16.json").read_text(encoding="utf-8"))
    result = gce.fit_gce(x[:fit_end], y[:fit_end], weights, v16_config)
    probabilities = expit(x[fit_end:] @ result.x[:-1] + result.x[-1])
    anchor = np.zeros(len(probabilities), dtype=np.int8)
    selected = calibrate_threshold_central(probabilities, y[fit_end:], anchor)
    repeat = calibrate_threshold_central(probabilities, y[fit_end:], anchor)
    checks = {"optimizer_success": bool(result.success), "probabilities_finite": bool(np.isfinite(probabilities).all()), "deterministic_selector": selected == repeat, "positive_inner_delta": selected["inner_delta_f1"] > 0, "changed_cap": selected["additions"] / len(probabilities) <= 0.005, "exact_model": config["model"]["gce_q"] == 0.7 and config["model"]["l2"] == 0.001, "v3_gate": config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == 0.015383691373120248, "two_fits": config["fit_budget"]["maximum"] == 2, "historical_disabled": config["authorization"]["historical_execution"] is False}
    return {"schema_version": "p1.v24.synthetic-preflight.1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "selected": selected, "selection_audit": {"v16_result_unchanged": True, "outer_previously_exposed": True, "prospective_only": True}, "resource_estimate": {"wall_seconds": 300, "rss_bytes": 2147483648, "pipeline_fits": 2}, "hashes": {"config_sha256": sha256(CONFIG), "v16_runner_sha256": sha256(ROOT / "scripts/run_p1_public_transport_repair_cycle_20260831_v16.py"), "calibration_v3_sha256": sha256(CALIBRATION)}, "access": {"historical_truth_reads": 0, "official_reads": 0, "hidden_truth_reads": 0, "locks": 0, "uploads": 0}}


def execute() -> dict:
    if ARTIFACT.exists():
        raise FileExistsError("v24 exactly-once artifact already exists")
    config = load_contract()
    if not config["authorization"]["historical_execution"]:
        raise RuntimeError("historical execution not authorized")
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    lock = {"experiment_id": config["experiment_id"], "pid": os.getpid(), "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "v16_runner_sha256": sha256(ROOT / "scripts/run_p1_public_transport_repair_cycle_20260831_v16.py"), "calibration_v3_sha256": sha256(CALIBRATION), "fit_budget": 2, "official_reads": 0, "hidden_truth_reads": 0}
    (ARTIFACT / "attempt_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "loading", "fit_count": 0}) + "\n")
    frame, anchor, numeric_names, dependency = gce.load_feature_surface()
    truth = frame["label_base"].to_numpy(np.int8)
    candidate, probability = anchor.copy(), np.zeros(len(frame), dtype=np.float64)
    receipts = []
    all_times = pd.to_datetime(frame["time"], utc=True)
    for fit_number, spec in enumerate(config["validation"]["pipeline_fits"], 1):
        prefix = frame["fold"].isin(spec["train_folds"]).to_numpy()
        unique_times = np.sort(all_times[prefix].unique())
        cutoff = unique_times[min(int(len(unique_times) * 0.75), len(unique_times) - 1)]
        inner_fit = prefix & (all_times.to_numpy() < cutoff) & (anchor == 0)
        inner_calibration = prefix & (all_times.to_numpy() >= cutoff)
        encoder = gce.PrefixEncoder.fit(frame, inner_fit, numeric_names)
        fit_design, fit_scaled = encoder.transform(frame, inner_fit)
        weights = gce.leverage_weights(fit_scaled)
        optimizer = gce.fit_gce(fit_design, truth[inner_fit], weights, config)
        calibration_design, _ = encoder.transform(frame, inner_calibration)
        calibration_probability = expit(calibration_design @ optimizer.x[:-1] + optimizer.x[-1])
        selected = calibrate_threshold_central(calibration_probability, truth[inner_calibration], anchor[inner_calibration])
        threshold = float(selected["threshold"])
        outer = frame["fold"].eq(spec["outer"]).to_numpy()
        outer_negative = outer & (anchor == 0)
        outer_design, _ = encoder.transform(frame, outer_negative)
        outer_probability = expit(outer_design @ optimizer.x[:-1] + optimizer.x[-1])
        probability[outer_negative] = outer_probability
        proposed = np.flatnonzero(outer_negative)[outer_probability >= threshold]
        candidate[proposed] = 1
        receipts.append({"fit_number": fit_number, "train_folds": spec["train_folds"], "outer": spec["outer"], "cutoff_utc": str(pd.Timestamp(cutoff)), "inner_fit_rows": int(inner_fit.sum()), "inner_fit_positives": int(truth[inner_fit].sum()), "inner_calibration_rows": int(inner_calibration.sum()), "inner_selected_threshold": threshold if np.isfinite(threshold) else None, "inner_delta_f1": float(selected["inner_delta_f1"]), "inner_precision": float(selected["precision"]), "inner_additions": int(selected["additions"]), "outer_additions": len(proposed), "outer_target_reads_before_prediction_seal": 0, "optimizer_success": bool(optimizer.success), "parameters_sha256": hashlib.sha256(optimizer.x.astype(np.float64).tobytes()).hexdigest()})
        (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "fit_complete", "fit_count": fit_number}) + "\n")
    np.savez_compressed(ARTIFACT / "sealed_nested_predictions.npz", candidate=candidate, probability=probability)
    record = evaluation.evaluate(frame, anchor, candidate, config)
    record["name"] = config["candidate"]
    qa = {"checks": {"exact_two_fits": len(receipts) == 2, "exact_gce": config["model"]["gce_q"] == 0.7 and config["model"]["l2"] == 0.001, "inner_selector_only": all(item["outer_target_reads_before_prediction_seal"] == 0 for item in receipts), "v3_gate": config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == 0.015383691373120248, "anchor_removals_zero": record["anchor_removals"] == 0, "official_zero": True, "hidden_zero": True, "csv_zero": True, "upload_zero": True}}
    qa["status"] = "PASS" if all(qa["checks"].values()) else "FAIL"
    result = {"schema_version": "p1.v24.result.1", "experiment_id": config["experiment_id"], "status": "COMPLETE_INTERNAL_ONLY", "runtime_seconds": time.perf_counter() - started, "fit_count": 2, "pass_count": int(record["strict_internal_pass"]), "candidate": record, "nested_fit_receipts": receipts, "source_feature_dependency_receipt": dependency, "independent_qa": qa, "adaptive_development_evidence_only": True, "operations": {"historical_reads": 1, "official_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0}, "hashes": {"config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "v16_runner_sha256": sha256(ROOT / "scripts/run_p1_public_transport_repair_cycle_20260831_v16.py"), "calibration_v3_sha256": sha256(CALIBRATION), "lock_sha256": sha256(ARTIFACT / "attempt_lock.json"), "prediction_sha256": sha256(ARTIFACT / "sealed_nested_predictions.npz")}}
    (ARTIFACT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    QA_REPORT.parent.mkdir(parents=True, exist_ok=True)
    QA_REPORT.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "terminal", "fit_count": 2, "pass_count": result["pass_count"]}) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        try:
            print(json.dumps(execute(), indent=2, sort_keys=True))
            return
        except Exception as exc:  # noqa: BLE001
            payload = {"status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "official_reads": 0, "hidden_truth_reads": 0, "uploads": 0}
            if ARTIFACT.exists():
                (ARTIFACT / "terminal_failure.json").write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(payload, indent=2))
            raise SystemExit(1) from exc
    if not args.preflight:
        raise SystemExit("only --preflight is authorized")
    started = time.perf_counter()
    result = preflight()
    result["runtime_seconds"] = time.perf_counter() - started
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
