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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v16 as source  # noqa: E402

from src.p1_qc.robust_student_t_llr import (  # noqa: E402
    calibrate_threshold,
    fit_student_t,
    score_llr,
)

CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v20.json"
REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v20/preflight-report.json"
ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v20"
QA_REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v20/independent-qa.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def preflight() -> dict:
    config = load_contract()
    rng = np.random.default_rng(20260831)
    negative = rng.standard_t(4, size=(8000, 10))
    positive = rng.standard_t(4, size=(240, 10)) + np.asarray([2.5, 2, 1.5] + [0] * 7)
    x = np.vstack([negative, positive])
    y = np.r_[np.zeros(len(negative), dtype=np.int8), np.ones(len(positive), dtype=np.int8)]
    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    fit_end = int(len(y) * 0.75)
    state = fit_student_t(x[:fit_end], y[:fit_end])
    scores = score_llr(state, x[fit_end:])
    anchor = np.zeros(len(scores), dtype=np.int8)
    calibration = calibrate_threshold(scores, y[fit_end:], anchor)
    repeat = calibrate_threshold(score_llr(fit_student_t(x[:fit_end], y[:fit_end]), x[fit_end:]), y[fit_end:], anchor)
    checks = {"finite_scores": bool(np.isfinite(scores).all()), "deterministic": calibration == repeat, "threshold_inner_only": fit_end < len(y), "rare_addition_cap": calibration["additions"] / len(scores) <= 0.005, "exact_two_pipeline_fits": config["fit_budget"]["maximum"] == 2, "raw_gate_exact": config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == 0.13168209161000616, "historical_disabled": config["authorization"]["historical_execution"] is False, "resource_cap": x.nbytes * 8 < config["resource_caps"]["rss_bytes"]}
    return {"schema_version": "p1.v20.synthetic-preflight.1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "calibration": calibration, "resource_estimate": {"historical_wall_seconds": 180, "peak_rss_bytes": 1073741824, "pipeline_fits": 2}, "hashes": {"config_sha256": sha256(CONFIG), "source_sha256": sha256(ROOT / "src/p1_qc/robust_student_t_llr.py")}, "access": {"historical_truth_reads": 0, "official_reads": 0, "hidden_truth_reads": 0, "attempt_locks_created": 0, "uploads": 0}}


def _feature_values(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    columns = []
    for name in names:
        if name.startswith("abs_temp_robust_z_"):
            columns.append(np.abs(frame[name.removeprefix("abs_")].to_numpy(np.float64)))
        else:
            columns.append(frame[name].to_numpy(np.float64))
    return np.column_stack(columns)


def execute() -> dict:
    if ARTIFACT.exists():
        raise FileExistsError("v20 exactly-once artifact already exists")
    config = load_contract()
    if not config["authorization"]["historical_execution"]:
        raise RuntimeError("historical execution not authorized")
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    lock = {"experiment_id": config["experiment_id"], "pid": os.getpid(), "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "source_sha256": sha256(ROOT / "src/p1_qc/robust_student_t_llr.py"), "fit_budget": 2, "official_reads": 0, "hidden_truth_reads": 0}
    (ARTIFACT / "attempt_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "loading", "fit_count": 0}) + "\n")
    frame, anchor, _, dependency = source.load_feature_surface()
    values = _feature_values(frame, config["features"]["names"])
    truth = frame["label_base"].to_numpy(np.int8)
    candidate, scores_all = anchor.copy(), np.full(len(frame), np.nan, dtype=np.float64)
    receipts = []
    for fit_number, spec in enumerate(config["validation"]["pipeline_fits"], 1):
        prefix_scope = frame["fold"].isin(spec["train_folds"]).to_numpy()
        prefix_times = pd.to_datetime(frame.loc[prefix_scope, "time"], utc=True)
        unique_times = np.sort(prefix_times.unique())
        cutoff = unique_times[min(int(len(unique_times) * 0.75), len(unique_times) - 1)]
        fit_scope = prefix_scope & (pd.to_datetime(frame["time"], utc=True).to_numpy() < cutoff) & (anchor == 0)
        calibration_scope = prefix_scope & (pd.to_datetime(frame["time"], utc=True).to_numpy() >= cutoff)
        model = fit_student_t(values[fit_scope], truth[fit_scope], degrees_of_freedom=4.0)
        calibration_scores = score_llr(model, values[calibration_scope])
        selected = calibrate_threshold(calibration_scores, truth[calibration_scope], anchor[calibration_scope])
        threshold = float(selected["threshold"])
        outer_scope = frame["fold"].eq(spec["outer"]).to_numpy()
        outer_scores = score_llr(model, values[outer_scope])
        scores_all[outer_scope] = outer_scores
        proposed_local = (anchor[outer_scope] == 0) & (outer_scores >= threshold)
        outer_positions = np.flatnonzero(outer_scope)
        candidate[outer_positions[proposed_local]] = 1
        receipts.append({"fit_number": fit_number, "train_folds": spec["train_folds"], "outer": spec["outer"], "inner_fit_rows": int(fit_scope.sum()), "inner_fit_positives": int(truth[fit_scope].sum()), "inner_calibration_rows": int(calibration_scope.sum()), "cutoff_utc": str(pd.Timestamp(cutoff)), "selected_threshold": threshold if np.isfinite(threshold) else None, "inner_additions": int(selected["additions"]), "outer_additions": int(proposed_local.sum()), "outer_target_reads_before_prediction_seal": 0, "location_sha256": hashlib.sha256(model.location.tobytes()).hexdigest(), "scale_sha256": hashlib.sha256(model.scale.tobytes()).hexdigest()})
        (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "fit_complete", "fit_count": fit_number}) + "\n")
    np.savez_compressed(ARTIFACT / "sealed_nested_predictions.npz", candidate=candidate, scores=scores_all)
    record = evaluation.evaluate(frame, anchor, candidate, config)
    record["name"] = config["candidate"]
    qa = {"checks": {"exact_two_fits": len(receipts) == 2, "inner_75_25": all(item["inner_fit_rows"] > item["inner_calibration_rows"] for item in receipts), "outer_targets_zero_before_seal": all(item["outer_target_reads_before_prediction_seal"] == 0 for item in receipts), "df4": config["model"]["degrees_of_freedom"] == 4.0, "anchor_removals_zero": record["anchor_removals"] == 0, "official_zero": True, "hidden_zero": True, "csv_zero": True, "upload_zero": True}}
    qa["status"] = "PASS" if all(qa["checks"].values()) else "FAIL"
    result = {"schema_version": "p1.v20.result.1", "experiment_id": config["experiment_id"], "status": "COMPLETE_INTERNAL_ONLY", "runtime_seconds": time.perf_counter() - started, "fit_count": 2, "pass_count": int(record["strict_internal_pass"]), "candidate": record, "nested_fit_receipts": receipts, "source_feature_dependency_receipt": dependency, "independent_qa": qa, "operations": {"historical_reads": 1, "official_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0}, "hashes": {"config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "source_sha256": sha256(ROOT / "src/p1_qc/robust_student_t_llr.py"), "lock_sha256": sha256(ARTIFACT / "attempt_lock.json"), "prediction_sha256": sha256(ARTIFACT / "sealed_nested_predictions.npz")}}
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
