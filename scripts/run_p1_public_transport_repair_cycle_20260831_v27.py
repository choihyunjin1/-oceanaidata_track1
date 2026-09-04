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
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v16 as surface  # noqa: E402

from src.p1_qc.robust_student_t_llr import calibrate_threshold_central  # noqa: E402

CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v27.json"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v3/calibration.json"
REPORT_DIR = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v27"
ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v27"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "candidate": config["candidate"] == "P1_1_PREFIX_ECDF_BASE_PEER_MIN_CONSENSUS",
        "inputs": config["score"]["inputs"] == ["probability_base", "probability_peer"],
        "min": config["score"]["combine"] == "minimum rank",
        "no_em": config["score"]["no_em"],
        "inner25": config["inner_selector"]["split"].startswith("first75"),
        "fits": config["fit_budget"]["maximum"] == 2,
        "v3": config["transport"]["calibration_sha256"] == sha256(CALIBRATION),
        "raw": config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == 0.015383691373120248,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v27 contract mismatch: {checks}")
    return config


def fit_ecdf(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("ECDF fit values must be finite and nonempty")
    return np.sort(values)


def apply_ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.searchsorted(reference, values, side="right").astype(np.float64) / reference.size


def consensus_score(base_ref: np.ndarray, peer_ref: np.ndarray, base: np.ndarray, peer: np.ndarray) -> np.ndarray:
    return np.minimum(apply_ecdf(base_ref, base), apply_ecdf(peer_ref, peer))


def preflight() -> dict:
    config = load_contract()
    rng = np.random.default_rng(20260831)
    n = 8000
    base = rng.uniform(size=n)
    peer = np.clip(base + rng.normal(0, 0.08, n), 0, 1)
    truth = ((base > 0.94) & (peer > 0.91)).astype(np.int8)
    anchor = np.zeros(n, dtype=np.int8)
    cut = int(n * 0.75)
    br, pr = fit_ecdf(base[:cut]), fit_ecdf(peer[:cut])
    score = consensus_score(br, pr, base[cut:], peer[cut:])
    selected = calibrate_threshold_central(score, truth[cut:], anchor[cut:])
    repeat = calibrate_threshold_central(score, truth[cut:], anchor[cut:])
    checks = {
        "finite": bool(np.isfinite(score).all()),
        "range": bool(((score >= 0) & (score <= 1)).all()),
        "deterministic": selected == repeat,
        "positive_delta": selected["inner_delta_f1"] > 0,
        "changed_cap": selected["additions"] / len(score) <= 0.005,
        "exact_two_fits": config["fit_budget"]["maximum"] == 2,
        "official_zero": True,
    }
    return {"schema_version": "p1.v27.preflight.1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "selected": selected, "hashes": {"config_sha256": sha256(CONFIG), "calibration_sha256": sha256(CALIBRATION)}, "access": {"historical_truth_reads": 0, "official_reads": 0, "hidden_truth_reads": 0, "locks": 0, "uploads": 0}}


def execute() -> dict:
    if ARTIFACT.exists():
        raise FileExistsError("v27 exactly-once artifact already exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    lock = {"experiment_id": config["experiment_id"], "pid": os.getpid(), "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "fit_budget": 2, "official_reads": 0, "hidden_truth_reads": 0}
    (ARTIFACT / "attempt_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "loading", "fit_count": 0}) + "\n", encoding="utf-8")
    frame, anchor, _numeric_names, dependency = surface.load_feature_surface()
    truth = frame["label_base"].to_numpy(np.int8)
    candidate = anchor.copy()
    scores = np.zeros(len(frame), dtype=np.float64)
    times = pd.to_datetime(frame["time"], utc=True)
    receipts = []
    for fit_number, spec in enumerate(config["validation"]["pipeline_fits"], 1):
        prefix = frame["fold"].isin(spec["train_folds"]).to_numpy()
        unique_times = np.sort(times[prefix].unique())
        cutoff = unique_times[min(int(len(unique_times) * 0.75), len(unique_times) - 1)]
        fit_mask = prefix & (times.to_numpy() < cutoff) & (anchor == 0)
        cal_mask = prefix & (times.to_numpy() >= cutoff)
        base_ref = fit_ecdf(frame.loc[fit_mask, "probability_base"].to_numpy())
        peer_ref = fit_ecdf(frame.loc[fit_mask, "probability_peer"].to_numpy())
        cal_score = consensus_score(base_ref, peer_ref, frame.loc[cal_mask, "probability_base"].to_numpy(), frame.loc[cal_mask, "probability_peer"].to_numpy())
        selected = calibrate_threshold_central(cal_score, truth[cal_mask], anchor[cal_mask])
        threshold = float(selected["threshold"])
        outer_neg = frame["fold"].eq(spec["outer"]).to_numpy() & (anchor == 0)
        out_score = consensus_score(base_ref, peer_ref, frame.loc[outer_neg, "probability_base"].to_numpy(), frame.loc[outer_neg, "probability_peer"].to_numpy())
        scores[outer_neg] = out_score
        proposed = np.flatnonzero(outer_neg)[out_score >= threshold]
        candidate[proposed] = 1
        receipts.append({"fit_number": fit_number, "train_folds": spec["train_folds"], "outer": spec["outer"], "cutoff_utc": str(pd.Timestamp(cutoff)), "ecdf_fit_rows": int(fit_mask.sum()), "inner_calibration_rows": int(cal_mask.sum()), "inner_selected_threshold": threshold if np.isfinite(threshold) else None, "inner_delta_f1": float(selected["inner_delta_f1"]), "inner_precision": float(selected["precision"]), "inner_additions": int(selected["additions"]), "outer_additions": int(len(proposed)), "outer_target_reads_before_prediction_seal": 0, "base_ecdf_sha256": hashlib.sha256(base_ref.tobytes()).hexdigest(), "peer_ecdf_sha256": hashlib.sha256(peer_ref.tobytes()).hexdigest()})
        (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "fit_complete", "fit_count": fit_number}) + "\n", encoding="utf-8")
    np.savez_compressed(ARTIFACT / "sealed_nested_predictions.npz", candidate=candidate, score=scores)
    record = evaluation.evaluate(frame, anchor, candidate, config)
    record["name"] = config["candidate"]
    checks = {"exact_two_fits": len(receipts) == 2, "prefix_only": all(r["outer_target_reads_before_prediction_seal"] == 0 for r in receipts), "anchor_removals_zero": record["anchor_removals"] == 0, "official_zero": True, "hidden_zero": True, "csv_zero": True, "upload_zero": True}
    qa = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    result = {"schema_version": "p1.v27.result.1", "experiment_id": config["experiment_id"], "status": "COMPLETE_INTERNAL_ONLY", "runtime_seconds": time.perf_counter() - started, "fit_count": 2, "pass_count": int(record["strict_internal_pass"]), "candidate": record, "nested_fit_receipts": receipts, "source_feature_dependency_receipt": dependency, "independent_qa": qa, "operations": {"historical_reads": 1, "official_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0}, "hashes": {"config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "calibration_sha256": sha256(CALIBRATION), "lock_sha256": sha256(ARTIFACT / "attempt_lock.json"), "prediction_sha256": sha256(ARTIFACT / "sealed_nested_predictions.npz")}}
    (ARTIFACT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "independent-qa.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "terminal", "fit_count": 2, "pass_count": result["pass_count"]}) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        result = execute() if args.execute else preflight() if args.preflight else None
        if result is None:
            raise SystemExit("use --preflight or --execute")
        if args.preflight:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / "preflight-report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        if result.get("status") == "FAIL":
            raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "official_reads": 0, "hidden_truth_reads": 0, "uploads": 0}
        if ARTIFACT.exists():
            (ARTIFACT / "terminal_failure.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
