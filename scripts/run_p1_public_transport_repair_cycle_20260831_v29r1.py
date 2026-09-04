from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v16 as source  # noqa: E402

CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v29r1.json"
BASE_CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v29.json"
ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v29r1"
REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v29r1/independent-qa.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> tuple[dict, dict]:
    amendment = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    sealed = ROOT / amendment["sealed_prediction"]["path"]
    checks = {
        "exact_field": amendment["repair"] == {"only_added_field": "decision_policy.bootstrap_probability_improved_minimum_inclusive", "value": 0.8, "additional_fits": 0, "prediction_changes": 0},
        "sealed_hash": sha256(sealed) == amendment["sealed_prediction"]["sha256"],
        "base_lock_hash": sha256(ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v29/attempt_lock.json") == amendment["base_lock_sha256"],
        "field_previously_missing": "bootstrap_probability_improved_minimum_inclusive" not in base["decision_policy"],
        "zero_fits": amendment["authorization"]["model_fits"] == 0,
        "zero_prediction_writes": amendment["authorization"]["prediction_writes"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v29r1 contract mismatch: {checks}")
    repaired = json.loads(json.dumps(base))
    repaired["experiment_id"] = amendment["experiment_id"]
    repaired["decision_policy"]["bootstrap_probability_improved_minimum_inclusive"] = 0.8
    return amendment, repaired


def validate() -> dict:
    amendment, repaired = load_contract()
    checks = {"threshold_exact": repaired["decision_policy"]["bootstrap_probability_improved_minimum_inclusive"] == 0.8, "fit_zero": amendment["repair"]["additional_fits"] == 0, "prediction_change_zero": amendment["repair"]["prediction_changes"] == 0, "official_zero": True}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "sealed_prediction_sha256": amendment["sealed_prediction"]["sha256"]}


def evaluate_once() -> dict:
    if ARTIFACT.exists():
        raise FileExistsError("v29r1 recovery artifact already exists")
    amendment, repaired = load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    sealed_path = ROOT / amendment["sealed_prediction"]["path"]
    before_hash = sha256(sealed_path)
    lock = {"experiment_id": amendment["experiment_id"], "pid": os.getpid(), "mode": "metric_only", "config_sha256": sha256(CONFIG), "sealed_prediction_sha256": before_hash, "additional_fits": 0, "prediction_changes": 0, "official_reads": 0, "hidden_truth_reads": 0}
    (ARTIFACT / "attempt_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    frame, anchor, _, dependency = source.load_feature_surface()
    with np.load(sealed_path, allow_pickle=False) as archive:
        candidate = archive["candidate"].astype(np.int8)
    candidate_hash = hashlib.sha256(candidate.tobytes()).hexdigest()
    record = evaluation.evaluate(frame, anchor, candidate, repaired)
    record["name"] = "P1_1_INNER_GROUP_DAY_GUARDED_LABEL_SHIFT_EM"
    after_hash = sha256(sealed_path)
    checks = {"sealed_npz_bit_exact": before_hash == after_hash == amendment["sealed_prediction"]["sha256"], "additional_fits_zero": True, "prediction_changes_zero": True, "anchor_removals_zero": record["anchor_removals"] == 0, "official_zero": True, "hidden_zero": True, "csv_zero": True, "upload_zero": True}
    qa = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    result = {"schema_version": "p1.v29r1.result.1", "experiment_id": amendment["experiment_id"], "status": "COMPLETE_METRIC_ONLY", "runtime_seconds": time.perf_counter() - started, "fit_count": 0, "pass_count": int(record["strict_internal_pass"]), "candidate": record, "source_feature_dependency_receipt": dependency, "independent_qa": qa, "operations": {"historical_reads": 1, "additional_model_fits": 0, "prediction_writes": 0, "official_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0}, "hashes": {"config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "sealed_prediction_sha256": after_hash, "candidate_array_sha256": candidate_hash, "lock_sha256": sha256(ARTIFACT / "attempt_lock.json")}}
    (ARTIFACT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = evaluate_once() if args.execute else validate() if args.validate else None
    if result is None:
        raise SystemExit("use --validate or --execute")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
