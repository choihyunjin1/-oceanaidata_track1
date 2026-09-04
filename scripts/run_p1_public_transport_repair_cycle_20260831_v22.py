from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_p1_public_transport_repair_cycle_20260831_v20 as base  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v20r1 as repair  # noqa: E402

from src.p1_qc.robust_student_t_llr import (  # noqa: E402
    calibrate_threshold_central,
    fit_student_t,
    score_llr,
)

CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v22.json"
REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v22/preflight-report.json"
ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v22"
QA_REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v22/independent-qa.json"
BASE_CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v20.json"


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
    scores = score_llr(fit_student_t(x[:fit_end], y[:fit_end], 4.0), x[fit_end:])
    anchor = np.zeros(len(scores), dtype=np.int8)
    selected = calibrate_threshold_central(scores, y[fit_end:], anchor)
    repeat = calibrate_threshold_central(scores, y[fit_end:], anchor)
    checks = {"finite": bool(np.isfinite(scores).all()), "deterministic": selected == repeat, "positive_inner_delta": selected["inner_delta_f1"] > 0, "central_precision_positive": selected["precision"] > 0, "changed_cap": selected["additions"] / len(scores) <= 0.005, "wilson_removed_only_inner": config["inner_selector"]["wilson_lcb_gate"] is False, "outer_gate_unchanged": config["decision_policy"] == {"minimum_raw_expected_point_delta_inclusive": 0.13168209161000616, "minimum_calibrated_expected_point_delta_inclusive": 0.01}, "two_fits": config["fit_budget"]["maximum"] == 2, "historical_disabled": config["authorization"]["historical_execution"] is False}
    return {"schema_version": "p1.v22.synthetic-preflight.1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "selected": selected, "selection_bias_audit": {"inner_threshold_candidates_data_dependent": True, "outer_separation_required": True, "outer_surface_previously_exposed": True, "promotion_evidence_if_run": "adaptive development evidence only"}, "resource_estimate": {"wall_seconds": 180, "rss_bytes": 1073741824, "vram_bytes": 0}, "hashes": {"config_sha256": sha256(CONFIG), "source_sha256": sha256(ROOT / "src/p1_qc/robust_student_t_llr.py")}, "access": {"historical_truth_reads": 0, "official_reads": 0, "hidden_truth_reads": 0, "locks": 0, "uploads": 0}}


def execution_contract() -> dict:
    sealed = load_contract()
    contract = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    checks = {
        "df4": contract["model"]["degrees_of_freedom"] == sealed["model"]["degrees_of_freedom"] == 4.0,
        "features": contract["features"]["names"] == sealed["features"]["names"],
        "inner75": contract["inner_calibration"]["fit_fraction"] == 0.75,
        "fits2": sealed["fit_budget"]["maximum"] == contract["fit_budget"]["maximum"] == 2,
        "raw_gate": sealed["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == contract["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
        "central_selector": sealed["inner_selector"]["wilson_lcb_gate"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v22 execution contract drift: {checks}")
    contract["experiment_id"] = sealed["experiment_id"]
    contract["candidate"] = sealed["candidate"]
    contract["status"] = sealed["status"]
    return contract


def execute() -> dict:
    schema = repair.historical_schema_preflight()
    if schema["status"] != "PASS":
        raise RuntimeError(f"historical schema preflight failed: {schema}")
    original_sha = base.sha256

    def runner_sha(path: Path) -> str:
        return sha256(Path(__file__)) if Path(path).resolve() == Path(base.__file__).resolve() else original_sha(path)

    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.QA_REPORT = QA_REPORT
    base.load_contract = execution_contract
    base._feature_values = repair.repaired_feature_values
    base.calibrate_threshold = calibrate_threshold_central
    base.sha256 = runner_sha
    result = base.execute()
    result["selection_contract"] = {"inner_wilson_lcb": False, "inner_delta_strict_positive": True, "central_precision_above_anchor_half": True, "changed_fraction_max": 0.005, "tie_break": "higher threshold then fewer additions", "outer_labels_used_for_selection": 0, "adaptive_development_evidence_only": True, "schema_preflight": schema}
    (ARTIFACT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
            import traceback

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
