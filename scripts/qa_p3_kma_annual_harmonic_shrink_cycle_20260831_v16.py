"""Independent QA for the P3 annual-harmonic smooth-shrink cycle."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

from run_p3_kma_annual_harmonic_shrink_cycle_20260831_v16 import (  # noqa: E402
    ARTIFACT_DIR,
    CALIBRATION,
    CALIBRATION_SHA,
    CONFIG,
    EXPERIMENT_ID,
    LOCK,
    REPORT_DIR,
    attach_energy,
    evaluate,
    load_energy_history,
    load_historical,
    sha256,
)
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    canonical,
    write_new,
)


def main() -> int:
    result_path = ARTIFACT_DIR / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    frame, _ = load_historical()
    history = load_energy_history()
    recomputed = evaluate(attach_energy(frame, history), history)
    candidate = result["candidate"]
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    runner = Path(__file__).with_name("run_p3_kma_annual_harmonic_shrink_cycle_20260831_v16.py")
    failed = {key for key, value in candidate["gate_checks"].items() if not value}
    checks = {
        "terminal_complete": result["status"] == "COMPLETE",
        "runner_hash_matches": sha256(runner) == lock["runner_sha256"] == result["provenance"]["runner_sha256"],
        "config_hash_matches": sha256(CONFIG) == result["provenance"]["config_sha256"],
        "calibration_hash_frozen": sha256(CALIBRATION) == CALIBRATION_SHA == result["provenance"]["calibration_sha256"],
        "delta_recomputed": np.isclose(candidate["delta_candidate_minus_reference_rmse_m"], recomputed["delta_candidate_minus_reference_rmse_m"], atol=1e-15),
        "episode_ci_recomputed": np.allclose(candidate["episode_bootstrap"]["ci90_m"], recomputed["episode_bootstrap"]["ci90_m"], atol=1e-15),
        "group_ci_recomputed": np.allclose(candidate["block_station_bootstrap"]["ci90_m"], recomputed["block_station_bootstrap"]["ci90_m"], atol=1e-15),
        "gate_recomputed": candidate["gate_checks"] == recomputed["gate_checks"],
        "only_transport_lcb_gates_failed": failed == {"raw_lcb_points_meets_family_threshold", "calibrated_lcb_at_least_0p01"},
        "six_solves_zero_models": result["fit_budget"]["prefix_ridge_solves"] == 6 and result["fit_budget"]["model_fits"] == 0,
        "early_insufficient_phase_abstained": all(item["theta_c"] == 0 and item["theta_d"] == 0 for item in candidate["calibration_receipts"][:3]),
        "outer_truth_sealed": all(item["outer_target_rows_read_before_theta_fixed"] == 0 for item in candidate["calibration_receipts"]),
        "stability_gates_passed": all(candidate["gate_checks"][key] for key in ["pooled_rmse_improves", "minimum_four_improved_bimonth_blocks", "episode_ci90_upper_below_zero", "block_station_ci90_upper_below_zero", "worst_station_lead_within_0p01m"]),
        "no_submission": result["outputs"] == [],
        "official_hidden_upload_zero": all(value == 0 for value in result["data_access"].values()),
        "no_result_based_tuning": not result["execution"]["result_based_tuning_or_retry"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "schema_version": "p3.kma_annual_harmonic_shrink.independent_qa.v16",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "result_sha256": sha256(result_path),
    }
    output = REPORT_DIR / "independent-qa.json"
    if output.exists():
        raise RuntimeError("independent QA artifact already exists")
    write_new(output, canonical(payload))
    print(json.dumps({"status": payload["status"], "checks": f"{payload['passed_checks']}/{payload['total_checks']}"}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
