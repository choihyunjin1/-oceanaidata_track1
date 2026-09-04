"""Independent QA for the P3 continuous-wave-power KMA factor v19."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

from run_p3_kma_continuous_wave_power_factor_cycle_20260831_v19 import (  # noqa: E402
    ARTIFACT_DIR,
    CALIBRATION,
    CALIBRATION_SHA,
    CONFIG,
    DUPLICATION_AUDIT,
    EXPERIMENT_ID,
    LOCK,
    REPORT_DIR,
    attach_wave_power,
    canonical,
    evaluate,
    load_historical,
    load_wave_power_history,
    sha256,
    write_new,
)


def main() -> int:
    result_path = ARTIFACT_DIR / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    frame, _ = load_historical()
    history = load_wave_power_history()
    recomputed = evaluate(attach_wave_power(frame, history), history)
    candidate = result["candidate"]
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    runner = Path(__file__).with_name("run_p3_kma_continuous_wave_power_factor_cycle_20260831_v19.py")
    checks = {
        "terminal_complete": result["status"] == "COMPLETE",
        "runner_hash_matches": sha256(runner) == lock["runner_sha256"] == result["provenance"]["runner_sha256"],
        "config_hash_matches": sha256(CONFIG) == result["provenance"]["config_sha256"],
        "calibration_hash_matches": sha256(CALIBRATION) == CALIBRATION_SHA == result["provenance"]["calibration_sha256"],
        "duplication_audit_hash_matches": sha256(DUPLICATION_AUDIT) == result["provenance"]["duplication_audit_sha256"],
        "delta_recomputed": np.isclose(candidate["delta_candidate_minus_reference_rmse_m"], recomputed["delta_candidate_minus_reference_rmse_m"], atol=1e-15),
        "episode_ci_recomputed": np.allclose(candidate["episode_bootstrap"]["ci90_m"], recomputed["episode_bootstrap"]["ci90_m"], atol=1e-15),
        "group_ci_recomputed": np.allclose(candidate["block_station_bootstrap"]["ci90_m"], recomputed["block_station_bootstrap"]["ci90_m"], atol=1e-15),
        "gate_recomputed": candidate["gate_checks"] == recomputed["gate_checks"],
        "central_transport_gate_only": "central_raw_points_meets_family_threshold_inclusive" in candidate["gate_checks"],
        "six_ecdf_zero_models": result["fit_budget"]["ecdf_calibration_fits"] == 6 and result["fit_budget"]["model_fits"] == 0,
        "surface_not_independent": candidate["surface_claim"] == "adaptive_182_case_development_surface_not_independent_confirmation",
        "no_result_based_tuning": not result["execution"]["result_based_tuning_or_retry"],
        "formula_not_retuned": not result["execution"]["formula_or_orientation_or_span_retuned"],
        "hidden_upload_zero": result["data_access"]["hidden_truth_rows_read"] == 0 and result["data_access"]["uploads"] == 0,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "schema_version": "p3.kma_continuous_wave_power.independent_qa.v19",
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
