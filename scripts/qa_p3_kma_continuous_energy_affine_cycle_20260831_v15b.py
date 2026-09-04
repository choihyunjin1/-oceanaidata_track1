"""Independent QA for P3 prefix-ridge continuous-energy affine v15b."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

from run_p3_kma_continuous_energy_affine_cycle_20260831_v15b import (  # noqa: E402
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
    runner_path = Path(__file__).with_name(
        "run_p3_kma_continuous_energy_affine_cycle_20260831_v15b.py"
    )
    checks = {
        "terminal_complete": result["status"] == "COMPLETE",
        "runner_hash_matches": sha256(runner_path) == lock["runner_sha256"] == result["provenance"]["runner_sha256"],
        "config_hash_matches": sha256(CONFIG) == result["provenance"]["config_sha256"],
        "calibration_hash_frozen": sha256(CALIBRATION) == CALIBRATION_SHA == result["provenance"]["calibration_sha256"],
        "delta_rmse_recomputed": np.isclose(candidate["delta_candidate_minus_reference_rmse_m"], recomputed["delta_candidate_minus_reference_rmse_m"], atol=1e-15),
        "episode_ci_recomputed": np.allclose(candidate["episode_bootstrap"]["ci90_m"], recomputed["episode_bootstrap"]["ci90_m"], atol=1e-15),
        "block_station_ci_recomputed": np.allclose(candidate["block_station_bootstrap"]["ci90_m"], recomputed["block_station_bootstrap"]["ci90_m"], atol=1e-15),
        "gate_recomputed": candidate["gate_checks"] == recomputed["gate_checks"],
        "fit_count_12_and_models_0": result["fit_budget"]["coefficient_calibration_fits"] == 12 and result["fit_budget"]["model_fits"] == 0,
        "first_outer_prior_only": candidate["calibration_receipts"][0]["train_rows"] == 0,
        "all_outer_truth_sealed_before_theta": all(item["outer_target_rows_read_before_theta_fixed"] == 0 for item in candidate["calibration_receipts"]),
        "changed_share_within_one_third": candidate["changed_rows_share"] <= 1.0 / 3.0 + 1e-12,
        "decision_matches_zero_pass": result["decision"] == "NO_GO_PREFIX_RIDGE_AFFINE_GATE" and result["passing_candidate_count"] == 0,
        "no_submission": result["outputs"] == [],
        "official_hidden_upload_zero": all(value == 0 for value in result["data_access"].values()),
        "no_result_based_tuning_or_retry": not result["execution"]["result_based_tuning_or_retry"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "schema_version": "p3.kma_continuous_energy_affine.independent_qa.v15b",
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
