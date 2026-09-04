"""Independent QA for the sealed P3 continuous-energy KMA factor cycle."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

from run_p3_kma_continuous_energy_factor_cycle_20260831_v14b import (  # noqa: E402
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
    checks = {
        "terminal_complete": result["status"] == "COMPLETE",
        "runner_hash_matches_lock_and_result": sha256(Path(__file__).with_name("run_p3_kma_continuous_energy_factor_cycle_20260831_v14b.py"))
        == result["provenance"]["runner_sha256"]
        == json.loads(LOCK.read_text(encoding="utf-8"))["runner_sha256"],
        "config_hash_matches_result": sha256(CONFIG) == result["provenance"]["config_sha256"],
        "calibration_hash_frozen": sha256(CALIBRATION) == CALIBRATION_SHA == result["provenance"]["calibration_sha256"],
        "delta_rmse_recomputed": np.isclose(candidate["delta_candidate_minus_reference_rmse_m"], recomputed["delta_candidate_minus_reference_rmse_m"], atol=1e-15),
        "episode_ci_recomputed": np.allclose(candidate["episode_bootstrap"]["ci90_m"], recomputed["episode_bootstrap"]["ci90_m"], atol=1e-15),
        "block_station_ci_recomputed": np.allclose(candidate["block_station_bootstrap"]["ci90_m"], recomputed["block_station_bootstrap"]["ci90_m"], atol=1e-15),
        "gate_recomputed": candidate["gate_checks"] == recomputed["gate_checks"],
        "six_ecdf_calibration_fits": result["fit_budget"]["ecdf_calibration_fits"] == 6 and result["fit_budget"]["model_fits"] == 0,
        "every_prefix_precedes_boundary": all(item["prefix_max_anchor_time_utc"] <= item["prefix_end_boundary_utc"] for item in candidate["ecdf_calibration_receipts"]),
        "short_leads_bit_exact": candidate["short_leads_bit_exact"],
        "changed_share_within_one_third": candidate["changed_rows_share"] <= 1.0 / 3.0 + 1e-12,
        "decision_matches_zero_pass": result["decision"] == "NO_GO_ENERGY_KMA_DIRECTION_CLOSED" and result["passing_candidate_count"] == 0,
        "no_submission_materialized": result["outputs"] == [],
        "official_hidden_upload_zero": all(value == 0 for value in result["data_access"].values()),
        "no_result_based_tuning_or_retry": not result["execution"]["result_based_tuning_or_retry"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "schema_version": "p3.kma_continuous_energy_factor.independent_qa.v14b",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "result_sha256": sha256(result_path),
        "official_hidden_upload_rows": 0,
    }
    output = REPORT_DIR / "independent-qa.json"
    if output.exists():
        raise RuntimeError("independent QA artifact already exists")
    write_new(output, canonical(payload))
    print(json.dumps({"status": payload["status"], "checks": f"{payload['passed_checks']}/{payload['total_checks']}"}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
