"""Independent QA for v18r1 evaluation-only recovery."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

from run_p3_kma_v14b_v16_midpoint_cycle_20260831_v18 import (  # noqa: E402
    attach_energy,
    evaluate,
    load_energy_history,
    load_historical,
    sha256,
)
from run_p3_kma_v14b_v16_midpoint_cycle_20260831_v18r1 import (  # noqa: E402
    ARTIFACT_DIR,
    CONFIG,
    EXPERIMENT_ID,
    LOCK,
    REPORT_DIR,
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
        "internal_pass": result["decision"] == "PASS_INTERNAL_AWAITING_SEPARATE_MATERIALIZER" and result["passing_candidate_count"] == 1,
        "delta_recomputed": np.isclose(candidate["delta_candidate_minus_reference_rmse_m"], recomputed["delta_candidate_minus_reference_rmse_m"], atol=1e-15),
        "episode_ci_recomputed": np.allclose(candidate["episode_bootstrap"]["ci90_m"], recomputed["episode_bootstrap"]["ci90_m"], atol=1e-15),
        "group_ci_recomputed": np.allclose(candidate["block_station_bootstrap"]["ci90_m"], recomputed["block_station_bootstrap"]["ci90_m"], atol=1e-15),
        "gate_recomputed": candidate["gate_checks"] == recomputed["gate_checks"],
        "config_hash_matches": sha256(CONFIG) == result["provenance"]["config_sha256"],
        "lock_runner_hash_matches": json.loads(LOCK.read_text(encoding="utf-8"))["runner_sha256"] == result["provenance"]["runner_sha256"],
        "official_hidden_upload_zero": all(value == 0 for value in result["data_access"].values()),
        "no_outputs": result["outputs"] == [],
        "candidate_or_gate_unchanged": not result["execution"]["candidate_or_gate_changed"],
        "no_result_based_tuning": not result["execution"]["result_based_tuning_or_retry"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {"schema_version": "p3.midpoint_recovery.independent_qa.v18r1", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "passed_checks": sum(checks.values()), "total_checks": len(checks), "result_sha256": sha256(result_path)}
    output = REPORT_DIR / "independent-qa.json"
    if output.exists():
        raise RuntimeError("QA artifact exists")
    write_new(output, canonical(payload))
    print(json.dumps({"status": payload["status"], "checks": f"{payload['passed_checks']}/{payload['total_checks']}"}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
