"""Independent QA for P3 v76 moist-air momentum zero-fit closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_moist_air_momentum_residual_cycle_20260901_v76"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RUNNER = ROOT / "scripts/run_p3_moist_air_momentum_residual_cycle_20260901_v76.py"
LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
RESULT = ROOT / "artifacts" / EXPERIMENT_ID / "result.json"
REPORT = ROOT / "reports" / EXPERIMENT_ID


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    support = result["support_receipt"]
    checks = {
        "terminal_complete": result["status"] == "COMPLETE",
        "decision_zero_fit": result["decision"] == "STOP_SUPPORT_GATE_ZERO_FIT",
        "fit_count_zero": result["fit_count"] == 0,
        "support_failed": support["passed"] is False,
        "density_below_sealed_gate": support["density_min_kg_m3"]
        < config["encoder"]["support_gate"]["minimum_density_kg_m3"],
        "feature_shape": support["rows"] == 182 and support["columns"] == 48,
        "all_features_vary": support["positive_variance_features"] == 48,
        "target_free_gate": support["target_used"] is False
        and result["data_access"]["historical_target_rows"] == 0,
        "config_hash": sha256(CONFIG) == result["provenance"]["config_sha256"],
        "runner_hash": sha256(RUNNER) == result["provenance"]["runner_sha256"],
        "lock_consumed": lock["status"] == "ATTEMPT_CONSUMED_ONE_SHOT",
        "official_hidden_csv_upload_zero": all(
            result["data_access"][key] == 0
            for key in (
                "official_test_rows",
                "official_sample_rows",
                "official_submission_rows",
                "hidden_truth_rows",
                "csv_materializations",
                "uploads",
            )
        ),
        "no_prior_or_official_selection": "excluded"
        in config["duplication_audit"]["official_exclusion"]
        and config["duplication_audit"]["posthoc_prior_cycle_adjustment"] is False,
    }
    payload = {
        "schema_version": "p3.moist_air_momentum_residual.independent_qa.v76",
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "check_count": len(checks),
        "passed": sum(checks.values()),
        "failed": len(checks) - sum(checks.values()),
        "model_fits": 0,
        "support_receipt": support,
        "hashes": {
            "config": sha256(CONFIG),
            "runner": sha256(RUNNER),
            "result": sha256(RESULT),
        },
        "official_rows": 0,
        "csv_materializations": 0,
        "uploads": 0,
    }
    (REPORT / "independent-qa.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "checks": payload["check_count"],
                "passed": payload["passed"],
                "failed": payload["failed"],
                "model_fits": payload["model_fits"],
                "official_rows": 0,
                "csv_materializations": 0,
                "uploads": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
