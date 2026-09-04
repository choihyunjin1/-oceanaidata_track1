"""Independent structural and adjudication QA for the P3 v7 candidate."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p3_v5_extratrees_competition_adjudication_20260831_v7 as runner  # noqa: E402
from run_p3_parallel_candidate_cycle_20260831_v4 import KEYS, P3_DATA  # noqa: E402


def main() -> int:
    result_path = runner.ARTIFACT_DIR / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output = result["output"]
    submission_path = Path(output["path"])
    submission = pd.read_csv(submission_path, dtype={"case_id": "string", "station": "string"})
    test_index = pd.read_csv(P3_DATA / "test_index.csv", dtype={"case_id": "string", "station": "string"})
    champion = pd.read_csv(runner.CHAMPION_PATH, dtype={"case_id": "string", "station": "string"})
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    check("runner_hash", runner.sha256(Path(runner.__file__)) == result["runner_sha256"], result["runner_sha256"])
    check("attempt_lock_hash", runner.sha256(runner.ATTEMPT_LOCK) == result["attempt_lock_sha256"], result["attempt_lock_sha256"])
    check("v5_runner_hash", runner.sha256(runner.V5_RUNNER) == result["v5_runner_sha256"], result["v5_runner_sha256"])
    check("v5_result_hash", runner.sha256(runner.V5_RESULT) == result["v5_result_sha256"], result["v5_result_sha256"])
    check("policy_hash", runner.sha256(runner.POLICY_PATH) == result["governing_policy_sha256"], result["governing_policy_sha256"])
    check("submission_hash", runner.sha256(submission_path) == output["sha256"], output["sha256"])
    check("rows_and_schema", len(submission) == 1200 and list(submission.columns) == [*KEYS, "hs_pred"], list(submission.columns))
    check("key_order", submission[KEYS].equals(test_index[KEYS]), len(test_index))
    check("key_unique", not submission.duplicated(KEYS).any(), int(submission.duplicated(KEYS).sum()))
    prediction = submission["hs_pred"].to_numpy(float)
    champion_prediction = champion["hs_pred"].to_numpy(float)
    check("finite_domain", np.isfinite(prediction).all() and prediction.min() >= 0.0 and prediction.max() <= 30.0, [float(prediction.min()), float(prediction.max())])
    inactive = ~submission["lead_h"].isin(runner.ACTIVE_LEADS).to_numpy()
    check("short_lead_exact_noop", np.array_equal(prediction[inactive], champion_prediction[inactive]), int(inactive.sum()))
    changed = int(np.sum(np.abs(prediction - champion_prediction) > 1e-12))
    check("nonduplicate_and_changed_count", changed == output["changed_rows_vs_champion"] and changed > 0, changed)
    scientific = result["adjudication"]["scientific"]
    competition = result["adjudication"]["competition"]
    check("scientific_inconclusive", scientific["ci90_low_m"] < 0.0 < scientific["ci90_high_m"] and scientific["status"].startswith("SCIENTIFIC_INCONCLUSIVE"), scientific)
    check("competition_expected_value", competition["central_projected_point_delta"] > 0.0 and competition["conservative_projected_point_delta"] < 0.0 and competition["heuristic_probability_weighted_action_points"] > 0.0, competition)
    check("exact_full_fit_once", result["full_fit_count"] == 1 and output["full_fit_seed"] == runner.FULL_FIT_SEED, [result["full_fit_count"], output["full_fit_seed"]])
    access = result["official_access"]
    check("hidden_and_upload_zero", access["hidden_truth_rows_read"] == 0 and access["uploads"] == 0 and result["hidden_truth_rows_read"] == 0 and result["uploads"] == 0, access)
    payload = {
        "schema_version": "p3.v5_extratrees_competition_adjudication.independent_qa.20260831.v7",
        "experiment_id": runner.EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "check_count": len(checks),
        "checks": checks,
        "submission_sha256": output["sha256"],
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }
    path = runner.REPORT_DIR / "independent-qa.json"
    runner.write_new(path, runner.json_bytes(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
