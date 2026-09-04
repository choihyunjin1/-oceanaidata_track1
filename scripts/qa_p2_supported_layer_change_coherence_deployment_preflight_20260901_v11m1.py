"""Independent hash/schema/action QA for P2 v11m1 candidate pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = "p2_supported_layer_change_coherence_deployment_preflight_20260901_v11m1"
ART = ROOT / "artifacts" / EXP
REPORT = ROOT / "reports" / EXP


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Use --execute")
    result = json.loads((ART / "result.json").read_text(encoding="utf-8"))
    action_path = Path(result["action_artifact"]["path"])
    data = np.load(action_path, allow_pickle=False)
    action = data["action"]
    candidate = data["candidate"]
    champion = data["champion"]
    weight = data["weight"]
    checks = {
        "action_hash": sha(action_path) == result["action_artifact"]["sha256"],
        "formula": np.array_equal(candidate - champion, action),
        "inactive_exact": np.array_equal(candidate[weight == 1.0], champion[weight == 1.0]),
        "finite": bool(np.isfinite(candidate).all()),
        "guard_decision_consistent": result["guard_result"]["passed"]
        == all(result["guard_result"]["checks"].values()),
        "access_zero": result["operation_counters"]["hidden_truth_rows_read"] == 0
        and result["operation_counters"]["score_file_rows_read"] == 0
        and result["operation_counters"]["uploads"] == 0,
    }
    submission = result["submission"]
    if result["status"] == "READY_NOT_UPLOADED":
        path = Path(submission["path"])
        frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
        checks.update(
            {
                "submission_exists": path.is_file(),
                "submission_hash": sha(path) == submission["sha256"],
                "submission_rows": len(frame) == 26061,
                "submission_schema": list(frame.columns) == ["station", "layer", "time", "temp"],
                "submission_finite": bool(np.isfinite(frame["temp"]).all()),
                "submission_candidate_exact": np.array_equal(frame["temp"].to_numpy(float), candidate),
            }
        )
    else:
        checks["no_csv_on_fail"] = submission["created"] is False
    qa = {
        "experiment_id": EXP,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "geometry": result["guard_result"]["geometry"],
        "hidden_score_upload_access": 0,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "independent-qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
