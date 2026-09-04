"""Independent QA for P2 v7 PASS and submission."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)

EXP = "p2_public_feature_benefit_gate_cycle_20260831_v7"
ART = ROOT / "artifacts" / EXP
REP = ROOT / "reports" / EXP
OBS = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    result = json.loads((ART / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((ART / "materialization.json").read_text(encoding="utf-8"))
    submission_path = Path(receipt["path"])
    submission = pd.read_csv(submission_path)
    submission["time"] = pd.to_datetime(submission["time"], utc=True)
    observations = pd.read_csv(OBS)
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    reprojection = project_profiles_vectorized(
        submission[["station", "layer", "time"]],
        submission["temp"].to_numpy(float),
        public_endpoint_frame(observations),
    ).prediction
    passes = [candidate for candidate in result["candidates"] if candidate["pass"]]
    checks = {
        "terminal_with_exactly_one_pass": result["status"] == "COMPLETE_WITH_PASS"
        and len(passes) == 1,
        "calibrated_gate_recomputed": passes[0]["calibrated_expected_points_delta"] >= 0.01,
        "prediction_hashes_match": all(
            sha256_file(ART / f"{candidate['name']}.npz") == candidate["prediction_sha256"]
            for candidate in result["candidates"]
        ),
        "submission_hash_matches": sha256_file(submission_path) == receipt["sha256"],
        "submission_qa_pass": receipt["status"] == "READY_NOT_UPLOADED"
        and all(receipt["checks"].values()),
        "pava_idempotent": bool(
            np.allclose(reprojection, submission["temp"], rtol=0.0, atol=1e-10)
        ),
        "total_fit_count_3": result["fit_count"] + receipt["full_fit_count"] == 3,
        "official_access_only_after_pass": receipt["official_test_index_rows_read"] == 26061,
        "hidden_truth_and_upload_zero": receipt["hidden_truth_rows_read"]
        == receipt["upload_count"]
        == 0,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "result_sha256": sha256_file(ART / "result.json"),
        "submission_sha256": receipt["sha256"],
    }
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "independent-qa.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
