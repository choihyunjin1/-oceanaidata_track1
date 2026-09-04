"""Independently recompute P3 v6 OOF metrics and integrity checks."""

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

import run_p3_inner_lcb_router_cycle_20260831_v6 as runner  # noqa: E402
from run_p3_parallel_candidate_cycle_20260831_v4 import (  # noqa: E402
    ACTIVE_LEADS,
    episode_bootstrap,
    load_historical,
    rmse,
)


def main() -> int:
    result_path = runner.ARTIFACT_DIR / "result.json"
    oof_path = runner.ARTIFACT_DIR / "internal_oof_active_leads.parquet"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    oof = pd.read_parquet(oof_path)
    frame, _ = load_historical()
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    check("runner_hash", runner.sha256(Path(runner.__file__)) == result["runner_sha256"], result["runner_sha256"])
    check("attempt_lock_hash", runner.sha256(runner.ATTEMPT_LOCK) == result["attempt_lock_sha256"], result["attempt_lock_sha256"])
    check("oof_hash", runner.sha256(oof_path) == result["internal_oof_active_leads"]["sha256"], result["internal_oof_active_leads"]["sha256"])
    check("oof_grain", len(oof) == 364 and not oof.duplicated(["anchor_id", "station", "lead_h"]).any(), len(oof))
    recomputed: list[dict[str, object]] = []
    active_mask = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    for record in result["candidates"]:
        name = record["spec"]["name"]
        column = f"prediction__{name}"
        candidate = frame["reference"].to_numpy(float).copy()
        mapping = frame.loc[active_mask, ["anchor_id", "station", "lead_h"]].merge(
            oof[["anchor_id", "station", "lead_h", column]],
            on=["anchor_id", "station", "lead_h"],
            how="left",
            validate="one_to_one",
        )
        candidate[active_mask] = mapping[column].to_numpy(float)
        reference = frame["reference"].to_numpy(float)
        target = frame["target_hs"].to_numpy(float)
        delta = rmse(target, candidate) - rmse(target, reference)
        bootstrap = episode_bootstrap(frame, candidate)
        gates = {
            "validity_hard_pass": bool(np.isfinite(candidate).all()),
            "pooled_rmse_improves": bool(delta < 0.0),
            "episode_bootstrap_ci90_upper_below_zero": bool(bootstrap["ci90_high"] < 0.0),
        }
        same = (
            abs(delta - record["delta_rmse"]) < 1e-12
            and abs(bootstrap["ci90_low"] - record["bootstrap"]["ci90_low"]) < 1e-12
            and abs(bootstrap["ci90_high"] - record["bootstrap"]["ci90_high"]) < 1e-12
            and gates == record["governing_gates"]
        )
        check(f"metrics::{name}", same, {"delta_rmse": delta, "ci90": [bootstrap["ci90_low"], bootstrap["ci90_high"]], "gates": gates})
        recomputed.append({"name": name, "delta_rmse": delta, "bootstrap": bootstrap, "governing_gates": gates})
    access = result["official_access"]
    check(
        "official_hidden_upload_zero",
        access["test_index_rows_read"] == 0
        and access["official_test_feature_rows_read"] == 0
        and access["hidden_truth_rows_read"] == 0
        and access["uploads"] == 0
        and result["hidden_truth_rows_read"] == 0
        and result["uploads"] == 0,
        access,
    )
    payload = {
        "schema_version": "p3.inner_lcb_router.independent_qa.20260831.v6",
        "experiment_id": runner.EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "check_count": len(checks),
        "checks": checks,
        "recomputed_candidates": recomputed,
    }
    path = runner.REPORT_DIR / "independent-qa.json"
    runner.write_new(path, runner.json_bytes(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
