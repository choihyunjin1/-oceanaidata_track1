"""Lifecycle-safe independent QA for the consumed zero-fit P1 v6 gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v6_metric_consistency_preflight_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
RESULT = ROOT / f"artifacts/{EXPERIMENT_ID}/result.json"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = {
        "consumed_zero_fit": result["decision"] == "CLOSE_ZERO_FIT_DUPLICATE_AND_MISSING_INNER_ANCHOR"
        and all(value == 0 for value in result["counters"].values()),
        "rocket_duplicate": result["rocket_decision"] == "CLOSE_ZERO_FIT_SEMANTIC_DUPLICATE",
        "inner_anchor_block": result["metric_decision"] == "CLOSE_ZERO_FIT_MISSING_AUTHENTICATED_Q2_INNER_ANCHOR",
        "outer_surface_explicit": result["surface"] == "EXPLORATORY_REUSED_SURFACE",
        "outer_not_rescored": result["counters"]["outer_targets"] == 0
        and config["metric_reassessment"]["v5r1_result_use_for_threshold_selection"] == 0,
        "hashes": result["hashes"]
        == {"config": _sha(CONFIG), "runner": _sha(RUNNER), "lock": _sha(LOCK)},
        "access_zero": result["counters"]["official"] == result["counters"]["csv"] == result["counters"]["uploads"] == 0,
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "result_sha256": _sha(RESULT),
    }
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    if payload["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
