"""Zero-fit P1 metric-consistency and ROCKET duplication gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v6_metric_consistency_preflight_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2).encode() + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def add_only_delta(anchor_tp: int, anchor_fp: int, anchor_fn: int, true_add: int, false_add: int) -> tuple[float, float]:
    denominator = 2 * anchor_tp + anchor_fp + anchor_fn
    anchor = 2 * anchor_tp / denominator if denominator else 0.0
    candidate_denominator = denominator + true_add + false_add
    candidate = 2 * (anchor_tp + true_add) / candidate_denominator if candidate_denominator else 0.0
    return anchor, candidate - anchor


def exact_break_even(anchor_tp: int, anchor_fp: int, anchor_fn: int, true_add: int, false_add: int) -> bool:
    anchor, delta = add_only_delta(anchor_tp, anchor_fp, anchor_fn, true_add, false_add)
    count = true_add + false_add
    precision = true_add / count if count else 0.0
    return (delta > 0) == (count > 0 and precision > anchor / 2)


def preflight() -> dict[str, Any]:
    if ARTIFACT.exists() or LOCK.exists():
        raise FileExistsError("namespace consumed")
    config = _read(CONFIG)
    for relative, expected in config["rocket_semantic_audit"]["evidence"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"duplicate evidence drifted: {relative}")
    metric = config["metric_reassessment"]
    if _sha(ROOT / metric["v5r1_result"]["path"]) != metric["v5r1_result"]["sha256"]:
        raise RuntimeError("v5r1 result binding drifted")
    oof_path = ROOT / metric["authenticated_oof"]["path"]
    if _sha(oof_path) != metric["authenticated_oof"]["sha256"]:
        raise RuntimeError("authenticated OOF drifted")
    oof_times = pd.to_datetime(pd.read_parquet(oof_path, columns=["time"])["time"], utc=True, format="mixed")
    minimum = oof_times.min()
    q2_inner_end = pd.Timestamp(metric["q2_inner_end"])
    if not minimum > q2_inner_end:
        raise RuntimeError("Q2 inner anchor blocker no longer holds")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_ZERO_OPERATION_WITH_SCIENTIFIC_BLOCK",
        "config_sha256": _sha(CONFIG),
        "runner_sha256": _sha(Path(__file__)),
        "rocket_decision": config["rocket_semantic_audit"]["decision"],
        "metric_decision": metric["scientific_decision"],
        "surface": metric["surface"],
        "authenticated_oof_minimum": minimum.isoformat(),
        "q2_inner_end": q2_inner_end.isoformat(),
        "algebra_examples": [
            {"counts": [10, 2, 8, 3, 2], "exact": exact_break_even(10, 2, 8, 3, 2)},
            {"counts": [10, 2, 8, 1, 4], "exact": exact_break_even(10, 2, 8, 1, 4)},
        ],
        "counters": {"fits": 0, "outer_targets": 0, "official": 0, "csv": 0, "uploads": 0},
    }


def qa() -> dict[str, Any]:
    ready = preflight()
    checks = {
        "zero": all(value == 0 for value in ready["counters"].values()),
        "rocket_duplicate": ready["rocket_decision"] == "CLOSE_ZERO_FIT_SEMANTIC_DUPLICATE",
        "metric_blocked": ready["metric_decision"] == "CLOSE_ZERO_FIT_MISSING_AUTHENTICATED_Q2_INNER_ANCHOR",
        "outer_reused": ready["surface"] == "EXPLORATORY_REUSED_SURFACE",
        "algebra": all(item["exact"] for item in ready["algebra_examples"]),
    }
    return {"experiment_id": EXPERIMENT_ID, "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def execute() -> dict[str, Any]:
    ready = preflight()
    _write(LOCK, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_ZERO_FIT", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT.mkdir(exist_ok=False)
    result = {
        **ready,
        "decision": "CLOSE_ZERO_FIT_DUPLICATE_AND_MISSING_INNER_ANCHOR",
        "scientific_interpretation": "No v5r1 threshold was changed and no outer metric was rescored.",
        "next_axis": _read(CONFIG)["next_axis"],
        "hashes": {"config": ready["config_sha256"], "runner": ready["runner_sha256"], "lock": _sha(LOCK)},
    }
    _write(ARTIFACT / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = preflight() if args.preflight else qa() if args.qa else execute()
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False), end="")


if __name__ == "__main__":
    main()
