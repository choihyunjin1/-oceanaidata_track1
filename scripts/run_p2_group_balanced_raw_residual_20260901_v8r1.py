"""Datetime-unit contract repair for the sealed P2 v8 exploratory experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_group_balanced_raw_residual_20260901_v8 as engine  # noqa: E402

EXPERIMENT_ID = "p2_group_balanced_raw_residual_20260901_v8r1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID

engine.EXPERIMENT_ID = EXPERIMENT_ID
engine.CONFIG = CONFIG
engine.ARTIFACT = ARTIFACT
engine.REPORT = REPORT
engine.RUNNER_PATH = Path(__file__)
engine.RESULT_SCHEMA = "p2.group_balanced_raw_residual.result.20260901.v8r1"
engine.REPORT_TITLE = "# P2 group-balanced raw-residual exploratory cycle 20260901 v8r1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def preflight() -> dict[str, object]:
    us = pd.DatetimeIndex(["2024-09-01T00:00:00Z"]).as_unit("us")
    ns = pd.DatetimeIndex(["2024-09-01T00:00:00Z"]).as_unit("ns")
    us_canonical = engine.canonical_time_ns(us).tolist()
    ns_canonical = engine.canonical_time_ns(ns).tolist()
    if us_canonical != ns_canonical:
        raise engine.ContractError("mixed-unit nanosecond canonicalization failed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "mixed_unit_keys_equal": True,
        "candidate_names": [item["name"] for item in config["candidates"]],
        "seeds": config["training"]["seeds"],
        "config_sha256": engine.sha256_file(CONFIG),
        "runner_sha256": engine.sha256_file(Path(__file__)),
        "engine_sha256": engine.sha256_file(engine.ENGINE_PATH),
        "data_rows_read": 0,
        "model_fits": 0,
        "artifacts_written": 0,
        "official_rows_read": 0,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["preflight_sha256"] = sha256_text(canonical)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight or --execute")
    if args.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, indent=2, allow_nan=False))
        return
    print(json.dumps(engine.run(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
