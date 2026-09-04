"""One-line datetime-resolution recovery for the sealed P1 v18 execution."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v18 as original  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v18r1"
RECOVERY_CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
BASE_CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v18.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RECEIPT = REPORT / "technical-recovery-receipt.json"
ORIGINAL_LOCK = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v18/attempt_lock.json"
ORIGINAL_FAILURE = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v18/terminal_failure.json"


def fixed_utc_minutes(values: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(values, utc=True).to_numpy(dtype="datetime64[ns]").astype("int64")
    return (timestamps // 60_000_000_000).astype(np.int64)


def load_recovery_contract() -> dict[str, Any]:
    config = json.loads(RECOVERY_CONFIG.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    hashes = receipt["hash_chain"]
    checks = {
        "base_config": config["base_method_config_sha256"] == original.sha256(BASE_CONFIG),
        "receipt_base_config": hashes["base_method_config_sha256"] == original.sha256(BASE_CONFIG),
        "original_runner": hashes["original_runner_sha256"]
        == original.sha256(ROOT / "scripts/run_p1_public_transport_repair_cycle_20260831_v18.py"),
        "original_lock": hashes["original_attempt_lock_sha256"] == original.sha256(ORIGINAL_LOCK),
        "original_failure": hashes["original_terminal_failure_sha256"] == original.sha256(ORIGINAL_FAILURE),
        "only_one_change": config["method_changes"] == config["feature_changes"] == config["model_changes"]
        == config["threshold_changes"] == config["gate_changes"] == config["fit_budget_changes"] == 0,
        "prefit": config["original_fit_count"] == 0 and config["original_metric_rows"] == 0,
        "external_zero": all(
            config[name] == 0
            for name in ("official_reads", "hidden_truth_reads", "submission_csv_created", "uploads")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v18r1 recovery contract mismatch: {checks}")
    return config


def prepare_original() -> None:
    load_recovery_contract()
    original.EXPERIMENT_ID = EXPERIMENT_ID
    original.CONFIG = BASE_CONFIG
    original.ARTIFACT = ARTIFACT
    original.REPORT = REPORT
    original._utc_minutes = fixed_utc_minutes
    original.__file__ = str(Path(__file__).resolve())


def validate_only() -> dict[str, Any]:
    prepare_original()
    payload = original.validate_only()
    payload["recovery_manifest_sha256"] = original.sha256(RECOVERY_CONFIG)
    payload["technical_receipt_sha256"] = original.sha256(RECEIPT)
    payload["datetime_resolution_regression"] = "PASS"
    return payload


def execute() -> dict[str, Any]:
    prepare_original()
    result = original.execute()
    result["schema_version"] = "p1.public_transport_repair_cycle.20260831.v18r1.result"
    result["technical_recovery"] = {
        "base_experiment_id": "p1_public_transport_repair_cycle_20260831_v18",
        "original_fit_count": 0,
        "original_metric_rows": 0,
        "only_change": "datetime64[ns] normalization before minute conversion",
        "recovery_manifest_sha256": original.sha256(RECOVERY_CONFIG),
        "technical_receipt_sha256": original.sha256(RECEIPT),
    }
    original.base.write_json(ARTIFACT / "result.json", result, exclusive=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate_only == args.execute:
        parser.error("choose exactly one mode")
    try:
        payload = validate_only() if args.validate_only else execute()
        print(json.dumps(original.base.native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            original.base.write_json(ARTIFACT / "terminal_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
