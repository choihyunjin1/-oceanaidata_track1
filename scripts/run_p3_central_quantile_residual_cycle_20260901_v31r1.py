"""Science-neutral scorer-adapter recovery for terminal P3 v31."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

import run_p3_central_quantile_residual_cycle_20260901_v31 as source  # noqa: E402

EXPERIMENT_ID = "p3_central_quantile_residual_cycle_20260901_v31r1"
SOURCE_ID = "p3_central_quantile_residual_cycle_20260901_v31"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
SOURCE_CONFIG = ROOT / "configs/experiments" / f"{SOURCE_ID}.json"
SOURCE_RUNNER = ROOT / "scripts" / f"run_{SOURCE_ID}.py"
SOURCE_LOCK = ROOT / "artifacts" / f"{SOURCE_ID}.ATTEMPT_LOCK.json"
SOURCE_RESULT = ROOT / "artifacts" / SOURCE_ID / "result.json"
SOURCE_FAILURE = ROOT / "reports" / SOURCE_ID / "failure-receipt.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


class ContractError(RuntimeError):
    """Raised when the v31r1 recovery boundary differs."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def science_receipt(source_config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "features": source_config["features"],
        "inputs": source_config["inputs"],
        "model": source_config["model"],
        "reference": source_config["reference"],
        "score": source_config["score"],
        "validation": source_config["validation"],
    }
    return {
        "payload_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
        "quantiles": list(source.QUANTILES),
        "l1_alpha": source.L1_ALPHA,
        "blend": source.BLEND,
        "case_feature_count": source.CASE_FEATURE_COUNT,
        "row_feature_count": source.ROW_FEATURE_COUNT,
        "maximum_total_fits": 12,
    }


def load_recovery() -> tuple[dict[str, Any], dict[str, Any]]:
    recovery = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "schema": recovery["schema_version"]
        == "p3.central_quantile_residual.recovery.config.v31r1",
        "experiment": recovery["experiment_id"] == EXPERIMENT_ID,
        "source_experiment": recovery["source_experiment_id"] == SOURCE_ID,
        "source_config_hash": recovery["source_config_sha256"]
        == sha256(SOURCE_CONFIG),
        "source_runner_hash": recovery["source_runner_sha256"]
        == sha256(SOURCE_RUNNER),
        "source_lock_hash": recovery["source_lock_sha256"] == sha256(SOURCE_LOCK),
        "failure_hash": recovery["source_failure_receipt_sha256"]
        == sha256(SOURCE_FAILURE),
        "source_result_absent": recovery["source_result_must_be_absent"]
        and not SOURCE_RESULT.exists(),
        "source_lock_consumed": json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))[
            "status"
        ]
        == "ATTEMPT_CONSUMED_ONE_SHOT",
        "allowed_change": recovery["only_allowed_change"].startswith(
            "temporarily register the unchanged source SPEC"
        ),
        "official_zero": all(value == 0 for value in recovery["official_policy"].values()),
    }
    if not all(checks.values()):
        raise ContractError(f"v31r1 recovery contract failed: {checks}")
    source_config = source.load_config()
    sealed = recovery["science_contract"]
    science_checks = {
        "source_verbatim": sealed["source_config_used_verbatim"],
        "quantiles": tuple(sealed["quantiles"]) == source.QUANTILES,
        "alpha": float(sealed["l1_alpha"]) == source.L1_ALPHA,
        "blend": float(sealed["additive_residual_weight"]) == source.BLEND,
        "case_features": sealed["case_feature_count"] == source.CASE_FEATURE_COUNT,
        "row_features": sealed["row_feature_count"] == source.ROW_FEATURE_COUNT,
        "blocks": tuple(sealed["outer_blocks"]) == tuple(source.v23.BLOCKS),
        "purge": sealed["purge_hours"] == 78,
        "fits": sealed["maximum_total_fits"] == 12,
        "tail_unchanged": not sealed["tail_gate_changed"],
        "no_tuning": not sealed["result_based_tuning"],
    }
    if not all(science_checks.values()):
        raise ContractError(f"v31r1 science changed: {science_checks}")
    return recovery, source_config


def score_with_registered_spec(frame, prediction: np.ndarray) -> dict[str, Any]:
    """Register only the unchanged source SPEC and restore foreign state."""
    original_specs = source.v28.SPECS
    source.v28.SPECS = (source.SPEC,)
    try:
        return source.v28.score(frame, prediction, source.SPEC)
    finally:
        source.v28.SPECS = original_specs


def preflight_payload() -> dict[str, Any]:
    _, source_config = load_recovery()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v31r1 exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.central_quantile_residual.recovery_preflight.v31r1",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE_SCIENCE_NEUTRAL_RECOVERY",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "source_config_sha256": sha256(SOURCE_CONFIG),
        "source_runner_sha256": sha256(SOURCE_RUNNER),
        "source_lock_sha256": sha256(SOURCE_LOCK),
        "source_failure_receipt_sha256": sha256(SOURCE_FAILURE),
        "source_result_absent": not SOURCE_RESULT.exists(),
        "science_receipt": science_receipt(source_config),
        "maximum_model_fits": 12,
        "official_access": 0,
        "csv_materializations": 0,
        "uploads": 0,
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(
    recovery: dict[str, Any], source_config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, profile = source.v23.case_surface()
    features, feature_receipt = source.surface_features(cases)
    prediction, receipts = source.crossfit(cases, features, targets, reference)
    frame = source.v23.long_frame(cases, targets, reference)
    scored = score_with_registered_spec(frame, prediction)
    result = {
        "schema_version": "p3.central_quantile_residual.recovery_result.v31r1",
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE"
        if scored["decision"] != "NO_GO"
        else "NO_GO_CENTRAL_QUANTILE_CANDIDATE",
        "surface_claim": source_config["validation"]["surface"],
        "reference": source_config["reference"],
        "duplication_audit": source_config["duplication_audit"],
        "primary_sources": source_config["primary_sources"],
        "recovery_contract": recovery,
        "science_receipt": science_receipt(source_config),
        "feature_receipt": feature_receipt,
        "candidate": scored,
        "fit_receipts": receipts,
        "fit_count": 12,
        "data_profile": profile,
        "data_access": {
            "historical_target_rows": 1092,
            "official_test_rows": 0,
            "official_sample_rows": 0,
            "official_submission_rows": 0,
            "hidden_truth_rows": 0,
            "csv_materializations": 0,
            "uploads": 0,
        },
        "execution": {
            "python": platform.python_version(),
            "elapsed_seconds": time.perf_counter() - started,
            "candidate_count": 1,
            "result_based_tuning": False,
            "outer_result_parameter_changes": 0,
            "science_changes": 0,
            "scorer_adapter_changes": 1,
            "row_deletion": 0,
        },
    }
    arrays = {
        "truth": targets,
        "uniform": reference,
        "candidate_1": prediction,
        "anchor_id": cases["anchor_id"].to_numpy(np.int32),
        "lead_h": np.asarray(source.v23.LEADS, dtype=np.int16),
        "block": cases["block"].to_numpy(dtype="U5"),
        "station": cases["station"].to_numpy(dtype="U5"),
        "episode": cases["episode_id"].to_numpy(dtype="U32"),
    }
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    item = result["candidate"]
    metric, points = item["rmse_m"], item["expected_points"]
    return (
        "# P3 central quantile recovery v31r1\n\n## 결론\n\n"
        f"- overall decision: **{result['decision']}**.\n"
        "- This is a science-neutral recovery: only the scorer SPEC registry adapter changed.\n"
        f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; "
        f"raw {points['raw_gain']:+.6f} points; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; "
        f"episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.\n"
        "- The 182-case surface is EXPLORATORY_ONLY. Official/hidden/CSV/upload access is zero.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT.exists() or REPORT.exists() or LOCK.exists():
        raise ContractError("v31r1 exactly-once namespace already exists")
    recovery, source_config = load_recovery()
    preflight = preflight_payload()
    write_new(
        LOCK,
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "ATTEMPT_CONSUMED_ONE_SHOT",
                "runner_sha256": sha256(Path(__file__)),
                "config_sha256": sha256(CONFIG),
                "preflight_receipt_sha256": preflight["receipt_sha256"],
                "source_failure_receipt_sha256": sha256(SOURCE_FAILURE),
                "official_access": 0,
            }
        ),
    )
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    result, arrays = execute(recovery, source_config)
    array_path = ARTIFACT / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {
        "runner_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(CONFIG),
        "source_runner_sha256": sha256(SOURCE_RUNNER),
        "source_config_sha256": sha256(SOURCE_CONFIG),
        "source_lock_sha256": sha256(SOURCE_LOCK),
        "source_failure_receipt_sha256": sha256(SOURCE_FAILURE),
        "evaluation_arrays_sha256": sha256(array_path),
        "preflight_receipt_sha256": preflight["receipt_sha256"],
    }
    result_path = ARTIFACT / "result.json"
    write_new(result_path, canonical(result))
    report_path = REPORT / "report-source.md"
    write_new(report_path, render_report(result).encode())
    write_new(REPORT / "result.json", canonical(result))
    write_new(
        REPORT / "run-manifest.json",
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "result_sha256": sha256(result_path),
                "arrays_sha256": sha256(array_path),
                "report_sha256": sha256(report_path),
                "fit_count": 12,
                "science_changes": 0,
                "scorer_adapter_changes": 1,
                "official_access": 0,
                "csv_materializations": 0,
                "uploads": 0,
            }
        ),
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "decision": result["decision"],
                "fit_count": 12,
                "science_changes": 0,
                "official_access": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
