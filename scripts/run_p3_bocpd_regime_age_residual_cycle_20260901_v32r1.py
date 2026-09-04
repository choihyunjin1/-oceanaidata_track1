"""Log-domain numerical recovery for terminal P3 v32 BOCPD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

import run_p3_bocpd_regime_age_residual_cycle_20260901_v32 as source  # noqa: E402

EXPERIMENT_ID = "p3_bocpd_regime_age_residual_cycle_20260901_v32r1"
SOURCE_ID = "p3_bocpd_regime_age_residual_cycle_20260901_v32"
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
    """Raised when the v32r1 recovery boundary differs."""


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


def logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        raise ContractError("logsumexp received no finite mass")
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def log_normal_density(value: float, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    return -0.5 * (
        np.log(2.0 * np.pi * variance) + np.square(value - mean) / variance
    )


def bocpd_summary_log(values: np.ndarray) -> np.ndarray:
    """Exact v32 recursion in log probability space."""
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    q25, q75 = np.quantile(values, (0.25, 0.75))
    normalized = (values - median) / max(float(q75 - q25), 1e-8)
    log_probabilities = np.asarray([0.0], dtype=np.float64)
    means = np.asarray([0.0], dtype=np.float64)
    precisions = np.asarray([1.0], dtype=np.float64)
    changepoint_history: list[float] = []
    for value in normalized:
        log_predictive = log_normal_density(value, means, 1.0 + 1.0 / precisions)
        log_prior = float(
            log_normal_density(value, np.asarray([0.0]), np.asarray([2.0]))[0]
        )
        size = min(len(log_probabilities) + 1, source.RUN_CAP + 1)
        updated = np.full(size, -np.inf, dtype=np.float64)
        updated[0] = np.log(source.HAZARD) + log_prior + logsumexp(log_probabilities)
        growth_count = min(len(log_probabilities), source.RUN_CAP)
        updated[1 : growth_count + 1] = (
            log_probabilities[:growth_count]
            + np.log1p(-source.HAZARD)
            + log_predictive[:growth_count]
        )
        log_probabilities = updated - logsumexp(updated)
        probabilities = np.exp(log_probabilities)
        new_means = np.empty(size, dtype=np.float64)
        new_precisions = np.empty(size, dtype=np.float64)
        new_means[0] = value / 2.0
        new_precisions[0] = 2.0
        if growth_count:
            new_means[1 : growth_count + 1] = (
                precisions[:growth_count] * means[:growth_count] + value
            ) / (precisions[:growth_count] + 1.0)
            new_precisions[1 : growth_count + 1] = precisions[:growth_count] + 1.0
        means, precisions = new_means, new_precisions
        changepoint_history.append(float(probabilities[0]))
    run_length = np.arange(len(probabilities), dtype=np.float64)
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive))) / np.log(source.RUN_CAP + 1.0)
    output = np.asarray(
        [
            probabilities[0],
            float(probabilities @ run_length) / source.RUN_CAP,
            entropy,
            float(probabilities.max()),
            float(probabilities[: min(7, len(probabilities))].sum()),
            float(probabilities[48:].sum()) if len(probabilities) > 48 else 0.0,
            float(np.mean(changepoint_history[-12:])),
            float(np.max(changepoint_history[-24:])),
        ],
        dtype=np.float64,
    )
    if output.shape != (8,) or not np.isfinite(output).all():
        raise ContractError("log-domain BOCPD summary differs")
    return output


def load_recovery() -> tuple[dict[str, Any], dict[str, Any]]:
    recovery = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "schema": recovery["schema_version"]
        == "p3.bocpd_regime_age_residual.recovery.config.v32r1",
        "experiment": recovery["experiment_id"] == EXPERIMENT_ID,
        "source_config": recovery["source_config_sha256"] == sha256(SOURCE_CONFIG),
        "source_runner": recovery["source_runner_sha256"] == sha256(SOURCE_RUNNER),
        "source_lock": recovery["source_lock_sha256"] == sha256(SOURCE_LOCK),
        "source_failure": recovery["source_failure_receipt_sha256"]
        == sha256(SOURCE_FAILURE),
        "source_result_absent": recovery["source_result_must_be_absent"]
        and not SOURCE_RESULT.exists(),
        "lock_consumed": json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["status"]
        == "ATTEMPT_CONSUMED_ONE_SHOT",
        "allowed_change": recovery["only_allowed_change"].startswith(
            "mathematically equivalent log-domain"
        ),
        "official_zero": all(value == 0 for value in recovery["official_policy"].values()),
    }
    if not all(checks.values()):
        raise ContractError(f"v32r1 recovery contract failed: {checks}")
    source_config = source.load_config()
    science = recovery["science_contract"]
    science_checks = {
        "hazard": float(science["hazard"]) == source.HAZARD,
        "cap": science["maximum_run_length"] == source.RUN_CAP,
        "variance": float(science["known_observation_variance"]) == 1.0,
        "features": science["feature_count"] == source.FEATURE_COUNT,
        "ridge": tuple(science["ridge"]) == tuple(item.alpha for item in source.SPECS),
        "blend": float(science["additive_residual_weight"]) == 0.10,
        "purge": science["purge_hours"] == 78,
        "fits": science["maximum_total_fits"] == 12,
        "tail": not science["tail_gate_changed"],
        "tuning": not science["result_based_tuning"],
    }
    if not all(science_checks.values()):
        raise ContractError(f"v32r1 science changed: {science_checks}")
    return recovery, source_config


def preflight_payload() -> dict[str, Any]:
    load_recovery()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v32r1 exactly-once namespace is consumed")
    benign = np.sin(np.linspace(0.0, 20.0, 289))
    ordinary = source.bocpd_summary(benign)
    stable = bocpd_summary_log(benign)
    if not np.allclose(ordinary, stable, atol=1e-12, rtol=1e-12):
        raise ContractError("benign ordinary/log BOCPD disagreement")
    extreme = np.zeros(289, dtype=np.float64)
    extreme[-1] = 1e12
    extreme_receipt = bocpd_summary_log(extreme)
    payload = {
        "schema_version": "p3.bocpd_regime_age_residual.recovery_preflight.v32r1",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE_SCIENCE_NEUTRAL_RECOVERY",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "source_config_sha256": sha256(SOURCE_CONFIG),
        "source_runner_sha256": sha256(SOURCE_RUNNER),
        "source_lock_sha256": sha256(SOURCE_LOCK),
        "source_failure_receipt_sha256": sha256(SOURCE_FAILURE),
        "source_result_absent": not SOURCE_RESULT.exists(),
        "benign_max_abs_difference": float(np.max(np.abs(ordinary - stable))),
        "extreme_finite": bool(np.isfinite(extreme_receipt).all()),
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
    original = source.bocpd_summary
    source.bocpd_summary = bocpd_summary_log
    try:
        result, arrays = source.execute(source_config)
    finally:
        source.bocpd_summary = original
    result["schema_version"] = "p3.bocpd_regime_age_residual.recovery_result.v32r1"
    result["experiment_id"] = EXPERIMENT_ID
    result["source_experiment_id"] = SOURCE_ID
    result["recovery_contract"] = recovery
    result["execution"]["science_changes"] = 0
    result["execution"]["numerical_adapter_changes"] = 1
    return result, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT.exists() or REPORT.exists() or LOCK.exists():
        raise ContractError("v32r1 exactly-once namespace already exists")
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
    report = source.render_report(result).replace(
        "cycle v32", "recovery cycle v32r1"
    ) + "\nScience changes: 0; numerical log-domain adapter changes: 1.\n"
    write_new(report_path, report.encode())
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
                "numerical_adapter_changes": 1,
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
