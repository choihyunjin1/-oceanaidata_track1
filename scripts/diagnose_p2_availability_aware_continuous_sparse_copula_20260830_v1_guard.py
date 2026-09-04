"""Aggregate-only diagnostic for the sealed P2 v1 physical-domain failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _directory in (ROOT, SRC):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from p2_restore import (  # noqa: E402
    p2_availability_aware_continuous_sparse_copula_20260830_v1 as sealed,
)

FAILURE_RECEIPT = (
    ROOT
    / "reports/p2_availability_aware_continuous_sparse_copula_20260830_v1"
    / "failure-receipt.json"
)
EXPECTED_FAILURE_RECEIPT_SHA256 = (
    "8a511b151bad037965fd1d93ef38e8b69ca25fbde0688e8a07f72ff90abe922a"
)
EXPECTED_OUTPUT = (
    ROOT
    / "reports/p2_availability_aware_continuous_sparse_copula_20260830_v1"
    / "guard-diagnostic.json"
)
LOWER_C = -5.0
UPPER_C = 45.0


class DiagnosticContractError(RuntimeError):
    """Raised when the aggregate-only diagnostic contract is violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector_receipt(values: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vector)
    return {
        "rows": int(len(vector)),
        "finite_rows": int(finite.sum()),
        "minimum_c": float(np.min(vector[finite])) if finite.any() else None,
        "maximum_c": float(np.max(vector[finite])) if finite.any() else None,
        "below_lower_count": int((vector < LOWER_C).sum()),
        "above_upper_count": int((vector > UPPER_C).sum()),
        "outside_count": int(((vector < LOWER_C) | (vector > UPPER_C)).sum()),
    }


def _capture_operands(frame: FrameType) -> dict[str, Any]:
    local = frame.f_locals
    required = {
        "fold",
        "candidate",
        "reference",
        "correction",
        "active_rows",
        "row_prediction",
    }
    if missing := required.difference(local):
        raise DiagnosticContractError(f"guard frame lacks operands: {sorted(missing)}")
    fold = str(local["fold"])
    candidate = np.asarray(local["candidate"], dtype=np.float64)
    reference = np.asarray(local["reference"], dtype=np.float64)
    correction = np.asarray(local["correction"], dtype=np.float64)
    active = np.asarray(local["active_rows"], dtype=bool)
    row_prediction = local["row_prediction"]
    if not (candidate.shape == reference.shape == correction.shape == active.shape):
        raise DiagnosticContractError("guard operand shapes differ")
    outside_reference = (reference < LOWER_C) | (reference > UPPER_C)
    outside_candidate = (candidate < LOWER_C) | (candidate > UPPER_C)
    newly_outside = outside_candidate & ~outside_reference
    preexisting_outside = outside_reference
    correction_identity_error = np.max(np.abs(candidate - reference - correction))
    if correction_identity_error > 1e-12:
        raise DiagnosticContractError("candidate != reference + correction")

    # Minimal contract repair candidate: preserve every pre-existing extreme as
    # an exact no-op; clip only an active row whose frozen reference was already
    # inside the declared absolute domain.
    repaired = reference.copy()
    repairable = active & ~outside_reference
    repaired[repairable] = np.clip(candidate[repairable], LOWER_C, UPPER_C)
    repaired_correction = repaired - reference
    repaired_outside = (repaired < LOWER_C) | (repaired > UPPER_C)
    allowed_preserved_extreme = repaired_outside & outside_reference & (
        repaired_correction == 0.0
    )
    invalid_repaired = repaired_outside & ~allowed_preserved_extreme

    layers = row_prediction["layer"].to_numpy(dtype=int)
    violations_by_layer = {
        str(layer): {
            "rows": int((layers == layer).sum()),
            "reference_outside_count": int((outside_reference & (layers == layer)).sum()),
            "candidate_outside_count": int((outside_candidate & (layers == layer)).sum()),
            "new_candidate_outside_count": int((newly_outside & (layers == layer)).sum()),
        }
        for layer in sorted(set(layers.tolist()))
    }
    return {
        "failed_fold": fold,
        "absolute_domain_c": [LOWER_C, UPPER_C],
        "reference": _vector_receipt(reference),
        "bounded_pre_guard_correction": {
            **_vector_receipt(correction),
            "minimum_c": float(np.min(correction)),
            "maximum_c": float(np.max(correction)),
            "maximum_absolute_c": float(np.max(np.abs(correction))),
        },
        "raw_candidate": _vector_receipt(candidate),
        "guard_predicates": {
            "candidate_all_finite": bool(np.isfinite(candidate).all()),
            "reference_outside_count": int(outside_reference.sum()),
            "candidate_outside_count": int(outside_candidate.sum()),
            "preexisting_reference_outside_and_candidate_outside_count": int(
                (preexisting_outside & outside_candidate).sum()
            ),
            "new_candidate_outside_count": int(newly_outside.sum()),
            "active_candidate_outside_count": int((active & outside_candidate).sum()),
            "inactive_candidate_outside_count": int((~active & outside_candidate).sum()),
            "candidate_minus_reference_minus_correction_max_abs_c": float(
                correction_identity_error
            ),
        },
        "violations_by_layer": violations_by_layer,
        "minimal_relative_domain_repair_diagnostic": {
            "description": (
                "pre-existing reference extremes are exact no-ops; active rows whose "
                "reference is in-domain are clipped only to the fixed absolute domain"
            ),
            "candidate": _vector_receipt(repaired),
            "changed_rows_from_raw_candidate": int(
                (~np.isclose(repaired, candidate, rtol=0.0, atol=1e-12)).sum()
            ),
            "preexisting_reference_extreme_noop_rows": int(
                (outside_reference & (repaired_correction == 0.0)).sum()
            ),
            "invalid_outside_rows_after_relative_contract": int(invalid_repaired.sum()),
            "maximum_absolute_correction_c": float(np.max(np.abs(repaired_correction))),
            "inactive_rows_changed": int(
                (~np.isclose(repaired[~active], reference[~active], rtol=0.0, atol=0.0)).sum()
            ),
        },
    }


def diagnose(p2_dir: Path) -> dict[str, Any]:
    if _sha256_file(FAILURE_RECEIPT) != EXPECTED_FAILURE_RECEIPT_SHA256:
        raise DiagnosticContractError("sealed v1 failure receipt changed")
    config = sealed.load_config()
    observations, source_receipt, access = sealed._read_training_source(p2_dir, config)
    state_table = sealed._public_state_table(observations, config)
    block_observations = sealed.legacy._assign_blocks(observations, config)
    diagnostic_rows = block_observations.loc[block_observations["block"].notna()].copy()
    marked = sealed.legacy.stage0._mark_row_diagnostics(diagnostic_rows, config)
    profile_flags = sealed.legacy.stage0._profile_flag_table(marked, config)
    anchor_record = config["immutable_training_inputs"]["alpha50_oof_anchor"]
    anchor_path = ROOT / anchor_record["path"]
    fold, fold_spec = next(iter(config["frozen_historical_windows"].items()))
    captured: dict[str, Any] = {}

    target_code = sealed._predict_outer.__code__

    def tracer(frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is target_code and event == "exception":
            _exception_type, exception, _traceback = arg
            if isinstance(exception, sealed.ExperimentContractError) and str(exception) == (
                "candidate left the finite physical temperature domain"
            ):
                captured.update(_capture_operands(frame))
        return tracer

    started = time.perf_counter()
    sys.settrace(tracer)
    try:
        sealed._predict_outer(
            fold=fold,
            fold_spec=fold_spec,
            config=config,
            observations=observations,
            state_table=state_table,
            profile_flags=profile_flags,
            anchor_path=anchor_path,
        )
    except sealed.ExperimentContractError as error:
        if str(error) != "candidate left the finite physical temperature domain":
            raise
    else:
        raise DiagnosticContractError("sealed v1 guard did not reproduce")
    finally:
        sys.settrace(None)
    if not captured:
        raise DiagnosticContractError("failed guard operands were not captured")
    elapsed = time.perf_counter() - started
    return {
        "schema_version": "p2.availability_aware_continuous_sparse_copula.guard_diagnostic.20260830.v1",
        "experiment_id": sealed.EXPERIMENT_ID,
        "status": "AGGREGATE_TECHNICAL_DIAGNOSTIC_NO_PERFORMANCE_SCORING",
        "sealed_failure_receipt": {
            "path": str(FAILURE_RECEIPT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": EXPECTED_FAILURE_RECEIPT_SHA256,
        },
        "diagnostic_scope": {
            "v1_main_called": False,
            "validation_truth_bound": False,
            "performance_metrics_computed": False,
            "bootstrap_computed": False,
            "decision_computed": False,
            "diagnostic_outer_fits": 1,
            "result_based_tuning": False,
            "model_or_threshold_changes": False,
        },
        "guard_operands": captured,
        "source": source_receipt,
        "source_open_counts": access.open_counts,
        "access_receipt": {
            "official_interface_rows_read": 0,
            "query_support_rows_read": 0,
            "csv_output_count": 0,
            "submission_generated": False,
            "upload_count": 0,
            "hard_deleted_training_profiles": 0,
        },
        "runtime_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--diagnose", action="store_true", required=True)
    args = parser.parse_args()
    output = args.output_json.resolve(strict=False)
    if output != EXPECTED_OUTPUT.resolve(strict=False):
        raise DiagnosticContractError("--output-json must equal sealed diagnostic path")
    if output.exists():
        raise FileExistsError(output)
    result = diagnose(args.p2_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_json": str(output),
                "failed_fold": result["guard_operands"]["failed_fold"],
                "reference_outside_count": result["guard_operands"]["guard_predicates"][
                    "reference_outside_count"
                ],
                "new_candidate_outside_count": result["guard_operands"]["guard_predicates"][
                    "new_candidate_outside_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
