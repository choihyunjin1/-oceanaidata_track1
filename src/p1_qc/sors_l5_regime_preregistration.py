"""Fail-closed audit for the non-executable S-ORS layer-5 hypothesis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "P1_sors_l5_regime_invariance_v1"
FAMILY_ID = "sors_l5_deployment_regime_invariance"
REFERENCE_HASHES = {
    "config": "7c0f5630717567b7c2fbeacb6e86b600e01b77951cc9cea0cbd34ec8c68f7947",
    "train": "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2",
    "test": "6d5c6522c282651b99f4261ffa803cf99950596028e996de1e7714db77408387",
}
EXACT_CHANGE = {
    "kind": "deterministic_categorical_value_transform_before_encoder_fit",
    "station": "S-ORS",
    "layer": 5,
    "column": "depth_regime",
    "value": "S-ORS|deployment_unknown|l5",
    "apply_to": ["fold_train", "fold_validation", "full_train", "test"],
    "mask_inputs": ["station", "layer"],
    "uses_labels": False,
    "numeric_depth_unchanged": True,
    "non_target_rows_bitwise_unchanged": True,
    "category_map_fit": "fold_train_only",
}
EXACT_SUPPORT = {"2025_q2": 238, "2025_q3": 4, "2025_q4": 270}
RECOVERABLE_WEIGHTED_UPPER_BOUND = 0.002463588960835761
PROMOTION_DELTA_REQUIRED = 0.005


class SORSL5PreregistrationError(ValueError):
    """Raised when the fixed contract drifts or an outer run is requested."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SORSL5PreregistrationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_sors_l5_preregistration(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise SORSL5PreregistrationError(f"invalid preregistration JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SORSL5PreregistrationError("preregistration root must be an object")
    return value


def canonical_sors_l5_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SORSL5PreregistrationError(f"{name} must be an object")
    return value


def _exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise SORSL5PreregistrationError(f"{name} must equal the frozen contract")


def validate_sors_l5_preregistration(
    payload: Mapping[str, Any],
    *,
    ledger_rows: Sequence[Mapping[str, Any]] = (),
    observed_hashes: Mapping[str, str] | None = None,
    require_outer_authorized: bool = False,
) -> dict[str, Any]:
    """Validate implementation provenance and unconditionally deny outer CV."""

    _exact(payload.get("schema_version"), "1.0-draft", "schema_version")
    _exact(payload.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _exact(payload.get("family_id"), FAMILY_ID, "family_id")
    _exact(payload.get("status"), "implementation_only_outer_no_go", "status")
    _exact(
        payload.get("outer_decision"),
        "no_go_due_experiment_budget",
        "outer decision",
    )

    input_contract = _mapping(payload.get("input_contract"), "input_contract")
    for name, expected in REFERENCE_HASHES.items():
        _exact(input_contract.get(f"{name}_sha256"), expected, f"{name} hash")

    hypothesis = _mapping(payload.get("hypothesis"), "hypothesis")
    change = _mapping(hypothesis.get("exactly_one_change"), "exactly_one_change")
    _exact(dict(change), EXACT_CHANGE, "exactly_one_change")
    forbidden = hypothesis.get("forbidden_changes")
    required_forbidden = {
        "numeric_depth_features",
        "non_sors_layer5_rows",
        "feature_columns",
        "xgboost_parameters",
        "iteration_counts",
        "postprocess",
        "thresholds",
        "augmentation",
        "ensemble",
    }
    if not isinstance(forbidden, list) or not required_forbidden.issubset(forbidden):
        raise SORSL5PreregistrationError("forbidden_changes is incomplete")

    comparison = _mapping(payload.get("comparison"), "comparison")
    _exact(comparison.get("additional_hyperparameters"), [], "additional hyperparameters")
    _exact(comparison.get("adaptive_search_allowed"), False, "adaptive search")
    _exact(comparison.get("inner_reselection_allowed"), False, "inner reselection")
    _exact(comparison.get("outer_execution_authorized"), False, "outer execution")

    audit = _mapping(payload.get("outer_budget_audit"), "outer_budget_audit")
    _exact(audit.get("no_virgin_holdout_remains"), True, "virgin holdout audit")
    _exact(audit.get("fold_positive_support"), EXACT_SUPPORT, "fold positive support")
    _exact(
        audit.get("recoverable_test_share_weighted_f1_upper_bound"),
        RECOVERABLE_WEIGHTED_UPPER_BOUND,
        "recoverable weighted upper bound",
    )
    _exact(
        audit.get("promotion_delta_required"),
        PROMOTION_DELTA_REQUIRED,
        "promotion delta",
    )
    _exact(audit.get("upper_bound_meets_promotion_gate"), False, "upper-bound decision")
    if RECOVERABLE_WEIGHTED_UPPER_BOUND >= PROMOTION_DELTA_REQUIRED:
        raise SORSL5PreregistrationError("frozen budget arithmetic is inconsistent")

    authorization = _mapping(payload.get("authorization"), "authorization")
    _exact(authorization.get("implementation"), True, "implementation authorization")
    _exact(authorization.get("unit_tests"), True, "unit-test authorization")
    for action in (
        "outer_cv",
        "competition_upload",
        "commit",
        "push",
        "external_observations",
    ):
        _exact(authorization.get(action), False, f"authorization.{action}")

    family_exposure = sum(
        int(row.get("outer_result_count", 0))
        for row in ledger_rows
        if row.get("family_id") == FAMILY_ID
        and isinstance(row.get("outer_result_count", 0), int)
        and not isinstance(row.get("outer_result_count", 0), bool)
    )

    if observed_hashes is not None:
        if set(observed_hashes) != set(REFERENCE_HASHES):
            raise SORSL5PreregistrationError(
                f"observed_hashes must contain exactly {sorted(REFERENCE_HASHES)}"
            )
        for name, expected in REFERENCE_HASHES.items():
            _exact(str(observed_hashes[name]).lower(), expected, f"observed {name} hash")

    if require_outer_authorized:
        raise SORSL5PreregistrationError(
            "outer CV denied: no_go_due_experiment_budget; the recoverable weighted "
            "upper bound is below the fixed promotion gate and no virgin holdout remains"
        )

    return {
        "status": "valid_implementation_only",
        "experiment_id": EXPERIMENT_ID,
        "family_id": FAMILY_ID,
        "preregistration_sha256": canonical_sors_l5_sha256(payload),
        "outer_execution_authorized": False,
        "outer_decision": "no_go_due_experiment_budget",
        "family_outer_exposure": family_exposure,
        "recoverable_test_share_weighted_f1_upper_bound": RECOVERABLE_WEIGHTED_UPPER_BOUND,
        "promotion_delta_required": PROMOTION_DELTA_REQUIRED,
        "no_virgin_holdout_remains": True,
    }


def _read_ledger(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    target = Path(path)
    if not target.is_file():
        raise SORSL5PreregistrationError(f"ledger does not exist: {target}")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise SORSL5PreregistrationError(
                f"invalid ledger JSON on line {number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise SORSL5PreregistrationError(f"ledger line {number} is not an object")
        rows.append(value)
    return rows


def validate_sors_l5_preregistration_files(
    preregistration_path: str | Path,
    *,
    ledger_path: str | Path | None,
    observed_hashes: Mapping[str, str] | None = None,
    require_outer_authorized: bool = False,
) -> dict[str, Any]:
    return validate_sors_l5_preregistration(
        load_sors_l5_preregistration(preregistration_path),
        ledger_rows=_read_ledger(ledger_path),
        observed_hashes=observed_hashes,
        require_outer_authorized=require_outer_authorized,
    )


__all__ = [
    "EXACT_CHANGE",
    "EXACT_SUPPORT",
    "EXPERIMENT_ID",
    "FAMILY_ID",
    "PROMOTION_DELTA_REQUIRED",
    "RECOVERABLE_WEIGHTED_UPPER_BOUND",
    "REFERENCE_HASHES",
    "SORSL5PreregistrationError",
    "canonical_sors_l5_sha256",
    "load_sors_l5_preregistration",
    "validate_sors_l5_preregistration",
    "validate_sors_l5_preregistration_files",
]
