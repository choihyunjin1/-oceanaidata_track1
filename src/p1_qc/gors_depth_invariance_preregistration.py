"""Fail-closed preregistration for the one-shot G-ORS depth-invariance family."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FAMILY_ID = "gors_deployment_depth_invariance"
EXPERIMENT_ID = "P1_gors_depth_invariance_v1"
FOLD_NAMES = ("2025_q2", "2025_q3", "2025_q4")
REFERENCE_RUN_ID = "20260813T153038+0900_cv_378a4e89"
REFERENCE_MICRO_F1 = 0.8603708380408055
REFERENCE_WEIGHTED_F1 = 0.8133155525620019
REFERENCE_HASHES = {
    "config": "7c0f5630717567b7c2fbeacb6e86b600e01b77951cc9cea0cbd34ec8c68f7947",
    "train": "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2",
    "test": "6d5c6522c282651b99f4261ffa803cf99950596028e996de1e7714db77408387",
    "oof": "d1b9439db6d0d906fa080bd01f1eb8fc21d051c3d056a274e2b02e43c1e55f4a",
    "metrics": "de7d92153df58b10177ac6c79a0733164c9c4630fa2c317495ddd8a1eb487e36",
    "selection": "ebc2dcdd57f5a610151d84a3e37f9bb4295030f0dcb40314972be4c419e01de8",
    "deployment_stress": "ea2a3e797dc9edbd2716b517f3ff35611abfbc4bd7e1b29eddcfb8ca2971bc8e",
}
FOLD_POSTPROCESS = {
    "2025_q2": {
        "close_gap_rows": 0,
        "high_threshold": 0.15,
        "low_threshold": 0.075,
        "minimum_positive_run": 12,
    },
    "2025_q3": {
        "close_gap_rows": 0,
        "high_threshold": 0.2,
        "low_threshold": 0.1,
        "minimum_positive_run": 12,
    },
    "2025_q4": {
        "close_gap_rows": 6,
        "high_threshold": 0.15,
        "low_threshold": 0.075,
        "minimum_positive_run": 6,
    },
}
EXACT_CHANGE = {
    "kind": "deterministic_feature_transform_before_encoder_fit",
    "name": "gors_fold_symmetric_depth_mask",
    "station": "G-ORS",
    "depth_numeric_columns": [
        "depth_raw",
        "nominal_depth_m",
        "depth_diff_1",
        "depth_abs_diff_1",
    ],
    "depth_numeric_value": "NaN",
    "depth_missing": 1,
    "depth_regime_template": "{station}|unknown|l{layer}",
    "apply_to": ["fold_train", "fold_validation", "full_train", "test"],
    "mask_inputs": ["station", "layer"],
    "uses_labels": False,
    "category_map_fit": "fold_train_only",
    "categorical_encoding_policy": "preserve original fold-train codes for shared categories; candidate-only fallback=max(original)+1; validation-only=-1",
    "arm_encoding_consequence": "A G-ORS fallback is -1 when unseen in natural-depth fold-train; B G-ORS fallback is known because symmetric mask exposes it in fold-train",
}
EXACT_PRIMARY = {
    "id": "A_deployment_matched_frozen_baseline",
    "training": "original natural-depth fold-train",
    "validation": "mask G-ORS depth inputs only",
    "model": "frozen XGBoost refit with same fold, seed, parameters, and 700 trees",
    "postprocess": "exact frozen per-fold values",
    "expected_test_share_weighted_f1": 0.8043980282796417,
    "expected_gors_f1": 0.7633410672853829,
}
EXACT_CANDIDATE = {
    "id": "B_gors_fold_symmetric_depth_mask",
    "training": "mask G-ORS depth inputs in fold-train",
    "validation": "mask G-ORS depth inputs in fold-validation",
    "model": "refit XGBoost with same fold, seed, parameters, and 700 trees",
    "postprocess": "exact frozen per-fold values",
}
EXACT_GATES = {
    "primary_delta": "B_minus_A",
    "test_share_weighted_f1_delta_min": 0.005,
    "paired_bootstrap_90pct_lower_bound_gt": 0.0,
    "gors_group_f1_delta_min": 0.02,
    "non_g_test_share_weighted_f1_delta_min": -0.001,
    "non_g_station_layer_f1_drop_max": 0.01,
    "folds_non_degrading_min": 2,
    "normal_fp_day_relative_increase_lt": 0.1,
}


class GORSDepthPreregistrationError(ValueError):
    """Raised when the single allowed experiment contract is not exact."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GORSDepthPreregistrationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_gors_depth_preregistration(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise GORSDepthPreregistrationError(f"invalid preregistration JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GORSDepthPreregistrationError("preregistration root must be an object")
    return value


def canonical_gors_depth_sha256(payload: Mapping[str, Any]) -> str:
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
        raise GORSDepthPreregistrationError(f"{name} must be an object")
    return value


def _exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise GORSDepthPreregistrationError(f"{name} must equal the frozen contract")


def _read_ledger(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    target = Path(path)
    if not target.is_file():
        raise GORSDepthPreregistrationError(f"ledger does not exist: {target}")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise GORSDepthPreregistrationError(
                f"invalid ledger JSON on line {number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise GORSDepthPreregistrationError(f"ledger line {number} is not an object")
        rows.append(value)
    return rows


def validate_gors_depth_preregistration(
    payload: Mapping[str, Any],
    *,
    ledger_rows: Sequence[Mapping[str, Any]] = (),
    observed_hashes: Mapping[str, str] | None = None,
    require_outer_authorized: bool = False,
) -> dict[str, Any]:
    """Validate the exact one-change contract and family exposure state."""

    _exact(payload.get("schema_version"), "1.0-draft", "schema_version")
    _exact(payload.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _exact(payload.get("family_id"), FAMILY_ID, "family_id")

    input_contract = _mapping(payload.get("input_contract"), "input_contract")
    _exact(input_contract.get("config_sha256"), REFERENCE_HASHES["config"], "config hash")
    _exact(input_contract.get("train_sha256"), REFERENCE_HASHES["train"], "train hash")
    _exact(input_contract.get("test_sha256"), REFERENCE_HASHES["test"], "test hash")

    hypothesis = _mapping(payload.get("hypothesis"), "hypothesis")
    change = _mapping(hypothesis.get("exactly_one_change"), "exactly_one_change")
    _exact(dict(change), EXACT_CHANGE, "exactly_one_change")
    forbidden = hypothesis.get("forbidden_changes")
    required_forbidden = {
        "non_gors_rows",
        "sors_layer5",
        "feature_columns",
        "xgboost_parameters",
        "iteration_counts",
        "postprocess",
        "thresholds",
        "augmentation",
        "ensemble",
    }
    if not isinstance(forbidden, list) or not required_forbidden.issubset(forbidden):
        raise GORSDepthPreregistrationError("forbidden_changes is incomplete")

    baseline = _mapping(payload.get("baseline"), "baseline")
    _exact(baseline.get("run_id"), REFERENCE_RUN_ID, "baseline run")
    _exact(baseline.get("backend"), "xgboost", "baseline backend")
    _exact(baseline.get("feature_mode"), "offline", "baseline feature mode")
    _exact(baseline.get("oof_sha256"), REFERENCE_HASHES["oof"], "baseline OOF hash")
    _exact(baseline.get("micro_f1"), REFERENCE_MICRO_F1, "baseline micro F1")
    _exact(
        baseline.get("test_share_weighted_f1"),
        REFERENCE_WEIGHTED_F1,
        "baseline weighted F1",
    )
    _exact(baseline.get("fold_iteration_counts"), [700, 700, 700], "fold iterations")
    artifacts = _mapping(baseline.get("reference_artifacts"), "reference_artifacts")
    _exact(
        dict(artifacts),
        {
            f"{name}_sha256": REFERENCE_HASHES[name]
            for name in ("oof", "metrics", "selection", "deployment_stress")
        },
        "reference artifact hashes",
    )
    _exact(baseline.get("fold_postprocess"), FOLD_POSTPROCESS, "fold postprocess")
    _exact(baseline.get("reuse_exact_fold_postprocess"), True, "reuse postprocess")
    _exact(baseline.get("inner_reselection"), False, "inner reselection")

    comparison = _mapping(payload.get("comparison"), "comparison")
    _exact(comparison.get("primary_comparator"), EXACT_PRIMARY, "primary comparator A")
    _exact(comparison.get("candidate"), EXACT_CANDIDATE, "candidate B")
    _exact(
        comparison.get("secondary_context"),
        "original frozen natural-depth OOF; never substitutes for comparator A",
        "secondary context",
    )
    _exact(comparison.get("additional_hyperparameters"), [], "additional hyperparameters")
    _exact(
        comparison.get("candidate_refits"),
        "same frozen folds, seeds, XGBoost parameters, exactly 700 trees, and exact per-fold postprocess",
        "candidate refit contract",
    )
    for name, expected in (
        ("adaptive_search_allowed", False),
        ("inner_reselection_allowed", False),
        ("inner_or_outer_labels_for_transform", False),
        ("outer_label_values_accessed_for_metrics_only_after_both_predictions", True),
        ("frozen_event_protected_fold_membership_acknowledged", True),
        ("non_g_encoded_inputs_bitwise_equal_before_fit", True),
    ):
        _exact(comparison.get(name), expected, f"comparison.{name}")

    evaluation = _mapping(
        payload.get("one_shot_evaluation_after_separate_authorization"),
        "one_shot evaluation",
    )
    _exact(
        evaluation.get("outer_label_values_accessed_for_metrics_only_after_both_predictions"),
        True,
        "prediction-before-label contract",
    )
    _exact(
        evaluation.get("frozen_event_protected_fold_membership_acknowledged"),
        True,
        "frozen fold-membership caveat",
    )
    _exact(evaluation.get("outer_is_independent_holdout"), False, "outer holdout caveat")
    _exact(evaluation.get("primary_promotion_gates"), EXACT_GATES, "promotion gates")
    _exact(
        evaluation.get("secondary_incumbent_safety_gate"),
        {
            "comparator": "original frozen natural-depth OOF",
            "test_share_weighted_f1_delta_min": -0.001,
            "role": "veto-only incumbent safety gate; cannot replace or rescue a failed B-minus-A primary gate",
        },
        "secondary incumbent safety gate",
    )
    support = _mapping(evaluation.get("known_support_caveat"), "known support caveat")
    _exact(support.get("2025_q4_gors_positive_rows"), 0, "Q4 G-ORS support")

    authorization = _mapping(payload.get("authorization"), "authorization")
    outer_authorized = authorization.get("outer_cv") is True
    _exact(
        comparison.get("outer_execution_authorized"),
        outer_authorized,
        "comparison/authorization agreement",
    )
    for action in ("competition_upload", "commit", "push", "external_observations"):
        _exact(authorization.get(action), False, f"authorization.{action}")
    expected_status = (
        "authorized_one_shot" if outer_authorized else "implementation_only_not_outer_evaluated"
    )
    _exact(payload.get("status"), expected_status, "status/authorization agreement")
    if require_outer_authorized and not outer_authorized:
        raise GORSDepthPreregistrationError(
            "outer CV is not authorized; set status=authorized_one_shot and both outer flags true"
        )

    preregistration_sha256 = canonical_gors_depth_sha256(payload)
    exposure = 0
    family_rows = 0
    matching_preregistered = 0
    for row in ledger_rows:
        if row.get("family_id") != FAMILY_ID:
            continue
        family_rows += 1
        if (
            row.get("event") == "preregistered"
            and row.get("experiment_id") == EXPERIMENT_ID
            and row.get("preregistration_sha256") == preregistration_sha256
            and row.get("outer_result_count") == 0
        ):
            matching_preregistered += 1
        count = row.get("outer_result_count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise GORSDepthPreregistrationError("family ledger outer_result_count is invalid")
        exposure += count
        if row.get("event") in {"outer_evaluated", "closed"}:
            exposure = max(exposure, 1)
    if exposure != 0:
        raise GORSDepthPreregistrationError("family already has outer exposure; rerun prohibited")
    if require_outer_authorized and matching_preregistered != 1:
        raise GORSDepthPreregistrationError(
            "authorized run requires exactly one matching preregistered ledger event"
        )

    if observed_hashes is not None:
        keys = set(REFERENCE_HASHES)
        if set(observed_hashes) != keys:
            raise GORSDepthPreregistrationError(
                f"observed_hashes must contain exactly {sorted(keys)}"
            )
        for name, expected in REFERENCE_HASHES.items():
            _exact(str(observed_hashes[name]).lower(), expected, f"observed {name} hash")

    return {
        "status": "valid",
        "experiment_id": EXPERIMENT_ID,
        "family_id": FAMILY_ID,
        "preregistration_sha256": preregistration_sha256,
        "outer_execution_authorized": outer_authorized,
        "family_outer_exposure": exposure,
        "family_ledger_rows": family_rows,
        "primary_comparator": "A_deployment_matched_frozen_baseline",
        "candidate": "B_gors_fold_symmetric_depth_mask",
        "additional_hyperparameters": 0,
        "inner_reselection": False,
        "outer_is_independent_holdout": False,
    }


def validate_gors_depth_preregistration_files(
    preregistration_path: str | Path,
    *,
    ledger_path: str | Path | None,
    observed_hashes: Mapping[str, str] | None = None,
    require_outer_authorized: bool = False,
) -> dict[str, Any]:
    return validate_gors_depth_preregistration(
        load_gors_depth_preregistration(preregistration_path),
        ledger_rows=_read_ledger(ledger_path),
        observed_hashes=observed_hashes,
        require_outer_authorized=require_outer_authorized,
    )


__all__ = [
    "EXACT_GATES",
    "EXPERIMENT_ID",
    "FAMILY_ID",
    "FOLD_NAMES",
    "FOLD_POSTPROCESS",
    "GORSDepthPreregistrationError",
    "REFERENCE_HASHES",
    "REFERENCE_RUN_ID",
    "canonical_gors_depth_sha256",
    "load_gors_depth_preregistration",
    "validate_gors_depth_preregistration",
    "validate_gors_depth_preregistration_files",
]
