"""Permanent-NO-GO contract and coverage audit for ring-residual features.

This module deliberately contains no model-training or scoring implementation.
The preregistered ring hypothesis failed its label-blind deployment-coverage
gate, so only immutable contract validation and metadata-only coverage auditing
remain available.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .features import FeatureBundle

EXPERIMENT_ID = "P1_ring_residual_inner_only_v1"
FAMILY_ID = "ring_residual_96_168h"
SCOPE_END = pd.Timestamp("2025-03-24T23:50:00+09:00").tz_convert("UTC")
RING_FEATURES = (
    "temp_ring_past_resid_96_168h",
    "temp_ring_future_resid_96_168h",
    "temp_ring_consensus_resid_96_168h",
    "temp_ring_flank_disagreement_96_168h",
)
ARM_IDS = ("frozen_features", "frozen_plus_4_ring")
FIXED_POSTPROCESS = {
    "high_threshold": 0.20,
    "low_threshold": 0.10,
    "close_gap_rows": 0,
    "minimum_positive_run": 12,
}
FIXED_MODEL = {
    "backend": "xgboost",
    "n_estimators": 700,
    "base_seed": 20260813,
    "block_seed_rule": "base_seed+block_ordinal",
    "hyperparameter_search": False,
    "early_stopping": False,
}
FIXED_SYNTHETIC_STRESS = {
    "enabled_for_model_selection": False,
    "types": ["offset", "drift"],
    "durations_hours_by_type": {"offset": [8, 24, 48, 86], "drift": [9, 24, 48, 86]},
    "signs": [-1, 1],
    "mad_amplitudes": [2, 4],
    "seed": 20260813,
    "source": "normal_contiguous_fold_fit_blocks",
    "separation": "remove every stress source row and its +/-168h context from model fitting",
    "feature_order": "inject into raw temp before rebuilding every feature",
    "copy_policy": "independent pristine frame copy per scenario",
}
FIXED_COVERAGE_CONTRACT = {
    "cadence_minutes": 10,
    "segment_boundaries_strict": True,
    "exclusion_hours_each_side": 96,
    "flank_hours": [96, 168],
    "nominal_rows_per_flank": 432,
    "min_rows_per_flank": 216,
    "require_both_flanks": True,
    "test_both_flank_coverage_upper_bound": 0.056286277224559346,
    "test_any_flank_coverage_upper_bound": 0.294341788404305,
    "zero_both_coverage_groups": 7,
    "total_test_groups": 15,
    "eligible_long_segments": 11,
    "total_test_segments": 929,
    "failure_if_both_coverage_lt": 0.85,
    "failure_if_any_group_both_coverage_lt": 0.75,
}
FIXED_STOP_REASONS = (
    "low_test_coverage",
    "internal_wave_contamination_uncontrolled",
    "composite_duration_assumption_not_guaranteed_by_problem_contract",
    "offline_future_context_requires_explicit_contract",
    "validation_covariate_context_must_not_use_validation_labels",
    "synthetic_signal_must_be_injected_before_feature_generation",
)
COVERAGE_ARTIFACT_SHA256 = "156588e92ccd64966aa8e9057dd4c4587a491e911c2bc19ea9944915a8f51f9e"


@dataclass(frozen=True)
class InnerBlock:
    name: str
    ordinal: int
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end_exclusive: pd.Timestamp
    role: str


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


FIXED_INNER_BLOCKS = (
    InnerBlock(
        "I1",
        0,
        _utc("2024-06-23T23:50:00+09:00"),
        _utc("2024-07-01T00:00:00+09:00"),
        _utc("2024-09-01T00:00:00+09:00"),
        "development",
    ),
    InnerBlock(
        "I2",
        1,
        _utc("2024-09-23T23:50:00+09:00"),
        _utc("2024-10-01T00:00:00+09:00"),
        _utc("2024-12-01T00:00:00+09:00"),
        "development",
    ),
    InnerBlock(
        "I3",
        2,
        _utc("2024-12-23T23:50:00+09:00"),
        _utc("2025-01-01T00:00:00+09:00"),
        _utc("2025-03-01T00:00:00+09:00"),
        "locked_confirmation",
    ),
)


class RingAppender(Protocol):
    """Sibling feature API accepted by a future audit caller."""

    def __call__(
        self, frame: pd.DataFrame, bundle: FeatureBundle, /, **kwargs: Any
    ) -> FeatureBundle: ...


class RingInnerContractError(ValueError):
    """Raised whenever the permanent-NO-GO contract drifts."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RingInnerContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_ring_inner_contract(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except json.JSONDecodeError as exc:
        raise RingInnerContractError(f"invalid contract JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RingInnerContractError("contract root must be an object")
    return value


def canonical_contract_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise RingInnerContractError(f"{name} differs from the permanent-NO-GO contract")


def _expected_blocks() -> list[dict[str, Any]]:
    return [
        {
            "name": block.name,
            "ordinal": block.ordinal,
            "train_end": block.train_end.isoformat(),
            "validation_start": block.validation_start.isoformat(),
            "validation_end_exclusive": block.validation_end_exclusive.isoformat(),
            "purge_days_minimum": 7,
            "dependency_embargo_minimum_minutes": 10090,
            "role": block.role,
        }
        for block in FIXED_INNER_BLOCKS
    ]


def validate_ring_inner_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact audit-only contract; model execution can never be enabled."""

    _exact(payload.get("schema_version"), "1.0-permanent-no-go", "schema_version")
    _exact(payload.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _exact(payload.get("family_id"), FAMILY_ID, "family_id")
    _exact(payload.get("audit_decision"), "permanent_no_go", "audit_decision")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise RingInnerContractError("scope must be an object")
    _exact(scope.get("maximum_time"), "2025-03-24T23:50:00+09:00", "scope maximum")
    _exact(scope.get("targets_after_scope_visible"), False, "post-scope target visibility")
    _exact(payload.get("features"), list(RING_FEATURES), "feature list")
    _exact(payload.get("arms"), list(ARM_IDS), "arms")
    _exact(payload.get("inner_blocks"), _expected_blocks(), "inner blocks")
    _exact(payload.get("model_contract"), FIXED_MODEL, "model contract")
    _exact(payload.get("postprocess"), FIXED_POSTPROCESS, "postprocess")
    _exact(payload.get("synthetic_stress"), FIXED_SYNTHETIC_STRESS, "synthetic stress")
    _exact(payload.get("coverage_contract"), FIXED_COVERAGE_CONTRACT, "coverage contract")
    _exact(payload.get("stop_reasons"), list(FIXED_STOP_REASONS), "stop reasons")
    _exact(
        payload.get("validity_caveats"),
        {
            "individual_86_5h_is_not_composite_union_upper_bound": True,
            "offline_future_dependency_hours": 168,
            "minimum_honest_boundary_gap": "168h10m",
            "inner_blocks_claimed_virgin": False,
        },
        "validity caveats",
    )
    gates = payload.get("execution_gates")
    expected_gates = {
        "outer_labels_allowed": False,
        "outer_oof_allowed": False,
        "model_fit_allowed": False,
        "official_submission_allowed": False,
        "parameter_search_allowed": False,
        "coverage_audit_allowed": True,
    }
    _exact(gates, expected_gates, "execution gates")
    provenance = payload.get("coverage_provenance")
    if not isinstance(provenance, Mapping):
        raise RingInnerContractError("coverage_provenance must be an object")
    _exact(
        provenance.get("artifact_path"),
        "artifacts/ring_coverage_audit_20260813/result.json",
        "coverage artifact path",
    )
    _exact(
        provenance.get("artifact_sha256"),
        "156588e92ccd64966aa8e9057dd4c4587a491e911c2bc19ea9944915a8f51f9e",
        "coverage artifact hash",
    )
    _exact(provenance.get("evidence_pending_reproduction"), False, "coverage evidence state")
    _exact(
        provenance.get("supersedes_buggy_both_coverage_fraction"),
        0.030980232055901686,
        "provenance",
    )
    _exact(provenance.get("corrected"), True, "provenance correction flag")
    authorization = payload.get("authorization")
    _exact(
        authorization,
        {"execution": False, "commit": False, "push": False, "competition_upload": False},
        "authorization",
    )
    return {
        "status": "valid_permanent_no_go",
        "experiment_id": EXPERIMENT_ID,
        "family_id": FAMILY_ID,
        "contract_sha256": canonical_contract_sha256(payload),
        "model_fit_allowed": False,
        "coverage_audit_allowed": True,
    }


def label_blind_scoped_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only pre-cutoff covariates; no target column crosses the API boundary."""

    if "time" not in frame:
        raise KeyError("missing time column")
    time = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    keep = time.le(SCOPE_END)
    safe = frame.loc[keep].drop(columns=["label", "anomaly_type"], errors="ignore").copy()
    if safe.empty or pd.to_datetime(safe["time"], utc=True, format="mixed").gt(SCOPE_END).any():
        raise RingInnerContractError("pre-cutoff scope is empty or escaped its boundary")
    return safe.reset_index(drop=True)


def fixed_block_indices(frame: pd.DataFrame, block_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed label-blind fit/validation positions within the sealed scope."""

    if {"label", "anomaly_type"}.intersection(frame.columns):
        raise RingInnerContractError("block indexing accepts covariates only")
    matches = [block for block in FIXED_INNER_BLOCKS if block.name == block_name]
    if len(matches) != 1:
        raise RingInnerContractError(f"unknown fixed block: {block_name}")
    block = matches[0]
    time = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    if time.gt(SCOPE_END).any():
        raise RingInnerContractError("frame contains a row after the sealed scope")
    fit = np.flatnonzero(time.le(block.train_end).to_numpy())
    validation = np.flatnonzero(
        (time.ge(block.validation_start) & time.lt(block.validation_end_exclusive)).to_numpy()
    )
    if not len(fit) or not len(validation):
        raise RingInnerContractError(f"fixed block {block_name} is empty")
    if block.validation_start - block.train_end < pd.Timedelta(days=7):
        raise RingInnerContractError("fixed block purge is shorter than seven days")
    if np.intersect1d(fit, validation).size:
        raise RingInnerContractError("fixed block fit/validation overlap")
    return fit, validation


def validate_two_arm_bundle(base: FeatureBundle, candidate: FeatureBundle) -> None:
    """Require one and only one frozen configuration change: the four ring columns."""

    if len(base.frame) != len(candidate.frame) or not base.frame.index.equals(
        candidate.frame.index
    ):
        raise RingInnerContractError("two arms must align row-for-row")
    base_columns = tuple(base.feature_columns)
    candidate_columns = tuple(candidate.feature_columns)
    if candidate_columns != (*base_columns, *RING_FEATURES):
        raise RingInnerContractError(
            "candidate arm must append exactly the four fixed ring features"
        )
    if tuple(candidate.categorical_columns) != tuple(base.categorical_columns):
        raise RingInnerContractError("ring features cannot alter categorical columns")
    if not candidate.frame.loc[:, base_columns].equals(base.frame.loc[:, base_columns]):
        raise RingInnerContractError("frozen arm feature values changed")


def assert_synthetic_holdout_separation(
    model_fit_indices: Sequence[int] | np.ndarray,
    synthetic_source_indices: Sequence[int] | np.ndarray,
) -> None:
    """Fail if a preregistered synthetic-stress source row enters model fitting."""

    fit = np.asarray(model_fit_indices, dtype=np.int64)
    stress = np.asarray(synthetic_source_indices, dtype=np.int64)
    if len(np.unique(fit)) != len(fit) or len(np.unique(stress)) != len(stress):
        raise RingInnerContractError("synthetic audit indices must be unique")
    if np.intersect1d(fit, stress).size:
        raise RingInnerContractError("synthetic stress source rows overlap model fitting")


def assert_safe_audit_path(path: str | Path) -> Path:
    """Permit diagnostics only; reject run, prediction, failure-analysis, or score paths."""

    target = Path(path)
    lowered = [part.casefold() for part in target.parts]
    joined = "/".join(lowered)
    forbidden = ("artifacts/runs", "oof", "failure", "metric", "score", "submission")
    if any(token in joined for token in forbidden):
        raise RingInnerContractError("unsafe non-audit path rejected")
    return target


def verify_coverage_artifact(path: str | Path) -> dict[str, Any]:
    """Verify the corrected aggregate-only coverage evidence and its frozen SHA."""

    target = assert_safe_audit_path(path)
    if not target.is_file():
        raise RingInnerContractError(f"coverage evidence does not exist: {target}")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    _exact(digest, COVERAGE_ARTIFACT_SHA256, "coverage artifact hash")
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except json.JSONDecodeError as exc:
        raise RingInnerContractError(f"invalid coverage evidence JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RingInnerContractError("coverage evidence root must be an object")
    _exact(payload.get("raw_observation_rows_exported"), 0, "coverage raw-row export count")
    test = payload.get("test")
    if not isinstance(test, Mapping):
        raise RingInnerContractError("coverage evidence test section is absent")
    for observed, expected, name in (
        (test.get("both_flanks_coverage"), 0.056286277224559346, "both-flank coverage"),
        (test.get("any_flank_coverage"), 0.294341788404305, "any-flank coverage"),
        (test.get("segments_total"), 929, "test segments"),
        (test.get("segments_at_least_1585_rows"), 11, "eligible test segments"),
        (test.get("station_layer_groups_total"), 15, "test groups"),
        (test.get("station_layer_groups_with_zero_both_flanks"), 7, "zero groups"),
    ):
        _exact(observed, expected, name)
    _exact(payload.get("decision"), "permanent_no_go", "coverage decision")
    return {
        "artifact_sha256": digest,
        "both_flanks_coverage": test["both_flanks_coverage"],
        "any_flank_coverage": test["any_flank_coverage"],
        "decision": "permanent_no_go",
    }


def audit_runner_ast(path: str | Path) -> None:
    """Statically prove that the audit runner has no training/prediction call surface."""

    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {
        "p1_qc.pipeline",
        "p1_qc.models_tabular",
        "p1_qc.metrics",
        "p1_qc.validation",
    }
    forbidden_calls = {
        "fit",
        "predict",
        "predict_proba",
        "read_parquet",
        "read_pickle",
        "micro_f1",
        "evaluate_predictions",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
            raise RingInnerContractError(f"audit runner imports forbidden module: {node.module}")
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
            if names.intersection(forbidden_modules):
                raise RingInnerContractError("audit runner imports a forbidden module")
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name in forbidden_calls:
                raise RingInnerContractError(f"audit runner calls forbidden function: {name}")


def deny_model_execution(*_args: Any, **_kwargs: Any) -> None:
    """Permanent fail-close; no authorization flag can revive this family."""

    raise RingInnerContractError("ring-residual model execution is permanently NO-GO")


__all__ = [
    "ARM_IDS",
    "FIXED_COVERAGE_CONTRACT",
    "FIXED_INNER_BLOCKS",
    "FIXED_SYNTHETIC_STRESS",
    "RING_FEATURES",
    "RingInnerContractError",
    "assert_safe_audit_path",
    "assert_synthetic_holdout_separation",
    "audit_runner_ast",
    "canonical_contract_sha256",
    "deny_model_execution",
    "fixed_block_indices",
    "label_blind_scoped_frame",
    "load_ring_inner_contract",
    "validate_ring_inner_contract",
    "validate_two_arm_bundle",
    "verify_coverage_artifact",
]
