"""Fail-closed preregistration contract for P1 model-family experiments.

This module is intentionally independent from the training pipeline.  It does
not read observations or labels and cannot launch an experiment.  Its only job
is to reject an experiment description that would permit adaptive outer-fold
tuning, a growing search space, or repeated family-wise outer evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
FROZEN_BASELINE_RUN_ID = "20260813T153038+0900_cv_378a4e89"
FROZEN_BASELINE_OOF_SHA256 = "d1b9439db6d0d906fa080bd01f1eb8fc21d051c3d056a274e2b02e43c1e55f4a"
FROZEN_BASELINE_MICRO_F1 = 0.8603708380408055
FROZEN_BASELINE_WEIGHTED_F1 = 0.8133155525620019
OUTER_FOLDS = ("2025_q2", "2025_q3", "2025_q4")
PEER_GATE_FEATURES = (
    "peer_change_corr_24h",
    "peer_pair_coverage_24h",
    "peer_trust_gate_24h",
    "temp_abs_peer_residual_gated_24h",
)


class PreregistrationError(ValueError):
    """Raised when an experiment violates the anti-overfitting contract."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreregistrationError(f"{name} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = sorted(set(keys).difference(value))
    if missing:
        raise PreregistrationError(f"{name} is missing required keys: {missing}")


def _require_bool(value: Any, expected: bool, name: str) -> None:
    if value is not expected:
        raise PreregistrationError(f"{name} must be {expected}")


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreregistrationError(f"{name} must be numeric")
    return float(value)


def _require_exact_number(
    value: Any, expected: float, name: str, *, tolerance: float = 1e-12
) -> None:
    actual = _require_number(value, name)
    if abs(actual - expected) > tolerance:
        raise PreregistrationError(f"{name} must remain frozen at {expected}; got {actual}")


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise PreregistrationError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PreregistrationError(f"{name} must be hexadecimal") from exc


def load_preregistration(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON preregistration without accepting duplicate keys."""

    source = Path(path).expanduser().resolve(strict=True)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PreregistrationError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreregistrationError(f"cannot read preregistration {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreregistrationError("preregistration root must be an object")
    return payload


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Return the stable hash recorded by experiment runners before execution."""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def read_experiment_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Read an append-only JSONL experiment ledger and reject malformed rows."""

    source = Path(path).expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PreregistrationError(f"ledger line {line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise PreregistrationError(f"ledger line {line_number} must be an object")
        _require_keys(
            row,
            ("experiment_id", "family_id", "event", "outer_result_count"),
            f"ledger line {line_number}",
        )
        count = row["outer_result_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PreregistrationError(
                f"ledger line {line_number} outer_result_count must be a non-negative integer"
            )
        rows.append(row)
    return rows


def _validate_ledger(payload: Mapping[str, Any], ledger_rows: Sequence[Mapping[str, Any]]) -> None:
    family_id = payload["family_id"]
    experiment_id = payload["experiment_id"]
    current_outer = sum(
        int(row["outer_result_count"])
        for row in ledger_rows
        if row.get("family_id") == family_id and row.get("experiment_id") == experiment_id
    )
    current_closed = any(
        row.get("family_id") == family_id
        and row.get("experiment_id") == experiment_id
        and row.get("event") == "closed"
        for row in ledger_rows
    )
    if current_outer or current_closed:
        raise PreregistrationError(
            f"experiment {experiment_id!r} already has an outer result and its family is closed; "
            "rerun is prohibited"
        )
    previous_outer = sum(
        int(row["outer_result_count"])
        for row in ledger_rows
        if row.get("family_id") == family_id and row.get("experiment_id") != experiment_id
    )
    if previous_outer:
        raise PreregistrationError(
            f"family {family_id!r} already has {previous_outer} outer result(s); "
            "family-wise outer reuse is prohibited"
        )
    duplicate_preregistrations = sum(
        1
        for row in ledger_rows
        if row.get("experiment_id") == experiment_id and row.get("event") == "preregistered"
    )
    if duplicate_preregistrations > 1:
        raise PreregistrationError(
            f"experiment {experiment_id!r} appears more than once as preregistered"
        )


def validate_preregistration(
    payload: Mapping[str, Any],
    *,
    ledger_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate the single-hypothesis P1 experiment and return a compact receipt."""

    _require_keys(
        payload,
        (
            "schema_version",
            "experiment_id",
            "family_id",
            "status",
            "hypothesis",
            "baseline",
            "comparison",
            "inner_nuisance_selection",
            "outer_evaluation",
            "shadow_holdout",
            "stress_tests",
            "stop_rules",
            "authorization",
        ),
        "preregistration",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise PreregistrationError(f"schema_version must be {SCHEMA_VERSION!r}")
    if payload["status"] != "preregistered":
        raise PreregistrationError("status must be 'preregistered' before execution")
    for name in ("experiment_id", "family_id"):
        value = payload[name]
        if not isinstance(value, str) or not value.strip():
            raise PreregistrationError(f"{name} must be a non-empty string")

    hypothesis = _require_mapping(payload["hypothesis"], "hypothesis")
    _require_keys(
        hypothesis,
        ("single_sentence", "target_failure", "exactly_one_change", "forbidden_changes"),
        "hypothesis",
    )
    if (
        not isinstance(hypothesis["single_sentence"], str)
        or not hypothesis["single_sentence"].strip()
    ):
        raise PreregistrationError("hypothesis.single_sentence must be non-empty")
    change = _require_mapping(hypothesis["exactly_one_change"], "hypothesis.exactly_one_change")
    _require_keys(
        change,
        ("kind", "name", "feature_count", "features", "fixed_parameters"),
        "exactly_one_change",
    )
    if change["kind"] != "feature_bundle" or change["name"] != "fixed_24h_offline_peer_coherence":
        raise PreregistrationError("the only allowed change is fixed_24h_offline_peer_coherence")
    if change["feature_count"] != 4 or tuple(change["features"]) != PEER_GATE_FEATURES:
        raise PreregistrationError(
            "the fixed peer gate must add exactly the four registered features"
        )
    parameters = _require_mapping(change["fixed_parameters"], "fixed_parameters")
    expected_parameters = {"mode": "offline", "window_hours": 24, "min_period_fraction": 0.5}
    if dict(parameters) != expected_parameters:
        raise PreregistrationError(f"fixed_parameters must equal {expected_parameters}")
    forbidden = hypothesis["forbidden_changes"]
    required_forbidden = {
        "baseline_model",
        "baseline_features",
        "postprocessing",
        "threshold_grid",
        "calendar_or_station_rules",
    }
    if not isinstance(forbidden, list) or not required_forbidden.issubset(forbidden):
        raise PreregistrationError(f"forbidden_changes must include {sorted(required_forbidden)}")

    baseline = _require_mapping(payload["baseline"], "baseline")
    _require_keys(
        baseline,
        ("run_id", "model", "feature_mode", "trees", "oof_sha256", "micro_f1", "weighted_f1"),
        "baseline",
    )
    if baseline["run_id"] != FROZEN_BASELINE_RUN_ID:
        raise PreregistrationError("baseline.run_id does not identify the frozen XGBoost run")
    if (
        baseline["model"] != "xgboost"
        or baseline["feature_mode"] != "offline"
        or baseline["trees"] != 700
    ):
        raise PreregistrationError(
            "baseline model/mode/trees must remain frozen at XGBoost/offline/700"
        )
    _validate_sha256(baseline["oof_sha256"], "baseline.oof_sha256")
    if baseline["oof_sha256"] != FROZEN_BASELINE_OOF_SHA256:
        raise PreregistrationError("baseline.oof_sha256 differs from the frozen OOF")
    _require_exact_number(baseline["micro_f1"], FROZEN_BASELINE_MICRO_F1, "baseline.micro_f1")
    _require_exact_number(
        baseline["weighted_f1"], FROZEN_BASELINE_WEIGHTED_F1, "baseline.weighted_f1"
    )

    comparison = _require_mapping(payload["comparison"], "comparison")
    _require_keys(
        comparison,
        (
            "arm_count",
            "arms",
            "fixed_gate_execution_count",
            "family_selection_allowed",
            "additional_hyperparameters",
            "adaptive_search_allowed",
        ),
        "comparison",
    )
    if comparison["arm_count"] != 2:
        raise PreregistrationError("comparison.arm_count must be exactly 2 (no-op plus fixed24h)")
    arms = comparison["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        raise PreregistrationError("comparison.arms must contain exactly two entries")
    identifiers = [arm.get("id") for arm in arms if isinstance(arm, Mapping)]
    if identifiers != ["frozen_no_op", "fixed_24h_peer_coherence"]:
        raise PreregistrationError(
            "arm order/IDs must be frozen_no_op then fixed_24h_peer_coherence"
        )
    if arms[0].get("role") != "frozen_comparator" or arms[0].get("change") != "none":
        raise PreregistrationError("the no-op arm must remain a frozen comparator")
    if arms[1].get("role") != "fixed_ablation" or arms[1].get("change") != change["name"]:
        raise PreregistrationError("the gate arm must remain the fixed ablation")
    if comparison["fixed_gate_execution_count"] != 1:
        raise PreregistrationError("the fixed gate must execute exactly once")
    _require_bool(
        comparison["family_selection_allowed"], False, "comparison.family_selection_allowed"
    )
    if comparison["additional_hyperparameters"] != []:
        raise PreregistrationError("additional_hyperparameters must remain empty")
    _require_bool(
        comparison["adaptive_search_allowed"], False, "comparison.adaptive_search_allowed"
    )

    selection = _require_mapping(payload["inner_nuisance_selection"], "inner_nuisance_selection")
    _require_keys(
        selection,
        (
            "scope",
            "folds",
            "purpose",
            "applies_independently_to_each_arm",
            "selected_only",
            "model_iteration_rule",
            "postprocess_metric",
            "fixed_gate_outer_execution_depends_on_inner_win",
            "outer_labels_prohibited",
        ),
        "inner_nuisance_selection",
    )
    if selection["scope"] != "inner_validation_only" or tuple(selection["folds"]) != OUTER_FOLDS:
        raise PreregistrationError(
            "selection must use the three frozen inner-validation folds only"
        )
    if selection["purpose"] != "per_arm_iteration_and_existing_postprocess_only":
        raise PreregistrationError("inner selection may tune nuisance parameters only")
    _require_bool(
        selection["applies_independently_to_each_arm"],
        True,
        "inner_nuisance_selection.applies_independently_to_each_arm",
    )
    if selection["selected_only"] != ["model_iteration", "existing_postprocess"]:
        raise PreregistrationError(
            "inner selection is limited to model_iteration and existing_postprocess"
        )
    if selection["model_iteration_rule"] != "existing_backend_early_stopping":
        raise PreregistrationError("model iteration rule must remain the existing backend rule")
    if selection["postprocess_metric"] != "row_micro_f1":
        raise PreregistrationError("existing postprocess must remain selected by row_micro_f1")
    _require_bool(
        selection["fixed_gate_outer_execution_depends_on_inner_win"],
        False,
        "inner_nuisance_selection.fixed_gate_outer_execution_depends_on_inner_win",
    )
    _require_bool(selection["outer_labels_prohibited"], True, "selection.outer_labels_prohibited")

    outer = _require_mapping(payload["outer_evaluation"], "outer_evaluation")
    _require_keys(
        outer,
        (
            "folds",
            "one_shot",
            "max_family_outer_evaluations",
            "labels_used_for_selection",
            "adaptive_exposure_acknowledged",
            "promotion_gate",
        ),
        "outer_evaluation",
    )
    if tuple(outer["folds"]) != OUTER_FOLDS:
        raise PreregistrationError("outer folds must remain frozen")
    _require_bool(outer["one_shot"], True, "outer_evaluation.one_shot")
    if outer["max_family_outer_evaluations"] != 1:
        raise PreregistrationError("one family may produce at most one outer result")
    _require_bool(outer["labels_used_for_selection"], False, "outer labels_used_for_selection")
    _require_bool(
        outer["adaptive_exposure_acknowledged"], True, "outer adaptive_exposure_acknowledged"
    )
    gate = _require_mapping(outer["promotion_gate"], "outer promotion_gate")
    _require_keys(
        gate,
        (
            "weighted_f1_min",
            "weighted_f1_delta_min",
            "paired_bootstrap_90pct_lower_gt",
            "folds_non_degrading_min",
            "station_group_f1_drop_max",
            "normal_fp_day_relative_increase_lt",
        ),
        "promotion_gate",
    )
    minimum_weighted = FROZEN_BASELINE_WEIGHTED_F1 + 0.005
    if _require_number(gate["weighted_f1_min"], "promotion weighted_f1_min") < minimum_weighted:
        raise PreregistrationError(f"promotion weighted_f1_min must be at least {minimum_weighted}")
    if _require_number(gate["weighted_f1_delta_min"], "promotion weighted delta") < 0.005:
        raise PreregistrationError("promotion weighted-F1 delta must be at least +0.005")
    if _require_number(gate["paired_bootstrap_90pct_lower_gt"], "promotion CI gate") != 0.0:
        raise PreregistrationError(
            "paired bootstrap 90% CI lower bound must be strictly greater than 0"
        )
    if gate["folds_non_degrading_min"] != 2:
        raise PreregistrationError("at least two of three outer folds must be non-degrading")
    if _require_number(gate["station_group_f1_drop_max"], "station drop gate") > 0.01:
        raise PreregistrationError("station-group F1 drop allowance cannot exceed 0.01")
    if _require_number(gate["normal_fp_day_relative_increase_lt"], "outer FP gate") > 0.10:
        raise PreregistrationError("outer normal FP/day allowance cannot exceed 10%")

    shadow = _require_mapping(payload["shadow_holdout"], "shadow_holdout")
    _require_bool(shadow.get("available"), False, "shadow_holdout.available")
    if shadow.get("reason") != "no_rows_after_last_outer_end":
        raise PreregistrationError("shadow_holdout must record that no later train rows exist")
    if shadow.get("train_max_time_kst") != "2025-12-10T03:00:00+09:00":
        raise PreregistrationError("shadow_holdout.train_max_time_kst must remain frozen")

    stress = _require_mapping(payload["stress_tests"], "stress_tests")
    required_stress = stress.get("required")
    if required_stress != ["gors_depth_100pct_mask", "sors_year_transfer"]:
        raise PreregistrationError("both frozen stress tests are required in canonical order")
    if _require_number(stress.get("maximum_f1_drop"), "stress maximum_f1_drop") > 0.01:
        raise PreregistrationError("stress F1 drop allowance cannot exceed 0.01")

    stop = _require_mapping(payload["stop_rules"], "stop_rules")
    _require_bool(stop.get("close_family_on_any_gate_failure"), True, "stop family failure")
    _require_bool(stop.get("no_window_or_threshold_followup"), True, "stop follow-up")
    _require_bool(stop.get("no_outer_informed_erratum"), True, "stop outer-informed erratum")
    if stop.get("label_blind_bug_errata_max_before_outer") != 1:
        raise PreregistrationError(
            "at most one label-blind implementation erratum is allowed pre-outer"
        )

    authorization = _require_mapping(payload["authorization"], "authorization")
    for action in ("competition_upload", "commit", "push", "external_observations"):
        _require_bool(authorization.get(action), False, f"authorization.{action}")

    _validate_ledger(payload, ledger_rows)
    return {
        "status": "valid",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": payload["experiment_id"],
        "family_id": payload["family_id"],
        "candidate_count": 2,
        "preregistration_sha256": canonical_sha256(payload),
        "minimum_outer_weighted_f1": minimum_weighted,
        "outer_is_independent_holdout": False,
        "shadow_holdout_available": False,
    }


def record_ledger_entry(
    path: str | Path,
    entry: Mapping[str, Any],
) -> None:
    """Append one validated event to the JSONL ledger.

    This helper is for experiment runners.  Source-controlled seed ledgers are
    still edited through normal repository review; the function never rewrites
    or truncates prior history.
    """

    required = ("experiment_id", "family_id", "event", "outer_result_count")
    _require_keys(entry, required, "ledger entry")
    if entry["event"] not in {"preregistered", "inner_screened", "outer_evaluated", "closed"}:
        raise PreregistrationError("unsupported ledger event")
    count = entry["outer_result_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise PreregistrationError("outer_result_count must be a non-negative integer")
    if entry["event"] == "outer_evaluated" and count != 1:
        raise PreregistrationError("outer_evaluated entries must record exactly one outer result")
    timestamp = entry.get("recorded_at_kst")
    if timestamp is not None:
        try:
            parsed = datetime.fromisoformat(str(timestamp))
        except ValueError as exc:
            raise PreregistrationError("recorded_at_kst must be ISO-8601") from exc
        if parsed.utcoffset() is None:
            raise PreregistrationError("recorded_at_kst must include an offset")
    target = Path(path).expanduser().resolve()
    existing = read_experiment_ledger(target) if target.exists() else []
    identity = (entry["experiment_id"], entry["event"])
    if any((row["experiment_id"], row["event"]) == identity for row in existing):
        raise PreregistrationError(f"duplicate ledger event {identity}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(entry), ensure_ascii=False, sort_keys=True, allow_nan=False))
        handle.write("\n")


__all__ = [
    "PreregistrationError",
    "canonical_sha256",
    "load_preregistration",
    "read_experiment_ledger",
    "record_ledger_entry",
    "validate_preregistration",
]
