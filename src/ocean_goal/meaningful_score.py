"""Fail-closed contract for rolling meaningful official-score improvement.

The module deliberately separates three concepts:

* a locally curve-qualified challenger;
* a strict official point-score improvement; and
* a meaningful official promotion that satisfies both the curve and raw-metric gates.

It performs no upload and contains no platform credentials or network integration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

Direction = Literal["higher", "lower"]


class ContractError(ValueError):
    """Raised when evidence violates the sealed goal contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _finite_number(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ContractError(f"{name} must be finite")
    return parsed


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class ProblemContract:
    problem: str
    metric: str
    direction: Direction
    absolute_effect_threshold: float
    maximum_critical_slice_regression: float
    required_critical_slices: tuple[str, ...]

    @classmethod
    def from_mapping(cls, problem: str, value: Mapping[str, Any]) -> ProblemContract:
        direction = value.get("direction")
        if direction not in {"higher", "lower"}:
            raise ContractError(f"{problem} direction must be higher or lower")
        threshold = _finite_number(
            value.get("absolute_effect_threshold"), name=f"{problem} threshold"
        )
        regression = _finite_number(
            value.get("maximum_critical_slice_regression"),
            name=f"{problem} slice regression",
        )
        if threshold <= 0 or regression < 0:
            raise ContractError(f"{problem} thresholds must be positive")
        slices = tuple(str(item) for item in value.get("required_critical_slices", []))
        if len(slices) != len(set(slices)) or not slices:
            raise ContractError(f"{problem} required slices must be unique and non-empty")
        return cls(
            problem=problem,
            metric=str(value.get("metric", "")),
            direction=direction,
            absolute_effect_threshold=threshold,
            maximum_critical_slice_regression=regression,
            required_critical_slices=slices,
        )

    def improvement(self, candidate_minus_incumbent: float) -> float:
        return (
            candidate_minus_incumbent if self.direction == "higher" else -candidate_minus_incumbent
        )

    def improvement_interval(self, low: float, high: float) -> tuple[float, float]:
        if low > high:
            raise ContractError("confidence interval low exceeds high")
        if self.direction == "higher":
            return low, high
        return -high, -low


def load_contract(root: Path, relative: str) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ContractError("contract path must be workspace-relative and non-traversing")
    path = (workspace / relative_path).resolve(strict=True)
    if not path.is_relative_to(workspace):
        raise ContractError("contract path escapes workspace")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if parsed.get("canonical_paths", {}).get("contract") != relative:
        raise ContractError("contract does not bind its canonical path")
    expected_fractions = [0.4, 0.55, 0.7, 0.85, 1.0]
    if parsed.get("learning_curve", {}).get("training_prefix_fractions") != expected_fractions:
        raise ContractError("learning-curve fractions differ from the goal contract")
    problems = parsed.get("problems")
    if not isinstance(problems, dict) or set(problems) != {"P1", "P2", "P3"}:
        raise ContractError("contract must define exactly P1, P2, and P3")
    for problem, value in problems.items():
        ProblemContract.from_mapping(problem, value)
    scoring = parsed.get("official_scoring")
    if not isinstance(scoring, Mapping):
        raise ContractError("official_scoring must be an object")
    team_limit = scoring.get("daily_upload_limit_team_wide")
    if not isinstance(team_limit, int) or isinstance(team_limit, bool) or team_limit <= 0:
        raise ContractError("team-wide daily upload limit must be a positive integer")
    if "daily_upload_limit_per_problem" in scoring:
        raise ContractError("per-problem daily upload limit is forbidden by authenticated UI")
    return parsed


def verify_initial_pins(root: Path, contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    workspace = root.resolve(strict=True)
    verified: dict[str, dict[str, Any]] = {}
    for problem, problem_config in contract["problems"].items():
        verified[problem] = {}
        for role in ("immutable_baseline", "historical_research_only"):
            pin = problem_config[role]
            relative = Path(pin["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ContractError(f"{problem} {role} path is unsafe")
            path = (workspace / relative).resolve(strict=True)
            if not path.is_relative_to(workspace):
                raise ContractError(f"{problem} {role} path escapes workspace")
            observed = sha256_file(path)
            if observed != pin["sha256"]:
                raise ContractError(f"{problem} {role} SHA mismatch")
            verified[problem][role] = {
                "path": relative.as_posix(),
                "sha256": observed,
                "bytes": path.stat().st_size,
            }
    evidence = contract["evidence"]["meaningful_improvement_audit"]
    audit_path = (workspace / evidence["path"]).resolve(strict=True)
    if sha256_file(audit_path) != evidence["sha256"]:
        raise ContractError("meaningful-improvement audit SHA mismatch")
    ui_evidence = contract["official_scoring"].get("authenticated_ui_observation")
    if not isinstance(ui_evidence, Mapping):
        raise ContractError("authenticated UI observation pin is required")
    ui_relative = Path(str(ui_evidence.get("path", "")))
    if ui_relative.is_absolute() or not ui_relative.parts or ".." in ui_relative.parts:
        raise ContractError("authenticated UI observation path is unsafe")
    ui_path = (workspace / ui_relative).resolve(strict=True)
    if not ui_path.is_relative_to(workspace):
        raise ContractError("authenticated UI observation path escapes workspace")
    ui_sha = sha256_file(ui_path)
    if ui_sha != ui_evidence.get("sha256"):
        raise ContractError("authenticated UI observation SHA mismatch")
    ui_payload = json.loads(ui_path.read_text(encoding="utf-8"))
    ui_facts = ui_payload.get("facts", {})
    if ui_facts.get("answer_upload_daily_limit_scope") != "TEAM_WIDE":
        raise ContractError("authenticated UI does not bind a team-wide limit")
    if ui_facts.get("answer_upload_daily_limit") != contract["official_scoring"].get(
        "daily_upload_limit_team_wide"
    ):
        raise ContractError("authenticated UI and contract upload limits differ")
    verified["OFFICIAL_UI_OBSERVATION"] = {
        "source": {
            "path": ui_relative.as_posix(),
            "sha256": ui_sha,
            "bytes": ui_path.stat().st_size,
        }
    }
    return verified


def verify_curve_evidence_pins(root: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Verify pre-fit registration bytes referenced by learning-curve evidence."""

    workspace = root.resolve(strict=True)
    registration = evidence.get("preregistration")
    if not isinstance(registration, Mapping):
        raise ContractError("preregistration must be an object")
    relative = Path(str(registration.get("config_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ContractError("preregistration config path is unsafe")
    path = (workspace / relative).resolve(strict=True)
    if not path.is_relative_to(workspace):
        raise ContractError("preregistration config path escapes workspace")
    observed = sha256_file(path)
    if observed != registration.get("config_sha256"):
        raise ContractError("preregistration config SHA mismatch")
    return {
        "config_path": relative.as_posix(),
        "config_sha256": observed,
        "config_bytes": path.stat().st_size,
    }


def evaluate_learning_curve(
    contract: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    problem = str(evidence.get("problem"))
    try:
        problem_config = contract["problems"][problem]
    except KeyError as exc:
        raise ContractError(f"unknown problem {problem}") from exc
    spec = ProblemContract.from_mapping(problem, problem_config)
    expected = tuple(contract["learning_curve"]["training_prefix_fractions"])
    late = tuple(contract["learning_curve"]["late_fractions"])

    registration = evidence.get("preregistration")
    if not isinstance(registration, Mapping):
        raise ContractError("preregistration must be an object")
    hypothesis_count = registration.get("hypothesis_count")
    registration_pass = (
        bool(registration.get("generation_id"))
        and bool(registration.get("config_path"))
        and _is_sha256(registration.get("config_sha256"))
        and isinstance(hypothesis_count, int)
        and not isinstance(hypothesis_count, bool)
        and 1 <= hypothesis_count <= 3
        and registration.get("created_before_first_fit") is True
        and registration.get("score_derived_tuning") is False
    )

    protocol = evidence.get("curve_protocol")
    if not isinstance(protocol, Mapping):
        raise ContractError("curve_protocol must be an object")
    seed_ids = protocol.get("seed_ids")
    protocol_pass = (
        protocol.get("prefix_fractions") == list(expected)
        and isinstance(seed_ids, Sequence)
        and not isinstance(seed_ids, (str, bytes))
        and len(seed_ids) == contract["learning_curve"]["fixed_stochastic_seed_count"]
        and len(set(seed_ids)) == len(seed_ids)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seed_ids)
        and protocol.get("seed_aggregation") == "PREDICTION_MEAN_THEN_METRIC"
        and protocol.get("bootstrap_replicates")
        == contract["learning_curve"]["paired_cluster_bootstrap_replicates"]
        and protocol.get("bootstrap_cluster") == problem_config["bootstrap_cluster"]
        and protocol.get("incumbent_fresh_refit_each_prefix") is True
        and protocol.get("challenger_fresh_refit_each_prefix") is True
        and protocol.get("same_fold_keys_metric_postprocess") is True
        and protocol.get("incumbent_reference_seed_full_prediction_exact_to_frozen_oof") is True
    )

    points_raw = evidence.get("points")
    if not isinstance(points_raw, Sequence) or isinstance(points_raw, (str, bytes)):
        raise ContractError("points must be a sequence")
    points: dict[float, dict[str, float]] = {}
    for index, raw in enumerate(points_raw):
        if not isinstance(raw, Mapping):
            raise ContractError(f"point {index} must be an object")
        fraction = _finite_number(raw.get("fraction"), name=f"point {index} fraction")
        if fraction in points:
            raise ContractError("duplicate learning-curve fraction")
        incumbent = _finite_number(raw.get("incumbent"), name=f"point {index} incumbent")
        challenger = _finite_number(raw.get("challenger"), name=f"point {index} challenger")
        interval = raw.get("delta_ci90")
        if not isinstance(interval, Sequence) or len(interval) != 2:
            raise ContractError("delta_ci90 must have two values")
        ci_low = _finite_number(interval[0], name=f"point {index} ci low")
        ci_high = _finite_number(interval[1], name=f"point {index} ci high")
        for role in ("incumbent_seed_metrics", "challenger_seed_metrics"):
            seed_metrics = raw.get(role)
            if (
                not isinstance(seed_metrics, Sequence)
                or isinstance(seed_metrics, (str, bytes))
                or len(seed_metrics) != contract["learning_curve"]["fixed_stochastic_seed_count"]
            ):
                raise ContractError(f"point {index} {role} must contain exactly three values")
            for seed_index, value in enumerate(seed_metrics):
                _finite_number(value, name=f"point {index} {role} {seed_index}")
        improvement_ci_low, improvement_ci_high = spec.improvement_interval(ci_low, ci_high)
        delta = challenger - incumbent
        points[fraction] = {
            "incumbent": incumbent,
            "challenger": challenger,
            "delta_candidate_minus_incumbent": delta,
            "improvement": spec.improvement(delta),
            "improvement_ci90_low": improvement_ci_low,
            "improvement_ci90_high": improvement_ci_high,
        }
    if tuple(sorted(points)) != expected:
        raise ContractError(f"fractions must be exactly {expected}")

    late_all_improve = all(points[fraction]["improvement"] > 0 for fraction in late)
    full_ci_excludes_zero = points[1.0]["improvement_ci90_low"] > 0
    other_late_ci_count = sum(
        points[fraction]["improvement_ci90_low"] > 0 for fraction in late if fraction != 1.0
    )
    full_effect_pass = points[1.0]["improvement"] >= spec.absolute_effect_threshold

    fold_deltas_raw = evidence.get("fold_deltas_candidate_minus_incumbent")
    if not isinstance(fold_deltas_raw, Sequence) or len(fold_deltas_raw) != 3:
        raise ContractError("exactly three fold deltas are required")
    fold_improvements = [
        spec.improvement(_finite_number(value, name="fold delta")) for value in fold_deltas_raw
    ]
    improved_fold_count = sum(value > 0 for value in fold_improvements)
    fold_gate_pass = improved_fold_count >= int(
        contract["learning_curve"]["minimum_improved_outer_folds"]
    )

    slice_deltas_raw = evidence.get("slice_deltas_candidate_minus_incumbent")
    if not isinstance(slice_deltas_raw, Mapping):
        raise ContractError("slice deltas must be an object")
    if set(slice_deltas_raw) != set(spec.required_critical_slices):
        raise ContractError("critical-slice keys differ from the sealed contract")
    slice_improvements = {
        name: spec.improvement(_finite_number(value, name=f"slice {name} delta"))
        for name, value in slice_deltas_raw.items()
    }
    slice_gate_pass = all(
        improvement >= -spec.maximum_critical_slice_regression
        for improvement in slice_improvements.values()
    )

    leakage_checks = evidence.get("leakage_checks")
    if not isinstance(leakage_checks, Mapping) or not leakage_checks:
        raise ContractError("non-empty leakage checks are required")
    leakage_pass = all(value is True for value in leakage_checks.values())
    reproducibility_checks = evidence.get("reproducibility_checks")
    if not isinstance(reproducibility_checks, Mapping) or not reproducibility_checks:
        raise ContractError("non-empty reproducibility checks are required")
    reproducibility_pass = all(value is True for value in reproducibility_checks.values())

    gates = {
        "preregistration_contract_pass": registration_pass,
        "curve_protocol_contract_pass": protocol_pass,
        "late_fractions_all_improve": late_all_improve,
        "full_fraction_ci90_excludes_zero": full_ci_excludes_zero,
        "another_late_fraction_ci90_excludes_zero": other_late_ci_count >= 1,
        "full_effect_meets_absolute_threshold": full_effect_pass,
        "minimum_two_of_three_folds_improve": fold_gate_pass,
        "critical_slice_regression_within_limit": slice_gate_pass,
        "all_leakage_checks_pass": leakage_pass,
        "all_reproducibility_checks_pass": reproducibility_pass,
    }
    return {
        "schema_version": "meaningful_learning_curve_decision.v1",
        "problem": problem,
        "metric": spec.metric,
        "direction": spec.direction,
        "decision": "CURVE_QUALIFIED" if all(gates.values()) else "RESEARCH_ONLY",
        "passed": all(gates.values()),
        "absolute_effect_threshold": spec.absolute_effect_threshold,
        "full_fraction_improvement": points[1.0]["improvement"],
        "improved_fold_count": improved_fold_count,
        "other_late_ci_excluding_zero_count": other_late_ci_count,
        "points": [{"fraction": fraction, **points[fraction]} for fraction in expected],
        "fold_improvements": fold_improvements,
        "slice_improvements": slice_improvements,
        "gates": gates,
    }


def evaluate_official_score(
    contract: Mapping[str, Any],
    curve_decision: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    problem = str(evidence.get("problem"))
    if curve_decision.get("problem") != problem:
        raise ContractError("curve and official evidence problem mismatch")
    spec = ProblemContract.from_mapping(problem, contract["problems"][problem])
    incumbent_version = str(evidence.get("incumbent_scoring_version", ""))
    challenger_version = str(evidence.get("challenger_scoring_version", ""))
    incumbent_split = str(evidence.get("incumbent_split", ""))
    challenger_split = str(evidence.get("challenger_split", ""))
    same_surface = (
        bool(incumbent_version)
        and incumbent_version == challenger_version
        and bool(incumbent_split)
        and incumbent_split == challenger_split
    )
    incumbent_points = _finite_number(evidence.get("incumbent_points"), name="incumbent points")
    challenger_points = _finite_number(evidence.get("challenger_points"), name="challenger points")
    incumbent_metric = _finite_number(
        evidence.get("incumbent_raw_metric"), name="incumbent raw metric"
    )
    challenger_metric = _finite_number(
        evidence.get("challenger_raw_metric"), name="challenger raw metric"
    )
    raw_delta = challenger_metric - incumbent_metric
    raw_improvement = spec.improvement(raw_delta)
    score_gain = challenger_points - incumbent_points
    for name in ("incumbent_sha256", "challenger_sha256"):
        if not _is_sha256(evidence.get(name)):
            raise ContractError(f"{name} is not a lowercase SHA-256")
    operational_checks = {
        "same_scoring_version_and_split": same_surface,
        "strict_official_point_gain": score_gain > 0,
        "exact_user_approval_recorded": evidence.get("user_approval_recorded") is True,
        "saved_model_byte_reproduces": evidence.get("saved_model_byte_reproduces") is True,
        "schema_key_order_valid": evidence.get("schema_key_order_valid") is True,
    }
    score_promoted = all(operational_checks.values())
    meaningful_checks = {
        **operational_checks,
        "curve_qualified": curve_decision.get("passed") is True,
        "official_raw_effect_meets_threshold": raw_improvement >= spec.absolute_effect_threshold,
    }
    meaningful_promoted = all(meaningful_checks.values())
    final_confirmed = evidence.get("final_or_private_confirmed") is True
    if meaningful_promoted and final_confirmed:
        decision = "MEANINGFUL_PROMOTED_FINAL_CONFIRMED"
    elif meaningful_promoted:
        decision = "MEANINGFUL_PROMOTED_PROVISIONAL"
    elif score_promoted:
        decision = "SCORE_INCUMBENT_ONLY_SMALL_GAIN"
    else:
        decision = "OFFICIAL_REJECTED_OR_INCOMPARABLE"
    return {
        "schema_version": "official_score_decision.v1",
        "problem": problem,
        "decision": decision,
        "score_incumbent_updates": score_promoted,
        "meaningful_incumbent_updates": meaningful_promoted,
        "goal_problem_milestone_complete": meaningful_promoted and final_confirmed,
        "official_point_gain": score_gain,
        "official_raw_improvement": raw_improvement,
        "absolute_effect_threshold": spec.absolute_effect_threshold,
        "operational_checks": operational_checks,
        "meaningful_checks": meaningful_checks,
    }


def evaluate_goal_completion(
    contract: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the terminal predicate; local or provisional wins can never complete it."""

    problems = evidence.get("problems")
    if not isinstance(problems, Mapping) or set(problems) != set(contract["problems"]):
        raise ContractError("completion evidence must define exactly P1, P2, and P3")
    problem_checks: dict[str, dict[str, bool]] = {}
    initial_total = 0.0
    final_total = 0.0
    for problem in ("P1", "P2", "P3"):
        item = problems[problem]
        if not isinstance(item, Mapping):
            raise ContractError(f"{problem} completion evidence must be an object")
        initial_points = _finite_number(item.get("initial_points"), name=f"{problem} initial")
        final_points = _finite_number(item.get("final_points"), name=f"{problem} final")
        initial_total += initial_points
        final_total += final_points
        promotion_count = item.get("meaningful_promotion_count")
        checks = {
            "meaningful_promotion_at_least_once": (
                isinstance(promotion_count, int)
                and not isinstance(promotion_count, bool)
                and promotion_count >= 1
            ),
            "final_score_incumbent_is_problem_best": (
                item.get("final_score_incumbent_is_problem_best") is True
            ),
            "final_private_improvement_maintained": (
                item.get("final_private_improvement_maintained") is True
            ),
            "strict_problem_point_gain": final_points > initial_points,
            "scored_csv_byte_reproduced": item.get("scored_csv_byte_reproduced") is True,
            "saved_model_package_reproduced": (item.get("saved_model_package_reproduced") is True),
            "organizer_verification_confirmed": (
                item.get("organizer_verification_confirmed") is True
            ),
            "scoring_version_present": bool(item.get("scoring_version")),
            "final_split_present": bool(item.get("final_split")),
        }
        problem_checks[problem] = checks
    global_checks = {
        "all_problem_checks_pass": all(all(checks.values()) for checks in problem_checks.values()),
        "portfolio_total_strictly_improves": final_total > initial_total,
        "all_final_model_submissions_confirmed": (
            evidence.get("all_final_model_submissions_confirmed") is True
        ),
    }
    complete = all(global_checks.values())
    return {
        "schema_version": "goal_completion_decision.v1",
        "decision": "COMPLETE" if complete else "NOT_COMPLETE",
        "goal_complete": complete,
        "initial_total_points": initial_total,
        "final_total_points": final_total,
        "total_point_gain": final_total - initial_total,
        "problem_checks": problem_checks,
        "global_checks": global_checks,
    }


def validate_upload_approval(
    root: Path,
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    curve_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a user approval packet without performing an upload.

    Baseline anchors do not need a curve decision. Challengers must bind a passing
    curve decision. The exact confirmation string is intentionally inconvenient to
    synthesize accidentally and binds the problem and candidate bytes.
    """

    problem = str(receipt.get("problem", ""))
    if problem not in contract["problems"]:
        raise ContractError("approval receipt has an unknown problem")
    role = receipt.get("role")
    if role not in {"IMMUTABLE_BASELINE_ANCHOR", "CURVE_QUALIFIED_CHALLENGER"}:
        raise ContractError("approval receipt role is invalid")
    relative = Path(str(receipt.get("candidate_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ContractError("candidate path must be safe and workspace-relative")
    workspace = root.resolve(strict=True)
    candidate_path = (workspace / relative).resolve(strict=True)
    if not candidate_path.is_relative_to(workspace):
        raise ContractError("candidate path escapes workspace")
    observed_sha = sha256_file(candidate_path)
    if receipt.get("candidate_sha256") != observed_sha:
        raise ContractError("approval receipt candidate SHA mismatch")
    if role == "IMMUTABLE_BASELINE_ANCHOR":
        baseline = contract["problems"][problem]["immutable_baseline"]
        if relative.as_posix() != baseline["path"] or observed_sha != baseline["sha256"]:
            raise ContractError("baseline anchor differs from the immutable contract")
    else:
        if curve_decision is None:
            raise ContractError("challenger approval requires a curve decision")
        if curve_decision.get("problem") != problem or curve_decision.get("passed") is not True:
            raise ContractError("challenger is not curve-qualified")

    expected_confirmation = f"APPROVE_OFFICIAL_UPLOAD:{problem}:{observed_sha}"
    checks = {
        "explicit_user_confirmation": receipt.get("confirmation") == expected_confirmation,
        "user_message_reference_present": bool(receipt.get("user_message_reference")),
        "scoring_window_open": receipt.get("scoring_window_open") is True,
        "scoring_version_present": bool(receipt.get("scoring_version")),
        "split_identity_present": bool(receipt.get("split_identity")),
        "final_model_not_designated": receipt.get("final_model_already_designated") is False,
    }
    count_before = receipt.get("team_daily_upload_count_before")
    if not isinstance(count_before, int) or isinstance(count_before, bool):
        raise ContractError("team-wide daily upload count must be an integer")
    daily_limit = int(contract["official_scoring"]["daily_upload_limit_team_wide"])
    checks["daily_slot_available"] = 0 <= count_before < daily_limit
    reserve_required = contract["official_scoring"].get("reserve_one_daily_slot") is True
    if reserve_required and count_before >= daily_limit - 1:
        checks["reserve_slot_policy"] = (
            receipt.get("reserve_slot_use_reason") == "ERROR_RECOVERY"
            and receipt.get("reserve_slot_user_override") is True
        )
    else:
        checks["reserve_slot_policy"] = True
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ContractError(f"upload approval failed checks: {failed}")
    return {
        "schema_version": "upload_readiness_receipt.v1",
        "status": "UPLOAD_READY_PENDING_SEPARATE_PLATFORM_ACTION",
        "upload_performed": False,
        "problem": problem,
        "role": role,
        "candidate_path": relative.as_posix(),
        "candidate_sha256": observed_sha,
        "candidate_bytes": candidate_path.stat().st_size,
        "scoring_version": receipt["scoring_version"],
        "split_identity": receipt["split_identity"],
        "team_daily_upload_count_before": count_before,
        "team_daily_upload_count_after_if_successful": count_before + 1,
        "checks": checks,
    }


def validate_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    previous = "0" * 64
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ContractError(f"blank ledger line {line_number}")
        record = json.loads(line)
        expected_seq = len(records) + 1
        if record.get("seq") != expected_seq:
            raise ContractError(f"ledger sequence mismatch at line {line_number}")
        if record.get("previous_event_sha256") != previous:
            raise ContractError(f"ledger chain mismatch at line {line_number}")
        claimed = record.get("event_sha256")
        base = {key: value for key, value in record.items() if key != "event_sha256"}
        observed = hashlib.sha256(canonical_json_bytes(base)).hexdigest()
        if claimed != observed:
            raise ContractError(f"ledger event hash mismatch at line {line_number}")
        previous = claimed
        records.append(record)
    return records


def append_ledger_event(
    path: Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not event_type or not event_type.replace("_", "").isalnum():
        raise ContractError("event type must be a non-empty identifier")
    existing = validate_ledger(path)
    previous = existing[-1]["event_sha256"] if existing else "0" * 64
    base = {
        "seq": len(existing) + 1,
        "recorded_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "event_type": event_type,
        "previous_event_sha256": previous,
        "payload": dict(payload),
    }
    record = {**base, "event_sha256": hashlib.sha256(canonical_json_bytes(base)).hexdigest()}
    encoded = canonical_json_bytes(record) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    descriptor = os.open(path, flags, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short ledger write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    persisted = validate_ledger(path)
    if persisted[-1] != record:
        raise ContractError("ledger append failed round-trip validation")
    return record
