"""Static value-of-information preflight for expensive P1 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreflightDecision:
    decision: str
    score: int
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    nearest_closed_experiment: str | None
    nearest_similarity: float
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "score": self.score,
            "hard_failures": list(self.hard_failures),
            "warnings": list(self.warnings),
            "nearest_closed_experiment": self.nearest_closed_experiment,
            "nearest_similarity": self.nearest_similarity,
            "checks": self.checks,
        }


def _tags(value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    return frozenset(str(item).strip().lower() for item in value if str(item).strip())


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def evaluate_experiment_value(
    proposal: dict[str, Any],
    registry: dict[str, Any],
) -> PreflightDecision:
    """Reject duplicate or scientifically weak proposals before expensive execution."""

    mechanisms = _tags(proposal.get("mechanisms"))
    layer = str(proposal.get("intervention_layer", "")).strip().lower()
    closed = registry.get("closed_experiments", [])
    nearest_id: str | None = None
    nearest_similarity = 0.0
    exact_closed_match = False
    for record in closed:
        similarity = _jaccard(mechanisms, _tags(record.get("mechanisms")))
        if similarity > nearest_similarity:
            nearest_similarity = similarity
            nearest_id = str(record.get("experiment_id"))
        if (
            layer
            and layer == str(record.get("intervention_layer", "")).strip().lower()
            and similarity >= float(registry.get("duplicate_similarity_threshold", 0.67))
        ):
            exact_closed_match = True

    evaluation = proposal.get("evaluation", {})
    budget = proposal.get("budget", {})
    scope = proposal.get("scope", {})
    evidence = proposal.get("evidence", {})
    confirmation = evaluation.get("confirmation_windows", [])
    stations = evaluation.get("required_station_slices", [])
    stop_conditions = budget.get("stop_conditions", [])

    checks = {
        "binary_row_f1_primary": evaluation.get("primary_metric") == "binary_row_f1",
        "at_least_two_confirmation_windows": isinstance(confirmation, list)
        and len(set(map(str, confirmation))) >= 2,
        "all_three_station_slices": set(map(str, stations))
        >= {"G-ORS", "I-ORS", "S-ORS"},
        "low_fidelity_stage_registered": int(budget.get("low_fidelity_epochs", 0)) > 0,
        "explicit_stop_conditions": isinstance(stop_conditions, list)
        and len(stop_conditions) >= 3,
        "primary_sources_registered": isinstance(evidence.get("primary_sources"), list)
        and len(evidence["primary_sources"]) >= 2,
        "no_official_test_labels": scope.get("official_test_labels") is False,
        "no_submission_creation": scope.get("submission_created") is False,
        "mechanism_is_nonempty": len(mechanisms) >= 1,
        "not_duplicate_of_closed_family": not exact_closed_match,
    }

    hard_fail_names = {
        "binary_row_f1_primary",
        "at_least_two_confirmation_windows",
        "all_three_station_slices",
        "low_fidelity_stage_registered",
        "explicit_stop_conditions",
        "no_official_test_labels",
        "no_submission_creation",
        "mechanism_is_nonempty",
        "not_duplicate_of_closed_family",
    }
    hard_failures = tuple(name for name in hard_fail_names if not checks[name])
    warnings: list[str] = []
    if bool(evaluation.get("retrospective_confirmation", False)):
        warnings.append("confirmation windows were previously exposed; no fresh-generalization claim")
    if nearest_similarity >= 0.4 and not exact_closed_match:
        warnings.append("proposal is adjacent to a closed family; low-fidelity stop is mandatory")

    score = sum(int(value) for value in checks.values())
    minimum = int(registry.get("minimum_score", 9))
    decision = "PASS_TO_LOW_FIDELITY" if not hard_failures and score >= minimum else "REJECT_BEFORE_COMPUTE"
    return PreflightDecision(
        decision=decision,
        score=score,
        hard_failures=hard_failures,
        warnings=tuple(warnings),
        nearest_closed_experiment=nearest_id,
        nearest_similarity=float(nearest_similarity),
        checks=checks,
    )


__all__ = ["PreflightDecision", "evaluate_experiment_value"]
