"""Strict valid-combination contract for the P3 CatBoost HPO v2 cycle."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

from p3_wave.catboost_ordered_hpo import (
    CONTROL_ID,
    HPOContractError,
    apply_frozen_kma_alpha,
    evaluate_confirmation_gate,
    evaluate_selection_gate,
    metric_deltas,
    paired_case_bootstrap,
    rank_candidates,
    sha256_file,
    validate_windows,
)

EXPERIMENT_ID = "p3_catboost_valid_hpo_20260829_v2"
VALID_STRUCTURES = (
    ("Plain", "SymmetricTree"),
    ("Plain", "Depthwise"),
    ("Ordered", "SymmetricTree"),
)


def materialize_grid(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the sealed 3x3x2x2 valid-only grid in stable order."""

    if spec.get("experiment_id") != EXPERIMENT_ID:
        raise HPOContractError("v2 grid experiment id changed")
    structures = [tuple(row) for row in spec.get("structure_allowlist", [])]
    if tuple(structures) != VALID_STRUCTURES:
        raise HPOContractError("v2 structure allowlist changed")
    axes = spec.get("axes", {})
    if list(axes.get("depth", [])) != [5, 7, 9]:
        raise HPOContractError("v2 depth axis changed")
    if list(axes.get("bootstrap", [])) != ["bayesian_02", "mvs_08"]:
        raise HPOContractError("v2 bootstrap axis changed")
    if list(axes.get("regularization_profile", [])) != ["A", "B"]:
        raise HPOContractError("v2 regularization axis changed")

    fixed = dict(spec.get("fixed_parameters", {}))
    bootstrap_profiles = spec.get("bootstrap_profiles", {})
    regularization_profiles = spec.get("regularization_profiles", {})
    candidates: list[dict[str, Any]] = []
    combinations = itertools.product(
        structures,
        axes["depth"],
        axes["bootstrap"],
        axes["regularization_profile"],
    )
    for number, (structure, depth, bootstrap_name, regularization_name) in enumerate(
        combinations, start=1
    ):
        boosting_type, grow_policy = structure
        parameters = {
            **fixed,
            "boosting_type": boosting_type,
            "grow_policy": grow_policy,
            "depth": depth,
            **bootstrap_profiles[bootstrap_name],
            **regularization_profiles[regularization_name],
        }
        candidates.append(
            {
                "candidate_id": f"challenger_{number:02d}",
                "structure": f"{boosting_type}+{grow_policy}",
                "bootstrap_profile": bootstrap_name,
                "regularization_profile": regularization_name,
                "parameters": parameters,
            }
        )
    if len(candidates) != 36 or int(spec.get("candidate_count", -1)) != 36:
        raise HPOContractError("v2 challenger count is not 36")
    serialized = {repr(sorted(row["parameters"].items())) for row in candidates}
    if len(serialized) != 36:
        raise HPOContractError("v2 challenger parameters are not unique")
    return candidates


def control_candidate(spec: Mapping[str, Any]) -> dict[str, Any]:
    parameters = {**dict(spec["fixed_parameters"]), **dict(spec["control"])}
    candidate_id = str(parameters.pop("candidate_id"))
    if candidate_id != CONTROL_ID:
        raise HPOContractError("v2 control id changed")
    return {"candidate_id": candidate_id, "parameters": parameters}


def validate_schedule(config: Mapping[str, Any]) -> int:
    rungs = config["successive_halving"]["rungs"]
    observed = [
        (row["name"], row["iterations"], row["selection_fold_count"], row["challenger_keep"])
        for row in rungs
    ]
    expected = [
        ("rung_300", 300, 2, 9),
        ("rung_900", 900, 4, 3),
        ("rung_2500", 2500, 6, 1),
    ]
    if observed != expected:
        raise HPOContractError("v2 successive-halving schedule changed")
    fit_count = 37 * 2 + 10 * 4 + 4 * 6
    if fit_count != 138 or fit_count != config["successive_halving"]["maximum_fit_count"]:
        raise HPOContractError("v2 selection fit budget changed")
    return fit_count


__all__ = [
    "CONTROL_ID",
    "EXPERIMENT_ID",
    "HPOContractError",
    "VALID_STRUCTURES",
    "apply_frozen_kma_alpha",
    "control_candidate",
    "evaluate_confirmation_gate",
    "evaluate_selection_gate",
    "materialize_grid",
    "metric_deltas",
    "paired_case_bootstrap",
    "rank_candidates",
    "sha256_file",
    "validate_schedule",
    "validate_windows",
]
