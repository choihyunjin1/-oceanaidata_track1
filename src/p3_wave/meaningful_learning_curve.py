"""Aggregate evaluation helpers for the append-only P3 learning-curve run.

The helpers operate on corrected, complete six-lead OOF cases.  They never open the
anonymous test surface and intentionally keep model fitting in the canonical runner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .corrected_repeated_forward import OFFICIAL_LEADS, paired_case_bootstrap
from .validation import rmse

PREFIX_FRACTIONS = (0.40, 0.55, 0.70, 0.85, 1.00)
LATE_FRACTIONS = (0.70, 0.85, 1.00)
HYPOTHESES = (
    "single_horizon_residual_head",
    "multi_trajectory_residual_head",
    "fixed_horizon_splice",
)
CRITICAL_SLICE_KEYS = (
    "G-ORS",
    "I-ORS",
    "S-ORS",
    "winter",
    "lead_12",
    "lead_18",
    "lead_24",
)


def chronological_prefix_ids(
    anchors: pd.DataFrame,
    train_ids: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Return the earliest frozen fraction of an already-safe outer training set."""

    if fraction not in PREFIX_FRACTIONS:
        raise ValueError("fraction differs from the preregistered learning curve")
    required = {"anchor_id", "anchor_time", "station"}
    missing = required.difference(anchors.columns)
    if missing:
        raise ValueError(f"anchor metadata is missing: {sorted(missing)}")
    source = anchors.loc[anchors["anchor_id"].isin(train_ids), list(required)].copy()
    if len(source) != len(train_ids) or source["anchor_id"].duplicated().any():
        raise ValueError("outer training IDs are missing or duplicated")
    source["anchor_time"] = pd.to_datetime(source["anchor_time"], utc=True, errors="raise")
    source = source.sort_values(["anchor_time", "station", "anchor_id"])
    count = max(1, int(math.floor(len(source) * fraction)))
    if fraction == 1.0:
        count = len(source)
    result = source.iloc[:count]["anchor_id"].to_numpy(dtype=np.int64)
    if len(result) != count or np.unique(result).size != count:
        raise AssertionError("chronological prefix IDs are not unique")
    return result


def hypothesis_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map the three preregistered structures to their unsearched pre-shrink heads."""

    required = {"single_prediction", "multi_prediction", "equal_prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"component OOF is missing: {sorted(missing)}")
    lead = frame["lead_h"].to_numpy(int)
    single = frame["single_prediction"].to_numpy(float)
    multi = frame["multi_prediction"].to_numpy(float)
    result = {
        "single_horizon_residual_head": single,
        "multi_trajectory_residual_head": multi,
        "fixed_horizon_splice": np.where(np.isin(lead, [3, 6, 9, 12]), multi, single),
    }
    for name, values in result.items():
        if not np.isfinite(values).all() or not np.all((values >= 0.0) & (values <= 30.0)):
            raise ValueError(f"{name} predictions violate finite/range checks")
    return result


def _group_delta(
    frame: pd.DataFrame,
    *,
    group_column: str,
    candidate_column: str,
    incumbent_column: str,
) -> dict[str, float]:
    return {
        str(name): float(
            rmse(group["target_hs"], group[candidate_column])
            - rmse(group["target_hs"], group[incumbent_column])
        )
        for name, group in frame.groupby(group_column, sort=True, observed=True)
    }


def evaluate_point(
    frame: pd.DataFrame,
    *,
    candidate_column: str,
    incumbent_column: str = "incumbent_prediction",
    bootstrap_replicates: int = 5_000,
    bootstrap_seed: int = 20260823,
) -> dict[str, Any]:
    """Evaluate one candidate at one prefix against the paired refit incumbent."""

    required = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        candidate_column,
        incumbent_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"learning-curve frame is missing: {sorted(missing)}")
    if frame.duplicated(["fold", "anchor_id", "station", "lead_h"]).any():
        raise ValueError("learning-curve keys are duplicated")
    leads = frame.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(values.astype(int)))
    )
    if not leads.map(lambda values: values == OFFICIAL_LEADS).all():
        raise ValueError("learning-curve cases are not complete six-lead blocks")
    truth = frame["target_hs"].to_numpy(float)
    candidate = frame[candidate_column].to_numpy(float)
    incumbent = frame[incumbent_column].to_numpy(float)
    if not np.isfinite(np.column_stack([truth, candidate, incumbent])).all():
        raise ValueError("learning-curve values are not finite")
    pooled_candidate = rmse(truth, candidate)
    pooled_incumbent = rmse(truth, incumbent)
    bootstrap = paired_case_bootstrap(
        frame,
        candidate_column=candidate_column,
        baseline_column=incumbent_column,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    ci90 = bootstrap["delta_candidate_minus_persistence_ci90_m"]
    fold_delta = _group_delta(
        frame,
        group_column="fold",
        candidate_column=candidate_column,
        incumbent_column=incumbent_column,
    )
    station_delta = _group_delta(
        frame,
        group_column="station",
        candidate_column=candidate_column,
        incumbent_column=incumbent_column,
    )
    lead_delta = _group_delta(
        frame,
        group_column="lead_h",
        candidate_column=candidate_column,
        incumbent_column=incumbent_column,
    )
    winter = frame.loc[frame["fold"].astype(str).eq("winter_transition")]
    if winter.empty:
        raise ValueError("winter_transition slice is empty")
    slice_delta = {
        "G-ORS": station_delta["G-ORS"],
        "I-ORS": station_delta["I-ORS"],
        "S-ORS": station_delta["S-ORS"],
        "winter": float(
            rmse(winter["target_hs"], winter[candidate_column])
            - rmse(winter["target_hs"], winter[incumbent_column])
        ),
        "lead_12": lead_delta["12"],
        "lead_18": lead_delta["18"],
        "lead_24": lead_delta["24"],
    }
    return {
        "incumbent_rmse_m": float(pooled_incumbent),
        "challenger_rmse_m": float(pooled_candidate),
        "delta_candidate_minus_incumbent_m": float(pooled_candidate - pooled_incumbent),
        "delta_ci90_m": [float(ci90[0]), float(ci90[1])],
        "paired_whole_case_bootstrap": {
            "cases": int(bootstrap["cases"]),
            "replicates": int(bootstrap["replicates"]),
            "seed": int(bootstrap["seed"]),
            "median_delta_m": float(bootstrap["delta_candidate_minus_persistence_median_m"]),
            "probability_candidate_improves_descriptive": float(
                bootstrap["probability_candidate_improves_descriptive"]
            ),
        },
        "fold_deltas_candidate_minus_incumbent_m": fold_delta,
        "slice_deltas_candidate_minus_incumbent_m": slice_delta,
        "improved_fold_count": int(sum(value < 0.0 for value in fold_delta.values())),
        "worst_critical_slice_regression_m": float(max(slice_delta.values())),
    }


def evaluate_hypothesis_gate(
    points: Mapping[float, Mapping[str, Any]],
    *,
    leakage_checks: Mapping[str, bool],
    reproducibility_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the sealed P3 meaningful-curve gate to one hypothesis."""

    if tuple(sorted(points)) != PREFIX_FRACTIONS:
        raise ValueError("learning-curve points differ from the five-point contract")
    full = points[1.0]
    checks = {
        "late_70_85_100_all_improve": all(
            float(points[value]["delta_candidate_minus_incumbent_m"]) < 0.0
            for value in LATE_FRACTIONS
        ),
        "full_ci90_excludes_zero": float(full["delta_ci90_m"][1]) < 0.0,
        "another_late_ci90_excludes_zero": any(
            float(points[value]["delta_ci90_m"][1]) < 0.0 for value in (0.70, 0.85)
        ),
        "full_delta_at_most_minus_0p030m": float(full["delta_candidate_minus_incumbent_m"])
        <= -0.030,
        "minimum_two_of_three_folds_improve": int(full["improved_fold_count"]) >= 2,
        "critical_slice_worst_regression_at_most_0p0075m": float(
            full["worst_critical_slice_regression_m"]
        )
        <= 0.0075,
        "all_leakage_checks_pass": bool(leakage_checks)
        and all(value is True for value in leakage_checks.values()),
        "all_reproducibility_checks_pass": bool(reproducibility_checks)
        and all(value is True for value in reproducibility_checks.values()),
    }
    return {
        "passed": bool(all(checks.values())),
        "decision": "CURVE_QUALIFIED" if all(checks.values()) else "RESEARCH_ONLY",
        "checks": checks,
    }


def central_evidence(
    points: Mapping[float, Mapping[str, Any]],
    *,
    leakage_checks: Mapping[str, bool],
    reproducibility_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Adapt one hypothesis to ``ocean_goal.meaningful_score`` exactly."""

    full = points[1.0]
    fold_order = ("2024_h2_storm", "winter_transition", "2025_h1")
    return {
        "problem": "P3",
        "points": [
            {
                "fraction": float(fraction),
                "incumbent": float(points[fraction]["incumbent_rmse_m"]),
                "challenger": float(points[fraction]["challenger_rmse_m"]),
                "delta_ci90": [float(value) for value in points[fraction]["delta_ci90_m"]],
            }
            for fraction in PREFIX_FRACTIONS
        ],
        "fold_deltas_candidate_minus_incumbent": [
            float(full["fold_deltas_candidate_minus_incumbent_m"][name]) for name in fold_order
        ],
        "slice_deltas_candidate_minus_incumbent": {
            name: float(full["slice_deltas_candidate_minus_incumbent_m"][name])
            for name in CRITICAL_SLICE_KEYS
        },
        "leakage_checks": {str(key): bool(value) for key, value in leakage_checks.items()},
        "reproducibility_checks": {
            str(key): bool(value) for key, value in reproducibility_checks.items()
        },
    }


def next_structural_generation(points: Mapping[float, Mapping[str, Any]]) -> dict[str, Any]:
    """Return exactly one preregistration recommendation after a sealed no-pass result."""

    late = [float(points[value]["delta_candidate_minus_incumbent_m"]) for value in LATE_FRACTIONS]
    return {
        "count": 1,
        "generation_id": "p3_causal_forcing_sequence_residual_v1",
        "hypothesis": (
            "A lead-coupled sequence residual model using only the causal 48-hour local "
            "wave/atmospheric forcing context can improve the full-prefix RMSE by at least "
            "0.030 m without post-hoc shrink or blend tuning."
        ),
        "design": (
            "One preregistered trajectory model with explicit temporal forcing encoders; "
            "the same corrected folds, five prefixes, three fixed seeds, and untouched "
            "promotion gate must be reused."
        ),
        "diagnostic_basis": {
            "late_delta_m": late,
            "full_delta_m": float(points[1.0]["delta_candidate_minus_incumbent_m"]),
            "interpretation": (
                "The component-only structures test representation scaling; failure at the "
                "late points indicates the next generation must change temporal representation, "
                "not coefficients, shrink, alpha, or ensemble weights."
            ),
        },
    }


__all__ = [
    "CRITICAL_SLICE_KEYS",
    "HYPOTHESES",
    "LATE_FRACTIONS",
    "PREFIX_FRACTIONS",
    "central_evidence",
    "chronological_prefix_ids",
    "evaluate_hypothesis_gate",
    "evaluate_point",
    "hypothesis_predictions",
    "next_structural_generation",
]
