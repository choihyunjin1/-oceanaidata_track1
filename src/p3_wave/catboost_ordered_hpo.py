"""Frozen contract helpers for the P3 incumbent-core CatBoost HPO.

This module is deliberately data-source agnostic.  It materializes the preregistered
48-point grid, validates the successive-halving budget, and evaluates selection and
confirmation gates from already-produced historical predictions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_ID = "p3_catboost_ordered_hpo_20260829_v1"
OFFICIAL_LEADS = (3, 6, 9, 12, 18, 24)
SHORT_LEADS = (3, 6, 9, 12)
ACTIVE_KMA_LEADS = (18, 24)
CONTROL_ID = "control_incumbent"


class HPOContractError(RuntimeError):
    """Raised when the preregistered HPO contract is not exactly satisfied."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_grid(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the exact 2x2x3x2x2 Cartesian contract in stable JSON order."""

    if spec.get("experiment_id") != EXPERIMENT_ID:
        raise HPOContractError("grid experiment id changed")
    ordering = list(spec.get("ordering", []))
    expected_order = [
        "boosting_type",
        "grow_policy",
        "depth",
        "bootstrap",
        "regularization_profile",
    ]
    if ordering != expected_order:
        raise HPOContractError("grid axis ordering changed")
    axes = spec.get("axes", {})
    values = [axes.get(name, []) for name in ordering]
    if [len(value) for value in values] != [2, 2, 3, 2, 2]:
        raise HPOContractError("grid axis cardinality changed")

    bootstrap_profiles = spec.get("bootstrap_profiles", {})
    regularization_profiles = spec.get("regularization_profiles", {})
    fixed = dict(spec.get("fixed_parameters", {}))
    candidates: list[dict[str, Any]] = []
    for number, combination in enumerate(itertools.product(*values), start=1):
        choice = dict(zip(ordering, combination, strict=True))
        bootstrap_name = str(choice.pop("bootstrap"))
        regularization_name = str(choice.pop("regularization_profile"))
        if bootstrap_name not in bootstrap_profiles:
            raise HPOContractError(f"unknown bootstrap profile: {bootstrap_name}")
        if regularization_name not in regularization_profiles:
            raise HPOContractError(f"unknown regularization profile: {regularization_name}")
        parameters = {
            **fixed,
            **choice,
            **bootstrap_profiles[bootstrap_name],
            **regularization_profiles[regularization_name],
        }
        candidates.append(
            {
                "candidate_id": f"challenger_{number:02d}",
                "bootstrap_profile": bootstrap_name,
                "regularization_profile": regularization_name,
                "parameters": parameters,
            }
        )
    if len(candidates) != int(spec.get("candidate_count", -1)) or len(candidates) != 48:
        raise HPOContractError("materialized challenger count is not 48")
    if len({row["candidate_id"] for row in candidates}) != 48:
        raise HPOContractError("challenger ids are not unique")
    return candidates


def control_candidate(spec: Mapping[str, Any]) -> dict[str, Any]:
    fixed = dict(spec["fixed_parameters"])
    fixed.update(spec["control"])
    candidate_id = str(fixed.pop("candidate_id"))
    if candidate_id != CONTROL_ID:
        raise HPOContractError("control id changed")
    return {"candidate_id": candidate_id, "parameters": fixed}


def validate_schedule(config: Mapping[str, Any]) -> int:
    schedule = config["successive_halving"]
    rungs = schedule["rungs"]
    observed = [
        (row["name"], row["iterations"], row["selection_fold_count"], row["challenger_keep"])
        for row in rungs
    ]
    expected = [
        ("rung_300", 300, 2, 12),
        ("rung_900", 900, 4, 3),
        ("rung_2500", 2500, 6, 1),
    ]
    if observed != expected:
        raise HPOContractError("successive-halving schedule changed")
    fit_count = 49 * 2 + 13 * 4 + 4 * 6
    if fit_count != 174 or fit_count != schedule["maximum_fit_count"]:
        raise HPOContractError("maximum fit budget changed")
    return fit_count


def validate_windows(config: Mapping[str, Any]) -> None:
    selection = config["selection"]
    windows = selection["windows"]
    if len(windows) != 6:
        raise HPOContractError("selection must contain six windows")
    starts = [pd.Timestamp(row[1], tz="UTC") for row in windows]
    ends = [pd.Timestamp(row[2], tz="UTC") for row in windows]
    if any(start >= end for start, end in zip(starts, ends, strict=True)):
        raise HPOContractError("selection window is empty")
    if any(left != right for left, right in zip(ends[:-1], starts[1:], strict=True)):
        raise HPOContractError("selection windows are not contiguous")
    confirmation_start = pd.Timestamp(config["confirmation"]["windows"][0][1], tz="UTC")
    if ends[-1] > confirmation_start:
        raise HPOContractError("selection overlaps confirmation")
    if selection["embargo_hours"] != 78 or selection["footprint_hours"] != 72:
        raise HPOContractError("78h embargo or 72h footprint changed")


def rank_candidates(
    aggregate: pd.DataFrame,
    parameters: Mapping[str, Mapping[str, Any]],
    *,
    tie_tolerance_rmse_m: float,
) -> pd.DataFrame:
    """Rank complete candidate aggregates with the preregistered conservative tie rule."""

    required = {"candidate_id", "squared_error_sum", "row_count"}
    if not required.issubset(aggregate.columns):
        raise HPOContractError(f"ranking aggregate missing: {sorted(required - set(aggregate))}")
    grouped = aggregate.groupby("candidate_id", sort=True, observed=True).agg(
        squared_error_sum=("squared_error_sum", "sum"), row_count=("row_count", "sum")
    )
    if (grouped["row_count"] <= 0).any() or not np.isfinite(
        grouped["squared_error_sum"].to_numpy(dtype=float)
    ).all():
        raise HPOContractError("ranking aggregate is invalid")
    grouped["rmse"] = np.sqrt(grouped["squared_error_sum"] / grouped["row_count"])
    best = float(grouped["rmse"].min())

    def tie_key(candidate_id: str) -> tuple[int, int, float, str]:
        params = parameters[candidate_id]
        return (
            0 if candidate_id == CONTROL_ID else 1,
            int(params["depth"]),
            -float(params["l2_leaf_reg"]),
            candidate_id,
        )

    frame = grouped.reset_index()
    frame["within_tie_tolerance"] = frame["rmse"].le(best + tie_tolerance_rmse_m)
    ordered_ids = sorted(
        frame.loc[frame["within_tie_tolerance"], "candidate_id"], key=tie_key
    ) + sorted(
        frame.loc[~frame["within_tie_tolerance"], "candidate_id"],
        key=lambda candidate_id: (float(grouped.loc[candidate_id, "rmse"]), tie_key(candidate_id)),
    )
    rank = {candidate_id: number for number, candidate_id in enumerate(ordered_ids, start=1)}
    frame["rank"] = frame["candidate_id"].map(rank).astype(int)
    return frame.sort_values("rank").reset_index(drop=True)


def apply_frozen_kma_alpha(
    final_prediction: Sequence[float],
    calibrated_source: Sequence[float],
    leads: Sequence[int],
    *,
    alpha: float = 0.4,
) -> np.ndarray:
    """Apply the fixed alpha-0.4 KMA correction at 18/24h and exact no-op elsewhere."""

    base = np.asarray(final_prediction, dtype=np.float64)
    source = np.asarray(calibrated_source, dtype=np.float64)
    lead = np.asarray(leads, dtype=np.int64)
    if base.shape != source.shape or base.shape != lead.shape:
        raise HPOContractError("KMA integration arrays are not aligned")
    if alpha != 0.4:
        raise HPOContractError("KMA alpha is frozen at 0.4")
    if not set(np.unique(lead)).issubset(OFFICIAL_LEADS):
        raise HPOContractError("unexpected lead in KMA integration")
    result = base.copy()
    active = np.isin(lead, ACTIVE_KMA_LEADS)
    result[active] = (1.0 - alpha) * base[active] + alpha * source[active]
    if not np.array_equal(result[~active], base[~active]):
        raise AssertionError("KMA short leads are not an exact no-op")
    return np.clip(result, 0.0, 30.0)


def _rmse(frame: pd.DataFrame, column: str) -> float:
    return float(np.sqrt(np.mean(np.square(frame[column] - frame["target_hs"]))))


def metric_deltas(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        "control_prediction",
        "challenger_prediction",
    }
    if not required.issubset(frame.columns):
        raise HPOContractError(f"metric frame missing: {sorted(required - set(frame))}")
    if frame.duplicated(["fold", "anchor_id", "station", "lead_h"]).any():
        raise HPOContractError("metric rows are duplicated")

    def slices(column: str) -> dict[str, float]:
        return {
            str(key): _rmse(group, "challenger_prediction")
            - _rmse(group, "control_prediction")
            for key, group in frame.groupby(column, sort=True, observed=True)
        }

    control = _rmse(frame, "control_prediction")
    challenger = _rmse(frame, "challenger_prediction")
    station_lead = {
        f"{station}|{int(lead)}": _rmse(group, "challenger_prediction")
        - _rmse(group, "control_prediction")
        for (station, lead), group in frame.groupby(
            ["station", "lead_h"], sort=True, observed=True
        )
    }
    short = frame.loc[frame["lead_h"].isin(SHORT_LEADS)]
    return {
        "control_rmse_m": control,
        "challenger_rmse_m": challenger,
        "delta_rmse_m": challenger - control,
        "by_fold": slices("fold"),
        "by_station": slices("station"),
        "by_lead": slices("lead_h"),
        "by_station_lead": station_lead,
        "short_lead_delta_rmse_m": _rmse(short, "challenger_prediction")
        - _rmse(short, "control_prediction"),
    }


def paired_case_bootstrap(
    frame: pd.DataFrame, *, replicates: int, seed: int
) -> dict[str, float]:
    grouped = list(frame.groupby(["fold", "anchor_id", "station"], sort=True, observed=True))
    if not grouped:
        raise HPOContractError("bootstrap has no cases")
    control_sse = np.asarray(
        [np.square(group["control_prediction"] - group["target_hs"]).sum() for _, group in grouped]
    )
    challenger_sse = np.asarray(
        [
            np.square(group["challenger_prediction"] - group["target_hs"]).sum()
            for _, group in grouped
        ]
    )
    rows = np.asarray([len(group) for _, group in grouped], dtype=np.int64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        denominator = int(rows[sampled].sum())
        deltas[index] = np.sqrt(challenger_sse[sampled].sum() / denominator) - np.sqrt(
            control_sse[sampled].sum() / denominator
        )
    return {
        "ci90_lower_m": float(np.quantile(deltas, 0.05)),
        "ci90_upper_m": float(np.quantile(deltas, 0.95)),
    }


def evaluate_selection_gate(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "pooled_delta_at_most_threshold": metrics["delta_rmse_m"]
        <= gate["pooled_delta_rmse_m_max"],
        "minimum_nonworse_folds": sum(value <= 0.0 for value in metrics["by_fold"].values())
        >= gate["minimum_nonworse_folds"],
        "minimum_nonworse_stations": sum(
            value <= 0.0 for value in metrics["by_station"].values()
        )
        >= gate["minimum_nonworse_stations"],
        "lead_18_nonworse": metrics["by_lead"]["18"] <= gate["lead_18_delta_rmse_m_max"],
        "lead_24_nonworse": metrics["by_lead"]["24"] <= gate["lead_24_delta_rmse_m_max"],
    }
    return {"checks": checks, "pass": all(checks.values())}


def evaluate_confirmation_gate(
    metrics: Mapping[str, Any], bootstrap: Mapping[str, float], gate: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "pooled_delta_at_most_threshold": metrics["delta_rmse_m"]
        <= gate["pooled_delta_rmse_m_max"],
        "bootstrap_ci90_upper_strictly_below_zero": bootstrap["ci90_upper_m"]
        < gate["paired_case_bootstrap_ci90_upper_strictly_below_m"],
        "minimum_nonworse_folds": sum(value <= 0.0 for value in metrics["by_fold"].values())
        >= gate["minimum_nonworse_folds"],
        "minimum_nonworse_stations": sum(
            value <= 0.0 for value in metrics["by_station"].values()
        )
        >= gate["minimum_nonworse_stations"],
        "lead_18_nonworse": metrics["by_lead"]["18"] <= gate["lead_18_delta_rmse_m_max"],
        "lead_24_nonworse": metrics["by_lead"]["24"] <= gate["lead_24_delta_rmse_m_max"],
        "short_lead_within_limit": metrics["short_lead_delta_rmse_m"]
        <= gate["short_lead_pooled_delta_rmse_m_max"],
        "worst_station_lead_within_limit": max(metrics["by_station_lead"].values())
        <= gate["worst_station_by_lead_delta_rmse_m_max"],
    }
    return {"checks": checks, "pass": all(checks.values())}


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ACTIVE_KMA_LEADS",
    "CONTROL_ID",
    "EXPERIMENT_ID",
    "HPOContractError",
    "apply_frozen_kma_alpha",
    "canonical_json_sha256",
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
