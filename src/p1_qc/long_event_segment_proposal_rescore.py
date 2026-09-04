"""Frozen Cycle-1 long-event proposal and segment-rescore primitives.

This module has no filesystem entry point.  Proposal construction receives no
label or anomaly-type argument.  Labels are accepted only by the explicitly
named training-target and evaluation helpers.  The implementation constants
are derived from the preregistered design and the pre-existing label-free
``change_points`` primitive; predecessor results can select only the anchor
branch and cannot change this contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .change_points import ChangePointConfig, propose_change_intervals

EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v1"
DESIGN_SHA256 = "31b0bde27d8ef7e2b42135709563cca0bcca61c6ec6fdabefbb3530906869563"
OPERATIONAL_AMENDMENT_SHA256 = "b33f7d386e05cd7ab79976f58e9f4ab752f37cfe6a8856849867ef5f541cb276"
KEY_COLUMNS = ("station", "year", "layer", "time")
CONTEXT_BANKS_HOURS = ((24, 72), (48, 168), (24, 72, 168))
DECODER_MODES = ("CONNECTED_ONLY", "DUAL_BOUNDARY_DISCONNECTED_ALLOWED")
SEEDS = (20260826, 20260843, 20260871)
THRESHOLD_CANDIDATES = (0.75, 0.85, 0.92)
INNER_WINDOW_IDS = (
    "inner_2024_jul_aug",
    "inner_2024_oct_nov",
    "inner_2025_jan_feb",
)
OUTER_FOLDS = ("2025_q2", "2025_q3", "2025_q4")
MIN_INTERVAL_ROWS = 19
MIN_TARGET_OVERLAP = 0.80
MIN_INNER_INTERVAL_PRECISION = 0.80
CADENCE_MINUTES = 10
INNER_SEED_CELLS_PER_WINDOW = 18
INNER_ANCHOR_PHYSICAL_FITS = 9
INNER_SEGMENT_PHYSICAL_FITS = 54
OUTER_SEGMENT_PHYSICAL_FITS = 9
SEGMENT_PHYSICAL_FITS = 63
MAXIMUM_LIFETIME_PHYSICAL_FITS = 72
MAXIMUM_FEATURE_OR_PROPOSAL_MATERIALIZATIONS = 21

ROUND_B_REGISTERED_SEEDS = (20260813, 20260829, 20260847)
ROUND_B_PARAMETERS = {
    "n_estimators": 700,
    "learning_rate": 0.035,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 60,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "objective": "binary",
    "n_jobs": 8,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}
ROUND_B_DEPLOYMENT_POSTPROCESS = {
    "high_threshold": 0.2,
    "low_threshold": 0.1,
    "close_gap_rows": 0,
    "minimum_positive_run": 12,
}

# Engineering completion of proposal mechanics.  These values are inherited
# verbatim from the repository's already tested, label-free change-point
# primitive and are not selected from the predecessor result.
CHANGE_POINT_BASE = {
    "high_seed_threshold": 0.65,
    "low_seed_threshold": 0.35,
    "min_baseline_rows": 6,
    "min_return_rows": 3,
    "mean_gain_threshold": 0.5,
    "variance_gain_threshold": 0.25,
    "slope_gain_threshold": 0.25,
    "baseline_z_threshold": 3.0,
    "return_z_threshold": 3.0,
    "max_candidates_per_seed_run": 8,
    "robust_epsilon": 1.0e-6,
}

SEGMENT_MODEL_PARAMETERS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}

RESEARCH_GATES = {
    "pooled_f1_delta_gte": 0.0161,
    "paired_ci90_lower_gte": 0.007,
    "minimum_improving_outer_folds_of_3": 2,
    "minimum_improving_stations_of_3": 2,
    "equal_weight_supported_station_by_fold_f1_delta_gte": 0.008,
    "q3_f1_delta_gte": 0.0,
    "station_g_f1_delta_gte": 0.0,
    "noise_recall_delta_gte": -0.005,
    "offset_plus_drift_recall_delta_gte_any_of": (0.05,),
    "at_least_48h_event_recall_delta_gte_alternative": 0.10,
    "false_positives_per_day_ratio_lte": 1.05,
    "disconnected_interval_precision_gte": 0.75,
    "minimum_added_interval_rows_gte": 19,
}

SUBMISSION_GATES = {
    "pooled_f1_delta_gte": 0.0255,
    "paired_ci90_lower_gte": 0.012,
}


@dataclass(frozen=True)
class SegmentRecord:
    """One target-free half-open interval and its fixed structural metadata."""

    proposal_id: str
    station: str
    layer: int
    segment_id: int
    start: int
    stop: int
    start_boundary_score: float
    end_boundary_score: float
    source: str

    @property
    def duration_rows(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class InnerCellSummary:
    """Inner-only evidence used by the deterministic structure selector."""

    cell_id: str
    context_bank_hours: tuple[int, ...]
    decoder_mode: str
    threshold: float | None
    interval_precision: float
    window_f1_deltas: tuple[float, float, float]
    added_rows: int
    eligible: bool


def _deep_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def implementation_contract() -> dict[str, Any]:
    """Return the result-independent numerical/selection contract."""

    contract = {
        "experiment_id": EXPERIMENT_ID,
        "design_sha256": DESIGN_SHA256,
        "context_banks_hours": [list(bank) for bank in CONTEXT_BANKS_HOURS],
        "decoder_modes": list(DECODER_MODES),
        "seeds": list(SEEDS),
        "threshold_candidates": list(THRESHOLD_CANDIDATES),
        "minimum_interval_rows": MIN_INTERVAL_ROWS,
        "minimum_target_overlap": MIN_TARGET_OVERLAP,
        "minimum_inner_interval_precision": MIN_INNER_INTERVAL_PRECISION,
        "inner_windows": list(INNER_WINDOW_IDS),
        "outer_folds": list(OUTER_FOLDS),
        "fit_budget": {
            "inner_seed_cells_per_window": INNER_SEED_CELLS_PER_WINDOW,
            "inner_anchor_physical_fits": INNER_ANCHOR_PHYSICAL_FITS,
            "inner_segment_physical_fits": INNER_SEGMENT_PHYSICAL_FITS,
            "outer_segment_physical_fits": OUTER_SEGMENT_PHYSICAL_FITS,
            "segment_physical_fits": SEGMENT_PHYSICAL_FITS,
            "maximum_lifetime_physical_fits": MAXIMUM_LIFETIME_PHYSICAL_FITS,
            "maximum_feature_or_proposal_materializations": (
                MAXIMUM_FEATURE_OR_PROPOSAL_MATERIALIZATIONS
            ),
        },
        "round_b_anchor": {
            "registered_seeds": list(ROUND_B_REGISTERED_SEEDS),
            "fit_seed_rule": "three original registered seeds unchanged per inner window",
            "parameters": dict(ROUND_B_PARAMETERS),
            "postprocess": dict(ROUND_B_DEPLOYMENT_POSTPROCESS),
        },
        "change_point_base": dict(CHANGE_POINT_BASE),
        "segment_model_parameters": dict(SEGMENT_MODEL_PARAMETERS),
        "research_gates": dict(RESEARCH_GATES),
        "submission_gates": dict(SUBMISSION_GATES),
        "selection": (
            "lowest threshold with pooled accepted-interval precision >=0.80; "
            "then mean inner-window F1 delta, worst window, fewer rows, lexical id"
        ),
    }
    return {**contract, "contract_sha256": _deep_sha(contract)}


def assert_design_contract(design: Mapping[str, Any]) -> None:
    """Fail closed if any frozen design value used by this module differs."""

    if design.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("design experiment id changed")
    features = design["feature_contract"]
    search = design["fixed_structure_search"]
    firewall = design["data_features_target_firewall"]
    resource = design["resource_ceiling"]
    splits = design["leakage_safe_splits"]
    gates = design["decision_gates"]
    checks = {
        "context banks": tuple(tuple(v) for v in features["context_banks_hours"])
        == CONTEXT_BANKS_HOURS,
        "decoder modes": tuple(features["decoder_modes"]) == DECODER_MODES,
        "seeds": tuple(search["seeds"]) == SEEDS,
        "thresholds": tuple(search["segment_probability_threshold_candidates"])
        == THRESHOLD_CANDIDATES,
        "minimum interval": int(firewall["minimum_new_interval_rows"]) == MIN_INTERVAL_ROWS,
        "structure cells": int(search["structure_cells"]) == 6,
        "inner seed cells": int(search["inner_configuration_seed_cells"])
        == INNER_SEED_CELLS_PER_WINDOW,
        "inner fits": int(search["inner_physical_fit_calls"]) == INNER_SEGMENT_PHYSICAL_FITS,
        "outer fits": int(search["outer_locked_physical_fit_calls"]) == OUTER_SEGMENT_PHYSICAL_FITS,
        "lifetime fits search": int(search["maximum_lifetime_physical_fit_calls"])
        == SEGMENT_PHYSICAL_FITS,
        "lifetime fits resource": int(resource["maximum_lifetime_physical_fit_calls"])
        == SEGMENT_PHYSICAL_FITS,
        "inner windows": tuple(v["id"] for v in splits["inner_structure_windows"])
        == INNER_WINDOW_IDS,
        "outer folds": tuple(value.split()[0] for value in splits["outer_locked_windows"])
        == OUTER_FOLDS,
        "research gates": gates["RESEARCH_GO"]["pooled_f1_delta_gte"]
        == RESEARCH_GATES["pooled_f1_delta_gte"],
        "submission gates": gates["SUBMISSION_GO"]["pooled_f1_delta_gte"]
        == SUBMISSION_GATES["pooled_f1_delta_gte"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("frozen design contract changed: " + ", ".join(failed))


def assert_operational_amendment(amendment: Mapping[str, Any]) -> None:
    """Verify the prospective 9+54+9 operational completion exactly."""

    if amendment.get("status") != "PROSPECTIVE_OPERATIONAL_COMPLETION_NOT_AUTHORIZED":
        raise RuntimeError("operational amendment status changed")
    if amendment.get("scientific_experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("operational amendment identity changed")
    lineage = amendment["round_b_anchor_lineage"]
    inner = amendment["inner_anchor_fits"]
    resource = amendment["amended_resource_accounting"]
    checks = {
        "Round-B family": lineage["family"] == "event_day_balanced_binary_lgbm",
        "Round-B seeds": tuple(lineage["registered_seeds"]) == ROUND_B_REGISTERED_SEEDS,
        "Round-B parameters": lineage["fixed_parameters"]
        == {
            **ROUND_B_PARAMETERS,
            "feature_fraction_seed": "fit_seed",
            "bagging_seed": "fit_seed",
            "data_random_seed": "fit_seed",
            "extra_seed": "fit_seed",
            "random_state": "fit_seed",
        },
        "Round-B postprocess": {
            key: lineage["postprocess"][key] for key in ROUND_B_DEPLOYMENT_POSTPROCESS
        }
        == ROUND_B_DEPLOYMENT_POSTPROCESS,
        "anchor windows": tuple(item["id"] for item in inner["windows"]) == INNER_WINDOW_IDS,
        "anchor fits": int(inner["base_physical_fit_calls"]) == INNER_ANCHOR_PHYSICAL_FITS,
        "inner segment fits": int(resource["inner_segment_physical_fit_calls"])
        == INNER_SEGMENT_PHYSICAL_FITS,
        "outer segment fits": int(resource["outer_segment_physical_fit_calls"])
        == OUTER_SEGMENT_PHYSICAL_FITS,
        "lifetime fits": int(resource["maximum_lifetime_physical_fit_calls"])
        == MAXIMUM_LIFETIME_PHYSICAL_FITS,
        "materializations": int(
            resource["feature_or_proposal_materializations"]["maximum_lifetime"]
        )
        == MAXIMUM_FEATURE_OR_PROPOSAL_MATERIALIZATIONS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("operational amendment changed: " + ", ".join(failed))


def predecessor_anchor_branch(
    result: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    required_gate_names: Sequence[str],
) -> Literal["FROZEN_PREDECESSOR", "FROZEN_ROUND_B"]:
    """Use predecessor evidence only to select the preregistered anchor branch."""

    if result.get("status") != "COMPLETE_LOCAL_SCREEN_ONLY_PARENT_QA_PENDING":
        raise RuntimeError("predecessor lacks a terminal scientific status")
    if result.get("decision") not in {"GO_LOCAL_SCREEN_ONLY", "NO_GO_LOCAL_GATE"}:
        raise RuntimeError("predecessor lacks a scientific decision")
    checks = metrics.get("gate_checks")
    if not isinstance(checks, Mapping) or set(checks) != set(required_gate_names):
        raise RuntimeError("predecessor gate set is incomplete or changed")
    if not all(type(value) is bool for value in checks.values()):
        raise RuntimeError("predecessor gate values must be booleans")
    passed = bool(result.get("passed_all_gates"))
    if passed != bool(metrics.get("passed_all_gates")):
        raise RuntimeError("predecessor result/metrics gate decision differs")
    if passed != all(checks.values()):
        raise RuntimeError("predecessor aggregate decision is inconsistent")
    expected_decision = "GO_LOCAL_SCREEN_ONLY" if passed else "NO_GO_LOCAL_GATE"
    if result["decision"] != expected_decision:
        raise RuntimeError("predecessor decision label is inconsistent")
    return "FROZEN_PREDECESSOR" if passed else "FROZEN_ROUND_B"


def exact_gap_safe_segment_ids(frame: pd.DataFrame) -> np.ndarray:
    """Assign contiguous 10-minute station-layer segments in input order."""

    required = {"station", "layer", "time"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing segment columns: {missing}")
    if frame.empty:
        return np.empty(0, dtype=np.int64)
    time = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    station = frame["station"].astype(str).to_numpy()
    layer = frame["layer"].to_numpy()
    nanos = time.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    boundary = np.ones(len(frame), dtype=bool)
    if len(frame) > 1:
        delta = nanos[1:] - nanos[:-1]
        cadence = int(pd.Timedelta(minutes=CADENCE_MINUTES).value)
        boundary[1:] = (
            (station[1:] != station[:-1]) | (layer[1:] != layer[:-1]) | (delta != cadence)
        )
    return np.cumsum(boundary, dtype=np.int64) - 1


def _robust_z_by_segment(values: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for segment in np.unique(segment_ids):
        mask = segment_ids == segment
        current = np.asarray(values[mask], dtype=np.float64)
        finite = np.isfinite(current)
        if not finite.any():
            continue
        center = float(np.median(current[finite]))
        scale = 1.4826 * float(np.median(np.abs(current[finite] - center)))
        if not np.isfinite(scale) or scale < CHANGE_POINT_BASE["robust_epsilon"]:
            scale = max(float(np.nanstd(current)), CHANGE_POINT_BASE["robust_epsilon"])
        current[finite] = (current[finite] - center) / scale
        result[mask] = current
    return result


def _two_sided_cusum_by_segment(
    robust_z: np.ndarray,
    segment_ids: np.ndarray,
) -> np.ndarray:
    """Return a symmetric, label-free prefix/suffix CUSUM magnitude."""

    result = np.full(len(robust_z), np.nan, dtype=np.float64)
    for segment in np.unique(segment_ids):
        mask = segment_ids == segment
        values = np.asarray(robust_z[mask], dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        filled = np.where(finite, values, 0.0)
        forward = np.abs(np.cumsum(filled)) / np.sqrt(
            np.arange(1, len(filled) + 1, dtype=np.float64)
        )
        reverse = np.abs(np.cumsum(filled[::-1]))[::-1] / np.sqrt(
            np.arange(len(filled), 0, -1, dtype=np.float64)
        )
        score = np.maximum(forward, reverse)
        score[~finite] = np.nan
        result[mask] = score
    return result


def _causal_peer_residual(
    frame: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Use only same-time (therefore not future) peer observations for I/S."""

    values = frame[column].to_numpy(dtype=np.float64)
    station = frame["station"].astype(str).to_numpy()
    layer = frame["layer"].to_numpy()
    time = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    key_frame = pd.DataFrame(
        {
            "time": time,
            "layer": layer,
            "station": station,
            "value": values,
            "position": np.arange(len(frame), dtype=np.int64),
        }
    )
    residual = np.full(len(frame), np.nan, dtype=np.float64)
    peer_count = np.zeros(len(frame), dtype=np.int64)
    for _key, group in key_frame.groupby(["time", "layer"], sort=False, observed=True):
        finite_group = group[np.isfinite(group["value"].to_numpy(dtype=np.float64))]
        for row in group.itertuples(index=False):
            if row.station not in {"I-ORS", "S-ORS"} or not np.isfinite(row.value):
                continue
            peers = finite_group.loc[finite_group["station"].ne(row.station), "value"].to_numpy(
                dtype=np.float64
            )
            if len(peers):
                residual[int(row.position)] = float(row.value - np.median(peers))
                peer_count[int(row.position)] = len(peers)
    return residual, peer_count


def _measurement_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    candidates = (("temp", "temp_raw"), ("psal", "psal_raw"))
    selected = []
    for names in candidates:
        match = next((name for name in names if name in frame.columns), None)
        if match is None:
            raise KeyError(f"missing measurement column; expected one of {names}")
        selected.append(match)
    return tuple(selected)


def generate_target_free_proposals(
    frame: pd.DataFrame,
    anchor_probability: Sequence[float],
    context_bank_hours: Sequence[int],
) -> tuple[SegmentRecord, ...]:
    """Generate deterministic label-free change-point interval proposals."""

    forbidden = {"label", "anomaly_type", "derived_error_type"}.intersection(frame.columns)
    if forbidden:
        raise ValueError("proposal frame contains target/evaluation columns")
    bank = tuple(int(value) for value in context_bank_hours)
    if bank not in CONTEXT_BANKS_HOURS:
        raise ValueError("context bank is not preregistered")
    probability = np.asarray(anchor_probability, dtype=np.float64)
    if probability.shape != (len(frame),) or not np.isfinite(probability).all():
        raise ValueError("anchor probability shape/finite contract failed")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("anchor probability must lie in [0, 1]")
    segment_ids = exact_gap_safe_segment_ids(frame)
    channels = []
    cusum_channels = []
    for column in _measurement_columns(frame):
        robust_z = _robust_z_by_segment(frame[column].to_numpy(), segment_ids)
        channels.append(robust_z)
        cusum_channels.append(_two_sided_cusum_by_segment(robust_z, segment_ids))
    magnitude = np.nanmax(np.abs(np.column_stack(channels)), axis=1)
    cusum_magnitude = np.nanmax(np.column_stack(cusum_channels), axis=1)
    finite = np.isfinite(magnitude)
    proposal_probability = np.full(len(frame), np.nan, dtype=np.float64)
    # Label-free rescue seed: a physical change must coincide with an island
    # that the frozen row anchor scores as low-probability.  ``minimum`` is a
    # fixed conjunction, not a learned blend and not a target-derived rule.
    physical_change_score = np.clip(
        np.maximum(magnitude[finite], cusum_magnitude[finite]) / 6.0,
        0.0,
        1.0,
    )
    anchor_low_score = 1.0 - probability[finite]
    proposal_probability[finite] = np.clip(
        np.minimum(physical_change_score, anchor_low_score),
        0.0,
        1.0,
    )
    residual = np.nanmean(np.column_stack(channels), axis=1)
    config = ChangePointConfig(
        mode="offline",
        max_flank_rows=max(bank) * 6,
        min_interval_rows=MIN_INTERVAL_ROWS,
        max_interval_rows=max(bank) * 12,
        **CHANGE_POINT_BASE,
    )
    result = propose_change_intervals(
        residual,
        proposal_probability,
        segment_ids,
        station=frame["station"].astype(str).to_numpy(),
        layer=frame["layer"].to_numpy(),
        row_ids=np.arange(len(frame), dtype=np.int64),
        times=frame["time"].to_numpy(),
        config=config,
    )
    deduplicated: dict[tuple[str, int, int, int], SegmentRecord] = {}
    for proposal in result.proposals:
        if proposal.duration_rows < MIN_INTERVAL_ROWS:
            continue
        key = (
            str(proposal.station),
            int(proposal.layer),
            int(proposal.start),
            int(proposal.stop),
        )
        start_score = float(1.0 - np.exp(-abs(proposal.baseline_z)))
        end_score = float(1.0 - np.exp(-abs(proposal.return_z or 0.0)))
        record = SegmentRecord(
            proposal_id=proposal.proposal_id,
            station=key[0],
            layer=key[1],
            segment_id=int(proposal.segment_id),
            start=key[2],
            stop=key[3],
            start_boundary_score=start_score,
            end_boundary_score=end_score,
            source=proposal.source,
        )
        incumbent = deduplicated.get(key)
        if incumbent is None or (
            record.start_boundary_score + record.end_boundary_score
            > incumbent.start_boundary_score + incumbent.end_boundary_score
        ):
            deduplicated[key] = record
    return tuple(
        deduplicated[key]
        for key in sorted(
            deduplicated,
            key=lambda value: (value[0], value[1], value[2], value[3]),
        )
    )


def build_segment_features(
    frame: pd.DataFrame,
    anchor_probability: Sequence[float],
    anchor_prediction: Sequence[int],
    proposals: Sequence[SegmentRecord],
    context_bank_hours: Sequence[int],
) -> pd.DataFrame:
    """Build target-free robust pre/interior/post segment features."""

    forbidden = {"label", "anomaly_type", "derived_error_type"}.intersection(frame.columns)
    if forbidden:
        raise ValueError("segment feature frame contains target/evaluation columns")
    bank = tuple(int(value) for value in context_bank_hours)
    if bank not in CONTEXT_BANKS_HOURS:
        raise ValueError("context bank is not preregistered")
    probability = np.asarray(anchor_probability, dtype=np.float64)
    prediction = np.asarray(anchor_prediction, dtype=np.int8)
    if probability.shape != (len(frame),) or prediction.shape != (len(frame),):
        raise ValueError("anchor arrays differ from feature frame")
    if not np.isin(prediction, [0, 1]).all():
        raise ValueError("anchor prediction must be binary")
    measurements = _measurement_columns(frame)
    contiguous_segment_ids = exact_gap_safe_segment_ids(frame)
    peer = {column: _causal_peer_residual(frame, column) for column in measurements}
    time = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    rows: list[dict[str, float | int | str]] = []
    for proposal in proposals:
        if not 0 <= proposal.start < proposal.stop <= len(frame):
            raise ValueError("proposal coordinates exceed feature frame")
        proposal_segments = np.unique(contiguous_segment_ids[proposal.start : proposal.stop])
        if len(proposal_segments) != 1 or int(proposal_segments[0]) != proposal.segment_id:
            raise ValueError("proposal crosses or misidentifies a contiguous segment")
        segment_positions = np.flatnonzero(contiguous_segment_ids == proposal.segment_id)
        segment_start = int(segment_positions[0])
        segment_stop = int(segment_positions[-1]) + 1
        if (
            str(frame.iloc[proposal.start]["station"]) != proposal.station
            or int(frame.iloc[proposal.start]["layer"]) != proposal.layer
        ):
            raise ValueError("proposal station/layer differs from feature frame")
        interior = slice(proposal.start, proposal.stop)
        row: dict[str, float | int | str] = {
            "proposal_id": proposal.proposal_id,
            "station_code": {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}.get(proposal.station, -1),
            "layer": proposal.layer,
            "duration_rows": proposal.duration_rows,
            "start_boundary_score": proposal.start_boundary_score,
            "end_boundary_score": proposal.end_boundary_score,
            "anchor_probability_mean": float(np.mean(probability[interior])),
            "anchor_probability_min": float(np.min(probability[interior])),
            "anchor_probability_max": float(np.max(probability[interior])),
            "anchor_positive_fraction": float(np.mean(prediction[interior])),
            "anchor_negative_fraction": float(np.mean(prediction[interior] == 0)),
            "season_sin": float(np.sin(2.0 * np.pi * time.iloc[proposal.start].month / 12.0)),
            "season_cos": float(np.cos(2.0 * np.pi * time.iloc[proposal.start].month / 12.0)),
        }
        depth_column = "depth" if "depth" in frame.columns else "depth_raw"
        if depth_column not in frame.columns:
            raise KeyError("missing depth regime column")
        depth_values = frame[depth_column].to_numpy(dtype=np.float64)[interior]
        finite_depth = depth_values[np.isfinite(depth_values)]
        row["depth_regime_median"] = float(np.median(finite_depth)) if len(finite_depth) else 0.0
        for hours in bank:
            flank = int(hours) * 6
            pre = slice(max(segment_start, proposal.start - flank), proposal.start)
            post = slice(proposal.stop, min(segment_stop, proposal.stop + flank))
            for column in measurements:
                values = frame[column].to_numpy(dtype=np.float64)
                inside = values[interior]
                left = values[pre]
                right = values[post]
                finite_inside = inside[np.isfinite(inside)]
                finite_left = left[np.isfinite(left)]
                finite_right = right[np.isfinite(right)]
                prefix = f"{column}_{hours}h"
                row[f"{prefix}_missing_fraction"] = float(np.mean(~np.isfinite(inside)))
                row[f"{prefix}_interior_median"] = (
                    float(np.median(finite_inside)) if len(finite_inside) else 0.0
                )
                left_median = float(np.median(finite_left)) if len(finite_left) else 0.0
                right_median = float(np.median(finite_right)) if len(finite_right) else 0.0
                interior_median = float(row[f"{prefix}_interior_median"])
                row[f"{prefix}_pre_contrast"] = interior_median - left_median
                row[f"{prefix}_post_contrast"] = interior_median - right_median
                row[f"{prefix}_return_to_baseline"] = abs(left_median - right_median)
                leave_center_out = np.concatenate((finite_left, finite_right))
                if len(leave_center_out):
                    lco_median = float(np.median(leave_center_out))
                    lco_mad = 1.4826 * float(np.median(np.abs(leave_center_out - lco_median)))
                else:
                    lco_median = 0.0
                    lco_mad = 0.0
                row[f"{prefix}_leave_center_out_median"] = lco_median
                row[f"{prefix}_leave_center_out_mad"] = lco_mad
                if len(finite_inside) >= 2:
                    coordinate = np.arange(len(finite_inside), dtype=np.float64)
                    row[f"{prefix}_slope"] = float(np.polyfit(coordinate, finite_inside, 1)[0])
                    row[f"{prefix}_variance"] = float(np.var(finite_inside))
                else:
                    row[f"{prefix}_slope"] = 0.0
                    row[f"{prefix}_variance"] = 0.0
                if len(finite_inside) >= 3:
                    coordinate = np.arange(len(finite_inside), dtype=np.float64)
                    row[f"{prefix}_curvature"] = float(np.polyfit(coordinate, finite_inside, 2)[0])
                else:
                    row[f"{prefix}_curvature"] = 0.0
                flank_values = np.concatenate((finite_left, finite_right))
                flank_variance = float(np.var(flank_values)) if len(flank_values) >= 2 else 0.0
                row[f"{prefix}_variance_ratio"] = float(
                    row[f"{prefix}_variance"] / max(flank_variance, 1.0e-6)
                )
                peer_residual, peer_count = peer[column]
                peer_inside = peer_residual[interior]
                finite_peer = peer_inside[np.isfinite(peer_inside)]
                row[f"{prefix}_causal_peer_residual_median"] = (
                    float(np.median(finite_peer)) if len(finite_peer) else 0.0
                )
                row[f"{prefix}_causal_peer_available_fraction"] = float(
                    np.mean(peer_count[interior] > 0)
                )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    numeric = result.drop(columns=["proposal_id"])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("segment features contain non-finite values")
    return result


def segment_training_targets(
    truth: Sequence[int],
    anomaly_type: Sequence[object],
    metadata: pd.DataFrame,
    proposals: Sequence[SegmentRecord],
) -> np.ndarray:
    """Label proposals only for training, using the frozen event definition."""

    y = np.asarray(truth, dtype=np.int8)
    if y.shape != (len(metadata),) or not np.isin(y, [0, 1]).all():
        raise ValueError("truth must be a binary vector aligned to metadata")
    types = pd.Series(anomaly_type, dtype="string").fillna("")
    if len(types) != len(metadata):
        raise ValueError("anomaly type differs from metadata")
    segment_ids = exact_gap_safe_segment_ids(metadata)
    event_ids = np.full(len(metadata), -1, dtype=np.int64)
    event = -1
    previous_positive = False
    previous_segment = -1
    for index, (value, segment) in enumerate(zip(y, segment_ids, strict=True)):
        if value == 1 and (not previous_positive or segment != previous_segment):
            event += 1
        if value == 1:
            event_ids[index] = event
        previous_positive = value == 1
        previous_segment = int(segment)
    eligible_events: set[int] = set()
    for event_id in np.unique(event_ids[event_ids >= 0]):
        positions = np.flatnonzero(event_ids == event_id)
        tokens = set()
        for value in types.iloc[positions]:
            tokens.update(token.strip() for token in str(value).split("+") if token.strip())
        if len(positions) >= MIN_INTERVAL_ROWS and "spike" not in tokens:
            eligible_events.add(int(event_id))
    target = np.zeros(len(proposals), dtype=np.int8)
    for ordinal, proposal in enumerate(proposals):
        positions = np.arange(proposal.start, proposal.stop, dtype=np.int64)
        overlap_ids = set(event_ids[positions].tolist()).intersection(eligible_events)
        if len(overlap_ids) != 1:
            continue
        event_id = next(iter(overlap_ids))
        overlap = float(np.mean(event_ids[positions] == event_id))
        if overlap >= MIN_TARGET_OVERLAP:
            target[ordinal] = 1
    return target


def round_b_event_day_weight(
    metadata: pd.DataFrame,
    target: Sequence[int],
) -> np.ndarray:
    """Exact frozen Round-B event/day sample-weight implementation."""

    y = np.asarray(target, dtype=np.int8)
    if y.shape != (len(metadata),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Round-B target must be binary and aligned")
    if len(np.unique(y)) != 2:
        raise ValueError("Round-B prefix must contain both classes")
    work = metadata.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["__target"] = y
    work["__time"] = pd.to_datetime(
        work["time"],
        errors="raise",
        utc=True,
        format="mixed",
    )
    work.sort_values(["station", "layer", "__time", "__position"], inplace=True)
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(600)
    prior = grouped["__target"].shift(1).fillna(0).eq(1)
    starts = work["__target"].eq(1) & (~contiguous | ~prior)
    work["__event"] = starts.cumsum().where(work["__target"].eq(1), -1).astype(np.int64)
    positive = work["__target"].eq(1)
    event_length = work.loc[positive].groupby("__event", sort=False)["__event"].transform("size")
    pos_raw = 1.0 / np.sqrt(event_length.to_numpy(dtype=float))
    pos_raw /= pos_raw.mean()
    day = work["__time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    normal = ~positive
    normal_length = (
        work.loc[normal]
        .assign(__day=day.loc[normal])
        .groupby(["station", "layer", "__day"], sort=False, observed=True)["__day"]
        .transform("size")
    )
    normal_raw = 1.0 / np.sqrt(normal_length.to_numpy(dtype=float))
    normal_raw /= normal_raw.mean()
    ordered_weight = np.empty(len(work), dtype=np.float64)
    ordered_weight[positive.to_numpy()] = pos_raw * math.sqrt(
        max(1, int(normal.sum())) / max(1, int(positive.sum()))
    )
    ordered_weight[normal.to_numpy()] = normal_raw
    work["__weight"] = ordered_weight
    result = work.sort_values("__position", kind="mergesort")["__weight"].to_numpy(dtype=np.float32)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise RuntimeError("invalid Round-B event/day training weight")
    return result


def fit_round_b_anchor_model(
    encoded_features: np.ndarray,
    target: Sequence[int],
    metadata: pd.DataFrame,
    *,
    seed: int,
) -> Any:
    """Fit one of the nine operational Round-B anchor models."""

    if seed not in ROUND_B_REGISTERED_SEEDS:
        raise ValueError("Round-B anchor seed is not registered")
    matrix = np.asarray(encoded_features, dtype=np.float32)
    y = np.asarray(target, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[0] != len(y):
        raise ValueError("Round-B encoded feature matrix differs from target")
    weight = round_b_event_day_weight(metadata, y)
    import lightgbm as lgb

    seeded = {
        **ROUND_B_PARAMETERS,
        "random_state": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
        "extra_seed": int(seed),
    }
    model = lgb.LGBMClassifier(**seeded)
    model.fit(matrix, y, sample_weight=weight)
    return model


def fit_segment_model(
    features: pd.DataFrame,
    target: Sequence[int],
    *,
    seed: int,
) -> Any:
    """Fit one registered LightGBM segment model (one physical fit)."""

    if seed not in SEEDS:
        raise ValueError("model seed is not registered")
    y = np.asarray(target, dtype=np.int8)
    if y.shape != (len(features),) or len(np.unique(y)) != 2:
        raise ValueError("segment training target must contain both classes")
    if "proposal_id" not in features.columns:
        raise KeyError("segment features lack proposal_id")
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=int(seed),
        **SEGMENT_MODEL_PARAMETERS,
    )
    model.fit(features.drop(columns=["proposal_id"]), y)
    return model


def select_inner_threshold(
    probabilities: Sequence[float],
    interval_targets: Sequence[int],
) -> tuple[float | None, float]:
    """Choose the lowest fixed threshold meeting inner interval precision."""

    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(interval_targets, dtype=np.int8)
    if probability.shape != target.shape or not np.isin(target, [0, 1]).all():
        raise ValueError("threshold inputs differ or target is non-binary")
    if not np.isfinite(probability).all():
        raise ValueError("threshold probability is non-finite")
    for threshold in THRESHOLD_CANDIDATES:
        accepted = probability >= threshold
        if not accepted.any():
            continue
        precision = float(target[accepted].mean())
        if precision >= MIN_INNER_INTERVAL_PRECISION:
            return threshold, precision
    return None, 0.0


def select_structure_cell(cells: Sequence[InnerCellSummary]) -> InnerCellSummary:
    """Apply the exact inner-only deterministic cell selection tie-breaks."""

    eligible = [cell for cell in cells if cell.eligible and cell.threshold is not None]
    if not eligible:
        raise RuntimeError("no inner structure cell meets interval precision")
    if any(len(cell.window_f1_deltas) != 3 for cell in eligible):
        raise ValueError("every cell must contain all three inner windows")
    return sorted(
        eligible,
        key=lambda cell: (
            -float(np.mean(cell.window_f1_deltas)),
            -float(min(cell.window_f1_deltas)),
            int(cell.added_rows),
            cell.cell_id,
        ),
    )[0]


def decode_segments(
    anchor_prediction: Sequence[int],
    proposals: Sequence[SegmentRecord],
    segment_probability: Sequence[float],
    *,
    threshold: float,
    decoder_mode: str,
    spike_protected: Sequence[bool],
    flatline_protected: Sequence[bool],
) -> tuple[np.ndarray, dict[str, int]]:
    """Add accepted intervals while preserving the frozen anchor positives."""

    if threshold not in THRESHOLD_CANDIDATES or decoder_mode not in DECODER_MODES:
        raise ValueError("decoder cell is not preregistered")
    anchor = np.asarray(anchor_prediction, dtype=np.int8)
    probability = np.asarray(segment_probability, dtype=np.float64)
    spike = np.asarray(spike_protected, dtype=bool)
    flatline = np.asarray(flatline_protected, dtype=bool)
    if not np.isin(anchor, [0, 1]).all():
        raise ValueError("anchor prediction must be binary")
    if probability.shape != (len(proposals),):
        raise ValueError("segment probability differs from proposal count")
    if spike.shape != anchor.shape or flatline.shape != anchor.shape:
        raise ValueError("protection masks differ from anchor")
    candidate = anchor.copy()
    accepted = 0
    disconnected = 0
    for proposal, score in zip(proposals, probability, strict=True):
        if score < threshold:
            continue
        if not 0 <= proposal.start < proposal.stop <= len(anchor):
            raise ValueError("proposal coordinate exceeds anchor")
        interval = np.arange(proposal.start, proposal.stop, dtype=np.int64)
        left_connected = proposal.start > 0 and anchor[proposal.start - 1] == 1
        right_connected = proposal.stop < len(anchor) and anchor[proposal.stop] == 1
        overlaps_anchor = bool(anchor[interval].any())
        connected = left_connected or right_connected or overlaps_anchor
        if not connected:
            if decoder_mode == "CONNECTED_ONLY":
                continue
            if proposal.duration_rows < MIN_INTERVAL_ROWS:
                continue
            if proposal.start_boundary_score < threshold or proposal.end_boundary_score < threshold:
                continue
            disconnected += 1
        allowed = interval[~(spike[interval] | flatline[interval])]
        candidate[allowed] = 1
        accepted += 1
    if np.any(candidate < anchor):
        raise AssertionError("segment decoder removed an anchor-positive row")
    if not np.array_equal(candidate[spike], anchor[spike]):
        raise AssertionError("spike predictions changed")
    if not np.array_equal(candidate[flatline], anchor[flatline]):
        raise AssertionError("flatline predictions changed")
    return candidate, {
        "accepted_intervals": accepted,
        "accepted_disconnected_intervals": disconnected,
        "added_rows": int(np.sum((candidate == 1) & (anchor == 0))),
        "removed_anchor_rows": 0,
    }


def evaluate_decision_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate exact RESEARCH/SUBMISSION/preferred/stretch gates."""

    long_recall = (
        float(metrics["offset_plus_drift_recall_delta"])
        >= RESEARCH_GATES["offset_plus_drift_recall_delta_gte_any_of"][0]
        or float(metrics["at_least_48h_event_recall_delta"])
        >= RESEARCH_GATES["at_least_48h_event_recall_delta_gte_alternative"]
    )
    research = {
        "pooled_f1_delta": float(metrics["pooled_f1_delta"])
        >= RESEARCH_GATES["pooled_f1_delta_gte"],
        "paired_ci90_lower": float(metrics["paired_ci90_lower"])
        >= RESEARCH_GATES["paired_ci90_lower_gte"],
        "improving_outer_folds": int(metrics["improving_outer_folds"])
        >= RESEARCH_GATES["minimum_improving_outer_folds_of_3"],
        "improving_stations": int(metrics["improving_stations"])
        >= RESEARCH_GATES["minimum_improving_stations_of_3"],
        "equal_weight_supported_station_by_fold": float(
            metrics["equal_weight_supported_station_by_fold_f1_delta"]
        )
        >= RESEARCH_GATES["equal_weight_supported_station_by_fold_f1_delta_gte"],
        "q3": float(metrics["q3_f1_delta"]) >= RESEARCH_GATES["q3_f1_delta_gte"],
        "station_g": float(metrics["station_g_f1_delta"])
        >= RESEARCH_GATES["station_g_f1_delta_gte"],
        "noise_recall": float(metrics["noise_recall_delta"])
        >= RESEARCH_GATES["noise_recall_delta_gte"],
        "long_event_recall": long_recall,
        "spike_preserved": metrics["spike_predictions_exactly_preserved"] is True,
        "flatline_preserved": metrics["flatline_predictions_exactly_preserved"] is True,
        "fp_per_day": float(metrics["false_positives_per_day_ratio"])
        <= RESEARCH_GATES["false_positives_per_day_ratio_lte"],
        "disconnected_precision": float(metrics["disconnected_interval_precision"])
        >= RESEARCH_GATES["disconnected_interval_precision_gte"],
        "minimum_added_interval": int(metrics["minimum_added_interval_rows"])
        >= RESEARCH_GATES["minimum_added_interval_rows_gte"],
        "integrity": metrics["all_hash_chronology_leakage_and_key_checks"] is True,
    }
    research_go = all(research.values())
    submission = {
        "all_research_go": research_go,
        "pooled_f1_delta": float(metrics["pooled_f1_delta"])
        >= SUBMISSION_GATES["pooled_f1_delta_gte"],
        "paired_ci90_lower": float(metrics["paired_ci90_lower"])
        >= SUBMISSION_GATES["paired_ci90_lower_gte"],
        "all_seed_f1_deltas": all(float(value) >= 0.0 for value in metrics["seed_f1_deltas"]),
        "independent_aggregate_qa": metrics["independent_aggregate_QA"] == "PASS",
        "exact_reproduction": metrics["exact_reproduction_from_pinned_inputs"] is True,
    }
    return {
        "research_checks": research,
        "RESEARCH_GO": research_go,
        "submission_checks": submission,
        "SUBMISSION_GO_RESEARCH_ONLY": all(submission.values()),
        "preferred_effect": float(metrics["pooled_f1_delta"]) >= 0.0255,
        "stretch_effect": float(metrics["pooled_f1_delta"]) >= 0.055,
        "decision": (
            "RESEARCH_GO" if research_go else "RESEARCH_NO_GO_CLOSE_P1_LONG_EVENT_SEGMENT_FAMILY"
        ),
    }


__all__ = [
    "CONTEXT_BANKS_HOURS",
    "DECODER_MODES",
    "INNER_ANCHOR_PHYSICAL_FITS",
    "INNER_SEED_CELLS_PER_WINDOW",
    "INNER_SEGMENT_PHYSICAL_FITS",
    "MAXIMUM_LIFETIME_PHYSICAL_FITS",
    "OUTER_SEGMENT_PHYSICAL_FITS",
    "ROUND_B_DEPLOYMENT_POSTPROCESS",
    "ROUND_B_PARAMETERS",
    "ROUND_B_REGISTERED_SEEDS",
    "SEGMENT_PHYSICAL_FITS",
    "SEEDS",
    "THRESHOLD_CANDIDATES",
    "InnerCellSummary",
    "SegmentRecord",
    "assert_design_contract",
    "build_segment_features",
    "decode_segments",
    "evaluate_decision_gates",
    "exact_gap_safe_segment_ids",
    "fit_segment_model",
    "fit_round_b_anchor_model",
    "generate_target_free_proposals",
    "implementation_contract",
    "predecessor_anchor_branch",
    "round_b_event_day_weight",
    "segment_training_targets",
    "select_inner_threshold",
    "select_structure_cell",
]
