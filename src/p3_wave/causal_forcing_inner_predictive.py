"""Leakage-safe inner predictive utilities for P3 forcing analog v3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from p3_wave.episode_distinct_analog import (
    ACTIVE_LEADS,
    LEADS,
    EpisodeAnalogError,
    evaluate_inner_gate,
)

BLIND_CASE_COLUMNS = (
    "fold",
    "anchor_id",
    "station",
    "history_eligible",
    "conditioning_used",
    "fallback_reason",
    "query_mad_scale",
    "neighbor_anchor_ids_sha256",
    "neighbor_episode_ids_sha256",
    "neighbor_distance_mean",
    "neighbor_distance_max",
)
BLIND_PREDICTION_COLUMNS = (
    "fold",
    "anchor_id",
    "station",
    "lead_h",
    "current_hs",
    "query_mad_scale",
    "analog_applicable",
    "analog_prediction",
    "control_single_prediction",
    "control_final",
    "candidate_final",
)
EVALUATED_ADDITIONAL_COLUMNS = (
    "target_hs",
    "control_squared_error",
    "candidate_squared_error",
)


def hash_integer_array(values: np.ndarray | Sequence[int]) -> str:
    array = np.asarray(values, dtype="<i8")
    return sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class FoldScope:
    name: str
    train_ids: np.ndarray
    validation_ids: np.ndarray


class InnerTargetVault:
    """Release only chronological fit labels until blind predictions are sealed."""

    def __init__(
        self,
        official_targets: np.ndarray,
        scopes: Sequence[FoldScope],
    ) -> None:
        targets = np.asarray(official_targets, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != len(LEADS):
            raise EpisodeAnalogError("official targets must have six columns")
        if not np.isfinite(targets).all():
            raise EpisodeAnalogError("official targets must be finite")
        if len(scopes) != 3 or len({scope.name for scope in scopes}) != 3:
            raise EpisodeAnalogError("target vault requires three ordered unique folds")
        self._targets = targets
        self._scopes = tuple(scopes)
        self._by_name = {scope.name: (index, scope) for index, scope in enumerate(scopes)}
        self._blind_seal_sha256: str | None = None
        self._access_log: list[dict[str, Any]] = []
        all_validation = np.concatenate(
            [np.asarray(scope.validation_ids, dtype=np.int64) for scope in scopes]
        )
        if len(np.unique(all_validation)) != len(all_validation):
            raise EpisodeAnalogError("inner validation ids repeat across folds")

    @property
    def access_log(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._access_log]

    def read_fit(self, fold: str, anchor_ids: Sequence[int]) -> np.ndarray:
        index, scope = self._scope(fold)
        requested = np.asarray(anchor_ids, dtype=np.int64)
        expected = np.asarray(scope.train_ids, dtype=np.int64)
        if not np.array_equal(requested, expected):
            raise PermissionError(f"{fold} fit ids differ from the sealed inner split")
        current_future = np.concatenate(
            [
                np.asarray(item.validation_ids, dtype=np.int64)
                for item in self._scopes[index:]
            ]
        )
        overlap = np.intersect1d(requested, current_future)
        if overlap.size:
            raise PermissionError(
                f"{fold} fit attempted to open current/future validation labels"
            )
        prior = (
            np.concatenate(
                [
                    np.asarray(item.validation_ids, dtype=np.int64)
                    for item in self._scopes[:index]
                ]
            )
            if index
            else np.empty(0, dtype=np.int64)
        )
        self._access_log.append(
            {
                "fold": fold,
                "purpose": "control_and_analog_fit_labels",
                "anchor_count": int(len(requested)),
                "anchor_ids_sha256": hash_integer_array(requested),
                "current_or_future_validation_overlap_count": 0,
                "allowed_prior_validation_overlap_count": int(
                    np.intersect1d(requested, prior).size
                ),
                "blind_seal_required": False,
            }
        )
        return self._targets[requested].copy()

    def seal_blind_predictions(self, blind_seal_sha256: str) -> None:
        if self._blind_seal_sha256 is not None:
            raise PermissionError("blind predictions were already sealed")
        if len(blind_seal_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in blind_seal_sha256
        ):
            raise EpisodeAnalogError("blind seal must be a lowercase SHA-256")
        self._blind_seal_sha256 = blind_seal_sha256

    def read_validation(self, fold: str, anchor_ids: Sequence[int]) -> np.ndarray:
        if self._blind_seal_sha256 is None:
            raise PermissionError("validation labels require a fsynced blind seal")
        _, scope = self._scope(fold)
        requested = np.asarray(anchor_ids, dtype=np.int64)
        expected = np.asarray(scope.validation_ids, dtype=np.int64)
        if not np.array_equal(requested, expected):
            raise PermissionError(f"{fold} validation ids differ from the sealed split")
        self._access_log.append(
            {
                "fold": fold,
                "purpose": "C_inner_gate_validation_labels",
                "anchor_count": int(len(requested)),
                "anchor_ids_sha256": hash_integer_array(requested),
                "blind_seal_required": True,
                "blind_seal_sha256": self._blind_seal_sha256,
            }
        )
        return self._targets[requested].copy()

    def _scope(self, fold: str) -> tuple[int, FoldScope]:
        if fold not in self._by_name:
            raise EpisodeAnalogError(f"unknown inner fold: {fold}")
        return self._by_name[fold]


def expand_control_fit_rows(
    *,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    anchor_ids: Sequence[int],
    target_matrix: np.ndarray,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    targets = np.asarray(target_matrix, dtype=np.float64)
    if targets.shape != (len(ids), len(LEADS)) or not np.isfinite(targets).all():
        raise EpisodeAnalogError("fit target matrix shape or finiteness changed")
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    blocks: list[pd.DataFrame] = []
    residuals: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    for column, lead in enumerate(LEADS):
        block = feature_lookup.loc[ids, list(feature_columns)].reset_index(drop=True)
        station = anchor_lookup.loc[ids, "station"].astype(str).reset_index(drop=True)
        current = anchor_lookup.loc[ids, "current_hs"].to_numpy(dtype=np.float64)
        block.insert(0, "station", station)
        block.insert(1, "lead_h", str(lead))
        block.insert(2, "current_hs_for_residual", current)
        blocks.append(block)
        residuals.append(targets[:, column] - current)
        metadata.append(
            pd.DataFrame(
                {
                    "anchor_id": ids,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                }
            )
        )
    return (
        pd.concat(blocks, ignore_index=True),
        np.concatenate(residuals),
        pd.concat(metadata, ignore_index=True),
    )


def expand_control_prediction_rows(
    *,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    anchor_ids: Sequence[int],
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    blocks: list[pd.DataFrame] = []
    metadata: list[pd.DataFrame] = []
    for lead in LEADS:
        block = feature_lookup.loc[ids, list(feature_columns)].reset_index(drop=True)
        station = anchor_lookup.loc[ids, "station"].astype(str).reset_index(drop=True)
        current = anchor_lookup.loc[ids, "current_hs"].to_numpy(dtype=np.float64)
        block.insert(0, "station", station)
        block.insert(1, "lead_h", str(lead))
        block.insert(2, "current_hs_for_residual", current)
        blocks.append(block)
        metadata.append(
            pd.DataFrame(
                {
                    "anchor_id": ids,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True), pd.concat(metadata, ignore_index=True)


def apply_fixed_control_shrink(
    prediction: np.ndarray,
    current_hs: np.ndarray,
    lead_h: np.ndarray,
) -> np.ndarray:
    result = np.asarray(prediction, dtype=np.float64).copy()
    current = np.asarray(current_hs, dtype=np.float64)
    lead = np.asarray(lead_h, dtype=np.int64)
    if result.shape != current.shape or result.shape != lead.shape:
        raise EpisodeAnalogError("control shrink arrays differ in shape")
    active = np.isin(lead, ACTIVE_LEADS)
    result[active] = 0.8 * result[active] + 0.2 * current[active]
    return np.clip(result, 0.0, 30.0)


def validate_blind_cases(frame: pd.DataFrame, *, expected_cases: int) -> None:
    if tuple(frame.columns) != BLIND_CASE_COLUMNS:
        raise EpisodeAnalogError("blind case columns changed")
    if len(frame) != expected_cases or frame.duplicated(["fold", "anchor_id"]).any():
        raise EpisodeAnalogError("blind case count or uniqueness changed")
    forbidden = {column for column in frame if "target" in column or "truth" in column}
    if forbidden:
        raise EpisodeAnalogError(f"blind cases expose labels: {sorted(forbidden)}")
    if not frame["history_eligible"].isin([True, False]).all():
        raise EpisodeAnalogError("history eligibility must be boolean")
    eligible = frame["history_eligible"]
    if not np.isfinite(frame.loc[eligible, "query_mad_scale"]).all():
        raise EpisodeAnalogError("eligible blind case lacks a finite MAD scale")
    if not frame.loc[~eligible, "query_mad_scale"].isna().all():
        raise EpisodeAnalogError("ineligible blind case unexpectedly has a MAD scale")


def validate_blind_predictions(
    frame: pd.DataFrame,
    *,
    expected_cases: int,
) -> None:
    if tuple(frame.columns) != BLIND_PREDICTION_COLUMNS:
        raise EpisodeAnalogError("blind prediction columns changed")
    if len(frame) != expected_cases * len(LEADS):
        raise EpisodeAnalogError("blind prediction row count changed")
    if frame.duplicated(["fold", "anchor_id", "lead_h"]).any():
        raise EpisodeAnalogError("blind prediction keys are duplicated")
    forbidden = {column for column in frame if "target" in column or "truth" in column}
    if forbidden:
        raise EpisodeAnalogError(f"blind prediction exposes labels: {sorted(forbidden)}")
    expected_leads = set(LEADS)
    for _, group in frame.groupby(["fold", "anchor_id"], sort=False, observed=True):
        if set(group["lead_h"].astype(int)) != expected_leads:
            raise EpisodeAnalogError("blind case does not contain all six leads")
    finite_columns = (
        "current_hs",
        "control_single_prediction",
        "control_final",
        "candidate_final",
    )
    if not np.isfinite(frame.loc[:, finite_columns].to_numpy(dtype=np.float64)).all():
        raise EpisodeAnalogError("blind prediction contains a non-finite required value")
    applicable = frame["analog_applicable"].astype(bool)
    if not np.isfinite(frame.loc[applicable, "analog_prediction"]).all():
        raise EpisodeAnalogError("applicable analog prediction is non-finite")
    if not frame.loc[~applicable, "analog_prediction"].isna().all():
        raise EpisodeAnalogError("inapplicable analog prediction must be missing")
    short = frame["lead_h"].isin([3, 6, 9])
    inapplicable = ~applicable
    exact_no_op = short | inapplicable
    if not np.array_equal(
        frame.loc[exact_no_op, "candidate_final"].to_numpy(dtype=np.float64),
        frame.loc[exact_no_op, "control_final"].to_numpy(dtype=np.float64),
    ):
        raise EpisodeAnalogError("candidate changed a protected or inapplicable row")
    values = frame["candidate_final"].to_numpy(dtype=np.float64)
    if (values < 0.0).any() or (values > 30.0).any():
        raise EpisodeAnalogError("candidate lies outside [0,30]m")


def attach_validation_targets(
    blind: pd.DataFrame,
    targets_by_fold: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    result = blind.copy()
    target = np.full(len(result), np.nan, dtype=np.float64)
    lead_to_column = {lead: column for column, lead in enumerate(LEADS)}
    for fold, (anchor_ids, matrix) in targets_by_fold.items():
        ids = np.asarray(anchor_ids, dtype=np.int64)
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (len(ids), len(LEADS)):
            raise EpisodeAnalogError(f"{fold} validation target shape changed")
        id_to_row = {int(anchor_id): row for row, anchor_id in enumerate(ids)}
        mask = result["fold"].astype(str).eq(str(fold))
        positions = np.flatnonzero(mask.to_numpy())
        for position in positions:
            anchor_id = int(result.at[position, "anchor_id"])
            lead = int(result.at[position, "lead_h"])
            if anchor_id not in id_to_row or lead not in lead_to_column:
                raise EpisodeAnalogError("blind prediction key lacks a validation target")
            target[position] = values[id_to_row[anchor_id], lead_to_column[lead]]
    if not np.isfinite(target).all():
        raise EpisodeAnalogError("not every blind prediction received a target")
    result["target_hs"] = target
    result["control_squared_error"] = np.square(result["control_final"] - target)
    result["candidate_squared_error"] = np.square(result["candidate_final"] - target)
    expected = (*BLIND_PREDICTION_COLUMNS, *EVALUATED_ADDITIONAL_COLUMNS)
    if tuple(result.columns) != expected:
        raise EpisodeAnalogError("evaluated prediction columns changed")
    return result


def independently_recalculate_C_metrics(
    evaluated: pd.DataFrame,
    *,
    maximum_pooled_delta_m: float = -0.005,
    minimum_improved_folds: int = 2,
    maximum_station_degradation_m: float = 0.01,
) -> dict[str, Any]:
    required = {*BLIND_PREDICTION_COLUMNS, *EVALUATED_ADDITIONAL_COLUMNS}
    if set(evaluated.columns) != required:
        raise EpisodeAnalogError("evaluated prediction schema changed")
    control_error = np.square(
        evaluated["control_final"].to_numpy(dtype=np.float64)
        - evaluated["target_hs"].to_numpy(dtype=np.float64)
    )
    candidate_error = np.square(
        evaluated["candidate_final"].to_numpy(dtype=np.float64)
        - evaluated["target_hs"].to_numpy(dtype=np.float64)
    )
    if not np.allclose(
        control_error,
        evaluated["control_squared_error"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    ) or not np.allclose(
        candidate_error,
        evaluated["candidate_squared_error"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    ):
        raise EpisodeAnalogError("stored squared errors are not independently reproducible")
    gate = evaluate_inner_gate(
        evaluated,
        maximum_pooled_delta_m=maximum_pooled_delta_m,
        minimum_improved_folds=minimum_improved_folds,
        maximum_station_degradation_m=maximum_station_degradation_m,
    )
    gate["rows"] = int(len(evaluated))
    gate["cases"] = int(
        evaluated[["fold", "anchor_id"]].drop_duplicates().shape[0]
    )
    gate["analog_applicable_cases"] = int(
        evaluated.loc[evaluated["analog_applicable"]]
        [["fold", "anchor_id"]]
        .drop_duplicates()
        .shape[0]
    )
    return gate


__all__ = [
    "BLIND_CASE_COLUMNS",
    "BLIND_PREDICTION_COLUMNS",
    "EVALUATED_ADDITIONAL_COLUMNS",
    "FoldScope",
    "InnerTargetVault",
    "apply_fixed_control_shrink",
    "attach_validation_targets",
    "expand_control_fit_rows",
    "expand_control_prediction_rows",
    "hash_integer_array",
    "independently_recalculate_C_metrics",
    "validate_blind_cases",
    "validate_blind_predictions",
]
