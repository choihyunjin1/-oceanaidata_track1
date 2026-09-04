"""Core helpers for the frozen P3 lead-continuous fresh-episode check.

The helpers deliberately keep target columns out of the prediction path.  They
select anchors using timestamps and station identity, reconstruct the saved
CatBoost/router reference by inference only, and summarize paired six-lead RMSE
after a blind-prediction seal has been created by the runner.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd

from p3_wave.loss_router import (
    OBSERVED_FEATURES,
    build_inference_router_features,
    expand_case_router_features,
    route_row_predictions,
)
from p3_wave.persistence_shrink import (
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_wave.validation import rmse

LEADS = (3, 6, 9, 12, 18, 24)
ACTIVE_ROUTER_LEADS = (12, 18, 24)


class FrozenRouter(Protocol):
    """Minimal interface supplied by the saved component-loss router."""

    def predict_weights(self, frame: pd.DataFrame) -> np.ndarray: ...


def array_sha256(*arrays: np.ndarray) -> str:
    """Hash typed array values without serializing prediction rows."""

    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def frame_contract_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash a key/time contract while emitting no individual values."""

    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"frame contract columns missing: {sorted(missing)}")
    digest = hashlib.sha256()
    ordered = frame.loc[:, list(columns)].sort_values(list(columns), kind="stable")
    for row in ordered.itertuples(index=False, name=None):
        encoded: list[str] = []
        for value in row:
            if isinstance(value, pd.Timestamp):
                encoded.append(value.isoformat())
            else:
                encoded.append(str(value))
        digest.update(("\x1f".join(encoded) + "\n").encode("utf-8"))
    return digest.hexdigest()


def select_fresh_surface(
    anchors: pd.DataFrame,
    exposed_times: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    separation_hours: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a station-global chronological greedy surface disjoint from exposure.

    Selection uses only anchor identifiers, station, and time.  Target values and
    incumbent/candidate predictions are neither required nor inspected.
    """

    required_anchor = {"anchor_id", "station", "anchor_time"}
    required_exposed = {"station", "anchor_time"}
    if missing := required_anchor.difference(anchors.columns):
        raise ValueError(f"anchor metadata missing: {sorted(missing)}")
    if missing := required_exposed.difference(exposed_times.columns):
        raise ValueError(f"exposure metadata missing: {sorted(missing)}")
    if separation_hours <= 0:
        raise ValueError("separation_hours must be positive")

    metadata = anchors.loc[:, ["anchor_id", "station", "anchor_time"]].copy()
    metadata["anchor_time"] = pd.to_datetime(metadata["anchor_time"], utc=True)
    exposure = exposed_times.loc[:, ["station", "anchor_time"]].drop_duplicates().copy()
    exposure["anchor_time"] = pd.to_datetime(exposure["anchor_time"], utc=True)
    if metadata["anchor_time"].isna().any() or exposure["anchor_time"].isna().any():
        raise ValueError("anchor/exposure timestamps must be complete")

    window = metadata.loc[
        metadata["anchor_time"].ge(start) & metadata["anchor_time"].lt(end)
    ].copy()
    fresh_indices: list[int] = []
    nearest_gap: list[float] = []
    for index, row in window.iterrows():
        prior = exposure.loc[exposure["station"].eq(row["station"]), "anchor_time"]
        gaps = (prior - row["anchor_time"]).abs().dt.total_seconds().div(3600.0)
        minimum = float(gaps.min()) if len(gaps) else float("inf")
        if minimum >= separation_hours:
            fresh_indices.append(index)
            nearest_gap.append(minimum)
    eligible = window.loc[fresh_indices].copy()
    eligible["nearest_exposed_gap_h"] = nearest_gap

    selected_rows: list[int] = []
    for _, part in eligible.groupby("station", sort=True, observed=True):
        last: pd.Timestamp | None = None
        for row in part.sort_values("anchor_time").itertuples():
            timestamp = pd.Timestamp(row.anchor_time)
            if last is None or (timestamp - last).total_seconds() / 3600.0 >= separation_hours:
                selected_rows.append(int(row.Index))
                last = timestamp
    selected_meta = eligible.loc[selected_rows].sort_values(
        ["station", "anchor_time"], kind="stable"
    )
    selected = anchors.loc[selected_meta.index].copy().sort_values(
        ["station", "anchor_time"], kind="stable"
    )
    selected = selected.reset_index(drop=True)
    station_counts = {
        str(key): int(value)
        for key, value in selected.groupby("station", observed=True).size().items()
    }
    audit = {
        "window_anchor_count": int(len(window)),
        "exposure_time_union_count": int(len(exposure)),
        "fresh_dense_count": int(len(eligible)),
        "fresh_station_global_count": int(len(selected)),
        "fresh_by_station": station_counts,
        "minimum_gap_to_exposed_h": (
            float(selected_meta["nearest_exposed_gap_h"].min())
            if len(selected_meta)
            else None
        ),
        "selection_columns": ["anchor_id", "station", "anchor_time"],
        "target_value_columns_used_for_selection": 0,
        "prediction_value_columns_used_for_selection": 0,
    }
    return selected, audit


def build_safe_single_design(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build the saved single-lead CatBoost design without any future target."""

    forbidden = [name for name in feature_columns if str(name).startswith("target_")]
    if forbidden:
        raise ValueError(f"target columns forbidden in prediction design: {forbidden}")
    required_features = {"anchor_id", "station", *feature_columns}
    required_anchors = {"anchor_id", "station", "current_hs"}
    if missing := required_features.difference(features.columns):
        raise ValueError(f"prediction features missing: {sorted(missing)}")
    if missing := required_anchors.difference(anchors.columns):
        raise ValueError(f"safe anchors missing: {sorted(missing)}")
    safe_anchor_columns = set(anchors.columns)
    if any(str(name).startswith("target_") for name in safe_anchor_columns):
        raise ValueError("safe prediction anchors must not contain target columns")

    anchor_order = anchors["anchor_id"].to_numpy(dtype=np.int64)
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    if features["anchor_id"].duplicated().any() or anchors["anchor_id"].duplicated().any():
        raise ValueError("prediction anchor_id must be unique")
    blocks: list[pd.DataFrame] = []
    for lead in LEADS:
        block = feature_lookup.loc[anchor_order, list(feature_columns)].reset_index(drop=True)
        station = feature_lookup.loc[anchor_order, "station"].astype(str).reset_index(drop=True)
        current = anchor_lookup.loc[anchor_order, "current_hs"].to_numpy(dtype=float)
        block.insert(0, "station", station)
        block.insert(1, "lead_h", str(lead))
        block.insert(2, "current_hs_for_residual", current)
        blocks.append(block)
    design = pd.concat(blocks, ignore_index=True)
    current = anchor_lookup.loc[anchor_order, "current_hs"].to_numpy(dtype=float)
    stations = anchor_lookup.loc[anchor_order, "station"].astype(str).to_numpy()
    if not np.isfinite(current).all():
        raise ValueError("current Hs is non-finite")
    return design, current, stations


def predict_frozen_reference(
    features: pd.DataFrame,
    safe_anchors: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    single_predict: Callable[[pd.DataFrame], np.ndarray],
    multi_predict: Callable[[pd.DataFrame], np.ndarray],
    router: FrozenRouter,
    shrink_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer the saved corrected-v2 reference without fitting or target access."""

    single_x, current, stations = build_safe_single_design(
        features, safe_anchors, feature_columns
    )
    cases = len(safe_anchors)
    single_delta_lead_major = np.asarray(single_predict(single_x), dtype=float)
    if single_delta_lead_major.shape != (cases * len(LEADS),):
        raise ValueError("single-model prediction shape differs")
    single = np.clip(
        current[:, None] + single_delta_lead_major.reshape(len(LEADS), cases).T,
        0.0,
        30.0,
    )

    feature_lookup = features.set_index("anchor_id")
    ids = safe_anchors["anchor_id"].to_numpy(dtype=np.int64)
    multi_x = feature_lookup.loc[ids, ["station", *feature_columns]].reset_index(drop=True)
    multi_x["station"] = multi_x["station"].astype(str)
    multi_delta = np.asarray(multi_predict(multi_x), dtype=float)
    if multi_delta.shape != (cases, len(LEADS)):
        raise ValueError("multi-model prediction shape differs")
    multi = np.clip(current[:, None] + multi_delta, 0.0, 30.0)
    persistence = np.repeat(current[:, None], len(LEADS), axis=1)
    components = np.stack([single, multi, persistence], axis=2)

    observed = feature_lookup.loc[ids, list(OBSERVED_FEATURES)].reset_index(drop=True)
    case_x = build_inference_router_features(
        observed,
        stations,
        current,
        components,
    )
    case_meta = safe_anchors.loc[
        :, ["anchor_id", "station", "anchor_time"]
    ].copy()
    case_meta.insert(0, "fold", "post_h1_fresh_episode")
    row_x, row_meta, row_components = expand_case_router_features(
        case_x, case_meta, components
    )
    weights = np.asarray(router.predict_weights(row_x), dtype=float)
    inactive = ~row_meta["lead_h"].isin(ACTIVE_ROUTER_LEADS).to_numpy()
    weights[inactive] = np.array([0.5, 0.5, 0.0])
    routed = route_row_predictions(row_components, weights)
    final = apply_long_lead_persistence_shrink(
        routed,
        row_components[:, 2],
        row_meta["lead_h"].to_numpy(dtype=int),
        config=LongLeadPersistenceShrink(
            weight=float(shrink_weight), active_leads=ACTIVE_ROUTER_LEADS
        ),
    )
    if not np.isfinite(final).all() or np.any((final < 0.0) | (final > 30.0)):
        raise ValueError("frozen reference prediction violates finite/domain contract")
    receipt = {
        "cases": int(cases),
        "rows": int(len(final)),
        "catboost_fit_count": 0,
        "catboost_model_load_count": 2,
        "router_fit_count": 0,
        "router_model_load_count": 1,
        "target_value_columns_used": 0,
        "six_leads_intact": True,
        "prediction_sha256": array_sha256(final),
        "weight_sha256": array_sha256(weights),
    }
    return final, receipt


def build_comparison_frame(
    anchors: pd.DataFrame,
    incumbent: np.ndarray,
    candidate: np.ndarray,
) -> pd.DataFrame:
    """Attach target values only after prediction sealing."""

    cases = len(anchors)
    expected = cases * len(LEADS)
    left = np.asarray(incumbent, dtype=float)
    right = np.asarray(candidate, dtype=float)
    if left.shape != (expected,) or right.shape != (expected,):
        raise ValueError("paired prediction row count differs")
    truth = np.column_stack(
        [anchors[f"target_{lead}"].to_numpy(dtype=float) for lead in LEADS]
    ).reshape(-1)
    frame = pd.DataFrame(
        {
            "fold": "post_h1_fresh_episode",
            "anchor_id": np.repeat(anchors["anchor_id"].to_numpy(dtype=np.int64), len(LEADS)),
            "station": np.repeat(anchors["station"].astype(str).to_numpy(), len(LEADS)),
            "anchor_time": np.repeat(
                pd.to_datetime(anchors["anchor_time"], utc=True).to_numpy(), len(LEADS)
            ),
            "lead_h": np.tile(np.asarray(LEADS, dtype=int), cases),
            "target_hs": truth,
            "incumbent_prediction": left,
            "candidate_prediction": right,
            "persistence": np.repeat(
                anchors["current_hs"].to_numpy(dtype=float), len(LEADS)
            ),
        }
    )
    numeric = frame[
        ["target_hs", "incumbent_prediction", "candidate_prediction", "persistence"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or frame.duplicated(["anchor_id", "lead_h"]).any():
        raise ValueError("paired comparison contract failed")
    return frame


def comparison_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Report pooled six-lead RMSE first and lead/station diagnostics second."""

    def summarize(part: pd.DataFrame) -> dict[str, Any]:
        truth = part["target_hs"].to_numpy(dtype=float)
        incumbent = part["incumbent_prediction"].to_numpy(dtype=float)
        candidate = part["candidate_prediction"].to_numpy(dtype=float)
        persistence = part["persistence"].to_numpy(dtype=float)
        incumbent_rmse = rmse(truth, incumbent)
        candidate_rmse = rmse(truth, candidate)
        return {
            "cases": int(part["anchor_id"].nunique()),
            "rows": int(len(part)),
            "incumbent_rmse_m": incumbent_rmse,
            "candidate_rmse_m": candidate_rmse,
            "persistence_rmse_m": rmse(truth, persistence),
            "benefit_incumbent_minus_candidate_rmse_m": incumbent_rmse - candidate_rmse,
            "delta_candidate_minus_incumbent_rmse_m": candidate_rmse - incumbent_rmse,
        }

    return {
        "overall": summarize(frame),
        "by_station": {
            str(key): summarize(part)
            for key, part in frame.groupby("station", sort=True, observed=True)
        },
        "by_lead": {
            str(int(key)): summarize(part)
            for key, part in frame.groupby("lead_h", sort=True, observed=True)
        },
    }


def uncertainty_or_insufficient(
    frame: pd.DataFrame,
    *,
    minimum_blocks: int,
) -> dict[str, Any]:
    """Refuse a degenerate one-block CI instead of reporting false certainty."""

    case = frame.drop_duplicates("anchor_id", keep="first").copy()
    case["anchor_day"] = pd.to_datetime(case["anchor_time"], utc=True).dt.floor("D")
    blocks = int(case[["fold", "anchor_day"]].drop_duplicates().shape[0])
    point = comparison_metrics(frame)["overall"][
        "benefit_incumbent_minus_candidate_rmse_m"
    ]
    if blocks < int(minimum_blocks):
        return {
            "status": "NOT_ESTIMABLE_INSUFFICIENT_INDEPENDENT_BLOCKS",
            "unit": "fresh_episode_or_contiguous_anchor_day_block_with_six_leads_intact",
            "independent_blocks": blocks,
            "minimum_blocks_for_interval": int(minimum_blocks),
            "benefit_incumbent_minus_candidate_point_m": float(point),
            "benefit_ci90_m": None,
            "bootstrap_replicates_executed": 0,
        }
    raise NotImplementedError("multi-block bootstrap is not needed by the frozen one-episode contract")


def classify_terminal(
    *,
    integrity_checks: dict[str, bool],
    uncertainty: dict[str, Any],
) -> str:
    """Apply the governing hierarchy without an arbitrary magnitude threshold."""

    if not integrity_checks or not all(integrity_checks.values()):
        return "QA_BLOCKED"
    if uncertainty["status"] == "NOT_ESTIMABLE_INSUFFICIENT_INDEPENDENT_BLOCKS":
        return "INCONCLUSIVE_FRESH_SINGLE_EPISODE_INSUFFICIENT_DEPENDENCE_UNITS"
    return "INCONCLUSIVE"
