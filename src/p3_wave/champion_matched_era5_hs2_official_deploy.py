"""Pure helpers for the preregistered P3 ERA5 Hs-squared deployment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .champion_lineage_energy_residual import apply_champion_energy_residual
from .era5_context_transfer import LEADS, common_feature_columns, summarize_past_48h

KEYS = ("case_id", "station", "lead_h")
EXPECTED_STEPS = np.arange(-2880, 1, 10, dtype=np.int64)
SYNTHETIC_ANCHOR = pd.Timestamp("2000-01-03T00:00:00Z")
CANONICAL_VALUES = ("hs", "tp", "hmax", "wvdir", "wspd", "wdir", "airt", "relh", "caph")


def build_relative_test_features(
    context: pd.DataFrame, test_index: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the frozen 286 features using case-local elapsed time only."""
    expected_context = ["case_id", "station", "step_minute", *CANONICAL_VALUES[:4], "wspd", "gust", "wdir", "airt", "relh", "caph"]
    if list(context.columns) != expected_context:
        raise ValueError("official context schema drifted")
    if list(test_index.columns) != list(KEYS):
        raise ValueError("official index schema drifted")
    if len(context) != 57_800 or len(test_index) != 1_200:
        raise ValueError("official structural row count drifted")
    if context.duplicated(["case_id", "step_minute"]).any() or test_index.duplicated(list(KEYS)).any():
        raise ValueError("official keys contain duplicates")
    lead_contract = test_index.groupby("case_id", sort=False, observed=True)["lead_h"].agg(tuple)
    if len(lead_contract) != 200 or not lead_contract.map(lambda value: value == LEADS).all():
        raise ValueError("official case/lead contract drifted")

    rows: list[dict[str, float | str]] = []
    case_meta: list[dict[str, float | str]] = []
    for case_id, block in context.groupby("case_id", sort=False, observed=True):
        ordered = block.sort_values("step_minute", kind="mergesort").reset_index(drop=True)
        if not np.array_equal(ordered["step_minute"].to_numpy(dtype=np.int64), EXPECTED_STEPS):
            raise ValueError("case-local relative step grid drifted")
        if ordered["station"].nunique() != 1:
            raise ValueError("one official case spans multiple stations")
        relative = ordered.loc[:, CANONICAL_VALUES].copy()
        relative.insert(
            0,
            "relative_time",
            SYNTHETIC_ANCHOR + pd.to_timedelta(ordered["step_minute"], unit="m"),
        )
        row: dict[str, float | str] = {"case_id": str(case_id)}
        row.update(summarize_past_48h(relative, time_column="relative_time"))
        rows.append(row)
        case_meta.append(
            {
                "case_id": str(case_id),
                "station": str(ordered["station"].iloc[0]),
                "current_hs": float(row["hs_current"]),
            }
        )
    features = pd.DataFrame(rows)
    metadata = pd.DataFrame(case_meta)
    if tuple(features.columns) != ("case_id", *common_feature_columns()):
        raise ValueError("official 286-feature order drifted")
    expected_cases = test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    if not metadata[["case_id", "station"]].equals(expected_cases):
        raise ValueError("official context/index case order drifted")
    values = features.loc[:, common_feature_columns()].to_numpy(dtype=np.float64)
    if np.isinf(values).any() or not np.isfinite(metadata["current_hs"]).all():
        raise ValueError("official inference features contain invalid infinity/current values")
    return features, metadata


def align_transfer_predictions(
    test_index: pd.DataFrame,
    case_metadata: pd.DataFrame,
    predictions: np.ndarray,
) -> np.ndarray:
    """Align a 200 by 6 prediction matrix to the immutable official row order."""
    values = np.asarray(predictions, dtype=np.float64)
    if values.shape != (len(case_metadata), len(LEADS)) or not np.isfinite(values).all():
        raise ValueError("transfer prediction matrix drifted")
    case_position = {str(value): i for i, value in enumerate(case_metadata["case_id"])}
    lead_position = {int(value): i for i, value in enumerate(LEADS)}
    aligned = np.asarray(
        [values[case_position[str(row.case_id)], lead_position[int(row.lead_h)]] for row in test_index.itertuples(index=False)],
        dtype=np.float64,
    )
    return aligned


def make_candidate(
    champion: np.ndarray,
    transfer: np.ndarray,
    lead_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the single preregistered 18/24-hour Hs-squared correction."""
    candidate, active = apply_champion_energy_residual(
        champion,
        transfer,
        lead_h,
        energy_weight=0.25,
        active_leads=(18, 24),
    )
    inactive = ~active
    if not np.array_equal(candidate[inactive], np.asarray(champion, dtype=np.float64)[inactive]):
        raise AssertionError("inactive champion rows changed")
    if int(active.sum()) != 400 or int(inactive.sum()) != 800:
        raise AssertionError("active/inactive support drifted")
    if not np.isfinite(candidate).all() or ((candidate < 0.0) | (candidate > 30.0)).any():
        raise ValueError("candidate predictions violate 0..30 m")
    return candidate, active


__all__ = ["align_transfer_predictions", "build_relative_test_features", "make_candidate"]
