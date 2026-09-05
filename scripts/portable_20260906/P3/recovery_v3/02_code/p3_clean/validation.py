"""Minimal clean recipe functions extracted without algorithm changes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LEADS


def expand_leads(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    anchor_ids: np.ndarray,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Expand anchors to the official pooled station-lead regression grain."""

    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    blocks: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    for lead in LEADS:
        block = feature_lookup.loc[anchor_ids, feature_columns].reset_index(drop=True)
        station = feature_lookup.loc[anchor_ids, "station"].astype(str).reset_index(drop=True)
        current = anchor_lookup.loc[anchor_ids, "current_hs"].to_numpy(dtype=float)
        target = anchor_lookup.loc[anchor_ids, f"target_{lead}"].to_numpy(dtype=float)
        block.insert(0, "station", station)
        block.insert(1, "lead_h", lead)
        block.insert(2, "current_hs_for_residual", current)
        blocks.append(block)
        targets.append(target - current)
        metadata.append(
            pd.DataFrame(
                {
                    "anchor_id": anchor_ids,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                    "target_hs": target,
                }
            )
        )
    return (
        pd.concat(blocks, ignore_index=True),
        np.concatenate(targets),
        pd.concat(metadata, ignore_index=True),
    )


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def metric_slices(metadata: pd.DataFrame, prediction: np.ndarray) -> dict[str, object]:
    frame = metadata.copy()
    frame["prediction"] = prediction
    result: dict[str, object] = {
        "rmse": rmse(frame["target_hs"].to_numpy(), frame["prediction"].to_numpy()),
        "n": int(len(frame)),
        "by_lead": {},
        "by_station": {},
    }
    for lead, group in frame.groupby("lead_h", observed=True):
        result["by_lead"][str(int(lead))] = rmse(
            group["target_hs"].to_numpy(), group["prediction"].to_numpy()
        )
    for station, group in frame.groupby("station", observed=True):
        result["by_station"][str(station)] = rmse(
            group["target_hs"].to_numpy(), group["prediction"].to_numpy()
        )
    return result
