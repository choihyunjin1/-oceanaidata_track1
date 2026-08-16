"""Strict P3 submission construction and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_submission(frame: pd.DataFrame, test_index: pd.DataFrame) -> None:
    keys = ["case_id", "station", "lead_h"]
    required = keys + ["hs_pred"]
    if list(frame.columns) != required:
        raise ValueError(f"submission columns must be {required}")
    if len(frame) != 1_200 or len(test_index) != 1_200:
        raise ValueError("P3 submission must contain exactly 1,200 rows")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("submission keys contain missing values or duplicates")
    if not frame[keys].equals(test_index[keys]):
        raise ValueError("submission key set or order differs from test_index")
    if not frame["lead_h"].isin([3, 6, 9, 12, 18, 24]).all():
        raise ValueError("invalid lead_h")
    prediction = pd.to_numeric(frame["hs_pred"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(prediction).all() or not ((prediction >= 0.0) & (prediction <= 30.0)).all():
        raise ValueError("hs_pred must be finite and within 0..30 m")


def build_submission(test_index: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    values = np.asarray(prediction, dtype=float)
    if values.shape != (len(test_index),):
        raise ValueError("prediction length mismatch")
    frame = test_index.copy()
    frame["hs_pred"] = values
    validate_submission(frame, test_index)
    return frame


def write_submission(frame: pd.DataFrame, test_index: pd.DataFrame, path: str | Path) -> Path:
    validate_submission(frame, test_index)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    reread = pd.read_csv(target)
    validate_submission(reread, test_index)
    return target
