"""P2 submission construction and strict validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.data import KEYS


def build_submission(test_index: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    values = np.asarray(prediction, dtype=float)
    if (
        len(values) != len(test_index)
        or not np.isfinite(values).all()
        or not ((values >= -5) & (values <= 45)).all()
    ):
        raise ValueError("P2 predictions must be finite, in range, and match test rows")
    result = test_index.loc[:, KEYS].copy()
    result["temp"] = values
    return result


def validate_submission(
    submission: str | Path | pd.DataFrame, test_index: pd.DataFrame
) -> dict[str, object]:
    frame = (
        pd.read_csv(submission, dtype={"station": "string", "time": "string"})
        if not isinstance(submission, pd.DataFrame)
        else submission.copy()
    )
    if list(frame.columns) != KEYS + ["temp"] or len(frame) != len(test_index):
        raise ValueError("invalid P2 submission schema or row count")
    if frame.duplicated(KEYS).any() or not frame[KEYS].equals(test_index[KEYS]):
        raise ValueError("P2 submission keys/order differ from test_index")
    values = pd.to_numeric(frame["temp"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= -5) & (values <= 45)).all():
        raise ValueError("P2 temp must be finite and within -5..45 C")
    return {"rows": len(frame), "minimum": float(values.min()), "maximum": float(values.max())}
