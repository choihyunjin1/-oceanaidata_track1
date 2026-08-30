"""Query-contract-only adapter for the sealed Gaussian-copula v2 pack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p2_restore import (
    p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1 as sealed,
)

EXPERIMENT_ID = "p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v3"
INPUT_COLUMNS = ["station", "layer", "time", "nominal_depth"]
OUTPUT_COLUMNS = ["station", "layer", "time", "temp"]
KEYS = ["station", "layer", "time"]


def read_test_index_superset(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the exact official input superset without using its auxiliary as a feature."""
    frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
    if list(frame.columns) != INPUT_COLUMNS:
        raise sealed.SubmissionPackError(f"test_index schema mismatch: {path}")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise sealed.SubmissionPackError(f"test_index keys null or duplicate: {path}")
    nominal_depth = pd.to_numeric(frame["nominal_depth"], errors="coerce").to_numpy(
        np.float64
    )
    if not np.isfinite(nominal_depth).all():
        raise sealed.SubmissionPackError("test_index nominal_depth is nonfinite")
    frame["nominal_depth"] = nominal_depth
    receipt = {
        "columns_exact_order": list(frame.columns),
        "rows": len(frame),
        "key_columns_exact_order": KEYS,
        "nominal_depth_preserved": True,
        "nominal_depth_finite": True,
        "nominal_depth_used_as_model_feature": False,
    }
    return frame, receipt


def materialize_candidate(
    repo_root: Path,
    config: dict[str, Any],
    p2_dir: Path,
    base_u_path: Path,
    alpha50_path: Path,
    incumbent_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Invoke the unchanged model with only the documented query reader adapted."""
    original_reader = sealed.read_keyed_csv
    input_receipt: dict[str, object] = {}

    def repaired_reader(path: Path, *, require_temp: bool) -> pd.DataFrame:
        if require_temp:
            return original_reader(path, require_temp=True)
        if path.name != "test_index.csv":
            raise sealed.SubmissionPackError("unexpected key-only CSV in v3 adapter")
        frame, receipt = read_test_index_superset(path)
        input_receipt.update(receipt)
        return frame

    sealed.read_keyed_csv = repaired_reader
    try:
        candidate, materialization = sealed.materialize_candidate(
            repo_root,
            config,
            p2_dir,
            base_u_path,
            alpha50_path,
            incumbent_path,
        )
    finally:
        sealed.read_keyed_csv = original_reader

    if list(candidate.columns) != OUTPUT_COLUMNS:
        raise sealed.SubmissionPackError("candidate output schema drifted")
    materialization["query_contract"]["input_superset"] = input_receipt
    materialization["query_contract"]["output_columns_exact_order"] = list(
        candidate.columns
    )
    return candidate, materialization
