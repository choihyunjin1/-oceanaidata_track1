"""Key-only scored-population alignment for P2 dynamic sigmoid Gate 1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.dynamic_sigmoid_profile import TimeBlock


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_key_only_population(path: Path, *, expected_sha256: str) -> pd.DataFrame:
    """Read only the incumbent OOF keys; labels and predictions remain unopened."""

    if sha256(path) != expected_sha256:
        raise ValueError("incumbent OOF SHA differs from the pinned contract")
    keys = pd.read_parquet(path, columns=["time", "layer"])
    if list(keys.columns) != ["time", "layer"]:
        raise ValueError("key-only OOF reader returned unexpected columns")
    keys["time"] = pd.to_datetime(keys["time"], utc=True, errors="raise")
    if keys[["time", "layer"]].isna().any().any():
        raise ValueError("incumbent OOF keys contain missing values")
    if keys.duplicated(["time", "layer"]).any():
        raise ValueError("incumbent OOF keys are not unique")
    return keys


def key_aligned_gate_1(
    public_features: pd.DataFrame,
    key_population: pd.DataFrame,
    block: TimeBlock,
    *,
    minimum_public_points: int,
    minimum_depth_span_m: float,
    threshold: float,
) -> dict[str, object]:
    """Evaluate support over unique scored-key times, not the unscored full grid."""

    key_rows = key_population.loc[block.mask(key_population["time"]), ["time", "layer"]]
    scored_times = pd.DatetimeIndex(sorted(key_rows["time"].unique()))
    if len(scored_times) == 0:
        raise ValueError(f"block {block.name!r} has no scored-key times")
    if not scored_times.isin(public_features.index).all():
        raise ValueError(f"block {block.name!r} contains scored times absent from public features")
    scored = public_features.reindex(scored_times)
    supported = (scored["public_temp_count"].to_numpy(float) >= int(minimum_public_points)) & (
        scored["public_depth_span"].to_numpy(float) >= float(minimum_depth_span_m)
    )
    share = float(np.mean(supported))

    full = public_features.loc[block.mask(public_features.index)]
    full_supported = (full["public_temp_count"].to_numpy(float) >= int(minimum_public_points)) & (
        full["public_depth_span"].to_numpy(float) >= float(minimum_depth_span_m)
    )
    layer_counts = key_rows.groupby("time")["layer"].nunique().value_counts().sort_index()
    return {
        "population": "unique_incumbent_oof_key_times",
        "oof_columns_read": ["time", "layer"],
        "truth_read": False,
        "prediction_read_for_gate_1": False,
        "scored_key_rows": int(len(key_rows)),
        "validation_times": int(len(scored_times)),
        "scored_times_by_target_key_count": {
            str(int(count)): int(rows) for count, rows in layer_counts.items()
        },
        "supported_times": int(supported.sum()),
        "support_share": share,
        "threshold": float(threshold),
        "pass": bool(share >= threshold),
        "full_grid_diagnostic": {
            "validation_times": int(len(full)),
            "supported_times": int(full_supported.sum()),
            "support_share": float(np.mean(full_supported)) if len(full) else 0.0,
            "denominator_removed_times": int(len(full) - len(scored_times)),
            "supported_numerator_removed_times": int(full_supported.sum() - supported.sum()),
        },
    }
